#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import hashlib
import os

import pandas as pd
import torch
from dotenv import load_dotenv
from google import genai
from google.genai.types import EmbedContentConfig
from sparsemax import Sparsemax
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery: {query}"


class Embedder:
    def __init__(self, embedding_model: str, device="cpu"):
        self.embedding_model = embedding_model
        self.device = device

        if self.embedding_model == "qwen":
            self.batch_size = 32
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-Embedding-8B", padding_side="left"
            )
            self.embedder = AutoModel.from_pretrained(
                "Qwen/Qwen3-Embedding-8B", torch_dtype=torch.float16, device_map=self.device
            )
            self.tokenizer._tokenizer.model._resize_cache(0)

        elif self.embedding_model == "gemini":
            load_dotenv()
            self.task_type = "SEMANTIC_SIMILARITY"
            self.batch_size = 250
            self.client = genai.Client()

    def _embed(self, text: list[str]):
        if self.embedding_model == "qwen":
            num_batches = len(text) // self.batch_size + 1 * int(len(text) % self.batch_size > 0)

            embeddings = torch.empty(len(text), 4096, dtype=torch.float16)
            for batch_idx in tqdm(range(num_batches)):
                batch = text[
                    batch_idx * self.batch_size : min(self.batch_size * (batch_idx + 1), len(text))
                ]
                with torch.no_grad():
                    batch_dict = self.tokenizer(
                        batch, max_length=1024, padding=True, truncation=True, return_tensors="pt"
                    ).to(self.device)
                    outputs = self.embedder(**batch_dict)
                embeddings[
                    batch_idx * self.batch_size : min(self.batch_size * (batch_idx + 1), len(text))
                ] = last_token_pool(outputs.last_hidden_state, batch_dict["attention_mask"]).cpu()
                del outputs, batch_dict

        elif self.embedding_model == "gemini":
            num_batches = len(text) // self.batch_size + 1 * int(len(text) % self.batch_size > 0)

            embeddings = torch.empty(len(text), 1536, dtype=torch.float16)
            for batch_idx in tqdm(range(num_batches)):
                batch = text[
                    batch_idx * self.batch_size : min(self.batch_size * (batch_idx + 1), len(text))
                ]

                response = self.client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=batch,
                    config=EmbedContentConfig(
                        task_type=self.task_type,  # Optional
                        output_dimensionality=1536,  # Optional
                    ),
                )
                embed_list = [embed.values for embed in response.embeddings]
                # for embed in response.embeddings:
                #     print([embed])
                #     exit()
                # print([response.embeddings])
                embeddings[
                    batch_idx * self.batch_size : min(self.batch_size * (batch_idx + 1), len(text))
                ] = torch.tensor(embed_list)
        return embeddings

    def load_data(self, path, key, cache_dir=None):
        data_name = path.split("/")[-1].split(".")[
            0
        ]  ## +key to allow for caching of embeddings from same df ##filename without ext
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/")
        embedding_dir = os.path.join(cache_dir, "annotator-policy-models/embeddings")
        os.makedirs(embedding_dir, exist_ok=True)
        if self.embedding_model == "gemini":
            filename = self.embedding_model + "_" + self.task_type + "_" + data_name + ".pt"
        else:
            filename = self.embedding_model + "_" + data_name + ".pt"
        if os.path.exists(os.path.join(embedding_dir, filename)):  ## if found locally, download
            self.data_embeddings = torch.load(os.path.join(embedding_dir, filename), weights_only=True)
            return
        else:
            df = pd.read_csv(path, index_col=False)
            text = df[key].tolist()
            self.data_embeddings = self._embed(text)  ## Run embedder
            OUTPUT_DIR = embedding_dir
            os.makedirs(OUTPUT_DIR, exist_ok=True)

            torch.save(self.data_embeddings, os.path.join(OUTPUT_DIR, filename))
            return

    def _compute_features(self, features):
        if self.embedding_model == "qwen":
            task = "Given a text feature, retrieve passages which explicitly contain references to that specific feature"
            queries = []
            for feature in features:
                queries += [get_detailed_instruct(task, feature)]

            computed_features = self._embed(queries)
        else:
            computed_features = self._embed(features)
        return computed_features

    def load_features(self, features, cache_dir=None):
        """Load feature embeddings, using cache when available.

        Computes embeddings for the given feature strings and caches them to disk.
        On subsequent calls with the same features, the cached embeddings are loaded
        instead of calling the embedding API.
        """
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/")
        embedding_dir = os.path.join(cache_dir, "annotator-policy-models/embeddings")
        os.makedirs(embedding_dir, exist_ok=True)

        # Create a stable cache key from sorted feature strings
        features_key = "_".join(sorted(features))
        features_hash = hashlib.sha256(features_key.encode()).hexdigest()[:12]
        if self.embedding_model == "gemini":
            filename = f"gemini_{self.task_type}_features_{features_hash}.pt"
        else:
            filename = f"{self.embedding_model}_features_{features_hash}.pt"

        cache_path = os.path.join(embedding_dir, filename)
        if os.path.exists(cache_path):
            self.features = torch.load(cache_path, weights_only=True)
        else:
            self.features = self._compute_features(features)
            torch.save(self.features, cache_path)

        data_embeds_normalized = torch.nn.functional.normalize(self.data_embeddings, dim=1)
        feature_embeds_normalized = torch.nn.functional.normalize(self.features, dim=1)
        self.scores = (feature_embeds_normalized @ data_embeds_normalized.T).T

    def add_feature(self, features_to_add):
        if not isinstance(features_to_add, list):
            features_to_add = [features_to_add]
        newfeatures = self._compute_features(features_to_add)

        self.features = torch.concatenate((self.features, newfeatures))

        data_embeds_normalized = torch.nn.functional.normalize(self.data_embeddings, dim=1)
        feature_embeds_normalized = torch.nn.functional.normalize(newfeatures, dim=1)
        self.scores = torch.concatenate(
            (self.scores, feature_embeds_normalized @ data_embeds_normalized.T), dim=0
        ).T

    def get_binary_scores(self, method, target=None, **kwargs):
        binary_scores = torch.zeros_like(self.scores)
        if method == "percentile":
            percentile = kwargs.get("percentile", 95)
            for feat_idx in range(self.scores.shape[1]):
                optimal_threshold = torch.quantile(self.scores[:, feat_idx], percentile)
                binary_scores[:, feat_idx] = (self.scores[:, feat_idx] > optimal_threshold).int()
        elif method == "midpoint":
            unsafe_indices = torch.argwhere(target == 1)
            safe_indices = torch.argwhere(target == 0)
            for feat_idx in range(self.scores.shape[1]):
                optimal_threshold = (
                    (self.scores[unsafe_indices, feat_idx]).mean().item()
                    + (self.scores[safe_indices, feat_idx]).mean().item()
                ) / 2
                binary_scores[:, feat_idx] = (self.scores[:, feat_idx] > optimal_threshold).int()
        elif method == "sparsemax":
            alpha = kwargs.get("alpha", 4.0)
            sparsemax = Sparsemax(dim=-1)
            sparsemax_probs = sparsemax(alpha * self.scores)
            binary_scores = (sparsemax_probs > 0).int()
        else:
            print("method not supported")
            return
        return binary_scores
