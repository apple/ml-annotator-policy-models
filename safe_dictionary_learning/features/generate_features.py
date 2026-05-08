#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Given a dataset, prompt an LLM for concepts and aggregate into feature list. clustering optional."""

import argparse
import ast
import os

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from openai import OpenAI
from scipy.cluster.hierarchy import fcluster, linkage
from tqdm import tqdm

SYSTEM_PROMPT = "You are a knowledgable Linguist and CS researcher helping brainstorm text features relevant to safety data."


def get_response(client, model, prompt):
    """Send a prompt to the OpenAI chat completions API and return the response text."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content


def prompt_agent_for_concepts(client, model, df, prompt, max_features=10000, verbose=False):
    features = set()
    for i, row in tqdm(df.iterrows()):
        full_prompt = prompt + row.text
        try:
            response_string = get_response(client, model, full_prompt)
        except:
            continue
        try:
            opening_bracket = response_string.index("[")
            closing_bracket = response_string.index("]")
            response_string = response_string[opening_bracket : closing_bracket + 1]
            feature_list = set(ast.literal_eval(response_string))
        except:
            print(response_string)
            feature_list = set()
            # print(response["text"])
        if verbose:
            for f in feature_list:
                if f not in features:
                    print(f)
            print(row.text)
        features.update(feature_list)

        if len(features) > max_features:
            break
    return list(features)


def cluster(features, dataset, features_dir):
    embeds = torch.load(
        os.path.join(features_dir, f"{dataset}_clustered_features_embeddings.pt"), weights_only=True
    )

    embeds = embeds.numpy()
    embeds_normalized = embeds / np.linalg.norm(embeds, ord=2, axis=1)[:, np.newaxis]

    linkage_matrix = linkage(embeds_normalized, method="ward", metric="euclidean")

    n_clusters = 20

    clusters = fcluster(linkage_matrix, n_clusters, criterion="maxclust")

    return clusters


def summarize_cluster(clusters, client, model, features):
    unique_clusters = np.unique(clusters)

    higher_cluster_features = set()
    for cluster_id in unique_clusters:
        points_in_cluster = np.where(clusters == cluster_id)[0]
        featurelist = [features[i] for i in points_in_cluster]
        prompt_head = "Can you give me a set of higher-level concepts that capture the following text features? The higher-level concepts should be short words or phrases that are more general but still easily verifiable from text. Respond with just the higher-level concept in the form of python list and nothing else.\n\n"
        prompt = prompt_head + str(featurelist)
        response_string = get_response(client, model, prompt)
        opening_bracket = response_string.index("[")
        closing_bracket = response_string.index("]")
        response_string = response_string[opening_bracket : closing_bracket + 1]
        print(ast.literal_eval(response_string))
        try:
            response_set = set(ast.literal_eval(response_string))
        except:
            # print(response_string)
            response_set = set()
        higher_cluster_features.update(response_set)
    return higher_cluster_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/")
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--cluster", action="store_true")
    args = parser.parse_args()

    datasets_dir = os.path.join(args.data_root, "datasets")
    features_dir = os.path.join(args.data_root, "features")

    csv_path = os.path.join(datasets_dir, f"{args.dataset}.csv")
    if not os.path.exists(csv_path):
        print(f"dataset '{args.dataset}' not found at {csv_path}")
        print(
            f"  Hint: You can download it with:\n"
            f"    uv run python -m safe_dictionary_learning.data.download "
            f"--dataset <beavertails|wildguardmix> --output-dir {datasets_dir}"
        )
        exit()
    df = pd.read_csv(csv_path, index_col=False)

    load_dotenv()
    client = OpenAI()
    model = "gpt-4o-mini-2024-07-18"

    features = []
    with open(os.path.join(features_dir, f"{args.dataset}_features.txt")) as f:
        lines = f.readlines()
        for line in lines:
            features.append(line.strip())

    if args.cluster:
        clusters = cluster(features, args.dataset, features_dir)
        high_level_features = summarize_cluster(clusters, client, model, features)
        print(len(high_level_features))
        with open(
            os.path.join(features_dir, f"{args.dataset}_clustered_features2.txt"),
            "w",
        ) as f:
            for feat in high_level_features:
                f.write(feat)
                f.write("\n")
