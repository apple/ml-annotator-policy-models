#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import argparse
import asyncio
import json
import os

import numpy as np
import openai
import pandas as pd
import torch
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from tqdm import tqdm

load_dotenv()

client = AsyncOpenAI()

@retry(
    retry=retry_if_exception_type(openai.RateLimitError),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(6)
)
async def make_request(prompt):
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

async def process_batch(prompts, max_concurrent=5):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def limited_request(prompt):
        async with semaphore:
            return await make_request(prompt)

    tasks = [limited_request(p) for p in prompts]
    return await asyncio.gather(*tasks)

def apply_template(concept, text):
    template = "Please answer whether the following text contains the concept of {}. Respond only with Yes or No.\n\nText:\n{}\n\nAnswer:\n"
    return template.format(concept, text)

def apply_categorical_template(category, concepts, text):
    template = "Please answer with the {} of the following chat interaction. Respond only with one of the following: {}.\n\nText:\n{}\n\nAnswer:\n"
    return template.format(category,", ".join(concepts), text)

async def batch_get_labels(concepts, dataset, batch_size=100, output_csv="results/labeled_concepts.csv"):
    if os.path.exists(output_csv):
        df_labeled = pd.read_csv(output_csv)
        print(f"Loaded existing CSV with columns: {df_labeled.columns.tolist()}")
    else:
        df_labeled = pd.DataFrame(dataset['text'])
        df_labeled.columns = ['text']
    results = torch.zeros((len(dataset), len(concepts)), dtype=torch.float32)

    for i, concept in tqdm(enumerate(concepts), total=len(concepts), desc="concepts"):
        if concept in df_labeled.columns:
            print(f"Skipping '{concept}' - already exists in CSV")
            results[:, i] = torch.tensor(df_labeled[concept].values, dtype=torch.float32)
            continue
        prompts = []
        labels = []

        for index, row in dataset.iterrows():
            prompt = apply_template(concept, row['text'])
            prompts.append(prompt)
        all_responses = []
        for batch_start in tqdm(range(0, len(prompts), batch_size), desc=f"Batches for {concept}"):
            batch_prompts = prompts[batch_start:batch_start + batch_size]
            batch_results = await process_batch(batch_prompts, max_concurrent=10)
            all_responses.extend(batch_results)

            if batch_start + batch_size < len(prompts):
                await asyncio.sleep(1)

        for resp in all_responses:
            if resp.strip().lower() not in ["yes", "no"]:
                print(f"Unexpected response: {resp}", flush=True)
        labels = [1.0 if resp.strip().lower().startswith("yes") else 0.0 for resp in all_responses]

        df_labeled[concept] = labels
        df_labeled.to_csv(output_csv, index=False)

async def batch_get_categorical_labels(category, concepts, dataset, batch_size=100, output_csv="results/labeled_concepts.csv"):
    if os.path.exists(output_csv):
        df_labeled = pd.read_csv(output_csv)
        print(f"Loaded existing CSV with columns: {df_labeled.columns.tolist()}")
    else:
        df_labeled = pd.DataFrame(dataset['text'])
        df_labeled.columns = ['text']

    if category + ": " + concepts[0] in df_labeled.columns:
        print(f"Skipping '{category}' - already exists in CSV")
        return
    prompts = []
    for index, row in dataset.iterrows():
        prompt = apply_categorical_template(category, concepts, row['text'])
        prompts.append(prompt)
    all_responses = []
    for batch_start in tqdm(range(0, len(prompts), batch_size), desc=f"Batches for {category}"):
        batch_prompts = prompts[batch_start:batch_start + batch_size]
        batch_results = await process_batch(batch_prompts, max_concurrent=10)
        all_responses.extend(batch_results)

        if batch_start + batch_size < len(prompts):
            await asyncio.sleep(1)

    for resp in all_responses:
        if resp.strip() not in concepts:
            print(concepts)
            print(f"Unexpected response: {resp}", flush=True)
    labels = [resp.strip() for resp in all_responses]

    onehot_matrix = np.zeros((len(labels), len(concepts)), dtype=int)
    label_to_idx = {label.strip(): idx for idx, label in enumerate(concepts)}

    for i, label in enumerate(labels):
        onehot_matrix[i, label_to_idx[label]] = 1.

    for idx, label in enumerate(concepts):
        df_labeled[category + ": " + label] = onehot_matrix[:, idx]
    df_labeled.to_csv(output_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Label concepts in dataset using LLM')
    parser.add_argument('--feature-path', type=str, default="concepts/concept_templates.json")
    parser.add_argument('--output', type=str, default="results/labeled_concepts.csv")
    parser.add_argument('--dataset', type=str, default="data/diverse_safety_adversarial_dialog_350.csv")
    args = parser.parse_args()


    with open(args.feature_path, encoding='utf-8') as file:
        data = json.load(file)
    binary_concepts = data["binary_concepts"]

    dataset = pd.read_csv(args.dataset)
    dataset_unique = dataset.groupby('text').first().reset_index()

    asyncio.run(batch_get_labels(binary_concepts, dataset_unique, batch_size=512, output_csv=args.output))

    for category, concepts in data["categorical_concepts"].items():
        print(f"Processing category: {category}")
        asyncio.run(batch_get_categorical_labels(category, concepts, dataset_unique, batch_size=512, output_csv=args.output))
