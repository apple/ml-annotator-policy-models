#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import argparse
import json
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aix360.algorithms.rbm.BRCG import BRCGExplainer
from aix360.algorithms.rbm.boolean_rule_cg import BooleanRuleCG
from dotenv import load_dotenv
from safe_dictionary_learning.decision_functions.utils import convert_to_multiindex
from sklearn.model_selection import train_test_split
from tqdm import tqdm

load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy")

def sort_rules(rules):
    """Sort rules by their feature indices to ensure consistent ordering for comparison.
    """
    out = set([" AND ".join(sorted(r1.split(" AND "))) for r1 in rules])
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bootstrap DNF rule frequency estimation')
    parser.add_argument('--dataset', type=str, default="data/diverse_safety_adversarial_dialog_350_aggregated.csv")
    parser.add_argument('--category', type=str, default="rater_race",
                        choices=['rater_age', 'rater_gender', 'rater_race', 'rater_education'])
    parser.add_argument('--features-csv', type=str, default="results/cluster5.csv")
    parser.add_argument('--bootstrap-rounds', type=int, default=100)
    parser.add_argument('--output-json', type=str, default="results/dnf_parameter_uncertainty.json")
    parser.add_argument('--output-figures', type=str, default="figures")
    args = parser.parse_args()
    os.makedirs(args.output_figures, exist_ok=True)

    dataset = pd.read_csv(args.dataset)
    featuredf = pd.read_csv(args.features_csv)

    for col in featuredf.columns:
        if "model compliance" in col:
            featuredf = featuredf.drop(col, axis=1)

    keys_df = pd.DataFrame({'response': dataset['response']})
    features_expanded = keys_df.join(featuredf.set_index('response'), on='response', how='left').drop('response', axis=1)
    features = features_expanded.columns.tolist()
    binary_matrix = features_expanded.to_numpy().astype(int)

    bootstrap_rounds = args.bootstrap_rounds

    json_path = args.output_json

    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
    else:
        data = {}

    for cat in dataset[args.category].unique():
        print(cat)
        indices = dataset.loc[dataset[args.category] == cat].index

        X = binary_matrix[indices]
        y = (dataset.loc[indices, 'Q_overall'] == 'Yes').astype(int).values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_test_multiindex = convert_to_multiindex(pd.DataFrame(X_test, columns=features))

        weights = {}
        accs = []
        for i in tqdm(range(bootstrap_rounds)):
            X_train_resampled = np.random.choice(X_train.shape[0], size=X_train.shape[0], replace=True)
            X_train_multiindex_resampled = convert_to_multiindex(pd.DataFrame(X_train[X_train_resampled], columns=features))

            brcg = BRCGExplainer(BooleanRuleCG(
                    lambda0=0.00001,
                    lambda1=0.0001,
                ))
            brcg.fit(X_train_multiindex_resampled, y_train[X_train_resampled])
            rules = brcg.explain()['rules']
            print(len(rules))
            accs.append(np.mean(brcg.predict(X_test_multiindex) == y_test))
            sorted_rules = sort_rules(rules)
            for rule in sorted_rules:
                if rule in weights.keys():
                    weights[rule] += 1
                else:
                    weights[rule] = 1

        for rule in weights.keys():
            weights[rule] /= bootstrap_rounds

        plt.figure(figsize=(10, 6))
        plt.bar(range(len(weights)), list(weights.values()))
        plt.xlabel('Rule Index')
        plt.ylabel('Frequency of Appearance')
        plt.title(f'Rule Frequency for Category: {cat}')
        plt.savefig(os.path.join(args.output_figures, f"dnf_rule_frequency_{cat.replace('/', '_').replace(' ', '_')}.png"))
        plt.close()

        entropies = []
        for rule, weight in weights.items():
            entropy = -weight * np.log2(weight) - (1 - weight) * np.log2(1 - weight) if weight not in [0, 1] else 0
            entropies.append(entropy)
        print(np.mean(entropies))
        print(np.mean(list(weights.values())))
        data[cat] = {
            "entropy": np.mean(entropies),
            "mean_weight": np.mean(list(weights.values()))
        }
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
