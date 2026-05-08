#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from safe_dictionary_learning import NonNegativeLogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm

load_dotenv()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fit NNLR probe with bootstrap uncertainty')
    parser.add_argument('--dataset', type=str, default="data/diverse_safety_adversarial_dialog_350_aggregated.csv")
    parser.add_argument('--category', type=str, default="rater_education",
                        choices=['rater_age', 'rater_gender', 'rater_race', 'rater_education'])
    parser.add_argument('--features-csv', type=str, default="results/cluster5.csv")
    parser.add_argument('--exclude-features', type=str, default="model compliance",
                        help='Substring to match for excluding feature columns')
    parser.add_argument('--bootstrap-rounds', type=int, default=1000)
    parser.add_argument('--output-json', type=str, default="results/nnlr_parameter_uncertainty.json")
    parser.add_argument('--output-figures', type=str, default="figures/uncertainty")
    args = parser.parse_args()

    dataset = pd.read_csv(args.dataset)
    featuredf = pd.read_csv(args.features_csv)

    for col in featuredf.columns:
        if args.exclude_features in col:
            featuredf = featuredf.drop(col, axis=1)

    keys_df = pd.DataFrame({'response': dataset['response']})
    features_expanded = keys_df.join(featuredf.set_index('response'), on='response', how='left').drop('response', axis=1)
    features = features_expanded.columns.tolist()
    binary_matrix = features_expanded.to_numpy().astype(int)

    bootstrap_rounds = args.bootstrap_rounds

    json_path = args.output_json
    os.makedirs(args.output_figures, exist_ok=True)

    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
    else:
        data = {}

    for cat in dataset[args.category].unique():
        print(cat)
        indices = dataset.loc[dataset[args.category] == cat].index

        X = binary_matrix[indices]
        y = (dataset.loc[indices, 'Q2_harmful_content_overall'] == 'Yes').astype(int).values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        weights = np.zeros((X.shape[1], bootstrap_rounds))
        accs = []
        for i in tqdm(range(bootstrap_rounds)):
            X_train_resampled = np.random.choice(X_train.shape[0], size=X_train.shape[0], replace=True)
            clf = NonNegativeLogisticRegression(
                n_features=X.shape[1],
                lam=0.001
            )
            clf.fit(X_train[X_train_resampled], y_train[X_train_resampled])
            weights[:, i] = clf.coef_[0]
            accs.append(clf.score(X_test, y_test))
        print(np.mean(weights, axis=1))
        print(np.std(weights, axis=1))
        print(np.mean(accs))
        print(np.mean(np.linalg.norm(weights, ord=0, axis=0)))

        means = np.mean(weights, axis=1)
        lower = np.percentile(weights, 5, axis=1)
        upper = np.percentile(weights, 95, axis=1)

        sort_idx = np.argsort(means)[::-1]
        means_sorted = means[sort_idx]
        ci_lower_sorted = np.maximum(0, means_sorted - lower[sort_idx])
        ci_upper_sorted = np.maximum(0, upper[sort_idx] - means_sorted)
        names_sorted = [features[i] for i in sort_idx]

        data[cat] = {
            name: {
                'mean': float(mean),
                'lower': float(mean - ci_lo),
                'upper': float(mean + ci_hi)
            }
            for name, mean, ci_lo, ci_hi in zip(names_sorted, means_sorted, ci_lower_sorted, ci_upper_sorted)
        }

        nonzero_mask = means_sorted != 0
        means_sorted = means_sorted[nonzero_mask]
        ci_lower_sorted = ci_lower_sorted[nonzero_mask]
        ci_upper_sorted = ci_upper_sorted[nonzero_mask]
        names_sorted = [n for n, keep in zip(names_sorted, nonzero_mask) if keep]

        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

        fig, ax = plt.subplots(figsize=(14, 6))

        colors = ['#d73027' if v < 0 else '#4575b4' for v in means_sorted]

        bars = ax.bar(
            range(len(means_sorted)),
            means_sorted,
            yerr=[ci_lower_sorted, ci_upper_sorted],
            color=colors,
            alpha=0.85,
            capsize=4,
            error_kw=dict(elinewidth=1.2, ecolor='black', capthick=1.5),
            edgecolor='white',
            linewidth=0.5
        )

        ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

        ax.set_xticks(range(len(names_sorted)))
        ax.set_xticklabels(names_sorted, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Parameter Weight', fontsize=11)
        ax.set_title(f'NNLR Parameter Weights with 95% Bootstrap Confidence Intervals, {cat}, Accuracy: {np.mean(accs):.2f}', fontsize=13)
        ax.set_xlim(-0.5, len(means_sorted) - 0.5)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(
            os.path.join(args.output_figures, f'parameter_weights_{cat.replace("/","_").replace(" ", "_")}.png'),
            dpi=150, bbox_inches='tight'
        )
        plt.close()
