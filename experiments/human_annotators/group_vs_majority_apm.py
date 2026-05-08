#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from aix360.algorithms.rbm.BRCG import BRCGExplainer
from aix360.algorithms.rbm.boolean_rule_cg import BooleanRuleCG
from dotenv import load_dotenv
from safe_dictionary_learning.decision_functions.utils import (compute_rule_metrics, get_rule_examples,
                       format_example_output, format_rules, diff_rules,
                       convert_to_multiindex)
from sklearn.model_selection import train_test_split

load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fit DNF rules per demographic group and compare to aggregate')
    parser.add_argument('--dataset', type=str, default="350")
    parser.add_argument('--category', type=str, default="rater_gender",
                        choices=['rater_age', 'rater_gender', 'rater_race', 'rater_education'])
    parser.add_argument('--features-csv', type=str, default="results/cluster5.csv")
    parser.add_argument('--output-dir', type=str, default="results")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = pd.read_csv(f"data/diverse_safety_adversarial_dialog_{args.dataset}_aggregated.csv")
    featuredf = pd.read_csv(args.features_csv)

    for col in featuredf.columns:
        if "model compliance" in col:
            featuredf = featuredf.drop(col, axis=1)

    keys_df = pd.DataFrame({'response': dataset['response']})
    features_expanded = keys_df.join(featuredf.set_index('response'), on='response', how='left').drop('response', axis=1)
    features = features_expanded.columns.tolist()
    binary_matrix = features_expanded.to_numpy().astype(int)

    print(np.linalg.norm(binary_matrix, ord=0, axis=-1).mean())

    unique_indices = dataset['response'].drop_duplicates().index
    unique_binary_matrix = np.zeros((len(unique_indices), len(features)))
    aggregate_unique_labels = np.zeros(len(unique_indices))
    for i, idx in enumerate(unique_indices):
        unique_binary_matrix[i] = binary_matrix[idx]
        aggregate_unique_labels[i] = int(dataset.loc[idx, 'mean_rating'] > 0.5)
    X = convert_to_multiindex(pd.DataFrame(unique_binary_matrix, columns=features))
    y = aggregate_unique_labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    brcg = BRCGExplainer(BooleanRuleCG(
        lambda0=1e-05,
        lambda1=1e-04,
    ))
    brcg.fit(X, y)

    aggregate_rules = brcg.explain()['rules']

    save_path = os.path.join(args.output_dir, f"dnf_{args.dataset}_{args.category}_group_compared_rules.txt")
    with open(save_path, "w") as f:
        print("Aggregate Rules:", file=f)
        print(f"Accuracy: {(brcg.predict(X_test) == y_test).mean()}", file=f)
        print(format_rules(aggregate_rules), file=f)

        category = args.category

        detection_diffs_csv = os.path.join(args.output_dir, "detection_diffs_vs_aggregate.csv")
        if os.path.exists(detection_diffs_csv):
            detection_diff_df = pd.read_csv(detection_diffs_csv)
        else:
            detection_diff_df = pd.DataFrame(columns=['category', 'group', 'detection_diff'])
        detection_diffs = []
        unique_totals = []
        disagreement_totals = []
        precisions = []
        covered_totals = []
        for cat in dataset[category].unique():
            indices = dataset.loc[dataset[category] == cat].index
            group_dataset = dataset.loc[indices]

            unique_indices = dataset['response'].drop_duplicates().index
            unique_binary_matrix = np.zeros((len(unique_indices), len(features)))
            unique_labels = np.zeros(len(unique_indices))
            for i, idx in enumerate(unique_indices):
                response = dataset.loc[idx, 'response']
                annotations = group_dataset.loc[group_dataset['response'] == response]['Q_overall'].values
                mean_annotation = (np.mean(annotations == 'Yes') > 0.5).astype(int)
                unique_binary_matrix[i] = binary_matrix[idx]
                unique_labels[i] = mean_annotation

            X = unique_binary_matrix
            y = unique_labels
            X = convert_to_multiindex(pd.DataFrame(X, columns=features))
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            if len(np.unique(y)) < 2:
                print("Skipping due to only one class present")
                continue
            brcg = BRCGExplainer(BooleanRuleCG(
                lambda0=1e-5,
                lambda1=1e-4,
            ))
            brcg.fit(X, y)
            rules = brcg.explain()['rules']
            if len(rules) == 0 or len(rules[0].strip()) == 0:
                print("No rules found")
                continue

            print("\nRATER Rules:", file=f)
            print(dataset.loc[indices, category].unique()[0], file=f)
            print(format_rules(rules), file=f)
            print((brcg.predict(X_test) == y_test).mean(), file=f)

            print("Differences from aggregate rules:", file=f)
            diff_rule_list = diff_rules(aggregate_rules, rules)
            print(format_rules(diff_rule_list), file=f)
            for rule in diff_rule_list[:5]:
                examples = get_rule_examples(
                    rule=rule,
                    features=features,
                    dataset=dataset.loc[indices],
                    binary_matrix=binary_matrix[indices],
                    unique_on='response',
                    max_examples=5
                )

                print(f"Rule: {rule}", file=f)

                format_example_output(
                    examples_dict=examples,
                    dataset=dataset,
                    group_indices=indices,
                    include_context=True,
                    file=f
                )
            metrics = compute_rule_metrics(
                group_rules=rules,
                aggregate_rules=aggregate_rules,
                features=features,
                binary_matrix=unique_binary_matrix,
                labels=(unique_labels).astype(int) & (~aggregate_unique_labels.astype(int))
            )
            print(metrics, file=f)
            detection_diffs.append(metrics['p_unique'] - metrics['p_overlap'])
            unique_totals.append(metrics['unique_total'])
            disagreement_totals.append(metrics['disagree_total'])
            precisions.append(metrics['p_unique'])
            covered_totals.append(metrics['covered_total'])

            print("\n"+"-"*80+"\n", file=f)
            print("Aggregate rule differences from group rules:", file=f)

            diff_rule_list = diff_rules(rules, aggregate_rules)
            print(format_rules(diff_rule_list), file=f)
            for rule in diff_rule_list[:5]:
                examples = get_rule_examples(
                    rule=rule,
                    features=features,
                    dataset=dataset.loc[indices],
                    binary_matrix=binary_matrix[indices],
                    unique_on='response',
                    max_examples=5
                )

                print(f"Rule: {rule}", file=f)

                format_example_output(
                    examples_dict=examples,
                    dataset=dataset,
                    group_indices=indices,
                    include_context=True,
                    file=f
                )

            print("\n"+"="*80+"\n", file=f)
        detection_diffs = np.array(detection_diffs)
        unique_totals = np.array(unique_totals)
        disagreement_totals = np.array(disagreement_totals)
        precisions = np.array(precisions)
        dataframe = pd.DataFrame({'category': [category] * len(dataset[category].unique()),
                     'group': dataset[category].unique(),
                     'detection_diff': detection_diffs,
                     'unique_total': unique_totals,
                     'disagree_total': disagreement_totals,
                     'precision': precisions,
                     'covered_total': covered_totals})
        dataframe = pd.concat([detection_diff_df, dataframe], ignore_index=True)
        dataframe.to_csv(detection_diffs_csv, index=False)
