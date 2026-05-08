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
                       intersect_rules, convert_to_multiindex)
from sklearn.model_selection import train_test_split

load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fit DNF rules for each group and compare pairwise')
    parser.add_argument('--dataset', type=str, default="350")
    parser.add_argument('--category', type=str, default="rater_education",
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

    save_path = os.path.join(args.output_dir, f"dnf_{args.dataset}_{args.category}_intergroup_compared_rules.txt")
    with open(save_path, "w") as f:
        category = args.category

        brcgs = []
        ruleslist = []
        group = []
        group_indices = []
        group_labels = []
        group_data = []

        dataset['predictions'] = np.zeros(len(dataset))
        for cat in dataset[category].unique():
            indices = dataset.loc[dataset[category] == cat].index
            group_indices.append(indices)
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
            group_data.append(X)
            group_labels.append(y)
            X = convert_to_multiindex(pd.DataFrame(X, columns=features))
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            if len(np.unique(y)) < 2:
                print("Skipping due to only one class present")
                continue
            brcg = BRCGExplainer(BooleanRuleCG(
                lambda0=1e-4,
                lambda1=1e-5,
            ))
            brcg.fit(X, y)
            preds = brcg.predict(X)
            mapping = {response: pred for response, pred in zip(dataset.loc[unique_indices, 'response'], preds)}
            dataset.loc[indices, 'predictions'] = dataset.loc[indices, 'response'].map(mapping)

            rules = brcg.explain()['rules']
            brcgs.append(brcg)
            ruleslist.append(rules)
            group.append(cat)

            print("\nRATER Rules:", file=f)
            print(dataset.loc[indices, category].unique()[0], file=f)
            print(format_rules(rules), file=f)
            print((brcg.predict(X_test) == y_test).mean(), file=f)

        detection_diff_matrix = np.zeros((len(ruleslist), len(ruleslist)))
        unique_ratio_matrix = np.zeros((len(ruleslist), len(ruleslist)))
        precision_matrix = np.zeros((len(ruleslist), len(ruleslist)))
        for i in range(len(ruleslist)):
            for j in range(i+1, len(ruleslist)):
                print(f"\nComparing {group[i]} and {group[j]}:", file=f)
                print(f"Rules in {group[i]} but not in {group[j]}:", file=f)
                diff_rule_list = diff_rules(ruleslist[j], ruleslist[i])
                print(format_rules(diff_rule_list), file=f)

                indices_i = group_indices[i]
                indices_j = group_indices[j]
                for rule in diff_rule_list[:5]:
                    examples = get_rule_examples(
                        rule=rule,
                        features=features,
                        dataset=dataset.loc[indices_i],
                        binary_matrix=binary_matrix[indices_i],
                        unique_on='response',
                        max_examples=8
                    )

                    print(f"Rule: {rule}", file=f)
                    format_example_output(
                        examples_dict=examples,
                        dataset=dataset,
                        include_context=True,
                        group_indices=indices_i,
                        group_name=group[i],
                        second_group_indices=indices_j,
                        second_group_name=group[j],
                        file=f
                    )

                metrics = compute_rule_metrics(
                    group_rules=ruleslist[i],
                    aggregate_rules=ruleslist[j],
                    features=features,
                    binary_matrix=group_data[i],
                    labels=group_labels[i].astype(int) & ~group_labels[j].astype(int)
                )
                print(metrics, file=f)

                detection_diff_matrix[i, j] = metrics['p_unique'] - metrics['p_overlap']
                unique_ratio_matrix[i, j] = metrics['unique_ratio']
                precision_matrix[i, j] = metrics['p_unique']

                print(f"Rules in {group[j]} but not in {group[i]}:", file=f)
                diff_rule_list = diff_rules(ruleslist[i], ruleslist[j])
                print(format_rules(diff_rule_list), file=f)

                for rule in diff_rule_list[:5]:
                    examples = get_rule_examples(
                        rule=rule,
                        features=features,
                        dataset=dataset.loc[indices_j],
                        binary_matrix=binary_matrix[indices_j],
                        unique_on='response',
                        max_examples=8
                    )

                    print(f"Rule: {rule}", file=f)

                    format_example_output(
                        examples_dict=examples,
                        dataset=dataset,
                        include_context=True,
                        group_indices=indices_j,
                        group_name=group[j],
                        second_group_indices=indices_i,
                        second_group_name=group[i],
                        file=f
                    )
                metrics = compute_rule_metrics(
                    group_rules=ruleslist[j],
                    aggregate_rules=ruleslist[i],
                    features=features,
                    binary_matrix=group_data[j],
                    labels=~group_labels[i].astype(int) & group_labels[j].astype(int)
                )
                detection_diff_matrix[j, i] = metrics['p_unique'] - metrics['p_overlap']
                unique_ratio_matrix[j, i] = metrics['unique_ratio']
                precision_matrix[j,i] = metrics['p_unique']
                print(metrics, file=f)

                intersect_rules_list = intersect_rules(ruleslist[i], ruleslist[j])
                print(f"Rules in both {group[i]} and {group[j]}:", file=f)
                print(format_rules(intersect_rules_list), file=f)

                for rule in intersect_rules_list[:5]:
                    examples = get_rule_examples(
                        rule=rule,
                        features=features,
                        dataset=dataset.loc[indices_j],
                        binary_matrix=binary_matrix[indices_j],
                        unique_on='response',
                        max_examples=8
                    )

                    print(f"Rule: {rule}", file=f)

                    format_example_output(
                        examples_dict=examples,
                        dataset=dataset,
                        include_context=True,
                        group_indices=indices_j,
                        group_name=group[j],
                        second_group_indices=indices_i,
                        second_group_name=group[i],
                        file=f
                    )

        x, y = np.meshgrid(group, group)
        dataframe = pd.DataFrame({'group0': x.ravel(),
                     'group1': y.ravel(),
                     'detection_diff': detection_diff_matrix.ravel(),
                     'unique_ratio': unique_ratio_matrix.ravel(),
                     'precision': precision_matrix.ravel()})
        dataframe.to_csv(os.path.join(args.output_dir, f"detection_diff_matrix_{category}.csv"))
