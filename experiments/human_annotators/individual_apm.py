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
from safe_dictionary_learning.decision_functions.utils import format_rules, get_examples, diff_rules, convert_to_multiindex
from sklearn.model_selection import train_test_split

load_dotenv()
warnings.filterwarnings("ignore", category=UserWarning, module="cvxpy")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fit DNF rules per rater and compare to aggregate')
    parser.add_argument('--dataset', type=str, default="350")
    parser.add_argument('--category', type=str, default="rater_id",
                        choices=['rater_id', 'rater_age', 'rater_gender', 'rater_race', 'rater_education'])
    parser.add_argument('--features-csv', type=str, default="results/cluster5.csv")
    parser.add_argument('--output-dir', type=str, default="results")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = pd.read_csv(f"data/diverse_safety_adversarial_dialog_{args.dataset}_aggregated.csv")
    featuredf = pd.read_csv(args.features_csv)

    keys_df = pd.DataFrame({'response': dataset['response']})
    features_expanded = keys_df.join(featuredf.set_index('response'), on='response', how='left').drop('response', axis=1)
    features = features_expanded.columns.tolist()
    binary_matrix = features_expanded.to_numpy().astype(int)

    print(np.linalg.norm(binary_matrix, ord=0, axis=-1).mean())

    unique_indices = dataset['response'].drop_duplicates().index
    unique_binary_matrix = np.zeros((len(unique_indices), len(features)))
    unique_labels = np.zeros(len(unique_indices))
    for i, idx in enumerate(unique_indices):
        unique_binary_matrix[i] = binary_matrix[idx]
        unique_labels[i] = int(dataset.loc[idx, 'mean_rating'] > 0.5)
    X = convert_to_multiindex(pd.DataFrame(unique_binary_matrix, columns=features))
    y = unique_labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    brcg = BRCGExplainer(BooleanRuleCG(
        lambda0=1e-06,
        lambda1=1e-03,
    ))
    brcg.fit(X, y)

    aggregate_rules = brcg.explain()['rules']

    save_path = os.path.join(args.output_dir, f"dnf_{args.dataset}_{args.category}_compared_rules.txt")
    with open(save_path, "w") as f:
        print("Aggregate Rules:", file=f)
        print(f"Accuracy: {(brcg.predict(X_test) == y_test).mean()}", file=f)
        print(format_rules(aggregate_rules), file=f)

        for cat in dataset[args.category].unique():
            indices = dataset.loc[dataset[args.category] == cat].index

            X = binary_matrix[indices]
            print(X.shape)
            print(len(features))
            X = convert_to_multiindex(pd.DataFrame(X, columns=features))
            y = (dataset.loc[indices, 'Q_overall'] == 'Yes').astype(int).values
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
            print(dataset.loc[indices, 'rater_id'].unique()[0], file=f)
            print(dataset.loc[indices, 'rater_gender'].unique()[0], file=f)
            print(dataset.loc[indices, 'rater_race'].unique()[0], file=f)
            print(dataset.loc[indices, 'rater_age'].unique()[0], file=f)
            print(dataset.loc[indices, 'rater_education'].unique()[0], file=f)
            print(format_rules(rules), file=f)
            print((brcg.predict(X_test) == y_test).mean(), file=f)

            print("Differences from aggregate rules:", file=f)
            diff_rule_list = diff_rules(aggregate_rules, rules)
            print(format_rules(diff_rule_list), file=f)
            for rule in diff_rule_list[:5]:
                example_indices, example_features = get_examples(rule, features, binary_matrix[indices])
                example_texts = dataset.loc[indices].iloc[example_indices]['response'].unique()

                full_texts = []
                for text in example_texts:
                    convo = dataset.loc[indices].iloc[example_indices].loc[dataset['response'] == text].iloc[0]['context']
                    convo = ["\t" + line for line in convo.split("\n")]
                    full_texts.append("\n".join(convo)+"\n\tLAMBDA: "+text)
                print(f"Rule: {rule}", file=f)
                print("    Examples:", file=f)
                for i, text in enumerate(example_texts[:5]):
                    print(f"      - Mean: {dataset.loc[indices].loc[dataset['response'] == text].iloc[0]['mean_rating']}, Gold: {dataset.loc[indices].loc[dataset['response'] == text].iloc[0]['safety_gold']}, Rater: {dataset.loc[indices].loc[dataset['response'] == text].iloc[0]['Q_overall']}, \n\t{full_texts[i]}", file=f)
                    print("\t" + str(example_features[i]), file=f)

            print("\n"+"-"*80+"\n", file=f)
            print("Aggregate rule differences from group rules:", file=f)

            diff_rule_list = diff_rules(rules, aggregate_rules)
            print(format_rules(diff_rule_list), file=f)
            for rule in diff_rule_list[:5]:
                example_indices, example_features = get_examples(rule, features, binary_matrix[indices])
                example_texts = dataset.loc[indices].iloc[example_indices]['response'].unique()

                full_texts = []
                for text in example_texts:
                    convo = dataset.loc[indices].iloc[example_indices].loc[dataset['response'] == text].iloc[0]['context']
                    convo = ["\t" + line for line in convo.split("\n")]
                    full_texts.append("\n".join(convo)+"\n\tLAMBDA: "+text)
                print(f"Rule: {rule}", file=f)
                print("    Examples:", file=f)
                for i, text in enumerate(example_texts[:5]):
                    print(f"      - Mean: {dataset.loc[indices].loc[dataset['response'] == text].iloc[0]['mean_rating']}, Gold: {dataset.loc[indices].loc[dataset['response'] == text].iloc[0]['safety_gold']}, Rater: {dataset.loc[indices].loc[dataset['response'] == text].iloc[0]['Q_overall']}, \n\t{full_texts[i]}", file=f)
                    print("\t" + str(example_features[i]), file=f)

            print("\n"+"="*80+"\n", file=f)
