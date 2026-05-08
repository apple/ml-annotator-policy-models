#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import numpy as np
import pandas as pd


def parse_nnlr_string(nnlr_string):
    lines = nnlr_string.strip().split("\n")

    # Parse each line
    features = []
    for line in lines[2:]:
        line = line.strip()
        feature = line.split(":")[0]
        features.append(feature)
    return features


def parse_dnf_string(dnf_string):
    clauses = dnf_string.split("OR")
    features = []
    for clause in clauses:
        clause = " ".join(clause.strip().split(" ")[:-1]).strip()
        separated = clause.split(" AND ")
        separated.sort()
        joined = " AND ".join(separated)
        features.append(joined)
    return features


def convert_to_multiindex(df):
    """Convert binary DataFrame to MultiIndex format for BooleanRuleCG"""
    new_columns = pd.MultiIndex.from_tuples(
        [(col, "==", "present") for col in df.columns], names=["feature", "operator", "threshold"]
    )
    df_multiindex = df.copy()
    df_multiindex.columns = new_columns
    return df_multiindex


def format_rules(rules):
    formatted_rules = ""
    for rule in rules:
        predicates = rule.split(" AND ")
        rule = " AND ".join([p.split(" == ")[0].strip() for p in predicates])
        formatted_rules += rule + " OR\n"
    return formatted_rules


def diff_rules(rule1, rule2):
    """Returns rules in rule2 that are not in rule1, preserving rule2's order."""
    set1 = set([" AND ".join(sorted(r1.split(" AND "))) for r1 in rule1])
    set2 = set([" AND ".join(sorted(r2.split(" AND "))) for r2 in rule2])
    diff = set2 - set1

    out = []
    for rule in rule2:
        sorted_rule = " AND ".join(sorted(rule.split(" AND ")))
        if sorted_rule in diff:
            out.append(rule)

    return out


def intersect_rules(rule1, rule2):
    """Returns rules in rule2 that are also in rule1, preserving rule2's order."""
    set1 = set([" AND ".join(sorted(r1.split(" AND "))) for r1 in rule1])
    set2 = set([" AND ".join(sorted(r2.split(" AND "))) for r2 in rule2])
    common = set2 & set1

    out = []
    for rule in rule2:
        sorted_rule = " AND ".join(sorted(rule.split(" AND ")))
        if sorted_rule in common:
            out.append(rule)

    return out


def get_examples(rule, features, binary_matrix):
    feature_conditions = rule.split(" AND ")
    feature_terms = [fc.split(" == ")[0] for fc in feature_conditions]
    indices = [features.index(term) for term in feature_terms]
    data_indices = np.where(np.all(binary_matrix[:, indices] == 1, axis=1))[0]

    example_features = []
    for index in data_indices:
        example_indices = np.where(binary_matrix[index, :] == 1)[0]
        example_features.append([features[i] for i in example_indices])
    return data_indices, example_features


def get_ruleset_coverage(rules, features, binary_matrix):
    """Get boolean mask of samples covered by any rule in the set."""
    covered = np.zeros(len(binary_matrix), dtype=bool)
    for rule in rules:
        if not rule.strip():
            continue
        feature_conditions = rule.split(" AND ")
        required_features = [fc.split(" == ")[0].strip() for fc in feature_conditions]
        required_indices = [features.index(feat) for feat in required_features]
        matching_mask = np.all(binary_matrix[:, required_indices] == 1, axis=1)
        covered |= matching_mask
    return covered


def compute_rule_metrics(group_rules, aggregate_rules, features, binary_matrix, labels):
    """Compute novel coverage rate, Cohen's h, and lift.

    Returns dict with keys: novel_coverage_rate, cohens_h, lift, novel, overlap,
    p_unique, p_overlap, unique_ratio, unique_total, covered_total, disagree_rate, disagree_total.
    """
    group_covered = get_ruleset_coverage(group_rules, features, binary_matrix)
    aggregate_covered = get_ruleset_coverage(aggregate_rules, features, binary_matrix)

    novel = group_covered & ~aggregate_covered
    overlap = group_covered & aggregate_covered

    novel_coverage_rate = novel.sum() / len(labels)

    cohens_h = None
    lift = None

    p_unique = labels[novel].mean() if novel.sum() > 0 else 0
    p_overlap = labels[overlap].mean() if overlap.sum() > 0 else 0
    unique_ratio = 0

    if overlap.sum() > 0 and novel.sum() > 0:
        p_unique = labels[novel].mean()
        p_overlap = labels[overlap].mean()
        unique_ratio = labels[novel].sum() / labels.sum()

        cohens_h = 2 * (np.arcsin(np.sqrt(p_unique)) - np.arcsin(np.sqrt(p_overlap)))
        lift = p_unique / p_overlap if p_overlap > 0 else float("inf")

    return {
        "novel_coverage_rate": novel_coverage_rate,
        "cohens_h": cohens_h,
        "lift": lift,
        "novel": novel.sum(),
        "overlap": overlap.sum(),
        "p_unique": p_unique if p_unique is not None else 0,
        "p_overlap": p_overlap if p_overlap is not None else 0,
        "unique_ratio": unique_ratio,
        "unique_total": labels[novel].sum(),
        "covered_total": labels[group_covered].sum(),
        "disagree_rate": labels.mean(),
        "disagree_total": labels.sum(),
    }


def get_rule_examples(rule, features, dataset, binary_matrix, unique_on="response", max_examples=5):
    """Extract examples from dataset that satisfy a given rule."""
    feature_conditions = rule.split(" AND ")
    required_features = [fc.split(" == ")[0].strip() for fc in feature_conditions]
    required_indices = [features.index(feat) for feat in required_features]

    matching_mask = np.all(binary_matrix[:, required_indices] == 1, axis=1)
    matching_data_indices = np.where(matching_mask)[0]

    if len(matching_data_indices) == 0:
        return {"indices": [], "unique_examples": pd.DataFrame(), "features_per_example": []}

    subset_df = dataset.iloc[matching_data_indices].reset_index(drop=True)

    unique_mask = ~subset_df[unique_on].duplicated()
    unique_examples = subset_df[unique_mask].head(max_examples)

    unique_data_indices = matching_data_indices[unique_mask][:max_examples]
    features_per_example = []
    for idx in unique_data_indices:
        active_feature_indices = np.where(binary_matrix[idx, :] == 1)[0]
        features_per_example.append([features[i] for i in active_feature_indices])

    return {
        "indices": matching_data_indices,
        "unique_examples": unique_examples,
        "features_per_example": features_per_example,
    }


def format_example_output(
    examples_dict,
    dataset,
    include_context=True,
    group_indices=None,
    group_name="Group",
    second_group_indices=None,
    second_group_name="Second Group",
    file=None,
):
    """Format and print examples in a readable format. Supports optional second group comparison."""
    unique_examples = examples_dict["unique_examples"]
    features_per_example = examples_dict["features_per_example"]

    if unique_examples.empty:
        print("No examples found", file=file)
        return

    for i, (idx, row) in enumerate(unique_examples.iterrows()):
        response = row["response"]

        if group_indices is not None:
            group_data = dataset.loc[group_indices]
            group_responses = group_data[group_data["response"] == response]
            group_score = (group_responses["Q_overall"] == "Yes").astype(int).mean()
            group_score_str = f", {group_name} Score: {group_score:.2f}"
        else:
            group_score_str = ""

        if second_group_indices is not None:
            second_group_data = dataset.loc[second_group_indices]
            second_group_responses = second_group_data[second_group_data["response"] == response]
            second_group_score = (second_group_responses["Q_overall"] == "Yes").astype(int).mean()
            second_group_score_str = f", {second_group_name} Score: {second_group_score:.2f}"
        else:
            second_group_score_str = ""

        metadata = f"Mean: {row.get('mean_rating', 'N/A')}"
        if "safety_gold" in row:
            metadata += f", Gold: {row['safety_gold']}"
        metadata += group_score_str
        metadata += second_group_score_str

        print(f"  Example {i + 1}:", file=file)
        print(f"    {metadata}", file=file)

        if include_context and "context" in row:
            context_lines = ["\t" + line for line in row["context"].split("\n")]
            print("\t" + "\n\t".join(context_lines), file=file)
            print(f"\tLAMBDA: {response}", file=file)
        else:
            print(f"\t{response}", file=file)

        if i < len(features_per_example):
            print(f"\tFeatures: {features_per_example[i]}", file=file)
        print(file=file)
