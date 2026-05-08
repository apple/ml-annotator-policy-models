#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Given a dataset and feature space, trains nonnegative log reg model, saves model, and saves
features for counterfactual study.

Requires annotation CSVs where multiple LLM annotators have labeled each text as safe/unsafe,
along with a companion models JSON mapping annotator IDs to model names. To generate these,
use your own preferred LLM service to grade each text in the base dataset, then assemble into
a CSV with columns: text, ground_truth, annotator_0_label, annotator_1_label, ... and a
JSON file mapping annotator IDs to model names.

This script outputs CSVs with columns: text, annotator_label, predicted_label, features,
decision_function. To complete the counterfactual pipeline, a separate step must:
  1. Take each unsafe-predicted text and its triggering features/decision function.
  2. Prompt an LLM to rewrite the text removing the triggering features (generating a
     counterfactual).
  3. Re-label the counterfactual text (safe/unsafe) using the same annotator or model.
  4. Add `counterfactual` (rewritten text) and `counterfactual_label` columns to the CSV.

The resulting CSVs are then consumed by analyze_counterfactuals.py.

Note: For this script, use your own preferred LLM service for generating counterfactual rewrites and labels.
"""

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from safe_dictionary_learning import DNF, Embedder, NonNegativeLogisticRegression

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/")
    parser.add_argument("--embedding-model", type=str, default="gemini")
    parser.add_argument(
        "--dataset", type=str, default="beavertails_drug_abuse_weapons_banned_substance_annotations_full.csv"
    )
    parser.add_argument("--features", type=str, default="beavertails_drug_abuse_weapons_banned_substance_features.txt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--decision-function", type=str, default="NNLR")
    args = parser.parse_args()

    datasets_dir = os.path.join(args.data_root, "datasets")
    models_dir = os.path.join(args.data_root, "models")
    features_dir = os.path.join(args.data_root, "features")
    results_dir = os.path.join(args.data_root, "results")

    embedder = Embedder(args.embedding_model, device=args.device)
    embedder.load_data(os.path.join(datasets_dir, args.dataset), key="text")

    featurename = args.features.removesuffix(".txt")

    features = []
    featurepath = os.path.join(features_dir, featurename + ".txt")
    with open(featurepath) as f:
        lines = f.readlines()
        for line in lines:
            features.append(line.strip())

    embedder.load_features(features)

    binary_scores = embedder.get_binary_scores("sparsemax", alpha=5.0).numpy()

    df = pd.read_csv(os.path.join(datasets_dir, args.dataset))
    annotator_path = args.dataset[:-4].split("_10k")[0] + "_models.json"
    full_annotator_path = os.path.join(datasets_dir, annotator_path)
    with open(full_annotator_path) as f:
        annotator_to_model = json.loads(f.read())
    annotator_labels = [f"annotator_{a}_label" for a in annotator_to_model.keys()]

    minority_size = df["ground_truth"].value_counts().min()
    balanced_indices = df.groupby("ground_truth").sample(n=minority_size, random_state=42).index

    unsafe_df = df.loc[df.ground_truth == 1]
    n_unsafe_sample = min(10000, len(unsafe_df))
    df_10k_unsafe = unsafe_df.sample(n=n_unsafe_sample, random_state=42)
    df_10k_unsafe_indices = df_10k_unsafe.index

    ## for each annotator, fit probe
    annotators = []
    for annotator in annotator_labels + ["ground_truth"]:
        if annotator.startswith("annotator"):
            id = annotator.split("_")[1]
            annotators.append(annotator_to_model[id])
            filename = (
                args.dataset.split(".")[0]
                + "_"
                + annotator_to_model[id]
                + "_"
                + args.decision_function
            )
        else:
            annotators.append(annotator)
            filename = args.dataset.split(".")[0] + "_" + annotator + "_" + args.decision_function

        # scores_balanced, annotations_balanced = sample_balanced_subset(binary_scores, df[annotator].to_numpy(), random_state=42)

        X_train, X_test, y_train, y_test = train_test_split(
            binary_scores[balanced_indices],
            df.loc[balanced_indices, annotator].to_numpy(),
            random_state=42,
        )

        if args.decision_function == "NNLR":
            clf = NonNegativeLogisticRegression(binary_scores.shape[1], features, lam=0.01)

        elif args.decision_function == "DNF":
            clf = DNF(features, lambda0=0.001, lambda1=0.0001)

        clf.fit(X_train, y_train)
        print(f"{annotator} test accuracy:", clf.score(X_test, y_test))
        clf_string = clf.get_string_representation()
        feature_indices_used = clf.get_features_used()
        # model_weights = get_model_weights(clf)

        y_preds = clf.predict(binary_scores)

        ## intersection of features present in data and model weights
        final_features = []
        final_preds = []
        for i, row in df_10k_unsafe.iterrows():
            active_features = set(np.argwhere(binary_scores[i, :] > 0).flatten().tolist())
            feature_indices_used = set(feature_indices_used)
            final_features.append(
                json.dumps([features[j] for j in active_features & feature_indices_used])
            )
            final_preds.append(y_preds[i])

        ## just model weights
        # final_features = []
        # for i, row in df.iterrows():
        #     final_features.append(json.dumps([features[i] for i in np.argwhere(model_weights != 0).flatten()]))
        if args.decision_function == "NNLR":
            with open(os.path.join(models_dir, filename + ".pt"), "wb") as f:
                torch.save(clf.state_dict(), f)
        else:
            with open(os.path.join(models_dir, filename + ".pkl"), "wb") as f:
                pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)

        df_counterfactuals = pd.DataFrame(
            {
                "text": df_10k_unsafe.text,
                "annotator_label": df_10k_unsafe[annotator],
                "predicted_label": final_preds,
                "features": final_features,
                "decision_function": [clf_string] * len(df_10k_unsafe),
            }
        )
        # print(df_counterfactuals.iloc[0])
        df_counterfactuals.to_csv(
            os.path.join(datasets_dir, "counterfactuals", filename + ".csv"), index=False
        )
        df_counterfactuals.to_csv(
            os.path.join(datasets_dir, "counterfactuals", filename + ".csv"), index=False
        )
