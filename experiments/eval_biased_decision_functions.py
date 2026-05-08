#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Evaluate decision functions on biased annotator datasets.

Requires annotation CSVs where multiple biased LLM annotators have labeled each text as
safe/unsafe. To generate these, use your own preferred LLM service with different system
prompts or biasing strategies to grade each text, then assemble into a CSV with columns:
text, ground_truth, annotator_0_label, annotator_1_label, ...

See DOCUMENTATION.md for the expected file formats and directory layout.
"""

from safe_dictionary_learning import Embedder, NonNegativeLogisticRegression, DNF
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc
)
import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/")
    parser.add_argument("--features", type=str, default="beavertails_drug_abuse_weapons_banned_substance_features.txt")
    parser.add_argument("--dataset", type=str, default="biased_beavertails_drug_abuse_weapons_banned_substance_annotations_full.csv")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    datasets_dir = os.path.join(args.data_root, "datasets")
    models_dir = os.path.join(args.data_root, "models")
    features_dir = os.path.join(args.data_root, "features")
    results_dir = os.path.join(args.data_root, "results")

    if "beavertails" in args.dataset:
        dataset="beavertails"
    elif "wildguard" in args.dataset:
        dataset="wildguard"

    features = []
    with open(os.path.join(features_dir, args.features)) as f:
        lines = f.readlines()
        for line in lines:
            features.append(line.strip())

    embedder = Embedder("gemini", device=args.device)
    embedder.load_data(os.path.join(datasets_dir, args.dataset.removeprefix("biased_")), key="text")
    embedder.load_features(features)

    binary_scores = embedder.get_binary_scores("sparsemax", alpha=5).numpy()
    # print(torch.linalg.norm(binary_scores.to(torch.float32), ord=0, dim=1).mean())

    df = pd.read_csv(os.path.join(datasets_dir, args.dataset))

    annotator_labels = df.drop(columns=["text", "ground_truth"]).columns.to_list()
    print(annotator_labels)

    annotators = []

    resultsdict = []
    for annotator in annotator_labels:
        annotators.append(annotator)
        filename = args.dataset.split(".")[0]+"_"+annotator
        model = annotator

        minority_size = df[annotator].value_counts().min()
        balanced_indices = df.groupby(annotator).sample(n=minority_size, random_state=42).index

        X_train, X_test, y_train, y_test = train_test_split(binary_scores[balanced_indices], df.loc[balanced_indices, annotator].to_numpy(), random_state=42)
        # X_train, X_test, y_train, y_test = train_test_split(binary_scores, df[annotator].to_numpy(), random_state=42)

        clf = NonNegativeLogisticRegression(X_train.shape[1], features, lam=0.02)
        clf.fit(X_train, y_train, max_iter=10000)

        clf_string = clf.get_string_representation()
        y_pred = clf.predict(X_test).cpu().numpy()
        y_pred_proba = clf.predict_proba(X_test)[:, 1]  # Probabilities for positive class

        # ROC curve components
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        acc = clf.score(X_test, y_test).item()
        print(f"{model} NNLR test accuracy: {acc}")
        print(f"{model} NNLR AUC: {roc_auc}")

        print(clf.get_string_representation())
        nnlrdict = {
            "Accuracy": acc,
            "AUC": roc_auc,
            "Model": clf.get_string_representation(),
        }

        clf = DNF(features, lambda0=0.001, lambda1=0.001, verbose=False)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        dnf_fpr = fp / (fp + tn)
        dnf_tpr = tp / (tp + fn)
        acc = clf.score(X_test, y_test)
        print(f"{model} DNF test accuracy: {acc}")
        print(f"{model} DNF TPR: {dnf_tpr}")
        print(clf.get_string_representation())

        dnfdict = {
            "Accuracy": acc,
            "TPR": dnf_tpr,
            "Model": clf.get_string_representation()
        }

        # Extract all FPR and TPR values
        fpr_points = np.array(fpr)
        tpr_points = np.array(tpr)

        datastring = args.dataset[:-4]

        if args.save:
            roc_dict = {
                "NNLR_tpr": tpr_points,
                "NNLR_fpr": fpr_points,
                "DNF_tpr": dnf_tpr,
                "DNF_fpr": dnf_fpr
            }
            with open(os.path.join(results_dir, f"{datastring}_{model}_decision_function_results.npy"), "wb") as f:
                np.save(f, tpr_points)
        resultsdict.append(
            {
                "model": model,
                "NNLR": nnlrdict,
                "DNF": dnfdict,
            }
        )

    if args.save:
        with open(os.path.join(results_dir, f"{datastring}_results.jsonl"), "w") as f:
            for entry in resultsdict:
                json.dump(entry, f)
                f.write("\n")
