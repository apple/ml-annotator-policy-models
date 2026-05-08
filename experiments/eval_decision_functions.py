#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Evaluate decision functions (NNLR, DNF) on annotated safety datasets.

Requires annotation CSVs where multiple LLM annotators have labeled each text as safe/unsafe,
along with a companion models JSON mapping annotator IDs to model names. To generate these,
use your own preferred LLM service to grade each text in the base dataset, then assemble into
a CSV with columns: text, ground_truth, annotator_0_label, annotator_1_label, ... and a
JSON file mapping annotator IDs to model names.

See DOCUMENTATION.md for the expected file formats and directory layout.
"""

from safe_dictionary_learning import Embedder, NonNegativeLogisticRegression, DNF
import json
import pandas as pd
import numpy as np
import altair as alt
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
    parser.add_argument("--dataset", type=str, default="beavertails_drug_abuse_weapons_banned_substance_annotations_full.csv")
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
    embedder.load_data(os.path.join(datasets_dir, args.dataset), key="text")
    embedder.load_features(features)

    binary_scores = embedder.get_binary_scores("sparsemax", alpha=5).numpy() # n x d
    # print(torch.linalg.norm(binary_scores.to(torch.float32), ord=0, dim=1).mean())

    df = pd.read_csv(os.path.join(datasets_dir, args.dataset))
    annotator_path = args.dataset[:-4].split("_10k")[0] + "_models.json"
    full_annotator_path = os.path.join(datasets_dir, annotator_path)
    with open(full_annotator_path) as f:
        annotator_to_model = json.loads(f.read())
    annotator_labels = [f"annotator_{a}_label" for a in annotator_to_model.keys()]

    minority_size = df["ground_truth"].value_counts().min()
    balanced_indices = df.groupby('ground_truth').sample(n=minority_size, random_state=42).index

    annotators = []

    resultsdict = []
    for annotator in annotator_labels + ["ground_truth"]:
        if annotator.startswith("annotator"):
            id = annotator.split("_")[1]
            annotators.append(annotator_to_model[id])
            filename = args.dataset.split(".")[0]+"_"+annotator_to_model[id]
            model = annotator_to_model[id]
        else:
            annotators.append(annotator)
            filename = args.dataset.split(".")[0]+"_"+annotator
            model = annotator

        X_train, X_test, y_train, y_test = train_test_split(binary_scores[balanced_indices], df.loc[balanced_indices, annotator].to_numpy(), random_state=42)

        clf = NonNegativeLogisticRegression(X_train.shape[1], features, lam=0.01)
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

        clf = DNF(features, lambda0=0.001, lambda1=0.0001, verbose=False)
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

        # Create the full curve by adding endpoints and sorting
        # Add the endpoints (0,0) and (1,1)
        full_fpr = np.concatenate([[0], fpr_points, [1]])
        full_tpr = np.concatenate([[0], tpr_points, [1]])

        # Sort by FPR to create a proper ROC curve
        sort_indices = np.argsort(full_fpr)
        sorted_fpr = full_fpr[sort_indices]
        sorted_tpr = full_tpr[sort_indices]

        # Calculate AUC using trapezoidal rule
        auc_value = np.trapz(sorted_tpr, sorted_fpr)


        # Create DataFrames for plotting
        # ROC curve line data
        roc_line_df = pd.DataFrame({
            'fpr': sorted_fpr,
            'tpr': sorted_tpr
        })

        # All points data (including endpoints)
        scatter_df = pd.DataFrame({
            'fpr': full_fpr,
            'tpr': full_tpr
        })

        # Diagonal line data
        diagonal_df = pd.DataFrame({
            'fpr': [0, 1],
            'tpr': [0, 1]
        })

        # Add DNF points
        dnf_df = pd.DataFrame({
            'fpr': [dnf_fpr],
            'tpr': [dnf_tpr]
        })

        # Create the ROC curve line
        roc_line = alt.Chart(roc_line_df).mark_line(
            color='#0A84FF',
            strokeWidth=3,
            opacity=1
        ).encode(
            x=alt.X('fpr:Q', scale=alt.Scale(domain=[0, 1])),
            y=alt.Y('tpr:Q', scale=alt.Scale(domain=[0, 1]))
        )

        # Create scatter points
        scatter_points = alt.Chart(scatter_df).mark_circle(
            color='#0A84FF',
            size=30,
            opacity=1
        ).encode(
            x=alt.X('fpr:Q', title="FPR"),
            y=alt.Y('tpr:Q', title="TPR"),
        )

        dnf_points = alt.Chart(dnf_df).mark_circle(
            color='red',
            size=30,
            opacity=1
        ).encode(
            x=alt.X('fpr:Q', title="FPR"),
            y=alt.Y('tpr:Q', title="TPR")
        )

        # Create diagonal line for random classifier
        diagonal_line = alt.Chart(diagonal_df).mark_line(
            color='white',
            strokeDash=[5, 5],
            opacity=0.5
        ).encode(
            x='fpr:Q',
            y='tpr:Q'
        )

        # Combine all layers
        chart = (diagonal_line + roc_line + scatter_points + dnf_points).properties(
            width=400,
            height=320,
            title=f'ROC Curve for {model} - AUC = {auc_value:.3f}',
            background="black"
        ).configure_axis(
            labelColor='white',
            titleColor='white',
            grid=False
        ).configure_title(
            color='white'
        ).configure_view(
            strokeWidth=0
        )

        chart.save(f'figures/roc_{dataset}_{model}.png', format='png', ppi=300)
    if args.save:
        with open(os.path.join(results_dir, f"{datastring}_results.jsonl"), "w") as f:
            for entry in resultsdict:
                json.dump(entry, f, indent=2)
                f.write("\n")
