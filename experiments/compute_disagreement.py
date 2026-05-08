#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Compute annotator agreement confusion matrices from annotated safety datasets.

Requires annotation CSVs where multiple LLM annotators have labeled each text as safe/unsafe,
along with a companion models JSON mapping annotator IDs to model names. To generate these,
use your own preferred LLM service to grade each text in the base dataset, then assemble into
a CSV with columns: text, ground_truth, annotator_0_label, annotator_1_label, ... and a
JSON file mapping annotator IDs to model names.

See DOCUMENTATION.md for the expected file formats and directory layout.
"""

import argparse
import json
import os

import altair as alt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/")
    parser.add_argument(
        "--features",
        type=str,
        default="beavertails_drug_abuse_weapons_banned_substance_features.txt",
    )
    parser.add_argument(
        "--dataset", type=str, default="beavertails_drug_abuse_weapons_banned_substance_annotations_full.csv"
    )
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    datasets_dir = os.path.join(args.data_root, "datasets")
    results_dir = os.path.join(args.data_root, "results")

    if "beavertails" in args.dataset:
        datasetstr = "beavertails"
    elif "wildguard" in args.dataset:
        datasetstr = "wildguard"

    df = pd.read_csv(os.path.join(datasets_dir, args.dataset))
    annotator_path = args.dataset[:-4].split("_10k")[0] + "_models.json"
    full_annotator_path = os.path.join(datasets_dir, annotator_path)
    with open(full_annotator_path) as f:
        annotator_to_model = json.loads(f.read())
    annotator_labels = [f"annotator_{a}_label" for a in annotator_to_model.keys()] + [
        "ground_truth"
    ]
    annotators = list(annotator_to_model.values()) + ["ground_truth"]

    agreementmat = np.zeros((len(annotator_labels), len(annotator_labels)))
    agreements = []
    for i, annotator in enumerate(annotator_labels):
        print(annotator)
        for j, annotator2 in enumerate(annotator_labels):
            agree = df.loc[df[annotator] == df[annotator2]]
            agreement_rate = len(agree) / len(df)
            print(agreement_rate)
            agreementmat[i, j] = agreement_rate
            agreements.append(
                {
                    "Annotator 0": annotators[i],
                    "Annotator 1": annotators[j],
                    "agreement": agreement_rate,
                }
            )

    # Print ground truth comparison with all models
    print("\n" + "=" * 60)
    print("GROUND TRUTH COMPARISON WITH ALL MODELS")
    print("=" * 60)
    gt_index = annotator_labels.index("ground_truth")
    for i, (label, model_name) in enumerate(zip(annotator_labels, annotators)):
        if label != "ground_truth":
            agreement = agreementmat[gt_index, i]
            print(f"{model_name:50s} | Agreement: {agreement:.4f} ({agreement * 100:.2f}%)")
    print("=" * 60 + "\n")

    plotdf = pd.DataFrame(agreements)
    chart = (
        alt.Chart(plotdf)
        .mark_rect()
        .encode(
            x=alt.X("Annotator 0:O"),  # :O means ordinal
            y=alt.Y("Annotator 1:O"),
            color=alt.Color(
                "agreement:Q",  # :Q means quantitative
                scale=alt.Scale(scheme="blues"),  # use a predefined color scheme
                title="Agreement Rate",
            ),
        )
    )
    chart.save(f"figures/agreements_{datasetstr}.png", format="png", ppi=300)

    if args.save:
        plotdf.to_csv(os.path.join(results_dir, f"{args.dataset}_agreement.csv"))
        with open(os.path.join(results_dir, f"{args.dataset}_agreement.npy"), "wb") as f:
            np.save(f, agreementmat)
