#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
import torch

from safe_dictionary_learning import Embedder, NonNegativeLogisticRegression

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/")
    parser.add_argument(
        "--dataset",
        type=str,
        default="beavertails_drug_abuse_weapons_banned_substance_annotations_full.csv",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="beavertails_drug_abuse_weapons_banned_substance_features.txt",
    )
    parser.add_argument("--decision-function", type=str, default="NNLR")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    datasets_dir = os.path.join(args.data_root, "datasets")
    models_dir = os.path.join(args.data_root, "models")
    features_dir = os.path.join(args.data_root, "features")
    results_dir = os.path.join(args.data_root, "results")

    dataset = args.dataset[:-4]  # strip .csv

    # Load annotator model names from the companion JSON
    annotator_path = dataset.split("_10k")[0] + "_models.json"
    with open(os.path.join(datasets_dir, annotator_path)) as f:
        annotator_to_model = json.loads(f.read())
    models = list(annotator_to_model.values())

    featurename = args.features.removesuffix(".txt")
    features = []
    featurepath = os.path.join(features_dir, featurename + ".txt")
    with open(featurepath) as f:
        for line in f:
            features.append(line.strip())

    embedder = Embedder("gemini", device=args.device)
    embedder.load_data(
        os.path.join(
            datasets_dir,
            f"counterfactuals/{dataset}_counterfactual_annotations_{models[0]}_{args.decision_function}.csv",
        ),
        key="counterfactual",
    )
    embedder.load_features(features)

    faithfulness = []
    safe_flips = []
    original_unsafe = []
    new_unsafe = []
    for model in models:
        binary_scores = embedder.get_binary_scores("sparsemax", alpha=5.0).numpy()

        if args.decision_function == "NNLR":
            modelpath = dataset + "_" + model + "_" + args.decision_function + ".pt"
            clf = NonNegativeLogisticRegression(
                n_features=binary_scores.shape[1], features=features
            )
            with open(os.path.join(models_dir, modelpath), "rb") as f:
                clf.load_state_dict(torch.load(f, weights_only=True))
        elif args.decision_function == "DNF":
            modelpath = dataset + "_" + model + "_" + args.decision_function + ".pkl"
            with open(os.path.join(models_dir, modelpath), "rb") as f:
                clf = pickle.load(f)

        df = pd.read_csv(
            os.path.join(
                datasets_dir,
                f"counterfactuals/{dataset}_counterfactual_annotations_{model}_{args.decision_function}.csv",
            ),
            index_col=None,
        )
        unsafe_indices = df.index[df.predicted_label == 1].tolist()

        y_preds = clf.predict(binary_scores)
        if args.decision_function == "NNLR":
            y_preds = y_preds.numpy().astype(bool)

        faithful = (
            y_preds[unsafe_indices] == df.counterfactual_label.to_numpy()[unsafe_indices]
        ).astype(int)
        faithful = np.mean(faithful)
        safe_flip = df.counterfactual_label[unsafe_indices].mean()

        safe_flips.append(safe_flip)
        faithfulness.append(faithful)
        original_unsafe.append(df.predicted_label.sum())
        new_unsafe.append(df.counterfactual_label[unsafe_indices].sum())

        print(f"{model} faithfulness: ", faithful)
        print(f"{model} safe_flips: ", safe_flip)

    source = pd.DataFrame(
        {
            "faithfulness": faithfulness,
            "safe_flips": safe_flips,
            "original_unsafe": original_unsafe,
            "new_unsafe": new_unsafe,
            "models": models,
        }
    )

    if args.save:
        source.to_csv(
            os.path.join(
                results_dir,
                f"{dataset}_{args.decision_function}_counterfactual_results.csv",
            ),
            index=False,
        )
