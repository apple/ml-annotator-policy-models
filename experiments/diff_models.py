#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import json
import os
import argparse

from safe_dictionary_learning import parse_nnlr_string, parse_dnf_string

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="data/")
    parser.add_argument("--features", type=str, default="beavertails_drug_abuse_weapons_banned_substance_features.txt")
    parser.add_argument("--dataset", type=str, default="beavertails_drug_abuse_weapons_banned_substance_annotations_full.csv")
    parser.add_argument("--oracle", type=str, default="gpt-4o")
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    results_dir = os.path.join(args.data_root, "results")

    datastring = args.dataset[:-4]
    print(datastring)
    with open(os.path.join(results_dir, f"{datastring}_results.jsonl")) as f:
        content = f.read()
    models = []
    # Split by '}\n{' pattern to separate objects
    json_strings = content.strip().split('}\n{')

    for i, json_str in enumerate(json_strings):
        # Add back the braces that were removed by splitting
        if i == 0:
            json_str = json_str + '}'
        elif i == len(json_strings) - 1:
            json_str = '{' + json_str
        else:
            json_str = '{' + json_str + '}'

        # Remove extra whitespace and parse
        json_str = json_str.strip()
        models.append(json.loads(json_str))


    ## diff NNLR
    modelstrings = {}
    for model in models:
        if model["model"] != args.oracle:
            modelstrings[model["model"]] = model["NNLR"]["Model"]
        else:
            oracle_model = model["NNLR"]["Model"]
    oracle_features = set(parse_nnlr_string(oracle_model))
    nnlr_outfiles = []
    for annotator in modelstrings.keys():
        modelstr = modelstrings[annotator]
        features = set(parse_nnlr_string(modelstr))
        nnlr_outfiles.append({
            "model": annotator,
            "diff": list(oracle_features.difference(features))
        })

    ## diff DNF
    modelstrings = {}
    for model in models:
        if model["model"] != args.oracle:
            modelstrings[model["model"]] = model["DNF"]["Model"]
        else:
            oracle_model = model["DNF"]["Model"]
    oracle_features = set(parse_dnf_string(oracle_model))
    dnf_outfiles = []
    for annotator in modelstrings.keys():
        modelstr = modelstrings[annotator]
        features = set(parse_dnf_string(modelstr))
        dnf_outfiles.append({
            "model": annotator,
            "diff": list(oracle_features.difference(features))
        })


    resultsdict = {
        "oracle": args.oracle,
        "NNLR": nnlr_outfiles,
        "DNF": dnf_outfiles,
    }


    if args.save:
        with open(os.path.join(results_dir, f"{datastring}_{args.oracle}_diffs.jsonl"), "w") as f:
            json.dump(resultsdict, f, indent=2)
