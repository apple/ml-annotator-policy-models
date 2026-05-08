#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

"""Download HuggingFace datasets and convert to the CSV format expected by the pipeline."""

import argparse
import os

import pandas as pd


def load_beavertails(
    categories: list[str] | None = None,
    split: str = "330k_train",
    n_samples: int | None = None,
) -> pd.DataFrame:
    """Load BeaverTails from HuggingFace and return a DataFrame with `text` and `ground_truth`.

    Args:
        categories: Category names to filter on (e.g. ["drug_abuse", "weapons"]).
            Rows where any specified category is True are kept, plus safe examples.
            If None, all rows are included.
        split: HuggingFace dataset split to load. Available splits:
            ``330k_train``, ``330k_test``, ``30k_train``, ``30k_test``.
        n_samples: Optional limit on the number of rows returned.

    Returns:
        DataFrame with columns ``text`` (response text) and ``ground_truth`` (1=unsafe, 0=safe).

    """
    from datasets import load_dataset as _load_dataset  # noqa: PLC0415

    ds = _load_dataset("PKU-Alignment/BeaverTails", split=split)
    df = ds.to_pandas()

    if categories:
        # category column is a dict of {category_name: bool}
        mask_unsafe = df["category"].apply(lambda cat: any(cat.get(c, False) for c in categories))
        mask_safe = df["is_safe"]
        df = df[mask_unsafe | mask_safe].copy()

    df = df.rename(columns={"response": "text"})
    df["ground_truth"] = (~df["is_safe"]).astype(int)
    df = df[["text", "ground_truth"]].reset_index(drop=True)

    if n_samples is not None:
        df = df.sample(n=min(n_samples, len(df)), random_state=42).reset_index(drop=True)

    return df


def load_wildguardmix(
    split: str = "train",
    n_samples: int | None = None,
) -> pd.DataFrame:
    """Load WildGuardMix from HuggingFace and return a DataFrame with `text` and `ground_truth`.

    Args:
        split: HuggingFace dataset split to load.
        n_samples: Optional limit on the number of rows returned.

    Returns:
        DataFrame with columns ``text`` and ``ground_truth`` (1=harmful, 0=benign).

    """
    from datasets import load_dataset as _load_dataset  # noqa: PLC0415

    ds = _load_dataset("allenai/wildguardmix", "wildguardtrain", split=split)
    df = ds.to_pandas()

    df = df.rename(columns={"prompt": "text"})
    df["ground_truth"] = (df["prompt_harm_label"] == "harmful").astype(int)
    df = df[["text", "ground_truth"]].reset_index(drop=True)

    if n_samples is not None:
        df = df.sample(n=min(n_samples, len(df)), random_state=42).reset_index(drop=True)

    return df


def _build_output_filename(dataset: str, categories: list[str] | None) -> str:
    """Build a descriptive CSV filename from the dataset name and category filter."""
    parts = [dataset]
    if categories:
        # Replace commas in category keys with underscores for clean filenames
        parts.extend(c.replace(",", "_") for c in categories)
    else:
        parts.append("all")
    return "_".join(parts) + ".csv"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a HuggingFace dataset and save as CSV.")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["beavertails", "wildguardmix"],
        help="Dataset to download.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Pipe-separated category filter (BeaverTails only). Category keys may contain "
        'commas, e.g. "drug_abuse,weapons,banned_substance|violence,aiding_and_abetting,incitement".',
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="HF split. Default: 330k_train for beavertails, train for wildguardmix.",
    )
    parser.add_argument("--n-samples", type=int, default=None, help="Max rows to keep.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/datasets/",
        help="Output directory (default: data/datasets/).",
    )
    args = parser.parse_args()

    categories = [c.strip() for c in args.categories.split("|")] if args.categories else None

    if args.dataset == "beavertails":
        split = args.split or "330k_train"
        df = load_beavertails(categories=categories, split=split, n_samples=args.n_samples)
    else:
        split = args.split or "train"
        df = load_wildguardmix(split=split, n_samples=args.n_samples)

    os.makedirs(args.output_dir, exist_ok=True)
    filename = _build_output_filename(args.dataset, categories)
    output_path = os.path.join(args.output_dir, filename)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")
