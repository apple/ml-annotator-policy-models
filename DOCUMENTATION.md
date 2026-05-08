# Documentation

## Installation

Requires Python 3.10+. We use [uv](https://docs.astral.sh/uv/) as the package manager.

```bash
# Install dependencies and the package
uv sync
```

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

| Variable | Used by |
|----------|---------|
| `OPENAI_API_KEY` | `generate_features.py`, `label_dices_features.py` |
| `GOOGLE_API_KEY` | `Embedder("gemini", ...)` |

## Project Structure

```
safe_dictionary_learning/       # Core library
├── data/
│   └── download.py              # HuggingFace dataset download utility
├── decision_functions/
│   ├── nnlr.py                 # Non-Negative Logistic Regression (PyTorch)
│   ├── dnf.py                  # DNF classifier (aix360 BooleanRuleCG)
│   └── utils.py                # Parsing and rule comparison utilities
└── features/
    ├── embeddings.py            # Embedder: text → feature scores
    ├── generate_features.py     # LLM-based feature generation
    └── label_dices_features.py  # DICES dataset feature labeling

experiments/
├── eval_decision_functions.py   # Train and evaluate NNLR/DNF per annotator
├── eval_biased_decision_functions.py  # Biased annotator experiments
├── counterfactuals/             # Counterfactual generation and analysis
├── human_annotators/            # DICES/PRISM demographic experiments
├── compute_disagreement.py      # Annotator agreement matrices
└── diff_models.py               # Compare models against an oracle
```

## Data Setup

All experiment scripts expect a `--data-root` directory (default: `data/`) with four subdirectories:

```
data/
├── datasets/        # Annotation CSVs and model mappings
├── features/        # Feature lists and embeddings
├── models/          # Saved decision function weights
└── results/         # Experiment outputs
```

### Required datasets

**LLM annotator experiments** use safety-relevant text datasets. The repo supports [BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) and [WildGuard](https://huggingface.co/datasets/allenai/wildguardmix) out of the box. Use the built-in download utility to fetch them:

```bash
# BeaverTails (filtered to specific categories + safe examples)
uv run python -m safe_dictionary_learning.data.download \
    --dataset beavertails \
    --categories "drug_abuse,weapons,banned_substance" \
    --output-dir data/datasets/

# WildGuardMix
uv run python -m safe_dictionary_learning.data.download \
    --dataset wildguardmix \
    --output-dir data/datasets/
```

The utility downloads from HuggingFace, converts to the expected format (`text` + `ground_truth` columns), and saves to `data/datasets/`. Use `--n-samples` to limit the number of rows and `--split` to select a different split.

**Human annotator experiments** use the [DICES](https://github.com/google-research-datasets/dices-dataset) dataset. Download `diverse_safety_adversarial_dialog_350.csv` and/or `diverse_safety_adversarial_dialog_990.csv` and place them in the data directory referenced by `--dataset`.

### Annotation format

Experiments expect two files per dataset:

1. **Annotation CSV** — columns: `text`, `ground_truth`, `annotator_0_label`, `annotator_1_label`, ...
2. **Models JSON** — maps annotator IDs to model names:
   ```json
   {"0": "gpt-4o", "1": "gpt-4o-mini", "2": "qwen-2-7b-instruct"}
   ```

These are produced by having multiple LLMs label each text as safe (0) or unsafe (1). This labeling step requires your own LLM service.

## Pipeline

```
HuggingFace dataset
    │
    ├─► download.py ──► datasets/*.csv (text + ground_truth)
    │       │
    │       ├─► generate_features.py ──► features/*.txt
    │
    ├─► [LLM annotation] ──► datasets/*_annotations*.csv + *_models.json
    │       │
    │       ├─► eval_decision_functions.py ──► results/, models/
    │       │       │
    │       │       └─► diff_models.py ──► model diffs
    │       │
    │       ├─► compute_disagreement.py ──► agreement matrices
    │       │
    │       ├─► eval_biased_decision_functions.py ──► biased grader results
    │       │
    │       └─► generate_counterfactuals.py ──► counterfactual CSVs + models/
    │               │
    │               ├─► [LLM counterfactual rewriting + re-labeling]
    │               │       │
    │               │       └─► analyze_counterfactuals.py ──► faithfulness results
    │
    (steps marked with [] require your own LLM service)
```

All experiment modules are run with `uv run python -m experiments.<module_name>`. Each accepts `--help` for full argument details.

## Experiments

### 0. Download dataset

Downloads a HuggingFace dataset and converts it to the CSV format the pipeline expects.

```bash
uv run python -m safe_dictionary_learning.data.download \
    --dataset beavertails \
    --categories "drug_abuse,weapons,banned_substance" \
    --output-dir data/datasets/
```

**Outputs** (in `datasets/`): `beavertails_drug_abuse_weapons_banned_substance.csv` with `text` and `ground_truth` columns.

### 1. Generate features

Prompts an LLM to brainstorm safety-relevant text features, then optionally clusters them.

```bash
uv run python -m safe_dictionary_learning.features.generate_features \
    --dataset beavertails_drug_abuse_weapons_banned_substance
uv run python -m safe_dictionary_learning.features.generate_features \
    --dataset beavertails_drug_abuse_weapons_banned_substance --cluster
```

**Outputs** (in `features/`): `{dataset}_features.txt`, and with `--cluster`: `{dataset}_clustered_features2.txt` and `{dataset}_clustered_features_embeddings.pt`.

### 2. Evaluate decision functions

Trains per-annotator NNLR and DNF models, reports accuracy, AUC, TPR, and FPR.

```bash
uv run python -m experiments.eval_decision_functions \
    --dataset beavertails_drug_abuse_weapons_banned_substance_annotations_full.csv \
    --features beavertails_drug_abuse_weapons_banned_substance_features.txt \
    --save
```

**Outputs**: `results/{dataset}_results.jsonl` and saved models in `models/`.

### 3. Diff models

Compares trained decision functions against an oracle model to show which features differ.

```bash
uv run python -m experiments.diff_models \
    --dataset beavertails_drug_abuse_weapons_banned_substance_annotations_full.csv \
    --oracle gpt-4o --save
```

**Inputs**: `results/{dataset}_results.jsonl` (from step 2).
**Outputs**: `results/{dataset}_{oracle}_diffs.jsonl`.

### 4. Compute disagreement

Generates annotator agreement confusion matrices.

```bash
uv run python -m experiments.compute_disagreement \
    --dataset beavertails_drug_abuse_weapons_banned_substance_annotations_full.csv --save
```

### 5. Biased grader evaluation

Evaluates decision functions on datasets where each annotator uses a different biasing strategy.

```bash
uv run python -m experiments.eval_biased_decision_functions \
    --dataset biased_beavertails_drug_abuse_weapons_banned_substance_annotations_full.csv --save
```

### 6. Counterfactual analysis

A three-step process:

**Step 1** — Identify triggering features for unsafe-predicted texts:
```bash
uv run python -m experiments.counterfactuals.generate_counterfactuals \
    --dataset beavertails_drug_abuse_weapons_banned_substance_annotations_full.csv \
    --decision-function NNLR
```

**Step 2** — Using your own LLM service, rewrite each unsafe-predicted text by removing the triggering features, then re-label. Add two columns to each output CSV: `counterfactual` (rewritten text) and `counterfactual_label` (new label).

**Step 3** — Evaluate counterfactual faithfulness:
```bash
uv run python -m experiments.counterfactuals.analyze_counterfactuals \
    --dataset beavertails_drug_abuse_weapons_banned_substance_annotations_full.csv \
    --decision-function NNLR --save
```

### 7. Human annotator experiments (DICES/PRISM)

These experiments analyze demographic differences in safety labeling using human annotator data.

```bash
# Per-rater DNF rules
uv run python -m experiments.human_annotators.individual_apm \
    --dataset 350 --category rater_gender

# Demographic group vs. majority comparison
uv run python -m experiments.human_annotators.group_vs_majority_apm \
    --dataset 350 --category rater_race

# Pairwise demographic group comparison
uv run python -m experiments.human_annotators.group_vs_group_apm \
    --dataset 350 --category rater_education

# Bootstrap uncertainty (NNLR)
uv run python -m experiments.human_annotators.nnlr_uncertainty \
    --bootstrap-rounds 1000

# Bootstrap uncertainty (DNF)
uv run python -m experiments.human_annotators.dnf_uncertainty \
    --bootstrap-rounds 100
```

The `--category` argument selects the demographic attribute: `rater_id`, `rater_age`, `rater_gender`, `rater_race`, or `rater_education`.

## Core Library

### Decision Functions

**`NonNegativeLogisticRegression`** (`nnlr.py`) — A PyTorch `nn.Module` with non-negative weight constraints for interpretability. Supports binary and multi-class classification with optional L1 regularization. Weights are clamped to be non-negative after each gradient step, ensuring features can only contribute positively to the unsafe prediction.

**`DNF`** (`dnf.py`) — A disjunctive normal form classifier wrapping aix360's `BooleanRuleCG`. Produces human-readable AND/OR rules from binary features.

Both models implement a consistent interface: `fit()`, `predict()`, `score()`, `get_string_representation()`, and `get_features_used()`.

### Embedder

`Embedder` (`embeddings.py`) converts text and feature descriptions into embeddings, then computes similarity scores. The default embedding model is Google Gemini (requires `GOOGLE_API_KEY`). Qwen is also supported as an alternative local model.

Embeddings are cached in `~/.cache/annotator-policy-models/embeddings/`. Similarity scores can be binarized via `get_binary_scores()` using `"percentile"`, `"midpoint"`, or `"sparsemax"` methods.

## Linting

```bash
ruff check .          # lint
ruff check --fix .    # lint with auto-fix
ruff format .         # format
```

Ruff is configured in `pyproject.toml`: line length 100, target Python 3.12, double quotes.
