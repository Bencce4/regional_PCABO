# regional_PCABO

This repository contains PCA-assisted Bayesian optimization experiments on BBOB benchmark functions.

## Main scripts

- `compare_bo_dynamic_orthogonal_turbo_v5.py`: local dynamic orthogonal PCA-BO.
- `compare_pcabo_dynamic_orthogonal_clean.py`: dynamic orthogonal PCA-BO without the local trust-region extension.
- `compare_pcabo_pei_de_matched.py`: matched PCA-BO and dynamic orthogonal PCA-BO comparisons.
- `PCA_BO.py`: shared PCA-BO utilities and the original standalone workflow.

## Installation

The project targets Python `3.11.8`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Running

Example:

```bash
python compare_bo_dynamic_orthogonal_turbo_v5.py \
  --function-ids 2 8 22 \
  --dim 10 \
  --num-runs 30 \
  --budget 300
```

Use `python <script>.py --help` to see the available options for each script.

## Outputs

Experiment outputs are written under `comparison_runs/`. New generated runs, figures, logs, and temporary files should remain local rather than being added to Git.
