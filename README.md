# regional_PCABO

This repository contains PCA-assisted Bayesian optimization experiments on BBOB benchmark functions.

The main active entrypoints are:
- `PCA_BO.py`: original PCA-BO workflow.
- `PCA_BO_ivan.py`: Ivan's variant with configurable kernels, `torch.pca_lowrank`, and CMA-ES acquisition optimization.
- `compare_bo.py`: IOH-based comparison between PCA-BO and an official-style BoTorch baseline.
- `compare_bo_with_gaussian.py`: same comparison, but with orthogonal Gaussian noise added to the PCA-BO candidate before evaluation.

## Tested Environment

The comparison scripts were tested in this environment:
- Python `3.11.8`
- `numpy==1.26.4`
- `torch==2.1.0`
- `botorch==0.16.1`
- `gpytorch==1.15.1`
- `scipy==1.11.4`
- `matplotlib==3.8.0`
- `ioh==0.3.18`
- `iohinspector==0.0.6`
- `coco-experiment==2.8.2`
- `cma==4.4.4`

The repository targets Python `3.11.8`. Using older Python versions such as `3.9` may fail due to missing or incompatible package versions.

## Installation

Create a Python `3.11.8` environment and install the pinned requirements:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If you use conda instead of `venv`, make sure the active interpreter is Python `3.11.8`.

Check the environment:

```bash
python -V
python -c "import numpy, torch, botorch, gpytorch, scipy, matplotlib, ioh; print('ok')"
```

## Main Comparison Scripts

### `compare_bo.py`

This script compares:
- `pca_bo`: PCA-BO with:
  - weighted PCA computed via `torch.pca_lowrank`
  - PCA-side explicit `Matern(nu=2.5)` ARD kernel with bounded lengthscales
- `botorch_baseline`: official-style BoTorch baseline using:
  - `SingleTaskGP`
  - `LogExpectedImprovement`
  - `optimize_acqf`

The baseline is intentionally kept close to the standard BoTorch closed-loop pattern.

### `compare_bo_with_gaussian.py`

This script is the same as `compare_bo.py`, except that the PCA-BO branch adds Gaussian noise orthogonal to the current PCA manifold before evaluation:

- optimize on the current PCA manifold
- map the candidate back to the original space
- add orthogonal Gaussian perturbation
- evaluate the perturbed point in the original space

The orthogonal Gaussian schedule is fixed in code:
- `noise_sigma0_scale = 0.05`
- `noise_decay_alpha = 1.0`

## Command-Line Arguments

Both comparison scripts accept the same arguments:
- `--base-seed`
- `--function-ids`
- `--dim`
- `--n0`
- `--budget`
- `--num-runs`
- `--grid-size`
- `--run-root`

Defaults:
- `base_seed = 12`
- `function_ids = [2]`
- `dim = 2`
- `n0 = 8`
- `budget = 50 * dim`
- `num_runs = 30`
- `grid_size = 120`

## Running

Run the standard comparison:

```bash
python compare_bo.py --function-ids 2 --dim 2 --n0 8 --num-runs 30 --budget 100
```

Run the Gaussian-noise comparison:

```bash
python compare_bo_with_gaussian.py --function-ids 2 --dim 2 --n0 8 --num-runs 30 --budget 100
```

Run both sequentially:

```bash
python compare_bo.py --function-ids 2 --dim 2 --n0 8 --num-runs 30 --budget 100 && python compare_bo_with_gaussian.py --function-ids 2 --dim 2 --n0 8 --num-runs 30 --budget 100
```

Run a multi-function batch:

```bash
python compare_bo.py --function-ids 2 8 9 12 22 --dim 2 --n0 8 --num-runs 30 --budget 100
```

```bash
python compare_bo_with_gaussian.py --function-ids 2 8 9 12 22 --dim 2 --n0 8 --num-runs 30 --budget 100
```

## Output Format

The comparison scripts write separate per-method IOH-style `.dat` files.

Example layout:

```text
comparison_runs/<timestamp>_<suffix>/f2/
  pca_bo/
    seed_12/IOHprofiler_f2_DIM2.dat
  botorch_baseline/
    seed_12/IOHprofiler_f2_DIM2.dat
  convergence_overlay.png
  target_precision_overlay.png
```

For the Gaussian script, the PCA method folder is:
- `pca_bo_gaussian`

Each `.dat` file has exactly these columns:

```text
evaluations raw_y current_y raw_y_best current_y_best
```

Notes:
- `seed_<n>` folders correspond to independent runs.
- the first `n0` rows are the shared initial design
- the remaining rows are BO-selected evaluations

## Plots

The comparison scripts generate:
- `convergence_overlay.png`: mean best-so-far objective value across runs
- `target_precision_overlay.png`: mean best-so-far minus `f*` across runs

The scripts also generate per-iteration visualizations for the first run only, but only when `dim == 2`.

2D-only plots:
- PCA-BO iteration plots
- PCA weighted-point plots
- baseline iteration plots

Higher-dimensional runs still work, but these 2D visualizations are skipped.

## Legacy Scripts

### `PCA_BO.py`

Original PCA-BO implementation using:
- weighted PCA
- default `SingleTaskGP`
- `differential_evolution` to optimize the penalized acquisition function

Run it with:

```bash
python PCA_BO.py
```

### `PCA_BO_ivan.py`

Ivan's variant includes:
- explicit kernel configuration
- `torch.pca_lowrank`
- CMA-ES for acquisition optimization

Run it with:

```bash
python PCA_BO_ivan.py
```

## Notes

- `compare_bo.py` and `compare_bo_with_gaussian.py` use IOH as the benchmark and logging layer.
- `PCA_BO.py` and `PCA_BO_ivan.py` are older standalone experiment scripts and still depend on COCO/BBOB via `cocoex`.
- There is currently no single combined evaluation CSV; the comparison scripts write per-method `.dat` files and final overlay plots directly.
