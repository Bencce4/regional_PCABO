import argparse
import csv
import json
import math
import multiprocessing
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import ioh
import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.optim import optimize_acqf
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.optimize import differential_evolution
from scipy.stats import qmc
from torch import pca_lowrank

from PCA_BO import PEI, plot_pcabo_iteration, plot_weighted_points_iteration


METHOD_PCA_VANILLA = "pca_bo"
METHOD_PCA = "pca_bo_dynamic_orthogonal_v2"
METHOD_BASELINE = "botorch_baseline"
RUN_SUFFIX = "pca_kernel_pcalowrank_vanilla_dynamic_orthogonal_v2"
KERNEL_NU = 2.5
KERNEL_LENGTHSCALE_BOUNDS = (0.005, 4.0)
METHOD_LABELS = {
    METHOD_PCA_VANILLA: "PCA-BO",
    METHOD_PCA: "PCA-BO + Dynamic Orthogonal Exploration V2",
    METHOD_BASELINE: "BoTorch Baseline",
}
METHOD_COLORS = {
    METHOD_PCA_VANILLA: "#2ca02c",
    METHOD_PCA: "#1f77b4",
    METHOD_BASELINE: "#d62728",
}

ORTH_SIGMA_SCALE_INIT = 0.02
ORTH_SIGMA_SCALE_MIN = 0.0
ORTH_SIGMA_SCALE_MAX = 0.10
ORTH_SIGMA_SCALE_STEP = 0.01
ORTH_K_INIT = 1
ORTH_K_MIN = 1
ORTH_K_MAX = 4
UBR_IQM_WINDOW = 7
UBR_EPSILON = 0.1
STAGNATION_PATIENCE = 10
STAGNATION_MIN_IMPROVE = 1e-8
TRIGGER_MODE = "and"
TRIGGER_CONSECUTIVE_REQUIRED = 2
ORTHOGONAL_COOLDOWN = 4
MIXED_R_TAKE = 2
MIXED_M_TAKE = 2
LOCAL_SUBSET_SIZE = 40
KERNEL_TYPE = "matern"
PHASE_A_NUM_RUNS = 8
PHASE_B_NUM_RUNS = 30
PHASE_B_TOP_K = 4


@dataclass
class RunContext:
    method: str
    function_id: int
    seed: int
    run_id: str
    dim: int


@dataclass
class DynamicOrthogonalState:
    sigma_scale: float = ORTH_SIGMA_SCALE_INIT
    orth_k: int = ORTH_K_INIT
    ubr_history: List[float] = field(default_factory=list)
    best_history: List[float] = field(default_factory=list)
    last_selected_mode: str = "manifold"
    last_eval_improved: bool = False
    consecutive_trigger_count: int = 0
    cooldown_remaining: int = 0
    last_adjust_iter: int = -1


@dataclass
class MethodConfig:
    kernel_type: str = KERNEL_TYPE
    kernel_nu: float = KERNEL_NU
    lengthscale_bounds: Tuple[float, float] = KERNEL_LENGTHSCALE_BOUNDS
    orth_sigma_init: float = ORTH_SIGMA_SCALE_INIT
    orth_sigma_min: float = ORTH_SIGMA_SCALE_MIN
    orth_sigma_max: float = ORTH_SIGMA_SCALE_MAX
    orth_sigma_step: float = ORTH_SIGMA_SCALE_STEP
    orth_k_init: int = ORTH_K_INIT
    orth_k_min: int = ORTH_K_MIN
    orth_k_max: int = ORTH_K_MAX
    ubr_iqm_window: int = UBR_IQM_WINDOW
    ubr_epsilon: float = UBR_EPSILON
    stagnation_patience: int = STAGNATION_PATIENCE
    stagnation_min_improve: float = STAGNATION_MIN_IMPROVE
    trigger_mode: str = TRIGGER_MODE
    trigger_consecutive_required: int = TRIGGER_CONSECUTIVE_REQUIRED
    orthogonal_cooldown: int = ORTHOGONAL_COOLDOWN
    mixed_r_take: int = MIXED_R_TAKE
    mixed_m_take: int = MIXED_M_TAKE
    local_subset_size: int = LOCAL_SUBSET_SIZE


@dataclass
class NormalizationStats:
    lower: torch.Tensor
    upper: torch.Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_problem(fid: int, dim: int):
    return ioh.get_problem(fid, 1, dim, ioh.ProblemClass.BBOB)


def generate_covar_module(
    active_dim: int,
    kernel_type: str,
    kernel_nu: float,
    lengthscale_bounds: Tuple[float, float],
):
    if kernel_type == "rbf":
        internal_kernel = RBFKernel(
            ard_num_dims=active_dim,
            lengthscale_constraint=Interval(*lengthscale_bounds),
        )
    else:
        internal_kernel = MaternKernel(
            nu=kernel_nu,
            ard_num_dims=active_dim,
            lengthscale_constraint=Interval(*lengthscale_bounds),
        )
    return ScaleKernel(internal_kernel)


def get_bounds(problem) -> Tuple[np.ndarray, np.ndarray]:
    lb = np.asarray(problem.bounds.lb, dtype=float)
    ub = np.asarray(problem.bounds.ub, dtype=float)
    return lb, ub


def make_normalization_stats(values: torch.Tensor) -> NormalizationStats:
    lower = values.min(dim=0).values
    upper = values.max(dim=0).values
    needs_pad = (upper - lower) < 1e-12
    upper = torch.where(needs_pad, lower + 1.0, upper)
    return NormalizationStats(lower=lower, upper=upper)


def normalize_values(values: torch.Tensor, stats: NormalizationStats) -> torch.Tensor:
    scale = torch.clamp(stats.upper - stats.lower, min=1e-12)
    return (values - stats.lower) / scale


def denormalize_values(values: torch.Tensor, stats: NormalizationStats) -> torch.Tensor:
    scale = torch.clamp(stats.upper - stats.lower, min=1e-12)
    return stats.lower + values * scale


def compute_pca_lowrank(init_x: torch.Tensor, init_y: torch.Tensor, alpha: float = 0.95):
    x_mean = init_x.mean(dim=0)
    x_bar = init_x - x_mean

    y = init_y.squeeze(-1)
    _, sort_idx = torch.sort(y, dim=0, descending=False)
    n = y.shape[0]
    ranks = torch.empty(n, dtype=init_x.dtype, device=init_x.device)
    ranks[sort_idx] = torch.arange(1, n + 1, dtype=init_x.dtype, device=init_x.device)
    w_tilde = torch.log(torch.tensor(float(n), dtype=init_x.dtype, device=init_x.device)) - torch.log(ranks)
    w = w_tilde / w_tilde.sum()

    x_weighted = torch.diag(w) @ x_bar
    x_weighted_mean = x_weighted.mean(dim=0)
    x_weighted_bar = x_weighted - x_weighted_mean

    q = min(x_weighted_bar.shape)
    _, singular_vals, basis = pca_lowrank(x_weighted_bar, q=q, center=True)
    singular_vals_sq = singular_vals.square()
    total_var = singular_vals_sq.sum()
    if float(total_var.detach().cpu().item()) <= 0.0:
        p_r = torch.eye(init_x.shape[1], dtype=init_x.dtype, device=init_x.device)[:1]
        eigvals = torch.ones(1, dtype=init_x.dtype, device=init_x.device)
        return x_mean, x_weighted_mean, p_r, eigvals, w

    cum_ratio = torch.cumsum(singular_vals_sq, dim=0) / total_var
    cutoff = torch.tensor(alpha, dtype=cum_ratio.dtype, device=cum_ratio.device)
    rank = int(torch.searchsorted(cum_ratio, cutoff).item()) + 1
    p_r = basis[:, :rank].T.contiguous()
    eigvals = singular_vals_sq[:rank]
    return x_mean, x_weighted_mean, p_r, eigvals, w


def build_contour_cache(fid: int, dim: int, grid_size: int) -> Dict[str, np.ndarray]:
    plot_problem = create_problem(fid=fid, dim=dim)
    lb, ub = get_bounds(plot_problem)

    x1 = np.linspace(lb[0], ub[0], grid_size)
    x2 = np.linspace(lb[1], ub[1], grid_size)
    X1, X2 = np.meshgrid(x1, x2)
    grid = np.stack([X1.ravel(), X2.ravel()], axis=-1)
    Z = np.array([float(plot_problem(point)) for point in grid], dtype=float).reshape(grid_size, grid_size)
    return {"X1": X1, "X2": X2, "Z": Z, "lb": lb, "ub": ub}


class EvaluationDatLogger:
    fieldnames = ("evaluations", "raw_y", "current_y", "raw_y_best", "current_y_best")

    def __init__(self, dat_path: Path):
        self.dat_path = dat_path
        self.dat_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.dat_path.open("w", encoding="utf-8")
        self._file.write(" ".join(self.fieldnames) + "\n")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def log_evaluation(self, problem) -> None:
        info = problem.log_info
        values = (
            str(int(info.evaluations)),
            f"{float(info.raw_y):.16f}",
            f"{float(info.y):.16f}",
            f"{float(info.raw_y_best):.16f}",
            f"{float(info.y_best):.16f}",
        )
        self._file.write(" ".join(values) + "\n")

    def close(self) -> None:
        self._file.close()


def save_run_config(out_path: Path, config: Dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_runtime_record(csv_path: Path, record: Dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ("function_id", "method", "seed", "run_idx", "runtime_seconds")
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def summarize_runtime_records(csv_path: Path) -> Dict[str, Dict[str, Dict[str, float]]]:
    grouped: Dict[str, Dict[str, List[float]]] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            fid = str(row["function_id"])
            method = str(row["method"])
            runtime = float(row["runtime_seconds"])
            grouped.setdefault(fid, {}).setdefault(method, []).append(runtime)

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for fid, methods in grouped.items():
        summary[fid] = {}
        for method, runtimes in methods.items():
            values = np.asarray(runtimes, dtype=float)
            summary[fid][method] = {
                "num_runs": int(values.shape[0]),
                "mean_seconds": float(values.mean()),
                "median_seconds": float(np.median(values)),
                "min_seconds": float(values.min()),
                "max_seconds": float(values.max()),
            }
    return summary


def merge_runtime_records(output_path: Path, input_paths: Iterable[Path]) -> None:
    fieldnames = ("function_id", "method", "seed", "run_idx", "runtime_seconds")
    rows: List[Dict[str, str]] = []
    for csv_path in input_paths:
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            rows.extend(reader)

    rows.sort(key=lambda row: (int(row["function_id"]), row["method"], int(row["run_idx"]), int(row["seed"])))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_mean_ci(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0)
    if values.shape[0] > 1:
        std = values.std(axis=0, ddof=1)
        ci = 1.96 * std / np.sqrt(values.shape[0])
    else:
        ci = np.zeros_like(mean)
    return mean, ci


def load_best_so_far_series(dat_path: Path, field: str) -> np.ndarray:
    with dat_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=" ", skipinitialspace=True)
        series = [float(row[field]) for row in reader]
    return np.asarray(series, dtype=float)


def plot_overlay_with_ci(
    series_by_method: Dict[str, np.ndarray],
    out_path: Path,
    title: str,
    ylabel: str,
    reference_line: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    for method, values in series_by_method.items():
        mean, ci = compute_mean_ci(values)
        evaluations = np.arange(1, mean.shape[0] + 1)
        color = METHOD_COLORS.get(method, None)
        label = METHOD_LABELS.get(method, method)
        ax.plot(evaluations, mean, color=color, linewidth=2.0, label=label)
        ax.fill_between(evaluations, mean - ci, mean + ci, color=color, alpha=0.22)

    flat_values = np.concatenate([arr.reshape(-1) for arr in series_by_method.values()])
    positive_values = flat_values[flat_values > 0]
    if positive_values.size > 0:
        dynamic_range = positive_values.max() / max(positive_values.min(), 1e-12)
        if dynamic_range >= 100:
            ax.set_yscale("log")

    ax.axhline(reference_line, color="black", linestyle="--", linewidth=1.2, label="f*")
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def save_convergence_plots(function_root: Path, fid: int, dim: int, budget: int) -> None:
    methods = (METHOD_PCA_VANILLA, METHOD_PCA, METHOD_BASELINE)
    series_current_best: Dict[str, List[np.ndarray]] = {method: [] for method in methods}
    optimum = float(create_problem(fid=fid, dim=dim).optimum.y)

    for method in methods:
        method_root = function_root / method
        for dat_path in sorted(method_root.glob(f"seed_*/IOHprofiler_f{fid}_DIM{dim}.dat")):
            current_best = load_best_so_far_series(dat_path=dat_path, field="current_y_best")
            if current_best.shape[0] != budget:
                raise ValueError(
                    f"Expected {budget} rows in {dat_path}, found {current_best.shape[0]}."
                )
            series_current_best[method].append(current_best)

    current_by_method = {
        method: np.vstack(series_list)
        for method, series_list in series_current_best.items()
        if series_list
    }
    target_precision_by_method = {
        method: values - optimum for method, values in current_by_method.items()
    }

    plot_overlay_with_ci(
        series_by_method=current_by_method,
        out_path=function_root / "convergence_overlay.png",
        title=f"f{fid} d{dim} | Best-So-Far Loss",
        ylabel="Best-so-far loss",
        reference_line=optimum,
    )
    plot_overlay_with_ci(
        series_by_method=target_precision_by_method,
        out_path=function_root / "target_precision_overlay.png",
        title=f"f{fid} d{dim} | Target Precision",
        ylabel="Best-so-far - f*",
        reference_line=0.0,
    )


def sample_initial_design(lb: torch.Tensor, ub: torch.Tensor, dim: int, n0: int, seed: int) -> torch.Tensor:
    sampler = qmc.LatinHypercube(d=dim, seed=seed)
    unit_samples = torch.tensor(sampler.random(n=n0), dtype=lb.dtype)
    return lb + (ub - lb) * unit_samples


def evaluate_point(problem, x_np: np.ndarray, eval_logger: EvaluationDatLogger) -> float:
    current_y = float(problem(x_np))
    eval_logger.log_evaluation(problem)
    return current_y


def evaluate_initial_design(problem, init_x: torch.Tensor, eval_logger: EvaluationDatLogger) -> torch.Tensor:
    init_y_vals = []
    for idx in range(init_x.shape[0]):
        x_np = init_x[idx].detach().cpu().numpy()
        current_y = evaluate_point(problem=problem, x_np=x_np, eval_logger=eval_logger)
        init_y_vals.append(current_y)
    return torch.tensor(init_y_vals, dtype=init_x.dtype).unsqueeze(-1)


def plot_baseline_iteration(
    contour_cache: Dict[str, np.ndarray],
    X_hist: torch.Tensor,
    iter_idx: int,
    func_id: int,
    dim: int,
    out_dir: Path,
) -> None:
    zeros = torch.zeros(dim, dtype=X_hist.dtype, device=X_hist.device)
    dummy_dirs = torch.zeros((1, dim), dtype=X_hist.dtype, device=X_hist.device)
    dummy_eigvals = torch.zeros((1,), dtype=X_hist.dtype, device=X_hist.device)
    plot_pcabo_iteration(
        contour_cache=contour_cache,
        X_hist=X_hist,
        x_mean=zeros,
        x_weighted_mean=zeros,
        P_r=dummy_dirs,
        eigvals=dummy_eigvals,
        iter_idx=iter_idx,
        func_id=func_id,
        dim=dim,
        out_dir=str(out_dir),
    )


def posterior_mean_std(gp: SingleTaskGP, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        posterior = gp.posterior(z)
    mean = posterior.mean.squeeze(-1)
    std = torch.sqrt(torch.clamp(posterior.variance.squeeze(-1), min=1e-18))
    return mean, std


def estimate_latent_ubr(
    gp: SingleTaskGP,
    z_train: torch.Tensor,
    bounds_z: torch.Tensor,
    beta_t: float,
    seed: int,
    config: MethodConfig,
) -> float:
    beta = torch.tensor(beta_t, dtype=z_train.dtype, device=z_train.device)
    train_mean, train_std = posterior_mean_std(gp, z_train)
    min_observed_ucb = torch.min(train_mean + beta * train_std)

    lbz = bounds_z[0].detach().cpu().numpy()
    ubz = bounds_z[1].detach().cpu().numpy()
    de_bounds = [(float(lower), float(upper)) for lower, upper in zip(lbz, ubz)]

    def lcb_objective(z_np: Iterable[float]) -> float:
        z_t = torch.tensor(z_np, dtype=z_train.dtype, device=z_train.device).view(1, -1)
        mean, std = posterior_mean_std(gp, z_t)
        lcb = mean - beta * std
        return float(lcb.squeeze().detach().cpu().item())

    result = differential_evolution(
        lcb_objective,
        bounds=de_bounds,
        strategy="best1bin",
        maxiter=40,
        popsize=10,
        polish=False,
        seed=seed,
    )
    min_search_lcb = float(result.fun)
    return float(min_observed_ucb.detach().cpu().item()) - min_search_lcb


def moving_iqm(values: List[float], window: int) -> List[float]:
    smoothed = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        current = np.asarray(values[start : idx + 1], dtype=float)
        if current.size >= 4:
            current = np.sort(current)
            lower = int(np.floor(0.25 * current.size))
            upper = int(np.ceil(0.75 * current.size))
            middle = current[lower:upper]
            smoothed.append(float(np.mean(middle if middle.size else current)))
        else:
            smoothed.append(float(np.mean(current)))
    return smoothed


def ubr_has_plateaued(ubr_history: List[float], config: MethodConfig) -> bool:
    if len(ubr_history) < 4:
        return False

    smoothed = moving_iqm(ubr_history, window=config.ubr_iqm_window)
    gradients = np.diff(np.asarray(smoothed, dtype=float))
    if gradients.size < 2:
        return False

    abs_gradients = np.abs(gradients)
    max_abs_gradient = float(np.max(abs_gradients[:-1]))
    if max_abs_gradient <= 0.0:
        return False
    return float(abs_gradients[-1]) <= config.ubr_epsilon * max_abs_gradient


def best_has_stagnated(best_history: List[float], config: MethodConfig) -> bool:
    if len(best_history) < config.stagnation_patience + 1:
        return False
    window = best_history[-(config.stagnation_patience + 1) :]
    improvement = float(window[0] - window[-1])
    return improvement < config.stagnation_min_improve


def decrement_cooldown(state: DynamicOrthogonalState) -> None:
    if state.cooldown_remaining > 0:
        state.cooldown_remaining -= 1


def update_trigger_counter(
    state: DynamicOrthogonalState,
    ubr_trigger: bool,
    stagnation_trigger: bool,
    config: MethodConfig,
) -> bool:
    if config.trigger_mode == "or":
        trigger = ubr_trigger or stagnation_trigger
    else:
        trigger = ubr_trigger and stagnation_trigger

    if trigger:
        state.consecutive_trigger_count += 1
    else:
        state.consecutive_trigger_count = 0

    return trigger


def orthogonal_basis_from_projection(P_r: torch.Tensor) -> torch.Tensor:
    dim = P_r.shape[1]
    rank = P_r.shape[0]
    if rank >= dim:
        return torch.zeros(dim, 0, dtype=P_r.dtype, device=P_r.device)
    V_r = P_r.T
    identity = torch.eye(dim, dtype=P_r.dtype, device=P_r.device)
    projector_perp = identity - V_r @ V_r.T
    eigvals, eigvecs = torch.linalg.eigh(projector_perp)
    keep = eigvals > 0.5
    return eigvecs[:, keep]


def should_activate_orthogonal(
    state: DynamicOrthogonalState,
    rank: int,
    dim: int,
    config: MethodConfig,
) -> bool:
    if rank >= dim:
        return False
    if state.cooldown_remaining > 0:
        return False
    return state.consecutive_trigger_count >= config.trigger_consecutive_required


def adjust_dynamic_orthogonal_state(
    state: DynamicOrthogonalState,
    activate_orthogonal: bool,
    iter_idx: int,
    config: MethodConfig,
) -> None:
    if not activate_orthogonal:
        return

    if state.last_selected_mode == "orthogonal":
        state.sigma_scale = max(config.orth_sigma_min, state.sigma_scale - config.orth_sigma_step)
        if not state.last_eval_improved:
            state.orth_k = max(config.orth_k_min, state.orth_k - 1)
    else:
        state.sigma_scale = min(config.orth_sigma_max, state.sigma_scale + config.orth_sigma_step)
        state.orth_k = min(config.orth_k_max, state.orth_k + 1)

    state.consecutive_trigger_count = 0
    state.last_adjust_iter = iter_idx


def select_mixed_basis(P_r: torch.Tensor, U_perp: torch.Tensor, config: MethodConfig) -> Optional[torch.Tensor]:
    parts = []
    r_take = min(config.mixed_r_take, P_r.shape[0])
    m_take = min(config.mixed_m_take, U_perp.shape[1])
    if r_take > 0:
        parts.append(P_r[:r_take].T)
    if m_take > 0:
        parts.append(U_perp[:, :m_take])
    if not parts:
        return None
    return torch.cat(parts, dim=1)


def fit_gp_with_config(train_x: torch.Tensor, train_y: torch.Tensor, config: MethodConfig) -> SingleTaskGP:
    covar_module = generate_covar_module(
        active_dim=train_x.shape[1],
        kernel_type=config.kernel_type,
        kernel_nu=config.kernel_nu,
        lengthscale_bounds=config.lengthscale_bounds,
    )
    gp = SingleTaskGP(
        train_X=train_x,
        train_Y=train_y,
        covar_module=covar_module,
        outcome_transform=Standardize(m=1),
    )
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    return gp


def score_mixed_candidates(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    cand_x: torch.Tensor,
    config: MethodConfig,
) -> torch.Tensor:
    gp = fit_gp_with_config(train_x, train_y, config)
    try:
        best_f = train_y.min()
        acq = LogExpectedImprovement(model=gp, best_f=best_f, maximize=False)
        with torch.no_grad():
            scores = acq(cand_x.unsqueeze(-2)).reshape(-1)
    except Exception:
        with torch.no_grad():
            scores = -gp.posterior(cand_x).mean.squeeze(-1)
    return scores


def select_dynamic_candidate_v2(
    x_manifold: torch.Tensor,
    X_hist: torch.Tensor,
    init_y: torch.Tensor,
    P_r: torch.Tensor,
    x_mean: torch.Tensor,
    x_weighted_mean: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
    activate_orthogonal: bool,
    state: DynamicOrthogonalState,
    seed: int,
    config: MethodConfig,
) -> Tuple[torch.Tensor, str]:
    x_manifold = torch.clamp(x_manifold, min=lb, max=ub)
    if not activate_orthogonal:
        return x_manifold, "manifold"

    dim = x_manifold.shape[-1]
    rank = P_r.shape[0]
    if rank >= dim:
        return x_manifold, "manifold"

    U_perp = orthogonal_basis_from_projection(P_r)
    if U_perp.shape[1] == 0:
        return x_manifold, "manifold"

    generator = torch.Generator(device=x_manifold.device)
    generator.manual_seed(seed)
    sigma_t = max(config.orth_sigma_min, min(config.orth_sigma_max, state.sigma_scale))
    sigma_t = sigma_t * float(torch.mean(ub - lb).detach().cpu().item())

    gaussian = torch.randn(
        max(1, state.orth_k),
        dim,
        dtype=x_manifold.dtype,
        device=x_manifold.device,
        generator=generator,
    ) * sigma_t
    projector_perp = U_perp @ U_perp.T
    noise_perp = gaussian @ projector_perp.T
    nonzero_mask = torch.linalg.norm(noise_perp, dim=-1) > 1e-12
    if not bool(nonzero_mask.any().detach().cpu().item()):
        return x_manifold, "manifold"

    x_orth = torch.clamp(x_manifold + noise_perp[nonzero_mask], min=lb, max=ub)
    candidate_x = torch.cat([x_manifold, x_orth], dim=0)

    B = select_mixed_basis(P_r, U_perp, config)
    if B is None:
        return x_manifold, "manifold"

    subset_size = min(config.local_subset_size, X_hist.shape[0])
    dist2 = torch.sum((X_hist - x_manifold) * (X_hist - x_manifold), dim=1)
    nn_idx = torch.topk(dist2, k=subset_size, largest=False).indices
    train_x_orig = X_hist[nn_idx] @ B
    cand_x_orig = candidate_x @ B
    norm_stats = make_normalization_stats(train_x_orig)
    train_x = normalize_values(train_x_orig, norm_stats)
    cand_x = normalize_values(cand_x_orig, norm_stats)
    train_y = init_y[nn_idx]

    scores = score_mixed_candidates(train_x=train_x, train_y=train_y, cand_x=cand_x, config=config)
    best_idx = int(torch.argmax(scores).detach().cpu().item())
    selected_mode = "manifold" if best_idx == 0 else "orthogonal"
    return candidate_x[best_idx : best_idx + 1], selected_mode


def update_dynamic_orthogonal_state(
    state: DynamicOrthogonalState,
    should_adjust: bool,
) -> None:
    if not should_adjust:
        return

    if state.last_selected_mode == "orthogonal":
        state.sigma_scale = max(ORTH_SIGMA_SCALE_MIN, state.sigma_scale - ORTH_SIGMA_SCALE_STEP)
        state.prefer_orthogonal = False
        if not state.last_eval_improved:
            state.orth_k = max(ORTH_K_MIN, state.orth_k - 1)
    else:
        state.sigma_scale = min(ORTH_SIGMA_SCALE_MAX, state.sigma_scale + ORTH_SIGMA_SCALE_STEP)
        state.orth_k = min(ORTH_K_MAX, state.orth_k + 1)
        state.prefer_orthogonal = True


def select_dynamic_orthogonal_candidate(
    x_manifold: torch.Tensor,
    P_r: torch.Tensor,
    x_mean: torch.Tensor,
    x_weighted_mean: torch.Tensor,
    acquisition: PEI,
    state: DynamicOrthogonalState,
    lb: torch.Tensor,
    ub: torch.Tensor,
    seed: int,
) -> Tuple[torch.Tensor, str]:
    dim = x_manifold.shape[-1]
    rank = P_r.shape[0]
    x_manifold = torch.clamp(x_manifold, min=lb, max=ub)
    candidates = [x_manifold]

    if rank < dim and state.sigma_scale > 0.0:
        generator = torch.Generator(device=x_manifold.device)
        generator.manual_seed(seed)

        V_r = P_r.T
        identity = torch.eye(dim, dtype=x_manifold.dtype, device=x_manifold.device)
        projector_perp = identity - V_r @ V_r.T
        sigma_t = state.sigma_scale * torch.mean(ub - lb)

        gaussian = torch.randn(
            state.orth_k,
            dim,
            dtype=x_manifold.dtype,
            device=x_manifold.device,
            generator=generator,
        ) * sigma_t
        noise_perp = gaussian @ projector_perp.T
        nonzero_mask = torch.linalg.norm(noise_perp, dim=-1) > 1e-12
        if bool(nonzero_mask.any().detach().cpu().item()):
            x_orth = torch.clamp(x_manifold + noise_perp[nonzero_mask], min=lb, max=ub)
            candidates.append(x_orth)

    candidate_x = torch.cat(candidates, dim=0)
    candidate_z = ((candidate_x - x_mean) - x_weighted_mean) @ P_r.T
    with torch.no_grad():
        scores = acquisition(candidate_z.unsqueeze(-2)).reshape(-1)
    if state.prefer_orthogonal and candidate_x.shape[0] > 1:
        step_norms = torch.linalg.norm(candidate_x - x_manifold, dim=-1)
        step_scale = torch.clamp(torch.mean(ub - lb), min=1e-12)
        scores = scores + 1e-9 * (step_norms / step_scale)
    best_idx = int(torch.argmax(scores).detach().cpu().item())
    selected_mode = "manifold" if best_idx == 0 else "orthogonal"
    if selected_mode == "orthogonal":
        state.prefer_orthogonal = False
    return candidate_x[best_idx : best_idx + 1], selected_mode


def run_pca_bo(
    fid: int,
    dim: int,
    seed: int,
    run_idx: int,
    n0: int,
    budget: int,
    grid_size: int,
    dat_path: Path,
    initial_x: torch.Tensor,
    save_plots: bool,
    plot_root: Path,
    config: MethodConfig,
) -> None:
    set_seed(seed)
    problem = create_problem(fid=fid, dim=dim)
    contour_cache = build_contour_cache(fid=fid, dim=dim, grid_size=grid_size) if save_plots else None

    with EvaluationDatLogger(dat_path=dat_path) as eval_logger:
        lb = torch.tensor(problem.bounds.lb, dtype=torch.double)
        ub = torch.tensor(problem.bounds.ub, dtype=torch.double)

        init_x_local = initial_x.clone().to(dtype=torch.double)
        init_y = evaluate_initial_design(problem=problem, init_x=init_x_local, eval_logger=eval_logger)
        state = DynamicOrthogonalState(
            sigma_scale=config.orth_sigma_init,
            orth_k=config.orth_k_init,
            best_history=[float(init_y.min().detach().cpu().item())],
        )

        while problem.state.evaluations < budget:
            decrement_cooldown(state)
            x_mean, x_weighted_mean, P_r, eigvals, w = compute_pca_lowrank(init_x_local, init_y, alpha=0.95)
            z_r = ((init_x_local - x_mean) - x_weighted_mean) @ P_r.T
            z_norm_stats = make_normalization_stats(z_r)
            z_r_norm = normalize_values(z_r, z_norm_stats)
            mapper = lambda z: z @ P_r + x_mean + x_weighted_mean
            mapper_norm = lambda z_norm: mapper(denormalize_values(z_norm, z_norm_stats))

            x_center = 0.5 * (lb + ub)
            z_center = ((x_center - x_mean) - x_weighted_mean) @ P_r.T
            rho = 0.5 * torch.min(ub - lb)
            bounds_z = torch.stack([z_center - rho, z_center + rho], dim=0)
            bounds_z_norm = normalize_values(bounds_z, z_norm_stats)

            gp = fit_gp_with_config(train_x=z_r_norm, train_y=init_y, config=config)

            best_f = init_y.min()
            acquisition = PEI(gp=gp, best_f=best_f, bounds=(lb, ub), penalty_weight=100.0, mapper=mapper_norm)
            beta_t = 2.0 * math.log(
                float(max(1.0000001, z_r.shape[1] * (problem.state.evaluations + 1) ** 2))
            )
            latent_ubr = estimate_latent_ubr(
                gp=gp,
                z_train=z_r_norm,
                bounds_z=bounds_z_norm,
                beta_t=beta_t,
                seed=seed + problem.state.evaluations,
                config=config,
            )
            state.ubr_history.append(latent_ubr)
            ubr_trigger = ubr_has_plateaued(state.ubr_history, config)
            stagnation_trigger = best_has_stagnated(state.best_history, config)
            update_trigger_counter(
                state=state,
                ubr_trigger=ubr_trigger,
                stagnation_trigger=stagnation_trigger,
                config=config,
            )
            activate_orthogonal = should_activate_orthogonal(state, rank=P_r.shape[0], dim=dim, config=config)
            adjust_dynamic_orthogonal_state(
                state=state,
                activate_orthogonal=activate_orthogonal,
                iter_idx=problem.state.evaluations,
                config=config,
            )

            lbz = bounds_z_norm[0].detach().cpu().numpy()
            ubz = bounds_z_norm[1].detach().cpu().numpy()
            de_bounds = [(float(lower), float(upper)) for lower, upper in zip(lbz, ubz)]

            def objective(z_np: Iterable[float]) -> float:
                z_t = torch.tensor(z_np, dtype=init_x_local.dtype).view(1, 1, -1)
                with torch.no_grad():
                    value = acquisition(z_t).squeeze()
                return -float(value.detach().cpu().item())

            result = differential_evolution(
                objective,
                bounds=de_bounds,
                strategy="best1bin",
                maxiter=80,
                popsize=15,
                polish=True,
                seed=seed,
            )

            new_z_norm = torch.tensor(result.x, dtype=init_x_local.dtype).view(1, -1)
            new_x = mapper_norm(new_z_norm)
            x_eval, selected_mode = select_dynamic_candidate_v2(
                x_manifold=new_x,
                X_hist=init_x_local,
                init_y=init_y,
                P_r=P_r,
                x_mean=x_mean,
                x_weighted_mean=x_weighted_mean,
                lb=lb,
                ub=ub,
                activate_orthogonal=activate_orthogonal,
                state=state,
                seed=seed + problem.state.evaluations,
                config=config,
            )

            if save_plots and dim == 2:
                plot_weighted_points_iteration(
                    contour_cache=contour_cache,
                    X_hist=init_x_local,
                    w=w,
                    x_mean=x_mean,
                    iter_idx=init_x_local.shape[0] - n0,
                    func_id=fid,
                    dim=dim,
                    out_dir=str(plot_root / METHOD_PCA / "weighted"),
                )

            previous_best = state.best_history[-1]
            current_y = evaluate_point(
                problem=problem,
                x_np=x_eval.detach().cpu().numpy().reshape(-1),
                eval_logger=eval_logger,
            )
            new_y = torch.tensor([[current_y]], dtype=init_y.dtype)
            current_best = min(previous_best, current_y)
            state.last_selected_mode = selected_mode
            state.last_eval_improved = current_best < previous_best - config.stagnation_min_improve
            state.best_history.append(current_best)
            if selected_mode == "orthogonal":
                state.cooldown_remaining = config.orthogonal_cooldown

            init_x_local = torch.cat((init_x_local, x_eval), dim=0)
            init_y = torch.cat((init_y, new_y), dim=0)

            if save_plots and dim == 2:
                plot_pcabo_iteration(
                    contour_cache=contour_cache,
                    X_hist=init_x_local,
                    x_mean=x_mean,
                    x_weighted_mean=x_weighted_mean,
                    P_r=P_r,
                    eigvals=eigvals,
                    iter_idx=init_x_local.shape[0] - n0,
                    func_id=fid,
                    dim=dim,
                    out_dir=str(plot_root / METHOD_PCA / "iterations"),
                )


def run_pca_bo_vanilla(
    fid: int,
    dim: int,
    seed: int,
    run_idx: int,
    n0: int,
    budget: int,
    grid_size: int,
    dat_path: Path,
    initial_x: torch.Tensor,
    save_plots: bool,
    plot_root: Path,
) -> None:
    set_seed(seed)
    problem = create_problem(fid=fid, dim=dim)
    contour_cache = build_contour_cache(fid=fid, dim=dim, grid_size=grid_size) if save_plots else None

    with EvaluationDatLogger(dat_path=dat_path) as eval_logger:
        lb = torch.tensor(problem.bounds.lb, dtype=torch.double)
        ub = torch.tensor(problem.bounds.ub, dtype=torch.double)

        init_x_local = initial_x.clone().to(dtype=torch.double)
        init_y = evaluate_initial_design(problem=problem, init_x=init_x_local, eval_logger=eval_logger)

        while problem.state.evaluations < budget:
            x_mean, x_weighted_mean, P_r, eigvals, w = compute_pca_lowrank(init_x_local, init_y, alpha=0.95)
            z_r = ((init_x_local - x_mean) - x_weighted_mean) @ P_r.T
            mapper = lambda z: z @ P_r + x_mean + x_weighted_mean

            x_center = 0.5 * (lb + ub)
            z_center = ((x_center - x_mean) - x_weighted_mean) @ P_r.T
            rho = 0.5 * torch.min(ub - lb)
            bounds_z = torch.stack([z_center - rho, z_center + rho], dim=0)

            covar_module = generate_covar_module(
                active_dim=z_r.shape[1],
                kernel_type=KERNEL_TYPE,
                kernel_nu=KERNEL_NU,
                lengthscale_bounds=KERNEL_LENGTHSCALE_BOUNDS,
            )
            gp = SingleTaskGP(
                train_X=z_r,
                train_Y=init_y,
                covar_module=covar_module,
                outcome_transform=Standardize(m=1),
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

            best_f = init_y.min()
            acquisition = PEI(gp=gp, best_f=best_f, bounds=(lb, ub), penalty_weight=100.0, mapper=mapper)

            lbz = bounds_z[0].detach().cpu().numpy()
            ubz = bounds_z[1].detach().cpu().numpy()
            de_bounds = [(float(lower), float(upper)) for lower, upper in zip(lbz, ubz)]

            def objective(z_np: Iterable[float]) -> float:
                z_t = torch.tensor(z_np, dtype=init_x_local.dtype).view(1, 1, -1)
                with torch.no_grad():
                    value = acquisition(z_t).squeeze()
                return -float(value.detach().cpu().item())

            result = differential_evolution(
                objective,
                bounds=de_bounds,
                strategy="best1bin",
                maxiter=80,
                popsize=15,
                polish=True,
                seed=seed,
            )

            new_z = torch.tensor(result.x, dtype=init_x_local.dtype).view(1, -1)
            new_x = mapper(new_z)

            if save_plots and dim == 2:
                plot_weighted_points_iteration(
                    contour_cache=contour_cache,
                    X_hist=init_x_local,
                    w=w,
                    x_mean=x_mean,
                    iter_idx=init_x_local.shape[0] - n0,
                    func_id=fid,
                    dim=dim,
                    out_dir=str(plot_root / METHOD_PCA_VANILLA / "weighted"),
                )

            current_y = evaluate_point(
                problem=problem,
                x_np=new_x.detach().cpu().numpy().reshape(-1),
                eval_logger=eval_logger,
            )
            new_y = torch.tensor([[current_y]], dtype=init_y.dtype)

            init_x_local = torch.cat((init_x_local, new_x), dim=0)
            init_y = torch.cat((init_y, new_y), dim=0)

            if save_plots and dim == 2:
                plot_pcabo_iteration(
                    contour_cache=contour_cache,
                    X_hist=init_x_local,
                    x_mean=x_mean,
                    x_weighted_mean=x_weighted_mean,
                    P_r=P_r,
                    eigvals=eigvals,
                    iter_idx=init_x_local.shape[0] - n0,
                    func_id=fid,
                    dim=dim,
                    out_dir=str(plot_root / METHOD_PCA_VANILLA / "iterations"),
                )


def run_botorch_baseline(
    fid: int,
    dim: int,
    seed: int,
    run_idx: int,
    n0: int,
    budget: int,
    grid_size: int,
    dat_path: Path,
    initial_x: torch.Tensor,
    save_plots: bool,
    plot_root: Path,
) -> None:
    set_seed(seed)
    problem = create_problem(fid=fid, dim=dim)
    contour_cache = build_contour_cache(fid=fid, dim=dim, grid_size=grid_size) if save_plots else None

    with EvaluationDatLogger(dat_path=dat_path) as eval_logger:
        train_x = initial_x.clone().to(dtype=torch.double)
        train_y = evaluate_initial_design(problem=problem, init_x=train_x, eval_logger=eval_logger)

        lb = torch.tensor(problem.bounds.lb, dtype=torch.double)
        ub = torch.tensor(problem.bounds.ub, dtype=torch.double)
        bounds = torch.stack([lb, ub], dim=0)

        while problem.state.evaluations < budget:
            gp = SingleTaskGP(
                train_X=train_x,
                train_Y=train_y,
                outcome_transform=Standardize(m=1),
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

            acquisition = LogExpectedImprovement(model=gp, best_f=train_y.min(), maximize=False)
            candidate, _ = optimize_acqf(
                acq_function=acquisition,
                bounds=bounds,
                q=1,
                num_restarts=20,
                raw_samples=256,
                options={"batch_limit": 5, "maxiter": 200},
            )

            current_y = evaluate_point(
                problem=problem,
                x_np=candidate.detach().cpu().numpy().reshape(-1),
                eval_logger=eval_logger,
            )
            new_y = torch.tensor([[current_y]], dtype=train_y.dtype)

            train_x = torch.cat((train_x, candidate), dim=0)
            train_y = torch.cat((train_y, new_y), dim=0)

            if save_plots and dim == 2:
                plot_baseline_iteration(
                    contour_cache=contour_cache,
                    X_hist=train_x,
                    iter_idx=train_x.shape[0] - n0,
                    func_id=fid,
                    dim=dim,
                    out_dir=plot_root / METHOD_BASELINE / "iterations",
                )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run vanilla PCA-BO, PCA-BO V2 with mixed-space orthogonal rescoring, "
            "and the official BoTorch baseline with IOH logging."
        ),
    )
    parser.add_argument("--base-seed", type=int, default=12)
    parser.add_argument("--function-ids", nargs="+", type=int, default=[2])
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--n0", type=int, default=8)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--num-runs", type=int, default=30)
    parser.add_argument("--grid-size", type=int, default=120)
    parser.add_argument("--run-root", type=str, default=None)
    parser.add_argument("--kernel-type", choices=["matern", "rbf"], default=KERNEL_TYPE)
    parser.add_argument("--kernel-nu", type=float, choices=[0.5, 1.5, 2.5], default=KERNEL_NU)
    parser.add_argument("--lengthscale-lb", type=float, default=KERNEL_LENGTHSCALE_BOUNDS[0])
    parser.add_argument("--lengthscale-ub", type=float, default=KERNEL_LENGTHSCALE_BOUNDS[1])
    parser.add_argument("--orth-sigma-init", type=float, default=ORTH_SIGMA_SCALE_INIT)
    parser.add_argument("--orth-sigma-max", type=float, default=ORTH_SIGMA_SCALE_MAX)
    parser.add_argument("--orth-sigma-step", type=float, default=ORTH_SIGMA_SCALE_STEP)
    parser.add_argument("--orth-k-max", type=int, default=ORTH_K_MAX)
    parser.add_argument("--ubr-epsilon", type=float, default=UBR_EPSILON)
    parser.add_argument("--stagnation-patience", type=int, default=STAGNATION_PATIENCE)
    parser.add_argument("--trigger-mode", choices=["and", "or"], default=TRIGGER_MODE)
    parser.add_argument("--trigger-consecutive-required", type=int, default=TRIGGER_CONSECUTIVE_REQUIRED)
    parser.add_argument("--orthogonal-cooldown", type=int, default=ORTHOGONAL_COOLDOWN)
    parser.add_argument("--mixed-r-take", type=int, default=MIXED_R_TAKE)
    parser.add_argument("--mixed-m-take", type=int, default=MIXED_M_TAKE)
    parser.add_argument("--local-subset-size", type=int, default=LOCAL_SUBSET_SIZE)
    parser.add_argument("--parallel-functions", type=int, default=1)
    return parser.parse_args()


def build_method_config(args) -> MethodConfig:
    return MethodConfig(
        kernel_type=args.kernel_type,
        kernel_nu=args.kernel_nu,
        lengthscale_bounds=(args.lengthscale_lb, args.lengthscale_ub),
        orth_sigma_init=args.orth_sigma_init,
        orth_sigma_min=ORTH_SIGMA_SCALE_MIN,
        orth_sigma_max=args.orth_sigma_max,
        orth_sigma_step=args.orth_sigma_step,
        orth_k_init=ORTH_K_INIT,
        orth_k_min=ORTH_K_MIN,
        orth_k_max=args.orth_k_max,
        ubr_iqm_window=UBR_IQM_WINDOW,
        ubr_epsilon=args.ubr_epsilon,
        stagnation_patience=args.stagnation_patience,
        stagnation_min_improve=STAGNATION_MIN_IMPROVE,
        trigger_mode=args.trigger_mode,
        trigger_consecutive_required=args.trigger_consecutive_required,
        orthogonal_cooldown=args.orthogonal_cooldown,
        mixed_r_take=args.mixed_r_take,
        mixed_m_take=args.mixed_m_take,
        local_subset_size=args.local_subset_size,
    )


def run_single_function(fid: int, args, budget: int, run_root: Path, method_config: MethodConfig) -> Path:
    function_root = run_root / f"f{fid}"
    runtime_csv_path = function_root / "runtime_log.csv"

    print(f"Running f{fid}: {args.num_runs} runs per method")

    for run_idx in range(args.num_runs):
        seed = args.base_seed + run_idx
        set_seed(seed)

        sampling_problem = create_problem(fid=fid, dim=args.dim)
        lb_np, ub_np = get_bounds(sampling_problem)
        lb = torch.tensor(lb_np, dtype=torch.double)
        ub = torch.tensor(ub_np, dtype=torch.double)
        initial_x = sample_initial_design(lb=lb, ub=ub, dim=args.dim, n0=args.n0, seed=seed)

        save_plots = False
        start = time.perf_counter()
        run_pca_bo_vanilla(
            fid=fid,
            dim=args.dim,
            seed=seed,
            run_idx=run_idx,
            n0=args.n0,
            budget=budget,
            grid_size=args.grid_size,
            dat_path=function_root / METHOD_PCA_VANILLA / f"seed_{seed}" / f"IOHprofiler_f{fid}_DIM{args.dim}.dat",
            initial_x=initial_x,
            save_plots=save_plots,
            plot_root=function_root,
        )
        append_runtime_record(
            runtime_csv_path,
            {
                "function_id": fid,
                "method": METHOD_PCA_VANILLA,
                "seed": seed,
                "run_idx": run_idx,
                "runtime_seconds": f"{time.perf_counter() - start:.6f}",
            },
        )

        start = time.perf_counter()
        run_pca_bo(
            fid=fid,
            dim=args.dim,
            seed=seed,
            run_idx=run_idx,
            n0=args.n0,
            budget=budget,
            grid_size=args.grid_size,
            dat_path=function_root / METHOD_PCA / f"seed_{seed}" / f"IOHprofiler_f{fid}_DIM{args.dim}.dat",
            initial_x=initial_x,
            save_plots=save_plots,
            plot_root=function_root,
            config=method_config,
        )
        append_runtime_record(
            runtime_csv_path,
            {
                "function_id": fid,
                "method": METHOD_PCA,
                "seed": seed,
                "run_idx": run_idx,
                "runtime_seconds": f"{time.perf_counter() - start:.6f}",
            },
        )

        start = time.perf_counter()
        run_botorch_baseline(
            fid=fid,
            dim=args.dim,
            seed=seed,
            run_idx=run_idx,
            n0=args.n0,
            budget=budget,
            grid_size=args.grid_size,
            dat_path=function_root / METHOD_BASELINE / f"seed_{seed}" / f"IOHprofiler_f{fid}_DIM{args.dim}.dat",
            initial_x=initial_x,
            save_plots=save_plots,
            plot_root=function_root,
        )
        append_runtime_record(
            runtime_csv_path,
            {
                "function_id": fid,
                "method": METHOD_BASELINE,
                "seed": seed,
                "run_idx": run_idx,
                "runtime_seconds": f"{time.perf_counter() - start:.6f}",
            },
        )

    save_convergence_plots(
        function_root=function_root,
        fid=fid,
        dim=args.dim,
        budget=budget,
    )
    save_run_config(function_root / "runtime_summary.json", summarize_runtime_records(runtime_csv_path))
    return runtime_csv_path


def run_experiment(args) -> Path:
    budget = args.budget if args.budget is not None else 50 * args.dim
    method_config = build_method_config(args)
    runtime_csv_path: Optional[Path] = None

    if args.run_root:
        run_root = Path(args.run_root)
    else:
        run_root = Path("comparison_runs") / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{RUN_SUFFIX}"
    run_root.mkdir(parents=True, exist_ok=True)
    runtime_csv_path = run_root / "runtime_log.csv"

    config = {
        "base_seed": args.base_seed,
        "function_ids": args.function_ids,
        "dim": args.dim,
        "n0": args.n0,
        "budget": budget,
        "num_runs": args.num_runs,
        "grid_size": args.grid_size,
        "kernel": {
            "type": method_config.kernel_type,
            "nu": method_config.kernel_nu,
            "lengthscale_bounds": method_config.lengthscale_bounds,
        },
        "pca_backend": "torch.pca_lowrank",
        "vanilla_pca_bo": {
            "kernel": {
                "type": "Matern",
                "nu": KERNEL_NU,
                "lengthscale_bounds": KERNEL_LENGTHSCALE_BOUNDS,
            },
            "acquisition": "PEI",
            "notes": "Closest comparison to the original compare_bo.py PCA-BO implementation.",
        },
        "dynamic_orthogonal_exploration_v2": {
            "controller_rule": "mixed_space_rescoring_with_conservative_activation",
            "sigma_scale_init": method_config.orth_sigma_init,
            "sigma_scale_min": method_config.orth_sigma_min,
            "sigma_scale_max": method_config.orth_sigma_max,
            "sigma_scale_step": method_config.orth_sigma_step,
            "orth_k_init": method_config.orth_k_init,
            "orth_k_min": method_config.orth_k_min,
            "orth_k_max": method_config.orth_k_max,
            "ubr_iqm_window": method_config.ubr_iqm_window,
            "ubr_epsilon": method_config.ubr_epsilon,
            "stagnation_patience": method_config.stagnation_patience,
            "stagnation_min_improve": method_config.stagnation_min_improve,
            "trigger_mode": method_config.trigger_mode,
            "trigger_consecutive_required": method_config.trigger_consecutive_required,
            "orthogonal_cooldown": method_config.orthogonal_cooldown,
            "mixed_r_take": method_config.mixed_r_take,
            "mixed_m_take": method_config.mixed_m_take,
            "local_subset_size": method_config.local_subset_size,
        },
        "baseline_model": "SingleTaskGP + LogExpectedImprovement + optimize_acqf",
        "runtime_logging": {
            "per_seed_csv": runtime_csv_path.name,
            "per_function_csv": "f*/runtime_log.csv",
            "summary_json": "runtime_summary.json",
            "clock": "time.perf_counter",
        },
        "parallel_functions": args.parallel_functions,
    }
    save_run_config(run_root / "config.json", config)

    function_runtime_csvs: List[Path] = []
    worker_count = min(max(1, args.parallel_functions), len(args.function_ids))
    if worker_count == 1:
        for fid in args.function_ids:
            function_runtime_csvs.append(run_single_function(fid, args, budget, run_root, method_config))
    else:
        mp_context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp_context) as executor:
            future_by_fid = {
                executor.submit(run_single_function, fid, args, budget, run_root, method_config): fid
                for fid in args.function_ids
            }
            for future in as_completed(future_by_fid):
                fid = future_by_fid[future]
                try:
                    function_runtime_csvs.append(future.result())
                except Exception as exc:
                    raise RuntimeError(f"Function worker failed for f{fid}") from exc

    merge_runtime_records(runtime_csv_path, function_runtime_csvs)
    print(f"Saved comparison outputs under: {run_root}")
    runtime_summary = summarize_runtime_records(runtime_csv_path)
    save_run_config(run_root / "runtime_summary.json", runtime_summary)
    return run_root


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
