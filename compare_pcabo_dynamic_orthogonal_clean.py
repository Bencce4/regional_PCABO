import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import ioh
import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.optimize import differential_evolution
from scipy.stats import qmc
from torch import pca_lowrank

from PCA_BO import PEI

METHOD_CLEAN = "pca_bo_dynamic_orthogonal_clean"
RUN_SUFFIX = "pca_bo_dynamic_orthogonal_clean"

KERNEL_TYPE = "matern"
KERNEL_NU = 2.5
KERNEL_LENGTHSCALE_BOUNDS = (0.005, 4.0)

ORTH_SIGMA_SCALE_INIT = 0.025
ORTH_SIGMA_SCALE_MIN = 0.0
ORTH_SIGMA_SCALE_MAX = 0.12
ORTH_SIGMA_SCALE_STEP = 0.0125
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
LOCAL_POINTS_PER_DIM = 6
DOE_POINTS_PER_DIM = 3
BUDGET_EVALS_PER_DIM = 30

CANDIDATE_MULTIPLIER = 100
CANDIDATE_MAX = 5000
ORTHOGONAL_REFINEMENT = "mixed_gp"


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
    candidate_multiplier: int = CANDIDATE_MULTIPLIER
    candidate_max: int = CANDIDATE_MAX
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
    orthogonal_refinement: str = ORTHOGONAL_REFINEMENT
    device: str = "auto"


@dataclass
class NormalizationStats:
    lower: torch.Tensor
    upper: torch.Tensor


CURRENT_INSTANCE = 1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_torch_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available.")
        return torch.device("cuda")
    if device_arg == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_problem(fid: int, dim: int, instance: Optional[int] = None):
    instance_id = CURRENT_INSTANCE if instance is None else instance
    return ioh.get_problem(fid, instance_id, dim, ioh.ProblemClass.BBOB)


def get_bounds(problem) -> Tuple[np.ndarray, np.ndarray]:
    lb = np.asarray(problem.bounds.lb, dtype=float)
    ub = np.asarray(problem.bounds.ub, dtype=float)
    return lb, ub


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


def bounds_violation_norm(x: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> float:
    x_2d = x.reshape(-1, lb.numel())
    lb_2d = lb.reshape(1, -1).to(dtype=x.dtype, device=x.device)
    ub_2d = ub.reshape(1, -1).to(dtype=x.dtype, device=x.device)
    lower_violation = torch.clamp(lb_2d - x_2d, min=0.0)
    upper_violation = torch.clamp(x_2d - ub_2d, min=0.0)
    violation = lower_violation + upper_violation
    return float(torch.linalg.norm(violation, dim=1).max().detach().cpu().item())


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


def make_global_latent_bounds(
    lb: torch.Tensor,
    ub: torch.Tensor,
    x_mean: torch.Tensor,
    x_weighted_mean: torch.Tensor,
    P_r: torch.Tensor,
) -> torch.Tensor:
    x_center = 0.5 * (lb + ub)
    half_width = 0.5 * torch.clamp(ub - lb, min=1e-12)
    z_center = ((x_center - x_mean) - x_weighted_mean) @ P_r.T
    z_half_width = half_width @ torch.abs(P_r.T)
    z_half_width = torch.clamp(z_half_width, min=1e-8)
    return torch.stack([z_center - z_half_width, z_center + z_half_width], dim=0)


def make_pcabo_latent_bounds(
    lb: torch.Tensor,
    ub: torch.Tensor,
    x_mean: torch.Tensor,
    x_weighted_mean: torch.Tensor,
    P_r: torch.Tensor,
) -> torch.Tensor:
    x_center = 0.5 * (lb + ub)
    z_center = ((x_center - x_mean) - x_weighted_mean) @ P_r.T
    half_width = 0.5 * torch.min(ub - lb)
    return torch.stack([z_center - half_width, z_center + half_width], dim=0)


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


def append_timing_row(timing_path: Path, row: Dict[str, object]) -> None:
    fieldnames = (
        "method",
        "function_id",
        "instance_id",
        "dim",
        "seed",
        "run_idx",
        "n0",
        "budget",
        "total_seconds",
        "seconds_per_evaluation",
    )
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not timing_path.exists()
    with timing_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def record_method_timing(
    timing_rows: List[Dict[str, object]],
    run_root: Path,
    function_root: Path,
    method: str,
    fid: int,
    instance_id: int,
    dim: int,
    seed: int,
    run_idx: int,
    n0: int,
    budget: int,
    total_seconds: float,
) -> None:
    row = {
        "method": method,
        "function_id": fid,
        "instance_id": instance_id,
        "dim": dim,
        "seed": seed,
        "run_idx": run_idx,
        "n0": n0,
        "budget": budget,
        "total_seconds": f"{total_seconds:.6f}",
        "seconds_per_evaluation": f"{total_seconds / max(1, budget):.6f}",
    }
    timing_rows.append(row)
    append_timing_row(run_root / "timing.csv", row)
    append_timing_row(function_root / "timing.csv", row)
    append_timing_row(function_root / method / f"seed_{seed}" / "timing.csv", row)


def save_timing_summary(run_root: Path, timing_rows: List[Dict[str, object]]) -> None:
    if not timing_rows:
        return

    grouped: Dict[Tuple[str, int, int, int], List[float]] = {}
    for row in timing_rows:
        key = (
            str(row["method"]),
            int(row["function_id"]),
            int(row["instance_id"]),
            int(row["budget"]),
        )
        grouped.setdefault(key, []).append(float(row["total_seconds"]))

    summary_path = run_root / "timing_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = (
            "method",
            "function_id",
            "instance_id",
            "budget",
            "runs",
            "mean_total_seconds",
            "median_total_seconds",
            "std_total_seconds",
            "min_total_seconds",
            "max_total_seconds",
            "mean_seconds_per_evaluation",
        )
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for (method, fid, instance_id, budget), values in sorted(grouped.items()):
            arr = np.asarray(values, dtype=float)
            writer.writerow(
                {
                    "method": method,
                    "function_id": fid,
                    "instance_id": instance_id,
                    "budget": budget,
                    "runs": int(arr.size),
                    "mean_total_seconds": f"{float(arr.mean()):.6f}",
                    "median_total_seconds": f"{float(np.median(arr)):.6f}",
                    "std_total_seconds": f"{float(arr.std(ddof=1)) if arr.size > 1 else 0.0:.6f}",
                    "min_total_seconds": f"{float(arr.min()):.6f}",
                    "max_total_seconds": f"{float(arr.max()):.6f}",
                    "mean_seconds_per_evaluation": f"{float(arr.mean() / max(1, budget)):.6f}",
                }
            )


ORTHOGONAL_STATS_FIELDS = (
    "iteration",
    "selected_mode",
    "pca_rank",
    "ubr_value",
    "ubr_trigger",
    "stagnation_trigger",
    "consecutive_trigger_count",
    "sigma_scale",
    "orthogonal_candidate_count",
    "cooldown_remaining",
)


def write_orthogonal_stats(out_path: Path, rows: List[Dict[str, object]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ORTHOGONAL_STATS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


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


def candidate_count_for_dim(active_dim: int, config: MethodConfig) -> int:
    return max(1, min(config.candidate_multiplier * max(1, active_dim), config.candidate_max))


def sobol_candidates_in_bounds(bounds: torch.Tensor, n_candidates: int, seed: int) -> torch.Tensor:
    lb = bounds[0].reshape(-1)
    ub = bounds[1].reshape(-1)
    active_dim = lb.numel()
    engine = torch.quasirandom.SobolEngine(dimension=active_dim, scramble=True, seed=seed)
    unit = engine.draw(max(1, n_candidates)).to(dtype=lb.dtype, device=lb.device)
    return lb.reshape(1, -1) + unit * (ub - lb).reshape(1, -1)


def posterior_mean_std(gp: SingleTaskGP, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        posterior = gp.posterior(z)
    mean = posterior.mean.squeeze(-1)
    std = torch.sqrt(torch.clamp(posterior.variance.squeeze(-1), min=1e-18))
    return mean, std


def squared_bounds_violation(x: torch.Tensor, bounds: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    lb, ub = bounds
    lb = lb.to(dtype=x.dtype, device=x.device).reshape(1, -1)
    ub = ub.to(dtype=x.dtype, device=x.device).reshape(1, -1)
    lower_violation = torch.clamp(lb - x, min=0.0)
    upper_violation = torch.clamp(x - ub, min=0.0)
    return torch.sum((lower_violation + upper_violation) ** 2, dim=-1)


def select_sobol_lcb_candidate(
    gp: SingleTaskGP,
    bounds: torch.Tensor,
    seed: int,
    config: MethodConfig,
    mapper=None,
    x_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    beta: float = 2.0,
    penalty_weight: float = 100.0,
    chunk_size: int = 2048,
) -> torch.Tensor:
    candidates = sobol_candidates_in_bounds(
        bounds=bounds,
        n_candidates=candidate_count_for_dim(bounds.shape[1], config),
        seed=seed,
    )
    best_idx = 0
    best_score = None
    beta_sqrt = math.sqrt(max(0.0, beta))
    with torch.no_grad():
        for start in range(0, candidates.shape[0], chunk_size):
            cand_chunk = candidates[start : start + chunk_size]
            mean, std = posterior_mean_std(gp, cand_chunk)
            score = mean - beta_sqrt * std
            if mapper is not None and x_bounds is not None:
                x_chunk = mapper(cand_chunk)
                score = score + penalty_weight * squared_bounds_violation(x_chunk, x_bounds)
            chunk_best_score, chunk_best_local_idx = torch.min(score.reshape(-1), dim=0)
            if best_score is None or bool((chunk_best_score < best_score).detach().cpu().item()):
                best_score = chunk_best_score
                best_idx = start + int(chunk_best_local_idx.detach().cpu().item())
    return candidates[best_idx : best_idx + 1]


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
        z_t = torch.tensor(
            z_np,
            dtype=z_train.dtype,
            device=z_train.device,
        ).view(1, -1)
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


def select_pei_de_candidate(
    gp: SingleTaskGP,
    bounds_z: torch.Tensor,
    best_f: torch.Tensor,
    x_bounds: Tuple[torch.Tensor, torch.Tensor],
    mapper,
    seed: int,
) -> torch.Tensor:
    acquisition = PEI(
        gp=gp,
        best_f=best_f,
        bounds=x_bounds,
        penalty_weight=100.0,
        mapper=mapper,
    )
    lbz = bounds_z[0].detach().cpu().numpy()
    ubz = bounds_z[1].detach().cpu().numpy()
    de_bounds = [(float(lower), float(upper)) for lower, upper in zip(lbz, ubz)]

    def objective(z_np: Iterable[float]) -> float:
        z_t = torch.tensor(
            z_np,
            dtype=bounds_z.dtype,
            device=bounds_z.device,
        ).view(1, 1, -1)
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
    return torch.tensor(result.x, dtype=bounds_z.dtype, device=bounds_z.device).view(1, -1)


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
    ).to(dtype=train_x.dtype, device=train_x.device)
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
    with torch.no_grad():
        mean, std = posterior_mean_std(gp, cand_x)
        scores = -(mean - math.sqrt(2.0) * std)
    return scores


def select_dynamic_candidate(
    x_manifold: torch.Tensor,
    X_hist: torch.Tensor,
    init_y: torch.Tensor,
    P_r: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
    activate_orthogonal: bool,
    state: DynamicOrthogonalState,
    seed: int,
    config: MethodConfig,
) -> Tuple[torch.Tensor, str, int]:
    manifold_violation = bounds_violation_norm(x_manifold, lb, ub)
    x_manifold = torch.clamp(x_manifold, min=lb, max=ub)
    if not activate_orthogonal:
        return x_manifold, "manifold", 0

    dim = x_manifold.shape[-1]
    rank = P_r.shape[0]
    if rank >= dim:
        return x_manifold, "manifold", 0

    U_perp = orthogonal_basis_from_projection(P_r)
    if U_perp.shape[1] == 0:
        return x_manifold, "manifold", 0
    if config.orthogonal_refinement not in {"simple", "mixed_gp"}:
        raise ValueError(f"Unknown orthogonal refinement mode: {config.orthogonal_refinement}")

    generator = torch.Generator(device=x_manifold.device)
    generator.manual_seed(seed)
    sigma_t = max(config.orth_sigma_min, min(config.orth_sigma_max, state.sigma_scale))
    sigma_t = sigma_t * float(torch.mean(ub - lb).detach().cpu().item())

    n_orth_samples = 1 if config.orthogonal_refinement == "simple" else max(1, state.orth_k)
    gaussian = torch.randn(
        n_orth_samples,
        dim,
        dtype=x_manifold.dtype,
        device=x_manifold.device,
        generator=generator,
    ) * sigma_t
    projector_perp = U_perp @ U_perp.T
    noise_perp = gaussian @ projector_perp.T
    nonzero_mask = torch.linalg.norm(noise_perp, dim=-1) > 1e-12
    if not bool(nonzero_mask.any().detach().cpu().item()):
        return x_manifold, "manifold", 0

    x_orth_raw = x_manifold + noise_perp[nonzero_mask]
    n_generated = int(x_orth_raw.shape[0])
    if config.orthogonal_refinement == "simple":
        x_orth = torch.clamp(x_orth_raw[:1], min=lb, max=ub)
        return x_orth, "orthogonal", n_generated

    x_orth = torch.clamp(x_orth_raw, min=lb, max=ub)
    candidate_x = torch.cat([x_manifold, x_orth], dim=0)

    B = select_mixed_basis(P_r, U_perp, config)
    if B is None:
        return x_manifold, "manifold", n_generated

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
    selected = candidate_x[best_idx : best_idx + 1]
    if manifold_violation > 1e-10 and selected_mode == "manifold":
        selected = torch.clamp(selected, min=lb, max=ub)
    return selected, selected_mode, n_generated


def run_clean_dynamic_orthogonal_pcabo(
    fid: int,
    dim: int,
    seed: int,
    run_idx: int,
    n0: int,
    budget: int,
    dat_path: Path,
    initial_x: torch.Tensor,
    config: MethodConfig,
) -> None:
    set_seed(seed)
    device = resolve_torch_device(config.device)
    problem = create_problem(fid=fid, dim=dim)
    stats_rows: List[Dict[str, object]] = []

    with EvaluationDatLogger(dat_path=dat_path) as eval_logger:
        lb = torch.tensor(problem.bounds.lb, dtype=torch.double, device=device)
        ub = torch.tensor(problem.bounds.ub, dtype=torch.double, device=device)

        X_hist = initial_x.clone().to(dtype=torch.double, device=device)
        y_hist = evaluate_initial_design(problem=problem, init_x=X_hist, eval_logger=eval_logger).to(device=device)
        state = DynamicOrthogonalState(
            sigma_scale=config.orth_sigma_init,
            orth_k=config.orth_k_init,
            best_history=[float(y_hist.min().detach().cpu().item())],
        )

        while problem.state.evaluations < budget:
            iteration = int(problem.state.evaluations)
            decrement_cooldown(state)

            x_mean, x_weighted_mean, P_r, _, _ = compute_pca_lowrank(X_hist, y_hist, alpha=0.95)
            z_r = ((X_hist - x_mean) - x_weighted_mean) @ P_r.T
            z_norm_stats = make_normalization_stats(z_r)
            z_r_norm = normalize_values(z_r, z_norm_stats)
            mapper = lambda z: z @ P_r + x_mean + x_weighted_mean
            mapper_norm = lambda z_norm: mapper(denormalize_values(z_norm, z_norm_stats))

            bounds_z = make_pcabo_latent_bounds(
                lb=lb,
                ub=ub,
                x_mean=x_mean,
                x_weighted_mean=x_weighted_mean,
                P_r=P_r,
            )
            bounds_z_norm = normalize_values(bounds_z, z_norm_stats)

            gp = fit_gp_with_config(train_x=z_r_norm, train_y=y_hist, config=config)
            beta_t = 2.0 * math.log(float(max(1.0000001, z_r.shape[1] * (problem.state.evaluations + 1) ** 2)))
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
            consecutive_trigger_count_for_row = state.consecutive_trigger_count
            activate_orthogonal = should_activate_orthogonal(state, rank=P_r.shape[0], dim=dim, config=config)
            adjust_dynamic_orthogonal_state(
                state=state,
                activate_orthogonal=activate_orthogonal,
                iter_idx=problem.state.evaluations,
                config=config,
            )

            new_z_norm = select_pei_de_candidate(
                gp=gp,
                bounds_z=bounds_z_norm,
                best_f=y_hist.min(),
                x_bounds=(lb, ub),
                mapper=mapper_norm,
                seed=seed + 1543 * (problem.state.evaluations + 1),
            )
            new_x = mapper_norm(new_z_norm)
            x_eval, selected_mode, orth_candidate_count = select_dynamic_candidate(
                x_manifold=new_x,
                X_hist=X_hist,
                init_y=y_hist,
                P_r=P_r,
                lb=lb,
                ub=ub,
                activate_orthogonal=activate_orthogonal,
                state=state,
                seed=seed + problem.state.evaluations,
                config=config,
            )
            x_eval = torch.clamp(x_eval, min=lb, max=ub)

            previous_best = state.best_history[-1]
            current_y = evaluate_point(
                problem=problem,
                x_np=x_eval.detach().cpu().numpy().reshape(-1),
                eval_logger=eval_logger,
            )
            new_y = torch.tensor([[current_y]], dtype=y_hist.dtype, device=device)
            current_best = min(previous_best, current_y)
            state.last_selected_mode = selected_mode
            state.last_eval_improved = current_best < previous_best - config.stagnation_min_improve
            state.best_history.append(current_best)
            if selected_mode == "orthogonal":
                state.cooldown_remaining = config.orthogonal_cooldown

            stats_rows.append(
                {
                    "iteration": iteration,
                    "selected_mode": selected_mode,
                    "pca_rank": int(P_r.shape[0]),
                    "ubr_value": f"{latent_ubr:.16e}",
                    "ubr_trigger": bool(ubr_trigger),
                    "stagnation_trigger": bool(stagnation_trigger),
                    "consecutive_trigger_count": int(consecutive_trigger_count_for_row),
                    "sigma_scale": f"{state.sigma_scale:.16f}",
                    "orthogonal_candidate_count": int(orth_candidate_count),
                    "cooldown_remaining": int(state.cooldown_remaining),
                }
            )

            X_hist = torch.cat((X_hist, x_eval), dim=0)
            y_hist = torch.cat((y_hist, new_y), dim=0)

    write_orthogonal_stats(dat_path.parent / "orthogonal_stats.csv", stats_rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run clean PCA-BO with dynamic orthogonal exploration and no local-region wrapper.",
    )
    parser.add_argument("--base-seed", type=int, default=12)
    parser.add_argument("--function-ids", nargs="+", type=int, default=[2])
    parser.add_argument("--instance-id", type=int, default=1)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--n0", type=int, default=None, help="Initial DoE size. Defaults to 3 * dim.")
    parser.add_argument("--budget", type=int, default=None, help="Evaluation budget. Defaults to 30 * dim.")
    parser.add_argument("--num-runs", type=int, default=30)
    parser.add_argument("--grid-size", type=int, default=120)
    parser.add_argument("--run-root", type=str, default=None)
    parser.add_argument("--kernel-type", choices=["matern", "rbf"], default=KERNEL_TYPE)
    parser.add_argument("--kernel-nu", type=float, choices=[0.5, 1.5, 2.5], default=KERNEL_NU)
    parser.add_argument("--lengthscale-lb", type=float, default=KERNEL_LENGTHSCALE_BOUNDS[0])
    parser.add_argument("--lengthscale-ub", type=float, default=KERNEL_LENGTHSCALE_BOUNDS[1])
    parser.add_argument("--candidate-multiplier", type=int, default=CANDIDATE_MULTIPLIER)
    parser.add_argument("--candidate-max", type=int, default=CANDIDATE_MAX)
    parser.add_argument("--orth-sigma-init", type=float, default=ORTH_SIGMA_SCALE_INIT)
    parser.add_argument("--orth-sigma-max", type=float, default=ORTH_SIGMA_SCALE_MAX)
    parser.add_argument("--orth-sigma-step", type=float, default=ORTH_SIGMA_SCALE_STEP)
    parser.add_argument("--orth-k-max", type=int, default=ORTH_K_MAX)
    parser.add_argument("--ubr-epsilon", type=float, default=UBR_EPSILON)
    parser.add_argument("--stagnation-patience", type=int, default=STAGNATION_PATIENCE)
    parser.add_argument("--trigger-mode", choices=["and", "or"], default=TRIGGER_MODE)
    parser.add_argument("--trigger-consecutive-required", type=int, default=TRIGGER_CONSECUTIVE_REQUIRED)
    parser.add_argument("--orthogonal-cooldown", type=int, default=ORTHOGONAL_COOLDOWN)
    parser.add_argument(
        "--orthogonal-refinement",
        choices=["simple", "mixed_gp"],
        default=ORTHOGONAL_REFINEMENT,
        help="simple evaluates one projected perturbation; mixed_gp rescoring follows the v5 non-wrapped path.",
    )
    parser.add_argument("--mixed-r-take", type=int, default=MIXED_R_TAKE)
    parser.add_argument("--mixed-m-take", type=int, default=MIXED_M_TAKE)
    parser.add_argument(
        "--local-subset-size",
        type=int,
        default=None,
        help="Nearest-neighbor training set size for mixed-space orthogonal rescoring. Defaults to 6 * dim.",
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def build_method_config(args) -> MethodConfig:
    return MethodConfig(
        kernel_type=args.kernel_type,
        kernel_nu=args.kernel_nu,
        lengthscale_bounds=(args.lengthscale_lb, args.lengthscale_ub),
        candidate_multiplier=args.candidate_multiplier,
        candidate_max=args.candidate_max,
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
        orthogonal_refinement=args.orthogonal_refinement,
        device=args.device,
    )


def run_experiment(args) -> Path:
    global CURRENT_INSTANCE
    CURRENT_INSTANCE = int(args.instance_id)

    scaled_n0 = args.n0 is None
    scaled_budget = args.budget is None
    scaled_local_subset = args.local_subset_size is None

    args.n0 = args.n0 if args.n0 is not None else DOE_POINTS_PER_DIM * args.dim
    args.budget = args.budget if args.budget is not None else BUDGET_EVALS_PER_DIM * args.dim
    args.local_subset_size = (
        args.local_subset_size if args.local_subset_size is not None else LOCAL_POINTS_PER_DIM * args.dim
    )
    budget = args.budget
    method_config = build_method_config(args)

    if args.run_root:
        run_root = Path(args.run_root)
    else:
        run_root = Path("comparison_runs") / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{RUN_SUFFIX}"
    run_root.mkdir(parents=True, exist_ok=True)

    config = {
        "base_seed": args.base_seed,
        "function_ids": args.function_ids,
        "instance_id": args.instance_id,
        "dim": args.dim,
        "n0": args.n0,
        "budget": budget,
        "dimension_scaled_defaults": {
            "n0": {
                "used": scaled_n0,
                "formula": f"{DOE_POINTS_PER_DIM} * D",
                "resolved_value": args.n0,
            },
            "budget": {
                "used": scaled_budget,
                "formula": f"{BUDGET_EVALS_PER_DIM} * D",
                "resolved_value": budget,
            },
            "local_subset_size": {
                "used": scaled_local_subset,
                "formula": f"{LOCAL_POINTS_PER_DIM} * D",
                "resolved_value": method_config.local_subset_size,
            },
            "D": args.dim,
        },
        "num_runs": args.num_runs,
        "grid_size": args.grid_size,
        "selected_methods": [METHOD_CLEAN],
        "method_descriptions": {
            METHOD_CLEAN: (
                "PCA-BO with global PCA latent candidate search and dynamic orthogonal exploration."
            ),
        },
        "kernel": {
            "type": method_config.kernel_type,
            "nu": method_config.kernel_nu,
            "lengthscale_bounds": method_config.lengthscale_bounds,
        },
        "pca_backend": "torch.pca_lowrank",
        "candidate_proposal": {
            "device": method_config.device,
            "strategy": "PEI optimized with scipy differential_evolution in normalized PCA latent coordinates",
            "candidate_count": "differential_evolution maxiter=80, popsize=15",
            "global_bounds": "PCA-BO latent search box centered at the BBOB box center",
        },
        "dynamic_orthogonal_exploration": {
            "controller_rule": "UBR plateau and best-value stagnation with consecutive checks and cooldown",
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
            "orthogonal_refinement": method_config.orthogonal_refinement,
            "mixed_r_take": method_config.mixed_r_take,
            "mixed_m_take": method_config.mixed_m_take,
            "local_subset_size": method_config.local_subset_size,
            "candidate_clamp": "global BBOB box bounds only",
        },
        "outputs": {
            "evaluation_log": "IOHprofiler_f{fid}_DIM{dim}.dat",
            "orthogonal_stats": "orthogonal_stats.csv",
            "timing": "timing.csv",
        },
    }
    save_run_config(run_root / "config.json", config)
    timing_rows: List[Dict[str, object]] = []

    for fid in args.function_ids:
        print(f"Running f{fid}, instance {args.instance_id}: {args.num_runs} runs")
        function_root = run_root / f"f{fid}"

        for run_idx in range(args.num_runs):
            seed = args.base_seed + run_idx
            set_seed(seed)

            sampling_problem = create_problem(fid=fid, dim=args.dim)
            lb_np, ub_np = get_bounds(sampling_problem)
            lb = torch.tensor(lb_np, dtype=torch.double)
            ub = torch.tensor(ub_np, dtype=torch.double)
            initial_x = sample_initial_design(lb=lb, ub=ub, dim=args.dim, n0=args.n0, seed=seed)

            start = time.perf_counter()
            run_clean_dynamic_orthogonal_pcabo(
                fid=fid,
                dim=args.dim,
                seed=seed,
                run_idx=run_idx,
                n0=args.n0,
                budget=budget,
                dat_path=function_root / METHOD_CLEAN / f"seed_{seed}" / f"IOHprofiler_f{fid}_DIM{args.dim}.dat",
                initial_x=initial_x,
                config=method_config,
            )
            record_method_timing(
                timing_rows=timing_rows,
                run_root=run_root,
                function_root=function_root,
                method=METHOD_CLEAN,
                fid=fid,
                instance_id=args.instance_id,
                dim=args.dim,
                seed=seed,
                run_idx=run_idx,
                n0=args.n0,
                budget=budget,
                total_seconds=time.perf_counter() - start,
            )

    save_timing_summary(run_root=run_root, timing_rows=timing_rows)
    print(f"Saved clean dynamic orthogonal PCA-BO outputs under: {run_root}")
    return run_root


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
