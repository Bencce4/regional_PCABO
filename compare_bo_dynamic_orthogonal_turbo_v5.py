import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import ioh
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Rectangle
import numpy as np
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.optimize import differential_evolution
from scipy.stats import qmc
from torch import pca_lowrank

from PCA_BO import PEI, plot_pcabo_iteration, plot_weighted_points_iteration


METHOD_PCA_VANILLA = "pca_bo"
METHOD_DYNAMIC = "pca_bo_dynamic_orthogonal_v2"
METHOD_TURBO = "pca_bo_dynamic_orthogonal_turbo_v1"
METHOD_TURBO_V2 = "pca_bo_dynamic_orthogonal_turbo_v2"
METHOD_TURBO_V3 = "pca_bo_dynamic_orthogonal_turbo_v3"
METHOD_TURBO_V4 = "pca_bo_dynamic_orthogonal_turbo_v4"
METHOD_TURBO_V5 = "pca_bo_dynamic_orthogonal_turbo_v5"
METHOD_BASELINE = "botorch_baseline"
METHODS = (METHOD_PCA_VANILLA, METHOD_DYNAMIC, METHOD_TURBO_V2, METHOD_TURBO_V5, METHOD_BASELINE)
RUN_SUFFIX = "pca_kernel_pcalowrank_dynamic_orthogonal_turbo_v5_region_radius"
KERNEL_NU = 2.5
KERNEL_LENGTHSCALE_BOUNDS = (0.005, 4.0)
METHOD_LABELS = {
    METHOD_PCA_VANILLA: "PCA-BO",
    METHOD_DYNAMIC: "PCA-BO + Dynamic Orthogonal Exploration V2",
    METHOD_TURBO: "PCA-BO + Dynamic Orthogonal Exploration + TuRBO-1",
    METHOD_TURBO_V2: "PCA-BO + Dynamic Orthogonal Exploration + TuRBO-2",
    METHOD_TURBO_V3: "PCA-BO + Dynamic Orthogonal Exploration + TuRBO-3",
    METHOD_TURBO_V4: "PCA-BO + Dynamic Orthogonal Exploration + TuRBO-4 Evaluated Restart",
    METHOD_TURBO_V5: "PCA-BO + Dynamic Orthogonal Exploration + TuRBO-5 Region Radius",
    METHOD_BASELINE: "BoTorch Baseline",
}
METHOD_COLORS = {
    METHOD_PCA_VANILLA: "#9467bd",
    METHOD_DYNAMIC: "#1f77b4",
    METHOD_TURBO: "#2ca02c",
    METHOD_TURBO_V2: "#ff7f0e",
    METHOD_TURBO_V3: "#17becf",
    METHOD_TURBO_V4: "#8c564b",
    METHOD_TURBO_V5: "#8c564b",
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
DOE_POINTS_PER_DIM = 3
BUDGET_EVALS_PER_DIM = 30
LOCAL_POINTS_PER_DIM = 6
TURBO_LENGTH_INIT = 0.8
TURBO_LENGTH_MIN = 2.0**-7
TURBO_LENGTH_MAX = 1.6
TURBO_SUCCESS_TOLERANCE = 3
TURBO_NUM_REGIONS = 5
TURBO_CANDIDATE_MULTIPLIER = 100
TURBO_CANDIDATE_MAX = 5000
TURBO_INIT_POINTS_PER_REGION = 20
TURBO_V3_ARD_SHAPE = False
TURBO_V3_ORTH_SCALE = 0.25
TURBO_V5_LATENT_RADIUS_SCALE = 0.75
ORTHOGONAL_REFINEMENT = "simple"
TURBO_RESTART_MODE = "distance"
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
class TrustRegionState:
    length: float = TURBO_LENGTH_INIT
    success_counter: int = 0
    failure_counter: int = 0
    restart_count: int = 0


@dataclass
class TurboV2RegionState:
    center: torch.Tensor
    length: float = TURBO_LENGTH_INIT
    success_counter: int = 0
    failure_counter: int = 0
    restart_count: int = 0
    best_x: Optional[torch.Tensor] = None
    best_y: float = float("inf")
    last_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    last_lengthscales: Optional[torch.Tensor] = None


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
    turbo_length_init: float = TURBO_LENGTH_INIT
    turbo_length_min: float = TURBO_LENGTH_MIN
    turbo_length_max: float = TURBO_LENGTH_MAX
    turbo_success_tolerance: int = TURBO_SUCCESS_TOLERANCE
    turbo_failure_tolerance: Optional[int] = None
    turbo_num_regions: int = TURBO_NUM_REGIONS
    turbo_candidate_multiplier: int = TURBO_CANDIDATE_MULTIPLIER
    turbo_candidate_max: int = TURBO_CANDIDATE_MAX
    turbo_init_points_per_region: int = TURBO_INIT_POINTS_PER_REGION
    turbo_v3_ard_shape: bool = TURBO_V3_ARD_SHAPE
    turbo_v3_orth_scale: float = TURBO_V3_ORTH_SCALE
    turbo_v5_latent_radius_scale: float = TURBO_V5_LATENT_RADIUS_SCALE
    turbo_restart_points: Optional[int] = None
    turbo_restart_mode: str = TURBO_RESTART_MODE
    orthogonal_refinement: str = ORTHOGONAL_REFINEMENT
    device: str = "auto"


@dataclass
class NormalizationStats:
    lower: torch.Tensor
    upper: torch.Tensor


@dataclass
class ClampDiagnostics:
    manifold_pre_clamp_violation: float = 0.0
    orthogonal_pre_clamp_violation: float = 0.0
    final_region_clamped: bool = False


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


CURRENT_INSTANCE = 1


def create_problem(fid: int, dim: int, instance: Optional[int] = None):
    instance_id = CURRENT_INSTANCE if instance is None else instance
    return ioh.get_problem(fid, instance_id, dim, ioh.ProblemClass.BBOB)


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


CLAMP_STATS_FIELDS = (
    "method",
    "function_id",
    "dim",
    "seed",
    "run_idx",
    "iteration",
    "mode",
    "rank",
    "region_length",
    "success_counter_before",
    "failure_counter_before",
    "restart_count_before",
    "manifold_pre_clamp_violation",
    "orthogonal_pre_clamp_violation",
    "final_region_clamped",
    "final_global_clamped",
    "region_length_after_update",
    "success_counter_after_update",
    "failure_counter_after_update",
    "restart_count_after_update",
    "restart_event",
)

CLAMP_SUMMARY_FIELDS = (
    "method",
    "function_id",
    "dim",
    "seed",
    "run_idx",
    "n_iterations",
    "manifold_clamp_count",
    "orthogonal_clamp_count",
    "final_region_clamp_count",
    "final_global_clamp_count",
    "manifold_clamp_rate",
    "orthogonal_clamp_rate",
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
    "region_length",
    "region_length_after_update",
    "restart_count_before",
    "restart_count_after_update",
    "restart_event",
    "orthogonal_pre_clamp_violation",
    "final_region_clamped",
    "final_global_clamped",
)


def write_clamp_rows(out_path: Path, rows: List[Dict[str, object]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLAMP_STATS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_clamp_summary(out_path: Path, rows: List[Dict[str, object]]) -> None:
    if rows:
        first = rows[0]
        n_iterations = len(rows)
        manifold_count = sum(float(row["manifold_pre_clamp_violation"]) > 1e-10 for row in rows)
        orthogonal_count = sum(float(row["orthogonal_pre_clamp_violation"]) > 1e-10 for row in rows)
        final_region_count = sum(bool(row["final_region_clamped"]) for row in rows)
        final_global_count = sum(bool(row["final_global_clamped"]) for row in rows)
        summary_rows = [
            {
                "method": first["method"],
                "function_id": first["function_id"],
                "dim": first["dim"],
                "seed": first["seed"],
                "run_idx": first["run_idx"],
                "n_iterations": n_iterations,
                "manifold_clamp_count": manifold_count,
                "orthogonal_clamp_count": orthogonal_count,
                "final_region_clamp_count": final_region_count,
                "final_global_clamp_count": final_global_count,
                "manifold_clamp_rate": manifold_count / max(1, n_iterations),
                "orthogonal_clamp_rate": orthogonal_count / max(1, n_iterations),
            }
        ]
    else:
        summary_rows = []

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLAMP_SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)


def write_orthogonal_stats(out_path: Path, rows: List[Dict[str, object]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ORTHOGONAL_STATS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def append_timing_row(timing_path: Path, row: Dict[str, object]) -> None:
    fieldnames = (
        "method",
        "function_id",
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

    grouped: Dict[Tuple[str, int, int], List[float]] = {}
    for row in timing_rows:
        key = (str(row["method"]), int(row["function_id"]), int(row["budget"]))
        grouped.setdefault(key, []).append(float(row["total_seconds"]))

    summary_path = run_root / "timing_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = (
            "method",
            "function_id",
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
        for (method, fid, budget), values in sorted(grouped.items()):
            arr = np.asarray(values, dtype=float)
            writer.writerow(
                {
                    "method": method,
                    "function_id": fid,
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


def save_convergence_plots(
    function_root: Path,
    fid: int,
    dim: int,
    budget: int,
    methods: Tuple[str, ...] = METHODS,
) -> None:
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


def _contour_norm(Z: np.ndarray):
    zmin = float(np.min(Z))
    zmax = float(np.max(Z))
    if zmin >= 0.0:
        zcap = float(np.percentile(Z, 99.5))
        if zcap <= zmin:
            zcap = zmax
        return mcolors.PowerNorm(gamma=0.35, vmin=zmin, vmax=zcap)
    linthresh = max(1e-3, 0.01 * max(abs(zmin), abs(zmax)))
    return mcolors.SymLogNorm(linthresh=linthresh, vmin=zmin, vmax=zmax, base=10)


def _pca_lines_in_box(
    center: np.ndarray,
    P_r: torch.Tensor,
    eigvals: torch.Tensor,
    box_lb: np.ndarray,
    box_ub: np.ndarray,
) -> List[Tuple[int, np.ndarray]]:
    lines: List[Tuple[int, np.ndarray]] = []
    if P_r.numel() == 0:
        return lines

    eig_used = eigvals[: P_r.shape[0]].detach().cpu().numpy()
    eig_used = np.clip(eig_used, a_min=0.0, a_max=None)
    scales = np.sqrt(eig_used + 1e-12)
    if scales.size == 0:
        scales = np.ones(P_r.shape[0], dtype=float)
    elif np.max(scales) > 0:
        scales = scales / np.max(scales)
    else:
        scales = np.ones_like(scales)

    base_extent = 0.5 * float(np.min(box_ub - box_lb))
    for k in range(P_r.shape[0]):
        v = P_r[k].detach().cpu().numpy()
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-12:
            continue
        v = v / v_norm

        t_min = -np.inf
        t_max = np.inf
        for j in range(2):
            if abs(v[j]) < 1e-12:
                if center[j] < box_lb[j] or center[j] > box_ub[j]:
                    t_min, t_max = 1.0, 0.0
                    break
                continue
            t1 = (box_lb[j] - center[j]) / v[j]
            t2 = (box_ub[j] - center[j]) / v[j]
            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))

        half_len = max(1e-12, base_extent * scales[min(k, len(scales) - 1)])
        t_min = max(t_min, -half_len)
        t_max = min(t_max, half_len)
        if not np.isfinite(t_min) or not np.isfinite(t_max) or t_min >= t_max:
            continue
        t = np.linspace(t_min, t_max, 160)
        lines.append((k, center[None, :] + t[:, None] * v[None, :]))
    return lines


def plot_turbo_v2_iteration(
    contour_cache: Dict[str, np.ndarray],
    X_hist: torch.Tensor,
    local_x: torch.Tensor,
    x_mean: torch.Tensor,
    x_weighted_mean: torch.Tensor,
    P_r: torch.Tensor,
    eigvals: torch.Tensor,
    region_center: torch.Tensor,
    tr_lb: torch.Tensor,
    tr_ub: torch.Tensor,
    thompson_candidate: torch.Tensor,
    final_candidate: torch.Tensor,
    iter_idx: int,
    func_id: int,
    dim: int,
    selected_mode: str,
    region_length: float,
    success_counter: int,
    failure_counter: int,
    restart_count: int,
    out_dir: Path,
) -> None:
    assert X_hist.shape[1] == 2, "X_hist must be (n,2)."
    X1 = contour_cache["X1"]
    X2 = contour_cache["X2"]
    Z = contour_cache["Z"]
    lb = contour_cache["lb"]
    ub = contour_cache["ub"]

    X_np = X_hist.detach().cpu().numpy()
    region_center_np = region_center.detach().cpu().numpy().reshape(2)
    tr_lb_np = np.maximum(tr_lb.detach().cpu().numpy().reshape(2), lb)
    tr_ub_np = np.minimum(tr_ub.detach().cpu().numpy().reshape(2), ub)
    pca_lines = _pca_lines_in_box(
        region_center_np,
        P_r.detach().cpu(),
        eigvals.detach().cpu(),
        tr_lb_np,
        tr_ub_np,
    )

    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    cs = ax.contourf(X1, X2, Z, levels=120, cmap="cividis", norm=_contour_norm(Z), alpha=0.94)
    ax.contour(X1, X2, Z, levels=22, colors="black", linewidths=0.35, alpha=0.35)
    plt.colorbar(cs, ax=ax, label="f(x)")

    rect = Rectangle(
        (tr_lb_np[0], tr_lb_np[1]),
        tr_ub_np[0] - tr_lb_np[0],
        tr_ub_np[1] - tr_lb_np[1],
        facecolor="none",
        edgecolor="black",
        linewidth=1.0,
        label="Active TuRBO region",
        zorder=2,
    )
    ax.add_patch(rect)

    cmap = plt.get_cmap("cool")
    for k, line_k in pca_lines:
        color = cmap(k / max(1, len(pca_lines) - 1))
        label = "Local PCA direction" if k == 0 else None
        ax.plot(line_k[:, 0], line_k[:, 1], color=color, linewidth=2.4, label=label, zorder=4)

    previous_np = X_np[:-1]
    current_np = X_np[-1:] if X_np.size else np.empty((0, 2))
    if previous_np.size:
        ax.scatter(
            previous_np[:, 0],
            previous_np[:, 1],
            c="white",
            s=24,
            edgecolors="black",
            linewidths=0.3,
            alpha=0.86,
            label="Previous evaluated points",
            zorder=3,
        )
    if current_np.size:
        ax.scatter(
            current_np[:, 0],
            current_np[:, 1],
            c="#d62728",
            s=52,
            edgecolors="black",
            linewidths=0.5,
            alpha=0.98,
            label="Current point",
            zorder=5,
        )

    ax.set_xlim(lb[0], ub[0])
    ax.set_ylim(lb[1], ub[1])
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(
        f"f{func_id} d{dim} | TuRBO1 iteration {iter_idx} | "
        f"mode={selected_mode}, L={region_length:.3f}, s/f={success_counter}/{failure_counter}, r={restart_count}"
    )
    ax.grid(alpha=0.18)
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / f"iter_{iter_idx:03d}.png", dpi=165)
    plt.close(fig)


def posterior_mean_std(gp: SingleTaskGP, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        posterior = gp.posterior(z)
    mean = posterior.mean.squeeze(-1)
    std = torch.sqrt(torch.clamp(posterior.variance.squeeze(-1), min=1e-18))
    return mean, std


def candidate_count_for_dim(active_dim: int, config: MethodConfig) -> int:
    return max(1, min(config.turbo_candidate_multiplier * max(1, active_dim), config.turbo_candidate_max))


def sobol_candidates_in_bounds(bounds: torch.Tensor, n_candidates: int, seed: int) -> torch.Tensor:
    lb = bounds[0].reshape(-1)
    ub = bounds[1].reshape(-1)
    active_dim = lb.numel()
    engine = torch.quasirandom.SobolEngine(dimension=active_dim, scramble=True, seed=seed)
    unit = engine.draw(max(1, n_candidates)).to(dtype=lb.dtype, device=lb.device)
    return lb.reshape(1, -1) + unit * (ub - lb).reshape(1, -1)


def select_sobol_acquisition_candidate(
    acquisition,
    bounds: torch.Tensor,
    seed: int,
    config: MethodConfig,
    chunk_size: int = 2048,
) -> torch.Tensor:
    candidates = sobol_candidates_in_bounds(
        bounds=bounds,
        n_candidates=candidate_count_for_dim(bounds.shape[1], config),
        seed=seed,
    )
    best_idx = 0
    best_score = None
    with torch.no_grad():
        for start in range(0, candidates.shape[0], chunk_size):
            cand_chunk = candidates[start : start + chunk_size]
            scores = acquisition(cand_chunk.unsqueeze(-2)).reshape(-1)
            chunk_best_score, chunk_best_local_idx = torch.max(scores, dim=0)
            if best_score is None or bool((chunk_best_score > best_score).detach().cpu().item()):
                best_score = chunk_best_score
                best_idx = start + int(chunk_best_local_idx.detach().cpu().item())
    return candidates[best_idx : best_idx + 1]


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

    search_points = sobol_candidates_in_bounds(
        bounds=bounds_z,
        n_candidates=candidate_count_for_dim(bounds_z.shape[1], config),
        seed=seed,
    )
    min_search_lcb = float("inf")
    with torch.no_grad():
        for start in range(0, search_points.shape[0], 2048):
            z_t = search_points[start : start + 2048]
            mean, std = posterior_mean_std(gp, z_t)
            chunk_lcb = mean - beta * std
            min_search_lcb = min(min_search_lcb, float(chunk_lcb.min().detach().cpu().item()))
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


def get_incumbent(X_hist: torch.Tensor, y_hist: torch.Tensor) -> Tuple[torch.Tensor, float]:
    best_idx = int(torch.argmin(y_hist.squeeze(-1)).detach().cpu().item())
    best_x = X_hist[best_idx : best_idx + 1]
    best_y = float(y_hist[best_idx].detach().cpu().item())
    return best_x, best_y


def make_trust_region_bounds(
    center: torch.Tensor,
    lb: torch.Tensor,
    ub: torch.Tensor,
    tr_state: TrustRegionState,
) -> Tuple[torch.Tensor, torch.Tensor]:
    center = center.reshape(-1)
    half_width = 0.5 * tr_state.length * (ub - lb)
    tr_lb = torch.maximum(lb, center - half_width)
    tr_ub = torch.minimum(ub, center + half_width)
    return tr_lb, tr_ub


def make_latent_trust_region_bounds(
    tr_center: torch.Tensor,
    tr_lb: torch.Tensor,
    tr_ub: torch.Tensor,
    x_mean: torch.Tensor,
    x_weighted_mean: torch.Tensor,
    P_r: torch.Tensor,
) -> torch.Tensor:
    z_center = ((tr_center.reshape(-1) - x_mean) - x_weighted_mean) @ P_r.T
    half_width = 0.5 * torch.clamp(tr_ub - tr_lb, min=1e-12)
    z_half_width = torch.sum(torch.abs(P_r) * half_width.unsqueeze(0), dim=1)
    z_half_width = torch.clamp(z_half_width, min=1e-8)
    return torch.stack([z_center - z_half_width, z_center + z_half_width], dim=0)


def make_latent_region_radius_bounds(
    tr_center: torch.Tensor,
    tr_lb: torch.Tensor,
    tr_ub: torch.Tensor,
    x_mean: torch.Tensor,
    x_weighted_mean: torch.Tensor,
    P_r: torch.Tensor,
    radius_scale: float = 1.0,
) -> torch.Tensor:
    z_center = ((tr_center.reshape(-1) - x_mean) - x_weighted_mean) @ P_r.T
    rho_tr = 0.5 * max(0.0, float(radius_scale)) * torch.min(torch.clamp(tr_ub - tr_lb, min=1e-12))
    rho_tr = torch.clamp(rho_tr, min=1e-8)
    return torch.stack([z_center - rho_tr, z_center + rho_tr], dim=0)


def update_trust_region_state(
    tr_state: TrustRegionState,
    improved: bool,
    dim: int,
    config: MethodConfig,
) -> None:
    failure_tolerance = config.turbo_failure_tolerance
    if failure_tolerance is None:
        failure_tolerance = max(1, dim)

    if improved:
        tr_state.success_counter += 1
        tr_state.failure_counter = 0
    else:
        tr_state.success_counter = 0
        tr_state.failure_counter += 1

    if tr_state.success_counter >= config.turbo_success_tolerance:
        tr_state.length = min(config.turbo_length_max, 2.0 * tr_state.length)
        tr_state.success_counter = 0
        tr_state.failure_counter = 0
    elif tr_state.failure_counter >= failure_tolerance:
        tr_state.length *= 0.5
        tr_state.success_counter = 0
        tr_state.failure_counter = 0

    if tr_state.length < config.turbo_length_min:
        tr_state.length = config.turbo_length_init
        tr_state.success_counter = 0
        tr_state.failure_counter = 0
        tr_state.restart_count += 1


def to_unit(values: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    return (values - lb) / torch.clamp(ub - lb, min=1e-12)


def from_unit(values: torch.Tensor, lb: torch.Tensor, ub: torch.Tensor) -> torch.Tensor:
    return lb + values * (ub - lb)


def generate_global_sobol_points(
    dim: int,
    n: int,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    engine = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=seed)
    return engine.draw(n).to(dtype=dtype, device=device)


def get_region_uniform_bounds_unit(
    region: TurboV2RegionState,
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    center = region.center.to(dtype=dtype, device=device).reshape(-1)
    half_width = torch.full((dim,), 0.5 * region.length, dtype=dtype, device=device)
    return torch.clamp(center - half_width, 0.0, 1.0), torch.clamp(center + half_width, 0.0, 1.0)


def make_ard_trust_region_bounds_unit(
    region: TurboV2RegionState,
    lengthscales: Optional[torch.Tensor],
    dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    center = region.center.to(dtype=dtype, device=device).reshape(-1)
    if lengthscales is None or lengthscales.numel() != dim:
        return get_region_uniform_bounds_unit(region, dim=dim, dtype=dtype, device=device)

    lambdas = torch.clamp(lengthscales.to(dtype=dtype, device=device).reshape(-1), min=1e-8)
    if not bool(torch.isfinite(lambdas).all().detach().cpu().item()):
        return get_region_uniform_bounds_unit(region, dim=dim, dtype=dtype, device=device)

    geom_mean = torch.exp(torch.mean(torch.log(lambdas)))
    side_lengths = torch.clamp(lambdas * region.length / torch.clamp(geom_mean, min=1e-8), min=1e-8)
    tr_lb = torch.clamp(center - 0.5 * side_lengths, 0.0, 1.0)
    tr_ub = torch.clamp(center + 0.5 * side_lengths, 0.0, 1.0)
    return tr_lb, tr_ub


def select_local_indices_for_region(
    X_unit: torch.Tensor,
    region: TurboV2RegionState,
    bounds_unit: Tuple[torch.Tensor, torch.Tensor],
    max_points: int,
) -> torch.Tensor:
    tr_lb, tr_ub = bounds_unit
    center = region.center.to(dtype=X_unit.dtype, device=X_unit.device).reshape(1, -1)
    inside = torch.all((X_unit >= tr_lb.reshape(1, -1)) & (X_unit <= tr_ub.reshape(1, -1)), dim=1)
    inside_idx = torch.nonzero(inside, as_tuple=False).reshape(-1)

    k = min(max_points, X_unit.shape[0])
    if inside_idx.numel() >= k:
        dist2_inside = torch.sum((X_unit[inside_idx] - center) ** 2, dim=1)
        return inside_idx[torch.topk(dist2_inside, k=k, largest=False).indices]

    dist2 = torch.sum((X_unit - center) ** 2, dim=1)
    nearest_idx = torch.topk(dist2, k=k, largest=False).indices
    if inside_idx.numel() == 0:
        return nearest_idx
    return torch.unique(torch.cat([inside_idx, nearest_idx], dim=0))[:k]


def extract_ard_lengthscales(gp: SingleTaskGP, dim: int) -> Optional[torch.Tensor]:
    try:
        lengthscale = gp.covar_module.base_kernel.lengthscale.detach().reshape(-1)
    except Exception:
        return None
    if lengthscale.numel() != dim:
        return None
    if not bool(torch.isfinite(lengthscale).all().detach().cpu().item()):
        return None
    return lengthscale


def fit_local_region_gp(
    X_unit: torch.Tensor,
    y: torch.Tensor,
    local_indices: torch.Tensor,
    config: MethodConfig,
) -> SingleTaskGP:
    train_x = X_unit[local_indices]
    train_y = y[local_indices]
    return fit_gp_with_config(train_x=train_x, train_y=train_y, config=config)


def initialize_turbo_v2_regions(
    X_unit: torch.Tensor,
    y: torch.Tensor,
    num_regions: int,
    config: MethodConfig,
    seed: int,
) -> List[TurboV2RegionState]:
    dim = X_unit.shape[1]
    dtype = X_unit.dtype
    device = X_unit.device
    y_flat = y.reshape(-1)
    sorted_idx = torch.argsort(y_flat)
    regions: List[TurboV2RegionState] = []

    for idx in sorted_idx[: min(num_regions, X_unit.shape[0])]:
        center = X_unit[idx].detach().clone()
        best_y = float(y_flat[idx].detach().cpu().item())
        regions.append(
            TurboV2RegionState(
                center=center,
                length=config.turbo_length_init,
                best_x=center.detach().clone(),
                best_y=best_y,
            )
        )

    if len(regions) < num_regions:
        extra = generate_global_sobol_points(
            dim=dim,
            n=num_regions - len(regions),
            seed=seed + 1009,
            dtype=dtype,
            device=device,
        )
        for point in extra:
            nearest = torch.argmin(torch.sum((X_unit - point.reshape(1, -1)) ** 2, dim=1))
            regions.append(
                TurboV2RegionState(
                    center=point.detach().clone(),
                    length=config.turbo_length_init,
                    best_x=X_unit[nearest].detach().clone(),
                    best_y=float(y_flat[nearest].detach().cpu().item()),
                )
            )
    return regions


def restart_turbo_v2_region(
    region: TurboV2RegionState,
    X_unit: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    config: MethodConfig,
) -> None:
    dim = X_unit.shape[1]
    fresh = generate_global_sobol_points(
        dim=dim,
        n=max(1, config.turbo_init_points_per_region),
        seed=seed,
        dtype=X_unit.dtype,
        device=X_unit.device,
    )
    if X_unit.shape[0] > 0:
        dist_to_history = torch.cdist(fresh, X_unit).min(dim=1).values
        center = fresh[int(torch.argmax(dist_to_history).detach().cpu().item())]
        nearest = torch.argmin(torch.sum((X_unit - center.reshape(1, -1)) ** 2, dim=1))
        best_x = X_unit[nearest].detach().clone()
        best_y = float(y[nearest].detach().cpu().item())
    else:
        center = fresh[0]
        best_x = None
        best_y = float("inf")

    region.center = center.detach().clone()
    region.length = config.turbo_length_init
    region.success_counter = 0
    region.failure_counter = 0
    region.restart_count += 1
    region.best_x = best_x
    region.best_y = best_y
    region.last_bounds = None
    region.last_lengthscales = None


def update_turbo_v2_region_after_eval(
    region: TurboV2RegionState,
    x_unit: torch.Tensor,
    y_value: float,
    dim: int,
    seed: int,
    X_unit: torch.Tensor,
    y: torch.Tensor,
    config: MethodConfig,
) -> None:
    failure_tolerance = config.turbo_failure_tolerance
    if failure_tolerance is None:
        failure_tolerance = max(1, dim)

    improved = y_value < region.best_y - config.stagnation_min_improve
    if improved:
        region.best_y = y_value
        region.best_x = x_unit.detach().clone()
        region.center = x_unit.detach().clone()
        region.success_counter += 1
        region.failure_counter = 0
    else:
        region.success_counter = 0
        region.failure_counter += 1

    if region.success_counter >= config.turbo_success_tolerance:
        region.length = min(config.turbo_length_max, 2.0 * region.length)
        region.success_counter = 0
        region.failure_counter = 0
    elif region.failure_counter >= failure_tolerance:
        region.length *= 0.5
        region.success_counter = 0
        region.failure_counter = 0

    if region.length < config.turbo_length_min:
        restart_turbo_v2_region(region, X_unit=X_unit, y=y, seed=seed, config=config)


def update_region_after_eval_for_evaluated_restart(
    region: TurboV2RegionState,
    x_unit: torch.Tensor,
    y_value: float,
    dim: int,
    config: MethodConfig,
) -> bool:
    failure_tolerance = config.turbo_failure_tolerance
    if failure_tolerance is None:
        failure_tolerance = max(1, dim)

    improved = y_value < region.best_y - config.stagnation_min_improve
    if improved:
        region.best_y = y_value
        region.best_x = x_unit.detach().clone()
        region.center = x_unit.detach().clone()
        region.success_counter += 1
        region.failure_counter = 0
    else:
        region.success_counter = 0
        region.failure_counter += 1

    if region.success_counter >= config.turbo_success_tolerance:
        region.length = min(config.turbo_length_max, 2.0 * region.length)
        region.success_counter = 0
        region.failure_counter = 0
    elif region.failure_counter >= failure_tolerance:
        region.length *= 0.5
        region.success_counter = 0
        region.failure_counter = 0

    return region.length < config.turbo_length_min


def evaluate_restart_batch_and_reset_region(
    region: TurboV2RegionState,
    problem,
    eval_logger: EvaluationDatLogger,
    X_cpu: torch.Tensor,
    y_cpu: torch.Tensor,
    lb_dev: torch.Tensor,
    ub_dev: torch.Tensor,
    seed: int,
    n_restart: int,
    config: MethodConfig,
) -> Tuple[torch.Tensor, torch.Tensor, List[float]]:
    if n_restart <= 0:
        return X_cpu, y_cpu, []

    device = lb_dev.device
    dim = lb_dev.numel()
    restart_unit = generate_global_sobol_points(
        dim=dim,
        n=n_restart,
        seed=seed,
        dtype=torch.double,
        device=device,
    )
    restart_x_cpu = from_unit(restart_unit, lb_dev, ub_dev).detach().cpu()

    restart_values: List[float] = []
    for row in restart_x_cpu:
        y_value = evaluate_point(
            problem=problem,
            x_np=row.detach().cpu().numpy().reshape(-1),
            eval_logger=eval_logger,
        )
        restart_values.append(float(y_value))
        X_cpu = torch.cat((X_cpu, row.reshape(1, -1).to(dtype=X_cpu.dtype)), dim=0)
        y_cpu = torch.cat((y_cpu, torch.tensor([[y_value]], dtype=y_cpu.dtype)), dim=0)

    best_idx = int(np.argmin(np.asarray(restart_values, dtype=float)))
    best_unit = restart_unit[best_idx].detach().clone()
    region.center = best_unit
    region.length = config.turbo_length_init
    region.success_counter = 0
    region.failure_counter = 0
    region.restart_count += 1
    region.best_x = best_unit.detach().clone()
    region.best_y = float(restart_values[best_idx])
    region.last_bounds = None
    region.last_lengthscales = None
    return X_cpu, y_cpu, restart_values


def generate_region_candidate_set_unit(
    region: TurboV2RegionState,
    bounds_unit: Tuple[torch.Tensor, torch.Tensor],
    n_candidates: int,
    seed: int,
) -> torch.Tensor:
    tr_lb, tr_ub = bounds_unit
    dim = tr_lb.numel()
    engine = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=seed)
    sobol = engine.draw(n_candidates).to(dtype=tr_lb.dtype, device=tr_lb.device)
    perturbed = tr_lb.reshape(1, -1) + sobol * (tr_ub - tr_lb).reshape(1, -1)
    center = region.center.to(dtype=tr_lb.dtype, device=tr_lb.device).reshape(1, -1)
    candidates = center.expand(n_candidates, dim).clone()

    perturb_prob = min(1.0, 20.0 / float(dim))
    mask = torch.rand(n_candidates, dim, dtype=tr_lb.dtype, device=tr_lb.device) < perturb_prob
    empty_rows = ~mask.any(dim=1)
    if bool(empty_rows.any().detach().cpu().item()):
        row_idx = torch.nonzero(empty_rows, as_tuple=False).reshape(-1)
        cols = torch.randint(0, dim, (row_idx.numel(),), device=tr_lb.device)
        mask[row_idx, cols] = True
    candidates[mask] = perturbed[mask]
    return torch.clamp(candidates, min=tr_lb.reshape(1, -1), max=tr_ub.reshape(1, -1))


def diagonal_thompson_values(gp: SingleTaskGP, candidates: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        posterior = gp.posterior(candidates)
        mean = posterior.mean.squeeze(-1)
        std = torch.sqrt(torch.clamp(posterior.variance.squeeze(-1), min=1e-18))
        return mean + std * torch.randn_like(mean)


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
    orth_noise_scale: float = 1.0,
    orthogonal_refinement: str = "mixed_gp",
    clamp_diagnostics: Optional[ClampDiagnostics] = None,
) -> Tuple[torch.Tensor, str]:
    manifold_violation = bounds_violation_norm(x_manifold, lb, ub)
    if clamp_diagnostics is not None:
        clamp_diagnostics.manifold_pre_clamp_violation = manifold_violation
    x_manifold = torch.clamp(x_manifold, min=lb, max=ub)
    if not activate_orthogonal:
        if clamp_diagnostics is not None:
            clamp_diagnostics.final_region_clamped = manifold_violation > 1e-10
        return x_manifold, "manifold"

    dim = x_manifold.shape[-1]
    rank = P_r.shape[0]
    if rank >= dim:
        if clamp_diagnostics is not None:
            clamp_diagnostics.final_region_clamped = manifold_violation > 1e-10
        return x_manifold, "manifold"

    U_perp = orthogonal_basis_from_projection(P_r)
    if U_perp.shape[1] == 0:
        if clamp_diagnostics is not None:
            clamp_diagnostics.final_region_clamped = manifold_violation > 1e-10
        return x_manifold, "manifold"
    if orthogonal_refinement not in {"simple", "mixed_gp"}:
        raise ValueError(f"Unknown orthogonal refinement mode: {orthogonal_refinement}")

    generator = torch.Generator(device=x_manifold.device)
    generator.manual_seed(seed)
    sigma_t = max(config.orth_sigma_min, min(config.orth_sigma_max, state.sigma_scale))
    sigma_t = sigma_t * float(torch.mean(ub - lb).detach().cpu().item())
    sigma_t = sigma_t * max(0.0, float(orth_noise_scale))

    n_orth_samples = 1 if orthogonal_refinement == "simple" else max(1, state.orth_k)
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
        if clamp_diagnostics is not None:
            clamp_diagnostics.final_region_clamped = manifold_violation > 1e-10
        return x_manifold, "manifold"

    x_orth_raw = x_manifold + noise_perp[nonzero_mask]
    orth_violation_values = [
        bounds_violation_norm(row.reshape(1, -1), lb, ub)
        for row in x_orth_raw
    ]
    if orthogonal_refinement == "simple":
        orth_violation = orth_violation_values[0]
        if clamp_diagnostics is not None:
            clamp_diagnostics.orthogonal_pre_clamp_violation = orth_violation
            clamp_diagnostics.final_region_clamped = orth_violation > 1e-10
        x_orth = torch.clamp(x_orth_raw[:1], min=lb, max=ub)
        return x_orth, "orthogonal"
    x_orth = torch.clamp(x_orth_raw, min=lb, max=ub)
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
    if clamp_diagnostics is not None:
        if best_idx == 0:
            clamp_diagnostics.final_region_clamped = manifold_violation > 1e-10
        else:
            selected_orth_violation = orth_violation_values[best_idx - 1]
            clamp_diagnostics.orthogonal_pre_clamp_violation = selected_orth_violation
            clamp_diagnostics.final_region_clamped = selected_orth_violation > 1e-10
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
    method_name: str,
    use_turbo: bool,
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
    device = resolve_torch_device(config.device)
    problem = create_problem(fid=fid, dim=dim)
    contour_cache = build_contour_cache(fid=fid, dim=dim, grid_size=grid_size) if save_plots else None

    with EvaluationDatLogger(dat_path=dat_path) as eval_logger:
        lb = torch.tensor(problem.bounds.lb, dtype=torch.double, device=device)
        ub = torch.tensor(problem.bounds.ub, dtype=torch.double, device=device)

        init_x_local = initial_x.clone().to(dtype=torch.double, device=device)
        init_y = evaluate_initial_design(problem=problem, init_x=init_x_local, eval_logger=eval_logger).to(device=device)
        state = DynamicOrthogonalState(
            sigma_scale=config.orth_sigma_init,
            orth_k=config.orth_k_init,
            best_history=[float(init_y.min().detach().cpu().item())],
        )
        tr_state = TrustRegionState(length=config.turbo_length_init) if use_turbo else None

        while problem.state.evaluations < budget:
            decrement_cooldown(state)
            if use_turbo:
                tr_center, _ = get_incumbent(init_x_local, init_y)
                tr_lb, tr_ub = make_trust_region_bounds(
                    center=tr_center,
                    lb=lb,
                    ub=ub,
                    tr_state=tr_state,
                )
            else:
                tr_center = 0.5 * (lb + ub)
                tr_lb, tr_ub = lb, ub

            x_mean, x_weighted_mean, P_r, eigvals, w = compute_pca_lowrank(init_x_local, init_y, alpha=0.95)
            z_r = ((init_x_local - x_mean) - x_weighted_mean) @ P_r.T
            z_norm_stats = make_normalization_stats(z_r)
            z_r_norm = normalize_values(z_r, z_norm_stats)
            mapper = lambda z: z @ P_r + x_mean + x_weighted_mean
            mapper_norm = lambda z_norm: mapper(denormalize_values(z_norm, z_norm_stats))

            if use_turbo:
                bounds_z = make_latent_trust_region_bounds(
                    tr_center=tr_center,
                    tr_lb=tr_lb,
                    tr_ub=tr_ub,
                    x_mean=x_mean,
                    x_weighted_mean=x_weighted_mean,
                    P_r=P_r,
                )
            else:
                x_center = 0.5 * (lb + ub)
                z_center = ((x_center - x_mean) - x_weighted_mean) @ P_r.T
                rho = 0.5 * torch.min(ub - lb)
                bounds_z = torch.stack([z_center - rho, z_center + rho], dim=0)
            bounds_z_norm = normalize_values(bounds_z, z_norm_stats)

            gp = fit_gp_with_config(train_x=z_r_norm, train_y=init_y, config=config)

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

            new_z_norm = select_sobol_lcb_candidate(
                gp=gp,
                bounds=bounds_z_norm,
                seed=seed + 1543 * (problem.state.evaluations + 1),
                config=config,
                mapper=mapper_norm,
                x_bounds=(tr_lb, tr_ub),
                beta=beta_t,
            )
            new_x = mapper_norm(new_z_norm)
            x_eval, selected_mode = select_dynamic_candidate_v2(
                x_manifold=new_x,
                X_hist=init_x_local,
                init_y=init_y,
                P_r=P_r,
                x_mean=x_mean,
                x_weighted_mean=x_weighted_mean,
                lb=tr_lb,
                ub=tr_ub,
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
                    out_dir=str(plot_root / method_name / "weighted"),
                )

            previous_best = state.best_history[-1]
            current_y = evaluate_point(
                problem=problem,
                x_np=x_eval.detach().cpu().numpy().reshape(-1),
                eval_logger=eval_logger,
            )
            new_y = torch.tensor([[current_y]], dtype=init_y.dtype, device=device)
            current_best = min(previous_best, current_y)
            state.last_selected_mode = selected_mode
            state.last_eval_improved = current_best < previous_best - config.stagnation_min_improve
            if use_turbo and tr_state is not None:
                update_trust_region_state(
                    tr_state=tr_state,
                    improved=state.last_eval_improved,
                    dim=dim,
                    config=config,
                )
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
                    out_dir=str(plot_root / method_name / "iterations"),
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
    config: MethodConfig,
) -> None:
    set_seed(seed)
    device = resolve_torch_device(config.device)
    problem = create_problem(fid=fid, dim=dim)
    contour_cache = build_contour_cache(fid=fid, dim=dim, grid_size=grid_size) if save_plots else None

    with EvaluationDatLogger(dat_path=dat_path) as eval_logger:
        lb = torch.tensor(problem.bounds.lb, dtype=torch.double, device=device)
        ub = torch.tensor(problem.bounds.ub, dtype=torch.double, device=device)

        init_x_local = initial_x.clone().to(dtype=torch.double, device=device)
        init_y = evaluate_initial_design(problem=problem, init_x=init_x_local, eval_logger=eval_logger).to(device=device)

        while problem.state.evaluations < budget:
            x_mean, x_weighted_mean, P_r, eigvals, w = compute_pca_lowrank(init_x_local, init_y, alpha=0.95)
            z_r = ((init_x_local - x_mean) - x_weighted_mean) @ P_r.T
            mapper = lambda z: z @ P_r + x_mean + x_weighted_mean

            x_center = 0.5 * (lb + ub)
            z_center = ((x_center - x_mean) - x_weighted_mean) @ P_r.T
            rho = 0.5 * torch.min(ub - lb)
            bounds_z = torch.stack([z_center - rho, z_center + rho], dim=0)

            gp = fit_gp_with_config(train_x=z_r, train_y=init_y, config=config)

            new_z = select_sobol_lcb_candidate(
                gp=gp,
                bounds=bounds_z,
                seed=seed + 1543 * (problem.state.evaluations + 1),
                config=config,
                mapper=mapper,
                x_bounds=(lb, ub),
                beta=2.0,
            )
            new_x = torch.clamp(mapper(new_z), min=lb, max=ub)

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
            new_y = torch.tensor([[current_y]], dtype=init_y.dtype, device=device)

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


def run_pca_bo_turbo_v2(
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
    device = resolve_torch_device(config.device)
    problem = create_problem(fid=fid, dim=dim)
    contour_cache = build_contour_cache(fid=fid, dim=dim, grid_size=grid_size) if save_plots else None

    with EvaluationDatLogger(dat_path=dat_path) as eval_logger:
        lb_cpu = torch.tensor(problem.bounds.lb, dtype=torch.double)
        ub_cpu = torch.tensor(problem.bounds.ub, dtype=torch.double)
        lb_dev = lb_cpu.to(device=device)
        ub_dev = ub_cpu.to(device=device)

        X_cpu = initial_x.clone().to(dtype=torch.double)
        y_cpu = evaluate_initial_design(problem=problem, init_x=X_cpu, eval_logger=eval_logger)
        X_unit_dev = to_unit(X_cpu.to(device=device), lb_dev, ub_dev)
        y_dev = y_cpu.to(device=device)
        regions = initialize_turbo_v2_regions(
            X_unit=X_unit_dev,
            y=y_dev,
            num_regions=max(1, config.turbo_num_regions),
            config=config,
            seed=seed,
        )
        state = DynamicOrthogonalState(
            sigma_scale=config.orth_sigma_init,
            orth_k=config.orth_k_init,
            best_history=[float(y_cpu.min().detach().cpu().item())],
        )

        while problem.state.evaluations < budget:
            decrement_cooldown(state)
            X_unit_dev = to_unit(X_cpu.to(device=device), lb_dev, ub_dev)
            y_dev = y_cpu.to(device=device)

            region_payloads = []
            for region_idx, region in enumerate(regions):
                uniform_bounds = get_region_uniform_bounds_unit(
                    region, dim=dim, dtype=torch.double, device=device
                )
                local_indices = select_local_indices_for_region(
                    X_unit=X_unit_dev,
                    region=region,
                    bounds_unit=uniform_bounds,
                    max_points=min(config.local_subset_size, X_unit_dev.shape[0]),
                )
                try:
                    gp = fit_local_region_gp(X_unit_dev, y_dev, local_indices, config=config)
                except Exception:
                    continue

                lengthscales = extract_ard_lengthscales(gp, dim=dim)
                ard_bounds = make_ard_trust_region_bounds_unit(
                    region=region,
                    lengthscales=lengthscales,
                    dim=dim,
                    dtype=torch.double,
                    device=device,
                )
                region.last_bounds = (ard_bounds[0].detach().clone(), ard_bounds[1].detach().clone())
                region.last_lengthscales = None if lengthscales is None else lengthscales.detach().clone()

                n_candidates = min(config.turbo_candidate_multiplier * dim, config.turbo_candidate_max)
                cand_unit = generate_region_candidate_set_unit(
                    region=region,
                    bounds_unit=ard_bounds,
                    n_candidates=max(1, n_candidates),
                    seed=seed + 7919 * (problem.state.evaluations + 1) + region_idx,
                )
                samples = diagonal_thompson_values(gp, cand_unit)
                best_idx = int(torch.argmin(samples).detach().cpu().item())
                best_sample = float(samples[best_idx].detach().cpu().item())
                region_payloads.append(
                    {
                        "region_idx": region_idx,
                        "region": region,
                        "gp": gp,
                        "local_indices": local_indices,
                        "bounds_unit": ard_bounds,
                        "candidate_unit": cand_unit[best_idx : best_idx + 1],
                        "sample": best_sample,
                    }
                )

            if not region_payloads:
                fallback_unit = generate_global_sobol_points(
                    dim=dim,
                    n=1,
                    seed=seed + 65537 * (problem.state.evaluations + 1),
                    dtype=torch.double,
                    device=device,
                )
                fallback_x_cpu = from_unit(fallback_unit, lb_dev, ub_dev).detach().cpu()
                current_y = evaluate_point(
                    problem=problem,
                    x_np=fallback_x_cpu.detach().cpu().numpy().reshape(-1),
                    eval_logger=eval_logger,
                )
                X_cpu = torch.cat((X_cpu, fallback_x_cpu), dim=0)
                y_cpu = torch.cat((y_cpu, torch.tensor([[current_y]], dtype=y_cpu.dtype)), dim=0)
                current_best = min(state.best_history[-1], current_y)
                state.last_selected_mode = "sobol_fallback"
                state.last_eval_improved = current_best < state.best_history[-1] - config.stagnation_min_improve
                state.best_history.append(current_best)
                restart_turbo_v2_region(
                    regions[0],
                    X_unit=to_unit(X_cpu.to(device=device), lb_dev, ub_dev),
                    y=y_cpu.to(device=device),
                    seed=seed + problem.state.evaluations,
                    config=config,
                )
                continue

            selected = min(region_payloads, key=lambda payload: payload["sample"])
            region = selected["region"]
            region_idx = int(selected["region_idx"])
            candidate_unit = selected["candidate_unit"]
            candidate_x_dev = from_unit(candidate_unit, lb_dev, ub_dev)
            tr_lb_unit, tr_ub_unit = selected["bounds_unit"]
            tr_lb_dev = from_unit(tr_lb_unit, lb_dev, ub_dev)
            tr_ub_dev = from_unit(tr_ub_unit, lb_dev, ub_dev)
            region_center_plot = from_unit(
                region.center.to(dtype=torch.double, device=device).reshape(1, -1), lb_dev, ub_dev
            ).detach().cpu().reshape(-1)
            region_length_plot = float(region.length)
            region_success_plot = int(region.success_counter)
            region_failure_plot = int(region.failure_counter)
            region_restarts_plot = int(region.restart_count)

            local_idx_cpu = selected["local_indices"].detach().cpu()
            local_x_dev = X_cpu[local_idx_cpu].to(device=device)
            local_y_dev = y_cpu[local_idx_cpu].to(device=device)
            if local_x_dev.shape[0] >= 2:
                x_mean, x_weighted_mean, P_r, eigvals, w = compute_pca_lowrank(local_x_dev, local_y_dev, alpha=0.95)
                activate_orthogonal = P_r.shape[0] < dim and state.cooldown_remaining == 0
                x_eval_dev, selected_mode = select_dynamic_candidate_v2(
                    x_manifold=candidate_x_dev,
                    X_hist=local_x_dev,
                    init_y=local_y_dev,
                    P_r=P_r,
                    x_mean=x_mean,
                    x_weighted_mean=x_weighted_mean,
                    lb=tr_lb_dev,
                    ub=tr_ub_dev,
                        activate_orthogonal=activate_orthogonal,
                        state=state,
                        seed=seed + problem.state.evaluations,
                        config=config,
                    )
            else:
                x_mean = torch.zeros(dim, dtype=torch.double, device=device)
                x_weighted_mean = torch.zeros(dim, dtype=torch.double, device=device)
                P_r = torch.eye(dim, dtype=torch.double, device=device)[:1]
                eigvals = torch.ones(1, dtype=torch.double, device=device)
                w = torch.ones(local_x_dev.shape[0], dtype=torch.double, device=device)
                selected_mode = "thompson"
                x_eval_dev = torch.clamp(candidate_x_dev, min=tr_lb_dev, max=tr_ub_dev)

            x_eval_cpu = torch.clamp(x_eval_dev.detach().cpu(), min=lb_cpu, max=ub_cpu)

            if save_plots and dim == 2 and local_x_dev.shape[0] >= 1:
                plot_weighted_points_iteration(
                    contour_cache=contour_cache,
                    X_hist=X_cpu,
                    w=torch.ones(X_cpu.shape[0], dtype=torch.double) / max(1, X_cpu.shape[0]),
                    x_mean=X_cpu.mean(dim=0),
                    iter_idx=X_cpu.shape[0] - n0,
                    func_id=fid,
                    dim=dim,
                    out_dir=str(plot_root / METHOD_TURBO_V2 / "weighted"),
                )

            previous_best = state.best_history[-1]
            current_y = evaluate_point(
                problem=problem,
                x_np=x_eval_cpu.detach().cpu().numpy().reshape(-1),
                eval_logger=eval_logger,
            )
            new_y_cpu = torch.tensor([[current_y]], dtype=y_cpu.dtype)
            X_cpu = torch.cat((X_cpu, x_eval_cpu), dim=0)
            y_cpu = torch.cat((y_cpu, new_y_cpu), dim=0)

            x_eval_unit_dev = to_unit(x_eval_cpu.to(device=device), lb_dev, ub_dev).reshape(-1)
            update_turbo_v2_region_after_eval(
                region=region,
                x_unit=x_eval_unit_dev,
                y_value=current_y,
                dim=dim,
                seed=seed + 104729 * (problem.state.evaluations + 1) + region_idx,
                X_unit=to_unit(X_cpu.to(device=device), lb_dev, ub_dev),
                y=y_cpu.to(device=device),
                config=config,
            )

            current_best = min(previous_best, current_y)
            state.last_selected_mode = selected_mode
            state.last_eval_improved = current_best < previous_best - config.stagnation_min_improve
            state.best_history.append(current_best)
            if selected_mode == "orthogonal":
                state.cooldown_remaining = config.orthogonal_cooldown

            if save_plots and dim == 2:
                plot_turbo_v2_iteration(
                    contour_cache=contour_cache,
                    X_hist=X_cpu,
                    local_x=local_x_dev.detach().cpu(),
                    x_mean=x_mean.detach().cpu(),
                    x_weighted_mean=x_weighted_mean.detach().cpu(),
                    P_r=P_r.detach().cpu(),
                    eigvals=eigvals.detach().cpu(),
                    region_center=region_center_plot,
                    tr_lb=tr_lb_dev.detach().cpu().reshape(-1),
                    tr_ub=tr_ub_dev.detach().cpu().reshape(-1),
                    thompson_candidate=candidate_x_dev.detach().cpu().reshape(-1),
                    final_candidate=x_eval_cpu.detach().cpu().reshape(-1),
                    iter_idx=X_cpu.shape[0] - n0,
                    func_id=fid,
                    dim=dim,
                    selected_mode=selected_mode,
                    region_length=region_length_plot,
                    success_counter=region_success_plot,
                    failure_counter=region_failure_plot,
                    restart_count=region_restarts_plot,
                    out_dir=plot_root / METHOD_TURBO_V2 / "iterations",
                )


def run_pca_bo_turbo_v5(
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
    config = replace(
        config,
        turbo_num_regions=1,
        turbo_v3_ard_shape=False,
        turbo_v3_orth_scale=TURBO_V3_ORTH_SCALE,
        turbo_v5_latent_radius_scale=config.turbo_v5_latent_radius_scale,
        turbo_restart_mode=TURBO_RESTART_MODE,
        orthogonal_refinement=ORTHOGONAL_REFINEMENT,
    )
    # V5 uses scipy differential evolution for the PCA-BO latent acquisition loop.
    # Keep that path on CPU to avoid thousands of tiny CUDA acquisition calls.
    device = torch.device("cpu")
    problem = create_problem(fid=fid, dim=dim)
    contour_cache = build_contour_cache(fid=fid, dim=dim, grid_size=grid_size) if save_plots else None

    with EvaluationDatLogger(dat_path=dat_path) as eval_logger:
        lb_cpu = torch.tensor(problem.bounds.lb, dtype=torch.double)
        ub_cpu = torch.tensor(problem.bounds.ub, dtype=torch.double)
        lb_dev = lb_cpu.to(device=device)
        ub_dev = ub_cpu.to(device=device)
        clamp_rows: List[Dict[str, object]] = []
        orthogonal_stats_rows: List[Dict[str, object]] = []

        X_cpu = initial_x.clone().to(dtype=torch.double)
        y_cpu = evaluate_initial_design(problem=problem, init_x=X_cpu, eval_logger=eval_logger)
        X_unit_dev = to_unit(X_cpu.to(device=device), lb_dev, ub_dev)
        y_dev = y_cpu.to(device=device)
        region = initialize_turbo_v2_regions(
            X_unit=X_unit_dev,
            y=y_dev,
            num_regions=1,
            config=config,
            seed=seed,
        )[0]
        state = DynamicOrthogonalState(
            sigma_scale=config.orth_sigma_init,
            orth_k=config.orth_k_init,
            best_history=[float(y_cpu.min().detach().cpu().item())],
        )

        while problem.state.evaluations < budget:
            decrement_cooldown(state)
            X_unit_dev = to_unit(X_cpu.to(device=device), lb_dev, ub_dev)
            y_dev = y_cpu.to(device=device)
            clamp_diagnostics = ClampDiagnostics()
            region_length_for_row = float(region.length)
            latent_ubr = float("nan")
            ubr_trigger = False
            stagnation_trigger = False
            activate_orthogonal = False
            orth_candidate_count = 0
            consecutive_trigger_count_for_row = int(state.consecutive_trigger_count)

            uniform_bounds = get_region_uniform_bounds_unit(
                region, dim=dim, dtype=torch.double, device=device
            )
            local_indices = select_local_indices_for_region(
                X_unit=X_unit_dev,
                region=region,
                bounds_unit=uniform_bounds,
                max_points=min(config.local_subset_size, X_unit_dev.shape[0]),
            )

            region_bounds = uniform_bounds
            if config.turbo_v3_ard_shape:
                try:
                    shape_gp = fit_local_region_gp(X_unit_dev, y_dev, local_indices, config=config)
                    lengthscales = extract_ard_lengthscales(shape_gp, dim=dim)
                    region_bounds = make_ard_trust_region_bounds_unit(
                        region=region,
                        lengthscales=lengthscales,
                        dim=dim,
                        dtype=torch.double,
                        device=device,
                    )
                    region.last_lengthscales = None if lengthscales is None else lengthscales.detach().clone()
                except Exception:
                    region_bounds = uniform_bounds
                    region.last_lengthscales = None

            region.last_bounds = (region_bounds[0].detach().clone(), region_bounds[1].detach().clone())
            tr_lb_unit, tr_ub_unit = region_bounds
            tr_lb_dev = from_unit(tr_lb_unit, lb_dev, ub_dev).reshape(-1)
            tr_ub_dev = from_unit(tr_ub_unit, lb_dev, ub_dev).reshape(-1)
            tr_center_dev = from_unit(
                region.center.to(dtype=torch.double, device=device).reshape(1, -1), lb_dev, ub_dev
            ).reshape(-1)

            local_idx_cpu = local_indices.detach().cpu()
            local_x_dev = X_cpu[local_idx_cpu].to(dtype=torch.double, device=device)
            local_y_dev = y_cpu[local_idx_cpu].to(dtype=torch.double, device=device)

            if local_x_dev.shape[0] < 2:
                x_center_raw = tr_center_dev.reshape(1, -1)
                clamp_diagnostics.manifold_pre_clamp_violation = bounds_violation_norm(
                    x_center_raw, tr_lb_dev, tr_ub_dev
                )
                x_eval_dev = torch.clamp(x_center_raw, min=tr_lb_dev, max=tr_ub_dev)
                clamp_diagnostics.final_region_clamped = (
                    clamp_diagnostics.manifold_pre_clamp_violation > 1e-10
                )
                selected_mode = "center_fallback"
                x_mean = torch.zeros(dim, dtype=torch.double, device=device)
                x_weighted_mean = torch.zeros(dim, dtype=torch.double, device=device)
                P_r = torch.eye(dim, dtype=torch.double, device=device)[:1]
                eigvals = torch.ones(1, dtype=torch.double, device=device)
                w = torch.ones(local_x_dev.shape[0], dtype=torch.double, device=device)
            else:
                x_mean, x_weighted_mean, P_r, eigvals, w = compute_pca_lowrank(
                    local_x_dev, local_y_dev, alpha=0.95
                )
                z_r = ((local_x_dev - x_mean) - x_weighted_mean) @ P_r.T
                z_norm_stats = make_normalization_stats(z_r)
                z_r_norm = normalize_values(z_r, z_norm_stats)
                mapper = lambda z: z @ P_r + x_mean + x_weighted_mean
                mapper_norm = lambda z_norm: mapper(denormalize_values(z_norm, z_norm_stats))

                bounds_z = make_latent_region_radius_bounds(
                    tr_center=tr_center_dev,
                    tr_lb=tr_lb_dev,
                    tr_ub=tr_ub_dev,
                    x_mean=x_mean,
                    x_weighted_mean=x_weighted_mean,
                    P_r=P_r,
                    radius_scale=config.turbo_v5_latent_radius_scale,
                )
                bounds_z_norm = normalize_values(bounds_z, z_norm_stats)

                gp = fit_gp_with_config(train_x=z_r_norm, train_y=local_y_dev, config=config)
                best_f = local_y_dev.min()
                acquisition = PEI(
                    gp=gp,
                    best_f=best_f,
                    bounds=(tr_lb_dev, tr_ub_dev),
                    penalty_weight=100.0,
                    mapper=mapper_norm,
                )

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
                consecutive_trigger_count_for_row = int(state.consecutive_trigger_count)
                activate_orthogonal = should_activate_orthogonal(
                    state, rank=P_r.shape[0], dim=dim, config=config
                )
                orth_candidate_count = int(activate_orthogonal and P_r.shape[0] < dim)
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
                    z_t = torch.tensor(
                        z_np,
                        dtype=local_x_dev.dtype,
                        device=device,
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
                    seed=seed + problem.state.evaluations,
                )

                new_z_norm = torch.tensor(result.x, dtype=local_x_dev.dtype, device=device).view(1, -1)
                new_x = mapper_norm(new_z_norm)
                x_eval_dev, selected_mode = select_dynamic_candidate_v2(
                    x_manifold=new_x,
                    X_hist=local_x_dev,
                    init_y=local_y_dev,
                    P_r=P_r,
                    x_mean=x_mean,
                    x_weighted_mean=x_weighted_mean,
                    lb=tr_lb_dev,
                    ub=tr_ub_dev,
                    activate_orthogonal=activate_orthogonal,
                    state=state,
                    seed=seed + problem.state.evaluations,
                    config=config,
                    orth_noise_scale=config.turbo_v3_orth_scale,
                    orthogonal_refinement=config.orthogonal_refinement,
                    clamp_diagnostics=clamp_diagnostics,
                )

            pca_rank_for_row = int(P_r.shape[0])
            x_eval_pre_global_cpu = x_eval_dev.detach().cpu()
            x_eval_cpu = torch.clamp(x_eval_pre_global_cpu, min=lb_cpu, max=ub_cpu)
            final_global_clamped = bool(
                torch.max(torch.abs(x_eval_cpu - x_eval_pre_global_cpu)).detach().cpu().item() > 1e-10
            )

            if save_plots and dim == 2 and local_x_dev.shape[0] >= 1:
                plot_weighted_points_iteration(
                    contour_cache=contour_cache,
                    X_hist=local_x_dev.detach().cpu(),
                    w=w.detach().cpu(),
                    x_mean=x_mean.detach().cpu(),
                    iter_idx=X_cpu.shape[0] - n0,
                    func_id=fid,
                    dim=dim,
                    out_dir=str(plot_root / METHOD_TURBO_V5 / "weighted"),
                )

            previous_best = state.best_history[-1]
            current_y = evaluate_point(
                problem=problem,
                x_np=x_eval_cpu.detach().cpu().numpy().reshape(-1),
                eval_logger=eval_logger,
            )
            clamp_row = {
                "method": METHOD_TURBO_V5,
                "function_id": fid,
                "dim": dim,
                "seed": seed,
                "run_idx": run_idx,
                "iteration": int(problem.state.evaluations),
                "mode": selected_mode,
                "rank": pca_rank_for_row,
                "region_length": f"{region_length_for_row:.16f}",
                "success_counter_before": int(region.success_counter),
                "failure_counter_before": int(region.failure_counter),
                "restart_count_before": int(region.restart_count),
                "manifold_pre_clamp_violation": f"{clamp_diagnostics.manifold_pre_clamp_violation:.16e}",
                "orthogonal_pre_clamp_violation": f"{clamp_diagnostics.orthogonal_pre_clamp_violation:.16e}",
                "final_region_clamped": bool(clamp_diagnostics.final_region_clamped),
                "final_global_clamped": final_global_clamped,
            }
            new_y_cpu = torch.tensor([[current_y]], dtype=y_cpu.dtype)
            X_cpu = torch.cat((X_cpu, x_eval_cpu), dim=0)
            y_cpu = torch.cat((y_cpu, new_y_cpu), dim=0)

            x_eval_unit_dev = to_unit(x_eval_cpu.to(device=device), lb_dev, ub_dev).reshape(-1)
            restart_needed = update_region_after_eval_for_evaluated_restart(
                region=region,
                x_unit=x_eval_unit_dev,
                y_value=current_y,
                dim=dim,
                config=config,
            )
            restart_values: List[float] = []
            if restart_needed:
                restart_seed = seed + 104729 * (problem.state.evaluations + 1)
                if config.turbo_restart_mode == "distance":
                    restart_turbo_v2_region(
                        region,
                        X_unit=to_unit(X_cpu.to(device=device), lb_dev, ub_dev),
                        y=y_cpu.to(device=device),
                        seed=restart_seed,
                        config=config,
                    )
                elif config.turbo_restart_mode == "evaluated":
                    restart_target = config.turbo_restart_points if config.turbo_restart_points is not None else n0
                    restart_n = min(max(0, int(restart_target)), budget - problem.state.evaluations)
                    X_cpu, y_cpu, restart_values = evaluate_restart_batch_and_reset_region(
                        region=region,
                        problem=problem,
                        eval_logger=eval_logger,
                        X_cpu=X_cpu,
                        y_cpu=y_cpu,
                        lb_dev=lb_dev,
                        ub_dev=ub_dev,
                        seed=restart_seed,
                        n_restart=restart_n,
                        config=config,
                    )
                else:
                    raise ValueError(f"Unknown TuRBO restart mode: {config.turbo_restart_mode}")

            clamp_row.update(
                {
                    "region_length_after_update": f"{float(region.length):.16f}",
                    "success_counter_after_update": int(region.success_counter),
                    "failure_counter_after_update": int(region.failure_counter),
                    "restart_count_after_update": int(region.restart_count),
                    "restart_event": int(region.restart_count) > int(clamp_row["restart_count_before"]),
                }
            )
            clamp_rows.append(clamp_row)

            current_best = min(previous_best, current_y)
            state.last_selected_mode = selected_mode
            state.last_eval_improved = current_best < previous_best - config.stagnation_min_improve
            state.best_history.append(current_best)
            for restart_y in restart_values:
                current_best = min(current_best, restart_y)
                state.best_history.append(current_best)
            if selected_mode == "orthogonal":
                state.cooldown_remaining = config.orthogonal_cooldown

            orthogonal_stats_rows.append(
                {
                    "iteration": int(clamp_row["iteration"]),
                    "selected_mode": selected_mode,
                    "pca_rank": int(pca_rank_for_row),
                    "ubr_value": f"{latent_ubr:.16e}",
                    "ubr_trigger": bool(ubr_trigger),
                    "stagnation_trigger": bool(stagnation_trigger),
                    "consecutive_trigger_count": int(consecutive_trigger_count_for_row),
                    "sigma_scale": f"{state.sigma_scale:.16f}",
                    "orthogonal_candidate_count": int(orth_candidate_count),
                    "cooldown_remaining": int(state.cooldown_remaining),
                    "region_length": f"{region_length_for_row:.16f}",
                    "region_length_after_update": f"{float(region.length):.16f}",
                    "restart_count_before": int(clamp_row["restart_count_before"]),
                    "restart_count_after_update": int(region.restart_count),
                    "restart_event": bool(clamp_row["restart_event"]),
                    "orthogonal_pre_clamp_violation": clamp_row["orthogonal_pre_clamp_violation"],
                    "final_region_clamped": bool(clamp_row["final_region_clamped"]),
                    "final_global_clamped": bool(clamp_row["final_global_clamped"]),
                }
            )

            if save_plots and dim == 2:
                plot_turbo_v2_iteration(
                    contour_cache=contour_cache,
                    X_hist=X_cpu,
                    local_x=local_x_dev.detach().cpu(),
                    x_mean=x_mean.detach().cpu(),
                    x_weighted_mean=x_weighted_mean.detach().cpu(),
                    P_r=P_r.detach().cpu(),
                    eigvals=eigvals.detach().cpu(),
                    region_center=tr_center_dev.detach().cpu().reshape(-1),
                    tr_lb=tr_lb_dev.detach().cpu().reshape(-1),
                    tr_ub=tr_ub_dev.detach().cpu().reshape(-1),
                    thompson_candidate=x_eval_cpu.detach().cpu().reshape(-1),
                    final_candidate=x_eval_cpu.detach().cpu().reshape(-1),
                    iter_idx=X_cpu.shape[0] - n0,
                    func_id=fid,
                    dim=dim,
                    selected_mode=selected_mode,
                        region_length=float(region.length),
                        success_counter=int(region.success_counter),
                        failure_counter=int(region.failure_counter),
                        restart_count=int(region.restart_count),
                        out_dir=plot_root / METHOD_TURBO_V5 / "iterations",
                    )

        clamp_dir = dat_path.parent
        write_clamp_rows(clamp_dir / "clamp_stats.csv", clamp_rows)
        write_clamp_summary(clamp_dir / "clamp_summary.csv", clamp_rows)
        write_orthogonal_stats(clamp_dir / "orthogonal_stats.csv", orthogonal_stats_rows)


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
    config: MethodConfig,
) -> None:
    set_seed(seed)
    device = resolve_torch_device(config.device)
    problem = create_problem(fid=fid, dim=dim)
    contour_cache = build_contour_cache(fid=fid, dim=dim, grid_size=grid_size) if save_plots else None

    with EvaluationDatLogger(dat_path=dat_path) as eval_logger:
        train_x = initial_x.clone().to(dtype=torch.double, device=device)
        train_y = evaluate_initial_design(problem=problem, init_x=train_x, eval_logger=eval_logger).to(device=device)

        lb = torch.tensor(problem.bounds.lb, dtype=torch.double, device=device)
        ub = torch.tensor(problem.bounds.ub, dtype=torch.double, device=device)
        bounds = torch.stack([lb, ub], dim=0)

        while problem.state.evaluations < budget:
            gp = SingleTaskGP(
                train_X=train_x,
                train_Y=train_y,
                outcome_transform=Standardize(m=1),
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

            candidate = select_sobol_lcb_candidate(
                gp=gp,
                bounds=bounds,
                seed=seed + 1543 * (problem.state.evaluations + 1),
                config=config,
                beta=2.0,
            )

            current_y = evaluate_point(
                problem=problem,
                x_np=candidate.detach().cpu().numpy().reshape(-1),
                eval_logger=eval_logger,
            )
            new_y = torch.tensor([[current_y]], dtype=train_y.dtype, device=device)

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
        description="Run comparison with TuRBO v5: PCA-BO inside trust regions with region-radius latent bounds.",
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
        help="Compatibility option; TuRBO v5 locks this to simple.",
    )
    parser.add_argument("--mixed-r-take", type=int, default=MIXED_R_TAKE)
    parser.add_argument("--mixed-m-take", type=int, default=MIXED_M_TAKE)
    parser.add_argument(
        "--local-subset-size",
        type=int,
        default=None,
        help="Maximum local trust-region training set size. Defaults to 6 * dim.",
    )
    parser.add_argument("--turbo-length-init", type=float, default=TURBO_LENGTH_INIT)
    parser.add_argument("--turbo-length-min", type=float, default=TURBO_LENGTH_MIN)
    parser.add_argument("--turbo-length-max", type=float, default=TURBO_LENGTH_MAX)
    parser.add_argument("--turbo-success-tolerance", type=int, default=TURBO_SUCCESS_TOLERANCE)
    parser.add_argument("--turbo-failure-tolerance", type=int, default=None)
    parser.add_argument("--turbo-num-regions", type=int, default=TURBO_NUM_REGIONS)
    parser.add_argument("--turbo-candidate-multiplier", type=int, default=TURBO_CANDIDATE_MULTIPLIER)
    parser.add_argument("--turbo-candidate-max", type=int, default=TURBO_CANDIDATE_MAX)
    parser.add_argument("--turbo-init-points-per-region", type=int, default=TURBO_INIT_POINTS_PER_REGION)
    parser.add_argument(
        "--turbo-v5-ard-shape",
        "--turbo-v4-ard-shape",
        "--turbo-v3-ard-shape",
        dest="turbo_v3_ard_shape",
        choices=["off", "on"],
        default="off",
        help="Compatibility option; TuRBO v5 locks ARD shaping off.",
    )
    parser.add_argument(
        "--turbo-v5-orth-scale",
        "--turbo-v4-orth-scale",
        "--turbo-v3-orth-scale",
        dest="turbo_v3_orth_scale",
        type=float,
        default=TURBO_V3_ORTH_SCALE,
        help="Compatibility option; TuRBO v5 locks this multiplier to 0.25.",
    )
    parser.add_argument(
        "--turbo-v5-latent-radius-scale",
        type=float,
        default=TURBO_V5_LATENT_RADIUS_SCALE,
        help="Multiplier on the v5 PCA latent trust-region radius. Values below 1.0 make it stricter.",
    )
    parser.add_argument(
        "--turbo-restart-points",
        type=int,
        default=None,
        help="Number of fresh Sobol points to evaluate if an evaluated restart mode is used. Defaults to n0.",
    )
    parser.add_argument(
        "--turbo-restart-mode",
        choices=["evaluated", "distance"],
        default=TURBO_RESTART_MODE,
        help=(
            "Compatibility option; TuRBO v5 locks this to distance restart, which chooses the "
            "Sobol center farthest from history without spending extra evaluations."
        ),
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(METHODS),
        default=list(METHODS),
        help="Methods to run. Use this to omit methods such as the older TuRBO v1 wrapper.",
    )
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
        turbo_length_init=args.turbo_length_init,
        turbo_length_min=args.turbo_length_min,
        turbo_length_max=args.turbo_length_max,
        turbo_success_tolerance=args.turbo_success_tolerance,
        turbo_failure_tolerance=args.turbo_failure_tolerance,
        turbo_num_regions=args.turbo_num_regions,
        turbo_candidate_multiplier=args.turbo_candidate_multiplier,
        turbo_candidate_max=args.turbo_candidate_max,
        turbo_init_points_per_region=args.turbo_init_points_per_region,
        turbo_v3_ard_shape=args.turbo_v3_ard_shape == "on",
        turbo_v3_orth_scale=args.turbo_v3_orth_scale,
        turbo_v5_latent_radius_scale=args.turbo_v5_latent_radius_scale,
        turbo_restart_points=args.turbo_restart_points,
        turbo_restart_mode=args.turbo_restart_mode,
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
    selected_methods = tuple(args.methods)

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
        "selected_methods": list(selected_methods),
        "method_descriptions": {
            METHOD_PCA_VANILLA: "vanilla PCA-BO from compare_bo_vanilla_vs_dynamic_v2.py",
            METHOD_DYNAMIC: "original dynamic orthogonal V2",
                METHOD_TURBO: "same dynamic orthogonal V2 wrapped in TuRBO-1-style trust region",
                METHOD_TURBO_V2: "dynamic orthogonal V2 inside multi-region TuRBO with local GPs, ARD trust regions, and batched Thompson candidate selection",
                METHOD_TURBO_V3: "dynamic orthogonal PCA-BO candidate selection inside a TuRBO-style trust region",
                METHOD_TURBO_V4: "dynamic orthogonal PCA-BO inside a TuRBO-style trust region with selectable restart and orthogonal refinement modes",
                METHOD_TURBO_V5: "dynamic orthogonal PCA-BO inside a TuRBO-style trust region with region-radius latent bounds, simple orthogonal exploration, and distance restart",
                METHOD_BASELINE: "BoTorch global GP baseline",
        },
        "kernel": {
            "type": method_config.kernel_type,
            "nu": method_config.kernel_nu,
            "lengthscale_bounds": method_config.lengthscale_bounds,
        },
        "pca_backend": "torch.pca_lowrank",
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
        "turbo_wrapper": {
            "variant": "TuRBO-1-style incumbent-centered trust region",
            "length_init": method_config.turbo_length_init,
            "length_min": method_config.turbo_length_min,
            "length_max": method_config.turbo_length_max,
            "success_tolerance": method_config.turbo_success_tolerance,
            "failure_tolerance": (
                method_config.turbo_failure_tolerance
                if method_config.turbo_failure_tolerance is not None
                else args.dim
            ),
            "candidate_generator": "batched Sobol PCA/PEI candidate scoring on the selected torch device, plus dynamic orthogonal rescoring constrained to trust region",
        },
        "turbo_v2": {
            "device": method_config.device,
            "num_regions": method_config.turbo_num_regions,
            "length_init": method_config.turbo_length_init,
            "length_min": method_config.turbo_length_min,
            "length_max": method_config.turbo_length_max,
            "success_tolerance": method_config.turbo_success_tolerance,
            "failure_tolerance": (
                method_config.turbo_failure_tolerance
                if method_config.turbo_failure_tolerance is not None
                else args.dim
            ),
            "candidate_count": f"min({method_config.turbo_candidate_multiplier} * dim, {method_config.turbo_candidate_max})",
            "init_points_per_region": method_config.turbo_init_points_per_region,
            "candidate_generator": "Sobol trust-region candidate tensors plus diagonal Thompson posterior sampling",
            "local_model": "local Matern/RBF ARD GP per trust region, using inside-region points plus nearest neighbors",
            "region_shape": "ARD side lengths length_i = lambda_i * L / geometric_mean(lambda)",
            "restart": "when length < length_min, choose a fresh global Sobol center far from existing history",
        },
            "turbo_v5": {
                "device": method_config.device,
                "num_regions": 1,
            "length_init": method_config.turbo_length_init,
            "length_min": method_config.turbo_length_min,
            "length_max": method_config.turbo_length_max,
            "success_tolerance": method_config.turbo_success_tolerance,
            "failure_tolerance": (
                method_config.turbo_failure_tolerance
                if method_config.turbo_failure_tolerance is not None
                else args.dim
            ),
            "local_subset_size": method_config.local_subset_size,
            "candidate_generator": "local PCA-BO in trust-region-radius latent bounds, followed by constrained orthogonal refinement",
            "main_model": "GP in local PCA latent space",
                "latent_bounds": "current_trust_region_radius",
                "latent_radius_scale": method_config.turbo_v5_latent_radius_scale,
                "latent_bounds_formula": "z_center = ((c_tr - x_mean) - x_weighted_mean) @ P_r.T; rho_tr = latent_radius_scale * 0.5 * min(u_tr - l_tr); bounds_z = [z_center - rho_tr, z_center + rho_tr]",
                "ard_shape": False,
                "ard_note": "if enabled, an auxiliary original-space local GP shapes the trust region; otherwise equal-side-length bounds are used",
                "orthogonal_refinement": ORTHOGONAL_REFINEMENT,
                "orthogonal_refinement_note": (
                    "simple uses one projected multivariate Gaussian perturbation with no mixed-space GP rescoring; "
                    "mixed_gp keeps the older multiple-candidate mixed-space GP rescoring path"
                ),
                "orthogonal_noise_scale": TURBO_V3_ORTH_SCALE,
                "orthogonal_noise_scale_note": "extra TuRBO v5 multiplier applied after scaling by the active trust-region width",
                "restart_mode": TURBO_RESTART_MODE,
                "evaluated_restart_points": method_config.turbo_restart_points if method_config.turbo_restart_points is not None else args.n0,
                "distance_restart_candidate_centers": method_config.turbo_init_points_per_region,
                "restart": (
                    "evaluated: when length < length_min, evaluate a fresh global Sobol batch and center the new region at the best restart point; "
                    "distance: v3 logic, choose the fresh global Sobol center farthest from evaluated history without extra evaluations"
                ),
                "locked_defaults": {
                    "orthogonal_refinement": ORTHOGONAL_REFINEMENT,
                    "turbo_restart_mode": TURBO_RESTART_MODE,
                    "turbo_v3_orth_scale": TURBO_V3_ORTH_SCALE,
                    "turbo_v5_latent_radius_scale": method_config.turbo_v5_latent_radius_scale,
                    "turbo_v3_ard_shape": False,
                    "num_regions": 1,
                },
                "clamp_tracking": {
                    "per_iteration_file": "clamp_stats.csv",
                    "summary_file": "clamp_summary.csv",
                    "tracked_events": [
                        "manifold candidate trust-region violation before clamp",
                        "orthogonal perturbation trust-region violation before clamp",
                        "final trust-region clamp",
                        "final global-bound safety clamp",
                    ],
                    "note": "diagnostic metadata only; it does not change candidate selection",
                },
            },
        "candidate_proposal": {
            "device": method_config.device,
            "applies_to": list(selected_methods),
            "candidate_count": f"min({method_config.turbo_candidate_multiplier} * active_dim, {method_config.turbo_candidate_max})",
            "note": "Selected methods fit/propose on the selected torch device; IOH objective calls and plotting remain CPU/NumPy.",
        },
        "baseline_model": "SingleTaskGP + batched Sobol posterior-LCB candidate scoring",
    }
    save_run_config(run_root / "config.json", config)
    timing_rows: List[Dict[str, object]] = []

    for fid in args.function_ids:
        print(f"Running f{fid}: {args.num_runs} runs for {len(selected_methods)} methods")
        function_root = run_root / f"f{fid}"

        for run_idx in range(args.num_runs):
            seed = args.base_seed + run_idx
            set_seed(seed)

            sampling_problem = create_problem(fid=fid, dim=args.dim)
            lb_np, ub_np = get_bounds(sampling_problem)
            lb = torch.tensor(lb_np, dtype=torch.double)
            ub = torch.tensor(ub_np, dtype=torch.double)
            initial_x = sample_initial_design(lb=lb, ub=ub, dim=args.dim, n0=args.n0, seed=seed)

            save_plots = run_idx == 0 and args.dim == 2
            if METHOD_PCA_VANILLA in selected_methods:
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
                    config=method_config,
                )
                record_method_timing(
                    timing_rows=timing_rows,
                    run_root=run_root,
                    function_root=function_root,
                    method=METHOD_PCA_VANILLA,
                    fid=fid,
                    dim=args.dim,
                    seed=seed,
                    run_idx=run_idx,
                    n0=args.n0,
                    budget=budget,
                    total_seconds=time.perf_counter() - start,
                )
            if METHOD_DYNAMIC in selected_methods:
                start = time.perf_counter()
                run_pca_bo(
                    method_name=METHOD_DYNAMIC,
                    use_turbo=False,
                    fid=fid,
                    dim=args.dim,
                    seed=seed,
                    run_idx=run_idx,
                    n0=args.n0,
                    budget=budget,
                    grid_size=args.grid_size,
                    dat_path=function_root / METHOD_DYNAMIC / f"seed_{seed}" / f"IOHprofiler_f{fid}_DIM{args.dim}.dat",
                    initial_x=initial_x,
                    save_plots=save_plots,
                    plot_root=function_root,
                    config=method_config,
                )
                record_method_timing(
                    timing_rows=timing_rows,
                    run_root=run_root,
                    function_root=function_root,
                    method=METHOD_DYNAMIC,
                    fid=fid,
                    dim=args.dim,
                    seed=seed,
                    run_idx=run_idx,
                    n0=args.n0,
                    budget=budget,
                    total_seconds=time.perf_counter() - start,
                )
            if METHOD_TURBO in selected_methods:
                start = time.perf_counter()
                run_pca_bo(
                    method_name=METHOD_TURBO,
                    use_turbo=True,
                    fid=fid,
                    dim=args.dim,
                    seed=seed,
                    run_idx=run_idx,
                    n0=args.n0,
                    budget=budget,
                    grid_size=args.grid_size,
                    dat_path=function_root / METHOD_TURBO / f"seed_{seed}" / f"IOHprofiler_f{fid}_DIM{args.dim}.dat",
                    initial_x=initial_x,
                    save_plots=save_plots,
                    plot_root=function_root,
                    config=method_config,
                )
                record_method_timing(
                    timing_rows=timing_rows,
                    run_root=run_root,
                    function_root=function_root,
                    method=METHOD_TURBO,
                    fid=fid,
                    dim=args.dim,
                    seed=seed,
                    run_idx=run_idx,
                    n0=args.n0,
                    budget=budget,
                    total_seconds=time.perf_counter() - start,
                )
            if METHOD_TURBO_V2 in selected_methods:
                start = time.perf_counter()
                run_pca_bo_turbo_v2(
                    fid=fid,
                    dim=args.dim,
                    seed=seed,
                    run_idx=run_idx,
                    n0=args.n0,
                    budget=budget,
                    grid_size=args.grid_size,
                    dat_path=function_root / METHOD_TURBO_V2 / f"seed_{seed}" / f"IOHprofiler_f{fid}_DIM{args.dim}.dat",
                    initial_x=initial_x,
                    save_plots=save_plots,
                    plot_root=function_root,
                    config=method_config,
                )
                record_method_timing(
                    timing_rows=timing_rows,
                    run_root=run_root,
                    function_root=function_root,
                    method=METHOD_TURBO_V2,
                    fid=fid,
                    dim=args.dim,
                    seed=seed,
                    run_idx=run_idx,
                    n0=args.n0,
                    budget=budget,
                    total_seconds=time.perf_counter() - start,
                )
            if METHOD_TURBO_V5 in selected_methods:
                start = time.perf_counter()
                run_pca_bo_turbo_v5(
                    fid=fid,
                    dim=args.dim,
                    seed=seed,
                    run_idx=run_idx,
                    n0=args.n0,
                    budget=budget,
                    grid_size=args.grid_size,
                    dat_path=function_root / METHOD_TURBO_V5 / f"seed_{seed}" / f"IOHprofiler_f{fid}_DIM{args.dim}.dat",
                    initial_x=initial_x,
                    save_plots=save_plots,
                    plot_root=function_root,
                    config=method_config,
                )
                record_method_timing(
                    timing_rows=timing_rows,
                    run_root=run_root,
                    function_root=function_root,
                    method=METHOD_TURBO_V5,
                    fid=fid,
                    dim=args.dim,
                    seed=seed,
                    run_idx=run_idx,
                    n0=args.n0,
                    budget=budget,
                    total_seconds=time.perf_counter() - start,
                )
            if METHOD_BASELINE in selected_methods:
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
                    config=method_config,
                )
                record_method_timing(
                    timing_rows=timing_rows,
                    run_root=run_root,
                    function_root=function_root,
                    method=METHOD_BASELINE,
                    fid=fid,
                    dim=args.dim,
                    seed=seed,
                    run_idx=run_idx,
                    n0=args.n0,
                    budget=budget,
                    total_seconds=time.perf_counter() - start,
                )

        save_convergence_plots(
            function_root=function_root,
            fid=fid,
            dim=args.dim,
            budget=budget,
            methods=selected_methods,
        )

    save_timing_summary(run_root=run_root, timing_rows=timing_rows)
    print(f"Saved comparison outputs under: {run_root}")
    return run_root


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
