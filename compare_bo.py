import argparse
import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from scipy.optimize import differential_evolution
from scipy.stats import qmc
from torch import pca_lowrank

from PCA_BO import PEI, plot_pcabo_iteration, plot_weighted_points_iteration


METHOD_PCA = "pca_bo"
METHOD_BASELINE = "botorch_baseline"
RUN_SUFFIX = "pca_kernel_pcalowrank"
KERNEL_NU = 2.5
KERNEL_LENGTHSCALE_BOUNDS = (0.005, 4.0)
METHOD_LABELS = {
    METHOD_PCA: "PCA-BO",
    METHOD_BASELINE: "BoTorch Baseline",
}
METHOD_COLORS = {
    METHOD_PCA: "#1f77b4",
    METHOD_BASELINE: "#d62728",
}


@dataclass
class RunContext:
    method: str
    function_id: int
    seed: int
    run_id: str
    dim: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_problem(fid: int, dim: int):
    return ioh.get_problem(fid, 1, dim, ioh.ProblemClass.BBOB)


def generate_covar_module(active_dim: int):
    internal_kernel = MaternKernel(
        nu=KERNEL_NU,
        ard_num_dims=active_dim,
        lengthscale_constraint=Interval(*KERNEL_LENGTHSCALE_BOUNDS),
    )
    return ScaleKernel(internal_kernel)


def get_bounds(problem) -> Tuple[np.ndarray, np.ndarray]:
    lb = np.asarray(problem.bounds.lb, dtype=float)
    ub = np.asarray(problem.bounds.ub, dtype=float)
    return lb, ub


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
    series_current_best: Dict[str, List[np.ndarray]] = {METHOD_PCA: [], METHOD_BASELINE: []}
    optimum = float(create_problem(fid=fid, dim=dim).optimum.y)

    for method in (METHOD_PCA, METHOD_BASELINE):
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
    # Reuse the same plotting function as PCA-BO; pass no PCA directions.
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

            covar_module = generate_covar_module(active_dim=z_r.shape[1])
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
                    out_dir=str(plot_root / METHOD_PCA / "weighted"),
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
                    out_dir=str(plot_root / METHOD_PCA / "iterations"),
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
        description="Run PCA-BO and official BoTorch baseline with IOH logging.",
    )
    parser.add_argument("--base-seed", type=int, default=12)
    parser.add_argument("--function-ids", nargs="+", type=int, default=[2])
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--n0", type=int, default=8)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--num-runs", type=int, default=30)
    parser.add_argument("--grid-size", type=int, default=120)
    parser.add_argument("--run-root", type=str, default=None)
    return parser.parse_args()


def run_experiment(args) -> Path:
    budget = args.budget if args.budget is not None else 50 * args.dim

    if args.run_root:
        run_root = Path(args.run_root)
    else:
        run_root = Path("comparison_runs") / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{RUN_SUFFIX}"
    run_root.mkdir(parents=True, exist_ok=True)

    config = {
        "base_seed": args.base_seed,
        "function_ids": args.function_ids,
        "dim": args.dim,
        "n0": args.n0,
        "budget": budget,
        "num_runs": args.num_runs,
        "grid_size": args.grid_size,
        "kernel": {
            "type": "Matern",
            "nu": KERNEL_NU,
            "lengthscale_bounds": KERNEL_LENGTHSCALE_BOUNDS,
        },
        "pca_backend": "torch.pca_lowrank",
        "baseline_model": "SingleTaskGP + LogExpectedImprovement + optimize_acqf",
    }
    save_run_config(run_root / "config.json", config)

    for fid in args.function_ids:
        print(f"Running f{fid}: {args.num_runs} runs per method")
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
            )
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

        save_convergence_plots(
            function_root=function_root,
            fid=fid,
            dim=args.dim,
            budget=budget,
        )

    print(f"Saved comparison outputs under: {run_root}")
    return run_root


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
