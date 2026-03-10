import os
from datetime import datetime

from botorch.acquisition.analytic import ExpectedImprovement, LogExpectedImprovement
from scipy.optimize import differential_evolution
import numpy as np
import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
import random
from botorch.models.transforms.outcome import Standardize
import cocoex
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from scipy.stats import qmc



def plot_pcabo_iteration(
    contour_cache,          # dict with keys: X1, X2, Z, lb, ub
    X_hist,                 # torch.Tensor, shape (n, 2)
    x_mean,                 # torch.Tensor, shape (2,)
    x_weighted_mean,        # torch.Tensor, shape (2,)
    P_r,                    # torch.Tensor, shape (r, 2)
    eigvals,                # torch.Tensor, shape (D,), sorted desc
    iter_idx,               # int
    func_id,                # int
    dim=2,                  # int
    out_dir=None,           # e.g. "figures/f2"
):
    """
    Plot one BO iteration without calling COCO problem(...) inside plotting.
    """
    assert X_hist.shape[1] == 2, "X_hist must be (n,2)."

    X1 = contour_cache["X1"]
    X2 = contour_cache["X2"]
    Z = contour_cache["Z"]
    lb = contour_cache["lb"]
    ub = contour_cache["ub"]
    zmin = float(np.min(Z))
    zmax = float(np.max(Z))
    if zmin >= 0.0:
        zcap = float(np.percentile(Z, 99.5))
        if zcap <= zmin:
            zcap = zmax
        norm = mcolors.PowerNorm(gamma=0.35, vmin=zmin, vmax=zcap)
    else:
        linthresh = max(1e-3, 0.01 * max(abs(zmin), abs(zmax)))
        norm = mcolors.SymLogNorm(linthresh=linthresh, vmin=zmin, vmax=zmax, base=10)

    # PCA lines in original space: x(t) = c + t * v_k
    c = (x_mean + x_weighted_mean).detach().cpu().numpy()
    pca_lines = []
    eig_used = eigvals[: P_r.shape[0]].detach().cpu().numpy()
    eig_used = np.clip(eig_used, a_min=0.0, a_max=None)
    scales = np.sqrt(eig_used + 1e-12)
    if np.max(scales) > 0:
        scales = scales / np.max(scales)
    else:
        scales = np.ones_like(scales)

    base_extent = 0.5 * min(ub[0] - lb[0], ub[1] - lb[1])

    for k in range(P_r.shape[0]):
        v = P_r[k].detach().cpu().numpy()
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-12:
            continue
        v = v / v_norm

        t_candidates = []
        for j in range(2):
            if abs(v[j]) < 1e-12:
                continue
            t_candidates.append((lb[j] - c[j]) / v[j])
            t_candidates.append((ub[j] - c[j]) / v[j])
        t_candidates = np.asarray(t_candidates, dtype=float)
        tmin, tmax = np.min(t_candidates), np.max(t_candidates)

        half_len = base_extent * scales[k]
        t = np.linspace(max(tmin, -half_len), min(tmax, half_len), 200)
        line_k = c[None, :] + t[:, None] * v[None, :]
        pca_lines.append((k, line_k))

    # History
    X_np = X_hist.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(7, 6))
    cs = ax.contourf(X1, X2, Z, levels=120, cmap="cividis", norm=norm, alpha=0.95)
    ax.contour(X1, X2, Z, levels=20, colors="black", linewidths=0.35, alpha=0.35)
    plt.colorbar(cs, ax=ax, label="f(x)")

    cmap = plt.get_cmap("cool")
    for k, line_k in pca_lines:
        color = cmap(k / max(1, len(pca_lines) - 1))
        label = "PCA manifold directions" if k == 0 else None
        ax.plot(line_k[:, 0], line_k[:, 1], color=color, linewidth=2.0, label=label)

    ax.scatter(
        X_np[:, 0],
        X_np[:, 1],
        c="white",
        s=26,
        edgecolors="black",
        linewidths=0.35,
        label="Evaluated points",
    )
    ax.scatter(
        X_np[-1, 0],
        X_np[-1, 1],
        c="#ff3b30",
        s=70,
        edgecolors="black",
        linewidths=0.5,
        zorder=5,
        label="Latest point",
    )

    ax.set_xlim(lb[0], ub[0])
    ax.set_ylim(lb[1], ub[1])
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(f"f{func_id} d{dim} | iteration {iter_idx}")
    ax.legend(loc="upper right")
    plt.tight_layout()

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        fname = f"iter_{iter_idx:03d}.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_weighted_points_iteration(
    contour_cache,
    X_hist,
    w,
    x_mean,
    iter_idx,
    func_id,
    dim=2,
    out_dir=None,
):
    assert X_hist.shape[1] == 2, "X_hist must be (n,2)."

    lb = contour_cache["lb"]
    ub = contour_cache["ub"]
    X1 = contour_cache["X1"]
    X2 = contour_cache["X2"]
    Z = contour_cache["Z"]
    zmin = float(np.min(Z))
    zmax = float(np.max(Z))
    if zmin >= 0.0:
        zcap = float(np.percentile(Z, 99.5))
        if zcap <= zmin:
            zcap = zmax
        norm = mcolors.PowerNorm(gamma=0.35, vmin=zmin, vmax=zcap)
    else:
        linthresh = max(1e-3, 0.01 * max(abs(zmin), abs(zmax)))
        norm = mcolors.SymLogNorm(linthresh=linthresh, vmin=zmin, vmax=zmax, base=10)
    X_np = X_hist.detach().cpu().numpy()
    w = w.detach().cpu().numpy()
    x_mean_np = x_mean.detach().cpu().numpy()

    # Visualize weighted "moved" points in original space:
    # x_moved_i = mu + w_i * (x_i - mu)
    X_moved = x_mean_np[None, :] + w[:, None] * (X_np - x_mean_np[None, :])

    fig, ax = plt.subplots(figsize=(7, 6))
    cs = ax.contourf(X1, X2, Z, levels=120, cmap="cividis", norm=norm, alpha=0.95)
    ax.contour(X1, X2, Z, levels=20, colors="black", linewidths=0.35, alpha=0.35)
    plt.colorbar(cs, ax=ax, label="f(x)")

    # Dashed correspondence lines: original point -> weighted moved point
    for i in range(X_np.shape[0]):
        ax.plot(
            [X_np[i, 0], X_moved[i, 0]],
            [X_np[i, 1], X_moved[i, 1]],
            linestyle="--",
            color="white",
            linewidth=0.8,
            alpha=0.55,
            zorder=2,
        )

    # Original points
    ax.scatter(
        X_np[:, 0],
        X_np[:, 1],
        c="none",
        s=28,
        edgecolors="white",
        linewidths=0.7,
        label="Original points",
        zorder=3,
    )

    # Weighted moved points (size+color by weight)
    sc = ax.scatter(
        X_moved[:, 0],
        X_moved[:, 1],
        c=w,
        s=70,
        cmap="magma",
        edgecolors="black",
        linewidths=0.4,
        label="Weighted moved points",
        zorder=4,
    )
    plt.colorbar(sc, ax=ax, label="Normalized weight")

    ax.set_xlim(lb[0], ub[0])
    ax.set_ylim(lb[1], ub[1])
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(f"f{func_id} d{dim} | weighted points | iteration {iter_idx}")
    ax.legend(loc="upper right")
    plt.tight_layout()

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        fname = f"iter_{iter_idx:03d}.png"
        plt.savefig(os.path.join(out_dir, fname), dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_target_precision_ci(tp_runs, func_id, out_dir, method_label="PCA-BO"):
    """
    tp_runs shape: (num_runs, num_evaluations)
    """
    mu = tp_runs.mean(axis=0)
    if tp_runs.shape[0] > 1:
        std = tp_runs.std(axis=0, ddof=1)
        ci = 1.96 * std / np.sqrt(tp_runs.shape[0])
    else:
        ci = np.zeros_like(mu)

    x = np.arange(1, tp_runs.shape[1] + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, mu, color="#1f77b4", linewidth=2.0, label=method_label)
    ax.fill_between(x, mu - ci, mu + ci, color="#1f77b4", alpha=0.25, label="95% CI")
    ax.set_xlabel("Function evaluations")
    ax.set_ylabel("Target precision (best-so-far - f_opt)")
    ax.set_title(f"f{func_id} | Mean Target Precision ± 95% CI")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, "target_precision_ci.png"), dpi=160)
    plt.close(fig)


def is_feasible(z, bounds, mapper):
    x = mapper(z)
    if isinstance(bounds, (tuple, list)):
        lb, ub = bounds
    else:
        lb, ub = bounds[0], bounds[1]
    eps = 1e-8
    lb = lb.to(dtype=x.dtype, device=x.device)
    ub = ub.to(dtype=x.dtype, device=x.device)
    feasible_mask = (x >= (lb - eps)) & (x <= (ub + eps))
    feasible = feasible_mask.all(dim=-1)
    return feasible


def penalty(z, bounds, weight=1.0, mapper=None):
    x = mapper(z)
    if isinstance(bounds, (tuple, list)):
        lb, ub = bounds
    else:
        lb, ub = bounds[0], bounds[1]

    lb = lb.to(dtype=x.dtype, device=x.device)
    ub = ub.to(dtype=x.dtype, device=x.device)

    lower_violation = torch.clamp(lb - x, min=0.0)
    upper_violation = torch.clamp(x - ub, min=0.0)
    violation = lower_violation + upper_violation
    return weight * torch.sum(violation * violation, dim=-1)


class PEI(LogExpectedImprovement):
    def __init__(self, gp, best_f, bounds, penalty_weight, mapper, penalty = penalty):
        super().__init__(gp, best_f, maximize=False)
        self.base = LogExpectedImprovement(gp, best_f, maximize=False)
        self.bounds = bounds
        self.penalty_weight = penalty_weight
        self.mapper = mapper
        self.penalty = penalty

    def forward(self, z):
        base = self.base(z).squeeze(-1)
        pen = self.penalty(z, self.bounds, mapper=self.mapper).squeeze(-1)
        return base - self.penalty_weight * pen


def compute_PCA(init_x, init_y, alpha=0.95):
    # compute mean and center
    x_mean = init_x.mean(dim=0)
    x_bar = init_x - x_mean #(n,D)

    # sort and calculate ranks
    y = init_y.squeeze(-1)
    _, sort_idx = torch.sort(y, dim=0, descending=False)  # minimization: best first
    n = y.shape[0]
    ranks = torch.empty(n, dtype=init_x.dtype, device=init_x.device)
    ranks[sort_idx] = torch.arange(1, n + 1, dtype=init_x.dtype, device=init_x.device)
    w_tilde = torch.log(torch.tensor(float(n), dtype=init_x.dtype, device=init_x.device)) - torch.log(ranks)
    w = w_tilde / w_tilde.sum()
    W = torch.diag(w) #(n,n)

    #rescale with weights
    X_weighted = W @ x_bar

    x_weighted_mean = X_weighted.mean(dim=0)
    x_weighted_bar = X_weighted - x_weighted_mean

    #covariance matrix
    C = (x_weighted_bar.T @ x_weighted_bar) / (n - 1) #(D,D)

    eigvals, eigvecs = torch.linalg.eigh(C)

    # sorting and reordering
    order = torch.argsort(eigvals, descending=True) 
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]          

    # choosing r using variance threshold
    cum_ratio = torch.cumsum(eigvals, dim=0) / eigvals.sum()
    r = int(torch.searchsorted(cum_ratio, torch.tensor(alpha, dtype=cum_ratio.dtype, device=cum_ratio.device)).item()) + 1

    V_r = eigvecs[:, :r] # (D, r)
    P_r = V_r.T # (r, D)

    return (x_mean, x_weighted_mean, P_r, eigvals, w)




def main():
    base_seed = 12
    function_ids = [2, 8, 9, 12, 22]
    dim = 2
    n0 = 8
    budget = 50 * dim
    num_runs = 30
    grid_size = 120
    save_iteration_plots_only_first_run = True

    observer = cocoex.Observer("bbob", "result_folder: results/pcabo_f2_f8_f9_f12_f22_d2")
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = os.path.join("figures_runs", run_stamp)
    print(f"Saving figures to: {run_root}")

    for fid in function_ids:
        # Build contour cache once per function (also used for f_opt approximation).
        plot_problem = next(iter(cocoex.Suite("bbob", "", f"dimensions:{dim} function_indices:{fid} instance_indices:1")))
        lb_np = np.asarray(plot_problem.lower_bounds, dtype=float)
        ub_np = np.asarray(plot_problem.upper_bounds, dtype=float)
        x1 = np.linspace(lb_np[0], ub_np[0], grid_size)
        x2 = np.linspace(lb_np[1], ub_np[1], grid_size)
        X1, X2 = np.meshgrid(x1, x2)
        grid = np.stack([X1.ravel(), X2.ravel()], axis=-1)
        Z = np.array([plot_problem(p) for p in grid], dtype=float).reshape(grid_size, grid_size)
        contour_cache = {"X1": X1, "X2": X2, "Z": Z, "lb": lb_np, "ub": ub_np}

        # 2D approximation of optimum (dense grid min)
        f_opt = float(np.min(Z))
        tp_runs = []
        print(f"Running f{fid}: {num_runs} runs")

        for run_idx in range(num_runs):
            seed = base_seed + run_idx
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            problem = next(iter(cocoex.Suite("bbob", "", f"dimensions:{dim} function_indices:{fid} instance_indices:1")))
            problem.observe_with(observer)

            lb = torch.tensor(problem.lower_bounds)
            ub = torch.tensor(problem.upper_bounds)

            sampler = qmc.LatinHypercube(d=problem.dimension)
            u = sampler.random(n=n0)
            u = torch.tensor(u, dtype=lb.dtype)
            init_x = lb + (ub - lb) * u

            init_y_vals = []
            y_trace = []
            for i in range(init_x.shape[0]):
                y = problem(init_x[i].detach().cpu().numpy())
                init_y_vals.append(y)
                y_trace.append(float(y))
            init_y = torch.tensor(init_y_vals, dtype=init_x.dtype).unsqueeze(-1)

            while problem.evaluations < budget:
                x_mean, x_weighted_mean, P_r, eigvals, w = compute_PCA(init_x, init_y, alpha=0.95)
                Z_r = ((init_x - x_mean) - x_weighted_mean) @ P_r.T
                mapper = lambda z: z @ P_r + x_mean + x_weighted_mean

                x_center = 0.5 * (lb + ub)
                z_center = ((x_center - x_mean) - x_weighted_mean) @ P_r.T
                rho = 0.5 * torch.min(ub - lb)
                bounds_z = torch.stack([z_center - rho, z_center + rho], dim=0)

                gp = SingleTaskGP(train_X=Z_r, train_Y=init_y, outcome_transform=Standardize(m=1))
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_mll(mll)

                best_f = init_y.min()
                acquisition = PEI(gp=gp, best_f=best_f, bounds=(lb, ub), penalty_weight=100.0, mapper=mapper)

                lbz = bounds_z[0].detach().cpu().numpy()
                ubz = bounds_z[1].detach().cpu().numpy()
                de_bounds = [(float(l), float(u)) for l, u in zip(lbz, ubz)]

                def obj(z_np):
                    z_t = torch.tensor(z_np, dtype=init_x.dtype, device=init_x.device).view(1, 1, -1)
                    with torch.no_grad():
                        val = acquisition(z_t).squeeze()
                    return -float(val.detach().cpu().item())

                res = differential_evolution(
                    obj,
                    bounds=de_bounds,
                    strategy="best1bin",
                    maxiter=80,
                    popsize=15,
                    polish=True,
                    seed=seed,
                )
                new_z = torch.tensor(res.x, dtype=init_x.dtype, device=init_x.device).view(1, -1)
                new_x = mapper(new_z)
                y_float = problem(new_x.detach().cpu().numpy().reshape(-1))
                new_y = torch.tensor([[y_float]], dtype=init_y.dtype)
                y_trace.append(float(y_float))

                if (not save_iteration_plots_only_first_run) or run_idx == 0:
                    plot_weighted_points_iteration(
                        contour_cache=contour_cache,
                        X_hist=init_x,
                        w=w,
                        x_mean=x_mean,
                        iter_idx=init_x.shape[0] - n0,
                        func_id=fid,
                        dim=dim,
                        out_dir=os.path.join(run_root, f"f{fid}", "weighted"),
                    )

                init_x = torch.cat((init_x, new_x), dim=0)
                init_y = torch.cat((init_y, new_y), dim=0)

                if (not save_iteration_plots_only_first_run) or run_idx == 0:
                    plot_pcabo_iteration(
                        contour_cache=contour_cache,
                        X_hist=init_x,
                        x_mean=x_mean,
                        x_weighted_mean=x_weighted_mean,
                        P_r=P_r,
                        eigvals=eigvals,
                        iter_idx=init_x.shape[0] - n0,
                        func_id=fid,
                        dim=dim,
                        out_dir=os.path.join(run_root, f"f{fid}", "iterations"),
                    )

            y_arr = np.array(y_trace[:budget], dtype=float)
            best_so_far = np.minimum.accumulate(y_arr)
            tp_runs.append(best_so_far - f_opt)
            print(f"f{fid} run {run_idx + 1}/{num_runs} complete")

        tp_runs = np.vstack(tp_runs)
        plot_target_precision_ci(
            tp_runs=tp_runs,
            func_id=fid,
            out_dir=os.path.join(run_root, f"f{fid}", "summary"),
            method_label="PCA-BO",
        )
        print(f"Saved summary plot for f{fid}")




if __name__ == "__main__":
    main()
