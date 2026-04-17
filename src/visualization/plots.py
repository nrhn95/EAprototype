import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

matplotlib.use("Agg")

GRAPH_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "results", "graphs")


def save_figure(fig, filename):
    os.makedirs(GRAPH_DIR, exist_ok=True)
    path = os.path.join(GRAPH_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_convergence(convergence_curves, title="PSO Convergence", label=""):

    curves = np.array(convergence_curves)
    mean   = curves.mean(axis=0)
    std    = curves.std(axis=0)
    iters  = np.arange(len(mean))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(iters, mean, lw=2, label=label)
    ax.fill_between(iters, mean - std, mean + std, alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average Delay")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_before_after(baseline, best_fitnesses):

    opt_mean = np.mean(best_fitnesses)
    opt_std  = np.std(best_fitnesses)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(
        ["Baseline", "PSO"],
        [baseline, opt_mean],
        yerr=[0, opt_std],
        capsize=5,
    )
    ax.set_ylabel("Average Delay")
    ax.set_title("Before vs After Optimisation")
    fig.tight_layout()
    return fig


def plot_parameter_study(results_dict, param_name):

    labels = list(results_dict.keys())
    data   = [results_dict[k] for k in labels]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.boxplot(data, labels=labels)
    ax.set_xlabel(param_name)
    ax.set_ylabel("Best Fitness")
    ax.set_title("Parameter Study: " + param_name)
    fig.tight_layout()
    return fig


def plot_topology_comparison(gbest_curves, lbest_curves):

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, curves, title in zip(
        axes,
        [gbest_curves, lbest_curves],
        ["gbest (global parent selection)", "lbest (local parent selection)"],
    ):
        arr  = np.array(curves)
        mean = arr.mean(axis=0)
        std  = arr.std(axis=0)
        iters = np.arange(len(mean))

        ax.plot(iters, mean)
        ax.fill_between(iters, mean - std, mean + std, alpha=0.2)
        ax.set_title(title)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Delay")

    fig.tight_layout()
    return fig


def plot_signal_timing(best_solution, baseline_solution, n_intersections, n_phases=2):

    n_vars = n_intersections * n_phases
    labels = [
        "INT-" + str(i + 1) + " P" + str(p + 1)
        for i in range(n_intersections)
        for p in range(n_phases)
    ]

    y = np.arange(n_vars)
    h = 0.3

    fig, ax = plt.subplots(figsize=(9, max(4, n_vars * 0.5)))
    ax.barh(y + h / 2, best_solution[:n_vars],     h, label="Optimised")
    ax.barh(y - h / 2, baseline_solution[:n_vars], h, label="Baseline")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Green Time (s)")
    ax.set_title("Signal Timing Comparison")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_operator_comparison(results_dict, suite_name):
    """
    Left panel  — mean convergence curves (one per variant, ±1 std band).
    Right panel — box-plot of final best-fitness distributions (30 runs).
    """
    labels  = list(results_dict.keys())
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, (ax_conv, ax_box) = plt.subplots(1, 2, figsize=(14, 5))

    for idx, label in enumerate(labels):
        runs   = results_dict[label]
        curves = np.array([r.convergence for r in runs])
        mean   = curves.mean(axis=0)
        std    = curves.std(axis=0)
        iters  = np.arange(len(mean))
        col    = colours[idx % len(colours)]

        ax_conv.plot(iters, mean, lw=2, label=label, color=col)
        ax_conv.fill_between(iters, mean - std, mean + std, alpha=0.15, color=col)

    ax_conv.set_title(suite_name + " — convergence")
    ax_conv.set_xlabel("Iteration")
    ax_conv.set_ylabel("Average Delay (lower is better)")
    ax_conv.legend(fontsize=9)

    fitness_data = [[r.best_fitness for r in results_dict[lbl]] for lbl in labels]

    bp = ax_box.boxplot(
        fitness_data,
        labels=labels,
        patch_artist=True,
        medianprops=dict(color="black", lw=2),
    )
    for patch, col in zip(bp["boxes"], colours):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)

    ax_box.set_title(suite_name + " — final fitness (30 runs)")
    ax_box.set_xlabel("Variant")
    ax_box.set_ylabel("Best Fitness")
    ax_box.tick_params(axis="x", rotation=15)

    fig.suptitle("Operator comparison: " + suite_name, fontsize=13,
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_statistical_significance(results_dict, suite_name, alpha=0.05):
    """
    Pairwise Wilcoxon rank-sum test heatmap.

    For every pair of configurations (A, B) we run a two-sided Wilcoxon
    rank-sum test on their 30-run best-fitness distributions.  The heatmap
    cell colour encodes the p-value:

        green  (p < alpha)   → statistically significant difference
        red    (p ≥ alpha)   → no significant difference

    The p-value is annotated inside each cell.  The diagonal is masked (a
    configuration compared with itself always yields p=1).

    Parameters
    ----------
    results_dict : dict[label, list[RunResult]]
    suite_name   : str  — used in the figure title
    alpha        : float — significance level (default 0.05)
    """
    labels = list(results_dict.keys())
    n      = len(labels)

    # Build matrix of p-values
    p_matrix = np.ones((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            fits_i = [r.best_fitness for r in results_dict[labels[i]]]
            fits_j = [r.best_fitness for r in results_dict[labels[j]]]
            _, p   = stats.ranksums(fits_i, fits_j)
            p_matrix[i, j] = p

    fig, ax = plt.subplots(figsize=(max(6, n * 1.4), max(5, n * 1.2)))

    # Colour: green if significant, red if not
    colours = np.where(p_matrix < alpha, p_matrix, np.nan)
    img = ax.imshow(
        np.where(p_matrix < alpha, 0.3, 0.8),
        cmap="RdYlGn_r",
        vmin=0, vmax=1,
        aspect="auto",
    )

    # Annotate each cell with the p-value
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=9, color="gray")
            else:
                p_val = p_matrix[i, j]
                sig   = "*" if p_val < alpha else ""
                ax.text(j, i, f"{p_val:.3f}{sig}",
                        ha="center", va="center", fontsize=8,
                        color="black")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    ax.set_title(
        f"Wilcoxon rank-sum p-values — {suite_name}\n"
        f"Green = significant (p < {alpha}), Red = not significant",
        fontsize=10,
    )

    fig.tight_layout()
    return fig
