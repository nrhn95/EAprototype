import os
import csv
import sys
import time
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.simulation.traffic_model import build_single_intersection, build_multi_intersection
from src.optimization.pso import PSO, PSOConfig
from src.visualization.plots import (
    save_figure,
    plot_convergence,
    plot_before_after,
    plot_parameter_study,
    plot_topology_comparison,
    plot_signal_timing,
    plot_operator_comparison,
    plot_statistical_significance,       # NEW
)
from experiments.configs import SEEDS, DEFAULT_CONFIG, ALL_SUITES


TABLES_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "tables")

# Suites that get the dedicated operator-comparison plot
OPERATOR_SUITES = {
    "velocity_strategy",
    "crossover",           # NEW — recombination suite
    "survivor_selection",  # NEW — renamed from pbest_strategy
    "diversity",           # NEW — diversity suite
}

# All suites get a Wilcoxon significance heatmap
STAT_TEST_SUITES = set(ALL_SUITES.keys())


def run_suite(network, configs, seeds, progress_cb=None):

    results = {}
    total   = len(configs) * len(seeds)
    done    = 0

    for label, cfg in configs.items():

        lb, ub = network.bounds()
        pso    = PSO(network.evaluate, lb, ub, cfg, network.repair)
        runs   = []

        for s in seeds:
            r = pso.run(s)
            runs.append(r)
            done += 1
            if progress_cb:
                progress_cb(done, total, label)

        results[label] = runs

    return results


def save_seeds_csv(seeds, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "seed"])
        for i, s in enumerate(seeds):
            w.writerow([i + 1, s])


def save_results_csv(name, results):

    os.makedirs(TABLES_DIR, exist_ok=True)
    path = os.path.join(TABLES_DIR, name + ".csv")

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "seed", "fitness", "evals", "iters"])
        for label, runs in results.items():
            for r in runs:
                w.writerow([
                    label, r.seed, r.best_fitness,
                    r.n_evaluations, len(r.convergence) - 1,
                ])

    return path


def save_summary_csv(name, results):

    os.makedirs(TABLES_DIR, exist_ok=True)
    path = os.path.join(TABLES_DIR, name + "_summary.csv")

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "mean", "std", "min", "max"])
        for label, runs in results.items():
            fits = [r.best_fitness for r in runs]
            w.writerow([label, np.mean(fits), np.std(fits),
                        np.min(fits), np.max(fits)])

    return path


def generate_plots(name, results, baseline):

    import matplotlib.pyplot as plt

    all_curves = {label: [r.convergence for r in runs]
                  for label, runs in results.items()}

    # ── Topology side-by-side (parent selection comparison) ──────────────────
    if name == "topology":
        g = [r.convergence for r in results.get("gbest", [])]
        l = [r.convergence for r in results.get("lbest", [])]
        if g and l:
            fig = plot_topology_comparison(g, l)
            save_figure(fig, "topology.png")
            plt.close(fig)

    # ── Operator-comparison plot (for all operator/selection/diversity suites)
    if name in OPERATOR_SUITES:
        fig = plot_operator_comparison(results, name)
        save_figure(fig, "operator_cmp_" + name + ".png")
        plt.close(fig)

    # ── Per-variant convergence curves ───────────────────────────────────────
    for label, curves in all_curves.items():
        fig = plot_convergence(curves, title=name + " " + label, label=label)
        save_figure(fig, "conv_" + name + "_" + label + ".png")
        plt.close(fig)

    # ── Box-plot across all variants ─────────────────────────────────────────
    best_fits = {label: [r.best_fitness for r in runs]
                 for label, runs in results.items()}
    fig = plot_parameter_study(best_fits, name)
    save_figure(fig, "param_" + name + ".png")
    plt.close(fig)

    # ── Before/after for first variant ───────────────────────────────────────
    first = list(results.keys())[0]
    fits  = [r.best_fitness for r in results[first]]
    fig   = plot_before_after(baseline, fits)
    save_figure(fig, "before_after_" + name + ".png")
    plt.close(fig)

    # ── Wilcoxon rank-sum significance heatmap ───────────────────────────────
    # Only meaningful when there are ≥2 variants with ≥2 runs each.
    if name in STAT_TEST_SUITES and len(results) >= 2:
        fig = plot_statistical_significance(results, name)
        save_figure(fig, "wilcoxon_" + name + ".png")
        plt.close(fig)


def main(argv=None):

    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true",
                        help="Use single-intersection network")
    parser.add_argument("--suite", type=str,
                        help="Run only one named suite")
    args = parser.parse_args(argv)

    network  = build_single_intersection() if args.single else build_multi_intersection(4)
    baseline = network.baseline_fitness()

    seed_path = os.path.join(TABLES_DIR, "seeds.csv")
    save_seeds_csv(SEEDS, seed_path)
    print("Seeds saved to:", seed_path)

    suites = ALL_SUITES
    if args.suite and args.suite in ALL_SUITES:
        suites = {args.suite: ALL_SUITES[args.suite]}

    def progress(done, total, label):
        print(f"{done}/{total}  {label}", end="\r")

    for name, configs in suites.items():

        print(f"\n{'─'*60}")
        print(f"Suite: {name}  ({len(configs)} variants × {len(SEEDS)} seeds)")
        t0 = time.time()

        results = run_suite(network, configs, SEEDS, progress)

        csv1 = save_results_csv(name, results)
        csv2 = save_summary_csv(name, results)
        generate_plots(name, results, baseline)

        print(f"\nSaved: {csv1}")
        print(f"Saved: {csv2}")
        print(f"Time:  {time.time() - t0:.1f}s")

    # ── Final single-run best-solution signal-timing plot ────────────────────
    lb, ub = network.bounds()
    pso    = PSO(network.evaluate, lb, ub, DEFAULT_CONFIG, network.repair)
    best   = pso.run(SEEDS[0])

    import matplotlib.pyplot as plt

    fig = plot_signal_timing(
        best.best_solution,
        network.baseline_solution(),
        network.n_intersections(),
        network.intersections[0].n_phases,
    )
    save_figure(fig, "final.png")
    plt.close(fig)

    pct = (baseline - best.best_fitness) / baseline * 100
    print(f"\nBest fitness : {best.best_fitness:.4f}")
    print(f"Baseline     : {baseline:.4f}")
    print(f"Improvement  : {pct:.1f}%")


if __name__ == "__main__":
    main()