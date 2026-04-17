"""
Genetic Algorithm for real-valued continuous optimisation.

Problem type : Real-valued constrained continuous optimisation.
Constraint   : Repair operator — clip each gene to [lb_i, ub_i].

EA component mapping
────────────────────
Representation       : Real-valued vector (green-time durations, one per phase).
Evaluation function  : Average simulated vehicle delay — lower is better.
Population           : `pop_size` individuals.
Parent selection     : Tournament selection  |  Roulette-wheel selection.
Variation operators  : (A) Mutation  — Uniform | Nonuniform (Michalewicz 1992)
                       (B) Crossover — Whole Arithmetic | Blend (BLX-α)
Survivor selection   : Generational (μ→λ) with elitism | Elitist (μ+λ).
Initialisation       : random | opposition-based (Tizhoosh 2005).
Termination          : max_generations  OR  stagnation_limit consecutive
                       generations without global-best improvement.
"""

import numpy as np
from src.optimization.pso import RunResult   # shared result container


class GAConfig:
    """All configurable parameters for the Genetic Algorithm."""

    def __init__(
        self,
        pop_size: int   = 50,
        max_generations: int = 200,
        crossover_rate: float = 0.90,
        mutation_rate:  float = 0.15,   # per-gene probability

        # ── PARENT SELECTION ──────────────────────────────────────────────────
        # "tournament" : deterministic k-way tournament — fast, tunable pressure.
        # "roulette"   : fitness-proportional (inverted for minimisation).
        selection: str  = "tournament",
        tournament_size: int = 3,

        # ── VARIATION OPERATOR A — MUTATION ───────────────────────────────────
        # "uniform"    : replace gene with U[lb_i, ub_i]  (gene-reset mutation).
        # "nonuniform" : Michalewicz (1992) — perturbation shrinks as generation
        #                count increases, giving exploration early and fine local
        #                search near convergence.
        mutation: str   = "uniform",
        nonuniform_b: float = 2.0,       # shape param — higher ⟹ faster decay

        # ── VARIATION OPERATOR B — CROSSOVER ──────────────────────────────────
        # "arithmetic" : Whole Arithmetic Crossover
        #                c1 = α·p1 + (1-α)·p2,  c2 = (1-α)·p1 + α·p2
        #                α ~ U(0,1).  Offspring lie inside convex hull of parents.
        # "blend"      : Blend Crossover BLX-α
        #                Each gene i drawn from
        #                  [ min(p1_i, p2_i) - α·d,  max(p1_i, p2_i) + α·d ]
        #                  d = |p1_i - p2_i|.
        #                Allows offspring *outside* each parent → more exploration.
        crossover: str  = "arithmetic",
        blend_alpha: float = 0.5,        # BLX-α exploration extent; 0.5 = standard

        # ── SURVIVOR SELECTION ─────────────────────────────────────────────────
        # "generational": μ→λ  — offspring fully replace parents; elite_count
        #                 best parents are preserved to prevent regression (elitism).
        # "elitist"     : (μ+λ) — parents + offspring pooled; best pop_size kept.
        replacement: str = "generational",
        elite_count: int  = 1,

        # ── INITIALISATION ─────────────────────────────────────────────────────
        init_strategy: str = "random",   # "random" | "opposition"

        # ── TERMINATION ────────────────────────────────────────────────────────
        stagnation_limit: int = 30,
    ):
        self.pop_size        = pop_size
        self.max_generations = max_generations
        self.crossover_rate  = crossover_rate
        self.mutation_rate   = mutation_rate
        self.selection       = selection
        self.tournament_size = tournament_size
        self.mutation        = mutation
        self.nonuniform_b    = nonuniform_b
        self.crossover       = crossover
        self.blend_alpha     = blend_alpha
        self.replacement     = replacement
        self.elite_count     = elite_count
        self.init_strategy   = init_strategy
        self.stagnation_limit = stagnation_limit


class GA:
    """Real-valued Genetic Algorithm with pluggable operators."""

    def __init__(self, fitness_fn, lb, ub, config=None):
        self.f      = fitness_fn
        self.lb     = np.array(lb, dtype=float)
        self.ub     = np.array(ub, dtype=float)
        self.cfg    = config if config is not None else GAConfig()
        self.dim    = len(self.lb)
        self.range_ = self.ub - self.lb

    # ─────────────────────────────────────────────────────────────────────────
    # Main optimisation loop
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, seed) -> RunResult:
        cfg = self.cfg
        rng = np.random.default_rng(seed)

        pop     = self._init_population(rng)
        fitness = np.array([self.f(ind) for ind in pop])
        n_eval  = cfg.pop_size

        best_idx = int(np.argmin(fitness))
        best_ind = pop[best_idx].copy()
        best_fit = float(fitness[best_idx])

        convergence = [best_fit]
        stagnation  = 0

        for gen in range(1, cfg.max_generations + 1):

            # ── breed offspring ──────────────────────────────────────────────
            offspring   = []
            off_fitness = []

            while len(offspring) < cfg.pop_size:
                p1 = pop[self._select(fitness, rng)].copy()
                p2 = pop[self._select(fitness, rng)].copy()

                if rng.random() < cfg.crossover_rate:
                    c1, c2 = self._crossover(p1, p2, rng)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                c1 = self._repair(self._mutate(c1, gen, rng))
                c2 = self._repair(self._mutate(c2, gen, rng))

                f1 = self.f(c1)
                f2 = self.f(c2)
                n_eval += 2

                offspring.extend([c1, c2])
                off_fitness.extend([f1, f2])

            offspring   = np.array(offspring[: cfg.pop_size])
            off_fitness = np.array(off_fitness[: cfg.pop_size])

            # ── survivor selection ───────────────────────────────────────────
            pop, fitness = self._replace(pop, fitness, offspring, off_fitness)

            # ── global best update ───────────────────────────────────────────
            gi = int(np.argmin(fitness))
            if fitness[gi] < best_fit:
                best_fit   = float(fitness[gi])
                best_ind   = pop[gi].copy()
                stagnation = 0
            else:
                stagnation += 1

            convergence.append(best_fit)

            if stagnation >= cfg.stagnation_limit:
                # Pad so all curves have the same length for uniform plotting
                remaining = cfg.max_generations - gen
                convergence.extend([best_fit] * remaining)
                break

        return RunResult(best_fit, best_ind, convergence, seed, cfg, n_eval)

    # ─────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ─────────────────────────────────────────────────────────────────────────

    def _init_population(self, rng):
        n = self.cfg.pop_size
        if self.cfg.init_strategy == "opposition":
            # Opposition-based initialisation (Tizhoosh 2005):
            # evaluate x and its opposite; keep the better one.
            pop = []
            for _ in range(n):
                x   = rng.uniform(self.lb, self.ub)
                opp = np.clip(self.lb + self.ub - x, self.lb, self.ub)
                pop.append(x if self.f(x) <= self.f(opp) else opp)
            return np.array(pop)
        return rng.uniform(self.lb, self.ub, (n, self.dim))

    # ─────────────────────────────────────────────────────────────────────────
    # Parent selection
    # ─────────────────────────────────────────────────────────────────────────

    def _select(self, fitness, rng) -> int:
        cfg = self.cfg
        if cfg.selection == "tournament":
            # k-way tournament: sample k candidates, return index of best
            k   = min(cfg.tournament_size, len(fitness))
            idx = rng.choice(len(fitness), size=k, replace=False)
            return int(idx[np.argmin(fitness[idx])])
        else:  # "roulette" — fitness-proportional, inverted for minimisation
            inv   = 1.0 / (fitness + 1e-9)
            probs = inv / inv.sum()
            return int(rng.choice(len(fitness), p=probs))

    # ─────────────────────────────────────────────────────────────────────────
    # Crossover (Variation Operator B)
    # ─────────────────────────────────────────────────────────────────────────

    def _crossover(self, p1, p2, rng):
        if self.cfg.crossover == "arithmetic":
            # Whole Arithmetic Crossover
            # α drawn fresh for every pair; both children are produced.
            alpha = rng.random()
            c1 = alpha * p1 + (1.0 - alpha) * p2
            c2 = (1.0 - alpha) * p1 + alpha * p2
            return c1, c2
        else:  # "blend" — BLX-α
            # Blend Crossover: gene i of each child sampled from an
            # interval that extends α * d beyond each parent's bounds,
            # where d = |p1_i - p2_i|.  Encourages wider exploration.
            alpha = self.cfg.blend_alpha
            lo    = np.minimum(p1, p2)
            hi    = np.maximum(p1, p2)
            d     = hi - lo
            c1    = rng.uniform(lo - alpha * d, hi + alpha * d)
            c2    = rng.uniform(lo - alpha * d, hi + alpha * d)
            return c1, c2

    # ─────────────────────────────────────────────────────────────────────────
    # Mutation (Variation Operator A)
    # ─────────────────────────────────────────────────────────────────────────

    def _mutate(self, ind, gen, rng):
        cfg    = self.cfg
        result = ind.copy()

        for i in range(self.dim):
            if rng.random() >= cfg.mutation_rate:
                continue

            if cfg.mutation == "uniform":
                # Gene-reset: replace with a uniformly random feasible value
                result[i] = rng.uniform(self.lb[i], self.ub[i])

            else:  # "nonuniform" — Michalewicz (1992)
                # delta(t, y) = y · (1 - r^((1 - t/T)^b))
                # As t → T the exponent → 0 so the perturbation → 0,
                # giving fine-grained local search near convergence.
                t      = gen
                T      = cfg.max_generations
                b      = cfg.nonuniform_b
                r      = rng.random()
                factor = 1.0 - r ** ((1.0 - t / T) ** b)

                if rng.random() < 0.5:
                    # Upward perturbation (toward upper bound)
                    result[i] += (self.ub[i] - result[i]) * factor
                else:
                    # Downward perturbation (toward lower bound)
                    result[i] -= (result[i] - self.lb[i]) * factor

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Survivor selection
    # ─────────────────────────────────────────────────────────────────────────

    def _replace(self, parents, par_fit, offspring, off_fit):
        cfg = self.cfg
        n   = cfg.pop_size

        if cfg.replacement == "elitist":
            # (μ+λ) strategy: pool all individuals, keep the best n
            combined     = np.vstack([parents, offspring])
            combined_fit = np.concatenate([par_fit, off_fit])
            keep         = np.argsort(combined_fit)[:n]
            return combined[keep], combined_fit[keep]

        else:  # "generational" with elitism
            # μ→λ: offspring replace parents, but the elite_count
            # best parents are injected to prevent regression.
            n_elite   = min(cfg.elite_count, n)
            elite_idx = np.argsort(par_fit)[:n_elite]
            n_off     = n - n_elite
            off_idx   = np.argsort(off_fit)[:n_off]
            new_pop   = np.vstack([parents[elite_idx], offspring[off_idx]])
            new_fit   = np.concatenate([par_fit[elite_idx], off_fit[off_idx]])
            return new_pop, new_fit

    # ─────────────────────────────────────────────────────────────────────────
    # Constraint handling
    # ─────────────────────────────────────────────────────────────────────────

    def _repair(self, x):
        """Clip each gene to [lb_i, ub_i] (repair operator)."""
        return np.clip(x, self.lb, self.ub)
