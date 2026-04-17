import numpy as np


class PSOConfig:
    """
    Configuration for the PSO optimiser.

    Problem type: Real-valued constrained continuous optimisation.
    Constraint handling: Repair operator (clip to feasible bounds).

    EA component mapping
    ────────────────────
    Representation       : Real-valued vector of green-time durations.
    Evaluation function  : Average simulated vehicle delay (lower = better).
    Population           : Swarm of `swarm_size` particles.
    Parent selection     : Neighbourhood topology (gbest = global, lbest = local ring).
    Variation operators  : (a) Velocity update strategy  — mutation-equivalent
                           (b) Crossover strategy         — recombination-equivalent
    Survivor selection   : pbest update strategy (greedy elitist vs SA-probabilistic).
    Diversity            : Crowding-based restart — restarted particles must be
                           placed at least `diversity_min_dist` away from gbest.
    Initialisation       : random | opposition-based.
    Termination          : max_iterations OR stagnation_limit consecutive
                           iterations without improvement.
    """

    def __init__(
        self,
        swarm_size=30,
        max_iterations=200,
        w_start=0.9,
        w_end=0.4,
        c1=2.0,
        c2=2.0,
        v_max_ratio=0.2,
        topology="gbest",
        neighbourhood_size=2,
        init_strategy="random",
        stagnation_limit=30,
        stagnation_fraction=0.3,

        # ── VARIATION OPERATOR A: velocity update (mutation-equivalent) ─────
        # "standard"    : classic inertia-weight PSO (Kennedy & Eberhart 1995)
        #                 v = w·v + c1·r1·(pbest-x) + c2·r2·(social-x)
        # "constriction": Clerc-Kennedy constriction factor (Clerc 1999)
        #                 v = chi·[v + c1·r1·(pbest-x) + c2·r2·(social-x)]
        #                 chi ≈ 0.729 guarantees convergence without hard clamp
        velocity_strategy="standard",

        # ── VARIATION OPERATOR B: crossover (recombination-equivalent) ──────
        # "none"        : no crossover — standard PSO behaviour
        # "arithmetic"  : when a stagnation restart fires, the restarted
        #                 particle's position is set to a random convex
        #                 combination of two randomly chosen pbests:
        #                 x_new = alpha*pbest_a + (1-alpha)*pbest_b
        #                 This injects crossover-like recombination of
        #                 existing good solutions into the restart step.
        crossover_strategy="none",

        # ── SURVIVOR SELECTION: pbest update policy ─────────────────────────
        # "greedy"       : accept new position only if it strictly improves pbest
        #                  (elitist — personal archive never degrades)
        # "probabilistic": SA-style Boltzmann acceptance; occasionally accepts
        #                  worse positions to escape local optima
        pbest_strategy="greedy",
        pbest_temperature=0.05,

        # ── DIVERSITY PRESERVATION: crowding-based restart ───────────────────
        # When a particle is restarted it is forced to land at least
        # `diversity_min_dist` (fraction of search range norm) away from gbest.
        # This prevents the swarm from collapsing onto the current attractor
        # and maintains phenotypic diversity in the population.
        # Set to 0.0 to disable (pure random restart — baseline for comparison).
        diversity_min_dist=0.0,
        diversity_max_attempts=10,
    ):
        self.swarm_size = swarm_size
        self.max_iterations = max_iterations
        self.w_start = w_start
        self.w_end = w_end
        self.c1 = c1
        self.c2 = c2
        self.v_max_ratio = v_max_ratio
        self.topology = topology
        self.neighbourhood_size = neighbourhood_size
        self.init_strategy = init_strategy
        self.stagnation_limit = stagnation_limit
        self.stagnation_fraction = stagnation_fraction
        self.velocity_strategy = velocity_strategy
        self.crossover_strategy = crossover_strategy
        self.pbest_strategy = pbest_strategy
        self.pbest_temperature = pbest_temperature
        self.diversity_min_dist = diversity_min_dist
        self.diversity_max_attempts = diversity_max_attempts


class RunResult:

    def __init__(self, best_fitness, best_solution,
                 convergence, seed, config, n_evaluations):

        self.best_fitness = best_fitness
        self.best_solution = best_solution
        self.convergence = convergence
        self.seed = seed
        self.config = config
        self.n_evaluations = n_evaluations


class PSO:

    def __init__(self, fitness_fn, lb, ub, config=None, repair_fn=None):

        self.f = fitness_fn
        self.lb = np.array(lb, dtype=float)
        self.ub = np.array(ub, dtype=float)

        self.cfg = config if config is not None else PSOConfig()

        if repair_fn is not None:
            self.repair = repair_fn
        else:
            self.repair = self._default_repair

        self.dim = len(self.lb)
        self.range_ = self.ub - self.lb

        # Pre-compute Clerc-Kennedy constriction factor
        # chi = 2 / |2 - phi - sqrt(phi^2 - 4*phi)|,  phi = c1 + c2 > 4
        phi = self.cfg.c1 + self.cfg.c2
        if phi <= 4.0:
            phi = 4.1
        self._chi = 2.0 / abs(2.0 - phi - (phi ** 2 - 4.0 * phi) ** 0.5)

        # Pre-compute the minimum Euclidean distance threshold for crowding
        # expressed in the same units as the search space norm.
        self._div_thresh = (
            self.cfg.diversity_min_dist * np.linalg.norm(self.range_)
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Main optimisation loop
    # ─────────────────────────────────────────────────────────────────────────

    def run(self, seed):

        cfg = self.cfg
        rng = np.random.default_rng(seed)

        X, V = self._init_swarm(rng)

        fitness = np.array([self.f(X[i]) for i in range(cfg.swarm_size)])
        n_eval = cfg.swarm_size

        pbest = X.copy()
        pbest_fit = fitness.copy()

        g_idx = np.argmin(pbest_fit)
        gbest = pbest[g_idx].copy()
        gbest_fit = pbest_fit[g_idx]

        if cfg.topology == "lbest":
            nbest, nbest_fit = self._lbest(pbest, pbest_fit)

        vmax = cfg.v_max_ratio * self.range_

        convergence = [gbest_fit]
        stagnation = 0
        prev_best = gbest_fit

        for t in range(1, cfg.max_iterations + 1):

            # Linear inertia weight decay (parameter control mechanism 1)
            w = cfg.w_start - (cfg.w_start - cfg.w_end) * t / cfg.max_iterations

            for i in range(cfg.swarm_size):

                r1 = rng.random(self.dim)
                r2 = rng.random(self.dim)

                social = gbest if cfg.topology == "gbest" else nbest[i]

                # ── VARIATION OPERATOR A: velocity update (mutation-equiv.) ──

                if cfg.velocity_strategy == "standard":
                    # Classic inertia-weight PSO (Kennedy & Eberhart 1995)
                    V[i] = (
                        w * V[i]
                        + cfg.c1 * r1 * (pbest[i] - X[i])
                        + cfg.c2 * r2 * (social   - X[i])
                    )
                    V[i] = np.clip(V[i], -vmax, vmax)

                else:  # "constriction"
                    # Constriction-factor PSO (Clerc 1999)
                    # chi ≈ 0.729 guarantees convergence without hard velocity clamp
                    V[i] = self._chi * (
                        V[i]
                        + cfg.c1 * r1 * (pbest[i] - X[i])
                        + cfg.c2 * r2 * (social   - X[i])
                    )
                    V[i] = np.clip(V[i], -vmax * 2, vmax * 2)

                X[i] = self.repair(X[i] + V[i])

                fit = self.f(X[i])
                n_eval += 1

                # ── SURVIVOR SELECTION: pbest update ─────────────────────────

                if cfg.pbest_strategy == "greedy":
                    # Elitist: accept only strict improvements.
                    # Personal archive never degrades → strong exploitation.
                    if fit < pbest_fit[i]:
                        pbest[i]     = X[i].copy()
                        pbest_fit[i] = fit

                else:  # "probabilistic"
                    # SA-style Boltzmann acceptance.
                    # Always accepts improvements; occasionally accepts worse
                    # positions to help particles escape local optima.
                    if fit < pbest_fit[i]:
                        pbest[i]     = X[i].copy()
                        pbest_fit[i] = fit
                    else:
                        delta = (fit - pbest_fit[i]) / (abs(pbest_fit[i]) + 1e-9)
                        prob  = np.exp(-delta / cfg.pbest_temperature)
                        if rng.random() < prob:
                            pbest[i]     = X[i].copy()
                            pbest_fit[i] = fit

                # Global best update (always greedy — tracks true optimum)
                if fit < gbest_fit:
                    gbest     = X[i].copy()
                    gbest_fit = fit

            if cfg.topology == "lbest":
                nbest, nbest_fit = self._lbest(pbest, pbest_fit)

            convergence.append(gbest_fit)

            # Stagnation detection
            if abs(gbest_fit - prev_best) < 1e-9:
                stagnation += 1
            else:
                stagnation = 0

            prev_best = gbest_fit

            if stagnation >= cfg.stagnation_limit:
                X, V = self._restart(X, V, gbest, pbest, rng)
                stagnation = 0

        return RunResult(
            gbest_fit,
            gbest,
            convergence,
            seed,
            self.cfg,
            n_eval
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _init_swarm(self, rng):

        cfg = self.cfg
        n   = cfg.swarm_size

        if cfg.init_strategy == "opposition":
            # Opposition-based initialisation (Tizhoosh 2005):
            # for each candidate x, also evaluate its opposite point
            # opp = lb + ub - x and keep whichever is better.
            X = []
            for _ in range(n):
                x   = rng.uniform(self.lb, self.ub)
                opp = np.clip(self.lb + self.ub - x, self.lb, self.ub)
                X.append(x if self.f(x) < self.f(opp) else opp)
            X = np.array(X)
        else:
            # Standard uniform random initialisation
            X = rng.uniform(self.lb, self.ub, (n, self.dim))

        V = rng.uniform(-self.range_ / 2, self.range_ / 2, (n, self.dim))
        return X, V

    def _lbest(self, pbest, pbest_fit):
        """
        Local best (ring topology) neighbourhood update.

        Each particle i looks at a ring of (2k+1) neighbours centred on i
        and adopts the best pbest in that neighbourhood as its social attractor.
        This slows information propagation vs gbest, preserving diversity and
        reducing the risk of premature convergence — functioning as the second
        parent-selection mechanism alongside the global gbest topology.
        """
        n = self.cfg.swarm_size
        k = self.cfg.neighbourhood_size

        nbest     = np.zeros_like(pbest)
        nbest_fit = np.zeros(n)

        for i in range(n):
            neighbors = [(i + j) % n for j in range(-k, k + 1)]
            best_idx  = min(neighbors, key=lambda idx: pbest_fit[idx])
            nbest[i]      = pbest[best_idx]
            nbest_fit[i]  = pbest_fit[best_idx]

        return nbest, nbest_fit

    def _restart(self, X, V, gbest, pbest, rng):
        """
        Stagnation-triggered restart combining two mechanisms:

        VARIATION OPERATOR B — arithmetic crossover (recombination-equivalent)
        ────────────────────────────────────────────────────────────────────────
        When crossover_strategy == "arithmetic", the new particle position is
        a random convex combination of two randomly selected pbests:

            x_new = alpha * pbest_a + (1 - alpha) * pbest_b,  alpha ~ U(0,1)

        This recombines genetic material from two good regions of the search
        space, analogous to arithmetic crossover in a GA.  Compare against
        crossover_strategy == "none" (pure random restart) to measure the
        effect of recombination independently.

        DIVERSITY PRESERVATION — crowding-based restart
        ────────────────────────────────────────────────
        When diversity_min_dist > 0, a restarted particle is placed only if
        it lands at least `_div_thresh` away from the current gbest.
        Up to `diversity_max_attempts` candidates are tried; the last one is
        kept even if below threshold to guarantee termination.
        This prevents the swarm from collapsing onto one attractor and is a
        lightweight form of crowding / niching.
        """
        cfg       = self.cfg
        n         = cfg.swarm_size
        n_restart = max(1, int(n * cfg.stagnation_fraction))

        # Identify the particles furthest from gbest (most stagnant)
        dist  = np.linalg.norm(X - gbest, axis=1)
        worst = np.argsort(dist)[-n_restart:]

        for i in worst:

            # ── VARIATION OPERATOR B: crossover ──────────────────────────────
            if cfg.crossover_strategy == "arithmetic" and len(pbest) >= 2:
                # Pick two distinct pbests at random
                idx_a, idx_b = rng.choice(n, size=2, replace=False)
                alpha        = rng.random()
                candidate    = alpha * pbest[idx_a] + (1.0 - alpha) * pbest[idx_b]
                candidate    = self.repair(candidate)
            else:
                # No crossover — pure random restart (baseline comparison)
                candidate = rng.uniform(self.lb, self.ub)

            # ── DIVERSITY: crowding check ─────────────────────────────────────
            if self._div_thresh > 0:
                for _ in range(cfg.diversity_max_attempts):
                    if np.linalg.norm(candidate - gbest) >= self._div_thresh:
                        break
                    # Re-sample until sufficiently distant (or give up)
                    if cfg.crossover_strategy == "arithmetic" and len(pbest) >= 2:
                        idx_a, idx_b = rng.choice(n, size=2, replace=False)
                        alpha        = rng.random()
                        candidate    = self.repair(
                            alpha * pbest[idx_a] + (1.0 - alpha) * pbest[idx_b]
                        )
                    else:
                        candidate = rng.uniform(self.lb, self.ub)

            X[i] = candidate
            V[i] = rng.uniform(-self.range_ / 2, self.range_ / 2)

        return X, V

    def _default_repair(self, x):
        """
        Repair operator (constraint-handling method):
        clips each decision variable to its feasible bounds [lb, ub].
        """
        return np.clip(x, self.lb, self.ub)
