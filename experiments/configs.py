from __future__ import annotations
from src.optimization.pso import PSOConfig

# ─────────────────────────────────────────────────────────────────────────────
# Reproducible seed list  (30 seeds, generated once and fixed)
# ─────────────────────────────────────────────────────────────────────────────
SEEDS = [
    101, 202, 303, 404, 505, 606, 707, 808, 909, 1010,
    1111, 1212, 1313, 1414, 1515, 1616, 1717, 1818, 1919, 2020,
    2121, 2222, 2323, 2424, 2525, 2626, 2727, 2828, 2929, 3030,
]

# ─────────────────────────────────────────────────────────────────────────────
# Baseline reference config
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = PSOConfig(
    swarm_size          = 30,
    max_iterations      = 200,
    w_start             = 0.90,
    w_end               = 0.40,
    c1                  = 2.0,
    c2                  = 2.0,
    v_max_ratio         = 0.20,
    topology            = "gbest",
    init_strategy       = "random",
    stagnation_limit    = 30,
    stagnation_fraction = 0.30,
    velocity_strategy   = "standard",
    crossover_strategy  = "none",
    pbest_strategy      = "greedy",
    diversity_min_dist  = 0.0,
)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Swarm size
# ─────────────────────────────────────────────────────────────────────────────
SWARM_SIZE_CONFIGS: dict[str, PSOConfig] = {
    "n=10":  PSOConfig(swarm_size=10,  max_iterations=200),
    "n=20":  PSOConfig(swarm_size=20,  max_iterations=200),
    "n=30":  PSOConfig(swarm_size=30,  max_iterations=200),
    "n=50":  PSOConfig(swarm_size=50,  max_iterations=200),
    "n=100": PSOConfig(swarm_size=100, max_iterations=200),
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. Inertia weight
# ─────────────────────────────────────────────────────────────────────────────
INERTIA_CONFIGS: dict[str, PSOConfig] = {
    "w=0.4_to_0.1": PSOConfig(w_start=0.40, w_end=0.10),
    "w=0.7_to_0.3": PSOConfig(w_start=0.70, w_end=0.30),
    "w=0.9_to_0.4": PSOConfig(w_start=0.90, w_end=0.40),
    "w=1.0_to_0.5": PSOConfig(w_start=1.00, w_end=0.50),
}

# ─────────────────────────────────────────────────────────────────────────────
# 3. Cognitive / social coefficients
# ─────────────────────────────────────────────────────────────────────────────
C1C2_CONFIGS: dict[str, PSOConfig] = {
    "c1=1.5,c2=2.5": PSOConfig(c1=1.5, c2=2.5),
    "c1=2.0,c2=2.0": PSOConfig(c1=2.0, c2=2.0),
    "c1=2.5,c2=1.5": PSOConfig(c1=2.5, c2=1.5),
    "c1=2.0,c2=0.0": PSOConfig(c1=2.0, c2=0.0),   # cognitive only
    "c1=0.0,c2=2.0": PSOConfig(c1=0.0, c2=2.0),   # social only
}

# ─────────────────────────────────────────────────────────────────────────────
# 4. Parent selection — neighbourhood topology
#
#   gbest (global best): every particle sees the single best position in the
#           entire swarm → fast convergence, higher premature-convergence risk.
#
#   lbest (local best):  each particle sees only its k nearest ring-neighbours
#           → slower information propagation, better diversity retention.
#
# Guideline e mapping: these are the two PARENT SELECTION mechanisms.
# In PSO the "social attractor" chosen for the velocity update plays the role
# of the selected parent in a traditional EA.
# ─────────────────────────────────────────────────────────────────────────────
TOPOLOGY_CONFIGS: dict[str, PSOConfig] = {
    "gbest": PSOConfig(topology="gbest"),
    "lbest": PSOConfig(topology="lbest"),
}

# ─────────────────────────────────────────────────────────────────────────────
# 5. Initialisation strategy  (bonus: ≥2 initialisations)
# ─────────────────────────────────────────────────────────────────────────────
INIT_CONFIGS: dict[str, PSOConfig] = {
    "random":     PSOConfig(init_strategy="random"),
    "opposition": PSOConfig(init_strategy="opposition"),
}

# ─────────────────────────────────────────────────────────────────────────────
# 6. Iteration budget
# ─────────────────────────────────────────────────────────────────────────────
ITER_CONFIGS: dict[str, PSOConfig] = {
    "iter=50":  PSOConfig(max_iterations=50),
    "iter=100": PSOConfig(max_iterations=100),
    "iter=200": PSOConfig(max_iterations=200),
    "iter=500": PSOConfig(max_iterations=500),
}

# ─────────────────────────────────────────────────────────────────────────────
# 7. Variation operator A — velocity update strategy  (mutation-equivalent)
#
#   standard      — classic inertia-weight PSO (Kennedy & Eberhart 1995)
#   constriction  — Clerc-Kennedy constriction factor (Clerc 1999)
#
# Guideline e mapping: these are the two MUTATION operators.
# ─────────────────────────────────────────────────────────────────────────────
VELOCITY_STRATEGY_CONFIGS: dict[str, PSOConfig] = {
    "standard":     PSOConfig(velocity_strategy="standard"),
    "constriction": PSOConfig(velocity_strategy="constriction"),
}

# ─────────────────────────────────────────────────────────────────────────────
# 8. Variation operator B — crossover strategy  (recombination-equivalent)
#
#   none        — no crossover; standard PSO restart places particle randomly.
#                 Baseline condition: recombination has zero influence.
#
#   arithmetic  — on stagnation restart, the new particle position is set to
#                 a random convex combination of two randomly chosen pbests:
#                 x_new = alpha * pbest_a + (1-alpha) * pbest_b
#                 This blends good regions of the search space, analogous to
#                 whole arithmetic crossover in a real-coded GA.
#
# Guideline e mapping: these are the two RECOMBINATION operators.
# Running them under identical hyper-parameters isolates the crossover effect.
# ─────────────────────────────────────────────────────────────────────────────
CROSSOVER_CONFIGS: dict[str, PSOConfig] = {
    "no_crossover": PSOConfig(crossover_strategy="none"),
    "arithmetic":   PSOConfig(crossover_strategy="arithmetic"),
}

# ─────────────────────────────────────────────────────────────────────────────
# 9. Survivor selection — pbest update strategy
#
#   greedy         — accept new position only when it strictly improves pbest.
#                    Deterministic and elitist: the personal archive never
#                    degrades.  Equivalent to (μ+λ) elitist replacement in a GA.
#
#   prob_T=0.01/   — SA-style Boltzmann acceptance: always accept improvements;
#   prob_T=0.05/     also accept worse positions with probability
#   prob_T=0.20      P = exp(-delta / T).  Higher T = more acceptance of bad
#                    moves (exploration); lower T ≈ greedy (exploitation).
#                    Equivalent to a probabilistic replacement policy.
#
# Guideline f mapping: these are the two (or more) SURVIVOR SELECTION /
# POPULATION MANAGEMENT models tested independently over 30 seeds each.
# ─────────────────────────────────────────────────────────────────────────────
SURVIVOR_SELECTION_CONFIGS: dict[str, PSOConfig] = {
    "greedy":      PSOConfig(pbest_strategy="greedy"),
    "prob_T=0.01": PSOConfig(pbest_strategy="probabilistic", pbest_temperature=0.01),
    "prob_T=0.05": PSOConfig(pbest_strategy="probabilistic", pbest_temperature=0.05),
    "prob_T=0.20": PSOConfig(pbest_strategy="probabilistic", pbest_temperature=0.20),
}

# ─────────────────────────────────────────────────────────────────────────────
# 10. Diversity preservation — crowding-based restart
#
#   no_crowding    — restarted particle placed at a purely random location.
#                    Baseline: no explicit diversity mechanism.
#
#   crowding_0.10  — restarted particle must land ≥ 10 % of the search-space
#   crowding_0.20    norm away from the current gbest.  If a candidate is too
#   crowding_0.30    close it is resampled (up to 10 attempts).  This prevents
#                    particles from collapsing back onto the current attractor
#                    and maintains phenotypic diversity — a lightweight form of
#                    niching / crowding as required by guideline h.
#
# Guideline h mapping: crowding_* variants implement DIVERSITY PRESERVATION.
# ─────────────────────────────────────────────────────────────────────────────
DIVERSITY_CONFIGS: dict[str, PSOConfig] = {
    "no_crowding":   PSOConfig(diversity_min_dist=0.0),
    "crowding_0.10": PSOConfig(diversity_min_dist=0.10),
    "crowding_0.20": PSOConfig(diversity_min_dist=0.20),
    "crowding_0.30": PSOConfig(diversity_min_dist=0.30),
}

# ─────────────────────────────────────────────────────────────────────────────
# GA experiment suites
# Each suite varies a single GA component while holding everything else at its
# default so the effect of that component can be isolated over 30 seeds.
# ─────────────────────────────────────────────────────────────────────────────
from src.optimization.ga import GAConfig  # noqa: E402  (placed here to avoid circular)

# 11. GA — Mutation operator  (guideline e: ≥2 mutation techniques)
GA_MUTATION_CONFIGS: dict[str, GAConfig] = {
    "uniform":    GAConfig(mutation="uniform"),
    "nonuniform": GAConfig(mutation="nonuniform", nonuniform_b=2.0),
}

# 12. GA — Crossover operator  (guideline e: ≥2 recombination techniques)
GA_CROSSOVER_CONFIGS: dict[str, GAConfig] = {
    "arithmetic": GAConfig(crossover="arithmetic"),
    "blend_0.3":  GAConfig(crossover="blend", blend_alpha=0.3),
    "blend_0.5":  GAConfig(crossover="blend", blend_alpha=0.5),
    "blend_0.7":  GAConfig(crossover="blend", blend_alpha=0.7),
}

# 13. GA — Parent selection  (guideline e: ≥2 parent selection techniques)
GA_SELECTION_CONFIGS: dict[str, GAConfig] = {
    "tournament_k2": GAConfig(selection="tournament", tournament_size=2),
    "tournament_k3": GAConfig(selection="tournament", tournament_size=3),
    "tournament_k5": GAConfig(selection="tournament", tournament_size=5),
    "roulette":      GAConfig(selection="roulette"),
}

# 14. GA — Survivor selection  (guideline f: ≥2 population management models)
GA_REPLACEMENT_CONFIGS: dict[str, GAConfig] = {
    "generational": GAConfig(replacement="generational", elite_count=1),
    "elitist_mu+lambda": GAConfig(replacement="elitist"),
}

# ─────────────────────────────────────────────────────────────────────────────
# Master list for the full experiment suite
# ─────────────────────────────────────────────────────────────────────────────
ALL_SUITES = {
    # ── PSO suites ────────────────────────────────────────────────────────────
    "swarm_size":        SWARM_SIZE_CONFIGS,
    "inertia_weight":    INERTIA_CONFIGS,
    "c1_c2":             C1C2_CONFIGS,
    "topology":          TOPOLOGY_CONFIGS,           # PSO parent selection
    "init_strategy":     INIT_CONFIGS,
    "iterations_budget": ITER_CONFIGS,
    "velocity_strategy": VELOCITY_STRATEGY_CONFIGS,  # PSO mutation
    "crossover":         CROSSOVER_CONFIGS,           # PSO recombination
    "survivor_selection": SURVIVOR_SELECTION_CONFIGS, # PSO survivor selection
    "diversity":          DIVERSITY_CONFIGS,           # PSO diversity
    # ── GA suites ─────────────────────────────────────────────────────────────
    "ga_mutation":    GA_MUTATION_CONFIGS,    # guideline e — mutation
    "ga_crossover":   GA_CROSSOVER_CONFIGS,   # guideline e — recombination
    "ga_selection":   GA_SELECTION_CONFIGS,   # guideline e — parent selection
    "ga_replacement": GA_REPLACEMENT_CONFIGS, # guideline f — survivor selection
}

