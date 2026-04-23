"""

The architecture is built programmatically (bypassing the formula DSL) so
that the parental-wealth lookup nodes are SHARED between edu.VT and
wealth.VT. The formula DSL would create independent founder draws per
mother()/father() call; legacy ey_sim.py stores parental wealth in a
single column (input_cindex vert_input) that both edu.vert and wealth.vert
read. Sharing here restores that behavior — the gen-0 founder draws for
parental wealth are the same draws seen by both aggregations, so the
expected gen-0 corr(edu.vert, wealth.vert) matches the paper.

NOTE: The new nstats.HasemanElstonEstimator runs on the full sample (no
RandomSiblingSubsampleFilter equivalent). ey_sim.py filters to a k=4000
sibling subsample before HE; HE estimates here will be on the full
unfiltered population, which under xAM slightly inflates HE h² vs. the
paper values.
"""

import numpy as np

from xftsim.founders import founder_haplotypes_uniform_AFs
from xftsim.neffect import AdditiveEffects
from xftsim.narch import (
    Architecture,
    GeneticComponent,
    NoiseComponent,
    MotherComponent,
    FatherComponent,
    AggregationComponent,
)
from xftsim.nmate import GeneralAssortativeMating, BatchedMating
from xftsim.reproduce import RecombinationMap
from xftsim.nsim import NSimulation
from xftsim.nstats import SampleStatistics, HasemanElstonEstimator

# ── Parameters (matched to ey_sim.py) ──────────────────────────────────────

n_individuals = 20_000
n_loci = 1_000
n_generations = 6           # gens 0..5  ⇒  "after 5 generations of xAM"

h2_height = 0.60
h2_edu = 0.01

# VT coefficients from the paper's transmission matrix
CC = np.sqrt(1.0 / 3.0)
AA = np.sqrt(2.0 / 3.0)
vt_edu_from_edu    = CC * np.sqrt(0.5)      # ≈ 0.4082
vt_edu_from_wealth = CC * np.sqrt(0.5)      # ≈ 0.4082
vt_wealth_from_wealth = AA * np.sqrt(0.5)   # ≈ 0.5774

# Founder variance for VT sources (ey_sim.py: founder_variances = [1,1,1,1])
vt_founder_var = 1.0

# Residual noise variances (ey_sim.py: [1/3 - H2, 0.4, 1/3])
var_edu = 1.0 / 3.0 - h2_edu
var_height = 0.40
var_wealth = 1.0 / 3.0

# Recombination probability (ey_sim.py: p = 50/m)
p_recomb = 50.0 / n_loci

# ── Founders and genetic effects ───────────────────────────────────────────

founder_haplotypes = founder_haplotypes_uniform_AFs(n=n_individuals, m=n_loci)

height_eff = AdditiveEffects.from_h2(h2=h2_height, m=n_loci, seed=1)
edu_eff = AdditiveEffects.from_h2(h2=h2_edu, m=n_loci, seed=2)

# ── Architecture (programmatic, with shared parental-wealth lookups) ───────

arch = Architecture()

# Genetic components (only height and edu have additive genetic effects)
arch.add('height.G', GeneticComponent(height_eff))
arch.add('edu.G', GeneticComponent(edu_eff))

# Residual noise
arch.add('height.E', NoiseComponent(var_height))
arch.add('edu.E', NoiseComponent(var_edu))
arch.add('wealth.E', NoiseComponent(var_wealth))

# SHARED parental-wealth lookups — computed ONCE per generation,
# referenced by both edu.VT (cross-trait term) and wealth.VT.
# Matches legacy LinearVerticalComponent behavior where parental
# wealth lives in a single stored column that both outputs read.
arch.add('wealth.VT_m', MotherComponent(
    'wealth',
    founder_component=NoiseComponent(vt_founder_var),
    normalize=True,
))
arch.add('wealth.VT_f', FatherComponent(
    'wealth',
    founder_component=NoiseComponent(vt_founder_var),
    normalize=True,
))

# Parental-edu lookups (used only by edu.VT — no sharing needed)
arch.add('edu.VT_m', MotherComponent(
    'edu',
    founder_component=NoiseComponent(vt_founder_var),
    normalize=True,
))
arch.add('edu.VT_f', FatherComponent(
    'edu',
    founder_component=NoiseComponent(vt_founder_var),
    normalize=True,
))

# edu.VT = CC·√0.5 × (edu.VT_m + edu.VT_f + wealth.VT_m + wealth.VT_f)
arch.add('edu.VT', AggregationComponent(
    f'{vt_edu_from_edu} * edu.VT_m + {vt_edu_from_edu} * edu.VT_f + '
    f'{vt_edu_from_wealth} * wealth.VT_m + {vt_edu_from_wealth} * wealth.VT_f'
))

# wealth.VT = AA·√0.5 × (wealth.VT_m + wealth.VT_f)  — reads the SAME
# two nodes as edu.VT, so gen-0 founder draws are shared.
arch.add('wealth.VT', AggregationComponent(
    f'{vt_wealth_from_wealth} * wealth.VT_m + {vt_wealth_from_wealth} * wealth.VT_f'
))

# Final trait phenotypes
arch.add('height', AggregationComponent('height.G + height.E'))
arch.add('edu',    AggregationComponent('edu.G + edu.E + edu.VT'))
arch.add('wealth', AggregationComponent('wealth.E + wealth.VT'))

# ── Recombination and mating ───────────────────────────────────────────────

rmap = RecombinationMap(p=p_recomb, m=n_loci)

# Cross-mate correlation matrix from ey_sim.py (xmatecorr), indexed as
# [edu, height, wealth]. Keeping that ordering here.
cross_corr = np.array([
    [0.24626319, 0.19158510, 0.25215290],   # edu    × {edu, height, wealth}
    [0.19158510, 0.12472120, 0.18323772],   # height × {edu, height, wealth}
    [0.25215290, 0.18323772, 0.25072489],   # wealth × {edu, height, wealth}
])

mating = BatchedMating(
    regime=GeneralAssortativeMating(
        component_names=['edu', 'height', 'wealth'],
        cross_corr=cross_corr,
        offspring_per_pair=2,
        solver_params=dict(
            nb_threads=8,
            time_limit=10,
            tolerance=1e-3,
            time_between_displays=5,
            termination_interval=5,
        ),
    ),
    max_batch_size=250,
)

# ── Simulation ─────────────────────────────────────────────────────────────

sim = NSimulation(
    founder_haplotypes=founder_haplotypes,
    architecture=arch,
    mating_regime=mating,
    recombination_map=rmap,
    retain_haplotypes=n_generations + 1,
    retain_phenotypes=n_generations + 1,
    statistics=[
        SampleStatistics(),
        HasemanElstonEstimator(phenotype_keys=['height', 'edu', 'wealth']),
    ],
    seed=42,
)

sim.run(n_generations=n_generations)
print(f"Simulation complete. Final generation: {sim.generation}\n")

# ── Post-hoc sibship-free HE on one-per-family subsample ──────────────────

def he_h2_unrelated(hap, pheno, keys):
    """Run GRM-based HE on one individual per FID.

    Mirrors HasemanElstonEstimator but first subsets to unrelated
    individuals (one per family) to remove VT-induced family
    relatedness that would otherwise inflate HE h² under xAM —
    matches ey_sim.py's RandomSiblingSubsampleFilter + HE pipeline.
    """
    fids = hap.samples.fid
    _, first_idx = np.unique(fids, return_index=True)
    keep = np.sort(first_idx)
    hap_sub = hap.subset(sample_idx=keep)

    G = hap_sub.to_diploid_standardized(scale=True).astype(np.float64)
    n, m = G.shape

    Y = np.column_stack([pheno[k][keep] for k in keys]).astype(np.float64)
    Y = (Y - Y.mean(0)) / np.where(Y.std(0) < 1e-15, 1.0, Y.std(0))

    GtY = G.T @ Y
    KY = G @ GtY / m
    GtG = G.T @ G
    trK2 = float(np.sum(GtG ** 2) / (m * m))
    denom = trK2 - n
    if abs(denom) < 1e-15:
        return {k: float('nan') for k in keys}
    cov_g = Y.T @ (KY - Y) / denom
    return {k: float(cov_g[i, i]) for i, k in enumerate(keys)}


# ── Results: True variance components + HE-estimated h² ───────────────────

traits = ['height', 'edu', 'wealth']
genetic_traits = ['height', 'edu']

print("=" * 138)
print(f"{'Gen':>3}  |  {'--- Phenotypic Variance ---':^30}  |  "
      f"{'--- True h² ---':^30}  |  {'--- HE h² (full) ---':^30}  |  "
      f"{'--- HE h² (1/family) ---':^30}")
print(f"{'':>3}  |  {'height':>8}  {'edu':>8}  {'wealth':>8}  |  "
      f"{'height':>8}  {'edu':>8}  {'wealth':>8}  |  "
      f"{'height':>8}  {'edu':>8}  {'wealth':>8}  |  "
      f"{'height':>8}  {'edu':>8}  {'wealth':>8}")
print("-" * 138)

for result in sim.results:
    stats = result.statistics
    gen = result.generation

    ss = stats.get('SampleStatistics')
    pheno_vars = {}
    if ss:
        keys = ss['keys']
        var = ss['var']
        for t in traits:
            if t in keys:
                pheno_vars[t] = var[keys.index(t)]

    pheno = sim.phenotype_history.get(gen)
    true_h2 = {}
    if pheno is not None:
        for t in traits:
            var_tot = np.var(pheno[t])
            if t in genetic_traits and var_tot > 0:
                true_h2[t] = float(np.var(pheno[f'{t}.G']) / var_tot)
            else:
                true_h2[t] = 0.0

    he = stats.get('HasemanElstonEstimator')
    he_h2 = {}
    if he:
        for t in traits:
            if t in he:
                he_h2[t] = he[t]['h2']

    hap = sim.haplotype_history.get(gen)
    he_unrel = {}
    if hap is not None and pheno is not None:
        he_unrel = he_h2_unrelated(hap, pheno, traits)

    pv = "  ".join(f"{pheno_vars.get(t, float('nan')):8.4f}" for t in traits)
    tv = "  ".join(f"{true_h2.get(t, float('nan')):8.4f}"  for t in traits)
    hv = "  ".join(f"{he_h2.get(t, float('nan')):8.4f}"    for t in traits)
    hu = "  ".join(f"{he_unrel.get(t, float('nan')):8.4f}" for t in traits)
    print(f"{gen:3d}  |  {pv}  |  {tv}  |  {hv}  |  {hu}")

print("=" * 138)

print("\nFigure 4 comparison (paper values after 5 generations of xAM):")
print("  Height true h²: 0.599 → 0.621   |  HE-estimated h²: 0.599 → 0.686")
print("  Edu    true h²: 0.010 → 0.005   |  HE-estimated h²: 0.010 → 0.068")
print("  Wealth true h²: 0.000 → 0.000   |  HE-estimated h²: 0.000 → 0.036")
