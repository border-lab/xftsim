"""
Wealth-Education-Height Joint Architecture Simulation
(Reproducing Figure 4 from the xAM manuscript)

Models three traits with distinct transmission channels:
- Height: heritable (h²=0.60), no VT. Connects to other traits only via xAM.
- Education: low heritability (h²=0.01), VT from parental edu AND parental wealth.
- Wealth: NO genetic component. Pure vertical transmission from parental wealth.

Parameters calibrated so that at generation 0:
- Height: 60% genetic, 40% noise (total variance = 1.0)
- Education: 1% genetic, 67% VT (from parental edu + wealth), 32% noise
- Wealth: 67% VT (from parental wealth), 33% noise
"""

import numpy as np

from xftsim.founders import founder_haplotypes_uniform_AFs
from xftsim.neffect import AdditiveEffects
from xftsim.narch import Architecture
from xftsim.nmate import GeneralAssortativeMating, BatchedMating
from xftsim.reproduce import RecombinationMap
from xftsim.nsim import NSimulation
from xftsim.nstats import SampleStatistics, HasemanElstonEstimator

# ── Parameters ──────────────────────────────────────────────────────────────

n_individuals = 8000
n_loci = 1000

# Heritabilities (only height and edu have genetic components)
h2_height = 0.60
h2_edu = 0.01

# VT coefficients (per-parent)

vt_edu_founder_var = 0.20
vt_wlth_founder_var = 0.15

vt_edu_self = 0.22    # parent edu    → offspring edu
vt_wlth_edu = 0.18    # parent wealth → offspring edu  (cross-trait)
vt_wlth_self = 0.20   # parent wealth → offspring wealth

# Residual (environmental noise) variances
# Chosen so gen-0 total variance is close to 1.0
# height: h2=0.60 + noise=0.40 = 1.0
# edu:    h2=0.01 + VT + noise ≈ 1.0
# wealth: VT + noise ≈ 1.0
var_height = 0.40
var_edu = 0.79
var_wealth = 0.88

# ── Founder haplotypes ─────────────────────────────────────────────────────

founder_haplotypes = founder_haplotypes_uniform_AFs(n=n_individuals, m=n_loci)

# ── Genetic effects (height and edu only — wealth has none) ────────────────

height_eff = AdditiveEffects.from_h2(h2=h2_height, m=n_loci, seed=1)
edu_eff = AdditiveEffects.from_h2(h2=h2_edu, m=n_loci, seed=2)

# ── Architecture (formula DSL) ─────────────────────────────────────────────

formula = f"""

height.G ~ genetic(height_eff)
height.E ~ noise({var_height})
height ~ height.G + height.E

edu.G ~ genetic(edu_eff)
edu.E ~ noise({var_edu})
edu.VT_edu_m ~ mother(edu, founder=noise({vt_edu_founder_var}))
edu.VT_edu_f ~ father(edu, founder=noise({vt_edu_founder_var}))
edu.VT_wlth_m ~ mother(wealth, founder=noise({vt_wlth_founder_var}))
edu.VT_wlth_f ~ father(wealth, founder=noise({vt_wlth_founder_var}))
edu.VT ~ {vt_edu_self} * edu.VT_edu_m + {vt_edu_self} * edu.VT_edu_f + {vt_wlth_edu} * edu.VT_wlth_m + {vt_wlth_edu} * edu.VT_wlth_f
edu ~ edu.G + edu.E + edu.VT

wealth.E ~ noise({var_wealth})
wealth.VT_m ~ mother(wealth, founder=noise({vt_wlth_founder_var}))
wealth.VT_f ~ father(wealth, founder=noise({vt_wlth_founder_var}))
wealth.VT ~ {vt_wlth_self} * wealth.VT_m + {vt_wlth_self} * wealth.VT_f
wealth ~ wealth.E + wealth.VT
"""

effects = {
    'height_eff': height_eff,
    'edu_eff': edu_eff,
}
arch = Architecture(formula=formula, effects=effects)


rmap = RecombinationMap(p=0.5, m=n_loci)

# Cross-mate correlation matrix from the manuscript (Figure 4)
# Rows/cols: [height, edu, wealth]
cross_corr = np.array([
    [0.246, 0.192, 0.252],
    [0.192, 0.125, 0.183],
    [0.252, 0.183, 0.251],
])

mating = BatchedMating(
    regime=GeneralAssortativeMating(
        component_names=['height', 'edu', 'wealth'],
        cross_corr=cross_corr,
        offspring_per_pair=2,
        solver_params=dict(
            time_limit=30,
            termination_interval=5,
            tolerance=1e-3,
        ),
    ),
    max_batch_size=1000,
)


sim = NSimulation(
    founder_haplotypes=founder_haplotypes,
    architecture=arch,
    mating_regime=mating,
    recombination_map=rmap,
    retain_haplotypes=1,
    retain_phenotypes=2,
    statistics=[
        SampleStatistics(),
        HasemanElstonEstimator(phenotype_keys=['height', 'edu', 'wealth']),
    ],
    seed=42,
)

n_generations = 6
sim.run(n_generations=n_generations)
print(f"Simulation complete. Final generation: {sim.generation}\n")

# ── Results: True variance components + HE-estimated h² ───────────────────

traits = ['height', 'edu', 'wealth']

print("=" * 90)
print(f"{'Gen':>3}  |  {'--- Phenotypic Variance ---':^30}  |  {'--- HE-Estimated h² ---':^30}")
print(f"{'':>3}  |  {'height':>8}  {'edu':>8}  {'wealth':>8}  |  {'height':>8}  {'edu':>8}  {'wealth':>8}")
print("-" * 90)

for result in sim.results:
    stats = result.statistics
    gen = result.generation

    # Phenotypic variances from SampleStatistics
    ss = stats.get('SampleStatistics')
    pheno_vars = {}
    if ss:
        keys = ss['keys']
        var = ss['var']
        for t in traits:
            if t in keys:
                pheno_vars[t] = var[keys.index(t)]

    # HE-estimated h² from HasemanElstonEstimator
    he = stats.get('HasemanElstonEstimator')
    he_h2 = {}
    if he:
        for t in traits:
            if t in he:
                he_h2[t] = he[t]['h2']

    pv = "  ".join(f"{pheno_vars.get(t, float('nan')):8.4f}" for t in traits)
    hv = "  ".join(f"{he_h2.get(t, float('nan')):8.4f}" for t in traits)
    print(f"{gen:3d}  |  {pv}  |  {hv}")

print("=" * 90)

# ── Summary comparison with Figure 4 ──────────────────────────────────────

print("\nFigure 4 comparison (paper values after 5 generations of xAM):")
print("  Height true h²: 0.599 → 0.621   |  HE-estimated h²: 0.599 → 0.686")
print("  Edu    true h²: 0.010 → 0.005   |  HE-estimated h²: 0.010 → 0.068")
print("  Wealth true h²: 0.000 → 0.000   |  HE-estimated h²: 0.000 → 0.036")
