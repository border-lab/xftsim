"""
Wealth-Education-Height Joint Architecture Simulation
(Reproducing Figure 4 from the xAM manuscript)

Parameters matched to ey_sim.py in the code supplement:
  https://github.com/border-lab/xftmanu_code_supplement

Models three traits with distinct transmission channels:
- Height: heritable (h²=0.60), no VT. Connects to other traits only via xAM.
- Education: low heritability (h²=0.01), VT from parental edu AND parental wealth.
- Wealth: NO genetic component. Pure vertical transmission from parental wealth.

Variance decomposition at generation 0 (paper targets):
- Height: 60% genetic, 40% noise
- Education: 1% genetic, ≈67% VT (from parental edu + wealth), ≈32% noise
- Wealth: ≈67% VT (from parental wealth), ≈33% noise

Vertical transmission (from ey_sim.py):
  The paper uses a LinearVerticalComponent with a transmission matrix:
      VT inputs: [edu_m, edu_f, wlth_m, wlth_f]
      edu_vert:    [CC, CC, CC, CC] * sqrt(0.5)     (CC = sqrt(1/3))
      height_vert: [0,  0,  0,  0 ]
      wealth_vert: [0,  0,  AA, AA] * sqrt(0.5)     (AA = sqrt(2/3))

  So each VT coefficient (per-parent) is:
      edu ← edu_parent:    CC * sqrt(0.5) = sqrt(1/6) ≈ 0.4082
      edu ← wealth_parent: CC * sqrt(0.5) = sqrt(1/6) ≈ 0.4082
      wealth ← wealth_parent: AA * sqrt(0.5) = sqrt(1/3) ≈ 0.5774

  The legacy LinearVerticalComponent with normalize=True divides by the
  std of the founder draws (which are N(0,1) since founder_variances=[1,1,1,1]).
  This means the effective per-parent coefficients are exactly as above.

Noise variances (from ey_sim.py):
  edu:    1/3 - h2_edu = 1/3 - 0.01 ≈ 0.3233
  height: 0.4
  wealth: 1/3 ≈ 0.3333

Mating regime:
  The paper uses GeneralAssortativeMatingRegime with a full cross-correlation matrix:
      [[0.246, 0.192, 0.252],
       [0.192, 0.125, 0.183],
       [0.252, 0.183, 0.251]]
  The new API only has LinearAssortativeMating (single scalar r). We use the
  mean of the cross-correlation matrix as the best approximation:
      mean ≈ 0.206
  This is an approximation — the paper's trait-specific correlations produce
  a different inflation pattern than a uniform scalar. Results will be
  qualitatively similar but not numerically identical to Figure 4.

Figure 4 target values (after 5 generations of xAM):
  Height: true h² 0.599 → 0.621  |  HE h² 0.599 → 0.686
  Edu:    true h² 0.010 → 0.005  |  HE h² 0.010 → 0.068
  Wealth: true h² 0.000 → 0.000  |  HE h² 0.000 → 0.036
"""

import numpy as np

from xftsim.founders import founder_haplotypes_uniform_AFs
from xftsim.neffect import AdditiveEffects
from xftsim.narch import Architecture
from xftsim.nmate import LinearAssortativeMating
from xftsim.reproduce import RecombinationMap
from xftsim.nsim import NSimulation
from xftsim.nstats import SampleStatistics, HasemanElstonEstimator

# ── Parameters (matched to ey_sim.py) ────────────────────────────────────

n_individuals = 128000
n_loci = 2000

# Heritabilities (only height and edu have genetic components)
h2_height = 0.60
h2_edu = 0.01

# VT coefficients from the paper's transmission matrix:
#   CC = sqrt(1/3), AA = sqrt(2/3), each multiplied by sqrt(0.5)
CC = np.sqrt(1 / 3)
AA = np.sqrt(2 / 3)
# Per-parent effective VT weight:
vt_edu_from_edu = CC * np.sqrt(0.5)      # ≈ 0.4082
vt_edu_from_wealth = CC * np.sqrt(0.5)   # ≈ 0.4082
vt_wealth_from_wealth = AA * np.sqrt(0.5)  # ≈ 0.5774

# Noise variances (from ey_sim.py)
var_edu = 1 / 3 - h2_edu    # ≈ 0.3233
var_height = 0.4
var_wealth = 1 / 3           # ≈ 0.3333

# Founder VT draw variance: the paper uses founder_variances = sqrt([1,1,1,1])
# which means N(0,1) draws. The normalize=True in LinearVerticalComponent
# divides by the observed std, so the effective founder variance is 1.0.
vt_founder_var = 1.0

# Mating: the paper uses a full cross-correlation matrix:
#   [[0.246, 0.192, 0.252],
#    [0.192, 0.125, 0.183],
#    [0.252, 0.183, 0.251]]
# Mean of all 9 entries ≈ 0.206
xmatecorr = np.array([[0.24626319, 0.1915851, 0.2521529],
                       [0.1915851, 0.1247212, 0.18323772],
                       [0.2521529, 0.18323772, 0.25072489]])
r_mean = float(np.mean(xmatecorr))

# Recombination rate (from ey_sim.py: p = 50/m)
p_recomb = 50 / n_loci


def build_ey_sim(n=n_individuals, m=n_loci, seed=42):
    """Build the education-height-wealth simulation matching ey_sim.py."""

    founder_haplotypes = founder_haplotypes_uniform_AFs(n=n, m=m)

    height_eff = AdditiveEffects.from_h2(h2=h2_height, m=m, seed=seed)
    edu_eff = AdditiveEffects.from_h2(h2=h2_edu, m=m, seed=seed + 1)

    formula = f"""
height.G ~ genetic(height_eff)
height.E ~ noise({var_height})
height ~ height.G + height.E

edu.G ~ genetic(edu_eff)
edu.E ~ noise({var_edu})
edu.VT_edu_m ~ mother(edu, founder=noise({vt_founder_var}))
edu.VT_edu_f ~ father(edu, founder=noise({vt_founder_var}))
edu.VT_wlth_m ~ mother(wealth, founder=noise({vt_founder_var}))
edu.VT_wlth_f ~ father(wealth, founder=noise({vt_founder_var}))
edu.VT ~ {vt_edu_from_edu} * edu.VT_edu_m + {vt_edu_from_edu} * edu.VT_edu_f + {vt_edu_from_wealth} * edu.VT_wlth_m + {vt_edu_from_wealth} * edu.VT_wlth_f
edu ~ edu.G + edu.E + edu.VT

wealth.E ~ noise({var_wealth})
wealth.VT_m ~ mother(wealth, founder=noise({vt_founder_var}))
wealth.VT_f ~ father(wealth, founder=noise({vt_founder_var}))
wealth.VT ~ {vt_wealth_from_wealth} * wealth.VT_m + {vt_wealth_from_wealth} * wealth.VT_f
wealth ~ wealth.E + wealth.VT
"""

    effects = {
        'height_eff': height_eff,
        'edu_eff': edu_eff,
    }
    arch = Architecture(formula=formula, effects=effects)

    rmap = RecombinationMap(p=p_recomb, m=m)
    mating = LinearAssortativeMating(
        component_names=['edu', 'height', 'wealth'],
        r=r_mean,
        offspring_per_pair=2,
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
            HasemanElstonEstimator(phenotype_keys=['edu', 'height', 'wealth']),
        ],
        seed=seed,
    )
    return sim


def extract_he_h2(sim, generation, trait):
    """Extract HE-estimated h² for a trait at a given generation."""
    for result in sim.results:
        if result.generation != generation:
            continue
        he = result.statistics.get('HasemanElstonEstimator')
        if he is None or trait not in he:
            raise ValueError(f"No HE result for '{trait}' at gen {generation}")
        return he[trait]['h2']
    raise ValueError(f"Generation {generation} not found in results")


def extract_he_rg(sim, generation):
    """Extract mean off-diagonal genetic correlation from HE at a generation."""
    for result in sim.results:
        if result.generation != generation:
            continue
        he = result.statistics.get('HasemanElstonEstimator')
        if he is None or '_cov_g' not in he:
            raise ValueError(f"No HE cov_g at generation {generation}")
        cov_g = he['_cov_g']
        k = cov_g.shape[0]
        d = np.sqrt(np.abs(np.diag(cov_g)))
        d[d < 1e-15] = 1.0
        rg = cov_g / np.outer(d, d)
        mask = np.triu(np.ones((k, k), dtype=bool), k=1)
        return rg, he['_keys']
    raise ValueError(f"Generation {generation} not found in results")


# ── Run as script ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    sim = build_ey_sim()
    sim.run(n_generations=6)
    print(f"Simulation complete. Final generation: {sim.generation}\n")

    traits = ['edu', 'height', 'wealth']

    print("=" * 90)
    print(f"{'Gen':>3}  |  {'--- Phenotypic Variance ---':^30}  |  {'--- HE-Estimated h² ---':^30}")
    print(f"{'':>3}  |  {'edu':>8}  {'height':>8}  {'wealth':>8}  |  {'edu':>8}  {'height':>8}  {'wealth':>8}")
    print("-" * 90)

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

    # HE genetic correlations
    print("\nHE-estimated genetic correlations:")
    for result in sim.results:
        gen = result.generation
        he = result.statistics.get('HasemanElstonEstimator')
        if he is None or '_cov_g' not in he:
            continue
        cov_g = he['_cov_g']
        d = np.sqrt(np.abs(np.diag(cov_g)))
        d[d < 1e-15] = 1.0
        rg = cov_g / np.outer(d, d)
        he_keys = he['_keys']
        pairs = []
        for i in range(len(he_keys)):
            for j in range(i + 1, len(he_keys)):
                pairs.append(f"rg({he_keys[i]},{he_keys[j]})={rg[i,j]:.3f}")
        print(f"  Gen {gen}: " + "  ".join(pairs))

    print("\n" + "=" * 90)
    print("Figure 4 target values (5 generations of xAM):")
    print("  Height: true h² 0.599 → 0.621   |  HE h² 0.599 → 0.686")
    print("  Edu:    true h² 0.010 → 0.005   |  HE h² 0.010 → 0.068")
    print("  Wealth: true h² 0.000 → 0.000   |  HE h² 0.000 → 0.036")
    print()
    print("NOTE: This simulation uses LinearAssortativeMating (single r={:.3f})".format(r_mean))
    print("      instead of GeneralAssortativeMatingRegime (full cross-correlation")
    print("      matrix). Results are qualitatively similar but not numerically")
    print("      identical to the paper. A GeneralAssortativeMating implementation")
    print("      in the new API would be needed for exact replication.")
