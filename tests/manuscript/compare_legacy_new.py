"""
New API vs paper's Figure 4 target values.

Runs the edu-height-wealth simulation with parameters matched to ey_sim.py,
using the new API (NSimulation, Architecture formula DSL, etc.), and compares
against the published Figure 4 results.

NOTE: The paper uses GeneralAssortativeMatingRegime (full cross-correlation
matrix). The new API only has LinearAssortativeMating (single scalar r).
We use r = mean(xmatecorr) ≈ 0.206 as an approximation. Results should be
qualitatively similar but not numerically identical.
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

N = 16000
M = 1000
SEED = 42
N_GEN = 6

h2_height = 0.60
h2_edu = 0.01

CC = np.sqrt(1 / 3)
AA = np.sqrt(2 / 3)

var_edu = 1 / 3 - h2_edu
var_height = 0.4
var_wealth = 1 / 3

vt_edu_from_edu = CC * np.sqrt(0.5)
vt_edu_from_wealth = CC * np.sqrt(0.5)
vt_wealth_from_wealth = AA * np.sqrt(0.5)

xmatecorr = np.array([[0.24626319, 0.1915851, 0.2521529],
                       [0.1915851, 0.1247212, 0.18323772],
                       [0.2521529, 0.18323772, 0.25072489]])
r_mate = float(np.mean(xmatecorr))

p_recomb = 50 / M

traits = ['edu', 'height', 'wealth']

# ── Paper's Figure 4 target values ────────────────────────────────────────
# These are from the GeneralAssortativeMatingRegime runs (ey_sim.py)
# with N=128K, M=2000. Our approximation uses LinearAssortativeMating
# with smaller N/M, so we expect qualitative but not exact agreement.

PAPER_TARGETS = {
    # (gen 0 → gen 5)
    'h2_true': {'height': (0.599, 0.621), 'edu': (0.010, 0.005), 'wealth': (0.0, 0.0)},
    'h2_HE':  {'height': (0.599, 0.686), 'edu': (0.010, 0.068), 'wealth': (0.0, 0.036)},
}


def run_simulation():
    founders = founder_haplotypes_uniform_AFs(n=N, m=M)
    height_eff = AdditiveEffects.from_h2(h2=h2_height, m=M, seed=SEED)
    edu_eff = AdditiveEffects.from_h2(h2=h2_edu, m=M, seed=SEED + 1)

    formula = f"""
height.G ~ genetic(height_eff)
height.E ~ noise({var_height})
height ~ height.G + height.E

edu.G ~ genetic(edu_eff)
edu.E ~ noise({var_edu})
edu.VT_edu_m ~ mother(edu, founder=noise(1.0), normalize=true)
edu.VT_edu_f ~ father(edu, founder=noise(1.0), normalize=true)
edu.VT_wlth_m ~ mother(wealth, founder=noise(1.0), normalize=true)
edu.VT_wlth_f ~ father(wealth, founder=noise(1.0), normalize=true)
edu.VT ~ {vt_edu_from_edu} * edu.VT_edu_m + {vt_edu_from_edu} * edu.VT_edu_f + {vt_edu_from_wealth} * edu.VT_wlth_m + {vt_edu_from_wealth} * edu.VT_wlth_f
edu ~ edu.G + edu.E + edu.VT

wealth.E ~ noise({var_wealth})
wealth.VT_m ~ mother(wealth, founder=noise(1.0), normalize=true)
wealth.VT_f ~ father(wealth, founder=noise(1.0), normalize=true)
wealth.VT ~ {vt_wealth_from_wealth} * wealth.VT_m + {vt_wealth_from_wealth} * wealth.VT_f
wealth ~ wealth.E + wealth.VT
"""

    effects = {'height_eff': height_eff, 'edu_eff': edu_eff}
    arch = Architecture(formula=formula, effects=effects)
    rmap = RecombinationMap(p=p_recomb, m=M)
    mating = LinearAssortativeMating(
        component_names=['edu', 'height', 'wealth'],
        r=r_mate,
        offspring_per_pair=2,
    )

    sim = NSimulation(
        founder_haplotypes=founders,
        architecture=arch,
        mating_regime=mating,
        recombination_map=rmap,
        retain_haplotypes=1,
        retain_phenotypes=2,
        statistics=[
            SampleStatistics(),
            HasemanElstonEstimator(phenotype_keys=['edu', 'height', 'wealth']),
        ],
        seed=SEED,
    )
    sim.run(n_generations=N_GEN)
    return sim


def print_results(sim):
    print(f"Parameters: N={N}, M={M}, seed={SEED}, r_mate={r_mate:.3f}, {N_GEN} generations")
    print(f"h2_edu={h2_edu}, h2_height={h2_height}, h2_wealth=0")
    print()

    # ── Combined table: Phenotypic Variance + HE h² ──────────────────────
    print("=" * 100)
    print(f"{'Gen':>3}  |  {'--- Phenotypic Variance ---':^30}  |  {'--- HE-Estimated h² ---':^30}")
    print(f"{'':>3}  |  {'edu':>8}  {'height':>8}  {'wealth':>8}  |  {'edu':>8}  {'height':>8}  {'wealth':>8}")
    print("-" * 100)

    for result in sim.results:
        gen = result.generation
        ss = result.statistics.get('SampleStatistics')
        he = result.statistics.get('HasemanElstonEstimator')

        pv_vals = []
        if ss:
            keys = ss['keys']
            for t in traits:
                if t in keys:
                    pv_vals.append(f"{ss['var'][keys.index(t)]:8.4f}")
                else:
                    pv_vals.append(f"{'nan':>8}")
        else:
            pv_vals = [f"{'nan':>8}"] * 3

        he_vals = []
        if he:
            for t in traits:
                if t in he:
                    he_vals.append(f"{he[t]['h2']:8.4f}")
                else:
                    he_vals.append(f"{'nan':>8}")
        else:
            he_vals = [f"{'nan':>8}"] * 3

        print(f"{gen:3d}  |  {'  '.join(pv_vals)}  |  {'  '.join(he_vals)}")

    print("=" * 100)

    # ── Paper targets ─────────────────────────────────────────────────────
    print()
    print("Figure 4 targets (paper, gen 0 → gen 5):")
    pt_true = PAPER_TARGETS['h2_true']
    pt_he = PAPER_TARGETS['h2_HE']
    print(f"  {'':>10}  {'true h²':>18}  {'HE h²':>18}")
    for t in traits:
        t0_true, t5_true = pt_true[t]
        t0_he, t5_he = pt_he[t]
        print(f"  {t:>10}  {t0_true:.3f} → {t5_true:.3f}          {t0_he:.3f} → {t5_he:.3f}")

    # ── HE genetic correlations ───────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"{'Gen':>3}  |  {'--- HE Genetic Correlations ---':^50}")
    print(f"{'':>3}  |  {'edu-hgt':>14}  {'edu-wlth':>14}  {'hgt-wlth':>14}")
    print("-" * 60)

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
                pairs.append(f"{rg[i, j]:14.4f}")
        print(f"{gen:3d}  |  {'  '.join(pairs)}")
    print("=" * 60)

    # ── Assessment ────────────────────────────────────────────────────────
    print()
    last_gen = N_GEN - 1
    he_last = None
    for result in sim.results:
        if result.generation == last_gen:
            he_last = result.statistics.get('HasemanElstonEstimator')
    he_gen0 = None
    for result in sim.results:
        if result.generation == 0:
            he_gen0 = result.statistics.get('HasemanElstonEstimator')

    print("Qualitative checks:")

    # 1. Height h2 inflated
    if he_gen0 and he_last and 'height' in he_gen0 and 'height' in he_last:
        h0 = he_gen0['height']['h2']
        h5 = he_last['height']['h2']
        ok = h5 > h0
        print(f"  Height HE h² inflates over generations: {h0:.3f} → {h5:.3f}  {'PASS' if ok else 'FAIL'}")

    # 2. Edu h2 inflated beyond true
    if he_last and 'edu' in he_last:
        h5 = he_last['edu']['h2']
        ok = h5 > h2_edu * 2  # should be well above true h2=0.01
        print(f"  Edu HE h² inflated beyond true (0.01): {h5:.3f}  {'PASS' if ok else 'FAIL'}")

    # 3. Wealth phantom h2
    if he_last and 'wealth' in he_last:
        h5 = he_last['wealth']['h2']
        ok = h5 > 0.005  # should be positive despite true h2=0
        print(f"  Wealth phantom HE h² > 0: {h5:.3f}  {'PASS' if ok else 'FAIL'}")

    # 4. Height h2 > edu h2 > wealth h2 at all generations
    if he_last:
        hh = he_last.get('height', {}).get('h2', 0)
        he = he_last.get('edu', {}).get('h2', 0)
        hw = he_last.get('wealth', {}).get('h2', 0)
        ok = hh > he > hw
        print(f"  Ordering height > edu > wealth: {hh:.3f} > {he:.3f} > {hw:.3f}  {'PASS' if ok else 'FAIL'}")

    print()
    print("NOTE: Using LinearAssortativeMating (r={:.3f}) instead of".format(r_mate))
    print("      GeneralAssortativeMatingRegime (full cross-correlation matrix).")
    print("      Exact magnitudes will differ from paper; qualitative patterns")
    print("      should match.")


if __name__ == '__main__':
    print("Running new API simulation...\n")
    sim = run_simulation()
    print(f"Simulation complete. Final generation: {sim.generation}\n")
    print_results(sim)
