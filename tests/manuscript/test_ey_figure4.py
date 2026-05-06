"""
Wealth-Education-Height Joint Architecture Simulation
(Reproducing Figure 4 from the xAM manuscript)

Parameters matched to ey_sim.py in the code supplement:
  https://github.com/border-lab/xftmanu_code_supplement

Models three traits with distinct transmission channels:
- Height: heritable (h²=0.60), no VT. Connects to other traits only via xAM.
- Education: low heritability (h²=0.01), VT from parental edu AND parental wealth.
- Wealth: NO genetic component. Pure vertical transmission from parental wealth.

Panels:
  (b) Per-trait h²_HE, h²_true, R²_G across generations
  (c) True PGI score correlations (r_score) and HE rg for trait pairs
  (d) GWAS beta correlations (r̂_β) for trait pairs

Figure 4 target values (after 5 generations of xAM):
  Height: true h² 0.599 → 0.621  |  HE h² 0.599 → 0.686
  Edu:    true h² 0.010 → 0.005  |  HE h² 0.010 → 0.068
  Wealth: true h² 0.000 → 0.000  |  HE h² 0.000 → 0.036

NOTE: This simulation uses LinearAssortativeMating (single scalar r)
      instead of GeneralAssortativeMatingRegime (full cross-correlation
      matrix). Results are qualitatively similar but not numerically
      identical to the paper.
"""

import numpy as np
import pandas as pd

from xftsim.founders import founder_haplotypes_uniform_AFs
from xftsim.effect import AdditiveEffects
from xftsim.arch import Architecture
from xftsim.mate import LinearAssortativeMating
from xftsim.reproduce import RecombinationMap
from xftsim.sim import NSimulation
from xftsim.stats import SampleStatistics, HasemanElstonEstimator
from xftsim.gwas import GWAS

# ── Parameters (matched to ey_sim.py) ────────────────────────────────────

n_individuals = 128000
n_loci = 2000
n_generations = 6

# Heritabilities (only height and edu have genetic components)
h2_height = 0.60
h2_edu = 0.01

# VT coefficients from the paper's transmission matrix
CC = np.sqrt(1 / 3)
AA = np.sqrt(2 / 3)
vt_edu_from_edu = CC * np.sqrt(0.5)
vt_edu_from_wealth = CC * np.sqrt(0.5)
vt_wealth_from_wealth = AA * np.sqrt(0.5)

# Noise variances (from ey_sim.py)
var_edu = 1 / 3 - h2_edu
var_height = 0.4
var_wealth = 1 / 3

vt_founder_var = 1.0

# Mating cross-correlation matrix (paper uses full matrix; we use mean)
xmatecorr = np.array([[0.24626319, 0.1915851, 0.2521529],
                       [0.1915851, 0.1247212, 0.18323772],
                       [0.2521529, 0.18323772, 0.25072489]])
r_mean = float(np.mean(xmatecorr))

p_recomb = 50 / n_loci

traits = ['edu', 'height', 'wealth']
# Only edu and height have genetic components
genetic_traits = ['edu', 'height']
trait_pairs = [('edu', 'height'), ('edu', 'wealth'), ('height', 'wealth')]


def build_ey_sim(n=n_individuals, m=n_loci, seed=42):
    """Build the education-height-wealth simulation.

    retain_haplotypes and retain_phenotypes are set to n_generations+1
    so all generations remain accessible for post-hoc analysis.
    """
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
        retain_haplotypes=n_generations + 1,
        retain_phenotypes=n_generations + 1,
        statistics=[
            SampleStatistics(),
            HasemanElstonEstimator(phenotype_keys=['edu', 'height', 'wealth']),
        ],
        seed=seed,
    )
    return sim


def extract_all(sim):
    """Extract all panel (b), (c), (d) quantities from a completed simulation.

    Returns a list of dicts (one per generation).
    """
    rows = []

    for result in sim.results:
        gen = result.generation
        stats = result.statistics
        row = {'generation': gen}

        pheno = sim.phenotype_history.get(gen)
        hap = sim.haplotype_history.get(gen)

        # ── Panel (b): h²_HE, h²_true, R²_G per trait ──────────────

        # HE h²
        he = stats.get('HasemanElstonEstimator')
        if he:
            for t in traits:
                if t in he:
                    row[f'h2_HE_{t}'] = he[t]['h2']

        if pheno is not None:
            for t in traits:
                total = pheno[t]
                var_total = np.var(total)

                # True h² (only edu and height have .G components)
                if t in genetic_traits:
                    g = pheno[f'{t}.G']
                    row[f'h2_true_{t}'] = np.var(g) / var_total if var_total > 0 else 0.0
                else:
                    row[f'h2_true_{t}'] = 0.0

                # R²_G: variance in each phenotype explained by each PGI
                for gt in genetic_traits:
                    g = pheno[f'{gt}.G']
                    r = np.corrcoef(g, total)[0, 1]
                    row[f'R2_{t}_from_{gt}_PGI'] = r ** 2

        # Phenotypic variances
        ss = stats.get('SampleStatistics')
        if ss:
            keys_ss = ss['keys']
            var_ss = ss['var']
            for t in traits:
                if t in keys_ss:
                    row[f'var_{t}'] = var_ss[keys_ss.index(t)]

        # ── Panel (c): PGI score correlations and HE rg ─────────────

        # True PGI score correlations (only between traits that have .G)
        if pheno is not None:
            for t1, t2 in trait_pairs:
                # r_score: correlation of true genetic values
                if t1 in genetic_traits and t2 in genetic_traits:
                    g1 = pheno[f'{t1}.G']
                    g2 = pheno[f'{t2}.G']
                    row[f'r_score_{t1}_{t2}'] = np.corrcoef(g1, g2)[0, 1]
                else:
                    row[f'r_score_{t1}_{t2}'] = np.nan

        # HE rg
        if he and '_cov_g' in he:
            cov_g = he['_cov_g']
            he_keys = he['_keys']
            d = np.sqrt(np.abs(np.diag(cov_g)))
            d[d < 1e-15] = 1.0
            rg = cov_g / np.outer(d, d)
            for t1, t2 in trait_pairs:
                if t1 in he_keys and t2 in he_keys:
                    i = he_keys.index(t1)
                    j = he_keys.index(t2)
                    row[f'rg_HE_{t1}_{t2}'] = rg[i, j]

        # ── Panel (d): GWAS beta correlations ────────────────────────

        if pheno is not None and hap is not None:
            gwas = GWAS(hap, pheno)
            gwas_results = gwas.run(traits)
            betas = {t: gwas_results[t].beta for t in traits}
            for t1, t2 in trait_pairs:
                row[f'r_beta_{t1}_{t2}'] = np.corrcoef(betas[t1], betas[t2])[0, 1]

        rows.append(row)

    return rows


# ── Run as script ─────────────────────────────────────────────────────────

if __name__ == '__main__':
    n_runs = 10
    all_rows = []

    for run in range(n_runs):
        seed = run + 1
        print(f"\n{'='*70}")
        print(f"Seed {seed}/{n_runs}")
        print(f"{'='*70}")

        sim = build_ey_sim(seed=seed)
        sim.run(n_generations=n_generations)
        print(f"  Done. Final generation: {sim.generation}")

        rows = extract_all(sim)
        for row in rows:
            row['seed'] = seed
        all_rows.extend(rows)

        del sim

    # ── Save to CSV ────────────────────────────────────────────────────────

    df = pd.DataFrame(all_rows)
    outpath = 'figure4_results.csv'
    df.to_csv(outpath, index=False)
    print(f"\nAll results saved to {outpath}")

    # ── Print averaged summary ─────────────────────────────────────────────

    print(f"\n{'='*100}")
    print(f"FIGURE 4 RESULTS — AVERAGED OVER {n_runs} SEEDS")
    print(f"{'='*100}")

    print(f"\n--- Panel (b): Heritability estimates ---")
    print(f"{'Gen':>3}  |  {'h2_HE_edu':>10} {'h2_HE_hgt':>10} {'h2_HE_wlt':>10}"
          f"  |  {'h2_true_edu':>11} {'h2_true_hgt':>11} {'h2_true_wlt':>11}")
    print("-" * 85)
    for gen in sorted(df['generation'].unique()):
        gdf = df[df['generation'] == gen]
        he = "  ".join(f"{gdf[f'h2_HE_{t}'].mean():10.4f}" for t in traits)
        tr = "  ".join(f"{gdf[f'h2_true_{t}'].mean():11.4f}" for t in traits)
        print(f"{gen:3d}  |  {he}  |  {tr}")

    print(f"\n--- Panel (c): PGI score correlations and HE rg ---")
    for t1, t2 in trait_pairs:
        print(f"\n  {t1}/{t2}:")
        print(f"  {'Gen':>3}  {'r_score':>10}  {'rg_HE':>10}")
        for gen in sorted(df['generation'].unique()):
            gdf = df[df['generation'] == gen]
            rs = gdf[f'r_score_{t1}_{t2}'].mean()
            rg = gdf[f'rg_HE_{t1}_{t2}'].mean()
            print(f"  {gen:3d}  {rs:10.4f}  {rg:10.4f}")

    print(f"\n--- Panel (d): GWAS beta correlations ---")
    for t1, t2 in trait_pairs:
        print(f"\n  {t1}/{t2}:")
        print(f"  {'Gen':>3}  {'r_beta':>10}")
        for gen in sorted(df['generation'].unique()):
            gdf = df[df['generation'] == gen]
            rb = gdf[f'r_beta_{t1}_{t2}'].mean()
            print(f"  {gen:3d}  {rb:10.4f}")

    print(f"\n{'='*100}")
    print("Figure 4 target values (5 generations of xAM):")
    print("  Height: true h² 0.599 → 0.621   |  HE h² 0.599 → 0.686")
    print("  Edu:    true h² 0.010 → 0.005   |  HE h² 0.010 → 0.068")
    print("  Wealth: true h² 0.000 → 0.000   |  HE h² 0.000 → 0.036")
    print(f"\nNOTE: Uses LinearAssortativeMating (r={r_mean:.3f}) as approximation.")
