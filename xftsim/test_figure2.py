"""
Figure 2 — Psychiatric Diagnoses under Empirical xAM
(6-way simulation, 50 seeds)

Models 6 psychiatric traits with empirical cross-mate correlations
and trait-specific heritabilities from psych_cors.csv.

Traits: ADHD, ALC, ANX, BIP, MDD, SCZ
- Each trait has independent genetic effects (true rg = 0)
- Empirical 6x6 cross-mate correlation matrix (non-exchangeable)
- Liability threshold model: continuous phenotypes binarized into diagnoses
- Mating operates on continuous phenotypes; HE estimated on both scales
- No vertical transmission

Saves per-seed, per-generation results to figure2_results.csv.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats

from xftsim.founders import founder_haplotypes_uniform_AFs
from xftsim.effect import AdditiveEffects
from xftsim.arch import Architecture
from xftsim.mate import GeneralAssortativeMating, BatchedMating
from xftsim.reproduce import RecombinationMap
from xftsim.sim import Simulation
from xftsim.stats import SampleStatistics, HasemanElstonEstimator

# ── Parameters ──────────────────────────────────────────────────────────────

n_individuals = 16000
n_loci = 2000
n_generations = 6
n_runs = 2

dx = ["ADHD", "ALC", "ANX", "BIP", "MDD", "SCZ"]
dx_diag = [f"{d}.dx" for d in dx]

# Prevalences and liability thresholds
prev = np.array([0.087, 0.291, 0.316, 0.025, 0.144, 0.04])
thresh = stats.norm.ppf(1 - prev)


def build_and_run_6way(seed):
    """Build and run the 6-way simulation for a given seed."""

    # Load empirical correlations and heritabilities
    psychdat = pd.read_csv(
        '/Users/ajayprabhakar/PycharmProjects/xftmanu_code_supplement/psych_cors.csv'
    )
    sdat = psychdat.loc[psychdat.seed == seed]

    R_mate = np.array([
        [sdat.rmate.loc[(sdat.dx1 == x) & (sdat.dx2 == y)].values[0]
         for x in dx]
        for y in dx
    ])

    h2 = np.array([
        sdat.vg1.loc[sdat.dx1 == d].mean() for d in dx
    ])
    ve = 1.0 - h2

    founder_haplotypes = founder_haplotypes_uniform_AFs(n=n_individuals, m=n_loci)

    effects = {}
    formula_lines = []
    for i, d in enumerate(dx):
        eff_name = f'{d}_eff'
        effects[eff_name] = AdditiveEffects.from_h2(
            h2=float(h2[i]), m=n_loci, seed=100 + i + seed * 1000
        )
        formula_lines.append(f"{d}.G ~ genetic({eff_name})")
        formula_lines.append(f"{d}.E ~ noise({float(ve[i])})")
        formula_lines.append(f"{d} ~ {d}.G + {d}.E")
        formula_lines.append(f"{d}.dx ~ threshold({d}, {float(thresh[i]):.6f})")
        formula_lines.append("")

    formula = "\n".join(formula_lines)
    arch = Architecture(formula=formula, effects=effects)
    rmap = RecombinationMap(p=0.5, m=n_loci)

    mating = BatchedMating(
        regime=GeneralAssortativeMating(
            component_names=dx,
            cross_corr=R_mate,
            offspring_per_pair=2,
            solver_params=dict(
                time_limit=30,
                termination_interval=5,
                tolerance=1e-3,
            ),
        ),
        max_batch_size=1000,
    )

    sim = Simulation(
        founder_haplotypes=founder_haplotypes,
        architecture=arch,
        mating_regime=mating,
        recombination_map=rmap,
        retain_haplotypes=1,
        retain_phenotypes=2,
        statistics=[
            SampleStatistics(),
            HasemanElstonEstimator(phenotype_keys=dx + dx_diag),
        ],
        seed=seed,
    )

    sim.run(n_generations=n_generations)
    return sim, h2, R_mate


def extract_results(sim):
    """Extract per-generation HE rg on liability and diagnosis scales."""
    rows = []

    for result in sim.results:
        gen = result.generation
        he = result.statistics.get('HasemanElstonEstimator')
        if he is None or '_cov_g' not in he:
            continue

        row = {'generation': gen}

        cov_g = he['_cov_g']
        he_keys = he['_keys']
        d_diag = np.sqrt(np.abs(np.diag(cov_g)))
        d_diag[d_diag < 1e-15] = 1.0
        rg = cov_g / np.outer(d_diag, d_diag)

        # HE h² per trait (liability and diagnosis)
        for trait in dx + dx_diag:
            if trait in he:
                row[f'h2_HE_{trait}'] = he[trait]['h2']

        # HE rg for all trait pairs — liability scale
        for i in range(len(dx)):
            for j in range(i + 1, len(dx)):
                t1, t2 = dx[i], dx[j]
                if t1 in he_keys and t2 in he_keys:
                    ii = he_keys.index(t1)
                    jj = he_keys.index(t2)
                    row[f'rg_liab_{t1}_{t2}'] = rg[ii, jj]

        # HE rg for all trait pairs — diagnosis scale
        for i in range(len(dx_diag)):
            for j in range(i + 1, len(dx_diag)):
                t1, t2 = dx_diag[i], dx_diag[j]
                if t1 in he_keys and t2 in he_keys:
                    ii = he_keys.index(t1)
                    jj = he_keys.index(t2)
                    row[f'rg_dx_{dx[i]}_{dx[j]}'] = rg[ii, jj]

        rows.append(row)

    return rows


# ── Main ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    all_rows = []

    for run in range(n_runs):
        seed = run + 1
        print(f"\n{'='*70}")
        print(f"Seed {seed}/{n_runs}")
        print(f"{'='*70}")

        sim, h2, R_mate = build_and_run_6way(seed)
        print(f"  Done. Final generation: {sim.generation}")

        rows = extract_results(sim)
        for row in rows:
            row['seed'] = seed
        all_rows.extend(rows)

        del sim

    # ── Save to CSV ────────────────────────────────────────────────────

    df = pd.DataFrame(all_rows)
    outpath = 'figure2_results.csv'
    df.to_csv(outpath, index=False)
    print(f"\nAll results saved to {outpath}")

    # ── Print summary ──────────────────────────────────────────────────

    print(f"\n{'='*100}")
    print(f"FIGURE 2 RESULTS — 6-WAY, AVERAGED OVER {n_runs} SEEDS")
    print(f"{'='*100}")

    print(f"\n--- HE h² per trait (liability scale) ---")
    print(f"{'Gen':>3}  " + "  ".join(f"{d:>8}" for d in dx))
    print("-" * 60)
    for gen in sorted(df['generation'].unique()):
        gdf = df[df['generation'] == gen]
        vals = "  ".join(
            f"{gdf[f'h2_HE_{d}'].mean():8.4f}" if f'h2_HE_{d}' in gdf.columns else f"{'nan':>8}"
            for d in dx
        )
        print(f"{gen:3d}  {vals}")

    print(f"\n--- Mean off-diagonal rg (liability scale) ---")
    rg_liab_cols = [c for c in df.columns if c.startswith('rg_liab_')]
    print(f"{'Gen':>3}  {'mean_rg':>10}  {'sd_rg':>10}")
    print("-" * 30)
    for gen in sorted(df['generation'].unique()):
        gdf = df[df['generation'] == gen]
        all_rg = gdf[rg_liab_cols].values.flatten()
        all_rg = all_rg[~np.isnan(all_rg)]
        print(f"{gen:3d}  {np.mean(all_rg):10.4f}  {np.std(all_rg):10.4f}")

    print(f"\n--- Mean off-diagonal rg (diagnosis scale) ---")
    rg_dx_cols = [c for c in df.columns if c.startswith('rg_dx_')]
    print(f"{'Gen':>3}  {'mean_rg':>10}  {'sd_rg':>10}")
    print("-" * 30)
    for gen in sorted(df['generation'].unique()):
        gdf = df[df['generation'] == gen]
        all_rg = gdf[rg_dx_cols].values.flatten()
        all_rg = all_rg[~np.isnan(all_rg)]
        print(f"{gen:3d}  {np.mean(all_rg):10.4f}  {np.std(all_rg):10.4f}")

    print(f"\n{'='*100}")
    print("Note: True genetic correlations are all zero (independent effects).")
    print("All rg inflation is an artifact of 6-way cross-trait assortative mating.")
