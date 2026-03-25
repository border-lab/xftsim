"""
Figure 2 — Psychiatric Diagnoses under Empirical xAM
(Single seed, liability threshold model, 6-way + pairwise)

Models 6 psychiatric traits with empirical cross-mate correlations
and trait-specific heritabilities from psych_cors.csv (seed=1).

Traits: ADHD, ALC, ANX, BIP, MDD, SCZ
- Each trait has independent genetic effects (true rg = 0)
- Empirical 6×6 cross-mate correlation matrix (non-exchangeable)
- Liability threshold model: continuous phenotypes binarized into diagnoses
- Mating operates on continuous phenotypes; HE estimated on both scales
- No vertical transmission

Two simulation types:
  1) 6-way: all 6 traits assorted simultaneously
  2) Pairwise: 15 separate sims, each assorting on only 2 traits
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from itertools import combinations

from xftsim.founders import founder_haplotypes_uniform_AFs
from xftsim.neffect import AdditiveEffects
from xftsim.narch import Architecture
from xftsim.nmate import GeneralAssortativeMating, BatchedMating
from xftsim.reproduce import RecombinationMap
from xftsim.nsim import NSimulation
from xftsim.nstats import SampleStatistics, HasemanElstonEstimator

# ── Parameters ──────────────────────────────────────────────────────────────

SEED = 1
n_individuals = 8000
n_loci = 2000
n_generations = 6

dx = ["ADHD", "ALC", "ANX", "BIP", "MDD", "SCZ"]
dx_diag = [f"{d}.dx" for d in dx]

# Prevalences and liability thresholds
prev = np.array([0.087, 0.291, 0.316, 0.025, 0.144, 0.04])
thresh = stats.norm.ppf(1 - prev)

# ── Load empirical correlations and heritabilities from CSV ─────────────────

psychdat = pd.read_csv(
    '/Users/ajayprabhakar/PycharmProjects/xftmanu_code_supplement/psych_cors.csv'
)
sdat = psychdat.loc[psychdat.seed == SEED]

# Build full 6×6 cross-mate correlation matrix
R_mate_full = np.array([
    [sdat.rmate.loc[(sdat.dx1 == x) & (sdat.dx2 == y)].values[0]
     for x in dx]
    for y in dx
])

# Trait-specific heritabilities
h2 = np.array([
    sdat.vg1.loc[sdat.dx1 == d].mean() for d in dx
])
ve = 1.0 - h2

print("Traits:", dx)
print(f"Heritabilities (seed={SEED}):")
for d, h in zip(dx, h2):
    print(f"  {d:>4s}: h² = {h:.4f}")
print(f"\nCross-mate correlation matrix:")
print("       " + "  ".join(f"{d:>6s}" for d in dx))
for i, d in enumerate(dx):
    print(f"  {d:>4s} " + "  ".join(f"{R_mate_full[i,j]:6.3f}" for j in range(6)))


# ── Helper: build and run a simulation ──────────────────────────────────────

def build_and_run(label, mate_component_names, cross_corr, sim_seed=42):
    """Build and run a simulation with given mating structure."""

    founder_haplotypes = founder_haplotypes_uniform_AFs(n=n_individuals, m=n_loci)

    # All 6 traits are always simulated
    effects = {}
    formula_lines = []
    for i, d in enumerate(dx):
        eff_name = f'{d}_eff'
        effects[eff_name] = AdditiveEffects.from_h2(
            h2=float(h2[i]), m=n_loci, seed=100 + i
        )
        formula_lines.append(f"{d}.G ~ genetic({eff_name})")
        formula_lines.append(f"{d}.E ~ noise({float(ve[i])})")
        formula_lines.append(f"{d} ~ {d}.G + {d}.E")
        formula_lines.append(f"{d}.dx ~ threshold({d}, {float(thresh[i]):.6f})")
        formula_lines.append("")

    formula = "\n".join(formula_lines)
    arch = Architecture(formula=formula, effects=effects)
    rmap = RecombinationMap(p=0.5, m=n_loci)

    # Mating only on the specified subset of traits
    mating = BatchedMating(
        regime=GeneralAssortativeMating(
            component_names=mate_component_names,
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
            HasemanElstonEstimator(phenotype_keys=dx + dx_diag),
        ],
        seed=sim_seed,
    )

    print(f"\n{'='*70}")
    print(f"Running: {label}")
    print(f"  Mating on: {mate_component_names}")
    print(f"{'='*70}")

    sim.run(n_generations=n_generations)
    print(f"  Done. Final generation: {sim.generation}")
    return sim


# ── Helpers ──────────────────────────────────────────────────────────────────

def extract_rg_per_gen(sim, scale='liability'):
    """Extract rg matrix for every generation."""
    n_traits = len(dx)
    results = []
    for result in sim.results:
        he = result.statistics.get('HasemanElstonEstimator')
        if he is None or '_cov_g' not in he:
            results.append((result.generation, None))
            continue
        cov_g = he['_cov_g']
        if scale == 'liability':
            cov = cov_g[:n_traits, :n_traits]
        else:
            cov = cov_g[n_traits:, n_traits:]
        diag = np.sqrt(np.diag(cov))
        diag[diag == 0] = 1.0
        rg = cov / np.outer(diag, diag)
        results.append((result.generation, rg))
    return results


def extract_pairwise_rg_per_gen(sim, idx1, idx2, scale='liability'):
    """Extract rg between two specific traits for every generation."""
    rg_by_gen = extract_rg_per_gen(sim, scale)
    out = []
    for gen, rg in rg_by_gen:
        if rg is not None:
            out.append((gen, rg[idx1, idx2]))
        else:
            out.append((gen, float('nan')))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Part 1: 6-way simulation
# ══════════════════════════════════════════════════════════════════════════════

sim_6way = build_and_run("6-way", dx, R_mate_full)

# Store 6-way per-generation rg for each pair
rg_6way_by_gen_liab = extract_rg_per_gen(sim_6way, 'liability')
rg_6way_by_gen_dx = extract_rg_per_gen(sim_6way, 'diagnosis')

# Free memory
del sim_6way


# ══════════════════════════════════════════════════════════════════════════════
# Part 2: Pairwise simulations (15 pairs)
# ══════════════════════════════════════════════════════════════════════════════

pairs = list(combinations(range(6), 2))

# Store per-generation rg for each pairwise sim: {(i,j): [(gen, rg), ...]}
pairwise_by_gen_liab = {}
pairwise_by_gen_dx = {}

for idx1, idx2 in pairs:
    d1, d2 = dx[idx1], dx[idx2]

    # 2×2 submatrix of cross-mate correlations
    sub_corr = R_mate_full[np.ix_([idx1, idx2], [idx1, idx2])]

    sim_pair = build_and_run(
        f"Pairwise: {d1}-{d2}",
        [d1, d2],
        sub_corr,
        sim_seed=42,
    )

    pairwise_by_gen_liab[(idx1, idx2)] = extract_pairwise_rg_per_gen(
        sim_pair, idx1, idx2, 'liability'
    )
    pairwise_by_gen_dx[(idx1, idx2)] = extract_pairwise_rg_per_gen(
        sim_pair, idx1, idx2, 'diagnosis'
    )

    del sim_pair


# ══════════════════════════════════════════════════════════════════════════════
# Results: Per-generation rg comparison (6-way vs Pairwise)
# ══════════════════════════════════════════════════════════════════════════════

for scale_name, rg_6way_list, pw_dict in [
    ('LIABILITY', rg_6way_by_gen_liab, pairwise_by_gen_liab),
    ('DIAGNOSIS', rg_6way_by_gen_dx, pairwise_by_gen_dx),
]:
    print(f"\n\n{'='*90}")
    print(f"HE-Estimated rg per generation — {scale_name} scale  (true rg = 0)")
    print(f"{'='*90}")

    for idx1, idx2 in pairs:
        d1, d2 = dx[idx1], dx[idx2]
        xmate_r = R_mate_full[idx1, idx2]

        print(f"\n  {d1}-{d2}  (cross-mate r = {xmate_r:.3f})")
        print(f"  {'Gen':>3s}  {'6-way rg':>10s}  {'pairwise rg':>12s}")
        print(f"  {'-'*30}")

        pw_vals = dict(pw_dict[(idx1, idx2)])

        for gen, rg_mat in rg_6way_list:
            rg6 = rg_mat[idx1, idx2] if rg_mat is not None else float('nan')
            rgp = pw_vals.get(gen, float('nan'))
            print(f"  {gen:3d}  {rg6:10.4f}  {rgp:12.4f}")

    print(f"\n{'='*90}")

print("\nNote: True genetic correlations are all zero (independent effects).")
print("6-way rg includes indirect xAM effects mediated through other traits.")
print("Pairwise rg reflects only the direct cross-mate similarity of that pair.")
