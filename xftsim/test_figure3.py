"""
Reproducing Figure 3 from the xAM manuscript.

Four scenarios with 5 traits, h²=0.5, orthogonal genetic effects, zero pleiotropy:
  1. RM        — Random mating, no VT
  2. RM + VT   — Random mating + vertical transmission (5% of phenotypic variance)
  3. 5xAM      — 5-trait exchangeable cross-mate correlations of 0.2, no VT
  4. 5xAM + VT — 5-trait xAM + VT (5%)

Uses LinearAssortativeMating (unidimensional sorting) for xAM scenarios,
matching the approach described in the manuscript methods.

Target values from Table S6 (generation 5):
  RM:         true h²=0.500, est h²=0.500, true rg=0.000, est rg=0.000
  RM + VT:    true h²=0.442, est h²=0.552, true rg=0.000, est rg=0.199
  5xAM:       true h²=0.535, est h²=0.658, true rg=0.132, est rg=0.296
  5xAM + VT:  true h²=0.396, est h²=0.749, true rg=0.078, est rg=0.513
"""

import numpy as np
import matplotlib.pyplot as plt

from xftsim.founders import founder_haplotypes_uniform_AFs
from xftsim.neffect import AdditiveEffects
from xftsim.narch import Architecture
from xftsim.nmate import RandomMating, LinearAssortativeMating
from xftsim.reproduce import RecombinationMap
from xftsim.nsim import NSimulation
from xftsim.nstats import SampleStatistics, HasemanElstonEstimator

# ── Parameters ──────────────────────────────────────────────────────────────

n_individuals = 32000
n_loci = 4000
n_generations = 6
K = 5  # number of traits

h2 = 0.50
theta = 0.05   # VT fraction of phenotypic variance at gen 0
xmate_r = 0.2  # exchangeable cross-mate correlation

# Per-parent, per-trait VT coefficient:
#   T_k = sqrt(theta/(2K)) * sum_j (Y*_j + Y**_j)
# Each of the 2K parent-trait terms gets coefficient sqrt(theta/(2K))
vt_coeff = np.sqrt(theta / (2 * K))  # ≈ 0.0707

# Noise variances
noise_var_no_vt = 1.0 - h2                # = 0.50
noise_var_with_vt = 1.0 - h2 - theta      # = 0.45

# Founder VT draw variance (parents at gen 0 have no actual parents,
# so VT components are initialized as noise draws with this variance)
vt_founder_var = 1.0

trait_names = [f't{i+1}' for i in range(K)]


def build_sim(scenario, seed=42):
    """Build an NSimulation for a given scenario.

    Parameters
    ----------
    scenario : str
        One of 'RM', 'RM+VT', '5xAM', '5xAM+VT'
    seed : int
        Random seed for reproducibility.
    """
    use_vt = 'VT' in scenario
    use_xam = 'xAM' in scenario

    noise_var = noise_var_with_vt if use_vt else noise_var_no_vt

    founder_haplotypes = founder_haplotypes_uniform_AFs(n=n_individuals, m=n_loci)

    # Independent genetic effects for each trait (orthogonal, zero pleiotropy)
    effects = {}
    effect_lines = []
    for i, name in enumerate(trait_names):
        eff_name = f'{name}_eff'
        effects[eff_name] = AdditiveEffects.from_h2(h2=h2, m=n_loci, seed=seed + i)
        effect_lines.append(f'{name}.G ~ genetic({eff_name})')

    # Build formula
    #
    # VT model from the manuscript:
    #   T_k = sqrt(theta/(2K)) * sum_j (Y*_j + Y**_j)
    # Each offspring trait k receives VT from ALL K parent traits (cross-trait VT).
    # This shared VT component creates within-person trait correlations that
    # the HE estimator attributes to genetic causes.
    lines = []

    if use_vt:
        # First, define all mother/father VT source components
        for src in trait_names:
            lines.append(f'{src}.VTsrc_m ~ mother({src}, normalize=True,  founder=noise({vt_founder_var}))')
            lines.append(f'{src}.VTsrc_f ~ father({src}, normalize=True, founder=noise({vt_founder_var}))')
        lines.append('')

    for i, name in enumerate(trait_names):
        lines.append(f'{name}.G ~ genetic({name}_eff)')
        lines.append(f'{name}.E ~ noise({noise_var})')

        if use_vt:
            # T_k = sqrt(theta/(2K)) * sum_j (mother(Y_j) + father(Y_j))
            vt_terms = []
            for src in trait_names:
                vt_terms.append(f'{vt_coeff} * {src}.VTsrc_m')
                vt_terms.append(f'{vt_coeff} * {src}.VTsrc_f')
            lines.append(f'{name}.VT ~ ' + ' + '.join(vt_terms))
            lines.append(f'{name} ~ {name}.G + {name}.E + {name}.VT')
        else:
            lines.append(f'{name} ~ {name}.G + {name}.E')

        lines.append('')  # blank line between traits

    formula = '\n'.join(lines)
    arch = Architecture(formula=formula, effects=effects)

    rmap = RecombinationMap(p=50 / n_loci, m=n_loci)

    if use_xam:
        mating = LinearAssortativeMating(
            component_names=trait_names,
            r=xmate_r,
            offspring_per_pair=2,
        )
    else:
        mating = RandomMating(offspring_per_pair=2)

    sim = NSimulation(
        founder_haplotypes=founder_haplotypes,
        architecture=arch,
        mating_regime=mating,
        recombination_map=rmap,
        retain_haplotypes=1,
        retain_phenotypes=2,
        statistics=[
            SampleStatistics(),
            HasemanElstonEstimator(phenotype_keys=trait_names),
        ],
        seed=seed,
    )
    return sim


def extract_results(sim):
    """Extract mean h² and mean rg across traits/pairs for each generation."""
    rows = []
    for result in sim.results:
        gen = result.generation
        stats = result.statistics

        # Mean HE-estimated h² across traits
        he = stats.get('HasemanElstonEstimator')
        he_h2_vals = []
        if he:
            for name in trait_names:
                if name in he:
                    he_h2_vals.append(he[name]['h2'])

        # Mean off-diagonal genetic correlation
        he_rg_vals = []
        if he and '_cov_g' in he:
            cov_g = he['_cov_g']
            d = np.sqrt(np.abs(np.diag(cov_g)))
            d[d < 1e-15] = 1.0
            rg = cov_g / np.outer(d, d)
            k = rg.shape[0]
            for i in range(k):
                for j in range(i + 1, k):
                    he_rg_vals.append(rg[i, j])

        # Phenotypic variances (mean across traits)
        ss = stats.get('SampleStatistics')
        mean_var = np.nan
        if ss:
            keys = ss['keys']
            var = ss['var']
            trait_vars = [var[keys.index(t)] for t in trait_names if t in keys]
            if trait_vars:
                mean_var = np.mean(trait_vars)

        rows.append({
            'gen': gen,
            'mean_pheno_var': mean_var,
            'mean_he_h2': np.mean(he_h2_vals) if he_h2_vals else np.nan,
            'mean_he_rg': np.mean(he_rg_vals) if he_rg_vals else np.nan,
        })
    return rows


# ── Run all four scenarios ─────────────────────────────────────────────────

if __name__ == '__main__':
    scenarios = ['RM', 'RM+VT', '5xAM', '5xAM+VT']
    n_runs = 10

    # {scenario: {gen: {'h2': [values], 'rg': [values], 'var': [values]}}}
    aggregated = {s: {} for s in scenarios}

    for scenario in scenarios:
        print(f"\n{'='*70}")
        print(f"Running: {scenario} ({n_runs} seeds)")
        print(f"{'='*70}")

        for run in range(n_runs):
            seed = run + 1
            print(f"  Seed {seed}/{n_runs} ... ", end="", flush=True)
            sim = build_sim(scenario, seed=seed)
            sim.run(n_generations=n_generations)
            rows = extract_results(sim)
            print(f"done (gen {sim.generation})")

            for row in rows:
                gen = row['gen']
                if gen not in aggregated[scenario]:
                    aggregated[scenario][gen] = {'h2': [], 'rg': [], 'var': []}
                aggregated[scenario][gen]['h2'].append(row['mean_he_h2'])
                aggregated[scenario][gen]['rg'].append(row['mean_he_rg'])
                aggregated[scenario][gen]['var'].append(row['mean_pheno_var'])

            del sim

    # ── Summary table (averaged over seeds) ────────────────────────────────

    print(f"\n\n{'='*100}")
    print(f"FIGURE 3 RESULTS — AVERAGED OVER {n_runs} SEEDS")
    print(f"{'='*100}")
    print(f"{'Scenario':<15} {'Gen':>3}  |  {'Mean h²(HE)':>11} {'± SD':>8}  "
          f"{'Mean rg(HE)':>11} {'± SD':>8}  {'Mean Var(Y)':>11}")
    print(f"{'-'*85}")

    for scenario in scenarios:
        gens = sorted(aggregated[scenario].keys())
        for gen in gens:
            vals = aggregated[scenario][gen]
            h2_arr = np.array(vals['h2'])
            rg_arr = np.array(vals['rg'])
            var_arr = np.array(vals['var'])
            print(f"{scenario:<15} {gen:3d}  |  "
                  f"{np.mean(h2_arr):11.4f} {np.std(h2_arr):8.4f}  "
                  f"{np.mean(rg_arr):11.4f} {np.std(rg_arr):8.4f}  "
                  f"{np.mean(var_arr):11.4f}")
        print()

    # ── Comparison with paper targets ──────────────────────────────────────

    print(f"{'='*100}")
    print("Table S6 targets (generation 5):")
    print(f"  {'Scenario':<15} {'true h²':>8} {'est h²':>8} {'true rg':>8} {'est rg':>8}")
    print(f"  {'RM':<15} {'0.500':>8} {'0.500':>8} {'0.000':>8} {'0.000':>8}")
    print(f"  {'RM + VT':<15} {'0.442':>8} {'0.552':>8} {'0.000':>8} {'0.199':>8}")
    print(f"  {'5xAM':<15} {'0.535':>8} {'0.658':>8} {'0.132':>8} {'0.296':>8}")
    print(f"  {'5xAM + VT':<15} {'0.396':>8} {'0.749':>8} {'0.078':>8} {'0.513':>8}")
    print(f"{'='*100}")

    # ── Line graphs: all models on same plot ─────────────────────────────────

    colors = {'RM': 'tab:blue', 'RM+VT': 'tab:orange',
              '5xAM': 'tab:green', '5xAM+VT': 'tab:red'}

    # h² plot
    fig_h2, ax_h2 = plt.subplots(figsize=(8, 5))
    for scenario in scenarios:
        gens = np.array(sorted(aggregated[scenario].keys()))
        h2_mean = np.array([np.mean(aggregated[scenario][g]['h2']) for g in gens])
        h2_std = np.array([np.std(aggregated[scenario][g]['h2']) for g in gens])
        ax_h2.plot(gens, h2_mean, 'o-', label=scenario, color=colors[scenario])
        ax_h2.fill_between(gens, h2_mean - h2_std, h2_mean + h2_std,
                           alpha=0.2, color=colors[scenario])
    ax_h2.set_xlabel('Generation')
    ax_h2.set_ylabel('h²(HE)')
    ax_h2.set_title(f'HE-Estimated h² Across Generations (mean ± SD, n={n_runs})')
    ax_h2.legend()
    ax_h2.grid(True, alpha=0.3)
    fig_h2.tight_layout()
    fig_h2.savefig('figure3_h2.png', dpi=150)
    print("\nPlot saved to figure3_h2.png")

    # rg plot
    fig_rg, ax_rg = plt.subplots(figsize=(8, 5))
    for scenario in scenarios:
        gens = np.array(sorted(aggregated[scenario].keys()))
        rg_mean = np.array([np.mean(aggregated[scenario][g]['rg']) for g in gens])
        rg_std = np.array([np.std(aggregated[scenario][g]['rg']) for g in gens])
        ax_rg.plot(gens, rg_mean, 's-', label=scenario, color=colors[scenario])
        ax_rg.fill_between(gens, rg_mean - rg_std, rg_mean + rg_std,
                           alpha=0.2, color=colors[scenario])
    ax_rg.set_xlabel('Generation')
    ax_rg.set_ylabel('rg(HE)')
    ax_rg.set_title(f'HE-Estimated rg Across Generations (mean ± SD, n={n_runs})')
    ax_rg.legend()
    ax_rg.grid(True, alpha=0.3)
    fig_rg.tight_layout()
    fig_rg.savefig('figure3_rg.png', dpi=150)
    print("Plot saved to figure3_rg.png")

    plt.show()
