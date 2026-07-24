"""
Reproducing Figure 3 from the xAM manuscript.

Four scenarios with 5 traits, h²=0.5, orthogonal genetic effects, zero
pleiotropy:

  - RM        — Random mating, no VT
  - RM + VT   — Random mating + vertical transmission (5% of phenotypic variance)
  - 5xAM      — 5-trait exchangeable cross-mate correlations of 0.2, no VT
  - 5xAM + VT — 5-trait xAM + VT (5%)

Verifies that the HE estimates at generation 5 approximately match Table S6.
Target values:

  Scenario      true h²   est h²    true rg   est rg
  RM            0.500     0.500     0.000     0.000
  RM + VT       0.442     0.552     0.000     0.199
  5xAM          0.535     0.658     0.132     0.296
  5xAM + VT     0.396     0.749     0.078     0.513

The paper uses n=32000, m=4000, n_runs=10. Tests default to a reduced grid
(n=4000, m=1000, n_runs=2) so the suite stays tractable on a developer
machine; tolerances are widened correspondingly. Override via env vars:

    XFTSIM_FIGURE3_N=8000 XFTSIM_FIGURE3_M=2000 XFTSIM_FIGURE3_RUNS=4 \
        pytest tests/manuscript/test_figure3.py

Set XFTSIM_SAVE_PLOTS=1 to also write figure3_h2.png / figure3_rg.png.
"""

import os

import numpy as np
import pytest

from xftsim.founders import founder_haplotypes_uniform_AFs
from xftsim.effect import AdditiveEffects
from xftsim.arch import Architecture
from xftsim.mate import RandomMating, LinearAssortativeMating
from xftsim.reproduce import RecombinationMap
from xftsim.sim import Simulation
from xftsim.stats import SampleStatistics, HasemanElstonEstimator


# ── Parameters ──────────────────────────────────────────────────────────────

N_INDIVIDUALS = int(os.environ.get("XFTSIM_FIGURE3_N", 4000))
N_LOCI = int(os.environ.get("XFTSIM_FIGURE3_M", 1000))
N_GENERATIONS = 6
N_RUNS = int(os.environ.get("XFTSIM_FIGURE3_RUNS", 2))

K = 5
H2 = 0.50
THETA = 0.05   # VT fraction of phenotypic variance at gen 0
XMATE_R = 0.2  # exchangeable cross-mate correlation

# Per-parent, per-trait VT coefficient (from the manuscript)
VT_COEFF = float(np.sqrt(THETA / (2 * K)))

NOISE_VAR_NO_VT = 1.0 - H2          # = 0.50
NOISE_VAR_WITH_VT = 1.0 - H2 - THETA  # = 0.45
VT_FOUNDER_VAR = 1.0

TRAIT_NAMES = [f't{i + 1}' for i in range(K)]


# ── Helpers ─────────────────────────────────────────────────────────────────

def build_sim(scenario: str, seed: int = 42,
              n: int = N_INDIVIDUALS, m: int = N_LOCI) -> Simulation:
    """Construct the Figure 3 simulation for a given scenario.

    Parameters
    ----------
    scenario : {'RM', 'RM+VT', '5xAM', '5xAM+VT'}
        Manuscript scenario label.
    seed : int
        Random seed.
    n, m : int
        Sample size and locus count.
    """
    use_vt = 'VT' in scenario
    use_xam = 'xAM' in scenario

    noise_var = NOISE_VAR_WITH_VT if use_vt else NOISE_VAR_NO_VT

    founder_haplotypes = founder_haplotypes_uniform_AFs(n=n, m=m)

    # Independent genetic effects per trait (orthogonal, zero pleiotropy)
    effects = {}
    for i, name in enumerate(TRAIT_NAMES):
        effects[f'{name}_eff'] = AdditiveEffects.from_h2(
            h2=H2, m=m, seed=seed + i,
        )

    # Build the formula. VT model from the manuscript:
    #   T_k = sqrt(theta / (2K)) * sum_j (Y*_j + Y**_j)
    lines = []
    if use_vt:
        for src in TRAIT_NAMES:
            lines.append(
                f'{src}.VTsrc_m ~ mother({src}, normalize=True, '
                f'founder=noise({VT_FOUNDER_VAR}))'
            )
            lines.append(
                f'{src}.VTsrc_f ~ father({src}, normalize=True, '
                f'founder=noise({VT_FOUNDER_VAR}))'
            )
        lines.append('')

    for name in TRAIT_NAMES:
        lines.append(f'{name}.G ~ genetic({name}_eff)')
        lines.append(f'{name}.E ~ noise({noise_var})')

        if use_vt:
            vt_terms = []
            for src in TRAIT_NAMES:
                vt_terms.append(f'{VT_COEFF} * {src}.VTsrc_m')
                vt_terms.append(f'{VT_COEFF} * {src}.VTsrc_f')
            lines.append(f'{name}.VT ~ ' + ' + '.join(vt_terms))
            lines.append(f'{name} ~ {name}.G + {name}.E + {name}.VT')
        else:
            lines.append(f'{name} ~ {name}.G + {name}.E')
        lines.append('')

    arch = Architecture(formula='\n'.join(lines), effects=effects)
    rmap = RecombinationMap(p=50 / m, m=m)

    if use_xam:
        mating = LinearAssortativeMating(
            component_names=TRAIT_NAMES,
            r=XMATE_R,
            offspring_per_pair=2,
        )
    else:
        mating = RandomMating(offspring_per_pair=2)

    return Simulation(
        founder_haplotypes=founder_haplotypes,
        architecture=arch,
        mating_regime=mating,
        recombination_map=rmap,
        retain_haplotypes=1,
        retain_phenotypes=2,
        statistics=[
            SampleStatistics(),
            HasemanElstonEstimator(phenotype_keys=TRAIT_NAMES),
        ],
        seed=seed,
    )


def extract_results(sim: Simulation) -> list[dict]:
    """Per-generation mean HE h² and mean off-diagonal HE rg across traits."""
    rows = []
    for result in sim.results:
        gen = result.generation
        stats = result.statistics

        he = stats.get('HasemanElstonEstimator') or {}
        he_h2 = [he[t]['h2'] for t in TRAIT_NAMES if t in he]

        he_rg = []
        if '_cov_g' in he:
            cov_g = he['_cov_g']
            d = np.sqrt(np.abs(np.diag(cov_g)))
            d[d < 1e-15] = 1.0
            rg = cov_g / np.outer(d, d)
            for i in range(rg.shape[0]):
                for j in range(i + 1, rg.shape[1]):
                    he_rg.append(rg[i, j])

        rows.append({
            'gen': gen,
            'mean_he_h2': float(np.mean(he_h2)) if he_h2 else np.nan,
            'mean_he_rg': float(np.mean(he_rg)) if he_rg else np.nan,
        })
    return rows


def aggregate_runs(scenario: str, n_runs: int = N_RUNS,
                   n: int = N_INDIVIDUALS, m: int = N_LOCI) -> dict[int, dict[str, list[float]]]:
    """Run `n_runs` seeds for one scenario, return {gen: {'h2': [...], 'rg': [...]}}."""
    aggregated: dict[int, dict[str, list[float]]] = {}
    for run in range(n_runs):
        seed = run + 1
        sim = build_sim(scenario, seed=seed, n=n, m=m)
        sim.run(n_generations=N_GENERATIONS)
        for row in extract_results(sim):
            gen = row['gen']
            bucket = aggregated.setdefault(gen, {'h2': [], 'rg': []})
            if not np.isnan(row['mean_he_h2']):
                bucket['h2'].append(row['mean_he_h2'])
            if not np.isnan(row['mean_he_rg']):
                bucket['rg'].append(row['mean_he_rg'])
        del sim
    return aggregated


# ── Tests ──────────────────────────────────────────────────────────────────

# Table S6 expected values at generation 5, plus tolerances widened for the
# reduced default n/m grid used in tests.
#
#   scenario          h2_target  rg_target  tol_h2  tol_rg
FIGURE3_TARGETS = [
    ('RM',            0.500,     0.000,     0.15,   0.10),
    ('RM+VT',         0.552,     0.199,     0.20,   0.15),
    ('5xAM',          0.658,     0.296,     0.20,   0.20),
    ('5xAM+VT',       0.749,     0.513,     0.30,   0.25),
]


class TestFigure3:
    """Per-scenario regression of HE estimates against Table S6."""

    @pytest.mark.parametrize(
        'scenario, h2_target, rg_target, tol_h2, tol_rg',
        FIGURE3_TARGETS,
        ids=[t[0] for t in FIGURE3_TARGETS],
    )
    def test_gen5_he_matches_table_s6(self, scenario, h2_target, rg_target,
                                      tol_h2, tol_rg):
        """At generation 5, HE estimates approximately match Table S6 targets."""
        aggregated = aggregate_runs(scenario)
        assert 5 in aggregated, f"no results recorded for gen 5 of {scenario}"

        h2_runs = aggregated[5]['h2']
        rg_runs = aggregated[5]['rg']
        assert h2_runs, f"no h² estimates collected for {scenario}"
        assert rg_runs, f"no rg estimates collected for {scenario}"

        mean_h2 = float(np.mean(h2_runs))
        mean_rg = float(np.mean(rg_runs))

        assert abs(mean_h2 - h2_target) < tol_h2, (
            f"{scenario}: mean HE h² @ gen 5 = {mean_h2:.3f} "
            f"(expected ≈ {h2_target:.3f} ± {tol_h2})"
        )
        assert abs(mean_rg - rg_target) < tol_rg, (
            f"{scenario}: mean HE rg @ gen 5 = {mean_rg:.3f} "
            f"(expected ≈ {rg_target:.3f} ± {tol_rg})"
        )

    def test_xam_inflates_rg_relative_to_rm(self):
        """5xAM yields a materially larger HE rg at gen 5 than random mating."""
        rm = aggregate_runs('RM')
        xam = aggregate_runs('5xAM')
        rm_rg = float(np.mean(rm[5]['rg']))
        xam_rg = float(np.mean(xam[5]['rg']))
        assert xam_rg > rm_rg + 0.10, (
            f"5xAM HE rg = {xam_rg:.3f} not materially above RM rg = {rm_rg:.3f}"
        )


# ── Optional plot generation ────────────────────────────────────────────────

def _save_plots(aggregated_by_scenario: dict[str, dict]):
    """Write figure3_h2.png and figure3_rg.png to the current working dir."""
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = {'RM': 'tab:blue', 'RM+VT': 'tab:orange',
              '5xAM': 'tab:green', '5xAM+VT': 'tab:red'}

    for stat_key, ylabel, marker, fname in [
        ('h2', 'h²(HE)', 'o', 'figure3_h2.png'),
        ('rg', 'rg(HE)', 's', 'figure3_rg.png'),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for scenario, by_gen in aggregated_by_scenario.items():
            gens = np.array(sorted(by_gen.keys()))
            mean = np.array([np.mean(by_gen[g][stat_key]) for g in gens])
            sd = np.array([np.std(by_gen[g][stat_key]) for g in gens])
            ax.plot(gens, mean, marker + '-', label=scenario, color=colors[scenario])
            ax.fill_between(gens, mean - sd, mean + sd, alpha=0.2,
                            color=colors[scenario])
        ax.set_xlabel('Generation')
        ax.set_ylabel(ylabel)
        ax.set_title(f'HE-Estimated {ylabel} Across Generations (mean ± SD)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(fname, dpi=150)
        plt.close(fig)


if __name__ == '__main__':
    # Quick standalone run with plot regeneration, e.g.:
    #   XFTSIM_SAVE_PLOTS=1 python -m tests.manuscript.test_figure3
    scenarios = ['RM', 'RM+VT', '5xAM', '5xAM+VT']
    aggregated_by_scenario = {s: aggregate_runs(s) for s in scenarios}

    print(f"\n{'Scenario':<10} {'Gen':>3} | {'mean h²':>8} {'sd h²':>8} | "
          f"{'mean rg':>8} {'sd rg':>8}")
    print('-' * 60)
    for s, by_gen in aggregated_by_scenario.items():
        for g in sorted(by_gen):
            h2_arr = np.array(by_gen[g]['h2'])
            rg_arr = np.array(by_gen[g]['rg'])
            print(f"{s:<10} {g:>3} | {h2_arr.mean():8.3f} {h2_arr.std():8.3f} | "
                  f"{rg_arr.mean():8.3f} {rg_arr.std():8.3f}")

    if os.environ.get('XFTSIM_SAVE_PLOTS'):
        _save_plots(aggregated_by_scenario)
        print('\nWrote figure3_h2.png and figure3_rg.png')
