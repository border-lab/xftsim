"""
Reproducing Figure 2 from the xAM manuscript — psychiatric diagnoses under
empirical 6-way cross-trait assortative mating.

Models 6 psychiatric traits with empirical cross-mate correlations and
trait-specific heritabilities. The empirical values (one seed from the
manuscript supplement's bootstrap, ``psych_cors.csv``) are bundled
directly below so the test runs without external data.

Traits: ADHD, ALC, ANX, BIP, MDD, SCZ
- Each trait has independent genetic effects (true rg = 0)
- Empirical 6x6 cross-mate correlation matrix (non-exchangeable)
- Liability threshold model: continuous phenotypes binarized into diagnoses
- Mating operates on continuous phenotypes; HE estimated on both scales
- No vertical transmission

Tests assert the manuscript's central claim: under 6-way xAM with truly
orthogonal genetic effects, HE-estimated rg is non-zero and grows
generation by generation.

Reduced from the paper for tractability; override via env vars:

    XFTSIM_FIGURE2_N=16000 XFTSIM_FIGURE2_M=2000 \
        pytest tests/manuscript/test_figure2.py
"""

import os

import numpy as np
import pytest
import scipy.stats as stats

from xftsim.founders import founder_haplotypes_uniform_AFs
from xftsim.effect import AdditiveEffects
from xftsim.arch import Architecture
from xftsim.mate import GeneralAssortativeMating, BatchedMating
from xftsim.reproduce import RecombinationMap
from xftsim.sim import Simulation
from xftsim.stats import SampleStatistics, HasemanElstonEstimator


# ── Empirical values (manuscript supplement, seed 1) ────────────────────────

# Trait order: ADHD, ALC, ANX, BIP, MDD, SCZ
DX = ['ADHD', 'ALC', 'ANX', 'BIP', 'MDD', 'SCZ']
DX_DIAG = [f'{d}.dx' for d in DX]

# Cross-mate correlation matrix R_mate[i, j] = corr(spouse_i_trait, spouse_j_trait')
R_MATE = np.array([
    [0.392920, 0.148046, 0.156768, 0.118861, 0.173807, 0.102213],
    [0.148046, 0.298260, 0.082886, 0.113442, 0.111826, 0.145308],
    [0.156768, 0.082886, 0.341487, 0.065107, 0.189212, 0.171701],
    [0.118861, 0.113442, 0.065107, 0.116765, 0.089918, 0.124731],
    [0.173807, 0.111826, 0.189212, 0.089918, 0.178439, 0.154842],
    [0.102213, 0.145308, 0.171701, 0.124731, 0.154842, 0.283989],
])

# Trait heritabilities (continuous-liability scale)
H2 = np.array([0.610823, 0.435485, 0.588905, 0.500659, 0.236732, 0.603718])

# Prevalences and liability thresholds (from the manuscript)
PREV = np.array([0.087, 0.291, 0.316, 0.025, 0.144, 0.04])
THRESH = stats.norm.ppf(1 - PREV)


# ── Parameters ──────────────────────────────────────────────────────────────

N_INDIVIDUALS = int(os.environ.get('XFTSIM_FIGURE2_N', 8000))
N_LOCI = int(os.environ.get('XFTSIM_FIGURE2_M', 1000))
N_GENERATIONS = 6


# ── Builder ────────────────────────────────────────────────────────────────

def build_and_run_6way(seed: int = 1,
                       n: int = N_INDIVIDUALS, m: int = N_LOCI) -> Simulation:
    """Build and run the 6-way psychiatric xAM simulation for a single seed."""
    ve = 1.0 - H2

    founder_haplotypes = founder_haplotypes_uniform_AFs(n=n, m=m)

    effects = {}
    formula_lines = []
    for i, d in enumerate(DX):
        eff_name = f'{d}_eff'
        effects[eff_name] = AdditiveEffects.from_h2(
            h2=float(H2[i]), m=m, seed=100 + i + seed * 1000,
        )
        formula_lines += [
            f'{d}.G ~ genetic({eff_name})',
            f'{d}.E ~ noise({float(ve[i])})',
            f'{d} ~ {d}.G + {d}.E',
            f'{d}.dx ~ threshold({d}, {float(THRESH[i]):.6f})',
            '',
        ]

    arch = Architecture(formula='\n'.join(formula_lines), effects=effects)
    rmap = RecombinationMap(p=0.5, m=m)

    # Native solver (the default since the Hexaly dependency was dropped).
    # 'auto' batching sizes batches to the smallest that can actually attain
    # tol at K = 6; the previous fixed 1000 individuals was ~500 pairs, below
    # the reachable floor, so every batch missed the target.
    mating = BatchedMating(
        regime=GeneralAssortativeMating(
            component_names=DX,
            cross_corr=R_MATE,
            offspring_per_pair=2,
            solver_params=dict(tol=0.005),
        ),
        max_batch_size='auto',
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
            HasemanElstonEstimator(phenotype_keys=DX + DX_DIAG),
        ],
        seed=seed,
    )
    sim.run(n_generations=N_GENERATIONS)
    return sim


def extract_mean_rg_by_gen(sim: Simulation, scale: str = 'liab') -> dict[int, float]:
    """Mean off-diagonal HE rg by generation, on liability or diagnosis scale."""
    assert scale in ('liab', 'dx')
    keys_used = DX if scale == 'liab' else DX_DIAG

    out: dict[int, float] = {}
    for result in sim.results:
        gen = result.generation
        he = result.statistics.get('HasemanElstonEstimator') or {}
        if '_cov_g' not in he:
            continue
        cov_g = he['_cov_g']
        he_keys = he['_keys']
        d = np.sqrt(np.abs(np.diag(cov_g)))
        d[d < 1e-15] = 1.0
        rg = cov_g / np.outer(d, d)

        pair_vals = []
        for i in range(len(keys_used)):
            for j in range(i + 1, len(keys_used)):
                t1, t2 = keys_used[i], keys_used[j]
                if t1 in he_keys and t2 in he_keys:
                    pair_vals.append(rg[he_keys.index(t1), he_keys.index(t2)])
        if pair_vals:
            out[gen] = float(np.nanmean(pair_vals))
    return out


# ── Tests ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def sim_seed1():
    """A single seed-1 simulation, cached for the module's tests."""
    return build_and_run_6way(seed=1)


class TestFigure2:
    """The qualitative manuscript claim: 6-way xAM inflates rg from zero."""

    def test_liability_rg_grows_over_generations(self, sim_seed1):
        """Mean off-diagonal liability-scale rg at gen 5 exceeds gen 1."""
        rg_by_gen = extract_mean_rg_by_gen(sim_seed1, scale='liab')
        assert 1 in rg_by_gen and 5 in rg_by_gen, (
            f"missing gens; got {sorted(rg_by_gen)}"
        )
        assert rg_by_gen[5] > rg_by_gen[1], (
            f"liability rg did not grow: gen1 = {rg_by_gen[1]:.3f}, "
            f"gen5 = {rg_by_gen[5]:.3f}"
        )

    def test_liability_rg_materially_above_zero_at_gen5(self, sim_seed1):
        """Despite zero true pleiotropy, HE rg at gen 5 is materially > 0."""
        rg_by_gen = extract_mean_rg_by_gen(sim_seed1, scale='liab')
        assert rg_by_gen[5] > 0.05, (
            f"liability rg @ gen 5 = {rg_by_gen[5]:.3f} (expected > 0.05 "
            f"under 6-way xAM)"
        )

    def test_diagnosis_rg_also_grows(self, sim_seed1):
        """Same claim on the diagnosis (binarized) scale."""
        rg_by_gen = extract_mean_rg_by_gen(sim_seed1, scale='dx')
        assert 5 in rg_by_gen, f"missing gen 5; got {sorted(rg_by_gen)}"
        assert rg_by_gen[5] > rg_by_gen.get(1, 0.0), (
            f"diagnosis rg did not grow: gen1 = {rg_by_gen.get(1, float('nan')):.3f}, "
            f"gen5 = {rg_by_gen[5]:.3f}"
        )


if __name__ == '__main__':
    sim = build_and_run_6way(seed=1)
    rg_liab = extract_mean_rg_by_gen(sim, scale='liab')
    rg_dx = extract_mean_rg_by_gen(sim, scale='dx')

    print(f"\n{'Gen':>3}  {'rg_liab':>9}  {'rg_dx':>9}")
    print('-' * 25)
    for gen in sorted(rg_liab):
        print(f"{gen:>3}  {rg_liab[gen]:9.4f}  {rg_dx.get(gen, float('nan')):9.4f}")
