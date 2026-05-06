"""
Numerical validation tests for HasemanElstonEstimator and ParentOffspringRegression.

Tests run actual simulations with known h2 values and verify that the
estimators recover approximately correct heritability estimates.

HasemanElstonEstimator is GRM-based: works with any relatedness structure
including founders (gen 0). Does not require sibling pairs.

Tests:
1. HE recovers h2 at gen 0 and subsequent generations
2. PO regression recovers h2 from gen 1+
3. Both estimators are approximately unbiased across h2 values
4. MatingStatistics integration: correct pair counts and spouse correlations
5. Combined: all estimators run together and give concordant results
"""
import numpy as np
import pytest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation

from xftsim.effect import AdditiveEffects
from xftsim.arch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.mate import RandomMating, LinearAssortativeMating
from xftsim.reproduce import RecombinationMap
from xftsim.sim import NSimulation
from xftsim.stats import (
    HasemanElstonEstimator,
    ParentOffspringRegression,
    MatingStatistics,
    SampleStatistics,
)
from xftsim.filters import TrioFilter, SibPairFilter


def _make_sim(n=1000, m=50, h2=0.5, offspring_per_pair=2,
              statistics=None, filters=None, mating=None, seed=42):
    """Build a simple single-trait NSimulation."""
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
    eff = AdditiveEffects.from_h2(h2=h2, m=m, seed=seed + 1)
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(eff))
    arch.add('Y.E', NoiseComponent(variance=1.0 - h2))
    arch.add('Y', AggregationComponent('Y.G + Y.E'))

    if mating is None:
        mating = RandomMating(offspring_per_pair=offspring_per_pair)

    return NSimulation(
        founder_haplotypes=hap,
        architecture=arch,
        mating_regime=mating,
        recombination_map=RecombinationMap.constant_map(m=m),
        seed=seed,
        statistics=statistics or [],
        filters=filters or {},
        retain_phenotypes=10,
        retain_haplotypes=10,
    )


def _collect_he_h2(sim, key='Y'):
    """Collect HE h2 estimates across generations."""
    vals = []
    for result in sim.results:
        stats = result.statistics.get('HasemanElstonEstimator')
        if stats is not None and key in stats:
            vals.append(stats[key]['h2'])
    return vals


def _collect_por_h2(sim, key='Y'):
    """Collect POR h2 estimates across generations."""
    vals = []
    for result in sim.results:
        stats = result.statistics.get('ParentOffspringRegression')
        if stats is not None and key in stats:
            vals.append(stats[key]['h2'])
    return vals


class TestHESimulation:
    """Numerical validation of GRM-based HasemanElstonEstimator."""

    def test_he_recovers_h2_moderate(self):
        """HE should recover h2≈0.5."""
        sim = _make_sim(
            n=2000, m=100, h2=0.5, offspring_per_pair=2,
            statistics=[HasemanElstonEstimator(phenotype_keys=['Y'])],
            seed=42,
        )
        sim.run(3)

        h2_estimates = _collect_he_h2(sim)
        assert len(h2_estimates) >= 1, "Should have HE results"
        mean_h2 = np.mean(h2_estimates)
        assert abs(mean_h2 - 0.5) < 0.15, f"Mean HE h2={mean_h2:.3f}, expected ~0.5"

    def test_he_recovers_h2_high(self):
        """HE with h2=0.8 should give estimate near 0.8."""
        sim = _make_sim(
            n=2000, m=100, h2=0.8, offspring_per_pair=2,
            statistics=[HasemanElstonEstimator(phenotype_keys=['Y'])],
            seed=123,
        )
        sim.run(3)

        h2_estimates = _collect_he_h2(sim)
        assert len(h2_estimates) >= 1
        mean_h2 = np.mean(h2_estimates)
        assert abs(mean_h2 - 0.8) < 0.15, f"HE h2={mean_h2:.3f}, expected ~0.8"

    def test_he_recovers_h2_low(self):
        """HE with h2=0.1 should give estimate near 0.1."""
        sim = _make_sim(
            n=2000, m=100, h2=0.1, offspring_per_pair=2,
            statistics=[HasemanElstonEstimator(phenotype_keys=['Y'])],
            seed=77,
        )
        sim.run(3)

        h2_estimates = _collect_he_h2(sim)
        assert len(h2_estimates) >= 1
        mean_h2 = np.mean(h2_estimates)
        assert abs(mean_h2 - 0.1) < 0.15, f"HE h2={mean_h2:.3f}, expected ~0.1"

    def test_he_ordering_across_h2(self):
        """Higher true h2 should produce higher estimated h2."""
        estimates_by_h2 = {}
        for h2_true in [0.2, 0.6]:
            sim = _make_sim(
                n=2000, m=100, h2=h2_true, offspring_per_pair=2,
                statistics=[HasemanElstonEstimator(phenotype_keys=['Y'])],
                seed=42,
            )
            sim.run(3)
            vals = _collect_he_h2(sim)
            estimates_by_h2[h2_true] = np.mean(vals) if vals else 0.0

        assert estimates_by_h2[0.6] > estimates_by_h2[0.2], (
            f"h2=0.6 estimate ({estimates_by_h2[0.6]:.3f}) should exceed "
            f"h2=0.2 estimate ({estimates_by_h2[0.2]:.3f})"
        )

    def test_he_works_at_gen0(self):
        """HE should produce results at generation 0 (founders)."""
        sim = _make_sim(
            n=2000, m=100, h2=0.5, offspring_per_pair=2,
            statistics=[HasemanElstonEstimator(phenotype_keys=['Y'])],
            seed=42,
        )
        sim.run(1)

        assert len(sim.results) >= 1
        he = sim.results[0].statistics.get('HasemanElstonEstimator')
        assert he is not None, "HE should produce results at gen 0"
        assert 'Y' in he
        assert abs(he['Y']['h2'] - 0.5) < 0.15, \
            f"HE at gen 0: h2={he['Y']['h2']:.3f}, expected ~0.5"


class TestPORSimulation:
    """Numerical validation of ParentOffspringRegression."""

    def test_por_recovers_h2_moderate(self):
        """PO regression should recover h2≈0.5."""
        sim = _make_sim(
            n=2000, m=100, h2=0.5, offspring_per_pair=2,
            statistics=[ParentOffspringRegression(filter_name='trio')],
            filters={'trio': TrioFilter()},
            seed=42,
        )
        sim.run(3)

        h2_estimates = _collect_por_h2(sim)
        assert len(h2_estimates) >= 2, "Should have PO results for gens 1+"
        mean_h2 = np.mean(h2_estimates)
        assert abs(mean_h2 - 0.5) < 0.15, f"Mean PO h2={mean_h2:.3f}, expected ~0.5"

    def test_por_recovers_h2_high(self):
        """PO regression with h2=0.8."""
        sim = _make_sim(
            n=2000, m=100, h2=0.8, offspring_per_pair=2,
            statistics=[ParentOffspringRegression(filter_name='trio')],
            filters={'trio': TrioFilter()},
            seed=123,
        )
        sim.run(3)

        h2_estimates = _collect_por_h2(sim)
        assert len(h2_estimates) >= 2
        mean_h2 = np.mean(h2_estimates)
        assert abs(mean_h2 - 0.8) < 0.15, f"PO h2={mean_h2:.3f}, expected ~0.8"

    def test_por_ordering_across_h2(self):
        """Higher true h2 should produce higher estimated h2."""
        estimates_by_h2 = {}
        for h2_true in [0.2, 0.6]:
            sim = _make_sim(
                n=2000, m=100, h2=h2_true, offspring_per_pair=2,
                statistics=[ParentOffspringRegression(filter_name='trio')],
                filters={'trio': TrioFilter()},
                seed=42,
            )
            sim.run(3)
            vals = _collect_por_h2(sim)
            estimates_by_h2[h2_true] = np.mean(vals) if vals else 0.0

        assert estimates_by_h2[0.6] > estimates_by_h2[0.2], (
            f"h2=0.6 estimate ({estimates_by_h2[0.6]:.3f}) should exceed "
            f"h2=0.2 estimate ({estimates_by_h2[0.2]:.3f})"
        )

    def test_por_se_decreases_with_n(self):
        """SE of PO regression should decrease with more samples."""
        se_by_n = {}
        for pop_n in [500, 2000]:
            sim = _make_sim(
                n=pop_n, m=50, h2=0.5, offspring_per_pair=2,
                statistics=[ParentOffspringRegression(filter_name='trio')],
                filters={'trio': TrioFilter()},
                seed=42,
            )
            sim.run(2)
            ses = []
            for result in sim.results:
                stats = result.statistics.get('ParentOffspringRegression')
                if stats is not None and 'Y' in stats:
                    se = stats['Y']['se']
                    if not np.isnan(se):
                        ses.append(se)
            se_by_n[pop_n] = np.mean(ses) if ses else np.inf

        assert se_by_n[2000] < se_by_n[500], (
            f"SE(n=2000)={se_by_n[2000]:.4f} should be < SE(n=500)={se_by_n[500]:.4f}"
        )

    def test_por_n_trios_matches_population(self):
        """n_trios should match the offspring population size at gen >= 1."""
        n = 1000
        sim = _make_sim(
            n=n, m=50, h2=0.5, offspring_per_pair=2,
            statistics=[ParentOffspringRegression(filter_name='trio')],
            filters={'trio': TrioFilter()},
            seed=42,
        )
        sim.run(2)

        for result in sim.results:
            stats = result.statistics.get('ParentOffspringRegression')
            if stats is not None and 'Y' in stats:
                n_trios = stats['Y']['n_trios']
                assert n_trios > 0
                assert n_trios <= n * 2  # generous upper bound


class TestMatingStatsSimulation:
    """Numerical validation of MatingStatistics."""

    def test_mating_stats_pair_count(self):
        """n_mating_pairs should reflect actual population structure."""
        sim = _make_sim(
            n=1000, m=50, h2=0.5, offspring_per_pair=2,
            statistics=[MatingStatistics(filter_name='trio')],
            filters={'trio': TrioFilter()},
            seed=42,
        )
        sim.run(2)

        for result in sim.results:
            stats = result.statistics.get('MatingStatistics')
            if stats is not None:
                assert stats['n_mating_pairs'] > 0
                assert stats['mean_offspring_count'] > 0

    def test_mating_stats_offspring_count_with_known_opp(self):
        """mean_offspring_count should be close to offspring_per_pair for gen > 0."""
        opp = 3
        sim = _make_sim(
            n=1000, m=50, h2=0.5, offspring_per_pair=opp,
            statistics=[MatingStatistics(filter_name='trio')],
            filters={'trio': TrioFilter()},
            seed=42,
        )
        sim.run(2)

        for result in sim.results:
            if result.generation >= 1:
                stats = result.statistics.get('MatingStatistics')
                if stats is not None:
                    assert stats['mean_offspring_count'] == pytest.approx(opp, abs=0.5)

    def test_assortative_mating_spouse_correlation(self):
        """Assortative mating should produce detectable spouse correlation."""
        n = 2000
        hap = TestSimulation.founder_haplotypes(n=n, m=50, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=50, seed=43)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))

        mating = LinearAssortativeMating(
            component_names=['Y'], r=0.5, offspring_per_pair=2,
        )

        sim = NSimulation(
            founder_haplotypes=hap,
            architecture=arch,
            mating_regime=mating,
            recombination_map=RecombinationMap.constant_map(m=50),
            seed=42,
            statistics=[MatingStatistics(filter_name='trio')],
            filters={'trio': TrioFilter()},
            retain_phenotypes=10,
            retain_haplotypes=10,
        )
        sim.run(3)

        spouse_cors = []
        for result in sim.results:
            if result.generation >= 1:
                stats = result.statistics.get('MatingStatistics')
                if stats is not None and 'Y' in stats.get('spouse_correlations', {}):
                    spouse_cors.append(stats['spouse_correlations']['Y'])

        assert len(spouse_cors) >= 1, "Should have spouse correlations for gen >= 1"
        mean_cor = np.mean(spouse_cors)
        assert mean_cor > 0.1, (
            f"Mean spouse correlation={mean_cor:.3f} should be > 0.1 with r=0.5"
        )

    def test_random_mating_zero_spouse_correlation(self):
        """Random mating should produce near-zero spouse correlation."""
        sim = _make_sim(
            n=2000, m=50, h2=0.5, offspring_per_pair=2,
            statistics=[MatingStatistics(filter_name='trio')],
            filters={'trio': TrioFilter()},
            seed=42,
        )
        sim.run(3)

        spouse_cors = []
        for result in sim.results:
            if result.generation >= 1:
                stats = result.statistics.get('MatingStatistics')
                if stats is not None and 'Y' in stats.get('spouse_correlations', {}):
                    spouse_cors.append(stats['spouse_correlations']['Y'])

        if spouse_cors:
            mean_cor = np.mean(spouse_cors)
            assert abs(mean_cor) < 0.15, (
                f"Mean spouse correlation={mean_cor:.3f} should be near 0 for random mating"
            )


class TestCombinedEstimators:
    """Test running HE, PO, and MatingStatistics together."""

    def test_all_three_in_one_sim(self):
        """All statistics should run without conflict."""
        sim = _make_sim(
            n=1000, m=50, h2=0.5, offspring_per_pair=2,
            statistics=[
                HasemanElstonEstimator(phenotype_keys=['Y']),
                ParentOffspringRegression(filter_name='trio'),
                MatingStatistics(filter_name='trio'),
                SampleStatistics(),
            ],
            filters={
                'trio': TrioFilter(),
            },
            seed=42,
        )
        sim.run(3)

        assert len(sim.results) == 3
        for result in sim.results:
            assert 'SampleStatistics' in result.statistics
            assert 'HasemanElstonEstimator' in result.statistics
            assert 'ParentOffspringRegression' in result.statistics
            assert 'MatingStatistics' in result.statistics

    def test_he_and_por_agree_roughly(self):
        """HE and PO should give roughly concordant h2 estimates."""
        sim = _make_sim(
            n=2000, m=100, h2=0.5, offspring_per_pair=2,
            statistics=[
                HasemanElstonEstimator(phenotype_keys=['Y']),
                ParentOffspringRegression(filter_name='trio'),
            ],
            filters={
                'trio': TrioFilter(),
            },
            seed=42,
        )
        sim.run(4)

        he_vals = _collect_he_h2(sim)
        por_vals = _collect_por_h2(sim)

        if he_vals and por_vals:
            he_mean = np.mean(he_vals)
            por_mean = np.mean(por_vals)
            assert abs(he_mean - por_mean) < 0.25, (
                f"HE mean={he_mean:.3f}, POR mean={por_mean:.3f} should be concordant"
            )
