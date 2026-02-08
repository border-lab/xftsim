"""
Integration test: filters and statistics working together in simulation.

Tests:
1. SampleStatistics collected each generation with cov/var/keys
2. TrioFilter produces filtered views for statistics
3. Statistics results list has correct length
4. Filter + statistics work together
5. Statistics + callbacks all work together
6. Variance from statistics is positive
"""
import numpy as np
import pytest

from xftsim.neffect import AdditiveEffects
from xftsim.narch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.nmate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.nsim import NSimulation
from xftsim.nstats import SampleStatistics
from xftsim.nfilter import TrioFilter, SibPairFilter

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


def _make_sim(n=200, m=20, filters=None, statistics=None, callbacks=None, seed=42):
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
    eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=seed)
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(eff))
    arch.add('Y.E', NoiseComponent(variance=0.5))
    arch.add('Y', AggregationComponent('Y.G + Y.E'))

    return NSimulation(
        founder_haplotypes=hap, architecture=arch,
        mating_regime=RandomMating(offspring_per_pair=2),
        recombination_map=RecombinationMap.constant_map(m=m),
        seed=seed,
        statistics=statistics or [],
        filters=filters or {},
        callbacks=callbacks or [],
        retain_phenotypes=10,
        retain_haplotypes=10,
    )


class TestFilterStatsPipeline:
    def test_sample_statistics_structure(self):
        """SampleStatistics returns dict with cov, var, keys."""
        sim = _make_sim(statistics=[SampleStatistics()])
        sim.run(3)

        assert len(sim.results) == 3
        for i, result in enumerate(sim.results):
            assert result.generation == i
            assert 'SampleStatistics' in result.statistics
            stats = result.statistics['SampleStatistics']
            assert 'cov' in stats
            assert 'var' in stats
            assert 'keys' in stats
            assert 'Y' in stats['keys']

    def test_stats_var_positive(self):
        """Variance from statistics should be positive for all components."""
        sim = _make_sim(statistics=[SampleStatistics()])
        sim.run(3)

        for result in sim.results:
            stats = result.statistics['SampleStatistics']
            assert np.all(stats['var'] > 0), \
                f"Gen {result.generation}: all variances should be positive"

    def test_stats_cov_symmetric(self):
        """Covariance matrix should be symmetric."""
        sim = _make_sim(statistics=[SampleStatistics()])
        sim.run(2)

        for result in sim.results:
            cov = result.statistics['SampleStatistics']['cov']
            np.testing.assert_array_almost_equal(cov, cov.T)

    def test_stats_with_filters(self):
        """Statistics and filters should both work in same simulation."""
        sim = _make_sim(
            statistics=[SampleStatistics()],
            filters={'trio': TrioFilter()},
        )
        sim.run(3)

        # Results should still be collected
        assert len(sim.results) == 3
        for result in sim.results:
            assert 'SampleStatistics' in result.statistics

    def test_stats_filter_callback_together(self):
        """All three features should work together without conflict."""
        gen_log = []
        def log_gen(sim):
            gen_log.append(sim.generation)

        sim = _make_sim(
            statistics=[SampleStatistics()],
            filters={'trio': TrioFilter()},
            callbacks=[log_gen],
        )
        sim.run(4)

        assert gen_log == [0, 1, 2, 3]
        assert len(sim.results) == 4
        for result in sim.results:
            assert 'SampleStatistics' in result.statistics

    def test_no_stats_no_results(self):
        """Without statistics, results list should be empty."""
        sim = _make_sim(statistics=[])
        sim.run(3)
        assert len(sim.results) == 0

    def test_filters_alone_no_results(self):
        """Filters without statistics should not produce results."""
        sim = _make_sim(filters={'trio': TrioFilter()}, statistics=[])
        sim.run(3)
        assert len(sim.results) == 0
