"""
Integration tests for callbacks and statistics within simulation.

Tests:
1. SampleStatistics collects correct keys
2. Multiple statistics generate distinct results
3. Callback + statistics run together
4. Filtered statistics (TrioFilter + SampleStatistics)
5. Results structure matches expected format
"""
import numpy as np
import pytest

from xftsim.neffect import AdditiveEffects
from xftsim.narch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.nmate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.nsim import NSimulation
from xftsim.nstats import SampleStatistics
from xftsim.nfilter import TrioFilter

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


def _make_sim(n=200, m=20, seed=42, callbacks=None, statistics=None, filters=None):
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
    eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=seed)
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(eff))
    arch.add('Y.E', NoiseComponent(variance=0.5))
    arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])
    return NSimulation(
        founder_haplotypes=hap, architecture=arch,
        mating_regime=RandomMating(offspring_per_pair=2),
        recombination_map=RecombinationMap.constant_map(m=m),
        seed=seed,
        callbacks=callbacks or [],
        statistics=statistics or [],
        filters=filters or {},
    )


class TestSampleStatisticsIntegration:
    def test_statistics_collected(self):
        sim = _make_sim(statistics=[SampleStatistics()])
        sim.run(3)
        assert len(sim.results) == 3
        for r in sim.results:
            assert 'SampleStatistics' in r.statistics

    def test_statistics_keys(self):
        sim = _make_sim(statistics=[SampleStatistics()])
        sim.run(2)
        stat = sim.results[0].statistics['SampleStatistics']
        assert 'cov' in stat
        assert 'var' in stat
        assert 'keys' in stat

    def test_statistics_keys_contain_phenotypes(self):
        sim = _make_sim(statistics=[SampleStatistics()])
        sim.run(1)
        stat = sim.results[0].statistics['SampleStatistics']
        assert 'Y' in stat['keys']
        assert 'Y.G' in stat['keys']
        assert 'Y.E' in stat['keys']

    def test_variance_positive(self):
        sim = _make_sim(statistics=[SampleStatistics()])
        sim.run(1)
        stat = sim.results[0].statistics['SampleStatistics']
        y_idx = stat['keys'].index('Y')
        assert stat['var'][y_idx] > 0


class TestMultipleStatistics:
    def test_duplicate_naming(self):
        """Two SampleStatistics should produce SampleStatistics and SampleStatistics_1."""
        sim = _make_sim(statistics=[SampleStatistics(), SampleStatistics()])
        sim.run(1)
        assert len(sim.results) == 1
        stat_keys = list(sim.results[0].statistics.keys())
        assert 'SampleStatistics' in stat_keys
        assert 'SampleStatistics_1' in stat_keys


class TestCallbacksWithStats:
    def test_callback_runs_with_stats(self):
        gen_record = []
        def cb(s):
            gen_record.append(s.generation)
        sim = _make_sim(
            callbacks=[cb],
            statistics=[SampleStatistics()],
        )
        sim.run(3)
        assert gen_record == [0, 1, 2]
        assert len(sim.results) == 3


class TestFilteredStatistics:
    def test_trio_filter_with_stats(self):
        """TrioFilter should produce filtered views used by statistics."""
        sim = _make_sim(
            statistics=[SampleStatistics()],
            filters={'trio': TrioFilter()},
        )
        sim.run(3)
        # Gen 0 has no trios (no pedigree), gen 1+ may have
        assert len(sim.results) == 3


class TestResultsStructure:
    def test_generation_result_attributes(self):
        sim = _make_sim(statistics=[SampleStatistics()])
        sim.run(2)
        r = sim.results[0]
        assert r.generation == 0
        assert isinstance(r.statistics, dict)

    def test_results_ordered_by_generation(self):
        sim = _make_sim(statistics=[SampleStatistics()])
        sim.run(4)
        gens = [r.generation for r in sim.results]
        assert gens == [0, 1, 2, 3]
