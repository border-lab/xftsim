"""
Integration tests for callback and filter interactions in NSimulation.

Tests:
1. Callbacks run each generation and accumulate results
2. sim.stop in callback terminates simulation early
3. Multiple callbacks all execute
4. Filters apply cross-generation (TrioFilter + SibPairFilter together)
5. Statistics produce results each generation
6. Simulation with all features: callbacks, filters, statistics, retention
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray
from xftsim.narch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.neffect import AdditiveEffects
from xftsim.nsim import NSimulation
from xftsim.nmate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.nstats import SampleStatistics
from xftsim.nfilter import TrioFilter, SibPairFilter

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


def _make_sim(n=100, m=10, seed=42):
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
    eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=seed)
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(eff))
    arch.add('Y.E', NoiseComponent(variance=0.5))
    arch.add('Y', AggregationComponent('Y.G + Y.E'))
    mate = RandomMating(offspring_per_pair=2)
    rmap = RecombinationMap.constant_map(m=m, p=0.5)
    return NSimulation(hap, arch, mate, rmap, seed=seed)


class TestCallbackExecution:
    def test_callback_runs_each_gen(self):
        """Callback should be called once per generation."""
        call_count = [0]
        def counter(sim):
            call_count[0] += 1

        sim = _make_sim()
        sim.callbacks = [counter]
        sim.run(5)
        assert call_count[0] == 5

    def test_callback_accumulates_data(self):
        """Callback can accumulate data across generations."""
        gen_tracker = []
        def track_gen(sim):
            gen_tracker.append(sim.generation)

        sim = _make_sim()
        sim.callbacks = [track_gen]
        sim.run(5)
        assert gen_tracker == [0, 1, 2, 3, 4]

    def test_early_stopping(self):
        """sim.stop in callback should stop simulation early."""
        def stop_at_2(sim):
            if sim.generation >= 2:
                sim.stop = True

        sim = _make_sim()
        sim.callbacks = [stop_at_2]
        sim.run(10)
        assert sim.generation == 2

    def test_multiple_callbacks(self):
        """Multiple callbacks should all execute."""
        results = {'a': 0, 'b': 0}
        def cb_a(sim):
            results['a'] += 1
        def cb_b(sim):
            results['b'] += 1

        sim = _make_sim()
        sim.callbacks = [cb_a, cb_b]
        sim.run(3)
        assert results['a'] == 3
        assert results['b'] == 3

    def test_callback_can_read_phenotypes(self):
        """Callback should be able to read current generation phenotypes."""
        means = []
        def track_mean(sim):
            pheno = sim.phenotype_history[sim.generation]
            means.append(np.mean(pheno['Y']))

        sim = _make_sim()
        sim.callbacks = [track_mean]
        sim.run(3)
        assert len(means) == 3
        assert all(np.isfinite(m) for m in means)


class TestFilterExecution:
    def test_trio_filter_runs(self):
        """TrioFilter should produce results when applicable."""
        sim = _make_sim()
        sim.filters = {'trio': TrioFilter()}
        sim.run(3)  # Need 2+ gens for trios
        assert sim.generation == 2

    def test_sibpair_filter_runs(self):
        """SibPairFilter should produce results."""
        sim = _make_sim()
        sim.filters = {'sibpair': SibPairFilter()}
        sim.run(3)
        assert sim.generation == 2

    def test_both_filters(self):
        """Both filters should run together."""
        sim = _make_sim()
        sim.filters = {'trio': TrioFilter(), 'sibpair': SibPairFilter()}
        sim.run(3)
        assert sim.generation == 2


class TestStatisticsExecution:
    def test_statistics_produce_results(self):
        """Statistics should produce GenerationResult entries."""
        sim = _make_sim()
        sim.statistics = [SampleStatistics()]
        sim.run(3)
        # Results should have entries for computed generations
        assert len(sim.results) > 0

    def test_statistics_with_callbacks(self):
        """Statistics and callbacks should not interfere."""
        gen_list = []
        def track(sim):
            gen_list.append(sim.generation)

        sim = _make_sim()
        sim.statistics = [SampleStatistics()]
        sim.callbacks = [track]
        sim.run(3)
        assert len(gen_list) == 3
        assert len(sim.results) > 0


class TestFullFeatureSim:
    def test_everything_together(self):
        """Simulation with callbacks, filters, statistics, and retention."""
        gen_list = []
        def track(sim):
            gen_list.append(sim.generation)

        sim = _make_sim()
        sim.callbacks = [track]
        sim.filters = {'trio': TrioFilter()}
        sim.statistics = [SampleStatistics()]
        sim.retain_haplotypes = 2
        sim.retain_phenotypes = 2
        sim.run(5)

        assert sim.generation == 4
        assert len(gen_list) == 5
        # Current gen should be in history
        assert sim.generation in sim.phenotype_history
        # Phenotypes at current gen should be finite
        pheno = sim.phenotype_history[sim.generation]
        assert np.all(np.isfinite(pheno['Y']))

    def test_early_stop_with_filters_stats(self):
        """Early stopping should work with filters and statistics."""
        def stop_at_1(sim):
            if sim.generation >= 1:
                sim.stop = True

        sim = _make_sim()
        sim.callbacks = [stop_at_1]
        sim.filters = {'trio': TrioFilter()}
        sim.statistics = [SampleStatistics()]
        sim.run(10)
        assert sim.generation == 1
