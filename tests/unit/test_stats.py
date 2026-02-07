"""
Tests for statistics module (SampleStatistics, GenerationResult).
"""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation

from xftsim.nstats import SampleStatistics, GenerationResult
from xftsim.nsim import NSimulation
from xftsim.nfilter import TrioFilter


class TestSampleStatistics:
    def test_shape(self):
        """SampleStatistics should return a cov matrix matching number of phenotype keys."""
        hap = TestSimulation.founder_haplotypes(n=500, m=50)
        arch = TestSimulation.simple_architecture(m=50, h2=0.5)
        rm = TestSimulation.mating_regime()
        rmap = TestSimulation.recombination_map(m=50)
        sim = NSimulation(hap, arch, rm, rmap, seed=42)
        sim.run(1)

        stat = SampleStatistics()
        result = stat.estimate(sim.phenotype_history, {}, 0)
        k = len(list(sim.phenotype_history[0].keys))
        assert result['cov'].shape == (k, k)
        assert len(result['var']) == k
        assert len(result['keys']) == k

    def test_diagonal(self):
        """Diagonal of cov should equal var."""
        hap = TestSimulation.founder_haplotypes(n=500, m=50)
        arch = TestSimulation.simple_architecture(m=50, h2=0.5)
        rm = TestSimulation.mating_regime()
        rmap = TestSimulation.recombination_map(m=50)
        sim = NSimulation(hap, arch, rm, rmap, seed=42)
        sim.run(1)

        stat = SampleStatistics()
        result = stat.estimate(sim.phenotype_history, {}, 0)
        np.testing.assert_allclose(np.diag(result['cov']), result['var'])


class TestGenerationResult:
    def test_dataclass(self):
        """GenerationResult should be a simple dataclass."""
        gr = GenerationResult(generation=5, statistics={'foo': 42})
        assert gr.generation == 5
        assert gr.statistics['foo'] == 42

    def test_default_empty_stats(self):
        """Default statistics should be empty dict."""
        gr = GenerationResult(generation=0)
        assert gr.statistics == {}


class TestSimWithStats:
    def test_produces_results(self):
        """Simulation with statistics should populate results list."""
        hap = TestSimulation.founder_haplotypes(n=500, m=50)
        arch = TestSimulation.simple_architecture(m=50, h2=0.5)
        rm = TestSimulation.mating_regime()
        rmap = TestSimulation.recombination_map(m=50)
        sim = NSimulation(hap, arch, rm, rmap, seed=42,
                         statistics=[SampleStatistics()])
        sim.run(3)
        assert len(sim.results) == 3  # one per generation (0, 1, 2)

    def test_multiple_statistics(self):
        """Multiple statistics of the same type should get deduplicated keys."""
        hap = TestSimulation.founder_haplotypes(n=500, m=50)
        arch = TestSimulation.simple_architecture(m=50, h2=0.5)
        rm = TestSimulation.mating_regime()
        rmap = TestSimulation.recombination_map(m=50)
        sim = NSimulation(hap, arch, rm, rmap, seed=42,
                         statistics=[SampleStatistics(), SampleStatistics()])
        sim.run(2)
        # Two SampleStatistics → keys 'SampleStatistics' and 'SampleStatistics_1'
        stats = sim.results[0].statistics
        assert 'SampleStatistics' in stats
        assert 'SampleStatistics_1' in stats
        assert len(stats) == 2

    def test_custom_statistic(self):
        """Custom Statistic subclass should work in the simulation."""
        from xftsim.nstats import Statistic

        class MeanStatistic(Statistic):
            def estimate(self, phenotype_history, filtered_views, generation):
                pheno = phenotype_history[generation]
                return {key: np.mean(pheno[key]) for key in pheno.keys}

        hap = TestSimulation.founder_haplotypes(n=500, m=50)
        arch = TestSimulation.simple_architecture(m=50, h2=0.5)
        rm = TestSimulation.mating_regime()
        rmap = TestSimulation.recombination_map(m=50)
        sim = NSimulation(hap, arch, rm, rmap, seed=42,
                         statistics=[SampleStatistics(), MeanStatistic()])
        sim.run(2)
        stats = sim.results[0].statistics
        assert 'SampleStatistics' in stats
        assert 'MeanStatistic' in stats

    def test_results_per_generation(self):
        """Each result should have the correct generation number."""
        hap = TestSimulation.founder_haplotypes(n=500, m=50)
        arch = TestSimulation.simple_architecture(m=50, h2=0.5)
        rm = TestSimulation.mating_regime()
        rmap = TestSimulation.recombination_map(m=50)
        sim = NSimulation(hap, arch, rm, rmap, seed=42,
                         statistics=[SampleStatistics()])
        sim.run(3)
        gens = [r.generation for r in sim.results]
        assert gens == [0, 1, 2]
        for r in sim.results:
            assert 'SampleStatistics' in r.statistics
