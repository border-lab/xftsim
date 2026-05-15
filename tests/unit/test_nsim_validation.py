"""
Unit tests for Simulation validation, callbacks, and edge cases.

Tests:
1. Dimension mismatch between effects and haplotypes
2. n_generations=1 (only gen 0)
3. Callback execution
4. Early stopping via callback
5. sim.stop flag
6. Current generation haplotypes/phenotypes properties
7. repr
8. Duplicate statistics naming
9. Multiple filters
"""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation

from xftsim.arch import (
    Architecture, GeneticComponent, NoiseComponent, AggregationComponent,
)
from xftsim.effect import AdditiveEffects
from xftsim.mate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.sim import Simulation
from xftsim.stats import SampleStatistics
from xftsim.filters import TrioFilter, SibPairFilter


def _make_sim(n=20, m=10, h2=0.5, seed=42, **kwargs):
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
    eff = AdditiveEffects.from_h2(m=m, h2=h2, seed=seed)
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(eff))
    arch.add('Y.E', NoiseComponent(variance=1.0 - h2))
    arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])
    mating = RandomMating(offspring_per_pair=2)
    rmap = RecombinationMap.constant_map(m=m)
    return Simulation(
        founder_haplotypes=hap, architecture=arch,
        mating_regime=mating, recombination_map=rmap,
        seed=seed, **kwargs,
    )


class TestDimensionValidation:
    def test_effect_m_mismatch_raises(self):
        """Effects with wrong m should raise ValueError at run time."""
        hap = TestSimulation.founder_haplotypes(n=20, m=10, seed=42)
        eff = AdditiveEffects.from_h2(m=15, h2=0.5, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])
        mating = RandomMating()
        rmap = RecombinationMap.constant_map(m=10)
        sim = Simulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=mating, recombination_map=rmap,
        )
        with pytest.raises(ValueError, match="Effect dimension mismatch"):
            sim.run(1)


class TestSingleGeneration:
    def test_run_one_generation(self):
        """run(1) should only compute gen 0 phenotypes, no mating."""
        sim = _make_sim()
        sim.run(1)
        assert sim.generation == 0
        assert 0 in sim.phenotype_history
        assert 'Y' in sim.phenotype_history[0]
        assert 0 not in sim.pedigree_history


class TestCallbacks:
    def test_callback_called_each_gen(self):
        """Callback should be called once per generation."""
        call_log = []
        def my_callback(sim):
            call_log.append(sim.generation)

        sim = _make_sim(callbacks=[my_callback])
        sim.run(3)
        assert call_log == [0, 1, 2]

    def test_early_stopping(self):
        """Setting sim.stop=True should halt after that generation."""
        def stop_at_gen_1(sim):
            if sim.generation == 1:
                sim.stop = True

        sim = _make_sim(callbacks=[stop_at_gen_1])
        sim.run(5)
        assert sim.generation == 1

    def test_multiple_callbacks(self):
        log1, log2 = [], []
        sim = _make_sim(callbacks=[lambda s: log1.append(s.generation),
                                    lambda s: log2.append(s.generation)])
        sim.run(2)
        assert log1 == [0, 1]
        assert log2 == [0, 1]

    def test_early_stopping_at_gen0(self):
        """Stopping at gen 0 should not crash."""
        def stop_at_gen_0(sim):
            sim.stop = True

        sim = _make_sim(callbacks=[stop_at_gen_0])
        sim.run(5)
        assert sim.generation == 0


class TestCurrentProperties:
    def test_haplotypes_property(self):
        sim = _make_sim()
        sim.run(2)
        assert sim.haplotypes is sim.haplotype_history[sim.generation]

    def test_phenotypes_property(self):
        sim = _make_sim()
        sim.run(2)
        assert sim.phenotypes is sim.phenotype_history[sim.generation]


class TestRepr:
    def test_repr(self):
        sim = _make_sim()
        sim.run(1)
        r = repr(sim)
        assert 'Simulation' in r
        assert 'generation=0' in r


class TestDuplicateStatistics:
    def test_duplicate_stat_names(self):
        """Multiple statistics of the same type should get unique keys."""
        sim = _make_sim(statistics=[SampleStatistics(), SampleStatistics()])
        sim.run(2)
        assert len(sim.results) == 2
        gen0_result = sim.results[0]
        assert 'SampleStatistics' in gen0_result.statistics
        assert 'SampleStatistics_1' in gen0_result.statistics


class TestMultipleFilters:
    def test_trio_and_sib_filters(self):
        """Both trio and sib pair filters should run without error."""
        sim = _make_sim(
            n=40, m=10,
            filters={'trio': TrioFilter(), 'sib': SibPairFilter()},
            statistics=[SampleStatistics()],
        )
        sim.run(3)
        assert sim.generation == 2
        assert len(sim.results) == 3


class TestContinueRun:
    def test_continue_from_gen0(self):
        """continue_run after run(1) should produce additional generations."""
        sim = _make_sim()
        sim.run(1)
        assert sim.generation == 0
        sim.continue_run(2)
        assert sim.generation == 2
        assert 2 in sim.phenotype_history

    def test_continue_zero_additional(self):
        """continue_run(0) should be a no-op."""
        sim = _make_sim()
        sim.run(2)
        gen_before = sim.generation
        sim.continue_run(0)
        assert sim.generation == gen_before
