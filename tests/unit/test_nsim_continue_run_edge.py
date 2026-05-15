"""
Unit tests for Simulation.continue_run edge cases.

Tests:
1. continue_run(0) is a no-op
2. continue_run with statistics collects results
3. continue_run with filters applies them
4. continue_run preserves generation continuity
5. run(1) then continue_run(2) gives 3 total generations
"""
import numpy as np
import pytest

from xftsim.effect import AdditiveEffects
from xftsim.arch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.mate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.sim import Simulation
from xftsim.stats import SampleStatistics
from xftsim.filters import TrioFilter

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


def _make_sim(n=100, m=20, seed=42, **kwargs):
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
    eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=seed)
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(eff))
    arch.add('Y.E', NoiseComponent(variance=0.5))
    arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])
    return Simulation(
        founder_haplotypes=hap, architecture=arch,
        mating_regime=RandomMating(offspring_per_pair=2),
        recombination_map=RecombinationMap.constant_map(m=m),
        seed=seed, **kwargs,
    )


class TestContinueRunZero:
    def test_continue_run_zero_is_noop(self):
        """continue_run(0) should not advance the simulation."""
        sim = _make_sim()
        sim.run(2)
        gen_before = sim.generation
        sim.continue_run(0)
        assert sim.generation == gen_before

    def test_continue_run_zero_preserves_histories(self):
        """continue_run(0) should not modify histories."""
        sim = _make_sim()
        sim.run(2)
        hap_gens = set(sim.haplotype_history.keys())
        pheno_gens = set(sim.phenotype_history.keys())
        sim.continue_run(0)
        assert set(sim.haplotype_history.keys()) == hap_gens
        assert set(sim.phenotype_history.keys()) == pheno_gens


class TestContinueRunWithStatsAndFilters:
    def test_continue_run_collects_statistics(self):
        """Statistics should be collected during continue_run."""
        sim = _make_sim(statistics=[SampleStatistics()])
        sim.run(2)
        n_results_after_run = len(sim.results)
        sim.continue_run(2)
        # Should have additional results from continued generations
        assert len(sim.results) == n_results_after_run + 2

    def test_continue_run_with_filters(self):
        """Filters should run during continue_run."""
        sim = _make_sim(
            statistics=[SampleStatistics()],
            filters={'trio': TrioFilter()},
        )
        sim.run(2)
        sim.continue_run(1)
        assert sim.generation == 2
        assert len(sim.results) == 3


class TestContinueRunGenerations:
    def test_run_then_continue_generations(self):
        """run(1) then continue_run(2) → generation 2."""
        sim = _make_sim()
        sim.run(1)
        assert sim.generation == 0
        sim.continue_run(2)
        assert sim.generation == 2

    def test_run_then_continue_phenotypes(self):
        """After run + continue, should have phenotypes for continued gens."""
        sim = _make_sim()
        sim.run(2)
        assert sim.generation == 1
        sim.continue_run(2)
        assert sim.generation == 3
        # Current gen phenotype should exist
        assert 3 in sim.phenotype_history
        assert 'Y' in sim.phenotypes

    def test_continue_run_with_callbacks(self):
        """Callbacks should fire during continue_run."""
        gen_record = []
        def cb(s):
            gen_record.append(s.generation)
        sim = _make_sim(callbacks=[cb])
        sim.run(2)
        gen_record.clear()
        sim.continue_run(2)
        assert gen_record == [2, 3]

    def test_continue_run_early_stop(self):
        """sim.stop in callback during continue_run should stop."""
        def cb(s):
            if s.generation >= 3:
                s.stop = True
        sim = _make_sim(callbacks=[cb])
        sim.run(2)  # gens 0, 1
        sim.continue_run(5)  # should stop at gen 3
        assert sim.generation <= 3
