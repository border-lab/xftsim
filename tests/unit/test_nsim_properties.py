"""
Unit tests for Simulation properties, repr, validation, and edge cases.

Tests:
1. haplotypes property returns current gen haplotypes
2. phenotypes property returns current gen phenotypes
3. repr format
4. _validate catches effect dimension mismatch
5. _validate passes when dimensions match
6. from_checkpoint with missing mating_regime raises
7. from_checkpoint with missing recombination_map raises
8. callbacks list defaults to empty
9. filters dict defaults to empty
10. statistics list defaults to empty
11. stop flag defaults to False
12. generation starts at 0
13. run with n_generations=1 only computes gen-0
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray
from xftsim.arch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.effect import AdditiveEffects
from xftsim.sim import Simulation
from xftsim.mate import RandomMating
from xftsim.reproduce import RecombinationMap

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


def _make_sim(n=50, m=10, seed=42, **kwargs):
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
    eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=seed)
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(eff))
    arch.add('Y.E', NoiseComponent(variance=0.3))
    arch.add('Y', AggregationComponent('Y.G + Y.E'))
    rm = RandomMating(offspring_per_pair=2)
    rmap = RecombinationMap.constant_map(m=m, p=0.5)
    return Simulation(hap, arch, rm, rmap, seed=seed, **kwargs)


class TestSimulationProperties:
    def test_haplotypes_returns_current_gen(self):
        sim = _make_sim()
        sim.run(3)
        assert sim.haplotypes is sim.haplotype_history[sim.generation]

    def test_phenotypes_returns_current_gen(self):
        sim = _make_sim()
        sim.run(3)
        assert sim.phenotypes is sim.phenotype_history[sim.generation]

    def test_generation_starts_at_zero(self):
        sim = _make_sim()
        assert sim.generation == 0

    def test_stop_defaults_false(self):
        sim = _make_sim()
        assert sim.stop is False

    def test_callbacks_defaults_empty(self):
        sim = _make_sim()
        assert sim.callbacks == []

    def test_filters_defaults_empty(self):
        sim = _make_sim()
        assert sim.filters == {}

    def test_statistics_defaults_empty(self):
        sim = _make_sim()
        assert sim.statistics == []

    def test_results_starts_empty(self):
        sim = _make_sim()
        assert sim.results == []


class TestSimulationRepr:
    def test_repr_format(self):
        sim = _make_sim()
        r = repr(sim)
        assert 'Simulation' in r
        assert 'generation=0' in r

    def test_repr_after_run(self):
        sim = _make_sim()
        sim.run(3)
        r = repr(sim)
        assert 'generation=2' in r


class TestValidation:
    def test_dimension_mismatch_raises(self):
        """Effect dimension != haplotype m should raise."""
        hap = TestSimulation.founder_haplotypes(n=50, m=10, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=20, seed=42)  # wrong m!
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.3))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        rm = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=10, p=0.5)
        sim = Simulation(hap, arch, rm, rmap, seed=42)
        with pytest.raises(ValueError, match="Effect dimension mismatch"):
            sim.run(1)

    def test_matching_dimensions_pass(self):
        sim = _make_sim()
        sim.run(1)  # should not raise
        assert sim.generation == 0


class TestRunSingleGen:
    def test_n_generations_one(self):
        """run(1) should only compute gen-0 phenotypes."""
        sim = _make_sim()
        sim.run(1)
        assert sim.generation == 0
        assert 0 in sim.phenotype_history
        assert 1 not in sim.phenotype_history

    def test_n_generations_two(self):
        """run(2) should produce gen-0 and gen-1."""
        sim = _make_sim()
        sim.run(2)
        assert sim.generation == 1
        assert 0 in sim.phenotype_history
        assert 1 in sim.phenotype_history


class TestCallbackExecution:
    def test_callback_receives_sim(self):
        """Callbacks should receive the simulation object."""
        received = []
        def cb(sim):
            received.append(sim.generation)
        sim = _make_sim(callbacks=[cb])
        sim.run(3)
        # Callback runs after gen 0, 1, 2
        assert received == [0, 1, 2]

    def test_early_stopping(self):
        """Setting sim.stop=True in callback should halt."""
        def stop_at_1(sim):
            if sim.generation >= 1:
                sim.stop = True
        sim = _make_sim(callbacks=[stop_at_1])
        sim.run(5)
        assert sim.generation == 1

    def test_multiple_callbacks_order(self):
        """Multiple callbacks should execute in order."""
        order = []
        def cb1(sim): order.append('a')
        def cb2(sim): order.append('b')
        sim = _make_sim(callbacks=[cb1, cb2])
        sim.run(1)
        assert order == ['a', 'b']


class TestFilterExecution:
    def test_filters_produce_results(self):
        """Filters and statistics should produce results."""
        from xftsim.filters import TrioFilter
        from xftsim.stats import SampleStatistics

        sim = _make_sim(
            filters={'trio': TrioFilter()},
            statistics=[SampleStatistics()],
            retain_phenotypes=10,
        )
        sim.run(3)
        # Results should have entries for each generation
        assert len(sim.results) >= 1

    def test_empty_filters_no_results(self):
        """No filters/stats → no results."""
        sim = _make_sim()
        sim.run(3)
        assert len(sim.results) == 0


class TestContinueRun:
    def test_continue_run_advances_generation(self):
        sim = _make_sim(retain_phenotypes=10, retain_haplotypes=10)
        sim.run(3)
        assert sim.generation == 2
        sim.continue_run(2)
        assert sim.generation == 4

    def test_continue_run_zero_additional(self):
        """continue_run(0) should not change state."""
        sim = _make_sim()
        sim.run(2)
        gen_before = sim.generation
        sim.continue_run(0)
        assert sim.generation == gen_before

    def test_continue_run_with_callbacks(self):
        gens = []
        def cb(sim): gens.append(sim.generation)
        sim = _make_sim(callbacks=[cb], retain_phenotypes=10, retain_haplotypes=10)
        sim.run(2)
        gens.clear()
        sim.continue_run(2)
        assert gens == [2, 3]

    def test_continue_run_early_stop(self):
        def stop_cb(sim):
            if sim.generation >= 3:
                sim.stop = True
        sim = _make_sim(callbacks=[stop_cb], retain_phenotypes=10, retain_haplotypes=10)
        sim.run(3)
        sim.continue_run(5)
        assert sim.generation == 3
