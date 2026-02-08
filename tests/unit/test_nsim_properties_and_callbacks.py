"""
Unit tests for NSimulation properties, callbacks, and edge cases.

Tests:
1. sim.haplotypes / sim.phenotypes convenience properties
2. sim.generation tracks correctly
3. Callback receives simulation reference
4. Callback modifies simulation state
5. sim.stop from callback ends simulation
6. Multiple callbacks in order
7. NSimulation repr
8. run(1) produces gen 0 only
"""
import numpy as np
import pytest

from xftsim.neffect import AdditiveEffects
from xftsim.narch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.nmate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.nsim import NSimulation

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


def _make_sim(n=100, m=20, seed=42, callbacks=None, **kwargs):
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
        seed=seed, callbacks=callbacks or [], **kwargs,
    )


class TestProperties:
    def test_haplotypes_property(self):
        sim = _make_sim()
        sim.run(2)
        assert sim.haplotypes is sim.haplotype_history[sim.generation]

    def test_phenotypes_property(self):
        sim = _make_sim()
        sim.run(2)
        assert sim.phenotypes is sim.phenotype_history[sim.generation]

    def test_generation_after_run(self):
        sim = _make_sim()
        sim.run(3)
        assert sim.generation == 2

    def test_generation_after_run_1(self):
        sim = _make_sim()
        sim.run(1)
        assert sim.generation == 0


class TestCallbacks:
    def test_callback_receives_sim(self):
        received = []
        def cb(s):
            received.append(s.generation)
        sim = _make_sim(callbacks=[cb])
        sim.run(3)
        # Callback runs once per generation (gen 0, 1, 2)
        assert received == [0, 1, 2]

    def test_callback_modifies_state(self):
        marker = []
        def cb(s):
            marker.append(np.mean(s.phenotypes['Y']))
        sim = _make_sim(callbacks=[cb])
        sim.run(2)
        assert len(marker) == 2
        assert all(np.isfinite(m) for m in marker)

    def test_stop_from_callback(self):
        def cb(s):
            if s.generation >= 1:
                s.stop = True
        sim = _make_sim(callbacks=[cb])
        sim.run(5)
        # Should stop at gen 1
        assert sim.generation <= 1

    def test_stop_at_gen0(self):
        def cb(s):
            s.stop = True
        sim = _make_sim(callbacks=[cb])
        sim.run(5)
        assert sim.generation == 0

    def test_multiple_callbacks_ordered(self):
        order = []
        def cb1(s): order.append('first')
        def cb2(s): order.append('second')
        sim = _make_sim(callbacks=[cb1, cb2])
        sim.run(1)
        assert order == ['first', 'second']


class TestRepr:
    def test_repr_before_run(self):
        sim = _make_sim()
        r = repr(sim)
        assert 'NSimulation' in r
        assert 'generation=0' in r

    def test_repr_after_run(self):
        sim = _make_sim()
        sim.run(2)
        r = repr(sim)
        assert 'generation=1' in r


class TestRunEdgeCases:
    def test_run_1_only_gen0(self):
        sim = _make_sim()
        sim.run(1)
        assert 0 in sim.phenotype_history
        assert 1 not in sim.phenotype_history

    def test_results_populated(self):
        """Results should be populated even without statistics."""
        from xftsim.nstats import SampleStatistics
        sim = _make_sim()
        sim.statistics = [SampleStatistics()]
        sim.run(2)
        assert len(sim.results) == 2
        assert sim.results[0].generation == 0
        assert sim.results[1].generation == 1
