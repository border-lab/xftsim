"""
Unit tests for Simulation callbacks and early stopping.

Tests:
1. Callback is called each generation
2. Early stopping via sim.stop = True in callback
3. Callback can access sim.generation and phenotype values
4. Multiple callbacks executed in order
5. Callback at gen 0 can stop simulation
"""
import numpy as np
import pytest

from xftsim.arch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.effect import AdditiveEffects
from xftsim.sim import Simulation
from xftsim.mate import RandomMating
from xftsim.reproduce import RecombinationMap

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


def _make_sim(n=50, m=5, callbacks=None, seed=42):
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
    eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=seed)
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(eff))
    arch.add('Y.E', NoiseComponent(variance=0.5))
    arch.add('Y', AggregationComponent('Y.G + Y.E'))
    mate = RandomMating(offspring_per_pair=2)
    rmap = RecombinationMap.constant_map(m=m, p=0.5)
    return Simulation(
        hap, arch, mate, rmap,
        callbacks=callbacks or [],
        seed=seed,
    )


class TestCallbacks:
    def test_callback_called_each_gen(self):
        """Callback should be called once per generation."""
        call_log = []
        def log_cb(sim):
            call_log.append(sim.generation)

        sim = _make_sim(callbacks=[log_cb])
        sim.run(4)
        # gen 0, 1, 2, 3 = 4 callbacks
        assert len(call_log) == 4
        assert call_log == [0, 1, 2, 3]

    def test_early_stopping(self):
        """sim.stop = True should halt after current generation."""
        def stop_at_2(sim):
            if sim.generation == 2:
                sim.stop = True

        sim = _make_sim(callbacks=[stop_at_2])
        sim.run(10)
        assert sim.generation == 2

    def test_callback_accesses_phenotypes(self):
        """Callback can read phenotype values from current generation."""
        phenotype_means = []
        def record_mean(sim):
            gen = sim.generation
            if gen in sim.phenotype_history:
                phenotype_means.append(float(np.mean(sim.phenotype_history[gen]['Y'])))

        sim = _make_sim(callbacks=[record_mean])
        sim.run(3)
        assert len(phenotype_means) == 3
        assert all(np.isfinite(m) for m in phenotype_means)

    def test_multiple_callbacks_order(self):
        """Multiple callbacks execute in order."""
        order = []
        def cb_a(sim):
            order.append('A')
        def cb_b(sim):
            order.append('B')

        sim = _make_sim(callbacks=[cb_a, cb_b])
        sim.run(2)
        # gen 0: A, B; gen 1: A, B
        assert order == ['A', 'B', 'A', 'B']

    def test_stop_at_gen0(self):
        """Callback can stop at generation 0."""
        def stop_immediately(sim):
            sim.stop = True

        sim = _make_sim(callbacks=[stop_immediately])
        sim.run(5)
        assert sim.generation == 0
