"""
Unit tests for simulation callbacks and early stopping.

Tests:
1. Callback is called each generation
2. Multiple callbacks all invoked
3. sim.stop() halts simulation early
4. Callback receives correct generation number
5. Callback can access phenotypes
6. Early stopping respects generation count
"""
import numpy as np
import pytest

from xftsim.effect import AdditiveEffects
from xftsim.arch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.mate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.sim import Simulation

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


def _make_sim(callbacks=None, seed=42):
    n, m = 100, 20
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
    eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=seed)
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(eff))
    arch.add('Y.E', NoiseComponent(variance=0.5))
    arch.add('Y', AggregationComponent('Y.G + Y.E'))

    return Simulation(
        founder_haplotypes=hap, architecture=arch,
        mating_regime=RandomMating(offspring_per_pair=2),
        recombination_map=RecombinationMap.constant_map(m=m),
        seed=seed,
        callbacks=callbacks or [],
    )


class TestCallbacks:
    def test_callback_called_each_gen(self):
        """Callback should be invoked once per generation."""
        call_count = [0]
        def counter(sim):
            call_count[0] += 1

        sim = _make_sim(callbacks=[counter])
        sim.run(5)  # gens 0-4
        assert call_count[0] == 5, \
            f"Callback called {call_count[0]} times, expected 5"

    def test_multiple_callbacks(self):
        """Multiple callbacks should all be invoked."""
        log_a = []
        log_b = []
        def cb_a(sim):
            log_a.append(sim.generation)
        def cb_b(sim):
            log_b.append(sim.generation)

        sim = _make_sim(callbacks=[cb_a, cb_b])
        sim.run(3)
        assert log_a == [0, 1, 2]
        assert log_b == [0, 1, 2]

    def test_early_stopping(self):
        """sim.stop() should halt simulation before reaching target gens."""
        gen_log = []
        def stop_at_gen_2(sim):
            gen_log.append(sim.generation)
            if sim.generation >= 2:
                sim.stop = True

        sim = _make_sim(callbacks=[stop_at_gen_2])
        sim.run(10)  # should stop at gen 2, not 9
        assert max(gen_log) == 2, \
            f"Max generation = {max(gen_log)}, expected 2"
        assert sim.generation == 2

    def test_callback_receives_generation(self):
        """Callback should see correct generation number."""
        gens_seen = []
        def track_gen(sim):
            gens_seen.append(sim.generation)

        sim = _make_sim(callbacks=[track_gen])
        sim.run(4)
        assert gens_seen == [0, 1, 2, 3]

    def test_callback_can_access_phenotypes(self):
        """Callback should be able to read current phenotypes."""
        phenotype_means = []
        def record_mean(sim):
            pheno = sim.phenotypes
            phenotype_means.append(np.mean(pheno['Y']))

        sim = _make_sim(callbacks=[record_mean])
        sim.run(3)
        assert len(phenotype_means) == 3
        assert all(np.isfinite(m) for m in phenotype_means)

    def test_no_callbacks_runs_normally(self):
        """Simulation with no callbacks should run fine."""
        sim = _make_sim(callbacks=[])
        sim.run(3)
        assert sim.generation == 2
        assert np.all(np.isfinite(sim.phenotypes['Y']))
