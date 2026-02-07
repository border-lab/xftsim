"""Tests for simulation callbacks."""
import numpy as np
import pytest

from tests.testdata import TestSimulation
from xftsim.nsim import NSimulation


def _make_sim(callbacks=None, seed=42, n_gen=3):
    hap = TestSimulation.founder_haplotypes(n=200, m=30, seed=seed)
    arch = TestSimulation.simple_architecture(m=30, h2=0.5, seed=123)
    rmap = TestSimulation.recombination_map(m=30)
    mate = TestSimulation.mating_regime()
    sim = NSimulation(
        founder_haplotypes=hap, architecture=arch,
        mating_regime=mate, recombination_map=rmap,
        callbacks=callbacks or [], seed=seed,
        retain_phenotypes=10, retain_haplotypes=10,
    )
    return sim, n_gen


class TestCallbacks:
    def test_fires_each_gen(self):
        gens = []
        def cb(sim):
            gens.append(sim.generation)
        sim, n_gen = _make_sim(callbacks=[cb])
        sim.run(n_gen)
        assert gens == [0, 1, 2]

    def test_multiple_callbacks_in_order(self):
        order = []
        def cb1(sim):
            order.append('a')
        def cb2(sim):
            order.append('b')
        sim, n_gen = _make_sim(callbacks=[cb1, cb2])
        sim.run(n_gen)
        # Should alternate: a,b for each gen
        assert order == ['a', 'b'] * 3

    def test_correct_generation(self):
        gen_vals = []
        def cb(sim):
            gen_vals.append(sim.generation)
        sim, _ = _make_sim(callbacks=[cb])
        sim.run(4)
        assert gen_vals == [0, 1, 2, 3]

    def test_has_phenotypes(self):
        """Callback should have access to current phenotypes."""
        def cb(sim):
            assert sim.generation in sim.phenotype_history
            pheno = sim.phenotype_history[sim.generation]
            assert 'Y' in pheno
        sim, n_gen = _make_sim(callbacks=[cb])
        sim.run(n_gen)

    def test_has_haplotypes(self):
        """Callback should have access to current haplotypes."""
        def cb(sim):
            assert sim.generation in sim.haplotype_history
        sim, n_gen = _make_sim(callbacks=[cb])
        sim.run(n_gen)

    def test_early_stopping_gen0(self):
        def cb(sim):
            sim.stop = True
        sim, _ = _make_sim(callbacks=[cb])
        sim.run(5)
        assert sim.generation == 0

    def test_early_stopping_gen1(self):
        def cb(sim):
            if sim.generation == 1:
                sim.stop = True
        sim, _ = _make_sim(callbacks=[cb])
        sim.run(5)
        assert sim.generation == 1

    def test_callback_reads_history(self):
        """At gen 2, callback should be able to read gen 0 and 1 phenotypes."""
        history_sizes = []
        def cb(sim):
            history_sizes.append(len(sim.phenotype_history))
        sim, _ = _make_sim(callbacks=[cb])
        sim.run(3)
        assert history_sizes[0] == 1  # gen 0
        assert history_sizes[1] == 2  # gen 0,1
        assert history_sizes[2] == 3  # gen 0,1,2
