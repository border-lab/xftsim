"""Integration tests for simple multi-generation simulations."""
import numpy as np
import pytest

from tests.testdata import TestSimulation
from xftsim.nsim import NSimulation
from xftsim.nmate import RandomMating


def _run_sim(n_gen=3, seed=42, callbacks=None, **kwargs):
    hap = TestSimulation.founder_haplotypes(n=500, m=50, seed=seed)
    arch = TestSimulation.simple_architecture(m=50, h2=0.5, seed=123)
    rmap = TestSimulation.recombination_map(m=50)
    mate = TestSimulation.mating_regime()
    sim = NSimulation(
        founder_haplotypes=hap, architecture=arch,
        mating_regime=mate, recombination_map=rmap,
        callbacks=callbacks or [], seed=seed, **kwargs,
    )
    sim.run(n_gen)
    return sim


class TestMultiGenRun:
    def test_runs_3_generations(self):
        sim = _run_sim(n_gen=3)
        assert sim.generation == 2

    def test_phenotype_keys_consistent(self):
        """All generations should produce the same phenotype keys."""
        sim = _run_sim(n_gen=3, retain_phenotypes=5)
        keys_0 = set(sim.phenotype_history[0].keys)
        for gen in range(1, 3):
            assert set(sim.phenotype_history[gen].keys) == keys_0

    def test_population_size_stable(self):
        """Population size should be consistent across generations."""
        sim = _run_sim(n_gen=3, retain_haplotypes=5)
        n0 = sim.haplotype_history[0].n
        for gen in range(1, 3):
            n_gen = sim.haplotype_history[gen].n
            # With 250 pairs * 2 offspring = 500
            assert n_gen > 0

    def test_history_retention(self):
        """Retention policy should prune old generations."""
        sim = _run_sim(n_gen=5, retain_phenotypes=2)
        # Generation 4 is the last; retain_phenotypes=2 means keep gen 4, 3, 2
        assert 4 in sim.phenotype_history
        assert 3 in sim.phenotype_history
        assert 2 in sim.phenotype_history
        assert 0 not in sim.phenotype_history

    def test_callback_fires_each_gen(self):
        gens_seen = []
        def cb(sim):
            gens_seen.append(sim.generation)
        sim = _run_sim(n_gen=3, callbacks=[cb])
        assert gens_seen == [0, 1, 2]

    def test_early_stopping_gen0(self):
        def stop_immediately(sim):
            sim.stop = True
        sim = _run_sim(n_gen=5, callbacks=[stop_immediately])
        assert sim.generation == 0

    def test_early_stopping_gen1(self):
        def stop_at_gen1(sim):
            if sim.generation == 1:
                sim.stop = True
        sim = _run_sim(n_gen=5, callbacks=[stop_at_gen1])
        assert sim.generation == 1

    def test_single_generation(self):
        """Running for 1 generation (gen 0 only) should work."""
        sim = _run_sim(n_gen=1)
        assert sim.generation == 0
        assert 0 in sim.phenotype_history
