"""Integration tests for pedigree consistency."""
import numpy as np
import pytest

from tests.testdata import TestSimulation
from xftsim.sim import Simulation


def _run_sim(n_gen=3, seed=42):
    hap = TestSimulation.founder_haplotypes(n=500, m=50, seed=seed)
    arch = TestSimulation.simple_architecture(m=50, h2=0.5, seed=123)
    rmap = TestSimulation.recombination_map(m=50)
    mate = TestSimulation.mating_regime(offspring_per_pair=2)
    sim = Simulation(
        founder_haplotypes=hap, architecture=arch,
        mating_regime=mate, recombination_map=rmap,
        retain_haplotypes=5, retain_phenotypes=5, seed=seed,
    )
    sim.run(n_gen)
    return sim


class TestPedigreeIntegrity:
    def test_every_offspring_has_two_parents(self):
        sim = _run_sim(n_gen=3)
        for gen in range(1, 3):
            ped = sim.pedigree_history[gen]
            n = ped.offspring_samples.n
            assert len(ped.maternal_idx) == n
            assert len(ped.paternal_idx) == n
            assert np.all(ped.maternal_idx >= 0)
            assert np.all(ped.paternal_idx >= 0)

    def test_haplotype_tracing(self):
        """Offspring haplotypes should have correct dimensions after meiosis."""
        sim = _run_sim(n_gen=2)
        hap0 = sim.haplotype_history[0]
        hap1 = sim.haplotype_history[1]
        assert hap1.m == hap0.m
        assert hap1.n > 0

    def test_fid_consistency(self):
        """Within each family, all siblings should share the same FID."""
        sim = _run_sim(n_gen=2)
        hap1 = sim.haplotype_history[1]
        fids = hap1.samples.fid
        ped = sim.pedigree_history[1]
        # Siblings (same maternal_idx AND paternal_idx) should have same FID
        for i in range(len(ped.maternal_idx)):
            for j in range(i+1, min(i+10, len(ped.maternal_idx))):
                if (ped.maternal_idx[i] == ped.maternal_idx[j] and
                        ped.paternal_idx[i] == ped.paternal_idx[j]):
                    assert fids[i] == fids[j]

    def test_no_self_mating(self):
        """No individual should mate with themselves."""
        sim = _run_sim(n_gen=3)
        for gen in range(1, 3):
            ped = sim.pedigree_history[gen]
            assert not np.any(ped.maternal_idx == ped.paternal_idx)

    def test_parent_indices_in_bounds(self):
        sim = _run_sim(n_gen=3)
        for gen in range(1, 3):
            ped = sim.pedigree_history[gen]
            assert np.all(ped.maternal_idx < ped.parent_n)
            assert np.all(ped.paternal_idx < ped.parent_n)
