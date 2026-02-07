"""
Tests for TrioFilter and SibPairFilter.
"""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation

from xftsim.nfilter import TrioFilter, SibPairFilter, TrioView, SibPairView
from xftsim.nsim import NSimulation
from xftsim.nmate import RandomMating


class TestTrioFilter:
    @pytest.fixture
    def sim_3gen(self):
        hap = TestSimulation.founder_haplotypes(n=500, m=50)
        arch = TestSimulation.simple_architecture(m=50, h2=0.5)
        rm = TestSimulation.mating_regime(offspring_per_pair=2)
        rmap = TestSimulation.recombination_map(m=50)
        sim = NSimulation(hap, arch, rm, rmap, seed=42,
                         retain_phenotypes=10)
        sim.run(3)
        return sim

    def test_trio_gen0_none(self, sim_3gen):
        """TrioFilter at gen 0 should return None (no parents)."""
        tf = TrioFilter()
        result = tf.apply(0, sim_3gen.phenotype_history, sim_3gen.pedigree_history)
        assert result is None

    def test_trio_gen1_shape(self, sim_3gen):
        """TrioFilter at gen 1 should produce correctly shaped arrays."""
        tf = TrioFilter()
        view = tf.apply(1, sim_3gen.phenotype_history, sim_3gen.pedigree_history)
        assert isinstance(view, TrioView)
        assert view.n_trios == sim_3gen.phenotype_history[1].samples.n
        for key in view.offspring_phenotypes:
            assert view.offspring_phenotypes[key].shape == (view.n_trios,)

    def test_trio_alignment_correct(self, sim_3gen):
        """Trio offspring[i] should match phenotype_history[gen][key][i]."""
        tf = TrioFilter()
        view = tf.apply(1, sim_3gen.phenotype_history, sim_3gen.pedigree_history)
        gen1_pheno = sim_3gen.phenotype_history[1]
        for key in view.offspring_phenotypes:
            np.testing.assert_array_equal(
                view.offspring_phenotypes[key],
                gen1_pheno[key]
            )

    def test_trio_all_keys(self, sim_3gen):
        """All offspring phenotype keys should appear in the trio view."""
        tf = TrioFilter()
        view = tf.apply(1, sim_3gen.phenotype_history, sim_3gen.pedigree_history)
        offspring_keys = set(view.offspring_phenotypes.keys())
        pheno_keys = set(sim_3gen.phenotype_history[1].keys)
        assert offspring_keys == pheno_keys


class TestSibPairFilter:
    @pytest.fixture
    def sim_2gen(self):
        hap = TestSimulation.founder_haplotypes(n=200, m=20)
        arch = TestSimulation.simple_architecture(m=20, h2=0.5)
        rm = RandomMating(offspring_per_pair=3)  # 3 per pair for sib pairs
        rmap = TestSimulation.recombination_map(m=20)
        sim = NSimulation(hap, arch, rm, rmap, seed=42,
                         retain_phenotypes=10)
        sim.run(2)
        return sim

    def test_sibpair_shape(self, sim_2gen):
        """SibPairFilter should produce paired arrays of equal length."""
        sf = SibPairFilter()
        view = sf.apply(1, sim_2gen.phenotype_history, sim_2gen.pedigree_history)
        assert isinstance(view, SibPairView)
        assert view.n_pairs > 0
        for key in view.sib1_phenotypes:
            assert view.sib1_phenotypes[key].shape == (view.n_pairs,)
            assert view.sib2_phenotypes[key].shape == (view.n_pairs,)

    def test_sibpair_same_fid(self, sim_2gen):
        """All sib pairs should share the same FID (verified via exposed indices)."""
        sf = SibPairFilter()
        view = sf.apply(1, sim_2gen.phenotype_history, sim_2gen.pedigree_history)
        pheno = sim_2gen.phenotype_history[1]
        fids = pheno.samples.fid

        # Direct FID verification using exposed indices
        assert view.sib1_idx is not None
        assert view.sib2_idx is not None
        fid1 = fids[view.sib1_idx]
        fid2 = fids[view.sib2_idx]
        np.testing.assert_array_equal(fid1, fid2)

    def test_sibpair_no_self_pairs(self, sim_2gen):
        """No individual should be paired with itself."""
        sf = SibPairFilter()
        view = sf.apply(1, sim_2gen.phenotype_history, sim_2gen.pedigree_history)
        # If sib1 and sib2 were the same person, their phenotypes would be identical
        # Check that at least some pairs differ
        diff = view.sib1_phenotypes['Y'] - view.sib2_phenotypes['Y']
        assert np.any(diff != 0)
