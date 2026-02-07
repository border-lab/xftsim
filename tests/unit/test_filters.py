"""
Tests for TrioFilter and SibPairFilter.
"""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation

from xftsim.nfilter import TrioFilter, SibPairFilter, TrioView, SibPairView
from xftsim.struct import SampleMeta, NPhenotypeArray
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


# ── Additional filter edge cases ──────────────────────────────────────────

class TestTrioFilterEdgeCases:
    """Edge case tests for TrioFilter."""

    def test_missing_generation_returns_none(self):
        """Requesting a generation not in history should return None."""
        tf = TrioFilter()
        result = tf.apply(5, {}, {})
        assert result is None

    def test_missing_parent_phenotypes_returns_none(self):
        """If prev gen phenotypes pruned by retention, should return None."""
        tf = TrioFilter()
        from xftsim.struct import PedigreeArray
        # gen=2 has pedigree but gen=1 not in phenotype_history
        ped = PedigreeArray(
            offspring_samples=SampleMeta(iid=np.arange(4), sex=np.array([0,1,0,1])),
            maternal_idx=np.array([0, 0, 1, 1]),
            paternal_idx=np.array([2, 2, 3, 3]),
            parent_n=4,
        )
        pheno2 = NPhenotypeArray(samples=SampleMeta(iid=np.arange(4), sex=np.array([0,1,0,1])))
        pheno2._values['Y'] = np.ones(4)
        result = tf.apply(2, {2: pheno2}, {2: ped})
        assert result is None

    def test_trio_gen2_uses_gen1_parents(self):
        """Gen-2 trios should index into gen-1 phenotypes."""
        hap = TestSimulation.founder_haplotypes(n=200, m=20)
        arch = TestSimulation.simple_architecture(m=20, h2=0.5)
        rm = RandomMating(offspring_per_pair=2)
        rmap = TestSimulation.recombination_map(m=20)
        sim = NSimulation(hap, arch, rm, rmap, seed=42, retain_phenotypes=10)
        sim.run(3)
        tf = TrioFilter()
        view = tf.apply(2, sim.phenotype_history, sim.pedigree_history)
        assert isinstance(view, TrioView)
        # Mother phenotypes should come from gen 1
        ped = sim.pedigree_history[2]
        gen1_Y = sim.phenotype_history[1]['Y']
        np.testing.assert_array_equal(
            view.mother_phenotypes['Y'],
            gen1_Y[ped.maternal_idx]
        )

    def test_trio_partial_key_overlap(self):
        """If parent gen has fewer keys, only matching keys appear in trio."""
        hap = TestSimulation.founder_haplotypes(n=100, m=20)
        arch = TestSimulation.simple_architecture(m=20, h2=0.5)
        rm = RandomMating(offspring_per_pair=2)
        rmap = TestSimulation.recombination_map(m=20)
        sim = NSimulation(hap, arch, rm, rmap, seed=42, retain_phenotypes=10)
        sim.run(2)
        # Add an extra key only to gen 1
        sim.phenotype_history[1]._values['EXTRA'] = np.ones(sim.phenotype_history[1].samples.n)
        tf = TrioFilter()
        view = tf.apply(1, sim.phenotype_history, sim.pedigree_history)
        # EXTRA was not in gen 0 parent, so should not appear in mother/father dicts
        assert 'EXTRA' not in view.mother_phenotypes


class TestSibPairFilterEdgeCases:
    """Edge case tests for SibPairFilter."""

    def test_all_singletons_returns_zero_pairs(self):
        """If no family has 2+ members, should return 0 pairs."""
        sf = SibPairFilter()
        n = 10
        samples = SampleMeta(
            iid=np.arange(n),
            fid=np.arange(n),  # each individual in own family
            sex=np.tile([0, 1], 5),
        )
        pheno = NPhenotypeArray(samples=samples)
        pheno._values['Y'] = np.random.RandomState(42).randn(n)
        view = sf.apply(0, {0: pheno}, {})
        assert isinstance(view, SibPairView)
        assert view.n_pairs == 0
        assert len(view.sib1_idx) == 0

    def test_missing_generation_returns_none(self):
        """If generation not in phenotype_history, should return None."""
        sf = SibPairFilter()
        result = sf.apply(5, {}, {})
        assert result is None

    def test_large_family_pair_count(self):
        """Family of size k should produce k*(k-1)/2 pairs."""
        sf = SibPairFilter()
        k = 5
        n = k
        samples = SampleMeta(
            iid=np.arange(n),
            fid=np.zeros(n, dtype=np.int64),  # all same family
            sex=np.tile([0, 1], (n + 1) // 2)[:n],
        )
        pheno = NPhenotypeArray(samples=samples)
        pheno._values['Y'] = np.arange(n, dtype=np.float64)
        view = sf.apply(0, {0: pheno}, {})
        expected_pairs = k * (k - 1) // 2
        assert view.n_pairs == expected_pairs

    def test_two_families_independent_pairs(self):
        """Pairs should only form within families, not across."""
        sf = SibPairFilter()
        n = 6
        samples = SampleMeta(
            iid=np.arange(n),
            fid=np.array([0, 0, 0, 1, 1, 1]),
            sex=np.tile([0, 1], 3),
        )
        pheno = NPhenotypeArray(samples=samples)
        pheno._values['Y'] = np.arange(n, dtype=np.float64)
        view = sf.apply(0, {0: pheno}, {})
        # 2 families of 3 → 3 pairs each = 6 total
        assert view.n_pairs == 6
        # All pairs should have same FID
        fids = samples.fid
        for i in range(view.n_pairs):
            assert fids[view.sib1_idx[i]] == fids[view.sib2_idx[i]]

    def test_gen0_sibpair_works(self):
        """SibPairFilter on gen 0 founders (with FIDs) should work."""
        sf = SibPairFilter()
        n = 20
        samples = SampleMeta(
            iid=np.arange(n),
            fid=np.repeat(np.arange(10), 2),  # 10 families of 2
            sex=np.tile([0, 1], 10),
        )
        pheno = NPhenotypeArray(samples=samples)
        pheno._values['Y'] = np.random.RandomState(42).randn(n)
        view = sf.apply(0, {0: pheno}, {})
        assert view.n_pairs == 10  # 10 families × 1 pair each
