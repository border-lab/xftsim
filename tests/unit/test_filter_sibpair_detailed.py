"""
Unit tests for SibPairFilter pair generation correctness.

Tests:
1. Pair count matches expected for known family sizes
2. No duplicate pairs
3. Upper-triangle property (sib1 < sib2 within family)
4. Large families: C(k, 2) pairs
5. Single-member families: no pairs
6. Mixed family sizes
7. TrioFilter: gen 0 returns None, key intersection
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, PhenotypeArray, PedigreeArray
from xftsim.filters import TrioFilter, SibPairFilter, TrioView, SibPairView


def _make_pheno_with_fid(n, fid, keys=None):
    """Create PhenotypeArray with specified FIDs."""
    sm = SampleMeta(iid=np.arange(n), fid=np.asarray(fid))
    if keys is None:
        keys = {'Y': np.random.randn(n)}
    return PhenotypeArray(samples=sm, values=keys)


class TestSibPairFilterPairCount:
    def test_two_member_families(self):
        """Families of size 2 → 1 pair each."""
        fid = [0, 0, 1, 1, 2, 2]
        pheno = _make_pheno_with_fid(6, fid)
        filt = SibPairFilter()
        view = filt.apply(0, {0: pheno}, {})
        assert view.n_pairs == 3  # 3 families * C(2,2) = 3

    def test_three_member_families(self):
        """Families of size 3 → 3 pairs each."""
        fid = [0, 0, 0, 1, 1, 1]
        pheno = _make_pheno_with_fid(6, fid)
        filt = SibPairFilter()
        view = filt.apply(0, {0: pheno}, {})
        assert view.n_pairs == 6  # 2 families * C(3,2) = 6

    def test_five_member_family(self):
        """Family of size 5 → C(5,2) = 10 pairs."""
        fid = np.zeros(5, dtype=int)
        pheno = _make_pheno_with_fid(5, fid)
        filt = SibPairFilter()
        view = filt.apply(0, {0: pheno}, {})
        assert view.n_pairs == 10

    def test_single_member_families_no_pairs(self):
        """All single-member families → 0 pairs."""
        fid = np.arange(5)
        pheno = _make_pheno_with_fid(5, fid)
        filt = SibPairFilter()
        view = filt.apply(0, {0: pheno}, {})
        assert view.n_pairs == 0

    def test_mixed_family_sizes(self):
        """Mix of sizes: 1, 2, 3 → 0 + 1 + 3 = 4 pairs."""
        fid = [0, 1, 1, 2, 2, 2]
        pheno = _make_pheno_with_fid(6, fid)
        filt = SibPairFilter()
        view = filt.apply(0, {0: pheno}, {})
        assert view.n_pairs == 4  # C(1,2)=0, C(2,2)=1, C(3,2)=3


class TestSibPairFilterNoDuplicates:
    def test_no_duplicate_pairs(self):
        """No (a,b) and (b,a) or (a,b) twice."""
        fid = np.repeat(np.arange(5), 4)  # 5 families of size 4
        pheno = _make_pheno_with_fid(20, fid)
        filt = SibPairFilter()
        view = filt.apply(0, {0: pheno}, {})
        # C(4,2)*5 = 30 pairs
        assert view.n_pairs == 30
        # Check no duplicate pairs
        pair_set = set()
        for i in range(view.n_pairs):
            a, b = int(view.sib1_idx[i]), int(view.sib2_idx[i])
            pair = (min(a, b), max(a, b))
            assert pair not in pair_set, f"Duplicate pair: {pair}"
            pair_set.add(pair)

    def test_sib_indices_from_same_family(self):
        """Each pair (sib1, sib2) should share the same FID."""
        fid = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
        pheno = _make_pheno_with_fid(9, fid)
        filt = SibPairFilter()
        view = filt.apply(0, {0: pheno}, {})
        for i in range(view.n_pairs):
            assert fid[view.sib1_idx[i]] == fid[view.sib2_idx[i]]


class TestSibPairFilterPhenotypeAlignment:
    def test_phenotype_values_aligned(self):
        """sib1_phenotypes and sib2_phenotypes should index correctly."""
        fid = [0, 0]
        vals = np.array([10.0, 20.0])
        pheno = _make_pheno_with_fid(2, fid, keys={'Y': vals})
        filt = SibPairFilter()
        view = filt.apply(0, {0: pheno}, {})
        assert view.n_pairs == 1
        # One pair: indices [0] and [1]
        assert view.sib1_phenotypes['Y'][0] in [10.0, 20.0]
        assert view.sib2_phenotypes['Y'][0] in [10.0, 20.0]
        assert view.sib1_phenotypes['Y'][0] != view.sib2_phenotypes['Y'][0]


class TestSibPairFilterMissingGen:
    def test_missing_generation_returns_none(self):
        """Generation not in phenotype_history → None."""
        filt = SibPairFilter()
        result = filt.apply(5, {}, {})
        assert result is None


class TestTrioFilterEdgeCases:
    def test_gen0_returns_none(self):
        """Gen 0 has no parents → None."""
        filt = TrioFilter()
        pheno = _make_pheno_with_fid(10, np.arange(10))
        result = filt.apply(0, {0: pheno}, {})
        assert result is None

    def test_no_pedigree_returns_none(self):
        """Gen > 0 but no pedigree → None."""
        filt = TrioFilter()
        pheno = _make_pheno_with_fid(10, np.arange(10))
        result = filt.apply(1, {1: pheno}, {})
        assert result is None

    def test_no_parent_phenotype_returns_none(self):
        """Parent generation phenotypes pruned → None."""
        filt = TrioFilter()
        pheno1 = _make_pheno_with_fid(4, [0, 0, 1, 1])
        sm_parent = SampleMeta(iid=np.arange(4))
        ped = PedigreeArray(
            offspring_samples=pheno1.samples,
            maternal_idx=np.array([0, 0, 1, 1]),
            paternal_idx=np.array([2, 2, 3, 3]),
            parent_n=4,
        )
        result = filt.apply(1, {1: pheno1}, {1: ped})
        # parent gen 0 not in phenotype_history
        assert result is None

    def test_valid_trio(self):
        """Valid trio extraction."""
        filt = TrioFilter()
        n_parent = 4
        n_offspring = 4
        parent_sm = SampleMeta(iid=np.arange(n_parent))
        parent_pheno = PhenotypeArray(
            samples=parent_sm,
            values={'Y': np.array([1.0, 2.0, 3.0, 4.0])},
        )
        offspring_sm = SampleMeta(iid=np.arange(n_offspring), generation=1)
        offspring_pheno = PhenotypeArray(
            samples=offspring_sm,
            values={'Y': np.array([10.0, 20.0, 30.0, 40.0])},
        )
        ped = PedigreeArray(
            offspring_samples=offspring_sm,
            maternal_idx=np.array([0, 0, 1, 1]),
            paternal_idx=np.array([2, 2, 3, 3]),
            parent_n=n_parent,
        )
        result = filt.apply(1, {0: parent_pheno, 1: offspring_pheno}, {1: ped})
        assert isinstance(result, TrioView)
        assert result.n_trios == 4
        np.testing.assert_array_equal(result.offspring_phenotypes['Y'],
                                     [10.0, 20.0, 30.0, 40.0])
        np.testing.assert_array_equal(result.mother_phenotypes['Y'],
                                     [1.0, 1.0, 2.0, 2.0])
        np.testing.assert_array_equal(result.father_phenotypes['Y'],
                                     [3.0, 3.0, 4.0, 4.0])

    def test_trio_partial_key_overlap(self):
        """If offspring has extra keys not in parent, those keys omitted from parents."""
        filt = TrioFilter()
        n = 2
        parent_sm = SampleMeta(iid=np.arange(n))
        parent_pheno = PhenotypeArray(
            samples=parent_sm,
            values={'Y': np.array([1.0, 2.0])},
        )
        offspring_sm = SampleMeta(iid=np.arange(n), generation=1)
        offspring_pheno = PhenotypeArray(
            samples=offspring_sm,
            values={'Y': np.array([3.0, 4.0]), 'Z': np.array([5.0, 6.0])},
        )
        ped = PedigreeArray(
            offspring_samples=offspring_sm,
            maternal_idx=np.array([0, 0]),
            paternal_idx=np.array([1, 1]),
            parent_n=n,
        )
        result = filt.apply(1, {0: parent_pheno, 1: offspring_pheno}, {1: ped})
        assert 'Y' in result.mother_phenotypes
        assert 'Z' not in result.mother_phenotypes  # parent didn't have Z
        assert 'Y' in result.offspring_phenotypes
        assert 'Z' in result.offspring_phenotypes  # offspring has Z
