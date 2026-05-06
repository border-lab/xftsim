"""
Unit tests for SibPairFilter with various family structures.

Tests:
1. Three siblings → 3 pairs
2. Four siblings → 6 pairs
3. Mixed family sizes (2, 3 siblings)
4. All singletons → 0 pairs
5. Single large family
6. Pair indices are valid sample indices
7. Phenotype values correctly extracted
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, NPhenotypeArray
from xftsim.filters import SibPairFilter, SibPairView


def _make_pheno_with_fids(fids, values=None):
    """Create NPhenotypeArray with given FIDs."""
    n = len(fids)
    sm = SampleMeta(iid=np.arange(n), fid=np.asarray(fids))
    pheno = NPhenotypeArray(samples=sm)
    if values is not None:
        for k, v in values.items():
            pheno[k] = v
    else:
        pheno['Y'] = np.arange(n, dtype=np.float64)
    return pheno


class TestSibPairFamilySizes:
    def test_two_sibs_one_pair(self):
        """Two siblings should produce 1 pair."""
        filt = SibPairFilter()
        pheno = _make_pheno_with_fids([0, 0, 1, 1])
        result = filt.apply(0, {0: pheno}, {})
        assert isinstance(result, SibPairView)
        assert result.n_pairs == 2  # two families, one pair each

    def test_three_sibs_three_pairs(self):
        """Three siblings should produce 3 pairs (C(3,2))."""
        filt = SibPairFilter()
        pheno = _make_pheno_with_fids([0, 0, 0])
        result = filt.apply(0, {0: pheno}, {})
        assert result.n_pairs == 3

    def test_four_sibs_six_pairs(self):
        """Four siblings should produce 6 pairs (C(4,2))."""
        filt = SibPairFilter()
        pheno = _make_pheno_with_fids([0, 0, 0, 0])
        result = filt.apply(0, {0: pheno}, {})
        assert result.n_pairs == 6

    def test_mixed_family_sizes(self):
        """Mixed: family of 2 (1 pair) + family of 3 (3 pairs) = 4 pairs."""
        filt = SibPairFilter()
        # Fam 0: 2 members (1 pair), Fam 1: 3 members (3 pairs)
        pheno = _make_pheno_with_fids([0, 0, 1, 1, 1])
        result = filt.apply(0, {0: pheno}, {})
        assert result.n_pairs == 4

    def test_all_singletons(self):
        """All singletons should produce 0 pairs."""
        filt = SibPairFilter()
        pheno = _make_pheno_with_fids([0, 1, 2, 3])
        result = filt.apply(0, {0: pheno}, {})
        assert result.n_pairs == 0
        assert len(result.sib1_idx) == 0
        assert len(result.sib2_idx) == 0

    def test_generation_not_in_history(self):
        """Missing generation should return None."""
        filt = SibPairFilter()
        result = filt.apply(5, {}, {})
        assert result is None


class TestSibPairIndices:
    def test_indices_are_valid(self):
        """Pair indices should be valid sample indices."""
        filt = SibPairFilter()
        n = 10
        pheno = _make_pheno_with_fids([0]*3 + [1]*2 + [2]*5)
        result = filt.apply(0, {0: pheno}, {})
        assert np.all(result.sib1_idx >= 0)
        assert np.all(result.sib1_idx < n)
        assert np.all(result.sib2_idx >= 0)
        assert np.all(result.sib2_idx < n)

    def test_pairs_share_fid(self):
        """Each pair should share the same FID."""
        filt = SibPairFilter()
        fids = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
        pheno = _make_pheno_with_fids(fids)
        result = filt.apply(0, {0: pheno}, {})
        for i in range(result.n_pairs):
            fid1 = fids[result.sib1_idx[i]]
            fid2 = fids[result.sib2_idx[i]]
            assert fid1 == fid2, f"Pair {i}: FIDs {fid1} != {fid2}"

    def test_pairs_are_distinct(self):
        """No pair should be (i, i)."""
        filt = SibPairFilter()
        pheno = _make_pheno_with_fids([0]*5)
        result = filt.apply(0, {0: pheno}, {})
        for i in range(result.n_pairs):
            assert result.sib1_idx[i] != result.sib2_idx[i]


class TestSibPairPhenotypes:
    def test_phenotype_values_extracted(self):
        """Pair phenotype values should match original array."""
        filt = SibPairFilter()
        vals = np.array([10.0, 20.0, 30.0])
        pheno = _make_pheno_with_fids([0, 0, 0], values={'Y': vals})
        result = filt.apply(0, {0: pheno}, {})
        # Verify extracted phenotype values match the original
        for i in range(result.n_pairs):
            assert result.sib1_phenotypes['Y'][i] == vals[result.sib1_idx[i]]
            assert result.sib2_phenotypes['Y'][i] == vals[result.sib2_idx[i]]

    def test_multiple_keys_extracted(self):
        """Multiple phenotype keys should all be extracted."""
        filt = SibPairFilter()
        pheno = _make_pheno_with_fids(
            [0, 0, 0],
            values={'A': np.array([1.0, 2.0, 3.0]),
                    'B': np.array([4.0, 5.0, 6.0])},
        )
        result = filt.apply(0, {0: pheno}, {})
        assert 'A' in result.sib1_phenotypes
        assert 'B' in result.sib1_phenotypes
        assert 'A' in result.sib2_phenotypes
        assert 'B' in result.sib2_phenotypes
