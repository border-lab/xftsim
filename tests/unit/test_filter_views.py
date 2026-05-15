"""
Unit tests for TrioFilter and SibPairFilter edge cases.

Tests:
1. TrioFilter at gen 0 returns None
2. TrioFilter with pruned parent gen returns None
3. TrioFilter key mismatch: parent missing some offspring keys
4. TrioFilter normal operation: trio alignment
5. SibPairFilter with all singletons (no pairs)
6. SibPairFilter with one large family
7. SibPairFilter mixed family sizes
8. SibPairFilter empty phenotype history
9. SibPairFilter pair counting correctness
10. TrioView/SibPairView data structures
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, PhenotypeArray, PedigreeArray
from xftsim.filters import TrioFilter, SibPairFilter, TrioView, SibPairView


def _make_pheno(n, keys, fid=None, generation=0):
    if fid is None:
        fid = np.arange(n)
    sm = SampleMeta(iid=np.arange(n), fid=fid, generation=generation)
    vals = {k: np.arange(n, dtype=float) for k in keys}
    return PhenotypeArray(samples=sm, values=vals)


class TestTrioFilterEdgeCases:
    def test_gen0_returns_none(self):
        filt = TrioFilter()
        pheno_hist = {0: _make_pheno(10, ['Y'])}
        result = filt.apply(0, pheno_hist, {})
        assert result is None

    def test_missing_pedigree_returns_none(self):
        filt = TrioFilter()
        pheno_hist = {0: _make_pheno(10, ['Y']), 1: _make_pheno(10, ['Y'])}
        result = filt.apply(1, pheno_hist, {})  # empty pedigree_history
        assert result is None

    def test_pruned_parent_gen_returns_none(self):
        filt = TrioFilter()
        # Gen 1 exists but gen 0 phenotypes pruned
        offspring_sm = SampleMeta(iid=np.arange(4), generation=1)
        ped = PedigreeArray(
            offspring_samples=offspring_sm,
            maternal_idx=np.array([0, 0, 2, 2]),
            paternal_idx=np.array([1, 1, 3, 3]),
            parent_n=10,
        )
        pheno_hist = {1: _make_pheno(4, ['Y'], generation=1)}
        result = filt.apply(1, pheno_hist, {1: ped})
        assert result is None  # prev gen (0) not in pheno_hist

    def test_key_mismatch(self):
        """If parent phenotypes lack some offspring keys, only shared keys appear."""
        filt = TrioFilter()
        parent_pheno = _make_pheno(10, ['Y'])  # only Y
        offspring_sm = SampleMeta(iid=np.arange(4), fid=np.array([0, 0, 1, 1]), generation=1)
        offspring_pheno = PhenotypeArray(
            samples=offspring_sm,
            values={
                'Y': np.array([1.0, 2.0, 3.0, 4.0]),
                'Z': np.array([5.0, 6.0, 7.0, 8.0]),
            },
        )
        ped = PedigreeArray(
            offspring_samples=offspring_sm,
            maternal_idx=np.array([0, 0, 2, 2]),
            paternal_idx=np.array([1, 1, 3, 3]),
            parent_n=10,
        )
        pheno_hist = {0: parent_pheno, 1: offspring_pheno}
        result = filt.apply(1, pheno_hist, {1: ped})

        assert isinstance(result, TrioView)
        assert 'Y' in result.mother_phenotypes
        # Z is not in parent_pheno, so shouldn't be in mother_phenotypes
        assert 'Z' not in result.mother_phenotypes
        # But Z IS in offspring_phenotypes (all offspring keys included)
        assert 'Z' in result.offspring_phenotypes

    def test_normal_trio_alignment(self):
        """Trio values should be correctly aligned by pedigree."""
        filt = TrioFilter()
        parent_sm = SampleMeta(iid=np.arange(4), generation=0)
        parent_pheno = PhenotypeArray(
            samples=parent_sm,
            values={'Y': np.array([10.0, 20.0, 30.0, 40.0])},
        )
        offspring_sm = SampleMeta(iid=np.arange(2), generation=1)
        offspring_pheno = PhenotypeArray(
            samples=offspring_sm,
            values={'Y': np.array([100.0, 200.0])},
        )
        ped = PedigreeArray(
            offspring_samples=offspring_sm,
            maternal_idx=np.array([0, 2]),
            paternal_idx=np.array([1, 3]),
            parent_n=4,
        )
        pheno_hist = {0: parent_pheno, 1: offspring_pheno}
        result = filt.apply(1, pheno_hist, {1: ped})

        assert result.n_trios == 2
        np.testing.assert_array_equal(result.offspring_phenotypes['Y'], [100.0, 200.0])
        np.testing.assert_array_equal(result.mother_phenotypes['Y'], [10.0, 30.0])
        np.testing.assert_array_equal(result.father_phenotypes['Y'], [20.0, 40.0])


class TestSibPairFilterEdgeCases:
    def test_all_singletons_returns_empty(self):
        """All unique FIDs → no sibling pairs."""
        filt = SibPairFilter()
        pheno = _make_pheno(5, ['Y'], fid=np.arange(5))  # each in own family
        result = filt.apply(0, {0: pheno}, {})

        assert isinstance(result, SibPairView)
        assert result.n_pairs == 0
        assert len(result.sib1_phenotypes['Y']) == 0

    def test_one_large_family(self):
        """Single family of size k → k*(k-1)/2 pairs."""
        n = 5
        filt = SibPairFilter()
        pheno = _make_pheno(n, ['Y'], fid=np.zeros(n, dtype=int))
        result = filt.apply(0, {0: pheno}, {})

        expected_pairs = n * (n - 1) // 2
        assert result.n_pairs == expected_pairs

    def test_two_pairs(self):
        """Two families of size 2 → 2 pairs."""
        filt = SibPairFilter()
        pheno = _make_pheno(4, ['Y'], fid=np.array([0, 0, 1, 1]))
        result = filt.apply(0, {0: pheno}, {})

        assert result.n_pairs == 2

    def test_mixed_sizes(self):
        """Mixed: family of 3 (3 pairs) + family of 2 (1 pair) + singleton (0)."""
        filt = SibPairFilter()
        pheno = _make_pheno(6, ['Y'], fid=np.array([0, 0, 0, 1, 1, 2]))
        result = filt.apply(0, {0: pheno}, {})

        assert result.n_pairs == 3 + 1  # C(3,2) + C(2,2)

    def test_sib_pair_indices_valid(self):
        """sib1_idx and sib2_idx should be valid sample indices."""
        filt = SibPairFilter()
        n = 10
        pheno = _make_pheno(n, ['Y'], fid=np.repeat(np.arange(5), 2))
        result = filt.apply(0, {0: pheno}, {})

        assert np.all(result.sib1_idx >= 0)
        assert np.all(result.sib1_idx < n)
        assert np.all(result.sib2_idx >= 0)
        assert np.all(result.sib2_idx < n)

    def test_sib_pairs_same_family(self):
        """All sib pairs should share the same FID."""
        filt = SibPairFilter()
        fids = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
        pheno = _make_pheno(len(fids), ['Y'], fid=fids)
        result = filt.apply(0, {0: pheno}, {})

        for i in range(result.n_pairs):
            fid1 = fids[result.sib1_idx[i]]
            fid2 = fids[result.sib2_idx[i]]
            assert fid1 == fid2, f"Pair {i}: FID mismatch {fid1} != {fid2}"

    def test_missing_generation_returns_none(self):
        filt = SibPairFilter()
        result = filt.apply(5, {0: _make_pheno(10, ['Y'])}, {})
        assert result is None

    def test_phenotype_values_correct(self):
        """Sib pair phenotype values should match original phenotypes."""
        filt = SibPairFilter()
        sm = SampleMeta(iid=np.arange(4), fid=np.array([0, 0, 1, 1]))
        pheno = PhenotypeArray(
            samples=sm,
            values={'Y': np.array([10.0, 20.0, 30.0, 40.0])},
        )
        result = filt.apply(0, {0: pheno}, {})

        for i in range(result.n_pairs):
            expected1 = pheno['Y'][result.sib1_idx[i]]
            expected2 = pheno['Y'][result.sib2_idx[i]]
            assert result.sib1_phenotypes['Y'][i] == expected1
            assert result.sib2_phenotypes['Y'][i] == expected2


class TestFilteredViewDataStructures:
    def test_trio_view_fields(self):
        tv = TrioView(
            offspring_phenotypes={'Y': np.array([1.0, 2.0])},
            mother_phenotypes={'Y': np.array([3.0, 4.0])},
            father_phenotypes={'Y': np.array([5.0, 6.0])},
            n_trios=2,
        )
        assert tv.n_trios == 2
        assert len(tv.offspring_phenotypes['Y']) == 2

    def test_sib_pair_view_fields(self):
        spv = SibPairView(
            sib1_phenotypes={'Y': np.array([1.0])},
            sib2_phenotypes={'Y': np.array([2.0])},
            n_pairs=1,
            sib1_idx=np.array([0]),
            sib2_idx=np.array([1]),
        )
        assert spv.n_pairs == 1
        assert len(spv.sib1_idx) == 1
