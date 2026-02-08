"""
Unit tests for DenseHaplotypeArray subsetting, indexing, and property methods.

Tests:
1. subset: sample-only, variant-only, both, copy vs no-copy
2. __getitem__: single index, tuple, 2-tuple, 3-tuple error
3. drop_isel: drop samples, drop variants, both
4. data/values: interleaved format, shape
5. attrs: generation in attrs
6. af_empirical: matches manual
7. diploid_matvec: matches manual
8. standardized_haploid_matvec: shape, centered
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


def _make_hap(n=10, m=5, seed=42):
    return TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)


class TestSubset:
    def test_sample_subset(self):
        """Subset by sample indices."""
        hap = _make_hap(n=10, m=5)
        sub = hap.subset(sample_idx=np.array([0, 2, 4]))
        assert sub.n == 3
        assert sub.m == 5
        np.testing.assert_array_equal(sub.genotypes, hap.genotypes[[0, 2, 4]])

    def test_variant_subset(self):
        """Subset by variant indices."""
        hap = _make_hap(n=10, m=5)
        sub = hap.subset(variant_idx=np.array([1, 3]))
        assert sub.n == 10
        assert sub.m == 2
        np.testing.assert_array_equal(sub.genotypes, hap.genotypes[:, [1, 3], :])

    def test_both_subset(self):
        """Subset by both samples and variants."""
        hap = _make_hap(n=10, m=5)
        sub = hap.subset(sample_idx=[0, 1], variant_idx=[0, 1])
        assert sub.n == 2
        assert sub.m == 2

    def test_subset_copy(self):
        """copy=True should produce independent data."""
        hap = _make_hap(n=10, m=5)
        sub = hap.subset(sample_idx=[0, 1], copy=True)
        sub.genotypes[0, 0, 0] = 99
        assert hap.genotypes[0, 0, 0] != 99

    def test_subset_no_copy(self):
        """copy=False should share memory."""
        hap = _make_hap(n=10, m=5)
        sub = hap.subset(sample_idx=slice(None), variant_idx=slice(None), copy=False)
        # Modifying sub should affect original (if data is contiguous)
        original_val = hap.genotypes[0, 0, 0].copy()
        sub.genotypes[0, 0, 0] = 99
        assert hap.genotypes[0, 0, 0] == 99
        hap.genotypes[0, 0, 0] = original_val  # restore

    def test_subset_preserves_generation(self):
        """Generation should be preserved through subset."""
        hap = _make_hap(n=10, m=5)
        sub = hap.subset(sample_idx=[0, 1])
        assert sub.generation == hap.generation

    def test_subset_empty_result(self):
        """Empty index produces 0-sample array."""
        hap = _make_hap(n=10, m=5)
        sub = hap.subset(sample_idx=np.array([], dtype=int))
        assert sub.n == 0


class TestGetitem:
    def test_single_index(self):
        """hap[idx] should subset samples."""
        hap = _make_hap(n=10, m=5)
        sub = hap[np.array([0, 1, 2])]
        assert sub.n == 3

    def test_tuple_one_element(self):
        """hap[(idx,)] should subset samples."""
        hap = _make_hap(n=10, m=5)
        sub = hap[(np.array([0, 1]),)]
        assert sub.n == 2

    def test_tuple_two_elements(self):
        """hap[sample_idx, variant_idx]."""
        hap = _make_hap(n=10, m=5)
        sub = hap[np.array([0, 1]), np.array([2, 3])]
        assert sub.n == 2
        assert sub.m == 2

    def test_tuple_three_elements_raises(self):
        """hap[a, b, c] should raise IndexError."""
        hap = _make_hap(n=10, m=5)
        with pytest.raises(IndexError, match="Too many"):
            hap[0, 1, 2]

    def test_slice_samples(self):
        """hap[:5] should subset first 5 samples."""
        hap = _make_hap(n=10, m=5)
        sub = hap[:5]
        assert sub.n == 5


class TestDropIsel:
    def test_drop_samples(self):
        """Dropping samples by index."""
        hap = _make_hap(n=10, m=5)
        result = hap.drop_isel(sample=[0, 1, 2])
        assert result.n == 7

    def test_drop_variants(self):
        """Dropping variants by index."""
        hap = _make_hap(n=10, m=5)
        result = hap.drop_isel(variant=[0])
        assert result.m == 4

    def test_drop_both(self):
        """Dropping both samples and variants."""
        hap = _make_hap(n=10, m=5)
        result = hap.drop_isel(sample=[0], variant=[0])
        assert result.n == 9
        assert result.m == 4

    def test_drop_none(self):
        """Dropping nothing returns full copy."""
        hap = _make_hap(n=10, m=5)
        result = hap.drop_isel()
        assert result.n == 10
        assert result.m == 5


class TestDataProperties:
    def test_data_shape(self):
        """data property should be (n, 2m)."""
        hap = _make_hap(n=10, m=5)
        assert hap.data.shape == (10, 10)

    def test_data_interleaved(self):
        """data should interleave haplotypes: [h0_v0, h1_v0, h0_v1, h1_v1, ...]."""
        hap = _make_hap(n=10, m=5)
        d = hap.data
        for v in range(5):
            np.testing.assert_array_equal(d[:, 2*v], hap.genotypes[:, v, 0])
            np.testing.assert_array_equal(d[:, 2*v+1], hap.genotypes[:, v, 1])

    def test_values_is_data(self):
        """values should be same as data."""
        hap = _make_hap(n=10, m=5)
        np.testing.assert_array_equal(hap.values, hap.data)

    def test_attrs(self):
        """attrs should contain generation."""
        hap = _make_hap(n=10, m=5)
        assert 'generation' in hap.attrs
        assert hap.attrs['generation'] == hap.generation

    def test_shape_property(self):
        """shape property should be (n, 2m)."""
        hap = _make_hap(n=10, m=5)
        assert hap.shape == (10, 10)


class TestAfEmpirical:
    def test_shape(self):
        """af_empirical should be (m,)."""
        hap = _make_hap(n=10, m=5)
        af = hap.af_empirical
        assert af.shape == (5,)

    def test_range(self):
        """AF should be in [0, 1]."""
        hap = _make_hap(n=100, m=20, seed=42)
        af = hap.af_empirical
        assert np.all(af >= 0)
        assert np.all(af <= 1)

    def test_matches_recompute(self):
        """af_empirical should match recompute_af."""
        hap = _make_hap(n=50, m=10, seed=42)
        np.testing.assert_allclose(hap.af_empirical, hap.recompute_af())

    def test_all_zero_genotypes(self):
        """All-zero genotypes → AF = 0."""
        sm = SampleMeta(iid=np.arange(5))
        vm = VariantMeta(vid=np.array(['v0', 'v1']))
        g = np.zeros((5, 2, 2), dtype=np.int8)
        hap = DenseHaplotypeArray(genotypes=g, generation=0, samples=sm, variants=vm)
        np.testing.assert_array_equal(hap.af_empirical, [0.0, 0.0])


class TestDiploidMatvec:
    def test_shape(self):
        """diploid_matvec with (m,) returns (n,)."""
        hap = _make_hap(n=10, m=5)
        result = hap.diploid_matvec(np.ones(5))
        assert result.shape == (10,)

    def test_matches_matvec(self):
        """diploid_matvec should equal matvec (both compute G@u)."""
        hap = _make_hap(n=10, m=5, seed=42)
        u = np.random.RandomState(0).randn(5)
        np.testing.assert_allclose(hap.diploid_matvec(u), hap.matvec(u))


class TestStandardizedHaploidMatvec:
    def test_shape(self):
        """standardized_haploid_matvec should return (n,)."""
        hap = _make_hap(n=20, m=5)
        result = hap.standardized_haploid_matvec(np.ones(5), haploid=0)
        assert result.shape == (20,)

    def test_approximately_centered(self):
        """Result should have approximately zero mean."""
        hap = _make_hap(n=200, m=20, seed=42)
        u = np.ones(20) / np.sqrt(20)
        result = hap.standardized_haploid_matvec(u, haploid=0)
        assert abs(np.mean(result)) < 0.2

    def test_haploid_0_vs_1_differ(self):
        """Maternal and paternal haploid matvec should differ."""
        hap = _make_hap(n=20, m=10, seed=42)
        u = np.random.RandomState(0).randn(10)
        r0 = hap.standardized_haploid_matvec(u, haploid=0)
        r1 = hap.standardized_haploid_matvec(u, haploid=1)
        assert not np.allclose(r0, r1)
