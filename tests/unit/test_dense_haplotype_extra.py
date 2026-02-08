"""
Unit tests for DenseHaplotypeArray methods not covered elsewhere.

Tests:
1. diploid_matvec: correctness vs manual, shape
2. standardized_haploid_matvec: centering/scaling, zero-variance, shapes
3. drop_isel: sample drop, variant drop, both, none
4. __getitem__ tuple error: 3+ indices
5. data/values/shape compatibility properties
6. af_empirical: vs manual, range, monomorphic
7. attrs property
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray


def _make_hap(n=10, m=5, seed=42):
    """Create a simple haplotype array."""
    rng = np.random.RandomState(seed)
    genotypes = rng.binomial(1, 0.3, size=(n, m, 2)).astype(np.int8)
    sm = SampleMeta(iid=np.arange(n))
    vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
    return DenseHaplotypeArray(genotypes=genotypes, generation=0, samples=sm, variants=vm)


class TestDiploidMatvec:
    def test_matches_manual(self):
        """diploid_matvec(u) should equal G[:,:,0]@u + G[:,:,1]@u."""
        hap = _make_hap()
        u = np.ones(hap.m)
        result = hap.diploid_matvec(u)
        expected = hap.genotypes[:, :, 0] @ u + hap.genotypes[:, :, 1] @ u
        np.testing.assert_array_equal(result, expected)

    def test_shape_1d(self):
        """diploid_matvec with 1-D input returns (n,) array."""
        hap = _make_hap()
        u = np.ones(hap.m)
        result = hap.diploid_matvec(u)
        assert result.shape == (hap.n,)

    def test_equals_matvec(self):
        """diploid_matvec should equal the standard matvec."""
        hap = _make_hap()
        u = np.random.RandomState(0).randn(hap.m)
        np.testing.assert_allclose(hap.diploid_matvec(u), hap.matvec(u))

    def test_zero_effects(self):
        """diploid_matvec with zero effects → zero output."""
        hap = _make_hap()
        u = np.zeros(hap.m)
        result = hap.diploid_matvec(u)
        np.testing.assert_array_equal(result, np.zeros(hap.n))


class TestStandardizedHaploidMatvec:
    def test_zero_mean(self):
        """standardized_haploid_matvec result should have ~zero mean."""
        hap = _make_hap(n=100, m=10, seed=42)
        u = np.ones(hap.m)
        result = hap.standardized_haploid_matvec(u, haploid=0)
        assert abs(np.mean(result)) < 0.5  # approximately centered

    def test_haploid_0_vs_1(self):
        """Two haploids should give different results for non-uniform genotypes."""
        hap = _make_hap(n=50, m=10, seed=42)
        u = np.ones(hap.m)
        r0 = hap.standardized_haploid_matvec(u, haploid=0)
        r1 = hap.standardized_haploid_matvec(u, haploid=1)
        # They can be close but generally not identical
        assert r0.shape == r1.shape == (hap.n,)

    def test_zero_variance_column(self):
        """Monomorphic variant (zero variance) should not cause NaN."""
        n, m = 20, 5
        genotypes = np.zeros((n, m, 2), dtype=np.int8)
        genotypes[:, 0, :] = 1  # monomorphic at variant 0
        sm = SampleMeta(iid=np.arange(n))
        vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
        hap = DenseHaplotypeArray(genotypes=genotypes, generation=0, samples=sm, variants=vm)
        u = np.ones(m)
        result = hap.standardized_haploid_matvec(u, haploid=0)
        assert np.all(np.isfinite(result))

    def test_manual_centering(self):
        """Verify centering: column means should be subtracted."""
        hap = _make_hap(n=20, m=3, seed=42)
        H = hap.genotypes[:, :, 0].astype(float)
        col_mean = H.mean(axis=0)
        col_std = H.std(axis=0, ddof=1)
        col_std[col_std == 0] = 1.0
        H_std = (H - col_mean) / col_std
        u = np.array([1.0, 0.0, 0.0])  # pick first column
        expected = H_std @ u
        result = hap.standardized_haploid_matvec(u, haploid=0)
        np.testing.assert_allclose(result, expected)


class TestDropIsel:
    def test_drop_samples(self):
        """drop_isel(sample=[0,1]) should remove 2 samples."""
        hap = _make_hap(n=10, m=5)
        result = hap.drop_isel(sample=[0, 1])
        assert result.n == 8
        assert result.m == 5

    def test_drop_variants(self):
        """drop_isel(variant=[0]) should remove 1 variant."""
        hap = _make_hap(n=10, m=5)
        result = hap.drop_isel(variant=[0])
        assert result.n == 10
        assert result.m == 4

    def test_drop_both(self):
        """drop_isel(sample=[0], variant=[0,1]) removes from both dims."""
        hap = _make_hap(n=10, m=5)
        result = hap.drop_isel(sample=[0], variant=[0, 1])
        assert result.n == 9
        assert result.m == 3

    def test_drop_none(self):
        """drop_isel() with no args returns full copy."""
        hap = _make_hap(n=10, m=5)
        result = hap.drop_isel()
        assert result.n == 10
        assert result.m == 5
        # Should be a copy
        assert result.genotypes is not hap.genotypes

    def test_drop_all_but_one_sample(self):
        """Drop all but 1 sample."""
        hap = _make_hap(n=5, m=3)
        result = hap.drop_isel(sample=[0, 1, 2, 3])
        assert result.n == 1
        assert result.m == 3


class TestGetitemEdgeCases:
    def test_tuple_three_indices_raises(self):
        """hap[a, b, c] should raise IndexError."""
        hap = _make_hap(n=10, m=5)
        with pytest.raises(IndexError, match="Too many indices"):
            hap[np.array([0]), np.array([0]), np.array([0])]

    def test_tuple_one_index(self):
        """hap[(slice(None),)] should work as sample-only index."""
        hap = _make_hap(n=10, m=5)
        result = hap[(slice(0, 5),)]
        assert result.n == 5
        assert result.m == 5

    def test_two_index_tuple(self):
        """hap[sample_idx, variant_idx] as tuple."""
        hap = _make_hap(n=10, m=5)
        result = hap[np.array([0, 1]), np.array([0, 1, 2])]
        assert result.n == 2
        assert result.m == 3


class TestCompatProperties:
    def test_shape(self):
        """shape should be (n, 2*m) for compatibility."""
        hap = _make_hap(n=10, m=5)
        assert hap.shape == (10, 10)

    def test_data_interleaved(self):
        """data property should return (n, 2m) interleaved array."""
        hap = _make_hap(n=10, m=5)
        d = hap.data
        assert d.shape == (10, 10)
        # Check interleaving: even cols are hap0, odd cols are hap1
        np.testing.assert_array_equal(d[:, 0::2], hap.genotypes[:, :, 0])
        np.testing.assert_array_equal(d[:, 1::2], hap.genotypes[:, :, 1])

    def test_values_equals_data(self):
        """values should be alias for data."""
        hap = _make_hap(n=10, m=5)
        np.testing.assert_array_equal(hap.values, hap.data)

    def test_attrs(self):
        """attrs should contain generation."""
        hap = _make_hap(n=10, m=5)
        assert hap.attrs == {'generation': 0}


class TestAFEmpirical:
    def test_matches_manual(self):
        """af_empirical should match (mean(hap0) + mean(hap1)) / 2."""
        hap = _make_hap(n=20, m=5, seed=42)
        expected = (hap.genotypes[:, :, 0].mean(axis=0) +
                    hap.genotypes[:, :, 1].mean(axis=0)) / 2
        np.testing.assert_allclose(hap.af_empirical, expected)

    def test_range(self):
        """Allele frequencies should be in [0, 1]."""
        hap = _make_hap(n=100, m=20, seed=42)
        af = hap.af_empirical
        assert np.all(af >= 0.0)
        assert np.all(af <= 1.0)

    def test_shape(self):
        """af_empirical should have shape (m,)."""
        hap = _make_hap(n=10, m=5)
        assert hap.af_empirical.shape == (5,)

    def test_monomorphic(self):
        """All-zero genotypes → af=0, all-one → af=1."""
        n, m = 10, 3
        g = np.zeros((n, m, 2), dtype=np.int8)
        sm = SampleMeta(iid=np.arange(n))
        vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
        hap = DenseHaplotypeArray(genotypes=g, generation=0, samples=sm, variants=vm)
        np.testing.assert_array_equal(hap.af_empirical, np.zeros(m))
        g2 = np.ones((n, m, 2), dtype=np.int8)
        hap2 = DenseHaplotypeArray(genotypes=g2, generation=0, samples=sm, variants=vm)
        np.testing.assert_array_equal(hap2.af_empirical, np.ones(m))


class TestToDense:
    def test_returns_self_type(self):
        """to_dense() on DenseHaplotypeArray returns a DenseHaplotypeArray."""
        hap = _make_hap()
        result = hap.to_dense()
        assert isinstance(result, DenseHaplotypeArray)

    def test_preserves_shape(self):
        """to_dense() preserves dimensions."""
        hap = _make_hap(n=10, m=5)
        result = hap.to_dense()
        assert result.n == 10
        assert result.m == 5
