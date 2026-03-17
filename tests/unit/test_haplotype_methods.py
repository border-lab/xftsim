"""
Unit tests for DenseHaplotypeArray methods that lack direct testing.

Tests:
1. matvec_maternal / matvec_paternal: correctness vs manual computation
2. standardized_matvec: centering, custom af
3. rmatvec: transpose property
4. recompute_af: range [0,1], shape, caching
5. to_diploid: shape, values
6. __getitem__: integer, slice, boolean, fancy indexing
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray


def _make_haplotypes(n=20, m=10, seed=42):
    rng = np.random.RandomState(seed)
    geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
    sm = SampleMeta(iid=np.arange(n))
    vm = VariantMeta(vid=np.arange(m), af=np.full(m, 0.5))
    return DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)


class TestMatvecMaternalPaternal:
    def test_maternal_manual(self):
        """matvec_maternal should be geno[:,:,0] @ v."""
        hap = _make_haplotypes()
        v = np.random.RandomState(0).randn(hap.m)
        result = hap.matvec_maternal(v)
        expected = hap.genotypes[:, :, 0].astype(np.float64) @ v
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_paternal_manual(self):
        """matvec_paternal should be geno[:,:,1] @ v."""
        hap = _make_haplotypes()
        v = np.random.RandomState(0).randn(hap.m)
        result = hap.matvec_paternal(v)
        expected = hap.genotypes[:, :, 1].astype(np.float64) @ v
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_maternal_plus_paternal_equals_diploid(self):
        """matvec_maternal(v) + matvec_paternal(v) == matvec(v)."""
        hap = _make_haplotypes()
        v = np.random.RandomState(1).randn(hap.m)
        mat = hap.matvec_maternal(v)
        pat = hap.matvec_paternal(v)
        diploid = hap.matvec(v)
        np.testing.assert_allclose(mat + pat, diploid, atol=1e-10)

    def test_maternal_shape_1d(self):
        """matvec_maternal with 1D v returns 1D result."""
        hap = _make_haplotypes()
        v = np.ones(hap.m)
        result = hap.matvec_maternal(v)
        assert result.ndim == 1
        assert result.shape == (hap.n,)

    def test_maternal_shape_2d(self):
        """matvec_maternal with 2D v returns 2D result."""
        hap = _make_haplotypes()
        v = np.ones((hap.m, 3))
        result = hap.matvec_maternal(v)
        assert result.ndim == 2
        assert result.shape == (hap.n, 3)

    def test_zero_effects(self):
        """Zero effect vector should produce all-zero genetic values."""
        hap = _make_haplotypes()
        v = np.zeros(hap.m)
        np.testing.assert_allclose(hap.matvec_maternal(v), 0.0)
        np.testing.assert_allclose(hap.matvec_paternal(v), 0.0)
        np.testing.assert_allclose(hap.matvec(v), 0.0)


class TestStandardizedMatvec:
    def test_centering(self):
        """standardized_matvec should produce mean closer to 0 than raw matvec."""
        hap = _make_haplotypes(n=500, m=20, seed=42)
        v = np.random.RandomState(0).randn(hap.m)
        af = hap.recompute_af()
        raw = hap.matvec(v)
        centered = hap.standardized_matvec(v, af=af)
        assert abs(np.mean(centered)) < abs(np.mean(raw)) + 1e-10

    def test_custom_af(self):
        """standardized_matvec with custom af should use it."""
        hap = _make_haplotypes()
        v = np.ones(hap.m)
        af_custom = np.full(hap.m, 0.3)
        result = hap.standardized_matvec(v, af=af_custom)
        G = hap.diploid_genotypes.astype(np.float64)
        denom = np.sqrt(2 * af_custom * (1 - af_custom))
        denom[denom == 0] = 1.0
        expected = ((G - 2 * af_custom) / denom) @ v
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_formula(self):
        """standardized_matvec = raw - 2*af@v."""
        hap = _make_haplotypes()
        v = np.random.RandomState(0).randn(hap.m)
        af = hap.recompute_af()
        result = hap.standardized_matvec(v, af=af)
        G = hap.diploid_genotypes.astype(np.float64)
        denom = np.sqrt(2 * af * (1 - af))
        denom[denom == 0] = 1.0
        expected = ((G - 2 * af) / denom) @ v
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestRmatvec:
    def test_rmatvec_transpose_property(self):
        """rmatvec(v) should equal G.T @ v where G is diploid matrix."""
        hap = _make_haplotypes()
        v = np.random.RandomState(0).randn(hap.n)
        result = hap.rmatvec(v)
        G = hap.genotypes.sum(axis=2).astype(np.float64)
        expected = G.T @ v
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_rmatvec_shape_1d(self):
        """rmatvec with 1D v returns shape (m,)."""
        hap = _make_haplotypes()
        v = np.ones(hap.n)
        result = hap.rmatvec(v)
        assert result.shape == (hap.m,)

    def test_rmatvec_shape_2d(self):
        """rmatvec with 2D v returns shape (m, k)."""
        hap = _make_haplotypes()
        v = np.ones((hap.n, 3))
        result = hap.rmatvec(v)
        assert result.shape == (hap.m, 3)


class TestRecomputeAF:
    def test_range(self):
        """Allele frequencies should be in [0, 1]."""
        hap = _make_haplotypes(n=100, m=20, seed=42)
        af = hap.recompute_af()
        assert np.all(af >= 0.0)
        assert np.all(af <= 1.0)

    def test_shape(self):
        """AF should be shape (m,)."""
        hap = _make_haplotypes()
        af = hap.recompute_af()
        assert af.shape == (hap.m,)

    def test_manual_computation(self):
        """AF should equal mean of diploid genotype / 2."""
        hap = _make_haplotypes()
        af = hap.recompute_af()
        diploid = hap.genotypes.sum(axis=2).astype(np.float64)
        expected = diploid.mean(axis=0) / 2.0
        np.testing.assert_allclose(af, expected, atol=1e-10)

    def test_consistent(self):
        """Two calls should return the same values."""
        hap = _make_haplotypes()
        af1 = hap.recompute_af()
        af2 = hap.recompute_af()
        np.testing.assert_array_equal(af1, af2)


class TestGetitem:
    def test_integer_index_via_list(self):
        """List-of-one indexing returns single-individual HaplotypeArray."""
        hap = _make_haplotypes()
        sub = hap[[0]]
        assert sub.n == 1
        assert sub.m == hap.m
        np.testing.assert_array_equal(sub.genotypes[0], hap.genotypes[0])

    def test_slice_index(self):
        """Slice indexing returns subset."""
        hap = _make_haplotypes()
        sub = hap[2:5]
        assert sub.n == 3
        np.testing.assert_array_equal(sub.genotypes, hap.genotypes[2:5])

    def test_boolean_index(self):
        """Boolean mask indexing."""
        hap = _make_haplotypes()
        mask = np.zeros(hap.n, dtype=bool)
        mask[0] = mask[3] = mask[7] = True
        sub = hap[mask]
        assert sub.n == 3
        np.testing.assert_array_equal(sub.genotypes, hap.genotypes[mask])

    def test_fancy_index(self):
        """Fancy (array) indexing."""
        hap = _make_haplotypes()
        idx = np.array([0, 5, 10, 15])
        sub = hap[idx]
        assert sub.n == 4
        np.testing.assert_array_equal(sub.genotypes, hap.genotypes[idx])

    def test_getitem_preserves_variants(self):
        """Subsetting should preserve variant metadata."""
        hap = _make_haplotypes()
        sub = hap[0:5]
        assert sub.m == hap.m
        np.testing.assert_array_equal(sub.variants.vid, hap.variants.vid)


class TestToDense:
    def test_returns_same_type(self):
        """to_dense() on DenseHaplotypeArray should return equivalent object."""
        hap = _make_haplotypes()
        dense = hap.to_dense()
        assert isinstance(dense, DenseHaplotypeArray)
        np.testing.assert_array_equal(dense.genotypes, hap.genotypes)

    def test_preserves_metadata(self):
        """to_dense() should preserve sample and variant metadata."""
        hap = _make_haplotypes()
        dense = hap.to_dense()
        assert dense.n == hap.n
        assert dense.m == hap.m


class TestMatvecConsistency:
    def test_matvec_agrees_with_manual_diploid(self):
        """matvec should compute diploid genotype @ effects."""
        hap = _make_haplotypes()
        v = np.random.RandomState(42).randn(hap.m)
        result = hap.matvec(v)
        G = hap.genotypes.sum(axis=2).astype(np.float64)
        expected = G @ v
        np.testing.assert_allclose(result, expected, atol=1e-10)
