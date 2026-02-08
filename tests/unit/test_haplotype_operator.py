"""
Unit tests for HaplotypeOperator ABC methods via DenseHaplotypeArray.

Tests:
1. matvec: 1D and 2D input shapes, manual verification
2. rmatvec: 1D and 2D, transpose relationship with matvec
3. standardized_matvec: zero-mean output
4. standardized_rmatvec: shape correctness
5. recompute_af: matches manual, range, shape
6. to_diploid_standardized: centered, shape
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray


def _make_hap(n=10, m=5, seed=42):
    """Create a DenseHaplotypeArray with known genotypes."""
    rng = np.random.RandomState(seed)
    genotypes = rng.binomial(1, 0.3, size=(n, m, 2)).astype(np.int8)
    sm = SampleMeta(iid=np.arange(n))
    vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
    return DenseHaplotypeArray(genotypes=genotypes, generation=0, samples=sm, variants=vm)


class TestMatvec:
    def test_1d_shape(self):
        """matvec with 1D vector should return (n,)."""
        hap = _make_hap(n=10, m=5)
        v = np.ones(5)
        result = hap.matvec(v)
        assert result.shape == (10,)

    def test_2d_shape(self):
        """matvec with 2D array (m, k) should return (n, k)."""
        hap = _make_hap(n=10, m=5)
        v = np.ones((5, 3))
        result = hap.matvec(v)
        assert result.shape == (10, 3)

    def test_manual_verification(self):
        """matvec should equal diploid_genotypes @ v."""
        hap = _make_hap(n=8, m=4, seed=0)
        v = np.array([1.0, -1.0, 0.5, -0.5])
        result = hap.matvec(v)
        expected = hap.diploid_genotypes @ v
        np.testing.assert_allclose(result, expected)

    def test_zero_vector(self):
        """matvec with zero vector should return zeros."""
        hap = _make_hap(n=10, m=5)
        result = hap.matvec(np.zeros(5))
        np.testing.assert_array_equal(result, np.zeros(10))


class TestRmatvec:
    def test_1d_shape(self):
        """rmatvec with 1D vector should return (m,)."""
        hap = _make_hap(n=10, m=5)
        v = np.ones(10)
        result = hap.rmatvec(v)
        assert result.shape == (5,)

    def test_2d_shape(self):
        """rmatvec with 2D array (n, k) should return (m, k)."""
        hap = _make_hap(n=10, m=5)
        v = np.ones((10, 3))
        result = hap.rmatvec(v)
        assert result.shape == (5, 3)

    def test_transpose_relationship(self):
        """v.T @ G @ w should equal (G.T @ v).T @ w."""
        hap = _make_hap(n=10, m=5, seed=0)
        v = np.random.randn(10)
        w = np.random.randn(5)
        lhs = v @ hap.matvec(w)
        rhs = hap.rmatvec(v) @ w
        np.testing.assert_allclose(lhs, rhs)


class TestStandardizedMatvec:
    def test_zero_mean(self):
        """Standardized matvec with ones should produce approximately zero mean."""
        hap = _make_hap(n=200, m=20, seed=42)
        v = np.ones(20) / np.sqrt(20)
        result = hap.standardized_matvec(v)
        assert abs(np.mean(result)) < 0.5  # Centered

    def test_shape(self):
        """Standardized matvec shape should match regular matvec."""
        hap = _make_hap(n=10, m=5)
        v = np.random.randn(5)
        result = hap.standardized_matvec(v)
        assert result.shape == (10,)

    def test_with_explicit_af(self):
        """standardized_matvec with explicit AF should work."""
        hap = _make_hap(n=10, m=5, seed=42)
        af = hap.recompute_af()
        v = np.ones(5)
        result = hap.standardized_matvec(v, af=af)
        assert result.shape == (10,)
        # Mean should be closer to zero than raw matvec
        raw = hap.matvec(v)
        assert abs(np.mean(result)) <= abs(np.mean(raw)) + 0.01


class TestStandardizedRmatvec:
    def test_shape_1d(self):
        """Standardized rmatvec with 1D input."""
        hap = _make_hap(n=10, m=5)
        v = np.random.randn(10)
        result = hap.standardized_rmatvec(v)
        assert result.shape == (5,)


class TestRecomputeAF:
    def test_shape(self):
        """AF should be (m,)."""
        hap = _make_hap(n=10, m=5)
        af = hap.recompute_af()
        assert af.shape == (5,)

    def test_range(self):
        """AF should be in [0, 1]."""
        hap = _make_hap(n=100, m=20, seed=42)
        af = hap.recompute_af()
        assert np.all(af >= 0)
        assert np.all(af <= 1)

    def test_manual_match(self):
        """AF should equal mean(diploid_genotypes) / 2."""
        hap = _make_hap(n=50, m=10, seed=42)
        af = hap.recompute_af()
        expected = hap.diploid_genotypes.mean(axis=0) / 2
        np.testing.assert_allclose(af, expected)

    def test_monomorphic(self):
        """All-zero genotypes → AF = 0."""
        n, m = 5, 3
        genotypes = np.zeros((n, m, 2), dtype=np.int8)
        sm = SampleMeta(iid=np.arange(n))
        vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
        hap = DenseHaplotypeArray(genotypes=genotypes, generation=0, samples=sm, variants=vm)
        af = hap.recompute_af()
        np.testing.assert_array_equal(af, np.zeros(m))


class TestToDiploidStandardized:
    def test_centered(self):
        """Standardized genotypes should have approximately zero column means."""
        hap = _make_hap(n=200, m=10, seed=42)
        G_std = hap.to_diploid_standardized()
        col_means = G_std.mean(axis=0)
        np.testing.assert_allclose(col_means, 0.0, atol=1e-10)

    def test_shape(self):
        """Standardized genotype matrix should be (n, m)."""
        hap = _make_hap(n=10, m=5)
        G_std = hap.to_diploid_standardized()
        assert G_std.shape == (10, 5)

    def test_with_scale(self):
        """With scale=True, columns should have unit variance (where possible)."""
        hap = _make_hap(n=200, m=10, seed=42)
        G_std = hap.to_diploid_standardized(scale=True)
        # Columns with non-zero variance should have var ~1
        col_vars = G_std.var(axis=0)
        nonzero = col_vars > 0
        if np.any(nonzero):
            np.testing.assert_allclose(col_vars[nonzero], 1.0, atol=0.15)
