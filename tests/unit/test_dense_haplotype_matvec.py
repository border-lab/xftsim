"""
Unit tests for DenseHaplotypeArray linear operator methods.

Tests cover:
1. matvec with 1-D vector: result shape is (n,)
2. matvec with 2-D vector: result shape is (n, k)
3. rmatvec: G.T @ v, result shape is (m,) or (m, k)
4. matvec_maternal + matvec_paternal = matvec (maternal + paternal = diploid)
5. diploid_matvec equals matvec (sum of both haplotypes)
6. standardized_matvec: centering by 2*af, result has mean≈0
7. standardized_matvec with custom af vector
8. standardized_haploid_matvec: works on individual haplotypes
9. to_diploid_standardized with scale=True: columns have unit variance
10. to_diploid_standardized with scale=False: only centered
11. matvec with zeros vector: result is zeros
12. matvec with ones vector: result is row sums of genotype matrix
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from testdata import TestSimulation

from xftsim.struct import DenseHaplotypeArray, SampleMeta, VariantMeta


class TestMatvec1D:
    """Test matvec with 1-D vectors."""

    def test_matvec_1d_shape(self):
        """matvec with 1-D vector should return (n,) array."""
        n, m = 10, 8
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        v = np.random.randn(m)
        result = hap.matvec(v)
        assert result.shape == (n,), f"Expected shape ({n},), got {result.shape}"

    def test_matvec_1d_equals_diploid_matmul(self):
        """matvec should match direct matrix multiplication."""
        n, m = 15, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=123)
        v = np.random.randn(m)
        result = hap.matvec(v)
        expected = hap.diploid_genotypes @ v
        np.testing.assert_allclose(result, expected)

    def test_matvec_zeros_vector(self):
        """matvec with zeros vector should return zeros."""
        n, m = 10, 5
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        v = np.zeros(m)
        result = hap.matvec(v)
        np.testing.assert_array_equal(result, np.zeros(n))

    def test_matvec_ones_vector(self):
        """matvec with ones vector should return row sums."""
        n, m = 10, 5
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        v = np.ones(m)
        result = hap.matvec(v)
        expected = hap.diploid_genotypes.sum(axis=1)
        np.testing.assert_array_equal(result, expected)


class TestMatvec2D:
    """Test matvec with 2-D matrices."""

    def test_matvec_2d_shape(self):
        """matvec with 2-D matrix should return (n, k) array."""
        n, m, k = 10, 8, 3
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        V = np.random.randn(m, k)
        result = hap.matvec(V)
        assert result.shape == (n, k), f"Expected shape ({n}, {k}), got {result.shape}"

    def test_matvec_2d_equals_diploid_matmul(self):
        """matvec with 2-D should match direct matrix multiplication."""
        n, m, k = 15, 10, 4
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=123)
        V = np.random.randn(m, k)
        result = hap.matvec(V)
        expected = hap.diploid_genotypes @ V
        np.testing.assert_allclose(result, expected)

    def test_matvec_2d_columns_match_1d(self):
        """Each column of 2-D matvec should match 1-D matvec of that column."""
        n, m, k = 12, 8, 3
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=456)
        V = np.random.randn(m, k)
        result = hap.matvec(V)

        for i in range(k):
            expected_i = hap.matvec(V[:, i])
            np.testing.assert_allclose(result[:, i], expected_i)


class TestRmatvec:
    """Test rmatvec (transposed matrix-vector product)."""

    def test_rmatvec_1d_shape(self):
        """rmatvec with 1-D vector should return (m,) array."""
        n, m = 10, 8
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        v = np.random.randn(n)
        result = hap.rmatvec(v)
        assert result.shape == (m,), f"Expected shape ({m},), got {result.shape}"

    def test_rmatvec_1d_equals_transpose_matmul(self):
        """rmatvec should match G.T @ v."""
        n, m = 15, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=123)
        v = np.random.randn(n)
        result = hap.rmatvec(v)
        expected = hap.diploid_genotypes.T @ v
        np.testing.assert_allclose(result, expected)

    def test_rmatvec_2d_shape(self):
        """rmatvec with 2-D matrix should return (m, k) array."""
        n, m, k = 10, 8, 3
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        V = np.random.randn(n, k)
        result = hap.rmatvec(V)
        assert result.shape == (m, k), f"Expected shape ({m}, {k}), got {result.shape}"

    def test_rmatvec_2d_equals_transpose_matmul(self):
        """rmatvec with 2-D should match G.T @ V."""
        n, m, k = 15, 10, 4
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=123)
        V = np.random.randn(n, k)
        result = hap.rmatvec(V)
        expected = hap.diploid_genotypes.T @ V
        np.testing.assert_allclose(result, expected)


class TestHaplotypeSplit:
    """Test maternal/paternal split and equivalence to diploid."""

    def test_matvec_maternal_plus_paternal_equals_matvec(self):
        """matvec_maternal + matvec_paternal should equal matvec."""
        n, m = 10, 8
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        v = np.random.randn(m)

        maternal = hap.matvec_maternal(v)
        paternal = hap.matvec_paternal(v)
        diploid = hap.matvec(v)

        np.testing.assert_allclose(maternal + paternal, diploid)

    def test_matvec_maternal_shape(self):
        """matvec_maternal should return (n,) or (n, k) array."""
        n, m = 10, 8
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        v = np.random.randn(m)
        result = hap.matvec_maternal(v)
        assert result.shape == (n,)

    def test_matvec_paternal_shape(self):
        """matvec_paternal should return (n,) or (n, k) array."""
        n, m = 10, 8
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        v = np.random.randn(m)
        result = hap.matvec_paternal(v)
        assert result.shape == (n,)

    def test_diploid_matvec_equals_matvec(self):
        """diploid_matvec should equal matvec (both compute G @ v)."""
        n, m = 15, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=123)
        v = np.random.randn(m)

        result1 = hap.diploid_matvec(v)
        result2 = hap.matvec(v)

        np.testing.assert_allclose(result1, result2)

    def test_maternal_paternal_2d(self):
        """Maternal/paternal should work with 2-D vectors."""
        n, m, k = 10, 8, 3
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        V = np.random.randn(m, k)

        maternal = hap.matvec_maternal(V)
        paternal = hap.matvec_paternal(V)
        diploid = hap.matvec(V)

        assert maternal.shape == (n, k)
        assert paternal.shape == (n, k)
        np.testing.assert_allclose(maternal + paternal, diploid)


class TestStandardizedMatvec:
    """Test standardized_matvec (centered diploid matvec)."""

    def test_standardized_matvec_centering(self):
        """standardized_matvec should center genotypes by 2*af."""
        n, m = 20, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        v = np.random.randn(m)

        result = hap.standardized_matvec(v)

        # Manual computation
        af = hap.af_empirical
        G = hap.diploid_genotypes.astype(np.float64)
        expected = (G - 2 * af) @ v

        np.testing.assert_allclose(result, expected)

    def test_standardized_matvec_mean_zero(self):
        """standardized_matvec with uniform vector should have mean ≈ 0."""
        n, m = 100, 50
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=123)
        v = np.ones(m)

        result = hap.standardized_matvec(v)

        # Should be approximately zero since E[G - 2*af] = 0
        # With large n, this should be close to zero
        np.testing.assert_allclose(result.mean(), 0.0, atol=0.5)

    def test_standardized_matvec_custom_af(self):
        """standardized_matvec should use custom AF when provided."""
        n, m = 15, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        custom_af = np.full(m, 0.5)  # Use 0.5 for all variants
        v = np.random.randn(m)

        result = hap.standardized_matvec(v, af=custom_af)

        # Manual computation with custom AF
        G = hap.diploid_genotypes.astype(np.float64)
        expected = (G - 2 * custom_af) @ v

        np.testing.assert_allclose(result, expected)

    def test_standardized_matvec_2d(self):
        """standardized_matvec should work with 2-D vectors."""
        n, m, k = 20, 10, 3
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=456)
        V = np.random.randn(m, k)

        result = hap.standardized_matvec(V)

        assert result.shape == (n, k)

        # Each column should match 1-D standardized_matvec
        for i in range(k):
            expected_i = hap.standardized_matvec(V[:, i])
            np.testing.assert_allclose(result[:, i], expected_i)


class TestStandardizedHaploidMatvec:
    """Test standardized_haploid_matvec (center & scale single haplotype)."""

    def test_standardized_haploid_matvec_maternal(self):
        """standardized_haploid_matvec should work on maternal haplotype."""
        n, m = 20, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        v = np.random.randn(m)

        result = hap.standardized_haploid_matvec(v, haploid=0)

        # Manual computation
        H = hap.genotypes[:, :, 0].astype(np.float64)
        col_mean = H.mean(axis=0)
        col_std = H.std(axis=0, ddof=1)
        col_std[col_std == 0] = 1.0
        H_std = (H - col_mean) / col_std
        expected = H_std @ v

        np.testing.assert_allclose(result, expected)

    def test_standardized_haploid_matvec_paternal(self):
        """standardized_haploid_matvec should work on paternal haplotype."""
        n, m = 20, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        v = np.random.randn(m)

        result = hap.standardized_haploid_matvec(v, haploid=1)

        # Manual computation
        H = hap.genotypes[:, :, 1].astype(np.float64)
        col_mean = H.mean(axis=0)
        col_std = H.std(axis=0, ddof=1)
        col_std[col_std == 0] = 1.0
        H_std = (H - col_mean) / col_std
        expected = H_std @ v

        np.testing.assert_allclose(result, expected)

    def test_standardized_haploid_matvec_shape(self):
        """standardized_haploid_matvec should return (n,) array."""
        n, m = 15, 8
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=123)
        v = np.random.randn(m)

        result_mat = hap.standardized_haploid_matvec(v, haploid=0)
        result_pat = hap.standardized_haploid_matvec(v, haploid=1)

        assert result_mat.shape == (n,)
        assert result_pat.shape == (n,)

    def test_standardized_haploid_matvec_zero_variance_protection(self):
        """standardized_haploid_matvec should protect against zero variance."""
        n, m = 5, 3
        # Create genotypes where first variant is all-zero (zero variance)
        geno = np.array(
            [
                [[0, 1], [1, 0], [0, 1]],
                [[0, 0], [0, 1], [1, 0]],
                [[0, 1], [1, 0], [0, 1]],
                [[0, 0], [0, 1], [1, 0]],
                [[0, 1], [1, 0], [0, 1]],
            ],
            dtype=np.int8,
        )
        sm = SampleMeta(iid=np.arange(n))
        vm = VariantMeta(vid=np.array([f"v{i}" for i in range(m)]))
        hap = DenseHaplotypeArray(geno, samples=sm, variants=vm)

        v = np.ones(m)
        result = hap.standardized_haploid_matvec(v, haploid=0)

        # Should not raise an error and should not contain NaN/inf
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))


class TestToDiploidStandardized:
    """Test to_diploid_standardized method."""

    def test_to_diploid_standardized_scale_false(self):
        """to_diploid_standardized with scale=False should only center."""
        n, m = 15, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)

        G_std = hap.to_diploid_standardized(scale=False)

        # Manual computation
        af = hap.af_empirical
        G = hap.diploid_genotypes.astype(np.float64)
        expected = G - 2 * af

        np.testing.assert_allclose(G_std, expected)

    def test_to_diploid_standardized_scale_true(self):
        """to_diploid_standardized with scale=True should center and scale."""
        n, m = 15, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)

        G_std = hap.to_diploid_standardized(scale=True)

        # Manual computation
        af = hap.af_empirical
        G = hap.diploid_genotypes.astype(np.float64)
        G_centered = G - 2 * af
        denom = np.sqrt(2 * af * (1 - af))
        denom[denom == 0] = 1.0
        expected = G_centered / denom

        np.testing.assert_allclose(G_std, expected)

    def test_to_diploid_standardized_scale_true_unit_variance(self):
        """to_diploid_standardized with scale=True should produce unit variance columns."""
        n, m = 100, 20
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=123)

        G_std = hap.to_diploid_standardized(scale=True)

        # Compute variance of each column (excluding those with zero variance in original)
        col_var = G_std.var(axis=0, ddof=1)

        # Variants with non-zero variance in original should have variance ≈ 1
        # (or variance = 0 if original had zero variance)
        af = hap.af_empirical
        non_zero_var = (af > 0) & (af < 1)

        # Check that non-zero-variance columns have variance close to 1
        # Use looser tolerance since we have finite sample size (n=100)
        if non_zero_var.any():
            np.testing.assert_allclose(
                col_var[non_zero_var], 1.0, atol=0.2,
                err_msg="Columns should have approximately unit variance"
            )

    def test_to_diploid_standardized_custom_af(self):
        """to_diploid_standardized should use custom AF when provided."""
        n, m = 15, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        custom_af = np.full(m, 0.3)

        G_std = hap.to_diploid_standardized(af=custom_af, scale=False)

        # Manual computation with custom AF
        G = hap.diploid_genotypes.astype(np.float64)
        expected = G - 2 * custom_af

        np.testing.assert_allclose(G_std, expected)

    def test_to_diploid_standardized_shape(self):
        """to_diploid_standardized should return (n, m) array."""
        n, m = 20, 15
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=456)

        G_std_no_scale = hap.to_diploid_standardized(scale=False)
        G_std_with_scale = hap.to_diploid_standardized(scale=True)

        assert G_std_no_scale.shape == (n, m)
        assert G_std_with_scale.shape == (n, m)


class TestEdgeCases:
    """Test edge cases for matvec operations."""

    def test_matvec_single_sample(self):
        """matvec should work with n=1."""
        n, m = 1, 5
        geno = np.random.randint(0, 2, size=(n, m, 2), dtype=np.int8)
        sm = SampleMeta(iid=np.arange(n))
        vm = VariantMeta(vid=np.array([f"v{i}" for i in range(m)]))
        hap = DenseHaplotypeArray(geno, samples=sm, variants=vm)

        v = np.random.randn(m)
        result = hap.matvec(v)

        assert result.shape == (n,)
        expected = hap.diploid_genotypes @ v
        np.testing.assert_allclose(result, expected)

    def test_matvec_single_variant(self):
        """matvec should work with m=1."""
        n, m = 10, 1
        geno = np.random.randint(0, 2, size=(n, m, 2), dtype=np.int8)
        sm = SampleMeta(iid=np.arange(n))
        vm = VariantMeta(vid=np.array([f"v{i}" for i in range(m)]))
        hap = DenseHaplotypeArray(geno, samples=sm, variants=vm)

        v = np.random.randn(m)
        result = hap.matvec(v)

        assert result.shape == (n,)
        expected = hap.diploid_genotypes @ v
        np.testing.assert_allclose(result, expected)

    def test_rmatvec_single_sample(self):
        """rmatvec should work with n=1."""
        n, m = 1, 5
        geno = np.random.randint(0, 2, size=(n, m, 2), dtype=np.int8)
        sm = SampleMeta(iid=np.arange(n))
        vm = VariantMeta(vid=np.array([f"v{i}" for i in range(m)]))
        hap = DenseHaplotypeArray(geno, samples=sm, variants=vm)

        v = np.random.randn(n)
        result = hap.rmatvec(v)

        assert result.shape == (m,)
        expected = hap.diploid_genotypes.T @ v
        np.testing.assert_allclose(result, expected)

    def test_all_zero_genotypes(self):
        """matvec should work with all-zero genotypes."""
        n, m = 10, 5
        geno = np.zeros((n, m, 2), dtype=np.int8)
        sm = SampleMeta(iid=np.arange(n))
        vm = VariantMeta(vid=np.array([f"v{i}" for i in range(m)]))
        hap = DenseHaplotypeArray(geno, samples=sm, variants=vm)

        v = np.random.randn(m)
        result = hap.matvec(v)

        np.testing.assert_array_equal(result, np.zeros(n))

    def test_all_one_genotypes(self):
        """matvec should work with all-one genotypes."""
        n, m = 10, 5
        geno = np.ones((n, m, 2), dtype=np.int8)
        sm = SampleMeta(iid=np.arange(n))
        vm = VariantMeta(vid=np.array([f"v{i}" for i in range(m)]))
        hap = DenseHaplotypeArray(geno, samples=sm, variants=vm)

        v = np.random.randn(m)
        result = hap.matvec(v)

        # Diploid genotypes are all 2, so result should be 2 * sum(v)
        expected = np.full(n, 2 * v.sum())
        np.testing.assert_allclose(result, expected)


class TestConsistency:
    """Test consistency between different methods."""

    def test_matvec_vs_diploid_genotypes(self):
        """matvec should always match direct multiplication with diploid_genotypes."""
        for n, m in [(5, 3), (10, 8), (20, 15), (100, 50)]:
            hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
            v = np.random.randn(m)

            result = hap.matvec(v)
            expected = hap.diploid_genotypes @ v

            np.testing.assert_allclose(
                result, expected,
                err_msg=f"Mismatch for n={n}, m={m}"
            )

    def test_rmatvec_vs_diploid_genotypes_transpose(self):
        """rmatvec should always match G.T @ v."""
        for n, m in [(5, 3), (10, 8), (20, 15), (100, 50)]:
            hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=123)
            v = np.random.randn(n)

            result = hap.rmatvec(v)
            expected = hap.diploid_genotypes.T @ v

            np.testing.assert_allclose(
                result, expected,
                err_msg=f"Mismatch for n={n}, m={m}"
            )

    def test_diploid_matvec_vs_matvec(self):
        """diploid_matvec should always equal matvec."""
        for n, m in [(5, 3), (10, 8), (20, 15)]:
            hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=456)
            v = np.random.randn(m)

            result1 = hap.diploid_matvec(v)
            result2 = hap.matvec(v)

            np.testing.assert_allclose(
                result1, result2,
                err_msg=f"Mismatch for n={n}, m={m}"
            )
