"""
Unit tests for DenseHaplotypeArray computed properties and methods.

Tests:
1. recompute_af returns same as af_empirical
2. af_empirical shape and range
3. to_diploid_standardized with scale=True handles zero-AF loci
4. to_diploid_standardized with custom AF
5. to_dense returns self
6. diploid_genotypes shape
7. matvec_maternal + matvec_paternal = matvec
8. data/values interleaved format
"""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestAFMethods:
    def test_recompute_af_equals_af_empirical(self):
        """recompute_af() should return same as af_empirical property."""
        hap = TestSimulation.founder_haplotypes(n=100, m=20, seed=42)
        np.testing.assert_array_equal(hap.recompute_af(), hap.af_empirical)

    def test_af_empirical_shape(self):
        """af_empirical should have shape (m,)."""
        hap = TestSimulation.founder_haplotypes(n=50, m=15, seed=42)
        af = hap.af_empirical
        assert af.shape == (15,)

    def test_af_empirical_range(self):
        """AF should be in [0, 1]."""
        hap = TestSimulation.founder_haplotypes(n=200, m=30, seed=42)
        af = hap.af_empirical
        assert np.all(af >= 0.0)
        assert np.all(af <= 1.0)


class TestStandardization:
    def test_standardized_centered(self):
        """to_diploid_standardized should center genotypes (mean ~0)."""
        hap = TestSimulation.founder_haplotypes(n=500, m=20, seed=42)
        G = hap.to_diploid_standardized()
        col_means = G.mean(axis=0)
        np.testing.assert_allclose(col_means, 0.0, atol=1e-10)

    def test_standardized_scale_true(self):
        """scale=True should produce unit-variance columns (for non-monomorphic)."""
        hap = TestSimulation.founder_haplotypes(n=500, m=20, seed=42)
        G = hap.to_diploid_standardized(scale=True)
        # For non-monomorphic loci, variance should be approximately 1
        af = hap.af_empirical
        non_mono = (af > 0) & (af < 1)
        variances = G[:, non_mono].var(axis=0)
        np.testing.assert_allclose(variances, 1.0, atol=0.15)

    def test_standardized_zero_af_loci(self):
        """to_diploid_standardized with scale=True should handle all-zero columns."""
        hap = TestSimulation.founder_haplotypes(n=50, m=10, seed=42)
        # Manually set one locus to all zeros (monomorphic)
        hap.genotypes[:, 0, :] = 0
        G = hap.to_diploid_standardized(scale=True)
        # Column 0 should be all zeros (0 - 0 = 0, denom=1 avoids div/0)
        np.testing.assert_array_equal(G[:, 0], 0.0)
        assert np.all(np.isfinite(G))

    def test_standardized_custom_af(self):
        """Providing custom AF should use those for centering."""
        hap = TestSimulation.founder_haplotypes(n=100, m=10, seed=42)
        custom_af = np.full(10, 0.5)
        G = hap.to_diploid_standardized(af=custom_af)
        # Centering should subtract 2*0.5 = 1.0 from diploid
        dip = hap.diploid_genotypes.astype(np.float64)
        expected = dip - 1.0
        np.testing.assert_array_equal(G, expected)


class TestDenseIdentity:
    def test_to_dense_is_self(self):
        """to_dense() should return the same object."""
        hap = TestSimulation.founder_haplotypes(n=20, m=5, seed=42)
        assert hap.to_dense() is hap

    def test_diploid_genotypes_shape(self):
        """diploid_genotypes should be (n, m)."""
        hap = TestSimulation.founder_haplotypes(n=30, m=10, seed=42)
        dip = hap.diploid_genotypes
        assert dip.shape == (30, 10)


class TestMatvecHaploid:
    def test_maternal_plus_paternal_equals_diploid(self):
        """matvec_maternal(v) + matvec_paternal(v) = matvec(v)."""
        hap = TestSimulation.founder_haplotypes(n=50, m=10, seed=42)
        v = np.random.RandomState(99).normal(0, 1, 10)
        result_mat = hap.matvec_maternal(v)
        result_pat = hap.matvec_paternal(v)
        result_dip = hap.matvec(v)
        np.testing.assert_allclose(result_mat + result_pat, result_dip)


class TestInterleavedFormat:
    def test_data_shape(self):
        """data property should be (n, 2*m) interleaved."""
        hap = TestSimulation.founder_haplotypes(n=20, m=8, seed=42)
        assert hap.data.shape == (20, 16)

    def test_values_equals_data(self):
        """values should be alias for data."""
        hap = TestSimulation.founder_haplotypes(n=10, m=5, seed=42)
        np.testing.assert_array_equal(hap.data, hap.values)

    def test_data_interleaving(self):
        """Even columns from hap 0, odd columns from hap 1."""
        hap = TestSimulation.founder_haplotypes(n=10, m=5, seed=42)
        d = hap.data
        np.testing.assert_array_equal(d[:, 0::2], hap.genotypes[:, :, 0])
        np.testing.assert_array_equal(d[:, 1::2], hap.genotypes[:, :, 1])
