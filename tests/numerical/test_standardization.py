"""
Numerical tests for standardization and matvec correctness.

Tests:
1. standardized_haploid_matvec: output is centered (~0 mean)
2. standardized_matvec (diploid): centering correct
3. to_diploid_standardized: centered and optionally scaled
4. diploid_matvec: equals (hap0 + hap1) @ u
5. matvec_maternal + matvec_paternal == diploid_matvec
6. rmatvec: G.T @ v identity
7. AF computation: empirical AF in [0, 1]
8. Standardized matvec with custom AF
"""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestStandardizedHaploidMatvec:
    def test_centered_output(self):
        """Standardized haploid matvec should produce ~0 mean output."""
        hap = TestSimulation.founder_haplotypes(n=500, m=50, seed=42)
        u = np.ones(50)
        result = hap.standardized_haploid_matvec(u, haploid=0)
        assert abs(np.mean(result)) < 0.5, f"Mean {np.mean(result)} not near 0"

    def test_haploid0_vs_haploid1(self):
        """Two haploid matvecs should generally differ."""
        hap = TestSimulation.founder_haplotypes(n=100, m=20, seed=42)
        u = np.ones(20)
        r0 = hap.standardized_haploid_matvec(u, haploid=0)
        r1 = hap.standardized_haploid_matvec(u, haploid=1)
        assert not np.allclose(r0, r1)

    def test_zero_effect_vector(self):
        """Zero effect vector should produce all zeros."""
        hap = TestSimulation.founder_haplotypes(n=50, m=10, seed=42)
        result = hap.standardized_haploid_matvec(np.zeros(10), haploid=0)
        np.testing.assert_array_equal(result, np.zeros(50))

    def test_single_variant_no_crash(self):
        """m=1 should work without crashing."""
        hap = TestSimulation.founder_haplotypes(n=20, m=1, seed=42)
        result = hap.standardized_haploid_matvec(np.array([1.0]), haploid=0)
        assert result.shape == (20,)


class TestDiploidStandardization:
    def test_to_diploid_standardized_centered(self):
        """Centered diploid genotypes should have ~0 column means."""
        hap = TestSimulation.founder_haplotypes(n=500, m=20, seed=42)
        G = hap.to_diploid_standardized()
        col_means = G.mean(axis=0)
        assert np.all(np.abs(col_means) < 0.1), f"Column means not near 0: max={np.max(np.abs(col_means))}"

    def test_to_diploid_standardized_scaled(self):
        """Scaled diploid genotypes should have ~1 column std (non-fixed variants)."""
        hap = TestSimulation.founder_haplotypes(n=500, m=20, seed=42)
        G = hap.to_diploid_standardized(scale=True)
        col_std = G.std(axis=0, ddof=1)
        # Filter out fixed variants (std would be 0 → 1 after protection)
        af = hap.af_empirical
        polymorphic = (af > 0.01) & (af < 0.99)
        if np.any(polymorphic):
            poly_std = col_std[polymorphic]
            assert np.all(np.abs(poly_std - 1.0) < 0.3), f"Scaled std not near 1: {poly_std}"

    def test_custom_af_centering(self):
        """Custom AF should change the centering point."""
        hap = TestSimulation.founder_haplotypes(n=100, m=10, seed=42)
        custom_af = np.full(10, 0.5)
        G = hap.to_diploid_standardized(af=custom_af)
        # Centering point is 2*0.5=1, so G = diploid - 1.0
        raw = hap.diploid_genotypes.astype(float)
        expected = raw - 1.0
        np.testing.assert_allclose(G, expected)


class TestDiploidMatvec:
    def test_equals_hap0_plus_hap1(self):
        """diploid_matvec(u) = hap[:,:,0]@u + hap[:,:,1]@u."""
        hap = TestSimulation.founder_haplotypes(n=50, m=20, seed=42)
        u = np.random.RandomState(42).normal(size=20)
        result = hap.diploid_matvec(u)
        expected = hap.genotypes[:, :, 0] @ u + hap.genotypes[:, :, 1] @ u
        np.testing.assert_allclose(result, expected)

    def test_maternal_plus_paternal_equals_diploid(self):
        """matvec_maternal(u) + matvec_paternal(u) == diploid_matvec(u)."""
        hap = TestSimulation.founder_haplotypes(n=50, m=20, seed=42)
        u = np.random.RandomState(42).normal(size=20)
        mat = hap.matvec_maternal(u)
        pat = hap.matvec_paternal(u)
        dip = hap.diploid_matvec(u)
        np.testing.assert_allclose(mat + pat, dip)

    def test_zero_effects(self):
        """Zero effects should produce zero genetic values."""
        hap = TestSimulation.founder_haplotypes(n=30, m=10, seed=42)
        result = hap.diploid_matvec(np.zeros(10))
        np.testing.assert_array_equal(result, np.zeros(30))


class TestRmatvec:
    def test_rmatvec_transpose_identity(self):
        """rmatvec(v) should equal G.T @ v where G = sum of haplotypes."""
        hap = TestSimulation.founder_haplotypes(n=50, m=20, seed=42)
        v = np.random.RandomState(42).normal(size=50)
        # rmatvec should implement G.T @ v (diploid)
        result = hap.rmatvec(v)
        G = hap.diploid_genotypes.astype(float)
        expected = G.T @ v
        np.testing.assert_allclose(result, expected)

    def test_rmatvec_shape(self):
        hap = TestSimulation.founder_haplotypes(n=50, m=20, seed=42)
        v = np.ones(50)
        result = hap.rmatvec(v)
        assert result.shape == (20,)


class TestAFComputation:
    def test_af_in_range(self):
        """Allele frequencies should be in [0, 1]."""
        hap = TestSimulation.founder_haplotypes(n=100, m=50, seed=42)
        af = hap.af_empirical
        assert np.all(af >= 0)
        assert np.all(af <= 1)

    def test_af_shape(self):
        hap = TestSimulation.founder_haplotypes(n=100, m=50, seed=42)
        af = hap.af_empirical
        assert af.shape == (50,)

    def test_af_equals_haplotype_mean(self):
        """AF should equal mean of (hap0 + hap1) / 2 across samples."""
        hap = TestSimulation.founder_haplotypes(n=100, m=20, seed=42)
        af = hap.af_empirical
        manual_af = (hap.genotypes[:, :, 0].mean(axis=0) +
                     hap.genotypes[:, :, 1].mean(axis=0)) / 2
        np.testing.assert_allclose(af, manual_af)

    def test_recompute_af_same(self):
        """recompute_af() should return same as af_empirical."""
        hap = TestSimulation.founder_haplotypes(n=100, m=20, seed=42)
        np.testing.assert_array_equal(hap.recompute_af(), hap.af_empirical)


class TestStandardizedMatvecWithCustomAF:
    def test_standardized_matvec_uses_custom_af(self):
        """standardized_matvec with custom AF should center differently."""
        hap = TestSimulation.founder_haplotypes(n=100, m=20, seed=42)
        u = np.ones(20)
        af1 = hap.af_empirical
        af2 = np.full(20, 0.5)
        r1 = hap.standardized_matvec(u, af=af1)
        r2 = hap.standardized_matvec(u, af=af2)
        # Different AFs should produce different results
        assert not np.allclose(r1, r2)
