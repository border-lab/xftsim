"""
Tests for allele frequency computation edge cases in DenseHaplotypeArray.

This module tests the af_empirical property and to_diploid_standardized method,
including:
- Edge cases (all-zero, all-one genotypes)
- AF shape and range validation
- Single individual/variant cases
- Standardization with known AF values
- Division by zero protection (p=0 or p=1 with scale=True)
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from testdata import TestSimulation

from xftsim.struct import DenseHaplotypeArray, SampleMeta, VariantMeta


class TestAFEmpirical:
    """Test allele frequency computation from genotype data."""

    def test_af_all_zero_genotypes(self):
        """AF of all-zero genotypes should be all 0.0."""
        n, m = 10, 5
        geno = np.zeros((n, m, 2), dtype=np.int8)
        sm = SampleMeta(iid=np.arange(n))
        vm = VariantMeta(vid=np.array([f"v{i}" for i in range(m)]))
        hap = DenseHaplotypeArray(geno, samples=sm, variants=vm)

        af = hap.af_empirical
        assert af.shape == (m,)
        np.testing.assert_array_equal(af, np.zeros(m))

    def test_af_all_one_genotypes(self):
        """AF of all-one genotypes should be all 1.0."""
        n, m = 10, 5
        geno = np.ones((n, m, 2), dtype=np.int8)
        sm = SampleMeta(iid=np.arange(n))
        vm = VariantMeta(vid=np.array([f"v{i}" for i in range(m)]))
        hap = DenseHaplotypeArray(geno, samples=sm, variants=vm)

        af = hap.af_empirical
        assert af.shape == (m,)
        np.testing.assert_array_equal(af, np.ones(m))

    def test_af_mixed_genotypes(self):
        """AF of mixed genotypes should match expected values."""
        # Create known genotype pattern
        # Individual 0: [0,0] [1,1] [0,1]
        # Individual 1: [1,1] [0,0] [0,1]
        # Individual 2: [0,1] [0,1] [1,1]
        # Expected AF: [0.5, 0.5, 2/3]
        n, m = 3, 3
        geno = np.array(
            [
                [[0, 0], [1, 1], [0, 1]],  # individual 0
                [[1, 1], [0, 0], [0, 1]],  # individual 1
                [[0, 1], [0, 1], [1, 1]],  # individual 2
            ],
            dtype=np.int8,
        )
        sm = SampleMeta(iid=np.arange(n))
        vm = VariantMeta(vid=np.array([f"v{i}" for i in range(m)]))
        hap = DenseHaplotypeArray(geno, samples=sm, variants=vm)

        af = hap.af_empirical
        expected = np.array([0.5, 0.5, 2.0 / 3.0])
        np.testing.assert_allclose(af, expected)

    def test_af_shape(self):
        """AF shape should be (m,) for any input."""
        for n, m in [(1, 1), (5, 10), (100, 50), (10, 1)]:
            geno = np.random.randint(0, 2, size=(n, m, 2), dtype=np.int8)
            sm = SampleMeta(iid=np.arange(n))
            vm = VariantMeta(vid=np.array([f"v{i}" for i in range(m)]))
            hap = DenseHaplotypeArray(geno, samples=sm, variants=vm)

            af = hap.af_empirical
            assert af.shape == (m,), f"Expected shape ({m},), got {af.shape} for n={n}, m={m}"

    def test_af_range(self):
        """AF should always be in [0, 1]."""
        n, m = 20, 15
        geno = np.random.randint(0, 2, size=(n, m, 2), dtype=np.int8)
        sm = SampleMeta(iid=np.arange(n))
        vm = VariantMeta(vid=np.array([f"v{i}" for i in range(m)]))
        hap = DenseHaplotypeArray(geno, samples=sm, variants=vm)

        af = hap.af_empirical
        assert np.all(af >= 0.0), "Some AFs are negative"
        assert np.all(af <= 1.0), "Some AFs are greater than 1"

    def test_af_single_individual(self):
        """AF computation with n=1 should work correctly."""
        n, m = 1, 5
        # Single individual with genotypes: [0,1], [1,1], [0,0], [1,0], [0,1]
        geno = np.array([[[0, 1], [1, 1], [0, 0], [1, 0], [0, 1]]], dtype=np.int8)
        sm = SampleMeta(iid=np.arange(n))
        vm = VariantMeta(vid=np.array([f"v{i}" for i in range(m)]))
        hap = DenseHaplotypeArray(geno, samples=sm, variants=vm)

        af = hap.af_empirical
        expected = np.array([0.5, 1.0, 0.0, 0.5, 0.5])
        np.testing.assert_array_equal(af, expected)

    def test_af_single_variant(self):
        """AF computation with m=1 should work correctly."""
        n, m = 5, 1
        # Five individuals with one variant: [0,1], [1,1], [0,0], [1,0], [1,1]
        geno = np.array(
            [[[0, 1]], [[1, 1]], [[0, 0]], [[1, 0]], [[1, 1]]], dtype=np.int8
        )
        sm = SampleMeta(iid=np.arange(n))
        vm = VariantMeta(vid=np.array([f"v{i}" for i in range(m)]))
        hap = DenseHaplotypeArray(geno, samples=sm, variants=vm)

        af = hap.af_empirical
        # Total alleles: 1 + 2 + 0 + 1 + 2 = 6 out of 10
        expected = np.array([0.6])
        np.testing.assert_array_equal(af, expected)

    def test_recompute_af(self):
        """recompute_af() should return the same as af_empirical."""
        n, m = 10, 8
        geno = np.random.randint(0, 2, size=(n, m, 2), dtype=np.int8)
        sm = SampleMeta(iid=np.arange(n))
        vm = VariantMeta(vid=np.array([f"v{i}" for i in range(m)]))
        hap = DenseHaplotypeArray(geno, samples=sm, variants=vm)

        af_direct = hap.af_empirical
        af_recompute = hap.recompute_af()
        np.testing.assert_array_equal(af_direct, af_recompute)


class TestDiploidStandardized:
    """Test to_diploid_standardized and standardized_matvec methods."""

    def test_standardized_centering_only(self):
        """to_diploid_standardized with scale=False should only center."""
        n, m = 5, 3
        # Known genotypes
        geno = np.array(
            [
                [[0, 0], [1, 1], [0, 1]],
                [[1, 1], [0, 0], [0, 1]],
                [[0, 1], [0, 1], [1, 1]],
                [[1, 0], [1, 0], [0, 0]],
                [[0, 0], [1, 1], [1, 0]],
            ],
            dtype=np.int8,
        )
        sm = SampleMeta(iid=np.arange(n))
        vm = VariantMeta(vid=np.array([f"v{i}" for i in range(m)]))
        hap = DenseHaplotypeArray(geno, samples=sm, variants=vm)

        af = hap.af_empirical
        G = hap.diploid_genotypes.astype(np.float64)
        G_std = hap.to_diploid_standardized(scale=False)

        # Check that it's centered: G - 2*af
        expected = G - 2 * af
        np.testing.assert_allclose(G_std, expected)

    def test_standardized_with_scaling(self):
        """to_diploid_standardized with scale=True should center and scale."""
        n, m = 5, 3
        geno = np.array(
            [
                [[0, 0], [1, 1], [0, 1]],
                [[1, 1], [0, 0], [0, 1]],
                [[0, 1], [0, 1], [1, 1]],
                [[1, 0], [1, 0], [0, 0]],
                [[0, 0], [1, 1], [1, 0]],
            ],
            dtype=np.int8,
        )
        sm = SampleMeta(iid=np.arange(n))
        vm = VariantMeta(vid=np.array([f"v{i}" for i in range(m)]))
        hap = DenseHaplotypeArray(geno, samples=sm, variants=vm)

        af = hap.af_empirical
        G = hap.diploid_genotypes.astype(np.float64)
        G_std = hap.to_diploid_standardized(scale=True)

        # Manual computation
        G_centered = G - 2 * af
        denom = np.sqrt(2 * af * (1 - af))
        denom[denom == 0] = 1.0  # protect from divide by zero
        expected = G_centered / denom

        np.testing.assert_allclose(G_std, expected)

    def test_standardized_with_custom_af(self):
        """to_diploid_standardized should use custom AF when provided."""
        n, m = 5, 3
        geno = np.random.randint(0, 2, size=(n, m, 2), dtype=np.int8)
        sm = SampleMeta(iid=np.arange(n))
        vm = VariantMeta(vid=np.array([f"v{i}" for i in range(m)]))
        hap = DenseHaplotypeArray(geno, samples=sm, variants=vm)

        # Use custom AF different from empirical
        custom_af = np.array([0.3, 0.5, 0.7])
        G = hap.diploid_genotypes.astype(np.float64)
        G_std = hap.to_diploid_standardized(af=custom_af, scale=False)

        expected = G - 2 * custom_af
        np.testing.assert_allclose(G_std, expected)

    def test_standardized_scale_division_by_zero_protection(self):
        """to_diploid_standardized with scale=True should protect against p=0 or p=1."""
        n, m = 3, 3
        # First variant: all 0 (p=0), second: all 1 (p=1), third: mixed (p=0.5)
        geno = np.array(
            [
                [[0, 0], [1, 1], [0, 1]],
                [[0, 0], [1, 1], [1, 0]],
                [[0, 0], [1, 1], [0, 1]],
            ],
            dtype=np.int8,
        )
        sm = SampleMeta(iid=np.arange(n))
        vm = VariantMeta(vid=np.array([f"v{i}" for i in range(m)]))
        hap = DenseHaplotypeArray(geno, samples=sm, variants=vm)

        G_std = hap.to_diploid_standardized(scale=True)

        # Should not raise an error
        assert not np.any(np.isnan(G_std)), "NaN values found in standardized genotypes"
        assert not np.any(np.isinf(G_std)), "Inf values found in standardized genotypes"

        # For p=0 and p=1, denom is set to 1, so G_std should just be centered
        af = hap.af_empirical
        G = hap.diploid_genotypes.astype(np.float64)

        # Variant 0: p=0, denom=1, so G_std[:,0] = G[:,0] - 0
        np.testing.assert_allclose(G_std[:, 0], G[:, 0] - 0)

        # Variant 1: p=1, denom=1, so G_std[:,1] = G[:,1] - 2
        np.testing.assert_allclose(G_std[:, 1], G[:, 1] - 2)

    def test_standardized_matvec_matches_manual(self):
        """standardized_matvec should match manual centering."""
        n, m = 10, 8
        sim = TestSimulation()
        hap = sim.founder_haplotypes(n=n, m=m, seed=42)

        v = np.random.randn(m)
        result = hap.standardized_matvec(v)

        # Manual computation: ((G - 2*af) / sqrt(2pq)) @ v
        af = hap.af_empirical
        G = hap.diploid_genotypes.astype(np.float64)
        denom = np.sqrt(2 * af * (1 - af))
        denom[denom == 0] = 1.0
        expected = ((G - 2 * af) / denom) @ v

        np.testing.assert_allclose(result, expected)

    def test_standardized_matvec_with_custom_af(self):
        """standardized_matvec with custom AF should use that AF."""
        n, m = 10, 8
        sim = TestSimulation()
        hap = sim.founder_haplotypes(n=n, m=m, seed=42)

        custom_af = np.full(m, 0.4)
        v = np.random.randn(m)
        result = hap.standardized_matvec(v, af=custom_af)

        # Manual computation with custom AF (per-SNP standardized)
        G = hap.diploid_genotypes.astype(np.float64)
        denom = np.sqrt(2 * custom_af * (1 - custom_af))
        denom[denom == 0] = 1.0
        expected = ((G - 2 * custom_af) / denom) @ v

        np.testing.assert_allclose(result, expected)

    def test_standardized_matvec_2d_vector(self):
        """standardized_matvec should work with 2D vectors."""
        n, m = 10, 8
        sim = TestSimulation()
        hap = sim.founder_haplotypes(n=n, m=m, seed=42)

        k = 3
        V = np.random.randn(m, k)
        result = hap.standardized_matvec(V)

        assert result.shape == (n, k)

        # Check each column independently
        for i in range(k):
            expected_i = hap.standardized_matvec(V[:, i])
            np.testing.assert_allclose(result[:, i], expected_i)


class TestAFWithTestSimulation:
    """Test AF computation using TestSimulation-generated data."""

    def test_af_from_founder_haplotypes(self):
        """AF from founder_haplotypes should be reasonable."""
        n, m = 50, 20
        sim = TestSimulation()
        hap = sim.founder_haplotypes(n=n, m=m, seed=123)

        af = hap.af_empirical
        assert af.shape == (m,)
        assert np.all(af >= 0.0)
        assert np.all(af <= 1.0)

    def test_af_consistency_across_calls(self):
        """Multiple calls to af_empirical should return same result."""
        n, m = 20, 15
        sim = TestSimulation()
        hap = sim.founder_haplotypes(n=n, m=m, seed=456)

        af1 = hap.af_empirical
        af2 = hap.af_empirical
        af3 = hap.recompute_af()

        np.testing.assert_array_equal(af1, af2)
        np.testing.assert_array_equal(af1, af3)

    def test_diploid_genotypes_match_af_computation(self):
        """Diploid genotypes should match AF computation."""
        n, m = 15, 10
        sim = TestSimulation()
        hap = sim.founder_haplotypes(n=n, m=m, seed=789)

        af = hap.af_empirical
        G = hap.diploid_genotypes

        # Manual AF computation from diploid genotypes
        manual_af = G.mean(axis=0) / 2.0

        np.testing.assert_allclose(af, manual_af)
