"""
Unit tests for genotypes_to_pseudo_haplotypes conversion (io.py).

Tests:
1. Homozygous ref (0) → both haplotypes 0
2. Homozygous alt (2) → both haplotypes 1
3. Heterozygous (1) → one 0 and one 1 (sum == 1)
4. Shape preserved: (n, m) → (n, m, 2)
5. dtype is int8
6. Diploid sum matches original genotypes
7. Empty input
8. Single sample, single variant
"""
import numpy as np
import pytest

from xftsim.io import genotypes_to_pseudo_haplotypes


class TestGenotypesToPseudoHaplotypes:
    def test_homozygous_ref(self):
        """Genotype 0 → both haplotypes 0."""
        g = np.array([[0, 0, 0]], dtype=np.int8)
        h = genotypes_to_pseudo_haplotypes(g)
        assert h.shape == (1, 3, 2)
        np.testing.assert_array_equal(h[0, :, 0], [0, 0, 0])
        np.testing.assert_array_equal(h[0, :, 1], [0, 0, 0])

    def test_homozygous_alt(self):
        """Genotype 2 → both haplotypes 1."""
        g = np.array([[2, 2]], dtype=np.int8)
        h = genotypes_to_pseudo_haplotypes(g)
        np.testing.assert_array_equal(h[0, :, 0], [1, 1])
        np.testing.assert_array_equal(h[0, :, 1], [1, 1])

    def test_heterozygous_sum(self):
        """Genotype 1 → haplotypes sum to 1 per site."""
        g = np.array([[1, 1, 1, 1, 1]], dtype=np.int8)
        h = genotypes_to_pseudo_haplotypes(g)
        sums = h[0, :, 0] + h[0, :, 1]
        np.testing.assert_array_equal(sums, [1, 1, 1, 1, 1])

    def test_shape(self):
        """(n, m) → (n, m, 2)."""
        g = np.zeros((10, 5), dtype=np.int8)
        h = genotypes_to_pseudo_haplotypes(g)
        assert h.shape == (10, 5, 2)

    def test_dtype(self):
        """Output should be int8."""
        g = np.array([[0, 1, 2]], dtype=np.int8)
        h = genotypes_to_pseudo_haplotypes(g)
        assert h.dtype == np.int8

    def test_diploid_sum_matches(self):
        """Sum of haplotypes should equal original genotypes."""
        rng = np.random.RandomState(42)
        g = rng.choice([0, 1, 2], size=(20, 10)).astype(np.int8)
        h = genotypes_to_pseudo_haplotypes(g)
        diploid = h[:, :, 0] + h[:, :, 1]
        np.testing.assert_array_equal(diploid, g)

    def test_single_sample_single_variant(self):
        """1x1 input."""
        for val in [0, 1, 2]:
            g = np.array([[val]], dtype=np.int8)
            h = genotypes_to_pseudo_haplotypes(g)
            assert h.shape == (1, 1, 2)
            assert h[0, 0, 0] + h[0, 0, 1] == val

    def test_values_binary(self):
        """All output values should be 0 or 1."""
        rng = np.random.RandomState(42)
        g = rng.choice([0, 1, 2], size=(50, 30)).astype(np.int8)
        h = genotypes_to_pseudo_haplotypes(g)
        assert np.all((h == 0) | (h == 1))

    def test_large_input(self):
        """Larger input should work without error."""
        g = np.ones((100, 200), dtype=np.int8)  # all het
        h = genotypes_to_pseudo_haplotypes(g)
        assert h.shape == (100, 200, 2)
        # All hets: each site sums to 1
        np.testing.assert_array_equal(h[:, :, 0] + h[:, :, 1], g)

    def test_float_input_coerced(self):
        """Float input should be coerced to int8."""
        g = np.array([[0.0, 1.0, 2.0]])
        h = genotypes_to_pseudo_haplotypes(g)
        assert h.dtype == np.int8
        assert h[0, 0, 0] + h[0, 0, 1] == 0
        assert h[0, 1, 0] + h[0, 1, 1] == 1
        assert h[0, 2, 0] + h[0, 2, 1] == 2
