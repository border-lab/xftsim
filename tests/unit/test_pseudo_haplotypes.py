"""
Unit tests for genotypes_to_pseudo_haplotypes.

Tests:
1. Homozygous 0 → (0, 0)
2. Homozygous 2 → (1, 1)
3. Heterozygous 1 → one of (0,1) or (1,0)
4. Output shape is (n, m, 2)
5. Output is binary (0 or 1)
6. Diploid sum preserved
7. Single variant
8. Single individual
9. All heterozygous
10. All homozygous
"""
import numpy as np
import pytest

from xftsim.io import genotypes_to_pseudo_haplotypes


class TestHomozygous:
    def test_hom_ref(self):
        """Genotype 0 → both haplotypes 0."""
        g = np.array([[0]], dtype=np.int8)
        h = genotypes_to_pseudo_haplotypes(g)
        assert h[0, 0, 0] == 0
        assert h[0, 0, 1] == 0

    def test_hom_alt(self):
        """Genotype 2 → both haplotypes 1."""
        g = np.array([[2]], dtype=np.int8)
        h = genotypes_to_pseudo_haplotypes(g)
        assert h[0, 0, 0] == 1
        assert h[0, 0, 1] == 1


class TestHeterozygous:
    def test_het_sum(self):
        """Genotype 1 → hap[0] + hap[1] = 1."""
        g = np.array([[1]], dtype=np.int8)
        h = genotypes_to_pseudo_haplotypes(g)
        assert h[0, 0, 0] + h[0, 0, 1] == 1


class TestOutputProperties:
    def test_shape(self):
        g = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int8)
        h = genotypes_to_pseudo_haplotypes(g)
        assert h.shape == (2, 3, 2)

    def test_binary_values(self):
        """Output values should be 0 or 1."""
        rng = np.random.RandomState(42)
        g = rng.randint(0, 3, size=(20, 10)).astype(np.int8)
        h = genotypes_to_pseudo_haplotypes(g)
        assert np.all((h == 0) | (h == 1))

    def test_diploid_sum_preserved(self):
        """Sum across haplotypes should equal original genotypes."""
        rng = np.random.RandomState(42)
        g = rng.randint(0, 3, size=(20, 10)).astype(np.int8)
        h = genotypes_to_pseudo_haplotypes(g)
        diploid = h[:, :, 0] + h[:, :, 1]
        np.testing.assert_array_equal(diploid, g)


class TestEdgeCases:
    def test_single_variant(self):
        g = np.array([[0], [1], [2]], dtype=np.int8)
        h = genotypes_to_pseudo_haplotypes(g)
        assert h.shape == (3, 1, 2)

    def test_single_individual(self):
        g = np.array([[0, 1, 2]], dtype=np.int8)
        h = genotypes_to_pseudo_haplotypes(g)
        assert h.shape == (1, 3, 2)

    def test_all_heterozygous(self):
        """All genotypes = 1."""
        g = np.ones((5, 5), dtype=np.int8)
        h = genotypes_to_pseudo_haplotypes(g)
        diploid = h[:, :, 0] + h[:, :, 1]
        np.testing.assert_array_equal(diploid, np.ones((5, 5)))

    def test_all_homozygous_ref(self):
        g = np.zeros((5, 5), dtype=np.int8)
        h = genotypes_to_pseudo_haplotypes(g)
        np.testing.assert_array_equal(h, np.zeros((5, 5, 2), dtype=np.int8))

    def test_all_homozygous_alt(self):
        g = np.full((5, 5), 2, dtype=np.int8)
        h = genotypes_to_pseudo_haplotypes(g)
        np.testing.assert_array_equal(h, np.ones((5, 5, 2), dtype=np.int8))
