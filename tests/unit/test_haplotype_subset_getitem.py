"""
Unit tests for DenseHaplotypeArray subset and __getitem__.

Tests:
1. __getitem__ with integer array
2. __getitem__ with tuple (samples, variants)
3. __getitem__ with slice
4. Too many indices raises IndexError
5. subset preserves metadata
6. drop_isel drops specified samples
7. Binary values preserved after subset
"""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestGetItem:
    def test_integer_array_index(self):
        """hap[[0, 2, 4]] should select 3 samples."""
        hap = TestSimulation.founder_haplotypes(n=10, m=5, seed=42)
        sub = hap[[0, 2, 4]]
        assert sub.n == 3
        assert sub.m == 5

    def test_tuple_samples_variants(self):
        """hap[sample_idx, variant_idx] should subset both dims."""
        hap = TestSimulation.founder_haplotypes(n=10, m=8, seed=42)
        sub = hap[np.array([0, 1, 2]), np.array([3, 4, 5])]
        assert sub.n == 3
        assert sub.m == 3

    def test_slice_index(self):
        """hap[:5] should select first 5 samples."""
        hap = TestSimulation.founder_haplotypes(n=10, m=5, seed=42)
        sub = hap[:5]
        assert sub.n == 5
        assert sub.m == 5

    def test_too_many_indices_raises(self):
        """hap[a, b, c] should raise IndexError."""
        hap = TestSimulation.founder_haplotypes(n=10, m=5, seed=42)
        with pytest.raises(IndexError, match="Too many indices"):
            hap[np.array([0]), np.array([0]), np.array([0])]


class TestSubsetMetadata:
    def test_subset_preserves_iid(self):
        """Subset should produce correct IIDs."""
        hap = TestSimulation.founder_haplotypes(n=10, m=5, seed=42)
        idx = np.array([1, 3, 5])
        sub = hap[idx]
        expected_iid = hap.samples.iid[idx]
        np.testing.assert_array_equal(sub.samples.iid, expected_iid)

    def test_subset_preserves_vid(self):
        """Variant subset should produce correct VIDs."""
        hap = TestSimulation.founder_haplotypes(n=10, m=8, seed=42)
        vidx = np.array([0, 4, 7])
        sub = hap[:, vidx]
        expected_vid = hap.variants.vid[vidx]
        np.testing.assert_array_equal(sub.variants.vid, expected_vid)

    def test_subset_binary_values(self):
        """Genotypes should remain 0/1 after subset."""
        hap = TestSimulation.founder_haplotypes(n=20, m=10, seed=42)
        sub = hap[[0, 5, 10]]
        assert set(np.unique(sub.genotypes)).issubset({0, 1})


class TestDropIsel:
    def test_drop_samples(self):
        """drop_isel(sample=...) should remove specified samples."""
        hap = TestSimulation.founder_haplotypes(n=10, m=5, seed=42)
        dropped = hap.drop_isel(sample=np.array([0, 1, 2]))
        assert dropped.n == 7
        assert dropped.m == 5

    def test_drop_variants(self):
        """drop_isel(variant=...) should remove specified variants."""
        hap = TestSimulation.founder_haplotypes(n=10, m=8, seed=42)
        dropped = hap.drop_isel(variant=np.array([2, 5]))
        assert dropped.n == 10
        assert dropped.m == 6
