"""
Unit tests for RecombinationMap construction and properties.

Tests:
1. constant_map: probabilities, repr
2. From p=float vs p=array
3. Chromosome boundary handling
4. from_haplotypes factory
5. Validation: p out of range, shape mismatch
6. Missing m and vid raises
"""
import numpy as np
import pytest

from xftsim.reproduce import RecombinationMap

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestConstantMap:
    def test_constant_probabilities(self):
        """constant_map should set all probabilities to p."""
        rmap = RecombinationMap.constant_map(m=5, p=0.3)
        assert rmap.m == 5
        np.testing.assert_array_equal(rmap._probabilities, [0.5, 0.3, 0.3, 0.3, 0.3])
        # Note: first position is always 0.5 (chromosome boundary)

    def test_constant_map_default_p(self):
        """Default p=0.5 should set all to 0.5."""
        rmap = RecombinationMap.constant_map(m=3)
        np.testing.assert_array_equal(rmap._probabilities, [0.5, 0.5, 0.5])

    def test_repr(self):
        rmap = RecombinationMap.constant_map(m=3, p=0.5)
        r = repr(rmap)
        assert '0.5' in r


class TestConstructor:
    def test_float_p(self):
        """Float p should create constant map."""
        rmap = RecombinationMap(p=0.1, m=4)
        assert rmap.m == 4
        # First position always 0.5 (boundary), rest 0.1
        assert rmap._probabilities[0] == 0.5
        np.testing.assert_allclose(rmap._probabilities[1:], 0.1)

    def test_array_p(self):
        """Array p should be used directly (with boundary override)."""
        p = np.array([0.1, 0.2, 0.3, 0.4])
        rmap = RecombinationMap(p=p, m=4)
        assert rmap._probabilities[0] == 0.5  # boundary
        np.testing.assert_allclose(rmap._probabilities[1:], [0.2, 0.3, 0.4])

    def test_default_p(self):
        """No p → all 0.5."""
        rmap = RecombinationMap(m=5)
        np.testing.assert_array_equal(rmap._probabilities, 0.5)

    def test_vid_provides_m(self):
        """m can be inferred from vid."""
        rmap = RecombinationMap(vid=np.array([10, 20, 30]))
        assert rmap.m == 3

    def test_missing_m_and_vid_raises(self):
        """Neither m nor vid should raise."""
        with pytest.raises(ValueError, match="m or vid"):
            RecombinationMap(p=0.5)

    def test_array_p_wrong_length_raises(self):
        """Array p with wrong length should raise."""
        with pytest.raises(AssertionError):
            RecombinationMap(p=np.array([0.1, 0.2]), m=5)


class TestChromosomeBoundary:
    def test_single_chrom(self):
        """Single chromosome: only first position is boundary."""
        rmap = RecombinationMap(p=0.1, m=5, chrom=np.array([1, 1, 1, 1, 1]))
        assert rmap._probabilities[0] == 0.5
        np.testing.assert_allclose(rmap._probabilities[1:], 0.1)

    def test_two_chroms(self):
        """Two chromosomes: boundary at start of each."""
        chrom = np.array([1, 1, 1, 2, 2])
        rmap = RecombinationMap(p=0.1, m=5, chrom=chrom)
        # Boundaries at positions 0 (start) and 3 (chrom switch)
        assert rmap._probabilities[0] == 0.5
        assert rmap._probabilities[3] == 0.5
        np.testing.assert_allclose(rmap._probabilities[[1, 2, 4]], 0.1)

    def test_three_chroms(self):
        """Three chromosomes."""
        chrom = np.array([1, 1, 2, 2, 3])
        rmap = RecombinationMap(p=0.2, m=5, chrom=chrom)
        # Boundaries at 0, 2, 4
        for i in [0, 2, 4]:
            assert rmap._probabilities[i] == 0.5


class TestFromHaplotypes:
    def test_from_haplotypes(self):
        """from_haplotypes should create map with correct m."""
        hap = TestSimulation.founder_haplotypes(n=10, m=5, seed=42)
        rmap = RecombinationMap.from_haplotypes(hap, p=0.3)
        assert rmap.m == 5

    def test_from_haplotypes_vid_propagation(self):
        """VIDs should come from the haplotype array."""
        hap = TestSimulation.founder_haplotypes(n=10, m=5, seed=42)
        rmap = RecombinationMap.from_haplotypes(hap)
        np.testing.assert_array_equal(rmap.vid, hap.vid)
