"""
Unit tests for RecombinationMap constructor edge cases.

Tests:
1. No m or vid raises ValueError
2. p as array with wrong length raises AssertionError
3. p=0.0 boundary — no recombination
4. p=1.0 boundary — always recombine
5. Multi-chromosome forces 0.5 at boundaries
6. vid-based construction infers m
7. Default p (None) uses 0.5
"""
import numpy as np
import pytest

from xftsim.reproduce import RecombinationMap


class TestRecombinationMapEdges:
    def test_no_m_no_vid_raises(self):
        """Must provide m or vid."""
        with pytest.raises(ValueError, match="Must provide m or vid"):
            RecombinationMap()

    def test_array_p_wrong_length_raises(self):
        """Array p with wrong length should raise."""
        with pytest.raises(AssertionError):
            RecombinationMap(p=np.array([0.1, 0.2, 0.3]), m=5)

    def test_p_zero(self):
        """p=0.0 should be valid (no recombination except at boundary)."""
        rmap = RecombinationMap.constant_map(m=10, p=0.0)
        # First position is a chromosome boundary → forced to 0.5
        assert rmap._probabilities[0] == 0.5
        # All other positions should be 0.0
        assert np.all(rmap._probabilities[1:] == 0.0)

    def test_p_one(self):
        """p=1.0 should be valid (always recombine, except boundary at 0.5)."""
        rmap = RecombinationMap.constant_map(m=10, p=1.0)
        # First position is chromosome boundary → forced to 0.5
        assert rmap._probabilities[0] == 0.5
        assert np.all(rmap._probabilities[1:] == 1.0)

    def test_multi_chromosome_boundaries(self):
        """Chromosome boundaries should always have p=0.5."""
        chrom = np.array([1, 1, 1, 2, 2, 3, 3, 3])
        rmap = RecombinationMap(p=0.01, m=8, chrom=chrom)
        # Boundary loci: index 0 (start of chrom 1), 3 (start of chrom 2), 5 (start of chrom 3)
        assert rmap._probabilities[0] == 0.5
        assert rmap._probabilities[3] == 0.5
        assert rmap._probabilities[5] == 0.5
        # Non-boundary loci should be 0.01
        assert rmap._probabilities[1] == 0.01
        assert rmap._probabilities[2] == 0.01
        assert rmap._probabilities[4] == 0.01

    def test_vid_infers_m(self):
        """Providing vid without m should infer m from vid length."""
        vid = np.array(['rs1', 'rs2', 'rs3', 'rs4'])
        rmap = RecombinationMap(p=0.1, vid=vid)
        assert rmap.m == 4

    def test_default_p_is_half(self):
        """No p argument should default to 0.5 everywhere."""
        rmap = RecombinationMap(m=5)
        np.testing.assert_array_equal(rmap._probabilities, 0.5)

    def test_array_p_exact(self):
        """Array p should be used exactly (except at boundaries)."""
        p = np.array([0.5, 0.1, 0.2, 0.3, 0.4])
        rmap = RecombinationMap(p=p, m=5)
        # First element is chromosome boundary → 0.5
        assert rmap._probabilities[0] == 0.5
        np.testing.assert_array_almost_equal(rmap._probabilities[1:], [0.1, 0.2, 0.3, 0.4])
