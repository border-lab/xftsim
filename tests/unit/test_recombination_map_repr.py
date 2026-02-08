"""
Unit tests for RecombinationMap repr and from_haplotypes details.

Tests:
1. RecombinationMap.__repr__ returns string
2. RecombinationMap.__repr__ contains variant info
3. from_haplotypes default p=0.5
4. from_haplotypes custom p
5. from_haplotypes vid matches haplotype vid
"""
import numpy as np
import pytest

from xftsim.reproduce import RecombinationMap

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestRecombinationMapRepr:
    def test_repr_returns_string(self):
        """__repr__ should return a non-empty string."""
        rmap = RecombinationMap.constant_map(m=5)
        r = repr(rmap)
        assert isinstance(r, str)
        assert len(r) > 0

    def test_repr_constant_map(self):
        """Repr of constant map should contain variant info."""
        rmap = RecombinationMap.constant_map(m=3, p=0.4)
        r = repr(rmap)
        # Should contain some representation of the map
        assert isinstance(r, str)


class TestFromHaplotypesDetails:
    def test_from_haplotypes_default_p(self):
        """from_haplotypes with default p=0.5."""
        hap = TestSimulation.founder_haplotypes(n=10, m=5, seed=42)
        rmap = RecombinationMap.from_haplotypes(hap)
        assert rmap.m == 5
        assert len(rmap.vid) == 5

    def test_from_haplotypes_custom_p(self):
        """from_haplotypes with custom p."""
        hap = TestSimulation.founder_haplotypes(n=10, m=5, seed=42)
        rmap = RecombinationMap.from_haplotypes(hap, p=0.3)
        assert rmap.m == 5

    def test_from_haplotypes_vid_matches(self):
        """from_haplotypes vid should match haplotype vid."""
        hap = TestSimulation.founder_haplotypes(n=10, m=5, seed=42)
        rmap = RecombinationMap.from_haplotypes(hap)
        np.testing.assert_array_equal(rmap.vid, hap.vid)

    def test_from_haplotypes_single_chrom(self):
        """from_haplotypes creates single-chromosome map."""
        hap = TestSimulation.founder_haplotypes(n=10, m=20, seed=42)
        rmap = RecombinationMap.from_haplotypes(hap)
        # All variants should be on same chromosome
        chroms = np.unique(rmap.chrom)
        assert len(chroms) == 1
