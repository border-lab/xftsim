"""
Unit tests for RecombinationMap.from_haplotypes and other constructors.

Tests:
1. from_haplotypes produces correct m
2. from_haplotypes preserves vid
3. from_haplotypes default p=0.5
4. from_haplotypes custom p
5. constant_map basic
6. RecombinationMap repr
7. Default p=None gives 0.5
"""
import numpy as np
import pytest

from xftsim.reproduce import RecombinationMap

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestFromHaplotypes:
    def test_correct_m(self):
        hap = TestSimulation.founder_haplotypes(n=10, m=8, seed=42)
        rmap = RecombinationMap.from_haplotypes(hap)
        assert rmap.m == 8

    def test_preserves_vid(self):
        hap = TestSimulation.founder_haplotypes(n=10, m=8, seed=42)
        rmap = RecombinationMap.from_haplotypes(hap)
        np.testing.assert_array_equal(rmap.vid, hap.variants.vid)

    def test_default_p_half(self):
        hap = TestSimulation.founder_haplotypes(n=10, m=5, seed=42)
        rmap = RecombinationMap.from_haplotypes(hap)
        np.testing.assert_allclose(rmap._probabilities, 0.5)

    def test_custom_p(self):
        hap = TestSimulation.founder_haplotypes(n=10, m=5, seed=42)
        rmap = RecombinationMap.from_haplotypes(hap, p=0.1)
        # Index 0 is chrom boundary → 0.5, rest 0.1
        assert rmap._probabilities[0] == 0.5
        for i in range(1, 5):
            assert rmap._probabilities[i] == 0.1


class TestConstantMap:
    def test_basic(self):
        rmap = RecombinationMap.constant_map(m=10, p=0.3)
        assert rmap.m == 10
        # Index 0 forced to 0.5; rest 0.3
        assert rmap._probabilities[0] == 0.5
        np.testing.assert_allclose(rmap._probabilities[1:], 0.3)

    def test_default_p(self):
        rmap = RecombinationMap.constant_map(m=5)
        np.testing.assert_allclose(rmap._probabilities, 0.5)


class TestRecombinationMapRepr:
    def test_repr_has_columns(self):
        rmap = RecombinationMap.constant_map(m=3, p=0.5)
        r = repr(rmap)
        assert 'vid' in r
        assert 'chrom' in r
        assert 'p' in r


class TestRecombinationMapDefaultP:
    def test_none_p_gives_half(self):
        """When p is neither float nor array, default to 0.5."""
        rmap = RecombinationMap(m=5)  # p=None default
        np.testing.assert_allclose(rmap._probabilities, 0.5)

    def test_vid_infers_m(self):
        """When m is not given but vid is, m is inferred."""
        vid = np.array(['a', 'b', 'c'])
        rmap = RecombinationMap(vid=vid)
        assert rmap.m == 3

    def test_no_m_or_vid_raises(self):
        """Must provide m or vid."""
        with pytest.raises(ValueError, match="Must provide m or vid"):
            RecombinationMap()
