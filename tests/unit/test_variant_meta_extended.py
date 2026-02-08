"""
Extended unit tests for VariantMeta.

Tests:
1. VariantMeta with chrom field
2. VariantMeta vid length
3. VariantMeta subset
4. VariantMeta default chrom
"""
import numpy as np
import pytest

from xftsim.struct import VariantMeta


class TestVariantMetaConstruction:
    def test_basic(self):
        vm = VariantMeta(vid=np.array(['rs1', 'rs2', 'rs3']))
        assert len(vm.vid) == 3

    def test_with_chrom(self):
        chrom = np.array([1, 1, 2])
        vm = VariantMeta(vid=np.array(['rs1', 'rs2', 'rs3']), chrom=chrom)
        np.testing.assert_array_equal(vm.chrom, [1, 1, 2])

    def test_default_chrom(self):
        vm = VariantMeta(vid=np.array(['rs1', 'rs2']))
        # Default: chrom may be None or an array
        assert vm.chrom is None or len(vm.chrom) == 2

    def test_m_property(self):
        vm = VariantMeta(vid=np.array(['v0', 'v1', 'v2', 'v3']))
        assert vm.m == 4


class TestVariantMetaSubset:
    def test_subset_by_index(self):
        vm = VariantMeta(vid=np.array(['v0', 'v1', 'v2', 'v3']),
                         chrom=np.array([1, 1, 2, 2]))
        sub = vm.subset(np.array([0, 2]))
        np.testing.assert_array_equal(sub.vid, ['v0', 'v2'])
        np.testing.assert_array_equal(sub.chrom, [1, 2])

    def test_subset_single(self):
        vm = VariantMeta(vid=np.array(['v0', 'v1', 'v2']))
        sub = vm.subset(np.array([1]))
        assert len(sub.vid) == 1
        assert sub.vid[0] == 'v1'
