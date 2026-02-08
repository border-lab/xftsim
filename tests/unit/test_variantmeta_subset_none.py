"""
Unit tests for VariantMeta subset with None optional fields and extras.

Tests:
1. Subset when chrom is None
2. Subset when pos_bp, pos_cM are None
3. Subset when af is None
4. Subset preserves extras
5. Subset when all optional fields are None
6. VariantMeta extras validation (length mismatch)
"""
import numpy as np
import pytest

from xftsim.struct import VariantMeta


class TestVariantMetaSubsetNone:
    def test_subset_chrom_none(self):
        """Subset should work when chrom is None."""
        vm = VariantMeta(vid=np.array(['v0', 'v1', 'v2']), chrom=None)
        sub = vm.subset(np.array([0, 2]))
        assert sub.chrom is None
        np.testing.assert_array_equal(sub.vid, ['v0', 'v2'])

    def test_subset_pos_none(self):
        """Subset should work when pos_bp and pos_cM are None."""
        vm = VariantMeta(vid=np.array(['v0', 'v1', 'v2']),
                         pos_bp=None, pos_cM=None)
        sub = vm.subset(np.array([1]))
        assert sub.pos_bp is None
        assert sub.pos_cM is None

    def test_subset_af_none(self):
        """Subset should work when af is None."""
        vm = VariantMeta(vid=np.array(['v0', 'v1', 'v2']), af=None)
        sub = vm.subset(np.array([0, 1]))
        assert sub.af is None

    def test_subset_preserves_extras(self):
        """Subset should propagate extra fields."""
        vm = VariantMeta(
            vid=np.array(['v0', 'v1', 'v2']),
            extra={'maf': np.array([0.1, 0.3, 0.5])},
        )
        sub = vm.subset(np.array([1, 2]))
        np.testing.assert_array_almost_equal(sub.extra['maf'], [0.3, 0.5])

    def test_subset_all_none(self):
        """Subset when all optional fields are None should work."""
        vm = VariantMeta(
            vid=np.array(['v0', 'v1', 'v2', 'v3']),
            chrom=None, pos_bp=None, pos_cM=None, af=None,
            zero_allele=None, one_allele=None,
        )
        sub = vm.subset(np.array([0, 3]))
        assert sub.m == 2
        assert sub.chrom is None
        assert sub.pos_bp is None
        assert sub.af is None

    def test_subset_with_all_fields(self):
        """Subset with all fields populated should preserve everything."""
        vm = VariantMeta(
            vid=np.array(['v0', 'v1', 'v2']),
            chrom=np.array([1, 1, 2]),
            pos_bp=np.array([100, 200, 300]),
            pos_cM=np.array([0.1, 0.2, 0.3]),
            af=np.array([0.1, 0.5, 0.9]),
            zero_allele=np.array(['A', 'G', 'T']),
            one_allele=np.array(['C', 'T', 'A']),
        )
        sub = vm.subset(np.array([0, 2]))
        np.testing.assert_array_equal(sub.vid, ['v0', 'v2'])
        np.testing.assert_array_equal(sub.chrom, [1, 2])
        np.testing.assert_array_equal(sub.pos_bp, [100, 300])
        np.testing.assert_array_almost_equal(sub.af, [0.1, 0.9])
        np.testing.assert_array_equal(sub.zero_allele, ['A', 'T'])
        np.testing.assert_array_equal(sub.one_allele, ['C', 'A'])
