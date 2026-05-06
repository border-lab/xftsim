"""
Unit tests for _resolve_grouping with unknown grouping variable.

Tests:
1. Unknown grouping raises ValueError with message
2. Extra field grouping works
3. Extra field with integer labels works
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray
from xftsim.arch import _resolve_grouping


def _make_hap(n=10, m=3, extra=None):
    sm = SampleMeta(iid=np.arange(n), fid=np.arange(n) // 2, extra=extra or {})
    vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
    geno = np.zeros((n, m, 2), dtype=np.int8)
    return DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)


class TestUnknownGrouping:
    def test_unknown_grouping_raises(self):
        """Unknown grouping variable (not FID/sex/mother/father/extra) should raise."""
        hap = _make_hap()
        with pytest.raises(ValueError, match="Unknown grouping variable"):
            _resolve_grouping('BOGUS', hap)

    def test_unknown_grouping_error_message(self):
        """Error message should list available variables."""
        hap = _make_hap()
        with pytest.raises(ValueError, match="Available"):
            _resolve_grouping('nonexistent', hap)


class TestExtraFieldGrouping:
    def test_extra_field_resolves(self):
        """Extra field should resolve to its values."""
        extra = {'batch': np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])}
        hap = _make_hap(extra=extra)
        labels = _resolve_grouping('batch', hap)
        np.testing.assert_array_equal(labels, extra['batch'])

    def test_extra_field_string_labels(self):
        """Extra field with string labels should work."""
        extra = {'site': np.array(['A', 'A', 'B', 'B', 'C', 'C', 'A', 'B', 'C', 'A'])}
        hap = _make_hap(extra=extra)
        labels = _resolve_grouping('site', hap)
        assert len(labels) == 10
        assert labels[0] == 'A'
        assert labels[2] == 'B'
