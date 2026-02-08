"""
Unit tests for _resolve_grouping function.

Tests:
1. None grouping → None
2. FID grouping → samples.fid
3. sex grouping → samples.sex
4. mother grouping with pedigree → maternal_idx
5. father grouping without pedigree at gen 0 → None with warning
6. Unknown grouping → ValueError
7. Extra field grouping works
8. mother grouping at gen 2 without pedigree → None with warning
"""
import numpy as np
import pytest
import warnings

from xftsim.narch import _resolve_grouping
from xftsim.struct import SampleMeta, DenseHaplotypeArray, VariantMeta, PedigreeArray


def _make_hap(n, m=3, fid=None, sex=None, extra=None):
    sm = SampleMeta(iid=np.arange(n), fid=fid, sex=sex, extra=extra or {})
    vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
    return DenseHaplotypeArray(np.zeros((n, m, 2), dtype=np.int8), samples=sm, variants=vm)


class TestResolveGrouping:
    def test_none_returns_none(self):
        hap = _make_hap(5)
        result = _resolve_grouping(None, hap)
        assert result is None

    def test_fid_grouping(self):
        fids = np.array([0, 0, 1, 1, 2])
        hap = _make_hap(5, fid=fids)
        result = _resolve_grouping('FID', hap)
        np.testing.assert_array_equal(result, fids)

    def test_sex_grouping(self):
        sex = np.array([0, 1, 0, 1, 0])
        hap = _make_hap(5, sex=sex)
        result = _resolve_grouping('sex', hap)
        np.testing.assert_array_equal(result, sex)

    def test_mother_grouping_with_pedigree(self):
        """mother grouping at gen 1 with pedigree → maternal_idx."""
        hap = _make_hap(4)
        ped = PedigreeArray(
            offspring_samples=hap.samples,
            maternal_idx=np.array([0, 0, 1, 1]),
            paternal_idx=np.array([5, 5, 6, 6]),
            parent_n=10,
        )
        result = _resolve_grouping(
            'mother', hap,
            generation=1,
            pedigree_history={1: ped},
        )
        np.testing.assert_array_equal(result, np.array([0, 0, 1, 1]))

    def test_father_gen0_no_pedigree_warns(self):
        """father grouping at gen 0 → None with warning."""
        hap = _make_hap(4)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _resolve_grouping(
                'father', hap,
                generation=0,
            )
            assert len(w) == 1
            assert 'IID grouping' in str(w[0].message)
        assert result is None

    def test_unknown_grouping_raises(self):
        hap = _make_hap(5)
        with pytest.raises(ValueError, match="Unknown grouping"):
            _resolve_grouping('nonexistent', hap)

    def test_extra_field_grouping(self):
        """Grouping by extra field in SampleMeta."""
        extra = {'batch': np.array([0, 0, 1, 1, 1])}
        hap = _make_hap(5, extra=extra)
        result = _resolve_grouping('batch', hap)
        np.testing.assert_array_equal(result, np.array([0, 0, 1, 1, 1]))

    def test_mother_gen2_no_pedigree_warns(self):
        """mother grouping at gen 2 but no pedigree for gen 2."""
        hap = _make_hap(4)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _resolve_grouping(
                'mother', hap,
                generation=2,
                pedigree_history={1: None},
            )
            assert len(w) == 1
        assert result is None
