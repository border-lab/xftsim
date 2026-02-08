"""
Unit tests for _resolve_grouping warning and error paths.

Tests:
1. mother/father grouping at gen 0 → warns + returns None
2. mother/father grouping with missing pedigree → warns + returns None
3. Unknown grouping variable → raises ValueError
4. Extra field grouping
"""
import numpy as np
import pytest
import warnings

from xftsim.struct import SampleMeta, DenseHaplotypeArray, VariantMeta, PedigreeArray
from xftsim.narch import _resolve_grouping


def _make_hap(n=10, m=3, generation=0, extra=None):
    sm_kwargs = dict(iid=np.arange(n), fid=np.arange(n) // 2, generation=generation)
    if extra is not None:
        sm_kwargs['extra'] = extra
    sm = SampleMeta(**sm_kwargs)
    vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
    geno = np.zeros((n, m, 2), dtype=np.int8)
    return DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm, generation=generation)


class TestMotherFatherGroupingGen0:
    def test_mother_grouping_gen0_warns(self):
        hap = _make_hap(generation=0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            result = _resolve_grouping('mother', hap, generation=0, pedigree_history={})
        assert result is None
        assert len(w) == 1
        assert 'falling back' in str(w[0].message)

    def test_father_grouping_gen0_warns(self):
        hap = _make_hap(generation=0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            result = _resolve_grouping('father', hap, generation=0, pedigree_history={})
        assert result is None
        assert len(w) == 1

    def test_mother_grouping_missing_pedigree_warns(self):
        """Pedigree exists but not for current generation."""
        hap = _make_hap(generation=2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            result = _resolve_grouping('mother', hap, generation=2, pedigree_history={})
        assert result is None
        assert len(w) == 1


class TestMotherFatherGroupingNormal:
    def test_mother_grouping_with_pedigree(self):
        hap = _make_hap(n=10, generation=1)
        n = 10
        sm = SampleMeta(iid=np.arange(n), fid=np.arange(n) // 2, generation=1)
        ped = PedigreeArray(
            offspring_samples=sm,
            maternal_idx=np.repeat(np.arange(5), 2),
            paternal_idx=np.repeat(np.arange(5, 10), 2),
            parent_n=10,
        )
        result = _resolve_grouping('mother', hap, generation=1, pedigree_history={1: ped})
        np.testing.assert_array_equal(result, ped.maternal_idx)

    def test_father_grouping_with_pedigree(self):
        hap = _make_hap(n=10, generation=1)
        n = 10
        sm = SampleMeta(iid=np.arange(n), fid=np.arange(n) // 2, generation=1)
        ped = PedigreeArray(
            offspring_samples=sm,
            maternal_idx=np.repeat(np.arange(5), 2),
            paternal_idx=np.repeat(np.arange(5, 10), 2),
            parent_n=10,
        )
        result = _resolve_grouping('father', hap, generation=1, pedigree_history={1: ped})
        np.testing.assert_array_equal(result, ped.paternal_idx)


class TestUnknownGrouping:
    def test_unknown_raises(self):
        hap = _make_hap()
        with pytest.raises(ValueError, match="Unknown grouping variable"):
            _resolve_grouping('UNKNOWN', hap, generation=0, pedigree_history={})


class TestExtraFieldGrouping:
    def test_extra_field(self):
        extra = {'cohort': np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0])}
        hap = _make_hap(extra=extra)
        result = _resolve_grouping('cohort', hap, generation=0, pedigree_history={})
        np.testing.assert_array_equal(result, extra['cohort'])
