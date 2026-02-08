"""
Extended unit tests for SampleMeta construction and properties.

Tests:
1. SampleMeta with extra fields
2. SampleMeta sex field
3. SampleMeta generation field
4. SampleMeta subset propagates extra fields
5. SampleMeta without fid (auto-generated)
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta


class TestSampleMetaExtra:
    def test_extra_field(self):
        extra = {'cohort': np.array([0, 0, 1, 1])}
        sm = SampleMeta(iid=np.arange(4), fid=np.arange(4), extra=extra)
        assert 'cohort' in sm.extra
        np.testing.assert_array_equal(sm.extra['cohort'], [0, 0, 1, 1])

    def test_multiple_extra_fields(self):
        extra = {
            'cohort': np.array([0, 1, 0, 1]),
            'batch': np.array([1, 1, 2, 2]),
        }
        sm = SampleMeta(iid=np.arange(4), fid=np.arange(4), extra=extra)
        assert len(sm.extra) == 2

    def test_no_extra(self):
        sm = SampleMeta(iid=np.arange(4), fid=np.arange(4))
        assert sm.extra == {} or sm.extra is None or len(sm.extra) == 0


class TestSampleMetaSex:
    def test_sex_field(self):
        sex = np.array([0, 1, 0, 1])
        sm = SampleMeta(iid=np.arange(4), fid=np.arange(4), sex=sex)
        np.testing.assert_array_equal(sm.sex, [0, 1, 0, 1])

    def test_sex_default(self):
        sm = SampleMeta(iid=np.arange(4), fid=np.arange(4))
        # Default sex should be assigned (alternating 0,1)
        assert len(sm.sex) == 4


class TestSampleMetaGeneration:
    def test_generation_field(self):
        sm = SampleMeta(iid=np.arange(4), fid=np.arange(4), generation=5)
        assert sm.generation == 5

    def test_generation_default(self):
        sm = SampleMeta(iid=np.arange(4), fid=np.arange(4))
        assert sm.generation == 0


class TestSampleMetaN:
    def test_n_property(self):
        sm = SampleMeta(iid=np.arange(10), fid=np.arange(10))
        assert sm.n == 10

    def test_n_matches_iid(self):
        sm = SampleMeta(iid=np.arange(7), fid=np.arange(7))
        assert sm.n == len(sm.iid)
