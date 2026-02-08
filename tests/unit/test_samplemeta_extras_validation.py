"""
Unit tests for SampleMeta extra field validation and properties.

Tests:
1. Extra field length mismatch raises ValueError
2. SampleMeta.n_fam counts unique families
3. SampleMeta.n_female and n_male counts
4. SampleMeta.unique_identifier format
5. SampleMeta.subset with boolean mask
6. SampleMeta.subset with fancy indexing preserves extras
7. SampleMeta.with_generation creates new generation
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta


class TestSampleMetaExtrasValidation:
    def test_extra_length_mismatch_raises(self):
        """Extra field with wrong length should raise ValueError."""
        with pytest.raises(ValueError, match="extra.*length"):
            SampleMeta(
                iid=np.arange(4),
                fid=np.arange(4),
                extra={'cohort': np.array([0, 1])},  # length 2 != 4
            )

    def test_extra_multiple_one_wrong_raises(self):
        """If one of several extra fields has wrong length, should raise."""
        with pytest.raises(ValueError, match="extra.*length"):
            SampleMeta(
                iid=np.arange(4),
                fid=np.arange(4),
                extra={
                    'ok': np.array([1, 2, 3, 4]),
                    'bad': np.array([1, 2]),
                },
            )


class TestSampleMetaProperties:
    def test_n_fam(self):
        """n_fam should count unique families."""
        sm = SampleMeta(
            iid=np.arange(6),
            fid=np.array([0, 0, 1, 1, 2, 2]),
        )
        assert sm.n_fam == 3

    def test_n_female_n_male(self):
        """n_female and n_male should match sex array."""
        sm = SampleMeta(
            iid=np.arange(5),
            fid=np.arange(5),
            sex=np.array([0, 0, 0, 1, 1]),
        )
        assert sm.n_female == 3
        assert sm.n_male == 2

    def test_unique_identifier_format(self):
        """unique_identifier should be '{generation}.{iid}.{fid}'."""
        sm = SampleMeta(
            iid=np.array([10, 20]),
            fid=np.array([1, 2]),
            generation=3,
        )
        uids = sm.unique_identifier
        assert uids[0] == '3.10.1'
        assert uids[1] == '3.20.2'


class TestSampleMetaSubsetAdvanced:
    def test_subset_boolean_mask(self):
        """Subset with boolean mask should work."""
        sm = SampleMeta(
            iid=np.arange(5),
            fid=np.arange(5),
            sex=np.array([0, 1, 0, 1, 0]),
            extra={'batch': np.array([10, 20, 30, 40, 50])},
        )
        mask = np.array([True, False, True, False, True])
        sub = sm.subset(mask)
        assert sub.n == 3
        np.testing.assert_array_equal(sub.iid, [0, 2, 4])
        np.testing.assert_array_equal(sub.extra['batch'], [10, 30, 50])

    def test_subset_preserves_generation(self):
        """Subset should preserve generation."""
        sm = SampleMeta(iid=np.arange(4), generation=7)
        sub = sm.subset(np.array([0, 2]))
        assert sub.generation == 7

    def test_with_generation(self):
        """with_generation should create new SampleMeta with different gen."""
        sm = SampleMeta(iid=np.arange(3), generation=0)
        sm2 = sm.with_generation(5)
        assert sm2.generation == 5
        assert sm.generation == 0  # original unchanged
        np.testing.assert_array_equal(sm2.iid, sm.iid)
