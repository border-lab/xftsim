"""
Unit tests for PedigreeArray validation and properties.

Tests:
1. PedigreeArray validation: maternal_idx length mismatch, paternal_idx length mismatch,
   maternal OOB, paternal OOB, negative indices, dtype coercion
2. PedigreeArray: repr, empty (n=0), valid construction properties
3. PedigreeArray: boundary index (parent_n-1), zero parent_n with empty arrays
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, PedigreeArray


class TestPedigreeArrayValidation:
    def test_maternal_idx_length_mismatch(self):
        """maternal_idx length != n should raise ValueError."""
        sm = SampleMeta(iid=np.arange(4))
        with pytest.raises(ValueError, match="maternal_idx length"):
            PedigreeArray(
                offspring_samples=sm,
                maternal_idx=np.array([0, 1, 2]),  # 3 != 4
                paternal_idx=np.array([0, 1, 2, 3]),
                parent_n=10,
            )

    def test_paternal_idx_length_mismatch(self):
        """paternal_idx length != n should raise ValueError."""
        sm = SampleMeta(iid=np.arange(4))
        with pytest.raises(ValueError, match="paternal_idx length"):
            PedigreeArray(
                offspring_samples=sm,
                maternal_idx=np.array([0, 1, 2, 3]),
                paternal_idx=np.array([0, 1]),  # 2 != 4
                parent_n=10,
            )

    def test_maternal_idx_oob_high(self):
        """maternal_idx >= parent_n should raise ValueError."""
        sm = SampleMeta(iid=np.arange(2))
        with pytest.raises(ValueError, match="maternal_idx out of bounds"):
            PedigreeArray(
                offspring_samples=sm,
                maternal_idx=np.array([0, 10]),  # 10 >= parent_n=5
                paternal_idx=np.array([0, 1]),
                parent_n=5,
            )

    def test_paternal_idx_oob_high(self):
        """paternal_idx >= parent_n should raise ValueError."""
        sm = SampleMeta(iid=np.arange(2))
        with pytest.raises(ValueError, match="paternal_idx out of bounds"):
            PedigreeArray(
                offspring_samples=sm,
                maternal_idx=np.array([0, 1]),
                paternal_idx=np.array([0, 5]),  # 5 >= parent_n=5
                parent_n=5,
            )

    def test_maternal_idx_negative(self):
        """Negative maternal_idx should raise ValueError."""
        sm = SampleMeta(iid=np.arange(2))
        with pytest.raises(ValueError, match="maternal_idx out of bounds"):
            PedigreeArray(
                offspring_samples=sm,
                maternal_idx=np.array([-1, 0]),
                paternal_idx=np.array([0, 1]),
                parent_n=5,
            )

    def test_paternal_idx_negative(self):
        """Negative paternal_idx should raise ValueError."""
        sm = SampleMeta(iid=np.arange(2))
        with pytest.raises(ValueError, match="paternal_idx out of bounds"):
            PedigreeArray(
                offspring_samples=sm,
                maternal_idx=np.array([0, 1]),
                paternal_idx=np.array([0, -1]),
                parent_n=5,
            )

    def test_dtype_coercion(self):
        """Indices should be coerced to intp."""
        sm = SampleMeta(iid=np.arange(2))
        ped = PedigreeArray(
            offspring_samples=sm,
            maternal_idx=np.array([0, 1], dtype=np.float64),
            paternal_idx=np.array([0, 1], dtype=np.int32),
            parent_n=5,
        )
        assert ped.maternal_idx.dtype == np.intp
        assert ped.paternal_idx.dtype == np.intp


class TestPedigreeArrayProperties:
    def test_valid_construction(self):
        """Valid PedigreeArray should store all fields."""
        sm = SampleMeta(iid=np.arange(4))
        ped = PedigreeArray(
            offspring_samples=sm,
            maternal_idx=np.array([0, 0, 1, 1]),
            paternal_idx=np.array([2, 2, 3, 3]),
            parent_n=10,
        )
        assert ped.parent_n == 10
        assert len(ped.maternal_idx) == 4
        assert len(ped.paternal_idx) == 4
        assert ped.offspring_samples.n == 4

    def test_boundary_index(self):
        """Index == parent_n - 1 should be valid."""
        sm = SampleMeta(iid=np.arange(2))
        ped = PedigreeArray(
            offspring_samples=sm,
            maternal_idx=np.array([4, 4]),  # parent_n - 1
            paternal_idx=np.array([4, 4]),
            parent_n=5,
        )
        np.testing.assert_array_equal(ped.maternal_idx, [4, 4])

    def test_empty_pedigree(self):
        """n=0 offspring should be valid (bounds check skipped)."""
        sm = SampleMeta(iid=np.array([], dtype=np.int64))
        ped = PedigreeArray(
            offspring_samples=sm,
            maternal_idx=np.array([], dtype=np.intp),
            paternal_idx=np.array([], dtype=np.intp),
            parent_n=10,
        )
        assert len(ped.maternal_idx) == 0
        assert len(ped.paternal_idx) == 0

    def test_same_parent_for_all(self):
        """All offspring from same parents should be valid."""
        sm = SampleMeta(iid=np.arange(10))
        ped = PedigreeArray(
            offspring_samples=sm,
            maternal_idx=np.zeros(10, dtype=np.intp),
            paternal_idx=np.ones(10, dtype=np.intp),
            parent_n=5,
        )
        np.testing.assert_array_equal(ped.maternal_idx, np.zeros(10))
        np.testing.assert_array_equal(ped.paternal_idx, np.ones(10))
