"""
Unit tests for PedigreeArray bounds checking.

Tests:
1. Valid construction
2. Maternal idx out of bounds (>= parent_n) → ValueError
3. Paternal idx out of bounds → ValueError
4. Negative maternal idx → ValueError
5. Negative paternal idx → ValueError
6. Length mismatch maternal → ValueError
7. Length mismatch paternal → ValueError
8. Empty PedigreeArray (n=0) is valid
9. dtype coercion (int32 → intp)
"""
import numpy as np
import pytest

from xftsim.struct import PedigreeArray, SampleMeta


class TestPedigreeArrayBounds:
    def test_valid_construction(self):
        sm = SampleMeta(iid=np.arange(4), generation=1)
        ped = PedigreeArray(
            offspring_samples=sm,
            maternal_idx=np.array([0, 0, 1, 1]),
            paternal_idx=np.array([5, 5, 6, 6]),
            parent_n=10,
        )
        assert ped.parent_n == 10
        np.testing.assert_array_equal(ped.maternal_idx, [0, 0, 1, 1])

    def test_maternal_out_of_bounds_raises(self):
        sm = SampleMeta(iid=np.arange(2), generation=1)
        with pytest.raises(ValueError, match="maternal_idx out of bounds"):
            PedigreeArray(
                offspring_samples=sm,
                maternal_idx=np.array([0, 10]),  # 10 >= parent_n=10
                paternal_idx=np.array([5, 5]),
                parent_n=10,
            )

    def test_paternal_out_of_bounds_raises(self):
        sm = SampleMeta(iid=np.arange(2), generation=1)
        with pytest.raises(ValueError, match="paternal_idx out of bounds"):
            PedigreeArray(
                offspring_samples=sm,
                maternal_idx=np.array([0, 0]),
                paternal_idx=np.array([5, 100]),  # 100 >= parent_n=10
                parent_n=10,
            )

    def test_negative_maternal_raises(self):
        sm = SampleMeta(iid=np.arange(2), generation=1)
        with pytest.raises(ValueError, match="maternal_idx out of bounds"):
            PedigreeArray(
                offspring_samples=sm,
                maternal_idx=np.array([-1, 0]),
                paternal_idx=np.array([5, 5]),
                parent_n=10,
            )

    def test_negative_paternal_raises(self):
        sm = SampleMeta(iid=np.arange(2), generation=1)
        with pytest.raises(ValueError, match="paternal_idx out of bounds"):
            PedigreeArray(
                offspring_samples=sm,
                maternal_idx=np.array([0, 0]),
                paternal_idx=np.array([-1, 5]),
                parent_n=10,
            )

    def test_maternal_length_mismatch(self):
        sm = SampleMeta(iid=np.arange(4), generation=1)
        with pytest.raises(ValueError, match="maternal_idx length"):
            PedigreeArray(
                offspring_samples=sm,
                maternal_idx=np.array([0, 0]),  # 2 != 4
                paternal_idx=np.array([5, 5, 6, 6]),
                parent_n=10,
            )

    def test_paternal_length_mismatch(self):
        sm = SampleMeta(iid=np.arange(4), generation=1)
        with pytest.raises(ValueError, match="paternal_idx length"):
            PedigreeArray(
                offspring_samples=sm,
                maternal_idx=np.array([0, 0, 1, 1]),
                paternal_idx=np.array([5]),  # 1 != 4
                parent_n=10,
            )

    def test_empty_valid(self):
        sm = SampleMeta(iid=np.array([], dtype=np.int64), generation=1)
        ped = PedigreeArray(
            offspring_samples=sm,
            maternal_idx=np.array([], dtype=np.int64),
            paternal_idx=np.array([], dtype=np.int64),
            parent_n=10,
        )
        assert len(ped.maternal_idx) == 0

    def test_dtype_coercion(self):
        sm = SampleMeta(iid=np.arange(2), generation=1)
        ped = PedigreeArray(
            offspring_samples=sm,
            maternal_idx=np.array([0, 1], dtype=np.int32),
            paternal_idx=np.array([5, 6], dtype=np.int32),
            parent_n=10,
        )
        assert ped.maternal_idx.dtype == np.intp
        assert ped.paternal_idx.dtype == np.intp
