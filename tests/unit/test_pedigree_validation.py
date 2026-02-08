"""
Unit tests for PedigreeArray validation.

Tests:
1. Valid construction
2. Maternal index length mismatch
3. Paternal index length mismatch
4. Maternal index out of bounds (too high)
5. Maternal index out of bounds (negative)
6. Paternal index out of bounds
7. Boundary value (index == parent_n - 1)
8. Dtype coercion
9. n=0 (empty) construction
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, PedigreeArray


class TestPedigreeArrayValidation:
    def test_valid_construction(self):
        sm = SampleMeta(iid=np.arange(4), generation=1)
        ped = PedigreeArray(
            offspring_samples=sm,
            maternal_idx=np.array([0, 0, 1, 1]),
            paternal_idx=np.array([2, 2, 3, 3]),
            parent_n=5,
        )
        assert len(ped.maternal_idx) == 4
        assert len(ped.paternal_idx) == 4

    def test_maternal_idx_length_mismatch(self):
        sm = SampleMeta(iid=np.arange(4), generation=1)
        with pytest.raises(ValueError, match="maternal_idx length"):
            PedigreeArray(
                offspring_samples=sm,
                maternal_idx=np.array([0, 0]),
                paternal_idx=np.array([2, 2, 3, 3]),
                parent_n=5,
            )

    def test_paternal_idx_length_mismatch(self):
        sm = SampleMeta(iid=np.arange(4), generation=1)
        with pytest.raises(ValueError, match="paternal_idx length"):
            PedigreeArray(
                offspring_samples=sm,
                maternal_idx=np.array([0, 0, 1, 1]),
                paternal_idx=np.array([2, 2]),
                parent_n=5,
            )

    def test_maternal_idx_too_high(self):
        sm = SampleMeta(iid=np.arange(2), generation=1)
        with pytest.raises(ValueError, match="maternal_idx out of bounds"):
            PedigreeArray(
                offspring_samples=sm,
                maternal_idx=np.array([5, 0]),  # 5 >= parent_n=5
                paternal_idx=np.array([0, 1]),
                parent_n=5,
            )

    def test_maternal_idx_negative(self):
        sm = SampleMeta(iid=np.arange(2), generation=1)
        with pytest.raises(ValueError, match="maternal_idx out of bounds"):
            PedigreeArray(
                offspring_samples=sm,
                maternal_idx=np.array([-1, 0]),
                paternal_idx=np.array([0, 1]),
                parent_n=5,
            )

    def test_paternal_idx_out_of_bounds(self):
        sm = SampleMeta(iid=np.arange(2), generation=1)
        with pytest.raises(ValueError, match="paternal_idx out of bounds"):
            PedigreeArray(
                offspring_samples=sm,
                maternal_idx=np.array([0, 1]),
                paternal_idx=np.array([0, 5]),
                parent_n=5,
            )

    def test_boundary_index_valid(self):
        """Index == parent_n - 1 should be valid."""
        sm = SampleMeta(iid=np.arange(2), generation=1)
        ped = PedigreeArray(
            offspring_samples=sm,
            maternal_idx=np.array([4, 4]),  # parent_n-1
            paternal_idx=np.array([0, 0]),
            parent_n=5,
        )
        assert ped.maternal_idx[0] == 4

    def test_dtype_coercion(self):
        sm = SampleMeta(iid=np.arange(2), generation=1)
        ped = PedigreeArray(
            offspring_samples=sm,
            maternal_idx=np.array([0, 1], dtype=np.int32),
            paternal_idx=np.array([1, 0], dtype=np.int32),
            parent_n=5,
        )
        assert ped.maternal_idx.dtype == np.intp
        assert ped.paternal_idx.dtype == np.intp
