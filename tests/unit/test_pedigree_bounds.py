"""
Unit tests for PedigreeArray bounds checking and edge cases.

Tests:
1. Out-of-bounds maternal_idx raises
2. Out-of-bounds paternal_idx raises
3. Length mismatch raises
4. Empty pedigree is valid
5. Dtype coercion to intp
6. Valid construction
7. repr
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, PedigreeArray


class TestPedigreeArrayBounds:
    def test_maternal_out_of_bounds_raises(self):
        sm = SampleMeta(iid=np.arange(2), generation=1)
        with pytest.raises(ValueError, match="maternal_idx out of bounds"):
            PedigreeArray(
                offspring_samples=sm,
                maternal_idx=np.array([0, 5]),  # 5 >= parent_n=4
                paternal_idx=np.array([1, 1]),
                parent_n=4,
            )

    def test_paternal_out_of_bounds_raises(self):
        sm = SampleMeta(iid=np.arange(2), generation=1)
        with pytest.raises(ValueError, match="paternal_idx out of bounds"):
            PedigreeArray(
                offspring_samples=sm,
                maternal_idx=np.array([0, 0]),
                paternal_idx=np.array([1, 10]),  # 10 >= parent_n=4
                parent_n=4,
            )

    def test_negative_maternal_raises(self):
        sm = SampleMeta(iid=np.arange(2), generation=1)
        with pytest.raises(ValueError, match="maternal_idx out of bounds"):
            PedigreeArray(
                offspring_samples=sm,
                maternal_idx=np.array([-1, 0]),
                paternal_idx=np.array([1, 1]),
                parent_n=4,
            )

    def test_negative_paternal_raises(self):
        sm = SampleMeta(iid=np.arange(2), generation=1)
        with pytest.raises(ValueError, match="paternal_idx out of bounds"):
            PedigreeArray(
                offspring_samples=sm,
                maternal_idx=np.array([0, 0]),
                paternal_idx=np.array([-1, 1]),
                parent_n=4,
            )

    def test_maternal_length_mismatch(self):
        sm = SampleMeta(iid=np.arange(3), generation=1)
        with pytest.raises(ValueError, match="maternal_idx length"):
            PedigreeArray(
                offspring_samples=sm,
                maternal_idx=np.array([0, 0]),  # length 2 != 3
                paternal_idx=np.array([1, 1, 2]),
                parent_n=4,
            )

    def test_paternal_length_mismatch(self):
        sm = SampleMeta(iid=np.arange(3), generation=1)
        with pytest.raises(ValueError, match="paternal_idx length"):
            PedigreeArray(
                offspring_samples=sm,
                maternal_idx=np.array([0, 0, 1]),
                paternal_idx=np.array([1, 1]),  # length 2 != 3
                parent_n=4,
            )


class TestPedigreeArrayValid:
    def test_valid_construction(self):
        sm = SampleMeta(iid=np.arange(4), generation=1)
        ped = PedigreeArray(
            offspring_samples=sm,
            maternal_idx=np.array([0, 0, 2, 2]),
            paternal_idx=np.array([1, 1, 3, 3]),
            parent_n=10,
        )
        assert ped.offspring_samples.n == 4
        assert ped.parent_n == 10

    def test_empty_pedigree(self):
        sm = SampleMeta(iid=np.array([]), generation=1)
        ped = PedigreeArray(
            offspring_samples=sm,
            maternal_idx=np.array([], dtype=np.int64),
            paternal_idx=np.array([], dtype=np.int64),
            parent_n=10,
        )
        assert ped.offspring_samples.n == 0

    def test_dtype_coercion(self):
        sm = SampleMeta(iid=np.arange(2), generation=1)
        ped = PedigreeArray(
            offspring_samples=sm,
            maternal_idx=np.array([0, 0], dtype=np.float64),
            paternal_idx=np.array([1, 1], dtype=np.int32),
            parent_n=4,
        )
        assert ped.maternal_idx.dtype == np.intp
        assert ped.paternal_idx.dtype == np.intp

    def test_boundary_indices(self):
        """Indices at the exact boundary (parent_n - 1) should be valid."""
        sm = SampleMeta(iid=np.arange(2), generation=1)
        ped = PedigreeArray(
            offspring_samples=sm,
            maternal_idx=np.array([0, 3]),   # max valid = parent_n - 1 = 3
            paternal_idx=np.array([0, 3]),
            parent_n=4,
        )
        assert ped.offspring_samples.n == 2

    def test_single_offspring(self):
        sm = SampleMeta(iid=np.array([0]), generation=1)
        ped = PedigreeArray(
            offspring_samples=sm,
            maternal_idx=np.array([0]),
            paternal_idx=np.array([1]),
            parent_n=2,
        )
        assert ped.offspring_samples.n == 1
