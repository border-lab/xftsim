"""
Unit tests for MateAssignment validation and edge cases.

Tests:
1. Valid construction
2. Maternal index length mismatch
3. Paternal index length mismatch
4. Negative maternal indices
5. Negative paternal indices
6. n_offspring property
7. repr
8. dtype coercion (int32 → int64)
9. LinearAssortativeMating boundary r values
10. LinearAssortativeMating r=0 fallback
11. LinearAssortativeMating no valid components
12. RandomMating offspring_per_pair validation
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, PhenotypeArray
from xftsim.mate import MateAssignment, RandomMating, LinearAssortativeMating


class TestNMateAssignmentValidation:
    def test_valid_construction(self):
        sm = SampleMeta(iid=np.arange(4), generation=1)
        ma = MateAssignment(
            offspring_samples=sm,
            maternal_idx=np.array([0, 0, 1, 1]),
            paternal_idx=np.array([2, 2, 3, 3]),
        )
        assert ma.n_offspring == 4

    def test_maternal_idx_length_mismatch(self):
        sm = SampleMeta(iid=np.arange(4), generation=1)
        with pytest.raises(ValueError, match="maternal_idx length"):
            MateAssignment(
                offspring_samples=sm,
                maternal_idx=np.array([0, 0]),
                paternal_idx=np.array([2, 2, 3, 3]),
            )

    def test_paternal_idx_length_mismatch(self):
        sm = SampleMeta(iid=np.arange(4), generation=1)
        with pytest.raises(ValueError, match="paternal_idx length"):
            MateAssignment(
                offspring_samples=sm,
                maternal_idx=np.array([0, 0, 1, 1]),
                paternal_idx=np.array([2, 2]),
            )

    def test_negative_maternal_idx(self):
        sm = SampleMeta(iid=np.arange(2), generation=1)
        with pytest.raises(ValueError, match="negative"):
            MateAssignment(
                offspring_samples=sm,
                maternal_idx=np.array([-1, 0]),
                paternal_idx=np.array([0, 1]),
            )

    def test_negative_paternal_idx(self):
        sm = SampleMeta(iid=np.arange(2), generation=1)
        with pytest.raises(ValueError, match="negative"):
            MateAssignment(
                offspring_samples=sm,
                maternal_idx=np.array([0, 1]),
                paternal_idx=np.array([0, -1]),
            )

    def test_dtype_coercion(self):
        """int32 arrays should be coerced to int64."""
        sm = SampleMeta(iid=np.arange(2), generation=1)
        ma = MateAssignment(
            offspring_samples=sm,
            maternal_idx=np.array([0, 1], dtype=np.int32),
            paternal_idx=np.array([1, 0], dtype=np.int32),
        )
        assert ma.maternal_idx.dtype == np.int64
        assert ma.paternal_idx.dtype == np.int64

    def test_repr(self):
        sm = SampleMeta(iid=np.arange(4), generation=2)
        ma = MateAssignment(
            offspring_samples=sm,
            maternal_idx=np.array([0, 0, 1, 1]),
            paternal_idx=np.array([2, 2, 3, 3]),
        )
        r = repr(ma)
        assert 'MateAssignment' in r
        assert 'n_offspring=4' in r


class TestRandomMatingValidation:
    def test_offspring_per_pair_zero_raises(self):
        with pytest.raises(ValueError, match="offspring_per_pair"):
            RandomMating(offspring_per_pair=0)

    def test_offspring_per_pair_negative_raises(self):
        with pytest.raises(ValueError, match="offspring_per_pair"):
            RandomMating(offspring_per_pair=-1)

    def test_all_one_sex_raises(self):
        sm = SampleMeta(
            iid=np.arange(10),
            sex=np.ones(10, dtype=np.int64),
        )
        mating = RandomMating()
        with pytest.raises(ValueError, match="at least one female"):
            mating.mate(sm, rng=np.random.RandomState(42))

    def test_minimal_population(self):
        """One female and one male."""
        sm = SampleMeta(
            iid=np.arange(2),
            sex=np.array([0, 1]),
        )
        mating = RandomMating(offspring_per_pair=2)
        ma = mating.mate(sm, rng=np.random.RandomState(42))
        assert ma.n_offspring == 2

    def test_repr(self):
        r = repr(RandomMating(offspring_per_pair=3))
        assert 'RandomMating' in r
        assert '3' in r


class TestLinearAssortativeMatingValidation:
    def test_r_at_negative_one_raises(self):
        with pytest.raises(ValueError, match="r must be"):
            LinearAssortativeMating(component_names=['Y'], r=-1.0)

    def test_r_at_positive_one_raises(self):
        with pytest.raises(ValueError, match="r must be"):
            LinearAssortativeMating(component_names=['Y'], r=1.0)

    def test_r_zero_fallback(self):
        """r=0 should behave like random mating."""
        sm = SampleMeta(
            iid=np.arange(10),
            sex=np.tile([0, 1], 5),
        )
        pheno = PhenotypeArray(samples=sm)
        pheno._values['Y'] = np.arange(10, dtype=float)

        mating = LinearAssortativeMating(component_names=['Y'], r=0.0)
        ma = mating.mate(sm, rng=np.random.RandomState(42), phenotypes=pheno)
        assert ma.n_offspring > 0

    def test_phenotypes_none_fallback(self):
        """phenotypes=None should fallback to random mating."""
        sm = SampleMeta(
            iid=np.arange(10),
            sex=np.tile([0, 1], 5),
        )
        mating = LinearAssortativeMating(component_names=['Y'], r=0.5)
        ma = mating.mate(sm, rng=np.random.RandomState(42), phenotypes=None)
        assert ma.n_offspring > 0

    def test_no_valid_components(self):
        """Component names not in phenotypes → zero composite, should not crash."""
        sm = SampleMeta(
            iid=np.arange(10),
            sex=np.tile([0, 1], 5),
        )
        pheno = PhenotypeArray(samples=sm)
        pheno._values['X'] = np.arange(10, dtype=float)

        mating = LinearAssortativeMating(component_names=['Y'], r=0.5)
        ma = mating.mate(sm, rng=np.random.RandomState(42), phenotypes=pheno)
        assert ma.n_offspring > 0

    def test_offspring_per_pair_validation(self):
        with pytest.raises(ValueError, match="offspring_per_pair"):
            LinearAssortativeMating(component_names=['Y'], r=0.5, offspring_per_pair=0)

    def test_repr(self):
        r = repr(LinearAssortativeMating(component_names=['A', 'B'], r=0.3))
        assert 'LinearAssortativeMating' in r
        assert '0.3' in r
