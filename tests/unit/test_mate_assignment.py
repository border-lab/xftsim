"""
Unit tests for NMateAssignment validation and properties.

Tests:
1. Valid construction and n_offspring
2. maternal_idx length mismatch raises
3. paternal_idx length mismatch raises
4. Negative maternal_idx raises
5. Negative paternal_idx raises
6. dtype coercion to int64
7. Empty offspring (n=0) is valid
8. repr
9. RandomMating offspring structure (fids, sex alternation)
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta
from xftsim.mate import NMateAssignment, RandomMating


class TestNMateAssignmentValidation:
    def test_valid_construction(self):
        sm = SampleMeta(iid=np.arange(4), generation=1)
        ma = NMateAssignment(
            offspring_samples=sm,
            maternal_idx=np.array([0, 0, 1, 1]),
            paternal_idx=np.array([2, 2, 3, 3]),
        )
        assert ma.n_offspring == 4

    def test_maternal_length_mismatch(self):
        sm = SampleMeta(iid=np.arange(4), generation=1)
        with pytest.raises(ValueError, match="maternal_idx"):
            NMateAssignment(
                offspring_samples=sm,
                maternal_idx=np.array([0, 0]),
                paternal_idx=np.array([2, 2, 3, 3]),
            )

    def test_paternal_length_mismatch(self):
        sm = SampleMeta(iid=np.arange(4), generation=1)
        with pytest.raises(ValueError, match="paternal_idx"):
            NMateAssignment(
                offspring_samples=sm,
                maternal_idx=np.array([0, 0, 1, 1]),
                paternal_idx=np.array([2, 2]),
            )

    def test_negative_maternal_raises(self):
        sm = SampleMeta(iid=np.arange(2), generation=1)
        with pytest.raises(ValueError, match="negative"):
            NMateAssignment(
                offspring_samples=sm,
                maternal_idx=np.array([-1, 0]),
                paternal_idx=np.array([0, 1]),
            )

    def test_negative_paternal_raises(self):
        sm = SampleMeta(iid=np.arange(2), generation=1)
        with pytest.raises(ValueError, match="negative"):
            NMateAssignment(
                offspring_samples=sm,
                maternal_idx=np.array([0, 0]),
                paternal_idx=np.array([1, -1]),
            )

    def test_dtype_coercion(self):
        sm = SampleMeta(iid=np.arange(2), generation=1)
        ma = NMateAssignment(
            offspring_samples=sm,
            maternal_idx=np.array([0, 0], dtype=np.int32),
            paternal_idx=np.array([1, 1], dtype=np.float64),
        )
        assert ma.maternal_idx.dtype == np.int64
        assert ma.paternal_idx.dtype == np.int64

    def test_empty_offspring(self):
        sm = SampleMeta(iid=np.array([]), generation=1)
        ma = NMateAssignment(
            offspring_samples=sm,
            maternal_idx=np.array([], dtype=np.int64),
            paternal_idx=np.array([], dtype=np.int64),
        )
        assert ma.n_offspring == 0

    def test_repr(self):
        sm = SampleMeta(iid=np.arange(4), generation=2)
        ma = NMateAssignment(
            offspring_samples=sm,
            maternal_idx=np.array([0, 0, 1, 1]),
            paternal_idx=np.array([2, 2, 3, 3]),
        )
        r = repr(ma)
        assert 'NMateAssignment' in r
        assert 'n_offspring=4' in r
        assert 'generation=2' in r


class TestRandomMatingOffspringStructure:
    def test_offspring_fids_per_pair(self):
        """Each pair's offspring should share a FID."""
        sm = SampleMeta(iid=np.arange(20))
        mate = RandomMating(offspring_per_pair=3)
        rng = np.random.RandomState(42)
        assignment = mate.mate(sm, rng=rng)
        # Each group of 3 should have same FID
        fids = assignment.offspring_samples.fid
        for i in range(0, len(fids), 3):
            group = fids[i:i+3]
            assert len(np.unique(group)) == 1

    def test_offspring_sex_alternates(self):
        """Offspring sex should alternate 0, 1, 0, 1, ..."""
        sm = SampleMeta(iid=np.arange(20))
        mate = RandomMating(offspring_per_pair=4)
        rng = np.random.RandomState(42)
        assignment = mate.mate(sm, rng=rng)
        sex = assignment.offspring_samples.sex
        for i in range(len(sex)):
            assert sex[i] == i % 2

    def test_offspring_count(self):
        """Total offspring = n_pairs * offspring_per_pair."""
        sm = SampleMeta(iid=np.arange(20))
        mate = RandomMating(offspring_per_pair=2)
        rng = np.random.RandomState(42)
        assignment = mate.mate(sm, rng=rng)
        n_female = np.sum(sm.sex == 0)
        n_male = np.sum(sm.sex == 1)
        expected = min(n_female, n_male) * 2
        assert assignment.n_offspring == expected

    def test_maternal_idx_are_females(self):
        """Maternal indices should point to females."""
        sm = SampleMeta(iid=np.arange(20))
        mate = RandomMating(offspring_per_pair=2)
        rng = np.random.RandomState(42)
        assignment = mate.mate(sm, rng=rng)
        for idx in assignment.maternal_idx:
            assert sm.sex[idx] == 0

    def test_paternal_idx_are_males(self):
        """Paternal indices should point to males."""
        sm = SampleMeta(iid=np.arange(20))
        mate = RandomMating(offspring_per_pair=2)
        rng = np.random.RandomState(42)
        assignment = mate.mate(sm, rng=rng)
        for idx in assignment.paternal_idx:
            assert sm.sex[idx] == 1

    def test_generation_incremented(self):
        """Offspring generation should be parent_gen + 1."""
        sm = SampleMeta(iid=np.arange(20), generation=3)
        mate = RandomMating(offspring_per_pair=2)
        rng = np.random.RandomState(42)
        assignment = mate.mate(sm, rng=rng)
        assert assignment.offspring_samples.generation == 4

    def test_deterministic_with_same_seed(self):
        """Same seed should produce same assignment."""
        sm = SampleMeta(iid=np.arange(20))
        mate = RandomMating(offspring_per_pair=2)
        a1 = mate.mate(sm, rng=np.random.RandomState(42))
        a2 = mate.mate(sm, rng=np.random.RandomState(42))
        np.testing.assert_array_equal(a1.maternal_idx, a2.maternal_idx)
        np.testing.assert_array_equal(a1.paternal_idx, a2.paternal_idx)
