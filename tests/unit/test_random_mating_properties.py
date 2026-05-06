"""
Unit tests for RandomMating offspring properties.

Tests:
1. offspring_per_pair=1 produces correct count
2. offspring_per_pair=3 produces correct count
3. Sex alternates within family
4. FIDs are pair-based
5. Generation incremented
6. Unequal sex ratio uses min(n_female, n_male)
7. Deterministic with same seed
8. Different seeds produce different assignments
"""
import numpy as np
import pytest

from xftsim.mate import RandomMating
from xftsim.struct import SampleMeta


def _make_parents(n_female, n_male, generation=0):
    n = n_female + n_male
    sex = np.concatenate([np.zeros(n_female, dtype=np.int64),
                          np.ones(n_male, dtype=np.int64)])
    return SampleMeta(iid=np.arange(n), sex=sex, generation=generation)


class TestRandomMatingOffspring:
    def test_opp_1_count(self):
        parents = _make_parents(5, 5)
        rm = RandomMating(offspring_per_pair=1)
        ma = rm.mate(parents, rng=np.random.RandomState(42))
        assert ma.n_offspring == 5

    def test_opp_3_count(self):
        parents = _make_parents(4, 4)
        rm = RandomMating(offspring_per_pair=3)
        ma = rm.mate(parents, rng=np.random.RandomState(42))
        assert ma.n_offspring == 12

    def test_sex_alternates(self):
        """Offspring sex alternates within each family."""
        parents = _make_parents(3, 3)
        rm = RandomMating(offspring_per_pair=4)
        ma = rm.mate(parents, rng=np.random.RandomState(42))
        sex = ma.offspring_samples.sex
        for fam_start in range(0, ma.n_offspring, 4):
            family_sex = sex[fam_start:fam_start+4]
            np.testing.assert_array_equal(family_sex, [0, 1, 0, 1])

    def test_fids_pair_based(self):
        parents = _make_parents(3, 3)
        rm = RandomMating(offspring_per_pair=2)
        ma = rm.mate(parents, rng=np.random.RandomState(42))
        expected_fid = np.array([0, 0, 1, 1, 2, 2])
        np.testing.assert_array_equal(ma.offspring_samples.fid, expected_fid)

    def test_generation_incremented(self):
        parents = _make_parents(5, 5, generation=3)
        rm = RandomMating(offspring_per_pair=2)
        ma = rm.mate(parents, rng=np.random.RandomState(42))
        assert ma.offspring_samples.generation == 4

    def test_unequal_sex_ratio(self):
        parents = _make_parents(10, 3)
        rm = RandomMating(offspring_per_pair=2)
        ma = rm.mate(parents, rng=np.random.RandomState(42))
        assert ma.n_offspring == 6  # 3 pairs × 2

    def test_deterministic(self):
        parents = _make_parents(10, 10)
        rm = RandomMating(offspring_per_pair=2)
        ma1 = rm.mate(parents, rng=np.random.RandomState(42))
        ma2 = rm.mate(parents, rng=np.random.RandomState(42))
        np.testing.assert_array_equal(ma1.maternal_idx, ma2.maternal_idx)
        np.testing.assert_array_equal(ma1.paternal_idx, ma2.paternal_idx)

    def test_different_seeds_differ(self):
        parents = _make_parents(10, 10)
        rm = RandomMating(offspring_per_pair=2)
        ma1 = rm.mate(parents, rng=np.random.RandomState(42))
        ma2 = rm.mate(parents, rng=np.random.RandomState(99))
        assert not np.array_equal(ma1.maternal_idx, ma2.maternal_idx)
