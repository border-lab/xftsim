"""
Unit tests for mating with unequal sex ratios and offspring count edge cases.

Tests:
1. Many females, few males → limited by males
2. Many males, few females → limited by females
3. offspring_per_pair=1 produces correct count
4. offspring_per_pair=3 produces correct count
5. Single pair (1F + 1M) with high offspring count
6. Sex assignment pattern: opp=2 gives 50/50 sex ratio
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta
from xftsim.mate import RandomMating


def _make_samples(n=20, sex=None):
    if sex is None:
        sex = np.tile([0, 1], n // 2)
    return SampleMeta(iid=np.arange(n), sex=sex)


class TestUnequalSexRatio:
    def test_more_females_limited_by_males(self):
        """8 females + 2 males → 2 pairs."""
        sex = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
        sm = _make_samples(10, sex=sex)
        mate = RandomMating(offspring_per_pair=2)
        result = mate.mate(sm, rng=np.random.RandomState(42))
        assert result.n_offspring == 4  # 2 pairs × 2

    def test_more_males_limited_by_females(self):
        """2 females + 8 males → 2 pairs."""
        sex = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
        sm = _make_samples(10, sex=sex)
        mate = RandomMating(offspring_per_pair=2)
        result = mate.mate(sm, rng=np.random.RandomState(42))
        assert result.n_offspring == 4  # 2 pairs × 2

    def test_single_pair_high_opp(self):
        """1 female + 1 male + opp=5 → 5 offspring."""
        sex = np.array([0, 1])
        sm = _make_samples(2, sex=sex)
        mate = RandomMating(offspring_per_pair=5)
        result = mate.mate(sm, rng=np.random.RandomState(42))
        assert result.n_offspring == 5


class TestOffspringCount:
    def test_opp_1(self):
        """offspring_per_pair=1 with 5 pairs → 5 offspring."""
        sm = _make_samples(10)  # 5F + 5M
        mate = RandomMating(offspring_per_pair=1)
        result = mate.mate(sm, rng=np.random.RandomState(42))
        assert result.n_offspring == 5

    def test_opp_3(self):
        """offspring_per_pair=3 with 5 pairs → 15 offspring."""
        sm = _make_samples(10)  # 5F + 5M
        mate = RandomMating(offspring_per_pair=3)
        result = mate.mate(sm, rng=np.random.RandomState(42))
        assert result.n_offspring == 15


class TestOffspringSexAssignment:
    def test_opp_2_balanced_sex(self):
        """offspring_per_pair=2 should produce 50/50 sex ratio."""
        sm = _make_samples(100)  # 50F + 50M
        mate = RandomMating(offspring_per_pair=2)
        result = mate.mate(sm, rng=np.random.RandomState(42))
        # sex_pattern = np.tile(np.arange(2) % 2, n_pairs) = [0,1,0,1,...]
        n_female = np.sum(result.offspring_samples.sex == 0)
        n_male = np.sum(result.offspring_samples.sex == 1)
        assert n_female == n_male, f"{n_female}F vs {n_male}M"

    def test_opp_4_balanced_sex(self):
        """offspring_per_pair=4 should produce 50/50 sex ratio."""
        sm = _make_samples(20)  # 10F + 10M
        mate = RandomMating(offspring_per_pair=4)
        result = mate.mate(sm, rng=np.random.RandomState(42))
        # sex_pattern = np.tile([0,1,2,3] % 2, n_pairs) = [0,1,0,1,...]
        n_female = np.sum(result.offspring_samples.sex == 0)
        n_male = np.sum(result.offspring_samples.sex == 1)
        assert n_female == n_male
