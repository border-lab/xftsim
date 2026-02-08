"""
Unit tests for RandomMating and LinearAssortativeMating edge cases.

Tests:
1. RandomMating: invalid offspring_per_pair, minimal population, sex alternation,
   unequal sex counts, large families, repr
2. NMateAssignment: validation, repr, n_offspring property
3. LinearAssortativeMating: invalid r, r=0 fallback, missing phenotypes fallback,
   disassortative mating (r<0), zero-variance component, repr
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, NPhenotypeArray
from xftsim.nmate import RandomMating, LinearAssortativeMating, NMateAssignment


class TestRandomMatingEdgeCases:
    def test_offspring_per_pair_zero_raises(self):
        """offspring_per_pair=0 should raise ValueError."""
        with pytest.raises(ValueError, match="offspring_per_pair"):
            RandomMating(offspring_per_pair=0)

    def test_offspring_per_pair_negative_raises(self):
        """offspring_per_pair < 0 should raise ValueError."""
        with pytest.raises(ValueError, match="offspring_per_pair"):
            RandomMating(offspring_per_pair=-1)

    def test_minimal_population(self):
        """1 female + 1 male should produce offspring_per_pair offspring."""
        sm = SampleMeta(iid=np.array([0, 1]), sex=np.array([0, 1]))
        mate = RandomMating(offspring_per_pair=3)
        result = mate.mate(sm, rng=np.random.RandomState(42))
        assert result.n_offspring == 3

    def test_all_same_sex_raises(self):
        """All female (no males) should raise ValueError."""
        sm = SampleMeta(iid=np.arange(5), sex=np.zeros(5, dtype=int))
        mate = RandomMating()
        with pytest.raises(ValueError, match="at least one"):
            mate.mate(sm, rng=np.random.RandomState(42))

    def test_all_male_raises(self):
        """All male (no females) should raise ValueError."""
        sm = SampleMeta(iid=np.arange(5), sex=np.ones(5, dtype=int))
        mate = RandomMating()
        with pytest.raises(ValueError, match="at least one"):
            mate.mate(sm, rng=np.random.RandomState(42))

    def test_sex_alternation_pattern(self):
        """Offspring sex should alternate within families."""
        sm = SampleMeta(iid=np.arange(10), sex=np.tile([0, 1], 5))
        mate = RandomMating(offspring_per_pair=4)
        result = mate.mate(sm, rng=np.random.RandomState(42))
        # Within each family of 4, sex should be [0, 1, 0, 1]
        for fam_id in np.unique(result.offspring_samples.fid):
            fam_mask = result.offspring_samples.fid == fam_id
            fam_sex = result.offspring_samples.sex[fam_mask]
            np.testing.assert_array_equal(fam_sex, [0, 1, 0, 1])

    def test_unequal_sex_counts(self):
        """More females than males: n_pairs = n_males."""
        sm = SampleMeta(
            iid=np.arange(8),
            sex=np.array([0, 0, 0, 0, 0, 1, 1, 1]),  # 5F, 3M
        )
        mate = RandomMating(offspring_per_pair=2)
        result = mate.mate(sm, rng=np.random.RandomState(42))
        assert result.n_offspring == 6  # 3 pairs * 2

    def test_generation_incremented(self):
        """Offspring generation should be parent generation + 1."""
        sm = SampleMeta(iid=np.arange(10), sex=np.tile([0, 1], 5), generation=5)
        mate = RandomMating()
        result = mate.mate(sm, rng=np.random.RandomState(42))
        assert result.offspring_samples.generation == 6

    def test_repr(self):
        """RandomMating repr should show offspring_per_pair."""
        mate = RandomMating(offspring_per_pair=3)
        r = repr(mate)
        assert '3' in r

    def test_large_offspring_per_pair(self):
        """Large family size should work."""
        sm = SampleMeta(iid=np.arange(4), sex=np.array([0, 0, 1, 1]))
        mate = RandomMating(offspring_per_pair=10)
        result = mate.mate(sm, rng=np.random.RandomState(42))
        assert result.n_offspring == 20  # 2 pairs * 10


class TestNMateAssignmentValidation:
    def test_maternal_idx_length_mismatch(self):
        """maternal_idx length != n should raise."""
        sm = SampleMeta(iid=np.arange(3))
        with pytest.raises(ValueError, match="maternal_idx length"):
            NMateAssignment(
                offspring_samples=sm,
                maternal_idx=np.array([0, 1], dtype=np.int64),
                paternal_idx=np.array([0, 1, 2], dtype=np.int64),
            )

    def test_paternal_idx_length_mismatch(self):
        """paternal_idx length != n should raise."""
        sm = SampleMeta(iid=np.arange(3))
        with pytest.raises(ValueError, match="paternal_idx length"):
            NMateAssignment(
                offspring_samples=sm,
                maternal_idx=np.array([0, 1, 2], dtype=np.int64),
                paternal_idx=np.array([0, 1], dtype=np.int64),
            )

    def test_negative_maternal_raises(self):
        """Negative maternal_idx should raise."""
        sm = SampleMeta(iid=np.arange(2))
        with pytest.raises(ValueError, match="negative"):
            NMateAssignment(
                offspring_samples=sm,
                maternal_idx=np.array([-1, 0], dtype=np.int64),
                paternal_idx=np.array([0, 1], dtype=np.int64),
            )

    def test_negative_paternal_raises(self):
        """Negative paternal_idx should raise."""
        sm = SampleMeta(iid=np.arange(2))
        with pytest.raises(ValueError, match="negative"):
            NMateAssignment(
                offspring_samples=sm,
                maternal_idx=np.array([0, 1], dtype=np.int64),
                paternal_idx=np.array([0, -1], dtype=np.int64),
            )

    def test_repr(self):
        """NMateAssignment repr should show n_offspring and generation."""
        sm = SampleMeta(iid=np.arange(4), generation=3)
        ma = NMateAssignment(
            offspring_samples=sm,
            maternal_idx=np.array([0, 0, 1, 1], dtype=np.int64),
            paternal_idx=np.array([2, 2, 3, 3], dtype=np.int64),
        )
        r = repr(ma)
        assert 'n_offspring=4' in r
        assert 'generation=3' in r

    def test_n_offspring_property(self):
        """n_offspring should equal n."""
        sm = SampleMeta(iid=np.arange(6))
        ma = NMateAssignment(
            offspring_samples=sm,
            maternal_idx=np.zeros(6, dtype=np.int64),
            paternal_idx=np.zeros(6, dtype=np.int64),
        )
        assert ma.n_offspring == 6


class TestLinearAssortativeMatingEdgeCases:
    def test_r_out_of_range_raises(self):
        """r >= 1 or r <= -1 should raise ValueError."""
        with pytest.raises(ValueError, match="r must be"):
            LinearAssortativeMating(component_names=['Y'], r=1.0)
        with pytest.raises(ValueError, match="r must be"):
            LinearAssortativeMating(component_names=['Y'], r=-1.0)

    def test_r_zero_fallback_to_random(self):
        """r=0 should behave like random mating."""
        sm = SampleMeta(iid=np.arange(10), sex=np.tile([0, 1], 5))
        pheno = NPhenotypeArray(samples=sm, values={'Y': np.random.randn(10)})
        mate = LinearAssortativeMating(component_names=['Y'], r=0.0, offspring_per_pair=2)
        result = mate.mate(sm, rng=np.random.RandomState(42), phenotypes=pheno)
        assert result.n_offspring == 10  # 5 pairs * 2

    def test_no_phenotypes_fallback_to_random(self):
        """phenotypes=None should fall back to random mating."""
        sm = SampleMeta(iid=np.arange(10), sex=np.tile([0, 1], 5))
        mate = LinearAssortativeMating(component_names=['Y'], r=0.5, offspring_per_pair=2)
        result = mate.mate(sm, rng=np.random.RandomState(42), phenotypes=None)
        assert result.n_offspring == 10

    def test_missing_component_names_still_works(self):
        """If none of the component_names exist in phenotypes, composite is zero."""
        sm = SampleMeta(iid=np.arange(10), sex=np.tile([0, 1], 5))
        pheno = NPhenotypeArray(samples=sm, values={'X': np.random.randn(10)})
        mate = LinearAssortativeMating(component_names=['Y', 'Z'], r=0.5)
        # Should still run (composite is all zeros, so it's essentially random)
        result = mate.mate(sm, rng=np.random.RandomState(42), phenotypes=pheno)
        assert result.n_offspring == 10

    def test_disassortative_mating(self):
        """r < 0 should negate scores for males."""
        sm = SampleMeta(iid=np.arange(20), sex=np.tile([0, 1], 10))
        vals = np.random.RandomState(42).randn(20)
        pheno = NPhenotypeArray(samples=sm, values={'Y': vals})
        mate = LinearAssortativeMating(component_names=['Y'], r=-0.5, offspring_per_pair=2)
        result = mate.mate(sm, rng=np.random.RandomState(42), phenotypes=pheno)
        assert result.n_offspring == 20

    def test_repr(self):
        """repr should show components, r, offspring_per_pair."""
        mate = LinearAssortativeMating(component_names=['Y', 'X'], r=0.5, offspring_per_pair=3)
        r = repr(mate)
        assert 'Y' in r
        assert '0.5' in r
        assert '3' in r

    def test_zero_variance_component(self):
        """Constant phenotype (zero variance) should not crash."""
        sm = SampleMeta(iid=np.arange(10), sex=np.tile([0, 1], 5))
        pheno = NPhenotypeArray(samples=sm, values={'Y': np.ones(10)})
        mate = LinearAssortativeMating(component_names=['Y'], r=0.5)
        result = mate.mate(sm, rng=np.random.RandomState(42), phenotypes=pheno)
        assert result.n_offspring == 10
