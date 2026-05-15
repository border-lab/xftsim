"""
Unit tests for LinearAssortativeMating edge cases and boundary conditions.

Tests:
1. r boundary: r=1 and r=-1 raise ValueError
2. r=0: falls back to RandomMating
3. phenotypes=None: falls back to RandomMating
4. Missing component names gracefully handled
5. Single component name
6. Multiple component names
7. Disassortative (r < 0) negates male scores
8. offspring_per_pair validation
9. repr
10. Deterministic with same seed
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, PhenotypeArray
from xftsim.mate import LinearAssortativeMating, MateAssignment


def _make_samples_and_phenotypes(n=50, seed=42):
    """Create SampleMeta and PhenotypeArray for testing."""
    sm = SampleMeta(iid=np.arange(n), generation=0)
    rng = np.random.RandomState(seed)
    pheno = PhenotypeArray(
        samples=sm,
        values={'Y': rng.normal(0, 1, n)},
    )
    return sm, pheno


class TestBoundaryR:
    def test_r_equals_one_raises(self):
        with pytest.raises(ValueError, match="r must be in"):
            LinearAssortativeMating(['Y'], r=1.0)

    def test_r_equals_neg_one_raises(self):
        with pytest.raises(ValueError, match="r must be in"):
            LinearAssortativeMating(['Y'], r=-1.0)

    def test_r_above_one_raises(self):
        with pytest.raises(ValueError, match="r must be in"):
            LinearAssortativeMating(['Y'], r=1.5)

    def test_r_below_neg_one_raises(self):
        with pytest.raises(ValueError, match="r must be in"):
            LinearAssortativeMating(['Y'], r=-1.5)

    def test_r_near_one_valid(self):
        lam = LinearAssortativeMating(['Y'], r=0.99)
        assert lam.r == 0.99

    def test_r_near_neg_one_valid(self):
        lam = LinearAssortativeMating(['Y'], r=-0.99)
        assert lam.r == -0.99


class TestFallbackToRandom:
    def test_r_zero_falls_back(self):
        sm, pheno = _make_samples_and_phenotypes()
        lam = LinearAssortativeMating(['Y'], r=0.0)
        rng = np.random.RandomState(42)
        assignment = lam.mate(sm, rng=rng, phenotypes=pheno)
        assert isinstance(assignment, MateAssignment)
        assert assignment.n_offspring > 0

    def test_no_phenotypes_falls_back(self):
        sm, _ = _make_samples_and_phenotypes()
        lam = LinearAssortativeMating(['Y'], r=0.5)
        rng = np.random.RandomState(42)
        assignment = lam.mate(sm, rng=rng, phenotypes=None)
        assert isinstance(assignment, MateAssignment)
        assert assignment.n_offspring > 0


class TestComponentNames:
    def test_missing_component_ignored(self):
        """If component name not in phenotypes, it contributes nothing."""
        sm, pheno = _make_samples_and_phenotypes()
        lam = LinearAssortativeMating(['NONEXISTENT'], r=0.5)
        rng = np.random.RandomState(42)
        # Should not raise, just no assortment
        assignment = lam.mate(sm, rng=rng, phenotypes=pheno)
        assert isinstance(assignment, MateAssignment)

    def test_single_component(self):
        sm, pheno = _make_samples_and_phenotypes()
        lam = LinearAssortativeMating(['Y'], r=0.5)
        rng = np.random.RandomState(42)
        assignment = lam.mate(sm, rng=rng, phenotypes=pheno)
        assert assignment.n_offspring > 0

    def test_multiple_components(self):
        sm = SampleMeta(iid=np.arange(50), generation=0)
        rng = np.random.RandomState(42)
        pheno = PhenotypeArray(
            samples=sm,
            values={
                'Y1': rng.normal(0, 1, 50),
                'Y2': rng.normal(0, 1, 50),
            },
        )
        lam = LinearAssortativeMating(['Y1', 'Y2'], r=0.5)
        assignment = lam.mate(sm, rng=np.random.RandomState(42), phenotypes=pheno)
        assert assignment.n_offspring > 0


class TestDisassortative:
    def test_negative_r_produces_assignment(self):
        sm, pheno = _make_samples_and_phenotypes()
        lam = LinearAssortativeMating(['Y'], r=-0.5)
        rng = np.random.RandomState(42)
        assignment = lam.mate(sm, rng=rng, phenotypes=pheno)
        assert isinstance(assignment, MateAssignment)
        assert assignment.n_offspring > 0


class TestOffspringPerPair:
    def test_zero_raises(self):
        with pytest.raises(ValueError, match="offspring_per_pair"):
            LinearAssortativeMating(['Y'], r=0.5, offspring_per_pair=0)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="offspring_per_pair"):
            LinearAssortativeMating(['Y'], r=0.5, offspring_per_pair=-1)

    def test_one_offspring(self):
        sm, pheno = _make_samples_and_phenotypes()
        lam = LinearAssortativeMating(['Y'], r=0.5, offspring_per_pair=1)
        rng = np.random.RandomState(42)
        assignment = lam.mate(sm, rng=rng, phenotypes=pheno)
        # Each pair produces exactly 1 offspring
        n_female = np.sum(sm.sex == 0)
        n_male = np.sum(sm.sex == 1)
        n_pairs = min(n_female, n_male)
        assert assignment.n_offspring == n_pairs

    def test_three_offspring(self):
        sm, pheno = _make_samples_and_phenotypes()
        lam = LinearAssortativeMating(['Y'], r=0.5, offspring_per_pair=3)
        rng = np.random.RandomState(42)
        assignment = lam.mate(sm, rng=rng, phenotypes=pheno)
        n_female = np.sum(sm.sex == 0)
        n_male = np.sum(sm.sex == 1)
        n_pairs = min(n_female, n_male)
        assert assignment.n_offspring == n_pairs * 3


class TestRepr:
    def test_repr_contains_r(self):
        lam = LinearAssortativeMating(['Y'], r=0.5)
        r = repr(lam)
        assert 'LinearAssortativeMating' in r
        assert '0.5' in r

    def test_repr_contains_components(self):
        lam = LinearAssortativeMating(['Y1', 'Y2'], r=0.3)
        r = repr(lam)
        assert 'Y1' in r
        assert 'Y2' in r


class TestDeterminism:
    def test_same_seed_same_assignment(self):
        sm, pheno = _make_samples_and_phenotypes()
        lam = LinearAssortativeMating(['Y'], r=0.5)
        a1 = lam.mate(sm, rng=np.random.RandomState(42), phenotypes=pheno)
        a2 = lam.mate(sm, rng=np.random.RandomState(42), phenotypes=pheno)
        np.testing.assert_array_equal(a1.maternal_idx, a2.maternal_idx)
        np.testing.assert_array_equal(a1.paternal_idx, a2.paternal_idx)

    def test_different_seed_different_assignment(self):
        sm, pheno = _make_samples_and_phenotypes()
        lam = LinearAssortativeMating(['Y'], r=0.5)
        a1 = lam.mate(sm, rng=np.random.RandomState(42), phenotypes=pheno)
        a2 = lam.mate(sm, rng=np.random.RandomState(99), phenotypes=pheno)
        # Very unlikely to be identical with different seeds
        assert not np.array_equal(a1.maternal_idx, a2.maternal_idx)


class TestNoMales:
    def test_all_female_raises(self):
        sm = SampleMeta(
            iid=np.arange(10),
            sex=np.zeros(10, dtype=int),  # all female
        )
        pheno = PhenotypeArray(
            samples=sm,
            values={'Y': np.ones(10)},
        )
        lam = LinearAssortativeMating(['Y'], r=0.5)
        with pytest.raises(ValueError, match="at least one female and one male"):
            lam.mate(sm, rng=np.random.RandomState(42), phenotypes=pheno)

    def test_all_male_raises(self):
        sm = SampleMeta(
            iid=np.arange(10),
            sex=np.ones(10, dtype=int),  # all male
        )
        pheno = PhenotypeArray(
            samples=sm,
            values={'Y': np.ones(10)},
        )
        lam = LinearAssortativeMating(['Y'], r=0.5)
        with pytest.raises(ValueError, match="at least one female and one male"):
            lam.mate(sm, rng=np.random.RandomState(42), phenotypes=pheno)
