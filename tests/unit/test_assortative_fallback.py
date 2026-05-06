"""
Unit tests for LinearAssortativeMating fallback paths and edge cases.

Tests:
1. r=0 fallback to random mating
2. phenotypes=None fallback to random mating
3. All components missing from phenotypes falls back gracefully
4. Zero-variance phenotype component
5. Very high |r| near boundary
6. Single component name
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, NPhenotypeArray
from xftsim.mate import RandomMating, LinearAssortativeMating, NMateAssignment


def _make_samples_and_phenotypes(n=100, seed=42):
    rng = np.random.RandomState(seed)
    sex = np.tile([0, 1], (n + 1) // 2)[:n]
    sm = SampleMeta(iid=np.arange(n), sex=sex)
    pheno = NPhenotypeArray(samples=sm, values={
        'Y': rng.randn(n),
        'X': rng.randn(n),
    })
    return sm, pheno


class TestAssortativeFallbackToRandom:
    def test_r_zero_produces_valid_assignment(self):
        """r=0 should fall back to random mating, producing valid output."""
        sm, pheno = _make_samples_and_phenotypes()
        mate = LinearAssortativeMating(component_names=['Y'], r=0.0, offspring_per_pair=2)
        rng = np.random.RandomState(42)
        result = mate.mate(sm, rng=rng, phenotypes=pheno)
        assert isinstance(result, NMateAssignment)
        assert result.offspring_samples.n > 0

    def test_phenotypes_none_produces_valid_assignment(self):
        """phenotypes=None should fall back to random mating."""
        sm, _ = _make_samples_and_phenotypes()
        mate = LinearAssortativeMating(component_names=['Y'], r=0.5, offspring_per_pair=2)
        rng = np.random.RandomState(42)
        result = mate.mate(sm, rng=rng, phenotypes=None)
        assert isinstance(result, NMateAssignment)
        assert result.offspring_samples.n > 0


class TestAssortativeComponentHandling:
    def test_all_components_missing(self):
        """All component_names missing from phenotypes → composites are zero."""
        sm, pheno = _make_samples_and_phenotypes()
        mate = LinearAssortativeMating(
            component_names=['nonexistent1', 'nonexistent2'],
            r=0.5, offspring_per_pair=2,
        )
        rng = np.random.RandomState(42)
        # Should still produce valid output — score will be mostly noise
        result = mate.mate(sm, rng=rng, phenotypes=pheno)
        assert isinstance(result, NMateAssignment)
        assert result.offspring_samples.n > 0

    def test_some_components_missing(self):
        """Some component_names present, some missing → uses what's available."""
        sm, pheno = _make_samples_and_phenotypes()
        mate = LinearAssortativeMating(
            component_names=['Y', 'nonexistent'],
            r=0.5, offspring_per_pair=2,
        )
        rng = np.random.RandomState(42)
        result = mate.mate(sm, rng=rng, phenotypes=pheno)
        assert isinstance(result, NMateAssignment)
        assert result.offspring_samples.n > 0

    def test_zero_variance_component(self):
        """Component with zero variance should not crash (sd=0 branch)."""
        n = 100
        sex = np.tile([0, 1], (n + 1) // 2)[:n]
        sm = SampleMeta(iid=np.arange(n), sex=sex)
        # Create constant phenotype — zero variance
        pheno = NPhenotypeArray(samples=sm, values={
            'Y': np.ones(n) * 5.0,  # constant, sd=0
        })
        mate = LinearAssortativeMating(
            component_names=['Y'], r=0.5, offspring_per_pair=2,
        )
        rng = np.random.RandomState(42)
        result = mate.mate(sm, rng=rng, phenotypes=pheno)
        assert isinstance(result, NMateAssignment)
        assert result.offspring_samples.n > 0


class TestAssortativeHighR:
    def test_high_positive_r(self):
        """r close to 1 should produce valid assignment."""
        sm, pheno = _make_samples_and_phenotypes()
        mate = LinearAssortativeMating(
            component_names=['Y'], r=0.999, offspring_per_pair=2,
        )
        rng = np.random.RandomState(42)
        result = mate.mate(sm, rng=rng, phenotypes=pheno)
        assert isinstance(result, NMateAssignment)
        assert result.offspring_samples.n > 0

    def test_high_negative_r(self):
        """r close to -1 should produce valid assignment."""
        sm, pheno = _make_samples_and_phenotypes()
        mate = LinearAssortativeMating(
            component_names=['Y'], r=-0.999, offspring_per_pair=2,
        )
        rng = np.random.RandomState(42)
        result = mate.mate(sm, rng=rng, phenotypes=pheno)
        assert isinstance(result, NMateAssignment)
        assert result.offspring_samples.n > 0


class TestAssortativeMultiComponent:
    def test_two_components(self):
        """Assortative mating with 2 component names."""
        sm, pheno = _make_samples_and_phenotypes()
        mate = LinearAssortativeMating(
            component_names=['Y', 'X'], r=0.5, offspring_per_pair=2,
        )
        rng = np.random.RandomState(42)
        result = mate.mate(sm, rng=rng, phenotypes=pheno)
        assert isinstance(result, NMateAssignment)

    def test_offspring_per_pair_one(self):
        """offspring_per_pair=1 should produce half as many offspring."""
        sm, pheno = _make_samples_and_phenotypes()
        mate = LinearAssortativeMating(
            component_names=['Y'], r=0.5, offspring_per_pair=1,
        )
        rng = np.random.RandomState(42)
        result = mate.mate(sm, rng=rng, phenotypes=pheno)
        n_pairs = min(np.sum(sm.sex == 0), np.sum(sm.sex == 1))
        assert result.offspring_samples.n == n_pairs

    def test_offspring_per_pair_four(self):
        """offspring_per_pair=4 should produce 4x as many offspring per pair."""
        sm, pheno = _make_samples_and_phenotypes()
        mate = LinearAssortativeMating(
            component_names=['Y'], r=0.5, offspring_per_pair=4,
        )
        rng = np.random.RandomState(42)
        result = mate.mate(sm, rng=rng, phenotypes=pheno)
        n_pairs = min(np.sum(sm.sex == 0), np.sum(sm.sex == 1))
        assert result.offspring_samples.n == n_pairs * 4


class TestAssortativeNumerical:
    def test_positive_r_produces_positive_correlation(self):
        """r > 0 should produce positive spouse correlation (average over seeds)."""
        n = 200
        cors = []
        for seed in range(10):
            rng = np.random.RandomState(seed)
            sex = np.tile([0, 1], n // 2)
            sm = SampleMeta(iid=np.arange(n), sex=sex)
            values = rng.randn(n)
            pheno = NPhenotypeArray(samples=sm, values={'Y': values})
            mate = LinearAssortativeMating(
                component_names=['Y'], r=0.8, offspring_per_pair=2,
            )
            ma = mate.mate(sm, rng=rng, phenotypes=pheno)
            # Compute spouse correlation
            mom_vals = values[ma.maternal_idx[::2]]
            dad_vals = values[ma.paternal_idx[::2]]
            if len(mom_vals) > 2:
                cors.append(np.corrcoef(mom_vals, dad_vals)[0, 1])
        assert np.mean(cors) > 0.0

    def test_negative_r_produces_negative_correlation(self):
        """r < 0 should produce negative spouse correlation (average over seeds)."""
        n = 200
        cors = []
        for seed in range(10):
            rng = np.random.RandomState(seed)
            sex = np.tile([0, 1], n // 2)
            sm = SampleMeta(iid=np.arange(n), sex=sex)
            values = rng.randn(n)
            pheno = NPhenotypeArray(samples=sm, values={'Y': values})
            mate = LinearAssortativeMating(
                component_names=['Y'], r=-0.8, offspring_per_pair=2,
            )
            ma = mate.mate(sm, rng=rng, phenotypes=pheno)
            mom_vals = values[ma.maternal_idx[::2]]
            dad_vals = values[ma.paternal_idx[::2]]
            if len(mom_vals) > 2:
                cors.append(np.corrcoef(mom_vals, dad_vals)[0, 1])
        assert np.mean(cors) < 0.0
