"""
Unit tests for LinearAssortativeMating fallback/edge case behavior.

Tests:
1. r=0 falls back to random mating
2. phenotypes=None falls back to random mating
3. All component_names missing from phenotypes → zero composites, noisy pairing
4. Zero-variance phenotype → composites not normalized (sd=0 branch)
5. Single-sex population raises ValueError
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, PhenotypeArray
from xftsim.mate import LinearAssortativeMating, RandomMating


def _make_phenotypes(n, seed=42):
    """Create simple phenotypes with 'Y' key."""
    rng = np.random.RandomState(seed)
    sm = SampleMeta(iid=np.arange(n), sex=np.tile([0, 1], n // 2))
    pheno = PhenotypeArray(sm)
    pheno['Y'] = rng.normal(0, 1, n)
    return sm, pheno


class TestAssortativeFallbacks:
    def test_r_zero_falls_back_to_random(self):
        """r=0 should produce random pairing (delegates to RandomMating)."""
        sm, pheno = _make_phenotypes(100)
        am = LinearAssortativeMating(component_names=['Y'], r=0.0,
                                     offspring_per_pair=2)
        rng = np.random.RandomState(42)
        ma = am.mate(sm, phenotypes=pheno, rng=rng)
        # Should produce a valid MateAssignment
        assert ma.maternal_idx is not None
        assert len(ma.maternal_idx) > 0

    def test_phenotypes_none_falls_back_to_random(self):
        """phenotypes=None with r!=0 should fall back to random mating."""
        sm, _ = _make_phenotypes(100)
        am = LinearAssortativeMating(component_names=['Y'], r=0.5,
                                     offspring_per_pair=2)
        rng = np.random.RandomState(42)
        ma = am.mate(sm, phenotypes=None, rng=rng)
        assert ma.maternal_idx is not None
        assert len(ma.maternal_idx) > 0

    def test_missing_component_names_zero_composites(self):
        """All component names missing from phenotypes → zero composites."""
        sm, pheno = _make_phenotypes(100, seed=42)
        am = LinearAssortativeMating(
            component_names=['NONEXISTENT'],
            r=0.9, offspring_per_pair=2,
        )
        rng = np.random.RandomState(99)
        # Should still work — composites are all zeros, pairing is noise-driven
        ma = am.mate(sm, phenotypes=pheno, rng=rng)
        assert ma.maternal_idx is not None

    def test_zero_variance_phenotype(self):
        """Phenotype with zero variance → sd=0 branch, composites not normalized."""
        sm = SampleMeta(iid=np.arange(100), sex=np.tile([0, 1], 50))
        pheno = PhenotypeArray(sm)
        pheno['Y'] = np.ones(100) * 5.0  # constant, sd=0

        am = LinearAssortativeMating(
            component_names=['Y'], r=0.9, offspring_per_pair=2,
        )
        rng = np.random.RandomState(42)
        # Should still produce valid pairing
        ma = am.mate(sm, phenotypes=pheno, rng=rng)
        assert ma.maternal_idx is not None
        assert len(ma.maternal_idx) > 0

    def test_no_females_raises(self):
        """All-male population should raise ValueError."""
        sm = SampleMeta(iid=np.arange(10), sex=np.ones(10, dtype=int))
        pheno = PhenotypeArray(sm)
        pheno['Y'] = np.random.normal(0, 1, 10)

        am = LinearAssortativeMating(
            component_names=['Y'], r=0.5, offspring_per_pair=2,
        )
        rng = np.random.RandomState(42)
        with pytest.raises(ValueError, match="female.*male"):
            am.mate(sm, phenotypes=pheno, rng=rng)

    def test_no_males_raises(self):
        """All-female population should raise ValueError."""
        sm = SampleMeta(iid=np.arange(10), sex=np.zeros(10, dtype=int))
        pheno = PhenotypeArray(sm)
        pheno['Y'] = np.random.normal(0, 1, 10)

        am = LinearAssortativeMating(
            component_names=['Y'], r=0.5, offspring_per_pair=2,
        )
        rng = np.random.RandomState(42)
        with pytest.raises(ValueError, match="female.*male"):
            am.mate(sm, phenotypes=pheno, rng=rng)
