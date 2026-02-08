"""
Unit tests for TrioFilter edge cases.

Tests:
1. Gen 0 returns None
2. Missing pedigree returns None
3. Missing parent phenotype history returns None
4. Key in offspring but not parent → only matching keys extracted
5. Normal trio extraction with matching indices
6. TrioView has correct n_trios
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, NPhenotypeArray, PedigreeArray
from xftsim.nfilter import TrioFilter, TrioView


def _make_pheno(n, keys=None, seed=42):
    """Create NPhenotypeArray with given keys."""
    rng = np.random.RandomState(seed)
    sm = SampleMeta(iid=np.arange(n))
    pheno = NPhenotypeArray(samples=sm)
    for k in (keys or ['Y']):
        pheno[k] = rng.randn(n)
    return pheno


def _make_ped(n_offspring, n_parents, seed=42):
    """Create PedigreeArray for n_offspring from n_parents."""
    rng = np.random.RandomState(seed)
    sm = SampleMeta(iid=np.arange(n_offspring), generation=1)
    maternal = rng.randint(0, n_parents // 2, size=n_offspring)
    paternal = rng.randint(n_parents // 2, n_parents, size=n_offspring)
    return PedigreeArray(
        offspring_samples=sm,
        maternal_idx=maternal,
        paternal_idx=paternal,
        parent_n=n_parents,
    )


class TestTrioFilterReturnsNone:
    def test_gen0_returns_none(self):
        """At generation 0, no trios exist."""
        filt = TrioFilter()
        pheno = _make_pheno(10)
        result = filt.apply(0, {0: pheno}, {})
        assert result is None

    def test_missing_pedigree_returns_none(self):
        """If gen not in pedigree_history, return None."""
        filt = TrioFilter()
        pheno = _make_pheno(10)
        result = filt.apply(1, {0: pheno, 1: pheno}, {})
        assert result is None

    def test_missing_parent_phenotype_returns_none(self):
        """If parent gen not in phenotype_history, return None."""
        filt = TrioFilter()
        offspring_pheno = _make_pheno(10)
        ped = _make_ped(10, 10)
        # Only gen 1 phenotypes, no gen 0
        result = filt.apply(1, {1: offspring_pheno}, {1: ped})
        assert result is None


class TestTrioFilterExtraction:
    def test_normal_trio_extraction(self):
        """Normal trio extraction produces TrioView with correct structure."""
        filt = TrioFilter()
        parent_pheno = _make_pheno(20, keys=['Y', 'Y.G'], seed=42)
        offspring_pheno = _make_pheno(10, keys=['Y', 'Y.G'], seed=43)
        ped = _make_ped(10, 20, seed=42)

        result = filt.apply(
            1,
            {0: parent_pheno, 1: offspring_pheno},
            {1: ped},
        )
        assert isinstance(result, TrioView)
        assert result.n_trios == 10
        assert 'Y' in result.offspring_phenotypes
        assert 'Y.G' in result.offspring_phenotypes
        assert 'Y' in result.mother_phenotypes
        assert 'Y.G' in result.mother_phenotypes

    def test_key_in_offspring_not_parent(self):
        """If a key exists in offspring but not parent, it's in offspring but not mother/father."""
        filt = TrioFilter()
        parent_pheno = _make_pheno(20, keys=['Y'], seed=42)
        offspring_pheno = _make_pheno(10, keys=['Y', 'Y.extra'], seed=43)
        ped = _make_ped(10, 20, seed=42)

        result = filt.apply(
            1,
            {0: parent_pheno, 1: offspring_pheno},
            {1: ped},
        )
        assert 'Y.extra' in result.offspring_phenotypes
        # Y.extra not in parent_pheno → not in mother/father dicts
        assert 'Y.extra' not in result.mother_phenotypes
        assert 'Y.extra' not in result.father_phenotypes

    def test_mother_indices_correct(self):
        """Mother phenotype values should come from parent gen at maternal_idx."""
        filt = TrioFilter()
        parent_vals = np.arange(20, dtype=np.float64)
        parent_sm = SampleMeta(iid=np.arange(20))
        parent_pheno = NPhenotypeArray(samples=parent_sm, values={'Y': parent_vals})

        offspring_sm = SampleMeta(iid=np.arange(4), generation=1)
        offspring_pheno = NPhenotypeArray(
            samples=offspring_sm,
            values={'Y': np.array([0.0, 1.0, 2.0, 3.0])},
        )

        maternal_idx = np.array([0, 2, 4, 6])
        paternal_idx = np.array([10, 12, 14, 16])
        ped = PedigreeArray(
            offspring_samples=offspring_sm,
            maternal_idx=maternal_idx,
            paternal_idx=paternal_idx,
            parent_n=20,
        )

        result = filt.apply(1, {0: parent_pheno, 1: offspring_pheno}, {1: ped})
        np.testing.assert_array_equal(
            result.mother_phenotypes['Y'],
            parent_vals[maternal_idx],
        )
        np.testing.assert_array_equal(
            result.father_phenotypes['Y'],
            parent_vals[paternal_idx],
        )
