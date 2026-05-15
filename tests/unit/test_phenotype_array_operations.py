"""
Unit tests for PhenotypeArray operations.

Tests:
1. Setting and getting values
2. __contains__ for key existence
3. Overwriting existing key (should warn)
4. keys property
5. subset by sample indices
6. Multiple keys stored independently
7. Values have correct dtype
8. n property
"""
import numpy as np
import pytest
import warnings

from xftsim.struct import SampleMeta, PhenotypeArray


def _make_pheno(n=10, keys=None, seed=42):
    rng = np.random.RandomState(seed)
    sm = SampleMeta(iid=np.arange(n))
    pheno = PhenotypeArray(samples=sm)
    if keys:
        for k in keys:
            pheno[k] = rng.normal(0, 1, n)
    return pheno


class TestPhenotypeSetGet:
    def test_set_and_get(self):
        pheno = _make_pheno(10)
        vals = np.arange(10, dtype=float)
        pheno['X'] = vals
        np.testing.assert_array_equal(pheno['X'], vals)

    def test_contains(self):
        pheno = _make_pheno(10, keys=['A', 'B'])
        assert 'A' in pheno
        assert 'B' in pheno
        assert 'C' not in pheno

    def test_keys(self):
        pheno = _make_pheno(10, keys=['A', 'B', 'C'])
        assert set(pheno.keys) == {'A', 'B', 'C'}

    def test_overwrite_warns(self):
        pheno = _make_pheno(10, keys=['X'])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pheno['X'] = np.zeros(10)
            assert any('overwriting' in str(warning.message).lower() or
                       'already' in str(warning.message).lower()
                       for warning in w), \
                "Expected warning about overwriting"


class TestPhenotypeSubset:
    def test_subset_by_indices(self):
        pheno = _make_pheno(10, keys=['A', 'B'])
        idx = np.array([0, 2, 5])
        sub = pheno.subset(idx)
        assert len(sub['A']) == 3
        np.testing.assert_array_equal(sub['A'], pheno['A'][idx])

    def test_subset_preserves_keys(self):
        pheno = _make_pheno(10, keys=['X', 'Y', 'Z'])
        idx = np.array([1, 3])
        sub = pheno.subset(idx)
        assert set(sub.keys) == {'X', 'Y', 'Z'}


class TestPhenotypeValues:
    def test_values_float64(self):
        pheno = _make_pheno(10)
        pheno['X'] = np.arange(10, dtype=float)
        assert pheno['X'].dtype == np.float64

    def test_multiple_keys_independent(self):
        pheno = _make_pheno(10)
        pheno['A'] = np.ones(10)
        pheno['B'] = np.zeros(10)
        assert np.all(pheno['A'] == 1.0)
        assert np.all(pheno['B'] == 0.0)

    def test_samples_n_property(self):
        pheno = _make_pheno(20)
        assert pheno.samples.n == 20

    def test_empty_phenotype(self):
        sm = SampleMeta(iid=np.arange(5))
        pheno = PhenotypeArray(samples=sm)
        assert len(pheno.keys) == 0
        assert pheno.samples.n == 5
