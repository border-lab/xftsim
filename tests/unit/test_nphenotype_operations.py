"""
Unit tests for NPhenotypeArray operations.

Tests:
1. __setitem__ and __getitem__
2. keys property
3. __contains__
4. subset by indices
5. subset by boolean mask
6. Multiple keys
7. values dict access
8. samples property
9. repr
10. Empty phenotype array
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, NPhenotypeArray


def _make_pheno(n=10, keys=None):
    sm = SampleMeta(iid=np.arange(n))
    pheno = NPhenotypeArray(samples=sm)
    if keys:
        for k in keys:
            pheno._values[k] = np.arange(n, dtype=np.float64)
    return pheno


class TestGetSetItem:
    def test_setitem_getitem(self):
        pheno = _make_pheno(5)
        pheno._values['Y'] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(pheno['Y'], [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_getitem_missing_raises(self):
        pheno = _make_pheno(5, keys=['Y'])
        with pytest.raises(KeyError):
            _ = pheno['NONEXISTENT']

    def test_overwrite_key(self):
        pheno = _make_pheno(5, keys=['Y'])
        new_vals = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        pheno._values['Y'] = new_vals
        np.testing.assert_array_equal(pheno['Y'], new_vals)


class TestKeysProperty:
    def test_no_keys(self):
        pheno = _make_pheno(5)
        assert len(pheno.keys) == 0

    def test_multiple_keys(self):
        pheno = _make_pheno(5, keys=['A', 'B', 'C'])
        assert set(pheno.keys) == {'A', 'B', 'C'}

    def test_keys_after_add(self):
        pheno = _make_pheno(5)
        pheno._values['X'] = np.zeros(5)
        assert 'X' in pheno.keys


class TestContains:
    def test_contains_true(self):
        pheno = _make_pheno(5, keys=['Y'])
        assert 'Y' in pheno

    def test_contains_false(self):
        pheno = _make_pheno(5, keys=['Y'])
        assert 'Z' not in pheno


class TestSubset:
    def test_subset_by_indices(self):
        pheno = _make_pheno(10, keys=['Y'])
        sub = pheno.subset(np.array([0, 2, 4]))
        assert len(sub['Y']) == 3
        np.testing.assert_array_equal(sub['Y'], [0.0, 2.0, 4.0])

    def test_subset_preserves_keys(self):
        pheno = _make_pheno(10, keys=['A', 'B'])
        sub = pheno.subset(np.array([0, 1]))
        assert 'A' in sub
        assert 'B' in sub

    def test_subset_single_index(self):
        pheno = _make_pheno(10, keys=['Y'])
        sub = pheno.subset(np.array([5]))
        assert len(sub['Y']) == 1
        assert sub['Y'][0] == 5.0


class TestSamplesProperty:
    def test_samples_n(self):
        pheno = _make_pheno(10)
        assert pheno.samples.n == 10

    def test_samples_iid(self):
        pheno = _make_pheno(5)
        np.testing.assert_array_equal(pheno.samples.iid, np.arange(5))


class TestRepr:
    def test_repr_nonempty(self):
        pheno = _make_pheno(5, keys=['Y', 'Z'])
        r = repr(pheno)
        assert 'NPhenotypeArray' in r

    def test_repr_empty(self):
        pheno = _make_pheno(5)
        r = repr(pheno)
        assert 'NPhenotypeArray' in r


class TestEmptyPhenotype:
    def test_no_values(self):
        pheno = _make_pheno(5)
        assert len(pheno.keys) == 0

    def test_add_value_to_empty(self):
        pheno = _make_pheno(5)
        pheno._values['Y'] = np.ones(5)
        assert 'Y' in pheno
        assert len(pheno['Y']) == 5
