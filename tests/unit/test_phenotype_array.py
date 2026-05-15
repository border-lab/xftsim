"""
Unit tests for PhenotypeArray operations.

Tests:
1. Construction: empty, with values, wrong shape raises
2. __getitem__/__setitem__: set/get, overwrite warns, missing key
3. __contains__: present and absent keys
4. keys: returns all keys
5. subset: correct data, correct samples
6. __repr__: contains class name and keys
"""
import numpy as np
import pytest
import warnings

from xftsim.struct import SampleMeta, PhenotypeArray


def _make_samples(n=10):
    return SampleMeta(iid=np.arange(n))


class TestConstruction:
    def test_empty(self):
        """Empty PhenotypeArray should have no keys."""
        sm = _make_samples()
        p = PhenotypeArray(samples=sm)
        assert len(list(p.keys)) == 0

    def test_with_values(self):
        """Construct with initial values dict."""
        sm = _make_samples(10)
        vals = {'A': np.ones(10), 'B': np.zeros(10)}
        p = PhenotypeArray(samples=sm, values=vals)
        assert 'A' in p
        assert 'B' in p
        np.testing.assert_array_equal(p['A'], np.ones(10))

    def test_wrong_shape_raises(self):
        """Value with wrong shape should raise ValueError."""
        sm = _make_samples(10)
        with pytest.raises(ValueError, match="shape"):
            PhenotypeArray(samples=sm, values={'X': np.ones(5)})

    def test_values_coerced_to_float64(self):
        """Integer values should be coerced to float64."""
        sm = _make_samples(10)
        p = PhenotypeArray(samples=sm, values={'X': np.ones(10, dtype=np.int32)})
        assert p['X'].dtype == np.float64


class TestGetSetItem:
    def test_set_and_get(self):
        """Set a value and retrieve it."""
        sm = _make_samples(5)
        p = PhenotypeArray(samples=sm)
        p['Y'] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_array_equal(p['Y'], [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_overwrite_warns(self):
        """Overwriting an existing key should warn."""
        sm = _make_samples(5)
        p = PhenotypeArray(samples=sm)
        p['Y'] = np.ones(5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            p['Y'] = np.zeros(5)
            assert len(w) == 1
            assert "Overwriting" in str(w[0].message)

    def test_missing_key_raises(self):
        """Accessing a missing key should raise KeyError."""
        sm = _make_samples(5)
        p = PhenotypeArray(samples=sm)
        with pytest.raises(KeyError):
            _ = p['nonexistent']

    def test_wrong_shape_on_set(self):
        """Setting a value with wrong shape should raise."""
        sm = _make_samples(5)
        p = PhenotypeArray(samples=sm)
        with pytest.raises(ValueError, match="shape"):
            p['X'] = np.ones(10)


class TestContains:
    def test_present(self):
        """Key should be found after setting."""
        sm = _make_samples(5)
        p = PhenotypeArray(samples=sm)
        p['Y'] = np.ones(5)
        assert 'Y' in p

    def test_absent(self):
        """Missing key should not be found."""
        sm = _make_samples(5)
        p = PhenotypeArray(samples=sm)
        assert 'Y' not in p


class TestKeys:
    def test_keys_empty(self):
        """Empty array has no keys."""
        p = PhenotypeArray(samples=_make_samples())
        assert list(p.keys) == []

    def test_keys_populated(self):
        """Keys should match what was set."""
        sm = _make_samples(5)
        p = PhenotypeArray(samples=sm)
        p['A'] = np.ones(5)
        p['B'] = np.ones(5)
        assert set(p.keys) == {'A', 'B'}


class TestSubset:
    def test_subset_data(self):
        """Subset should select correct data rows."""
        sm = _make_samples(10)
        p = PhenotypeArray(samples=sm, values={'X': np.arange(10, dtype=float)})
        sub = p.subset(np.array([0, 2, 4]))
        np.testing.assert_array_equal(sub['X'], [0, 2, 4])

    def test_subset_samples(self):
        """Subset should update samples metadata."""
        sm = _make_samples(10)
        p = PhenotypeArray(samples=sm)
        p['X'] = np.ones(10)
        sub = p.subset(np.array([0, 1]))
        assert sub.samples.n == 2

    def test_subset_preserves_all_keys(self):
        """All keys should be in the subset."""
        sm = _make_samples(10)
        p = PhenotypeArray(samples=sm, values={
            'A': np.ones(10), 'B': np.zeros(10), 'C': np.arange(10, dtype=float),
        })
        sub = p.subset(np.array([0]))
        assert set(sub.keys) == {'A', 'B', 'C'}

    def test_subset_is_copy(self):
        """Subset should not share memory with original."""
        sm = _make_samples(10)
        p = PhenotypeArray(samples=sm, values={'X': np.arange(10, dtype=float)})
        sub = p.subset(np.array([0, 1, 2]))
        sub._values['X'][0] = 999
        assert p['X'][0] == 0  # original unchanged


class TestRepr:
    def test_repr_contains_class(self):
        """Repr should mention PhenotypeArray."""
        p = PhenotypeArray(samples=_make_samples(5))
        assert 'PhenotypeArray' in repr(p)

    def test_repr_shows_n(self):
        """Repr should show sample count."""
        p = PhenotypeArray(samples=_make_samples(5))
        assert 'n=5' in repr(p)

    def test_repr_shows_keys(self):
        """Repr should show key names."""
        sm = _make_samples(5)
        p = PhenotypeArray(samples=sm, values={'alpha': np.ones(5)})
        assert 'alpha' in repr(p)
