"""
Unit tests for NPhenotypeArray advanced usage.

Tests:
1. Initialize with values dict
2. keys property
3. __contains__
4. Setting value with 2D array raises
5. Setting value with scalar broadcasts raises
6. Empty phenotype array has no keys
7. Multiple keys stored and retrieved independently
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, NPhenotypeArray


class TestNPhenotypeArrayInit:
    def test_init_with_values(self):
        """Initialize NPhenotypeArray with a values dict."""
        sm = SampleMeta(iid=np.arange(5))
        vals = {'Y': np.ones(5), 'X': np.zeros(5)}
        pheno = NPhenotypeArray(sm, values=vals)
        assert 'Y' in pheno
        assert 'X' in pheno
        np.testing.assert_array_equal(pheno['Y'], np.ones(5))

    def test_empty_keys(self):
        """Empty NPhenotypeArray should have no keys."""
        sm = SampleMeta(iid=np.arange(5))
        pheno = NPhenotypeArray(sm)
        assert len(list(pheno.keys)) == 0

    def test_keys_after_add(self):
        """Keys should update after setting values."""
        sm = SampleMeta(iid=np.arange(5))
        pheno = NPhenotypeArray(sm)
        pheno['Y'] = np.ones(5)
        pheno['X'] = np.zeros(5)
        keys = list(pheno.keys)
        assert 'Y' in keys
        assert 'X' in keys
        assert len(keys) == 2


class TestNPhenotypeArrayContains:
    def test_contains_true(self):
        """__contains__ should return True for set keys."""
        sm = SampleMeta(iid=np.arange(3))
        pheno = NPhenotypeArray(sm)
        pheno['Y'] = np.zeros(3)
        assert 'Y' in pheno

    def test_contains_false(self):
        """__contains__ should return False for missing keys."""
        sm = SampleMeta(iid=np.arange(3))
        pheno = NPhenotypeArray(sm)
        assert 'Y' not in pheno


class TestNPhenotypeArraySetItem:
    def test_2d_array_raises(self):
        """Setting a 2D value should raise ValueError."""
        sm = SampleMeta(iid=np.arange(5))
        pheno = NPhenotypeArray(sm)
        with pytest.raises(ValueError, match="shape"):
            pheno['Y'] = np.ones((5, 2))

    def test_wrong_length_raises(self):
        """Value with wrong length should raise ValueError."""
        sm = SampleMeta(iid=np.arange(5))
        pheno = NPhenotypeArray(sm)
        with pytest.raises(ValueError, match="shape"):
            pheno['Y'] = np.ones(10)

    def test_scalar_coerced_raises(self):
        """Scalar value should raise (shape mismatch)."""
        sm = SampleMeta(iid=np.arange(5))
        pheno = NPhenotypeArray(sm)
        with pytest.raises(ValueError, match="shape"):
            pheno['Y'] = 5.0


class TestNPhenotypeArrayIndependence:
    def test_values_independent(self):
        """Setting one key should not affect another."""
        sm = SampleMeta(iid=np.arange(5))
        pheno = NPhenotypeArray(sm)
        pheno['Y'] = np.ones(5) * 3.0
        pheno['X'] = np.ones(5) * 7.0

        assert np.all(pheno['Y'] == 3.0)
        assert np.all(pheno['X'] == 7.0)

    def test_overwrite(self):
        """Overwriting a key should replace the value."""
        sm = SampleMeta(iid=np.arange(5))
        pheno = NPhenotypeArray(sm)
        pheno['Y'] = np.ones(5)
        pheno['Y'] = np.ones(5) * 99.0
        np.testing.assert_array_equal(pheno['Y'], 99.0)
