"""
Unit tests for CNoiseComponent constructor validation.

Tests:
1. Non-square cov → ValueError
2. Valid cov accepted
3. k property
4. 1x1 cov (scalar multivariate noise)
5. 3x3 cov
6. Non-matrix (1D array) → ValueError
7. repr
"""
import numpy as np
import pytest

from xftsim.arch import CNoiseComponent


class TestCNoiseConstructor:
    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square matrix"):
            CNoiseComponent(cov=np.array([[1.0, 0.5], [0.5, 1.0], [0.1, 0.1]]))

    def test_valid_cov(self):
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        comp = CNoiseComponent(cov=cov)
        np.testing.assert_array_equal(comp.cov, cov)

    def test_k_property(self):
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        assert CNoiseComponent(cov=cov).k == 2

    def test_1x1_cov(self):
        cov = np.array([[2.0]])
        comp = CNoiseComponent(cov=cov)
        assert comp.k == 1

    def test_3x3_cov(self):
        cov = np.eye(3)
        comp = CNoiseComponent(cov=cov)
        assert comp.k == 3

    def test_1d_array_raises(self):
        with pytest.raises(ValueError, match="square matrix"):
            CNoiseComponent(cov=np.array([1.0, 0.5]))

    def test_repr(self):
        cov = np.eye(2)
        r = repr(CNoiseComponent(cov=cov))
        assert 'CNoiseComponent' in r
        assert '(2, 2)' in r
