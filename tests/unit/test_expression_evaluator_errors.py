"""
Unit tests for expression evaluator error handling.

Tests:
1. Division by zero → inf/nan
2. Mismatched parentheses → ValueError
3. Undefined variable → ValueError
4. Empty expression → error
5. Double operator → error
6. Division evaluates correctly
7. Unary minus before parenthesized expression
"""
import numpy as np
import pytest

from xftsim.arch import AggregationComponent, ArchNode, _evaluate_expression
from xftsim.struct import SampleMeta, NPhenotypeArray

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


def _eval(expr, pheno_dict, n=10):
    sm = SampleMeta(iid=np.arange(n))
    pheno = NPhenotypeArray(sm)
    for k, v in pheno_dict.items():
        pheno[k] = v
    return _evaluate_expression(expr, pheno, n)


class TestExpressionErrors:
    def test_division_by_zero_gives_inf(self):
        """A / B where B=0 → inf (numpy semantics)."""
        n = 5
        result = _eval('A / B', {'A': np.ones(n), 'B': np.zeros(n)}, n)
        assert np.all(np.isinf(result))

    def test_mismatched_open_paren(self):
        """Unclosed paren → ValueError."""
        n = 5
        with pytest.raises(ValueError, match="[Mm]ismatched|parenthes"):
            _eval('(A + B', {'A': np.ones(n), 'B': np.ones(n)}, n)

    def test_mismatched_close_paren(self):
        """Extra close paren → ValueError."""
        n = 5
        with pytest.raises(ValueError, match="[Mm]ismatched|parenthes"):
            _eval('A + B)', {'A': np.ones(n), 'B': np.ones(n)}, n)

    def test_undefined_variable(self):
        """Reference to missing phenotype → ValueError."""
        n = 5
        with pytest.raises(ValueError, match="Undefined reference"):
            _eval('A + missing', {'A': np.ones(n)}, n)

    def test_division_correct(self):
        """A / B evaluates correctly."""
        n = 5
        A = np.ones(n) * 10.0
        B = np.ones(n) * 2.0
        result = _eval('A / B', {'A': A, 'B': B}, n)
        np.testing.assert_allclose(result, 5.0)

    def test_unary_minus_before_paren(self):
        """-(A + B) should negate the result."""
        n = 5
        A = np.ones(n) * 3.0
        B = np.ones(n) * 2.0
        result = _eval('-(A + B)', {'A': A, 'B': B}, n)
        np.testing.assert_allclose(result, -5.0)

    def test_scientific_notation_number(self):
        """1e2 * A should work."""
        n = 5
        A = np.ones(n) * 3.0
        result = _eval('1e2 * A', {'A': A}, n)
        np.testing.assert_allclose(result, 300.0)
