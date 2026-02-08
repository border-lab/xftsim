"""
Unit tests for expression evaluator unary minus error paths.

Tests:
1. Unary minus at end of expression raises ValueError
2. Trailing standalone '-' raises ValueError
3. Unary minus before ')' raises error
4. Unary minus followed by operator raises ValueError
"""
import numpy as np
import pytest

from xftsim.narch import _tokenize, _shunting_yard, _evaluate_expression
from xftsim.struct import SampleMeta, NPhenotypeArray


def _make_pheno(**kwargs):
    """Helper to make a NPhenotypeArray with named arrays."""
    n = None
    for v in kwargs.values():
        n = len(v)
        break
    if n is None:
        n = 5
    sm = SampleMeta(iid=np.arange(n))
    pheno = NPhenotypeArray(samples=sm)
    for k, v in kwargs.items():
        pheno[k] = np.asarray(v, dtype=np.float64)
    return pheno, n


class TestUnaryMinusErrors:
    def test_unary_minus_at_end(self):
        """Expression ending in unary minus should raise."""
        with pytest.raises(ValueError, match="Unexpected end"):
            _shunting_yard(_tokenize('A + -'))

    def test_trailing_unary_minus_standalone(self):
        """Standalone '-' as expression should raise."""
        with pytest.raises(ValueError, match="Unexpected end"):
            _shunting_yard(_tokenize('-'))

    def test_unary_minus_in_expression_with_operator(self):
        """Unary minus followed by another operator should raise."""
        # '-*' means unary minus then '*', which is unexpected
        with pytest.raises(ValueError):
            _shunting_yard(_tokenize('- *'))

    def test_valid_unary_minus_before_name(self):
        """Unary minus before a name should produce negation."""
        pheno, n = _make_pheno(A=np.array([1.0, 2.0, 3.0]))
        result = _evaluate_expression('-A', pheno, n)
        np.testing.assert_array_equal(result, [-1.0, -2.0, -3.0])

    def test_valid_unary_minus_before_number(self):
        """Unary minus before number should negate."""
        pheno, n = _make_pheno(A=np.array([1.0, 2.0, 3.0]))
        result = _evaluate_expression('A + -2', pheno, n)
        np.testing.assert_array_equal(result, [-1.0, 0.0, 1.0])

    def test_valid_unary_minus_in_parens(self):
        """-(A) should negate A."""
        pheno, n = _make_pheno(A=np.array([1.0, 2.0, 3.0]))
        result = _evaluate_expression('-(A)', pheno, n)
        np.testing.assert_array_equal(result, [-1.0, -2.0, -3.0])

    def test_stack_underflow(self):
        """Not enough operands for operator should raise."""
        pheno, n = _make_pheno(A=np.array([1.0]))
        with pytest.raises(ValueError, match="not enough operands"):
            _evaluate_expression('+ A', pheno, n)

    def test_stack_overflow(self):
        """Expression that leaves multiple values on stack should raise."""
        pheno, n = _make_pheno(A=np.array([1.0]), B=np.array([2.0]))
        # Two values, no operator between them would require special input
        # Direct RPN with two values and no operator
        with pytest.raises(ValueError, match="stack has"):
            _evaluate_expression('A B', pheno, n)
