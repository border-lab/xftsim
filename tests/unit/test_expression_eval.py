"""
Unit tests for the shunting-yard expression evaluator in narch.py.

Tests:
1. _tokenize: numbers, identifiers, operators, scientific notation, dotted names
2. _shunting_yard: operator precedence, parentheses, unary minus, mismatched parens
3. _evaluate_expression: simple add, mul, nested parens, scalar mul, unary minus,
   division, undefined reference, not enough operands, too many values
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, NPhenotypeArray
from xftsim.arch import _tokenize, _shunting_yard, _evaluate_expression


def _make_pheno(n=5, **kwargs):
    """Helper to create NPhenotypeArray with given values."""
    sm = SampleMeta(iid=np.arange(n))
    return NPhenotypeArray(samples=sm, values=kwargs)


class TestTokenize:
    def test_simple_expression(self):
        """Simple a + b."""
        tokens = _tokenize('a + b')
        assert tokens == [('NAME', 'a'), ('OP', '+'), ('NAME', 'b')]

    def test_numbers(self):
        """Integer and float literals."""
        tokens = _tokenize('1 + 2.5')
        assert tokens == [('NUM', 1.0), ('OP', '+'), ('NUM', 2.5)]

    def test_scientific_notation(self):
        """Scientific notation like 1e-3."""
        tokens = _tokenize('1e-3 * x')
        assert tokens[0] == ('NUM', 0.001)
        assert tokens[2] == ('NAME', 'x')

    def test_dotted_names(self):
        """Dotted names like Y.G."""
        tokens = _tokenize('Y.G + Y.E')
        assert ('NAME', 'Y.G') in tokens
        assert ('NAME', 'Y.E') in tokens

    def test_operators(self):
        """All four operators."""
        tokens = _tokenize('a + b - c * d / e')
        ops = [t[1] for t in tokens if t[0] == 'OP']
        assert ops == ['+', '-', '*', '/']

    def test_parentheses(self):
        """Parenthesized expressions."""
        tokens = _tokenize('(a + b) * c')
        assert tokens[0] == ('OP', '(')
        assert tokens[4] == ('OP', ')')


class TestShuntingYard:
    def test_precedence_mul_before_add(self):
        """a + b * c → [a, b, c, *, +]."""
        tokens = _tokenize('a + b * c')
        rpn = _shunting_yard(tokens)
        # Check that * comes before + in output
        ops = [t[1] for t in rpn if t[0] == 'OP']
        assert ops.index('*') < ops.index('+')

    def test_parentheses_override(self):
        """(a + b) * c → [a, b, +, c, *]."""
        tokens = _tokenize('(a + b) * c')
        rpn = _shunting_yard(tokens)
        ops = [t[1] for t in rpn if t[0] == 'OP']
        assert ops.index('+') < ops.index('*')

    def test_unary_minus_number(self):
        """Unary minus before number."""
        tokens = _tokenize('-3 + x')
        rpn = _shunting_yard(tokens)
        nums = [t[1] for t in rpn if t[0] == 'NUM']
        assert -3.0 in nums

    def test_unary_minus_name(self):
        """Unary minus before identifier: -x → [-1, x, *]."""
        tokens = _tokenize('-x')
        rpn = _shunting_yard(tokens)
        assert ('NUM', -1.0) in rpn
        assert ('OP', '*') in rpn

    def test_mismatched_open_paren(self):
        """Unclosed paren should raise."""
        tokens = _tokenize('(a + b')
        with pytest.raises(ValueError, match="parentheses"):
            _shunting_yard(tokens)

    def test_mismatched_close_paren(self):
        """Extra closing paren should raise."""
        tokens = _tokenize('a + b)')
        with pytest.raises(ValueError, match="parentheses"):
            _shunting_yard(tokens)


class TestEvaluateExpression:
    def test_simple_addition(self):
        """a + b should add element-wise."""
        pheno = _make_pheno(n=3, a=np.array([1.0, 2.0, 3.0]),
                           b=np.array([10.0, 20.0, 30.0]))
        result = _evaluate_expression('a + b', pheno, 3)
        np.testing.assert_array_equal(result, [11.0, 22.0, 33.0])

    def test_subtraction(self):
        """a - b should subtract element-wise."""
        pheno = _make_pheno(n=3, a=np.array([10.0, 20.0, 30.0]),
                           b=np.array([1.0, 2.0, 3.0]))
        result = _evaluate_expression('a - b', pheno, 3)
        np.testing.assert_array_equal(result, [9.0, 18.0, 27.0])

    def test_multiplication(self):
        """a * b should multiply element-wise."""
        pheno = _make_pheno(n=3, a=np.array([2.0, 3.0, 4.0]),
                           b=np.array([5.0, 6.0, 7.0]))
        result = _evaluate_expression('a * b', pheno, 3)
        np.testing.assert_array_equal(result, [10.0, 18.0, 28.0])

    def test_division(self):
        """a / b should divide element-wise."""
        pheno = _make_pheno(n=3, a=np.array([10.0, 20.0, 30.0]),
                           b=np.array([2.0, 4.0, 5.0]))
        result = _evaluate_expression('a / b', pheno, 3)
        np.testing.assert_allclose(result, [5.0, 5.0, 6.0])

    def test_scalar_multiplication(self):
        """2.0 * a should multiply by scalar."""
        pheno = _make_pheno(n=3, a=np.array([1.0, 2.0, 3.0]))
        result = _evaluate_expression('2.0 * a', pheno, 3)
        np.testing.assert_array_equal(result, [2.0, 4.0, 6.0])

    def test_nested_parentheses(self):
        """((a + b) * (c - d))."""
        pheno = _make_pheno(n=2,
                           a=np.array([1.0, 2.0]),
                           b=np.array([3.0, 4.0]),
                           c=np.array([10.0, 20.0]),
                           d=np.array([5.0, 10.0]))
        result = _evaluate_expression('(a + b) * (c - d)', pheno, 2)
        # (1+3)*(10-5) = 20, (2+4)*(20-10) = 60
        np.testing.assert_array_equal(result, [20.0, 60.0])

    def test_unary_minus(self):
        """Unary minus: -a + b."""
        pheno = _make_pheno(n=3, a=np.array([1.0, 2.0, 3.0]),
                           b=np.array([10.0, 20.0, 30.0]))
        result = _evaluate_expression('-a + b', pheno, 3)
        np.testing.assert_array_equal(result, [9.0, 18.0, 27.0])

    def test_undefined_reference(self):
        """Referencing undefined phenotype should raise ValueError."""
        pheno = _make_pheno(n=3, a=np.array([1.0, 2.0, 3.0]))
        with pytest.raises(ValueError, match="Undefined reference"):
            _evaluate_expression('a + missing', pheno, 3)

    def test_dotted_names(self):
        """Dotted names like Y.G + Y.E."""
        sm = SampleMeta(iid=np.arange(3))
        pheno = NPhenotypeArray(samples=sm, values={
            'Y.G': np.array([1.0, 2.0, 3.0]),
            'Y.E': np.array([0.5, 0.5, 0.5]),
        })
        result = _evaluate_expression('Y.G + Y.E', pheno, 3)
        np.testing.assert_array_equal(result, [1.5, 2.5, 3.5])

    def test_complex_expression(self):
        """Multiple operations: a + b * c - d."""
        pheno = _make_pheno(n=2,
                           a=np.array([1.0, 1.0]),
                           b=np.array([2.0, 2.0]),
                           c=np.array([3.0, 3.0]),
                           d=np.array([4.0, 4.0]))
        result = _evaluate_expression('a + b * c - d', pheno, 2)
        # 1 + 2*3 - 4 = 3
        np.testing.assert_array_equal(result, [3.0, 3.0])

    def test_single_name(self):
        """Single name should return that value."""
        pheno = _make_pheno(n=3, x=np.array([7.0, 8.0, 9.0]))
        result = _evaluate_expression('x', pheno, 3)
        np.testing.assert_array_equal(result, [7.0, 8.0, 9.0])
