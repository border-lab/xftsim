"""
Unit tests for the shunting-yard expression evaluator in narch.py.

Tests:
1. _tokenize: numbers, identifiers, operators, scientific notation
2. _shunting_yard: basic operators, precedence, parentheses, unary minus
3. _evaluate_expression: arithmetic, variable lookup, error paths
4. AggregationComponent._extract_names: deduplication, dotted names
"""
import numpy as np
import pytest

from xftsim.arch import (
    _tokenize, _shunting_yard, _evaluate_expression,
    AggregationComponent,
)
from xftsim.struct import SampleMeta, NPhenotypeArray


def _make_pheno(n=5, **values):
    sm = SampleMeta(iid=np.arange(n))
    return NPhenotypeArray(samples=sm, values=values)


class TestTokenize:
    def test_simple_integer(self):
        tokens = _tokenize("42")
        assert tokens == [('NUM', 42.0)]

    def test_simple_float(self):
        tokens = _tokenize("3.14")
        assert tokens == [('NUM', 3.14)]

    def test_scientific_notation(self):
        tokens = _tokenize("1e-3")
        assert len(tokens) == 1
        assert tokens[0][0] == 'NUM'
        assert abs(tokens[0][1] - 0.001) < 1e-10

    def test_scientific_positive_exponent(self):
        tokens = _tokenize("2.5E+2")
        assert len(tokens) == 1
        assert abs(tokens[0][1] - 250.0) < 1e-10

    def test_identifier(self):
        tokens = _tokenize("height")
        assert tokens == [('NAME', 'height')]

    def test_dotted_identifier(self):
        tokens = _tokenize("height.G")
        assert tokens == [('NAME', 'height.G')]

    def test_operators(self):
        tokens = _tokenize("a + b * c")
        assert len(tokens) == 5
        assert tokens[1] == ('OP', '+')
        assert tokens[3] == ('OP', '*')

    def test_parentheses(self):
        tokens = _tokenize("(a + b)")
        assert tokens[0] == ('OP', '(')
        assert tokens[-1] == ('OP', ')')

    def test_complex_expression(self):
        tokens = _tokenize("0.5 * Y.G + Y.E - 0.1 * Y.VT")
        names = [t[1] for t in tokens if t[0] == 'NAME']
        assert 'Y.G' in names
        assert 'Y.E' in names
        assert 'Y.VT' in names


class TestShuntingYard:
    def test_addition_precedence(self):
        tokens = _tokenize("a + b")
        rpn = _shunting_yard(tokens)
        # Should be: a b +
        assert rpn[-1] == ('OP', '+')

    def test_multiplication_precedence(self):
        tokens = _tokenize("a + b * c")
        rpn = _shunting_yard(tokens)
        # Should be: a b c * +
        ops = [t for t in rpn if t[0] == 'OP']
        assert ops[0] == ('OP', '*')  # * applied first
        assert ops[1] == ('OP', '+')

    def test_parentheses_override(self):
        tokens = _tokenize("(a + b) * c")
        rpn = _shunting_yard(tokens)
        ops = [t for t in rpn if t[0] == 'OP']
        assert ops[0] == ('OP', '+')  # + applied first due to parens
        assert ops[1] == ('OP', '*')

    def test_unary_minus_number(self):
        tokens = _tokenize("-3")
        rpn = _shunting_yard(tokens)
        assert rpn == [('NUM', -3.0)]

    def test_unary_minus_identifier(self):
        tokens = _tokenize("-X")
        rpn = _shunting_yard(tokens)
        # Should produce: -1 X *
        assert ('NUM', -1.0) in rpn
        assert ('NAME', 'X') in rpn
        assert ('OP', '*') in rpn

    def test_unary_minus_after_open_paren(self):
        tokens = _tokenize("(-X)")
        rpn = _shunting_yard(tokens)
        names = [t for t in rpn if t[0] == 'NAME']
        assert names == [('NAME', 'X')]

    def test_unary_minus_after_operator(self):
        tokens = _tokenize("A + -B")
        rpn = _shunting_yard(tokens)
        names = [t[1] for t in rpn if t[0] == 'NAME']
        assert 'A' in names
        assert 'B' in names

    def test_mismatched_close_paren(self):
        tokens = _tokenize("a + b)")
        with pytest.raises(ValueError, match="Mismatched"):
            _shunting_yard(tokens)

    def test_mismatched_open_paren(self):
        tokens = _tokenize("(a + b")
        with pytest.raises(ValueError, match="Mismatched"):
            _shunting_yard(tokens)

    def test_unary_minus_end_of_expression(self):
        tokens = [('OP', '-')]
        with pytest.raises(ValueError, match="Unexpected end"):
            _shunting_yard(tokens)

    def test_nested_parentheses(self):
        tokens = _tokenize("((a + b))")
        rpn = _shunting_yard(tokens)
        assert rpn == [('NAME', 'a'), ('NAME', 'b'), ('OP', '+')]


class TestEvaluateExpression:
    def test_simple_addition(self):
        pheno = _make_pheno(
            n=3,
            X=np.array([1.0, 2.0, 3.0]),
            Y=np.array([4.0, 5.0, 6.0]),
        )
        result = _evaluate_expression("X + Y", pheno, 3)
        np.testing.assert_array_equal(result, [5.0, 7.0, 9.0])

    def test_subtraction(self):
        pheno = _make_pheno(
            n=3,
            X=np.array([10.0, 20.0, 30.0]),
            Y=np.array([1.0, 2.0, 3.0]),
        )
        result = _evaluate_expression("X - Y", pheno, 3)
        np.testing.assert_array_equal(result, [9.0, 18.0, 27.0])

    def test_multiplication(self):
        pheno = _make_pheno(
            n=3,
            X=np.array([1.0, 2.0, 3.0]),
        )
        result = _evaluate_expression("2 * X", pheno, 3)
        np.testing.assert_array_equal(result, [2.0, 4.0, 6.0])

    def test_division(self):
        pheno = _make_pheno(
            n=3,
            X=np.array([4.0, 6.0, 8.0]),
        )
        result = _evaluate_expression("X / 2", pheno, 3)
        np.testing.assert_array_equal(result, [2.0, 3.0, 4.0])

    def test_weighted_sum(self):
        pheno = _make_pheno(
            n=3,
            A=np.array([1.0, 0.0, 0.0]),
            B=np.array([0.0, 1.0, 0.0]),
        )
        result = _evaluate_expression("0.5 * A + 0.3 * B", pheno, 3)
        np.testing.assert_allclose(result, [0.5, 0.3, 0.0])

    def test_parenthesized_expression(self):
        pheno = _make_pheno(
            n=3,
            X=np.array([2.0, 4.0, 6.0]),
            Y=np.array([1.0, 1.0, 1.0]),
        )
        result = _evaluate_expression("(X + Y) * 2", pheno, 3)
        np.testing.assert_array_equal(result, [6.0, 10.0, 14.0])

    def test_unary_minus(self):
        pheno = _make_pheno(
            n=3,
            X=np.array([1.0, 2.0, 3.0]),
        )
        result = _evaluate_expression("-X", pheno, 3)
        np.testing.assert_array_equal(result, [-1.0, -2.0, -3.0])

    def test_constant_expression(self):
        pheno = _make_pheno(n=3)
        result = _evaluate_expression("42", pheno, 3)
        np.testing.assert_array_equal(result, [42.0, 42.0, 42.0])

    def test_undefined_reference_raises(self):
        pheno = _make_pheno(n=3)
        with pytest.raises(ValueError, match="Undefined reference"):
            _evaluate_expression("MISSING", pheno, 3)

    def test_scientific_notation_constant(self):
        pheno = _make_pheno(
            n=3,
            X=np.array([1.0, 1.0, 1.0]),
        )
        result = _evaluate_expression("1e-3 * X", pheno, 3)
        np.testing.assert_allclose(result, [0.001, 0.001, 0.001])

    def test_complex_nested(self):
        pheno = _make_pheno(
            n=2,
            A=np.array([1.0, 2.0]),
            B=np.array([3.0, 4.0]),
            C=np.array([5.0, 6.0]),
        )
        result = _evaluate_expression("(A + B) * C - 2", pheno, 2)
        np.testing.assert_array_equal(result, [(1+3)*5 - 2, (2+4)*6 - 2])

    def test_dotted_names(self):
        pheno = _make_pheno(
            n=2,
            **{'Y.G': np.array([1.0, 2.0]), 'Y.E': np.array([3.0, 4.0])},
        )
        result = _evaluate_expression("Y.G + Y.E", pheno, 2)
        np.testing.assert_array_equal(result, [4.0, 6.0])


class TestExtractNames:
    def test_simple(self):
        names = AggregationComponent._extract_names("X + Y")
        assert names == ['X', 'Y']

    def test_dotted(self):
        names = AggregationComponent._extract_names("Y.G + Y.E")
        assert names == ['Y.G', 'Y.E']

    def test_deduplication(self):
        names = AggregationComponent._extract_names("X + X")
        assert names == ['X']

    def test_with_constants(self):
        names = AggregationComponent._extract_names("0.5 * X + 0.3 * Y")
        assert names == ['X', 'Y']

    def test_complex(self):
        names = AggregationComponent._extract_names("(A + B) * C - D")
        assert names == ['A', 'B', 'C', 'D']
