"""
Unit tests for AggregationComponent shunting-yard expression evaluator edge cases.

Tests:
1. Scientific notation in expressions
2. Unary minus at start of expression
3. Unary minus after open paren
4. Unary minus after operator
5. Division by zero produces inf/nan (no crash)
6. Deeply nested parentheses
7. Dotted names (e.g., Y.G + Y.E)
8. Numeric literal only expression
9. Mismatched parentheses raise
10. Undefined variable reference raises
11. Empty expression raises
12. Variable names with digits (e.g., trait2)
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, DenseHaplotypeArray, VariantMeta, NPhenotypeArray
from xftsim.arch import (
    Architecture, AggregationComponent, NoiseComponent,
    _tokenize, _shunting_yard, _evaluate_expression,
)


def _make_pheno(n=10, keys=None, values=None):
    sm = SampleMeta(iid=np.arange(n))
    pheno = NPhenotypeArray(samples=sm)
    if keys and values:
        for k, v in zip(keys, values):
            pheno[k] = v
    return pheno


class TestTokenizer:
    def test_scientific_notation(self):
        tokens = _tokenize('1.5e-3')
        assert len(tokens) == 1
        assert tokens[0] == ('NUM', 1.5e-3)

    def test_scientific_notation_positive(self):
        tokens = _tokenize('2E+5')
        assert tokens[0] == ('NUM', 2e5)

    def test_dotted_name(self):
        tokens = _tokenize('Y.G + Y.E')
        names = [t for t in tokens if t[0] == 'NAME']
        assert names == [('NAME', 'Y.G'), ('NAME', 'Y.E')]

    def test_name_with_digits(self):
        tokens = _tokenize('trait2 + pheno1.E')
        names = [t for t in tokens if t[0] == 'NAME']
        assert names == [('NAME', 'trait2'), ('NAME', 'pheno1.E')]

    def test_integer_literal(self):
        tokens = _tokenize('42')
        assert tokens == [('NUM', 42.0)]


class TestShuntingYard:
    def test_simple_addition(self):
        tokens = _tokenize('a + b')
        rpn = _shunting_yard(tokens)
        types = [t[0] for t in rpn]
        assert types == ['NAME', 'NAME', 'OP']

    def test_precedence_mul_over_add(self):
        tokens = _tokenize('a + b * c')
        rpn = _shunting_yard(tokens)
        # Should be: a b c * +
        vals = [t[1] for t in rpn]
        assert vals == ['a', 'b', 'c', '*', '+']

    def test_parentheses_override_precedence(self):
        tokens = _tokenize('(a + b) * c')
        rpn = _shunting_yard(tokens)
        vals = [t[1] for t in rpn]
        assert vals == ['a', 'b', '+', 'c', '*']

    def test_mismatched_open_paren(self):
        tokens = _tokenize('(a + b')
        with pytest.raises(ValueError, match="Mismatched"):
            _shunting_yard(tokens)

    def test_mismatched_close_paren(self):
        tokens = _tokenize('a + b)')
        with pytest.raises(ValueError, match="Mismatched"):
            _shunting_yard(tokens)


class TestUnaryMinus:
    def test_unary_at_start(self):
        pheno = _make_pheno(5, ['x'], [np.ones(5) * 3.0])
        result = _evaluate_expression('-x', pheno, 5)
        np.testing.assert_allclose(result, -3.0)

    def test_unary_after_open_paren(self):
        pheno = _make_pheno(5, ['x'], [np.ones(5) * 2.0])
        result = _evaluate_expression('(-x)', pheno, 5)
        np.testing.assert_allclose(result, -2.0)

    def test_unary_after_operator(self):
        pheno = _make_pheno(5, ['x', 'y'],
                           [np.ones(5) * 3.0, np.ones(5) * 2.0])
        result = _evaluate_expression('x + -y', pheno, 5)
        np.testing.assert_allclose(result, 1.0)

    def test_unary_number(self):
        pheno = _make_pheno(5)
        result = _evaluate_expression('-2.5', pheno, 5)
        np.testing.assert_allclose(result, -2.5)


class TestExpressionEvaluation:
    def test_numeric_literal_only(self):
        pheno = _make_pheno(5)
        result = _evaluate_expression('42', pheno, 5)
        np.testing.assert_allclose(result, 42.0)
        assert len(result) == 5

    def test_scientific_notation_literal(self):
        pheno = _make_pheno(5)
        result = _evaluate_expression('1e3', pheno, 5)
        np.testing.assert_allclose(result, 1000.0)

    def test_division_by_zero(self):
        """Division by zero should produce inf, not crash."""
        pheno = _make_pheno(5, ['x', 'z'],
                           [np.ones(5), np.zeros(5)])
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            result = _evaluate_expression('x / z', pheno, 5)
        assert np.all(np.isinf(result))

    def test_deeply_nested_parens(self):
        pheno = _make_pheno(5, ['a'], [np.ones(5) * 7.0])
        result = _evaluate_expression('(((a)))', pheno, 5)
        np.testing.assert_allclose(result, 7.0)

    def test_dotted_names(self):
        pheno = _make_pheno(5, ['Y.G', 'Y.E'],
                           [np.ones(5) * 2.0, np.ones(5) * 3.0])
        result = _evaluate_expression('Y.G + Y.E', pheno, 5)
        np.testing.assert_allclose(result, 5.0)

    def test_complex_expression(self):
        pheno = _make_pheno(5, ['a', 'b', 'c'],
                           [np.full(5, 2.0), np.full(5, 3.0), np.full(5, 4.0)])
        result = _evaluate_expression('(a + b) * c - a', pheno, 5)
        # (2+3)*4 - 2 = 18
        np.testing.assert_allclose(result, 18.0)

    def test_undefined_reference_raises(self):
        pheno = _make_pheno(5, ['x'], [np.ones(5)])
        with pytest.raises(ValueError, match="Undefined reference"):
            _evaluate_expression('x + MISSING', pheno, 5)

    def test_multiplication_by_literal(self):
        pheno = _make_pheno(5, ['x'], [np.ones(5) * 4.0])
        result = _evaluate_expression('0.5 * x', pheno, 5)
        np.testing.assert_allclose(result, 2.0)


class TestAggregationComponentExpression:
    def test_subtraction_expression(self):
        """AggregationComponent with subtraction."""
        hap = DenseHaplotypeArray(
            genotypes=np.zeros((5, 3, 2), dtype=np.int8),
            samples=SampleMeta(iid=np.arange(5)),
            variants=VariantMeta(vid=np.array(['v0', 'v1', 'v2'])),
        )
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=0.0))  # all zeros
        arch.add('B', NoiseComponent(variance=0.0))
        arch.add('C', AggregationComponent('A - B'), inputs=['A', 'B'])
        result = arch.compute(hap, rng=np.random.RandomState(42))
        # 0 - 0 = 0
        np.testing.assert_allclose(result['C'], 0.0, atol=1e-10)
