"""
Unit tests for shunting-yard expression evaluator edge cases.

Targets _tokenize, _shunting_yard, _evaluate_expression in narch.py.

Tests:
1. Division by zero produces inf/nan
2. Unary minus at expression start
3. Unary minus after open paren
4. Deeply nested parentheses
5. Scientific notation in expressions
6. Single variable expression
7. Single number expression
8. Mismatched parentheses
9. Empty operand stack
10. Division operator
"""
import numpy as np
import pytest

from xftsim.arch import _tokenize, _shunting_yard, _evaluate_expression
from xftsim.struct import SampleMeta, PhenotypeArray


def _make_pheno(n=4, **kv):
    sm = SampleMeta(iid=np.arange(n), fid=np.arange(n))
    pheno = PhenotypeArray(samples=sm)
    for key, val in kv.items():
        pheno._values[key] = np.asarray(val, dtype=np.float64)
    return pheno


class TestTokenize:
    def test_simple_expression(self):
        tokens = _tokenize('A + B')
        assert ('NAME', 'A') in tokens
        assert ('OP', '+') in tokens
        assert ('NAME', 'B') in tokens

    def test_dotted_names(self):
        tokens = _tokenize('Y.G + Y.E')
        names = [t for t in tokens if t[0] == 'NAME']
        assert ('NAME', 'Y.G') in names
        assert ('NAME', 'Y.E') in names

    def test_scientific_notation(self):
        tokens = _tokenize('1.5e-3 * X')
        nums = [t for t in tokens if t[0] == 'NUM']
        assert len(nums) == 1
        np.testing.assert_allclose(nums[0][1], 1.5e-3)

    def test_integer_number(self):
        tokens = _tokenize('2 * X')
        nums = [t for t in tokens if t[0] == 'NUM']
        assert nums[0][1] == 2.0


class TestShuntingYard:
    def test_simple_addition(self):
        tokens = _tokenize('A + B')
        rpn = _shunting_yard(tokens)
        # Should be: A B +
        types = [t[0] for t in rpn]
        assert types == ['NAME', 'NAME', 'OP']

    def test_precedence_mul_over_add(self):
        tokens = _tokenize('A + B * C')
        rpn = _shunting_yard(tokens)
        # Should be: A B C * +
        vals = [t[1] for t in rpn]
        assert vals == ['A', 'B', 'C', '*', '+']

    def test_parentheses_override_precedence(self):
        tokens = _tokenize('(A + B) * C')
        rpn = _shunting_yard(tokens)
        vals = [t[1] for t in rpn]
        assert vals == ['A', 'B', '+', 'C', '*']

    def test_mismatched_open_paren(self):
        tokens = _tokenize('(A + B')
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            _shunting_yard(tokens)

    def test_mismatched_close_paren(self):
        tokens = _tokenize('A + B)')
        with pytest.raises(ValueError, match="Mismatched parentheses"):
            _shunting_yard(tokens)

    def test_unary_minus_start(self):
        tokens = _tokenize('-A')
        rpn = _shunting_yard(tokens)
        # -A = -1 * A
        vals = [t[1] for t in rpn]
        assert -1.0 in vals
        assert 'A' in vals

    def test_unary_minus_after_paren(self):
        tokens = _tokenize('(-A)')
        rpn = _shunting_yard(tokens)
        vals = [t[1] for t in rpn]
        assert -1.0 in vals


class TestEvaluateExpression:
    def test_simple_addition(self):
        pheno = _make_pheno(A=np.array([1.0, 2.0, 3.0, 4.0]),
                            B=np.array([10.0, 20.0, 30.0, 40.0]))
        result = _evaluate_expression('A + B', pheno, 4)
        np.testing.assert_allclose(result, [11.0, 22.0, 33.0, 44.0])

    def test_subtraction(self):
        pheno = _make_pheno(A=np.array([10.0, 20.0, 30.0, 40.0]),
                            B=np.array([1.0, 2.0, 3.0, 4.0]))
        result = _evaluate_expression('A - B', pheno, 4)
        np.testing.assert_allclose(result, [9.0, 18.0, 27.0, 36.0])

    def test_scalar_multiplication(self):
        pheno = _make_pheno(A=np.array([1.0, 2.0, 3.0, 4.0]))
        result = _evaluate_expression('0.5 * A', pheno, 4)
        np.testing.assert_allclose(result, [0.5, 1.0, 1.5, 2.0])

    def test_division(self):
        pheno = _make_pheno(A=np.array([10.0, 20.0, 30.0, 40.0]))
        result = _evaluate_expression('A / 2', pheno, 4)
        np.testing.assert_allclose(result, [5.0, 10.0, 15.0, 20.0])

    def test_division_by_zero(self):
        """Division by zero should produce inf/nan, not crash."""
        pheno = _make_pheno(A=np.array([1.0, 0.0, -1.0, 2.0]),
                            B=np.array([0.0, 0.0, 0.0, 1.0]))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            result = _evaluate_expression('A / B', pheno, 4)
        assert np.isinf(result[0]) or np.isnan(result[0])
        assert result[3] == 2.0

    def test_complex_expression(self):
        pheno = _make_pheno(A=np.array([2.0, 4.0, 6.0, 8.0]),
                            B=np.array([1.0, 1.0, 1.0, 1.0]))
        result = _evaluate_expression('0.5 * (A + B)', pheno, 4)
        np.testing.assert_allclose(result, [1.5, 2.5, 3.5, 4.5])

    def test_undefined_reference(self):
        pheno = _make_pheno(A=np.array([1.0, 2.0, 3.0, 4.0]))
        with pytest.raises(ValueError, match="Undefined reference"):
            _evaluate_expression('A + MISSING', pheno, 4)

    def test_single_variable(self):
        pheno = _make_pheno(A=np.array([1.0, 2.0, 3.0, 4.0]))
        result = _evaluate_expression('A', pheno, 4)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0, 4.0])

    def test_single_number(self):
        pheno = _make_pheno()
        result = _evaluate_expression('42', pheno, 4)
        np.testing.assert_allclose(result, [42.0, 42.0, 42.0, 42.0])

    def test_deeply_nested(self):
        pheno = _make_pheno(A=np.array([1.0, 2.0, 3.0, 4.0]))
        result = _evaluate_expression('((((A))))', pheno, 4)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0, 4.0])

    def test_unary_minus_expression(self):
        pheno = _make_pheno(A=np.array([1.0, 2.0, 3.0, 4.0]))
        result = _evaluate_expression('-1 * A', pheno, 4)
        np.testing.assert_allclose(result, [-1.0, -2.0, -3.0, -4.0])
