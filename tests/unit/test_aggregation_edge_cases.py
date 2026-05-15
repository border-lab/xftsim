"""
Unit tests for AggregationComponent edge cases.

Tests:
1. Division by zero behavior (NaN/Inf propagation)
2. NaN propagation through expressions
3. Deeply nested parentheses
4. Expression with only constants (no variable references)
5. Multiple references to same variable
6. Complex multi-variable weighted sums
7. Empty expression error
8. Stack underflow error
9. _extract_names deduplication
10. AggregationComponent repr
"""
import numpy as np
import pytest

from xftsim.arch import (
    AggregationComponent, ArchNode,
    _tokenize, _shunting_yard, _evaluate_expression,
)
from xftsim.struct import SampleMeta, PhenotypeArray


def _make_pheno(n, values_dict):
    """Create PhenotypeArray with given values."""
    sm = SampleMeta(iid=np.arange(n))
    pheno = PhenotypeArray(samples=sm)
    for k, v in values_dict.items():
        pheno._values[k] = np.asarray(v, dtype=np.float64)
    return pheno


class TestAggregationDivision:
    def test_division_by_zero_produces_inf(self):
        """Division by zero should produce Inf, not crash."""
        pheno = _make_pheno(3, {
            'A': np.array([1.0, 2.0, 3.0]),
            'B': np.array([0.0, 1.0, 0.0]),
        })
        result = _evaluate_expression('A / B', pheno, 3)
        assert np.isinf(result[0])  # 1/0 = inf
        assert result[1] == 2.0     # 2/1 = 2
        assert np.isinf(result[2])  # 3/0 = inf

    def test_zero_divided_by_zero_is_nan(self):
        """0/0 → NaN."""
        pheno = _make_pheno(2, {
            'A': np.array([0.0, 1.0]),
            'B': np.array([0.0, 1.0]),
        })
        result = _evaluate_expression('A / B', pheno, 2)
        assert np.isnan(result[0])
        assert result[1] == 1.0


class TestAggregationNaN:
    def test_nan_propagation_add(self):
        """NaN + x = NaN."""
        pheno = _make_pheno(2, {
            'A': np.array([np.nan, 1.0]),
            'B': np.array([1.0, 2.0]),
        })
        result = _evaluate_expression('A + B', pheno, 2)
        assert np.isnan(result[0])
        assert result[1] == 3.0

    def test_nan_propagation_multiply(self):
        """NaN * x = NaN."""
        pheno = _make_pheno(2, {
            'A': np.array([np.nan, 2.0]),
            'B': np.array([5.0, 3.0]),
        })
        result = _evaluate_expression('A * B', pheno, 2)
        assert np.isnan(result[0])
        assert result[1] == 6.0


class TestAggregationNesting:
    def test_deeply_nested_parens(self):
        """((((A + B)))) should work."""
        pheno = _make_pheno(2, {
            'A': np.array([1.0, 2.0]),
            'B': np.array([3.0, 4.0]),
        })
        result = _evaluate_expression('((((A + B))))', pheno, 2)
        np.testing.assert_array_equal(result, [4.0, 6.0])

    def test_nested_multiply_add(self):
        """(A + B) * (A - B) should work."""
        pheno = _make_pheno(3, {
            'A': np.array([5.0, 3.0, 1.0]),
            'B': np.array([3.0, 3.0, 2.0]),
        })
        result = _evaluate_expression('(A + B) * (A - B)', pheno, 3)
        # (5+3)*(5-3) = 16, (3+3)*(3-3) = 0, (1+2)*(1-2) = -3
        np.testing.assert_array_equal(result, [16.0, 0.0, -3.0])


class TestAggregationConstants:
    def test_constant_only_expression(self):
        """Expression with only numbers → constant array."""
        pheno = _make_pheno(3, {'A': np.array([1.0, 2.0, 3.0])})
        result = _evaluate_expression('2 + 3', pheno, 3)
        np.testing.assert_array_equal(result, np.full(3, 5.0))

    def test_scalar_multiply_variable(self):
        """2 * A should double values."""
        pheno = _make_pheno(3, {'A': np.array([1.0, 2.0, 3.0])})
        result = _evaluate_expression('2 * A', pheno, 3)
        np.testing.assert_array_equal(result, [2.0, 4.0, 6.0])


class TestAggregationMultiRef:
    def test_same_variable_twice(self):
        """A + A = 2*A."""
        pheno = _make_pheno(3, {'A': np.array([1.0, 2.0, 3.0])})
        result = _evaluate_expression('A + A', pheno, 3)
        np.testing.assert_array_equal(result, [2.0, 4.0, 6.0])

    def test_three_variable_weighted_sum(self):
        """0.5 * A + 0.3 * B + 0.2 * C."""
        pheno = _make_pheno(2, {
            'A': np.array([10.0, 20.0]),
            'B': np.array([10.0, 10.0]),
            'C': np.array([10.0, 0.0]),
        })
        result = _evaluate_expression('0.5 * A + 0.3 * B + 0.2 * C', pheno, 2)
        np.testing.assert_allclose(result, [10.0, 13.0])


class TestAggregationErrors:
    def test_undefined_variable_raises(self):
        pheno = _make_pheno(2, {'A': np.array([1.0, 2.0])})
        with pytest.raises(ValueError, match="Undefined reference"):
            _evaluate_expression('A + NONEXISTENT', pheno, 2)

    def test_stack_underflow_raises(self):
        """Operator with not enough operands."""
        pheno = _make_pheno(2, {'A': np.array([1.0, 2.0])})
        # Manually craft bad RPN
        with pytest.raises(ValueError, match="not enough operands"):
            _evaluate_expression('+ A', pheno, 2)


class TestAggregationComponent:
    def test_repr(self):
        comp = AggregationComponent('A + B')
        assert "AggregationComponent" in repr(comp)
        assert "A + B" in repr(comp)

    def test_extract_names_dedup(self):
        comp = AggregationComponent('A + B + A * B')
        assert comp._input_names == ['A', 'B']

    def test_extract_names_dotted(self):
        comp = AggregationComponent('Y.G + Y.E')
        assert comp._input_names == ['Y.G', 'Y.E']

    def test_extract_names_no_constants(self):
        """Numbers should not appear in extracted names."""
        comp = AggregationComponent('2 * A + 3.5 * B')
        assert comp._input_names == ['A', 'B']

    def test_compute_via_node(self):
        """Full compute path through node."""
        from xftsim.struct import VariantMeta, DenseHaplotypeArray
        sm = SampleMeta(iid=np.arange(3))
        vm = VariantMeta(vid=np.array(['v0']))
        geno = np.ones((3, 1, 2), dtype=np.int8)
        hap = DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)
        pheno = PhenotypeArray(samples=sm)
        pheno._values['A'] = np.array([1.0, 2.0, 3.0])
        pheno._values['B'] = np.array([4.0, 5.0, 6.0])

        comp = AggregationComponent('A + B')
        node = ArchNode(outputs=['C'], component=comp, inputs=['A', 'B'])
        result = comp.compute(node, hap, pheno)
        np.testing.assert_array_equal(result, [5.0, 7.0, 9.0])
