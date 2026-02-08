"""
Unit tests for complex expression evaluation.

Tests:
1. Weighted sum: 0.5 * A + 0.5 * B
2. Subtraction: A - B
3. Nested parentheses: (A + B) * (C - D)
4. Chain operations: A + B + C + D
5. Negative coefficient: -0.3 * A + 1.3 * B
"""
import numpy as np
import pytest

from xftsim.narch import AggregationComponent, ArchNode
from xftsim.struct import SampleMeta, NPhenotypeArray

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


def _eval_expression(expr, pheno_dict, n):
    """Helper: evaluate expression with given phenotype values."""
    sm = SampleMeta(iid=np.arange(n))
    pheno = NPhenotypeArray(sm)
    for k, v in pheno_dict.items():
        pheno[k] = v

    comp = AggregationComponent(expr)
    hap = TestSimulation.founder_haplotypes(n=n, m=5, seed=42)
    node = ArchNode(outputs=['Y'], component=comp, inputs=list(pheno_dict.keys()),
                    grouping=None)

    return comp.compute(node, hap, pheno)


class TestComplexExpressions:
    def test_weighted_sum(self):
        """0.5 * A + 0.5 * B"""
        n = 10
        A = np.ones(n) * 4.0
        B = np.ones(n) * 6.0
        result = _eval_expression('0.5 * A + 0.5 * B', {'A': A, 'B': B}, n)
        np.testing.assert_allclose(result, 5.0)

    def test_subtraction(self):
        """A - B"""
        n = 10
        A = np.ones(n) * 10.0
        B = np.ones(n) * 3.0
        result = _eval_expression('A - B', {'A': A, 'B': B}, n)
        np.testing.assert_allclose(result, 7.0)

    def test_nested_parens(self):
        """(A + B) * (C - D)"""
        n = 10
        result = _eval_expression(
            '(A + B) * (C - D)',
            {'A': np.ones(n) * 2, 'B': np.ones(n) * 3,
             'C': np.ones(n) * 5, 'D': np.ones(n) * 1},
            n,
        )
        # (2 + 3) * (5 - 1) = 5 * 4 = 20
        np.testing.assert_allclose(result, 20.0)

    def test_chain_addition(self):
        """A + B + C + D"""
        n = 10
        result = _eval_expression(
            'A + B + C + D',
            {'A': np.ones(n), 'B': np.ones(n) * 2,
             'C': np.ones(n) * 3, 'D': np.ones(n) * 4},
            n,
        )
        np.testing.assert_allclose(result, 10.0)

    def test_negative_coefficient(self):
        """-0.3 * A + 1.3 * B"""
        n = 10
        A = np.ones(n) * 10.0
        B = np.ones(n) * 10.0
        result = _eval_expression('-0.3 * A + 1.3 * B', {'A': A, 'B': B}, n)
        # -0.3*10 + 1.3*10 = -3 + 13 = 10
        np.testing.assert_allclose(result, 10.0)

    def test_multiplication_precedence(self):
        """A + B * C (B*C should evaluate before +)"""
        n = 10
        result = _eval_expression(
            'A + B * C',
            {'A': np.ones(n) * 2, 'B': np.ones(n) * 3, 'C': np.ones(n) * 4},
            n,
        )
        # 2 + 3*4 = 14, not (2+3)*4 = 20
        np.testing.assert_allclose(result, 14.0)

    def test_dotted_names(self):
        """Y.G + Y.E with dotted names."""
        n = 10
        result = _eval_expression(
            'Y.G + Y.E',
            {'Y.G': np.ones(n) * 3.0, 'Y.E': np.ones(n) * 7.0},
            n,
        )
        np.testing.assert_allclose(result, 10.0)
