"""
Unit tests for parser founder= error paths.

Tests:
1. Non-numeric variance in founder=noise() raises ValueError
2. Unsupported function in founder= raises ValueError
3. Valid founder=noise(0.5) works
4. founder=noise(1e-3) with scientific notation works
"""
import numpy as np
import pytest

from xftsim.parser import parse_formula
from xftsim.effect import AdditiveEffects


class TestFounderNoiseErrors:
    def test_non_numeric_variance(self):
        """founder=noise(abc) should raise ValueError."""
        formula = "Y.m ~ mother(Y, founder=noise(abc))"
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        with pytest.raises(ValueError, match="numeric variance"):
            parse_formula(formula, effects={'eff': eff})

    def test_unsupported_function_in_founder(self):
        """founder=foobar(1.0) should raise ValueError."""
        formula = "Y.m ~ mother(Y, founder=foobar(1.0))"
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        with pytest.raises(ValueError, match="unsupported function"):
            parse_formula(formula, effects={'eff': eff})

    def test_valid_founder_noise(self):
        """founder=noise(0.5) should parse without error."""
        formula = "Y.m ~ mother(Y, founder=noise(0.5))"
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        nodes = parse_formula(formula, effects={'eff': eff})
        assert len(nodes) == 1
        assert nodes[0].outputs == ['Y.m']

    def test_founder_noise_scientific_notation(self):
        """founder=noise(1e-3) should parse without error."""
        formula = "Y.m ~ mother(Y, founder=noise(1e-3))"
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        nodes = parse_formula(formula, effects={'eff': eff})
        assert len(nodes) == 1
