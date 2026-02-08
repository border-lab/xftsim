"""
Unit tests for SparseEffects edge cases.

Tests:
1. k_causal=0 → empty effects (all zero, no causal variants)
2. k_causal=m → all variants causal
3. SparseEffects variant_mask shape matches m
4. Non-causal effects are zero
"""
import numpy as np
import pytest

from xftsim.neffect import SparseEffects


class TestSparseEffectsKZero:
    def test_k_zero_raises(self):
        """k_causal=0 causes division by zero (h2/k_causal), should error."""
        with pytest.raises((ValueError, ZeroDivisionError)):
            SparseEffects.from_h2(h2=0.5, m=20, k_causal=0, seed=42)

    def test_k_equals_m(self):
        """k_causal=m should mark all variants causal."""
        eff = SparseEffects.from_h2(h2=0.5, m=10, k_causal=10, seed=42)
        assert np.sum(eff.variant_mask) == 10
        # All effects should be nonzero (with very high probability)
        assert np.sum(eff.effects != 0) == 10

    def test_variant_mask_shape(self):
        """variant_mask should have length m."""
        eff = SparseEffects.from_h2(h2=0.5, m=30, k_causal=5, seed=42)
        assert eff.variant_mask.shape == (30,)
        assert eff.variant_mask.dtype == bool

    def test_non_causal_zero(self):
        """Non-causal variant effects should be exactly zero."""
        eff = SparseEffects.from_h2(h2=0.5, m=20, k_causal=5, seed=42)
        non_causal = ~eff.variant_mask
        np.testing.assert_array_equal(eff.effects[non_causal], 0.0)

    def test_k_greater_than_m_raises(self):
        """k_causal > m should raise ValueError."""
        with pytest.raises(ValueError, match="k_causal.*>.*m"):
            SparseEffects.from_h2(h2=0.5, m=10, k_causal=15, seed=42)
