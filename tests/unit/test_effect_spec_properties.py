"""
Unit tests for EffectSpec properties and edge cases.

Tests:
1. AdditiveEffects.m property
2. AdditiveEffects.k == 1
3. MultivariateEffects.k matches number of traits
4. SparseEffects.m_causal matches k_causal
5. AdditiveEffects.m_causal == m (all causal)
6. Effect dtype is float64
7. Variant mask dtype is bool
"""
import numpy as np
import pytest

from xftsim.neffect import AdditiveEffects, MultivariateEffects, SparseEffects


class TestEffectSpecProperties:
    def test_additive_m(self):
        """m should match the variant count."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=20, seed=42)
        assert eff.m == 20

    def test_additive_k_one(self):
        """Univariate AdditiveEffects should have k=1."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=20, seed=42)
        assert eff.k == 1

    def test_multivariate_k(self):
        """MultivariateEffects.k should match number of traits."""
        mv = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3, 0.7], rg=0.2, m=10, seed=42)
        assert mv.k == 3

    def test_multivariate_m(self):
        """MultivariateEffects.m should match variant count."""
        mv = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.2, m=15, seed=42)
        assert mv.m == 15

    def test_sparse_m_causal(self):
        """SparseEffects.m_causal should match k_causal arg."""
        eff = SparseEffects.from_h2(h2=0.5, m=50, k_causal=10, seed=42)
        assert eff.m_causal == 10

    def test_additive_all_causal(self):
        """AdditiveEffects has all variants causal (m_causal == m)."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=20, seed=42)
        assert eff.m_causal == 20

    def test_effects_dtype_float64(self):
        """Effects should be float64."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        assert eff.effects.dtype == np.float64

    def test_variant_mask_dtype_bool(self):
        """Variant mask should be bool."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        assert eff.variant_mask.dtype == bool

    def test_multivariate_effects_shape(self):
        """MultivariateEffects.effects should be (m, k)."""
        mv = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.2, m=10, seed=42)
        assert mv.effects.shape == (10, 2)

    def test_standardized_flag(self):
        """standardized flag should be set correctly."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, standardized=True, seed=42)
        assert eff.standardized is True

        eff2 = AdditiveEffects.from_h2(h2=0.5, m=10, standardized=False, seed=42)
        assert eff2.standardized is False
