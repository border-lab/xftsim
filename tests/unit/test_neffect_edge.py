"""
Unit tests for EffectSpec edge cases and validation.

Tests:
1. AdditiveEffects: from_h2 h2=0, from_array, repr, properties (m, m_causal, k)
2. MultivariateEffects: from_h2_rg, from_covg, from_array, properties (k>1)
3. SparseEffects: from_h2 k_causal > m raises, k_causal == m, repr, partial mask
4. EffectSpec: standardized flag, variant_mask dtype
"""
import numpy as np
import pytest

from xftsim.neffect import AdditiveEffects, MultivariateEffects, SparseEffects


class TestAdditiveEffectsEdgeCases:
    def test_h2_zero_effects_zero(self):
        """h2=0 should produce all-zero effects."""
        eff = AdditiveEffects.from_h2(h2=0.0, m=10, seed=42)
        np.testing.assert_array_equal(eff.effects, np.zeros(10))

    def test_from_array(self):
        """from_array should store the exact effects given."""
        arr = np.array([1.0, 2.0, 3.0])
        eff = AdditiveEffects.from_array(arr)
        np.testing.assert_array_equal(eff.effects, arr)
        assert eff.m == 3
        assert eff.m_causal == 3
        assert eff.k == 1
        assert eff.standardized is True

    def test_from_array_not_standardized(self):
        """from_array with standardized=False."""
        eff = AdditiveEffects.from_array(np.ones(5), standardized=False)
        assert eff.standardized is False

    def test_properties(self):
        """m, m_causal, k properties should be correct."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=20, seed=0)
        assert eff.m == 20
        assert eff.m_causal == 20
        assert eff.k == 1
        assert eff.effects.shape == (20,)
        assert eff.variant_mask.shape == (20,)
        assert np.all(eff.variant_mask)

    def test_repr(self):
        """repr should include key info."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        r = repr(eff)
        assert 'm=10' in r
        assert 'm_causal=10' in r
        assert 'standardized=True' in r

    def test_deterministic_seed(self):
        """Same seed should produce identical effects."""
        eff1 = AdditiveEffects.from_h2(h2=0.5, m=100, seed=42)
        eff2 = AdditiveEffects.from_h2(h2=0.5, m=100, seed=42)
        np.testing.assert_array_equal(eff1.effects, eff2.effects)

    def test_different_seed_different_effects(self):
        """Different seeds should produce different effects."""
        eff1 = AdditiveEffects.from_h2(h2=0.5, m=100, seed=42)
        eff2 = AdditiveEffects.from_h2(h2=0.5, m=100, seed=99)
        assert not np.allclose(eff1.effects, eff2.effects)


class TestMultivariateEffectsEdgeCases:
    def test_from_h2_rg_properties(self):
        """MultivariateEffects should have correct k, m."""
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.4, m=15, seed=42)
        assert eff.m == 15
        assert eff.k == 2
        assert eff.effects.shape == (15, 2)
        assert eff.m_causal == 15

    def test_from_covg(self):
        """from_covg should produce valid effects."""
        covg = np.array([[0.5, 0.1], [0.1, 0.3]])
        eff = MultivariateEffects.from_covg(covg, m=20, seed=42)
        assert eff.m == 20
        assert eff.k == 2
        assert eff.effects.shape == (20, 2)

    def test_from_array_2d(self):
        """from_array with 2D array should set k correctly."""
        arr = np.random.randn(10, 3)
        eff = MultivariateEffects.from_array(arr)
        assert eff.m == 10
        assert eff.k == 3

    def test_repr(self):
        """repr should include k."""
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.0, m=5, seed=0)
        r = repr(eff)
        assert 'k=2' in r
        assert 'm=5' in r

    def test_three_traits(self):
        """3-trait effects should have k=3."""
        eff = MultivariateEffects.from_h2_rg(h2=[0.3, 0.4, 0.5], rg=0.2, m=20, seed=42)
        assert eff.k == 3
        assert eff.effects.shape == (20, 3)

    def test_rg_zero_independent(self):
        """rg=0 should produce approximately uncorrelated effects."""
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.5], rg=0.0, m=5000, seed=42)
        r = np.corrcoef(eff.effects[:, 0], eff.effects[:, 1])[0, 1]
        assert abs(r) < 0.1, f"Expected ~0 correlation, got {r}"


class TestSparseEffectsEdgeCases:
    def test_k_causal_gt_m_raises(self):
        """k_causal > m should raise ValueError."""
        with pytest.raises(ValueError, match="k_causal.*>.*m"):
            SparseEffects.from_h2(h2=0.5, m=10, k_causal=20, seed=42)

    def test_k_causal_equals_m(self):
        """k_causal == m should be valid (all causal)."""
        eff = SparseEffects.from_h2(h2=0.5, m=10, k_causal=10, seed=42)
        assert eff.m == 10
        assert eff.m_causal == 10
        assert np.all(eff.variant_mask)

    def test_partial_mask(self):
        """Only k_causal variants should be non-zero."""
        eff = SparseEffects.from_h2(h2=0.5, m=20, k_causal=5, seed=42)
        assert eff.m_causal == 5
        assert np.sum(eff.variant_mask) == 5
        # Non-causal should have zero effects
        assert np.all(eff.effects[~eff.variant_mask] == 0)
        # Causal should have non-zero effects (with high probability)
        assert np.any(eff.effects[eff.variant_mask] != 0)

    def test_repr(self):
        """repr should show m and m_causal."""
        eff = SparseEffects.from_h2(h2=0.5, m=20, k_causal=5, seed=42)
        r = repr(eff)
        assert 'm=20' in r
        assert 'm_causal=5' in r

    def test_k_causal_one(self):
        """Single causal variant."""
        eff = SparseEffects.from_h2(h2=0.5, m=10, k_causal=1, seed=42)
        assert eff.m_causal == 1
        assert np.sum(eff.effects != 0) == 1


class TestEffectSpecBase:
    def test_standardized_flag(self):
        """standardized flag should be stored correctly."""
        eff_std = AdditiveEffects.from_h2(h2=0.5, m=10, standardized=True, seed=0)
        eff_raw = AdditiveEffects.from_h2(h2=0.5, m=10, standardized=False, seed=0)
        assert eff_std.standardized is True
        assert eff_raw.standardized is False

    def test_variant_mask_dtype(self):
        """variant_mask should be bool dtype."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=0)
        assert eff.variant_mask.dtype == bool

    def test_effects_float64(self):
        """effects should be float64 dtype."""
        eff = AdditiveEffects.from_array(np.array([1, 2, 3], dtype=np.int32))
        assert eff.effects.dtype == np.float64
