"""
Unit tests for EffectSpec classes.
"""
import numpy as np
import pytest
from xftsim.neffect import AdditiveEffects, MultivariateEffects, SparseEffects


class TestAdditiveEffects:
    def test_from_h2_shape(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=100, seed=42)
        assert eff.effects.shape == (100,)
        assert eff.m == 100
        assert eff.m_causal == 100
        assert eff.k == 1

    def test_from_h2_standardized_flag(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=100, standardized=True)
        assert eff.standardized is True
        eff2 = AdditiveEffects.from_h2(h2=0.5, m=100, standardized=False)
        assert eff2.standardized is False

    def test_from_h2_seed_deterministic(self):
        e1 = AdditiveEffects.from_h2(h2=0.5, m=50, seed=42)
        e2 = AdditiveEffects.from_h2(h2=0.5, m=50, seed=42)
        np.testing.assert_array_equal(e1.effects, e2.effects)

    def test_from_h2_variance(self):
        """sum(beta^2) should be approximately h2 for large m."""
        m = 10000
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        var_g = np.sum(eff.effects**2)
        # generous tolerance: SE ≈ h2 * sqrt(2/m) ~ 0.007
        assert abs(var_g - 0.5) < 0.05

    def test_from_array(self):
        arr = np.array([0.1, 0.2, 0.3])
        eff = AdditiveEffects.from_array(arr)
        np.testing.assert_array_equal(eff.effects, arr)
        assert eff.m == 3
        assert eff.m_causal == 3
        assert np.all(eff.variant_mask)

    def test_from_array_roundtrip(self):
        arr = np.random.RandomState(42).randn(20)
        eff = AdditiveEffects.from_array(arr, standardized=False)
        np.testing.assert_array_equal(eff.effects, arr)
        assert eff.standardized is False


class TestMultivariateEffects:
    def test_from_h2_rg_shape(self):
        eff = MultivariateEffects.from_h2_rg(
            h2=[0.5, 0.3], rg=0.2, m=100, seed=42
        )
        assert eff.effects.shape == (100, 2)
        assert eff.m == 100
        assert eff.k == 2

    def test_from_h2_rg_correlation_range(self):
        eff = MultivariateEffects.from_h2_rg(
            h2=[0.5, 0.3], rg=0.5, m=5000, seed=42
        )
        # Empirical genetic correlation
        covg = eff.effects.T @ eff.effects
        varg = np.diag(covg)
        rg_emp = covg[0, 1] / np.sqrt(varg[0] * varg[1])
        # Should be close to 0.5, generous tolerance
        assert abs(rg_emp - 0.5) < 0.15

    def test_from_covg_shape(self):
        covg = np.array([[0.5, 0.1], [0.1, 0.3]])
        eff = MultivariateEffects.from_covg(covg=covg, m=100, seed=42)
        assert eff.effects.shape == (100, 2)

    def test_from_array(self):
        arr = np.random.RandomState(42).randn(50, 3)
        eff = MultivariateEffects.from_array(arr)
        np.testing.assert_array_equal(eff.effects, arr)
        assert eff.k == 3

    def test_standardized_flag(self):
        eff = MultivariateEffects.from_h2_rg(
            h2=[0.5, 0.3], rg=0.2, m=50, standardized=True
        )
        assert eff.standardized is True


class TestSparseEffects:
    def test_from_h2_shape(self):
        eff = SparseEffects.from_h2(h2=0.5, m=100, k_causal=10, seed=42)
        assert eff.effects.shape == (100,)
        assert eff.m == 100
        assert eff.m_causal == 10
        assert eff.k == 1

    def test_correct_nonzero_count(self):
        eff = SparseEffects.from_h2(h2=0.5, m=100, k_causal=10, seed=42)
        assert np.sum(eff.effects != 0) == 10
        assert np.sum(eff.variant_mask) == 10

    def test_variant_mask_matches_effects(self):
        eff = SparseEffects.from_h2(h2=0.5, m=50, k_causal=5, seed=42)
        nonzero = eff.effects != 0
        np.testing.assert_array_equal(eff.variant_mask, nonzero)

    def test_k_causal_exceeds_m(self):
        with pytest.raises(ValueError, match="k_causal"):
            SparseEffects.from_h2(h2=0.5, m=10, k_causal=20)

    def test_seed_deterministic(self):
        e1 = SparseEffects.from_h2(h2=0.5, m=50, k_causal=5, seed=42)
        e2 = SparseEffects.from_h2(h2=0.5, m=50, k_causal=5, seed=42)
        np.testing.assert_array_equal(e1.effects, e2.effects)
        np.testing.assert_array_equal(e1.variant_mask, e2.variant_mask)
