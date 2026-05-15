"""
Unit tests for effect.py edge cases and properties.

Tests:
1. AdditiveEffects.from_h2 boundary values (h2=0, h2=1)
2. AdditiveEffects.from_array
3. MultivariateEffects.from_h2_rg boundary rg values
4. MultivariateEffects.from_covg
5. MultivariateEffects.from_array
6. SparseEffects.from_h2 edge cases
7. EffectSpec properties: m, m_causal, k
8. repr for all types
"""
import numpy as np
import pytest

from xftsim.effect import AdditiveEffects, MultivariateEffects, SparseEffects


class TestAdditiveEffectsFromH2:
    def test_h2_zero(self):
        """h2=0 → all effects are 0 in expectation, variance ~ 0."""
        eff = AdditiveEffects.from_h2(h2=0.0, m=100, seed=42)
        assert eff.m == 100
        assert eff.m_causal == 100
        np.testing.assert_array_equal(eff.effects, np.zeros(100))

    def test_h2_one(self):
        """h2=1 → effects scaled to sum(beta^2) ≈ 1."""
        eff = AdditiveEffects.from_h2(h2=1.0, m=1000, seed=42)
        assert eff.m == 1000
        # E[sum(beta^2)] = h2 = 1.0
        var_g = np.sum(eff.effects ** 2)
        assert 0.5 < var_g < 1.5  # large tolerance for stochasticity

    def test_single_variant(self):
        """m=1: single variant."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=1, seed=42)
        assert eff.m == 1
        assert eff.effects.shape == (1,)
        assert eff.k == 1

    def test_standardized_flag(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42, standardized=False)
        assert eff.standardized is False

    def test_reproducibility(self):
        eff1 = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        eff2 = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        np.testing.assert_array_equal(eff1.effects, eff2.effects)


class TestAdditiveEffectsFromArray:
    def test_basic(self):
        arr = np.array([0.1, 0.2, 0.3])
        eff = AdditiveEffects.from_array(arr)
        np.testing.assert_array_equal(eff.effects, arr)
        assert eff.m == 3
        assert eff.m_causal == 3
        assert eff.standardized is True
        assert np.all(eff.variant_mask)

    def test_not_standardized(self):
        arr = np.array([0.1, 0.2])
        eff = AdditiveEffects.from_array(arr, standardized=False)
        assert eff.standardized is False

    def test_single_effect(self):
        eff = AdditiveEffects.from_array(np.array([1.0]))
        assert eff.m == 1
        assert eff.k == 1


class TestMultivariateEffectsFromH2Rg:
    def test_basic(self):
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.5, m=20, seed=42)
        assert eff.effects.shape == (20, 2)
        assert eff.k == 2
        assert eff.m == 20

    def test_rg_zero(self):
        """rg=0 → independent traits."""
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.5], rg=0.0, m=500, seed=42)
        # Cross-trait correlation should be near 0
        cross_cov = np.dot(eff.effects[:, 0], eff.effects[:, 1])
        assert abs(cross_cov) < 0.3  # wide tolerance

    def test_rg_positive(self):
        """rg=0.8 → positive correlation between trait effects."""
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.5], rg=0.8, m=500, seed=42)
        cross_cov = np.dot(eff.effects[:, 0], eff.effects[:, 1])
        assert cross_cov > 0  # should be positive

    def test_rg_negative(self):
        """rg=-0.8 → negative correlation."""
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.5], rg=-0.8, m=500, seed=42)
        cross_cov = np.dot(eff.effects[:, 0], eff.effects[:, 1])
        assert cross_cov < 0

    def test_three_traits(self):
        eff = MultivariateEffects.from_h2_rg(h2=[0.3, 0.4, 0.5], rg=0.2, m=30, seed=42)
        assert eff.effects.shape == (30, 3)
        assert eff.k == 3


class TestMultivariateEffectsFromCovg:
    def test_basic(self):
        covg = np.array([[0.5, 0.1], [0.1, 0.3]])
        eff = MultivariateEffects.from_covg(covg, m=50, seed=42)
        assert eff.effects.shape == (50, 2)
        assert eff.k == 2

    def test_diagonal_covg(self):
        """Diagonal covg → independent traits."""
        covg = np.diag([0.5, 0.5])
        eff = MultivariateEffects.from_covg(covg, m=100, seed=42)
        assert eff.effects.shape == (100, 2)

    def test_single_trait_covg(self):
        covg = np.array([[0.5]])
        eff = MultivariateEffects.from_covg(covg, m=10, seed=42)
        assert eff.effects.shape == (10, 1)
        assert eff.k == 1


class TestMultivariateEffectsFromArray:
    def test_basic(self):
        arr = np.random.RandomState(42).normal(0, 0.1, (10, 2))
        eff = MultivariateEffects.from_array(arr)
        np.testing.assert_array_equal(eff.effects, arr)
        assert eff.m == 10
        assert eff.k == 2


class TestSparseEffects:
    def test_basic(self):
        eff = SparseEffects.from_h2(h2=0.5, m=100, k_causal=10, seed=42)
        assert eff.m == 100
        assert eff.m_causal == 10
        assert np.sum(eff.variant_mask) == 10
        assert eff.effects.shape == (100,)

    def test_k_causal_exceeds_m(self):
        with pytest.raises(ValueError, match="k_causal.*>.*m"):
            SparseEffects.from_h2(h2=0.5, m=10, k_causal=20, seed=42)

    def test_k_causal_equals_m(self):
        """k_causal == m → all causal (like AdditiveEffects)."""
        eff = SparseEffects.from_h2(h2=0.5, m=10, k_causal=10, seed=42)
        assert eff.m_causal == 10
        assert np.all(eff.variant_mask)

    def test_k_causal_one(self):
        eff = SparseEffects.from_h2(h2=0.5, m=100, k_causal=1, seed=42)
        assert eff.m_causal == 1
        # Only one non-zero effect
        assert np.sum(eff.effects != 0) == 1

    def test_non_causal_effects_are_zero(self):
        eff = SparseEffects.from_h2(h2=0.5, m=20, k_causal=5, seed=42)
        non_causal = ~eff.variant_mask
        np.testing.assert_array_equal(eff.effects[non_causal], 0.0)

    def test_reproducibility(self):
        eff1 = SparseEffects.from_h2(h2=0.5, m=50, k_causal=10, seed=42)
        eff2 = SparseEffects.from_h2(h2=0.5, m=50, k_causal=10, seed=42)
        np.testing.assert_array_equal(eff1.effects, eff2.effects)
        np.testing.assert_array_equal(eff1.variant_mask, eff2.variant_mask)


class TestEffectSpecProperties:
    def test_additive_k(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        assert eff.k == 1

    def test_multivariate_k(self):
        eff = MultivariateEffects.from_h2_rg(h2=[0.3, 0.5], rg=0.5, m=10, seed=42)
        assert eff.k == 2

    def test_sparse_k(self):
        eff = SparseEffects.from_h2(h2=0.5, m=20, k_causal=5, seed=42)
        assert eff.k == 1

    def test_m_causal_additive(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=50, seed=42)
        assert eff.m_causal == 50  # all causal

    def test_variant_mask_dtype(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        assert eff.variant_mask.dtype == bool


class TestEffectRepr:
    def test_additive_repr(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        r = repr(eff)
        assert 'AdditiveEffects' in r
        assert 'm=10' in r

    def test_multivariate_repr(self):
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.5, m=10, seed=42)
        r = repr(eff)
        assert 'MultivariateEffects' in r
        assert 'k=2' in r

    def test_sparse_repr(self):
        eff = SparseEffects.from_h2(h2=0.5, m=20, k_causal=5, seed=42)
        r = repr(eff)
        assert 'SparseEffects' in r
        assert 'm_causal=5' in r
