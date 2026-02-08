"""
Extended edge case tests for neffect module.

Tests:
1. AdditiveEffects with h2=0 → all effects zero
2. AdditiveEffects non-standardized flag
3. MultivariateEffects k property
4. MultivariateEffects with rg=0 (uncorrelated)
5. MultivariateEffects 3 traits
6. SparseEffects k_causal > m raises
7. SparseEffects k_causal == m (all causal)
8. EffectSpec ABC properties
9. Effects array dtype always float64
"""
import numpy as np
import pytest

from xftsim.neffect import AdditiveEffects, MultivariateEffects, SparseEffects


class TestAdditiveEdgeCases:
    def test_h2_zero_effects_all_zero(self):
        eff = AdditiveEffects.from_h2(h2=0.0, m=100, seed=42)
        np.testing.assert_array_equal(eff.effects, np.zeros(100))

    def test_non_standardized_flag(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, standardized=False, seed=42)
        assert eff.standardized is False

    def test_from_array_preserves_values(self):
        arr = np.array([0.1, 0.2, 0.3])
        eff = AdditiveEffects.from_array(arr)
        np.testing.assert_allclose(eff.effects, arr)
        assert eff.m == 3
        assert eff.m_causal == 3

    def test_effects_dtype_float64(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        assert eff.effects.dtype == np.float64

    def test_from_array_int_coerced(self):
        arr = np.array([1, 2, 3])
        eff = AdditiveEffects.from_array(arr)
        assert eff.effects.dtype == np.float64

    def test_single_variant(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=1, seed=42)
        assert eff.m == 1
        assert eff.k == 1

    def test_variant_mask_all_true(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        assert np.all(eff.variant_mask)
        assert eff.variant_mask.dtype == bool


class TestMultivariateEdgeCases:
    def test_k_property(self):
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.2, m=10, seed=42)
        assert eff.k == 2

    def test_three_traits(self):
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3, 0.4], rg=0.1, m=20, seed=42)
        assert eff.k == 3
        assert eff.effects.shape == (20, 3)

    def test_uncorrelated_rg_zero(self):
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.5], rg=0.0, m=100, seed=42)
        assert eff.k == 2
        cross = np.sum(eff.effects[:, 0] * eff.effects[:, 1])
        self0 = np.sum(eff.effects[:, 0] ** 2)
        assert abs(cross / self0) < 0.5

    def test_from_covg_shape(self):
        covg = np.array([[0.5, 0.1], [0.1, 0.3]])
        eff = MultivariateEffects.from_covg(covg, m=50, seed=42)
        assert eff.effects.shape == (50, 2)
        assert eff.k == 2

    def test_from_array_2d(self):
        arr = np.random.RandomState(42).randn(10, 3)
        eff = MultivariateEffects.from_array(arr)
        assert eff.m == 10
        assert eff.k == 3
        np.testing.assert_allclose(eff.effects, arr)

    def test_effects_dtype(self):
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.2, m=10, seed=42)
        assert eff.effects.dtype == np.float64


class TestSparseEdgeCases:
    def test_k_causal_exceeds_m_raises(self):
        with pytest.raises(ValueError, match="k_causal.*>.*m"):
            SparseEffects.from_h2(h2=0.5, m=10, k_causal=20, seed=42)

    def test_k_causal_equals_m(self):
        eff = SparseEffects.from_h2(h2=0.5, m=10, k_causal=10, seed=42)
        assert eff.m_causal == 10
        assert np.all(eff.variant_mask)

    def test_k_causal_one(self):
        eff = SparseEffects.from_h2(h2=0.5, m=100, k_causal=1, seed=42)
        assert eff.m_causal == 1
        assert np.sum(eff.effects != 0) == 1

    def test_non_causal_exactly_zero(self):
        eff = SparseEffects.from_h2(h2=0.5, m=50, k_causal=10, seed=42)
        np.testing.assert_array_equal(eff.effects[~eff.variant_mask], 0.0)

    def test_causal_indices_sorted(self):
        eff = SparseEffects.from_h2(h2=0.5, m=100, k_causal=20, seed=42)
        causal_idx = np.where(eff.variant_mask)[0]
        assert np.all(np.diff(causal_idx) > 0)


class TestEffectSpecProperties:
    def test_k_univariate(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        assert eff.k == 1

    def test_m_matches_effects_len(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=25, seed=42)
        assert eff.m == 25
        assert len(eff.effects) == 25

    def test_repr_all_types(self):
        ae = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        assert 'AdditiveEffects' in repr(ae)
        me = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.2, m=10, seed=42)
        assert 'MultivariateEffects' in repr(me)
        se = SparseEffects.from_h2(h2=0.5, m=10, k_causal=5, seed=42)
        assert 'SparseEffects' in repr(se)
