"""
Advanced unit tests for effect.py edge cases.

Tests covering:
1. MultivariateEffects k property for k=1, k=2, k=3
2. MultivariateEffects.from_h2_rg: different rg values produce different correlation structures
3. MultivariateEffects: variant_mask when all variants are causal
4. SparseEffects: m_causal vs m relationship
5. SparseEffects: variant_mask has exactly k_causal True entries
6. AdditiveEffects: from_h2 with different m values
7. AdditiveEffects: from_h2 with standardized=True vs False
8. Effect matrix shape validation: 1-D vs 2-D
9. Effects repr/str output
10. MultivariateEffects from_h2_rg with rg=0 produces uncorrelated effects
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation

from xftsim.effect import AdditiveEffects, MultivariateEffects, SparseEffects


class TestMultivariateEffectsK:
    """Test MultivariateEffects k property for various trait counts."""

    def test_k_one_trait(self):
        """k=1 for single-trait MultivariateEffects."""
        mv = MultivariateEffects.from_h2_rg(h2=[0.5], rg=0.0, m=10, seed=42)
        assert mv.k == 1
        assert mv.effects.shape == (10, 1)

    def test_k_two_traits(self):
        """k=2 for bivariate MultivariateEffects."""
        mv = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.2, m=10, seed=42)
        assert mv.k == 2
        assert mv.effects.shape == (10, 2)

    def test_k_three_traits(self):
        """k=3 for trivariate MultivariateEffects."""
        mv = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3, 0.7], rg=0.1, m=15, seed=42)
        assert mv.k == 3
        assert mv.effects.shape == (15, 3)

    def test_k_five_traits(self):
        """k=5 for higher-dimensional MultivariateEffects."""
        mv = MultivariateEffects.from_h2_rg(
            h2=[0.5, 0.3, 0.7, 0.4, 0.6], rg=0.3, m=20, seed=42
        )
        assert mv.k == 5
        assert mv.effects.shape == (20, 5)


class TestMultivariateEffectsCorrelation:
    """Test that different rg values produce different correlation structures."""

    def test_rg_zero_uncorrelated(self):
        """rg=0 should produce uncorrelated effects across traits."""
        np.random.seed(42)
        m = 1000  # Large m for stable correlation estimates
        mv = MultivariateEffects.from_h2_rg(
            h2=[0.5, 0.5], rg=0.0, m=m, seed=42
        )
        # Empirical correlation of effect vectors
        corr = np.corrcoef(mv.effects[:, 0], mv.effects[:, 1])[0, 1]
        # Should be close to 0
        assert abs(corr) < 0.15, f"Expected rg~0, got {corr}"

    def test_rg_positive_correlated(self):
        """Positive rg should produce positively correlated effects."""
        np.random.seed(42)
        m = 1000
        rg_target = 0.5
        mv = MultivariateEffects.from_h2_rg(
            h2=[0.5, 0.5], rg=rg_target, m=m, seed=42
        )
        corr = np.corrcoef(mv.effects[:, 0], mv.effects[:, 1])[0, 1]
        # Should be close to target rg
        assert 0.3 < corr < 0.7, f"Expected rg~{rg_target}, got {corr}"

    def test_rg_negative_anticorrelated(self):
        """Negative rg should produce negatively correlated effects."""
        np.random.seed(42)
        m = 1000
        rg_target = -0.5
        mv = MultivariateEffects.from_h2_rg(
            h2=[0.5, 0.5], rg=rg_target, m=m, seed=42
        )
        corr = np.corrcoef(mv.effects[:, 0], mv.effects[:, 1])[0, 1]
        # Should be close to target rg (negative)
        assert -0.7 < corr < -0.3, f"Expected rg~{rg_target}, got {corr}"

    def test_different_rg_different_corr(self):
        """Different rg values should produce measurably different correlations."""
        m = 1000
        mv1 = MultivariateEffects.from_h2_rg(
            h2=[0.5, 0.5], rg=0.1, m=m, seed=42
        )
        mv2 = MultivariateEffects.from_h2_rg(
            h2=[0.5, 0.5], rg=0.8, m=m, seed=43
        )
        corr1 = np.corrcoef(mv1.effects[:, 0], mv1.effects[:, 1])[0, 1]
        corr2 = np.corrcoef(mv2.effects[:, 0], mv2.effects[:, 1])[0, 1]
        # Difference should be substantial
        assert abs(corr2 - corr1) > 0.4, \
            f"Expected different correlations, got {corr1} vs {corr2}"

    def test_rg_one_perfect_correlation(self):
        """rg=1 should produce perfectly correlated effects (up to scaling)."""
        m = 1000
        h2 = [0.5, 0.3]  # Different heritabilities
        mv = MultivariateEffects.from_h2_rg(h2=h2, rg=1.0, m=m, seed=42)
        # Correlation should be very close to 1
        corr = np.corrcoef(mv.effects[:, 0], mv.effects[:, 1])[0, 1]
        assert corr > 0.95, f"Expected rg~1, got {corr}"


class TestMultivariateEffectsVariantMask:
    """Test variant_mask behavior for MultivariateEffects."""

    def test_all_causal(self):
        """MultivariateEffects should have all variants causal."""
        mv = MultivariateEffects.from_h2_rg(
            h2=[0.5, 0.3], rg=0.2, m=20, seed=42
        )
        assert np.all(mv.variant_mask)
        assert mv.m_causal == mv.m

    def test_variant_mask_shape(self):
        """variant_mask should have length m."""
        mv = MultivariateEffects.from_h2_rg(
            h2=[0.5, 0.3, 0.7], rg=0.1, m=30, seed=42
        )
        assert mv.variant_mask.shape == (30,)
        assert mv.variant_mask.dtype == bool

    def test_variant_mask_from_covg(self):
        """from_covg should also have all variants causal."""
        covg = np.array([[0.5, 0.1], [0.1, 0.3]])
        mv = MultivariateEffects.from_covg(covg=covg, m=15, seed=42)
        assert np.all(mv.variant_mask)
        assert mv.m_causal == 15

    def test_variant_mask_from_array(self):
        """from_array should have all variants causal."""
        effects = np.random.randn(10, 3)
        mv = MultivariateEffects.from_array(effects=effects)
        assert np.all(mv.variant_mask)
        assert mv.m_causal == 10


class TestSparseEffectsMCausal:
    """Test m_causal vs m relationship for SparseEffects."""

    def test_m_causal_less_than_m(self):
        """m_causal should be less than m for sparse effects."""
        eff = SparseEffects.from_h2(h2=0.5, m=100, k_causal=20, seed=42)
        assert eff.m_causal < eff.m
        assert eff.m_causal == 20
        assert eff.m == 100

    def test_m_causal_ratio(self):
        """Test various sparsity ratios."""
        for m, k in [(50, 5), (100, 10), (200, 20), (100, 1)]:
            eff = SparseEffects.from_h2(h2=0.5, m=m, k_causal=k, seed=42)
            assert eff.m_causal == k
            assert eff.m == m
            assert eff.m_causal / eff.m == k / m

    def test_m_causal_equals_m_boundary(self):
        """When k_causal=m, m_causal should equal m."""
        eff = SparseEffects.from_h2(h2=0.5, m=25, k_causal=25, seed=42)
        assert eff.m_causal == eff.m
        assert eff.m_causal == 25


class TestSparseEffectsVariantMaskExact:
    """Test that variant_mask has exactly k_causal True entries."""

    def test_exact_count_small(self):
        """Small case: exactly k_causal True entries."""
        eff = SparseEffects.from_h2(h2=0.5, m=20, k_causal=5, seed=42)
        assert np.sum(eff.variant_mask) == 5

    def test_exact_count_large(self):
        """Large case: exactly k_causal True entries."""
        eff = SparseEffects.from_h2(h2=0.5, m=1000, k_causal=100, seed=42)
        assert np.sum(eff.variant_mask) == 100

    def test_exact_count_one(self):
        """Edge case: exactly 1 causal variant."""
        eff = SparseEffects.from_h2(h2=0.5, m=50, k_causal=1, seed=42)
        assert np.sum(eff.variant_mask) == 1

    def test_causal_indices_match_nonzero(self):
        """Causal indices should match nonzero effect indices."""
        eff = SparseEffects.from_h2(h2=0.5, m=50, k_causal=10, seed=42)
        causal_from_mask = np.where(eff.variant_mask)[0]
        causal_from_effects = np.where(eff.effects != 0)[0]
        np.testing.assert_array_equal(causal_from_mask, causal_from_effects)


class TestAdditiveEffectsVariousM:
    """Test AdditiveEffects.from_h2 with different m values."""

    def test_m_small(self):
        """Small m (10 variants)."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        assert eff.m == 10
        assert len(eff.effects) == 10

    def test_m_medium(self):
        """Medium m (100 variants)."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=100, seed=42)
        assert eff.m == 100
        assert len(eff.effects) == 100

    def test_m_large(self):
        """Large m (1000 variants)."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=1000, seed=42)
        assert eff.m == 1000
        assert len(eff.effects) == 1000

    def test_genetic_variance_scales_with_m(self):
        """Expected sum of squared effects should be ~h2 regardless of m."""
        h2 = 0.5
        # Try different m values
        for m in [10, 50, 100, 500]:
            eff = AdditiveEffects.from_h2(h2=h2, m=m, seed=42)
            # E[sum(beta^2)] = h2 for standardized genotypes
            sum_sq = np.sum(eff.effects**2)
            # Should be close to h2 (within ~3 std devs)
            # Var(sum(beta^2)) = m * 2 * (h2/m)^2 = 2*h2^2/m
            std_sum_sq = np.sqrt(2 * h2**2 / m)
            assert abs(sum_sq - h2) < 3 * std_sum_sq, \
                f"m={m}: sum(beta^2)={sum_sq}, expected {h2} ± {3*std_sum_sq}"


class TestAdditiveEffectsStandardized:
    """Test AdditiveEffects standardized=True vs False."""

    def test_standardized_flag_true(self):
        """standardized=True should be recorded."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=50, standardized=True, seed=42)
        assert eff.standardized is True

    def test_standardized_flag_false(self):
        """standardized=False should be recorded."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=50, standardized=False, seed=42)
        assert eff.standardized is False

    def test_effects_same_regardless_of_standardized(self):
        """from_h2 draws the same effects regardless of standardized flag."""
        eff_true = AdditiveEffects.from_h2(h2=0.5, m=50, standardized=True, seed=42)
        eff_false = AdditiveEffects.from_h2(h2=0.5, m=50, standardized=False, seed=42)
        # Effects should be identical (same seed)
        np.testing.assert_array_equal(eff_true.effects, eff_false.effects)

    def test_from_array_standardized_flag(self):
        """from_array should respect standardized flag."""
        effects = np.random.randn(20)
        eff_true = AdditiveEffects.from_array(effects, standardized=True)
        eff_false = AdditiveEffects.from_array(effects, standardized=False)
        assert eff_true.standardized is True
        assert eff_false.standardized is False


class TestEffectMatrixShape:
    """Test effect matrix shape validation for 1-D vs 2-D."""

    def test_additive_1d_shape(self):
        """AdditiveEffects should have 1-D effect array."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=20, seed=42)
        assert eff.effects.ndim == 1
        assert eff.effects.shape == (20,)

    def test_multivariate_2d_shape(self):
        """MultivariateEffects should have 2-D effect array."""
        mv = MultivariateEffects.from_h2_rg(
            h2=[0.5, 0.3], rg=0.2, m=20, seed=42
        )
        assert mv.effects.ndim == 2
        assert mv.effects.shape == (20, 2)

    def test_sparse_1d_shape(self):
        """SparseEffects should have 1-D effect array."""
        eff = SparseEffects.from_h2(h2=0.5, m=30, k_causal=10, seed=42)
        assert eff.effects.ndim == 1
        assert eff.effects.shape == (30,)

    def test_additive_from_array_1d(self):
        """AdditiveEffects.from_array with 1-D input."""
        effects = np.random.randn(15)
        eff = AdditiveEffects.from_array(effects)
        assert eff.effects.ndim == 1
        assert eff.effects.shape == (15,)

    def test_multivariate_from_array_2d(self):
        """MultivariateEffects.from_array with 2-D input."""
        effects = np.random.randn(15, 3)
        mv = MultivariateEffects.from_array(effects)
        assert mv.effects.ndim == 2
        assert mv.effects.shape == (15, 3)

    def test_k_property_respects_ndim(self):
        """k property should return 1 for 1-D, shape[1] for 2-D."""
        eff_1d = AdditiveEffects.from_h2(h2=0.5, m=20, seed=42)
        assert eff_1d.k == 1
        mv_2d = MultivariateEffects.from_h2_rg(
            h2=[0.5, 0.3, 0.7], rg=0.1, m=20, seed=42
        )
        assert mv_2d.k == 3


class TestEffectsRepr:
    """Test __repr__ and __str__ output for effect classes."""

    def test_additive_repr(self):
        """AdditiveEffects repr should include m, m_causal, standardized."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=25, seed=42)
        r = repr(eff)
        assert "AdditiveEffects" in r
        assert "m=25" in r
        assert "m_causal=25" in r
        assert "standardized=True" in r

    def test_multivariate_repr(self):
        """MultivariateEffects repr should include m, k, m_causal, standardized."""
        mv = MultivariateEffects.from_h2_rg(
            h2=[0.5, 0.3], rg=0.2, m=30, seed=42
        )
        r = repr(mv)
        assert "MultivariateEffects" in r
        assert "m=30" in r
        assert "k=2" in r
        assert "m_causal=30" in r
        assert "standardized=True" in r

    def test_sparse_repr(self):
        """SparseEffects repr should include m, m_causal, standardized."""
        eff = SparseEffects.from_h2(h2=0.5, m=100, k_causal=20, seed=42)
        r = repr(eff)
        assert "SparseEffects" in r
        assert "m=100" in r
        assert "m_causal=20" in r
        assert "standardized=True" in r

    def test_repr_standardized_false(self):
        """Repr should show standardized=False when appropriate."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, standardized=False, seed=42)
        r = repr(eff)
        assert "standardized=False" in r


class TestMultivariateEffectsRgZero:
    """Test that rg=0 produces uncorrelated effects."""

    def test_rg_zero_two_traits(self):
        """rg=0 for two traits should yield uncorrelated genetic values."""
        n, m = 500, 500  # Large n and m for stable estimates
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        mv = MultivariateEffects.from_h2_rg(
            h2=[0.5, 0.5], rg=0.0, m=m, seed=42
        )
        # Compute genetic values for both traits (matvec handles 2D effects)
        g = hap.matvec(mv.effects)
        corr = np.corrcoef(g[:, 0], g[:, 1])[0, 1]
        # Should be close to 0
        assert abs(corr) < 0.15, \
            f"Expected genetic correlation ~0, got {corr}"

    def test_rg_zero_three_traits(self):
        """rg=0 for three traits should yield pairwise uncorrelated genetic values."""
        n, m = 500, 500
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        mv = MultivariateEffects.from_h2_rg(
            h2=[0.5, 0.4, 0.6], rg=0.0, m=m, seed=42
        )
        g = hap.matvec(mv.effects)
        corr_matrix = np.corrcoef(g.T)
        # Off-diagonal should be close to 0
        for i in range(3):
            for j in range(i + 1, 3):
                assert abs(corr_matrix[i, j]) < 0.15, \
                    f"Expected rg~0 for traits {i},{j}, got {corr_matrix[i, j]}"

    def test_rg_zero_vs_nonzero_different(self):
        """rg=0 vs rg=0.5 should produce different genetic correlations."""
        n, m = 500, 500
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        mv0 = MultivariateEffects.from_h2_rg(
            h2=[0.5, 0.5], rg=0.0, m=m, seed=42
        )
        mv5 = MultivariateEffects.from_h2_rg(
            h2=[0.5, 0.5], rg=0.5, m=m, seed=43
        )
        g0 = hap.matvec(mv0.effects)
        g5 = hap.matvec(mv5.effects)
        corr0 = np.corrcoef(g0[:, 0], g0[:, 1])[0, 1]
        corr5 = np.corrcoef(g5[:, 0], g5[:, 1])[0, 1]
        # Difference should be substantial
        assert abs(corr5 - corr0) > 0.3, \
            f"Expected different correlations, got {corr0} vs {corr5}"


class TestEffectsDtypeConsistency:
    """Test that effects and variant_mask have consistent dtypes."""

    def test_additive_effects_dtype(self):
        """AdditiveEffects.effects should be float64."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=20, seed=42)
        assert eff.effects.dtype == np.float64

    def test_multivariate_effects_dtype(self):
        """MultivariateEffects.effects should be float64."""
        mv = MultivariateEffects.from_h2_rg(
            h2=[0.5, 0.3], rg=0.2, m=20, seed=42
        )
        assert mv.effects.dtype == np.float64

    def test_sparse_effects_dtype(self):
        """SparseEffects.effects should be float64."""
        eff = SparseEffects.from_h2(h2=0.5, m=30, k_causal=10, seed=42)
        assert eff.effects.dtype == np.float64

    def test_variant_mask_dtype_bool(self):
        """All effect classes should have bool variant_mask."""
        eff_add = AdditiveEffects.from_h2(h2=0.5, m=20, seed=42)
        eff_mv = MultivariateEffects.from_h2_rg(
            h2=[0.5, 0.3], rg=0.2, m=20, seed=42
        )
        eff_sparse = SparseEffects.from_h2(h2=0.5, m=30, k_causal=10, seed=42)
        assert eff_add.variant_mask.dtype == bool
        assert eff_mv.variant_mask.dtype == bool
        assert eff_sparse.variant_mask.dtype == bool
