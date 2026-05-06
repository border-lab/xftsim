"""
Numerical tests for effect size distributions.

Tests:
1. AdditiveEffects.from_h2: effect variance matches theoretical expectation
2. MultivariateEffects.from_h2_rg: genetic covariance structure
3. SparseEffects: only k_causal effects are nonzero
4. Effect-genotype product variance: Var(G@beta) ≈ h2 under standardization
"""
import numpy as np
import pytest

from xftsim.effect import AdditiveEffects, MultivariateEffects, SparseEffects

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestAdditiveEffectVariance:
    def test_genetic_variance_matches_h2(self):
        """For standardized effects, Var(Gβ) should ≈ h2."""
        m = 500
        n = 2000
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, standardized=True, seed=42)
        gv = hap.standardized_matvec(eff.effects)
        observed_var = np.var(gv)
        # With finite m and n, expect approximate match
        assert abs(observed_var - 0.5) < 0.35, f"Genetic variance {observed_var} far from 0.5"

    def test_effect_sum_of_squares(self):
        """Sum of effect squares should be approximately h2/m for standardized."""
        m = 100
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, standardized=True, seed=42)
        # For standardized effects drawn from N(0, h2/m):
        # E[sum(beta^2)] ≈ h2
        ss = np.sum(eff.effects ** 2)
        assert abs(ss - 0.5) < 0.3, f"Sum of squares {ss} far from h2=0.5"


class TestMultivariateEffectStructure:
    def test_genetic_covariance_positive(self):
        """Positive rg should produce positive genetic covariance."""
        m = 200
        n = 1000
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.5, m=m, seed=42)
        gv = hap.standardized_matvec(eff.effects)  # (n, 2)
        cov = np.cov(gv.T)
        assert cov[0, 1] > 0, f"Expected positive genetic covariance, got {cov[0, 1]}"

    def test_genetic_covariance_negative(self):
        """Negative rg should produce negative genetic covariance."""
        m = 200
        n = 1000
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=-0.5, m=m, seed=42)
        gv = hap.standardized_matvec(eff.effects)
        cov = np.cov(gv.T)
        assert cov[0, 1] < 0, f"Expected negative genetic covariance, got {cov[0, 1]}"

    def test_genetic_covariance_zero_rg(self):
        """rg=0 should produce approximately zero genetic covariance."""
        m = 200
        n = 1000
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.0, m=m, seed=42)
        gv = hap.standardized_matvec(eff.effects)
        cov = np.cov(gv.T)
        assert abs(cov[0, 1]) < 0.1, f"Expected ~0 covariance, got {cov[0, 1]}"


class TestSparseEffectStructure:
    def test_only_k_causal_nonzero(self):
        """Only k_causal variants should have nonzero effects."""
        m = 100
        k = 10
        eff = SparseEffects.from_h2(h2=0.5, m=m, k_causal=k, seed=42)
        n_nonzero = np.sum(eff.effects != 0)
        assert n_nonzero == k

    def test_variant_mask_matches(self):
        """variant_mask should be True exactly where effects are nonzero."""
        m = 100
        k = 20
        eff = SparseEffects.from_h2(h2=0.5, m=m, k_causal=k, seed=42)
        np.testing.assert_array_equal(eff.variant_mask, eff.effects != 0)

    def test_sparse_genetic_variance(self):
        """Sparse effects should still produce reasonable genetic variance."""
        m = 200
        k = 20
        n = 1000
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = SparseEffects.from_h2(h2=0.5, m=m, k_causal=k, standardized=True, seed=42)
        gv = hap.standardized_matvec(eff.effects)
        var = np.var(gv)
        assert var > 0, "Sparse effects should produce nonzero variance"
        assert np.isfinite(var), "Variance should be finite"


class TestEffectDeterminism:
    def test_same_seed_same_effects(self):
        """Same seed should produce identical effects."""
        e1 = AdditiveEffects.from_h2(h2=0.5, m=50, seed=42)
        e2 = AdditiveEffects.from_h2(h2=0.5, m=50, seed=42)
        np.testing.assert_array_equal(e1.effects, e2.effects)

    def test_different_seed_different_effects(self):
        """Different seeds should produce different effects."""
        e1 = AdditiveEffects.from_h2(h2=0.5, m=50, seed=42)
        e2 = AdditiveEffects.from_h2(h2=0.5, m=50, seed=43)
        assert not np.allclose(e1.effects, e2.effects)

    def test_multivariate_same_seed(self):
        e1 = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.4, m=50, seed=42)
        e2 = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.4, m=50, seed=42)
        np.testing.assert_array_equal(e1.effects, e2.effects)
