"""Unit tests for the LDAK/GCTA effect constructors and LDAK weights.

Expected values come from the spec (Speed et al. 2020, Nat Commun, Eq. 12),
not from the code under test:

  per-SNP heritability (standardized scale)  Var(beta_j) ∝ w_j^b * [2 f_j (1-f_j)]^a
  GCTA : a = b = 0  (flat per-SNP h2)
  LDAK : a = 0.75, b = 1
  LDAK weight w : non-negative LS solution of  R2 @ w = 1  within an LD block
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xftsim.effect import (compute_ldak_weights, AdditiveEffects,
                           MultivariateEffects)


def _independent_genotypes(n, m, seed=0):
    rng = np.random.RandomState(seed)
    p = rng.uniform(0.05, 0.5, size=m)
    return (rng.binomial(1, p, (n, m)) + rng.binomial(1, p, (n, m))).astype(float), p


# --------------------------------------------------------------------------- #
# compute_ldak_weights
# --------------------------------------------------------------------------- #
class TestLdakWeights:
    def test_independent_snps_weight_near_one(self):
        # Unlinked SNPs: R2 ~ I, so the NNLS solution of R2 w = 1 is w ~ 1.
        n, m = 8000, 40
        G, _ = _independent_genotypes(n, m, seed=2)
        _, w = compute_ldak_weights(G, bias_correct=True)
        assert abs(w.mean() - 1.0) < 0.05
        assert np.all(w >= 0.0)

    def test_perfectly_correlated_pair_shares_unit_weight(self):
        # Two identical SNPs in one block tag each other perfectly, so the
        # pair shares a total weight of 1 (each in [0, 1]).
        rng = np.random.RandomState(3)
        g = rng.binomial(2, 0.3, size=6000).astype(float)
        G = np.column_stack([g, g])
        _, w = compute_ldak_weights(G, bias_correct=True)
        np.testing.assert_allclose(w.sum(), 1.0, atol=1e-6)
        assert np.all((w >= -1e-9) & (w <= 1.0 + 1e-9))

    def test_chrom_blocks_isolate_tagging(self):
        # Identical SNPs on different chromosomes do NOT tag each other, so
        # each is isolated within its own block and gets weight ~1.
        rng = np.random.RandomState(5)
        g = rng.binomial(2, 0.3, size=6000).astype(float)
        G = np.column_stack([g, g])
        chrom = np.array([1, 2])
        _, w = compute_ldak_weights(G, chrom=chrom, bias_correct=True)
        np.testing.assert_allclose(w, [1.0, 1.0], atol=1e-6)

    def test_monomorphic_snp_zero(self):
        G, _ = _independent_genotypes(500, 5, seed=6)
        G[:, 2] = 2.0
        maf, w = compute_ldak_weights(G)
        assert maf[2] == 0.0
        assert w[2] == 0.0

    def test_too_few_individuals(self):
        with pytest.raises(ValueError, match="individuals"):
            compute_ldak_weights(np.ones((2, 4)))


# --------------------------------------------------------------------------- #
# GCTA model (univariate + multivariate)
# --------------------------------------------------------------------------- #
class TestGctaModel:
    def test_univariate_flat_per_snp_variance(self):
        # Under GCTA every SNP has the same expected variance h2/m regardless
        # of MAF; realized beta^2 should scale ~flat across MAF.
        rng = np.random.RandomState(0)
        maf = rng.uniform(0.01, 0.5, size=4000)
        eff = AdditiveEffects.from_gcta_model(maf, h2=0.5, seed=1)
        assert eff.effects.shape == (4000,)
        # design h2 recovered exactly (property of normalization)
        np.testing.assert_allclose(eff.effects.ravel() @ eff.effects.ravel(),
                                   0.5, rtol=0.1)
        np.testing.assert_allclose(eff.design_h2, 0.5)
        # No MAF tilt: mean beta^2 in rare vs common bins ~equal.
        rare = maf < 0.1
        mr = (eff.effects[rare] ** 2).mean()
        mc = (eff.effects[~rare] ** 2).mean()
        assert 0.7 < mr / mc < 1.3

    def test_univariate_matches_from_h2_distribution(self):
        # GCTA model == from_h2: both draw N(0, h2/m) per SNP. Compare second
        # moments at large m (draws differ in RNG path, not in distribution).
        m = 20000
        maf = np.random.RandomState(2).uniform(0.01, 0.5, size=m)
        g = AdditiveEffects.from_gcta_model(maf, h2=0.5, seed=7)
        np.testing.assert_allclose((g.effects ** 2).mean(), 0.5 / m, rtol=0.05)

    def test_multivariate_delegates_to_frequency_dependent(self):
        # from_gcta_model is from_frequency_dependent(alpha=-1, s=0): identical.
        maf = np.random.RandomState(4).uniform(0.01, 0.5, size=500)
        ld = np.ones_like(maf)
        a = MultivariateEffects.from_gcta_model(maf, h2=[0.4, 0.6], R=0.3, seed=9)
        b = MultivariateEffects.from_frequency_dependent(
            maf, ld, h2=[0.4, 0.6], R=0.3, alpha=-1.0, s=0.0, seed=9)
        np.testing.assert_array_equal(a.effects, b.effects)
        assert a.effects.shape == (500, 2)


# --------------------------------------------------------------------------- #
# LDAK model (univariate + multivariate)
# --------------------------------------------------------------------------- #
class TestLdakModel:
    def test_univariate_variance_follows_spec(self):
        # Realized beta^2 should track the spec variance w^b [2f(1-f)]^a.
        rng = np.random.RandomState(11)
        m = 3000
        maf = rng.uniform(0.01, 0.5, size=m)
        w = rng.uniform(0.2, 1.0, size=m)
        a, b, h2 = 0.75, 1.0, 0.6
        eff = AdditiveEffects.from_ldak_model(maf, w, h2=h2, a=a, b=b, seed=3)

        het = 2 * maf * (1 - maf)
        weight = (w ** b) * (het ** a)
        expected_var = h2 * weight / weight.sum()
        ratio = eff.effects ** 2 / expected_var
        assert 0.8 < ratio.mean() < 1.2
        np.testing.assert_allclose(eff.design_h2, h2)

    def test_ldak_reduces_to_gcta_when_a_b_zero(self):
        # a=b=0 => flat weights => identical to GCTA model with same seed.
        maf = np.random.RandomState(1).uniform(0.01, 0.5, size=800)
        w = np.random.RandomState(2).uniform(0.2, 1.0, size=800)
        ldak = AdditiveEffects.from_ldak_model(maf, w, h2=0.5, a=0.0, b=0.0, seed=5)
        gcta = AdditiveEffects.from_gcta_model(maf, h2=0.5, seed=5)
        np.testing.assert_allclose(ldak.effects, gcta.effects)

    def test_ldak_with_unit_weights_is_pure_maf_model(self):
        # w = 1 everywhere => Var ∝ [2f(1-f)]^a only (the MAF model).
        rng = np.random.RandomState(6)
        m = 2000
        maf = rng.uniform(0.01, 0.5, size=m)
        w = np.ones(m)
        eff = AdditiveEffects.from_ldak_model(maf, w, h2=0.7, a=0.75, b=1.0, seed=8)
        het = 2 * maf * (1 - maf)
        expected_var = 0.7 * (het ** 0.75) / (het ** 0.75).sum()
        ratio = eff.effects ** 2 / expected_var
        assert 0.8 < ratio.mean() < 1.2

    def test_multivariate_shape_and_rg(self):
        rng = np.random.RandomState(7)
        maf = rng.uniform(0.01, 0.5, size=600)
        w = rng.uniform(0.2, 1.0, size=600)
        mv = MultivariateEffects.from_ldak_model(
            maf, w, h2=[0.5, 0.5], R=0.6, a=0.75, b=1.0, seed=4)
        assert mv.effects.shape == (600, 2)
        # Shared a/b across traits => realized aggregate rg == per-SNP rg.
        np.testing.assert_allclose(mv.realized_rg[0, 1], 0.6, rtol=1e-12)
        np.testing.assert_allclose(mv.design_h2, [0.5, 0.5])


# --------------------------------------------------------------------------- #
# End-to-end: annotations feed the constructors
# --------------------------------------------------------------------------- #
class TestEndToEnd:
    def test_weights_feed_ldak_constructor(self):
        n, m = 3000, 60
        G, _ = _independent_genotypes(n, m, seed=7)
        chrom = np.repeat([1, 2], m // 2)
        maf, w = compute_ldak_weights(G, chrom=chrom)
        mv = MultivariateEffects.from_ldak_model(
            maf, w, h2=[0.4, 0.6], R=0.3, a=0.75, b=1.0, seed=8)
        assert mv.effects.shape == (m, 2)
        assert np.isfinite(mv.effects).all()
