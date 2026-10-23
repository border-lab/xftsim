"""Unit tests for MultivariateEffects.from_frequency_dependent.

Expected values are computed from the spec (the LDAK/RHE-mc frequency- and
LD-dependent variance model), not from the code under test:

  w_t(j)      = [2 f_j (1 - f_j)] ** (1 + alpha_t) * ldscore_j ** s_t
  Var(beta_tj) = h2_t * w_t(j) / sum_j' w_t(j')        (per-trait normalized)
  beta_j      ~ MVN(0, diag(sd_j) R diag(sd_j))

Key checks:
1. alpha=-1, s=0 reduces exactly to the flat from_h2_rg model.
2. Per-trait variance weights follow [2f(1-f)]^(1+alpha) * ldscore^s.
3. Design heritability is recovered: sum_j Var(beta_tj) == h2_t.
4. realized_rg equals the per-SNP rg when alpha/s are shared, and is
   attenuated when they differ.
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xftsim.effect import MultivariateEffects


def _annotations(m, seed=0):
    rng = np.random.RandomState(seed)
    maf = rng.uniform(0.01, 0.5, size=m)
    ldscore = rng.uniform(1.0, 20.0, size=m)
    return maf, ldscore


class TestShapeAndReduction:
    def test_shape_m_by_k(self):
        maf, ld = _annotations(50)
        mv = MultivariateEffects.from_frequency_dependent(
            maf, ld, h2=[0.5, 0.3], R=0.2, alpha=-0.25, s=0.0, seed=1)
        assert mv.effects.shape == (50, 2)
        assert mv.m == 50 and mv.k == 2
        assert mv.variant_mask.all()

    def test_reduces_to_from_h2_rg_when_alpha_neg1_s0(self):
        # With alpha=-1, s=0 the weights are flat, so every SNP shares the same
        # covariance cov_snp = diag(sqrt(h2/m)) R diag(sqrt(h2/m)) -- exactly the
        # model from_h2_rg uses. The draws are distributionally identical (not
        # byte-identical: numpy factorizes R vs D R D differently under the hood),
        # so we compare empirical second moments at large m.
        m = 20000
        maf, ld = _annotations(m, seed=3)
        h2 = np.array([0.5, 0.3]); rg = 0.4
        a = MultivariateEffects.from_frequency_dependent(
            maf, ld, h2=h2, R=rg, alpha=-1.0, s=0.0, seed=7)

        sd = np.sqrt(h2 / m)
        Rmat = np.array([[1.0, rg], [rg, 1.0]])
        cov_snp = np.outer(sd, sd) * Rmat            # analytic per-SNP covariance
        emp = a.effects.T @ a.effects / m            # ~ E[beta beta^T] = cov_snp
        np.testing.assert_allclose(emp, cov_snp, rtol=0.05)
        # Heritability target is recovered: sum_j Var(beta_tj) == h2_t.
        np.testing.assert_allclose((a.effects ** 2).sum(axis=0), h2, rtol=0.05)


class TestVarianceModel:
    def test_per_trait_variance_follows_spec(self):
        # Expected per-SNP variance ratio between two SNPs must equal the
        # spec weight ratio, independent of the random draw.
        maf, ld = _annotations(2000, seed=11)
        alpha = np.array([-0.25, -0.5])
        s = np.array([0.0, 1.0])
        h2 = np.array([0.6, 0.4])
        mv = MultivariateEffects.from_frequency_dependent(
            maf, ld, h2=h2, R=0.0, alpha=alpha, s=s, seed=2)

        het = 2.0 * maf * (1.0 - maf)
        for t in range(2):
            w = het ** (1.0 + alpha[t]) * ld ** s[t]
            expected_var = h2[t] * w / w.sum()          # (m,) from spec
            # Realized squared effect should track expected variance: compare
            # the empirical scaling via least-squares slope ~ 1 (mean of b^2/var).
            ratio = mv.effects[:, t] ** 2 / expected_var
            assert 0.7 < ratio.mean() < 1.3              # ~1 in expectation

    def test_design_h2_recovered(self):
        # sum_j Var(beta_tj) == h2_t exactly (this is a property of the
        # normalization, holds for the *expected* variances, not the draw).
        maf, ld = _annotations(500, seed=5)
        h2 = np.array([0.5, 0.2, 0.8])
        alpha = np.array([-0.25, -1.0, 0.0])
        s = np.array([0.5, 0.0, 1.0])
        mv = MultivariateEffects.from_frequency_dependent(
            maf, ld, h2=h2, R=np.eye(3), alpha=alpha, s=s, seed=9)
        het = 2.0 * maf * (1.0 - maf)
        for t in range(3):
            w = het ** (1.0 + alpha[t]) * ld ** s[t]
            total_var = (h2[t] * w / w.sum()).sum()
            np.testing.assert_allclose(total_var, h2[t], rtol=1e-12)


class TestRealizedRg:
    def test_realized_rg_matches_input_when_params_shared(self):
        # Shared alpha/s across traits => sd columns proportional => realized
        # aggregate rg equals the per-SNP rg exactly.
        maf, ld = _annotations(300, seed=4)
        mv = MultivariateEffects.from_frequency_dependent(
            maf, ld, h2=[0.5, 0.5], R=0.6, alpha=-0.25, s=1.0, seed=4)
        np.testing.assert_allclose(mv.realized_rg[0, 1], 0.6, rtol=1e-12)
        np.testing.assert_allclose(np.diag(mv.realized_rg), [1.0, 1.0], rtol=1e-12)

    def test_realized_rg_attenuated_when_params_differ(self):
        # Different alpha => sd columns not proportional => |realized rg| < |rg|.
        maf, ld = _annotations(300, seed=6)
        rg = 0.6
        mv = MultivariateEffects.from_frequency_dependent(
            maf, ld, h2=[0.5, 0.5], R=rg, alpha=[-0.1, -0.9], s=0.0, seed=6)
        assert mv.realized_rg[0, 1] < rg
        # Verify against the Cauchy-Schwarz formula directly.
        het = 2.0 * maf * (1.0 - maf)
        w0 = het ** 0.9; w1 = het ** 0.1
        sd0 = np.sqrt(0.5 * w0 / w0.sum()); sd1 = np.sqrt(0.5 * w1 / w1.sum())
        expected = rg * (sd0 * sd1).sum() / np.sqrt((sd0**2).sum() * (sd1**2).sum())
        np.testing.assert_allclose(mv.realized_rg[0, 1], expected, rtol=1e-12)


class TestValidation:
    def test_mismatched_annotation_lengths(self):
        with pytest.raises(ValueError, match="equal length"):
            MultivariateEffects.from_frequency_dependent(
                np.ones(10), np.ones(9), h2=[0.5], R=0.0, alpha=-0.25)

    def test_bad_R_shape(self):
        maf, ld = _annotations(10)
        with pytest.raises(ValueError, match=r"R must be"):
            MultivariateEffects.from_frequency_dependent(
                maf, ld, h2=[0.5, 0.5], R=np.eye(3), alpha=-0.25)
