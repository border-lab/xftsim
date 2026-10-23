"""Unit tests for effect.compute_maf_ldscore.

Expected values come from the spec (LD score = sum_k r_jk^2 over the SNP's
chromosome block, with the Bulik-Sullivan 2015 bias correction
r2 -> r2 - (1 - r2)/(n - 2)), not from the code under test.
"""
import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from xftsim.effect import compute_maf_ldscore, MultivariateEffects


def _independent_genotypes(n, m, seed=0):
    rng = np.random.RandomState(seed)
    p = rng.uniform(0.05, 0.5, size=m)
    # Hardy-Weinberg: each genotype = sum of two Bernoulli(p) alleles.
    return (rng.binomial(1, p, (n, m)) + rng.binomial(1, p, (n, m))).astype(float), p


class TestMaf:
    def test_maf_recovers_frequency(self):
        n, m = 5000, 30
        G, p = _independent_genotypes(n, m, seed=1)
        maf, _ = compute_maf_ldscore(G)
        expected = np.minimum(p, 1 - p)
        np.testing.assert_allclose(maf, expected, atol=0.02)

    def test_maf_folded_to_half(self):
        # A column with allele freq 0.8 should fold to maf 0.2.
        G = np.array([[2, 2, 2, 2, 0], [2, 2, 0, 2, 2], [2, 0, 2, 2, 2]], float).T
        # column means /2 give p; maf = min(p, 1-p) <= 0.5 always
        maf, _ = compute_maf_ldscore(G)
        assert np.all(maf <= 0.5 + 1e-12)


class TestLdScore:
    def test_independent_snps_ldscore_near_one(self):
        # Unlinked SNPs: each LD score ~ 1 (self term) after bias correction.
        n, m = 8000, 40
        G, _ = _independent_genotypes(n, m, seed=2)
        _, ld = compute_maf_ldscore(G, bias_correct=True)
        # Mean LD score across independent SNPs should be close to 1.
        assert abs(ld.mean() - 1.0) < 0.05

    def test_perfectly_correlated_pair(self):
        # Two identical SNPs in one block: r2 ~ 1 with each other and self,
        # so each LD score ~ 2 (before correction; ~2 after, since r2=1 -> 1).
        rng = np.random.RandomState(3)
        g = (rng.binomial(2, 0.3, size=6000)).astype(float)
        G = np.column_stack([g, g])               # duplicate column
        _, ld = compute_maf_ldscore(G, bias_correct=True)
        np.testing.assert_allclose(ld, [2.0, 2.0], atol=1e-6)

    def test_matches_spec_formula(self):
        # Compute LD score by hand from the bias-corrected r2 matrix.
        n, m = 2000, 10
        G, _ = _independent_genotypes(n, m, seed=4)
        _, ld = compute_maf_ldscore(G, bias_correct=True)
        Z = (G - G.mean(0)) / G.std(0)
        r2 = (Z.T @ Z / n) ** 2
        r2 = r2 - (1 - r2) / (n - 2)
        np.testing.assert_allclose(ld, r2.sum(1), rtol=1e-12)

    def test_chrom_blocks_isolate_correlation(self):
        # Put two identical SNPs on *different* chromosomes: they must NOT
        # contribute to each other's LD score (only self term ~ 1 each).
        rng = np.random.RandomState(5)
        g = rng.binomial(2, 0.3, size=6000).astype(float)
        G = np.column_stack([g, g])
        chrom = np.array([1, 2])
        _, ld = compute_maf_ldscore(G, chrom=chrom, bias_correct=True)
        np.testing.assert_allclose(ld, [1.0, 1.0], atol=1e-6)

    def test_monomorphic_snp_zero(self):
        G, _ = _independent_genotypes(500, 5, seed=6)
        G[:, 2] = 2.0                              # monomorphic column
        maf, ld = compute_maf_ldscore(G)
        assert maf[2] == 0.0
        assert ld[2] == 0.0


class TestValidation:
    def test_too_few_individuals(self):
        with pytest.raises(ValueError, match="individuals"):
            compute_maf_ldscore(np.ones((2, 4)))


class TestEndToEnd:
    def test_feeds_from_frequency_dependent(self):
        # The helper output should be directly consumable downstream.
        n, m = 3000, 50
        G, _ = _independent_genotypes(n, m, seed=7)
        chrom = np.repeat([1, 2], m // 2)
        maf, ld = compute_maf_ldscore(G, chrom=chrom)
        mv = MultivariateEffects.from_frequency_dependent(
            maf, ld, h2=[0.4, 0.6], R=0.3, alpha=[-0.25, -0.5], s=1.0, seed=8)
        assert mv.effects.shape == (m, 2)
        assert np.isfinite(mv.effects).all()
