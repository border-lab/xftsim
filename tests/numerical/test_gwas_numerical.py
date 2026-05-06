"""
Numerical tests for GWAS and PGS modules.

Verifies:
1. GWAS recovers true effect sizes (approximately) in a simulation
2. PGS correlates with true genetic value
3. PGS r^2 approximates h^2
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray, NPhenotypeArray
from xftsim.effect import AdditiveEffects
from xftsim.arch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.gwas import GWAS, PGS

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


N = 5000
M = 100


def _build_sim(n=N, m=M, h2=0.5, seed=42):
    """Build haplotypes, true effects, genetic values, and phenotypes."""
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
    eff = AdditiveEffects.from_h2(h2=h2, m=m, seed=seed + 1)
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(eff))
    arch.add('Y.E', NoiseComponent(variance=1.0 - h2))
    arch.add('Y', AggregationComponent('Y.G + Y.E'))
    pheno = arch.compute(hap, rng=np.random.RandomState(seed + 2))
    return hap, eff, pheno


class TestGWASRecoversTrueEffects:
    """GWAS marginal beta should correlate with true effects."""

    def test_beta_correlation_positive(self):
        """
        With large N, the marginal GWAS betas should correlate positively
        with the true (standardized) effect sizes.
        """
        hap, eff, pheno = _build_sim(n=N, m=M, h2=0.5, seed=42)
        res = GWAS(hap, pheno).run(['Y'])
        r = res['Y']

        beta_true = eff.effects  # (m,) standardized effects
        beta_hat = r.beta

        corr = np.corrcoef(beta_true, beta_hat)[0, 1]
        assert corr > 0.3, (
            f"Correlation between true and estimated betas = {corr:.3f}, expected > 0.3"
        )

    def test_significant_variants_enriched(self):
        """
        Variants with larger true effects should be more likely to have
        small p-values.
        """
        hap, eff, pheno = _build_sim(n=N, m=M, h2=0.5, seed=43)
        res = GWAS(hap, pheno).run(['Y'])
        r = res['Y']

        # Split variants into top-half and bottom-half by |true beta|
        abs_beta = np.abs(eff.effects)
        median_beta = np.median(abs_beta)
        large = abs_beta >= median_beta
        small = abs_beta < median_beta

        mean_neglogp_large = np.mean(-np.log10(np.maximum(r.p_value[large], 1e-300)))
        mean_neglogp_small = np.mean(-np.log10(np.maximum(r.p_value[small], 1e-300)))

        assert mean_neglogp_large > mean_neglogp_small, (
            f"Large-effect -log10(p) = {mean_neglogp_large:.2f} should exceed "
            f"small-effect {mean_neglogp_small:.2f}"
        )

    def test_gwas_se_decreases_with_n(self):
        """Standard errors should decrease as sample size increases."""
        m = 20
        ses = []
        for n in [200, 1000, 5000]:
            hap, eff, pheno = _build_sim(n=n, m=m, h2=0.5, seed=44)
            res = GWAS(hap, pheno).run(['Y'])
            ses.append(np.nanmean(res['Y'].se))

        for i in range(len(ses) - 1):
            assert ses[i + 1] < ses[i], (
                f"Mean SE should decrease with N: {ses}"
            )


class TestPGSCorrelatesWithGeneticValue:
    """PGS should capture true genetic value."""

    def test_pgs_true_weights_correlation(self):
        """
        PGS using true effect weights should correlate highly with
        the true genetic value (Y.G).
        """
        hap, eff, pheno = _build_sim(n=N, m=M, h2=0.5, seed=50)

        # PGS using true standardized weights
        pgs = PGS(hap, weights=eff.effects, standardized=True)
        scores = pgs.score()

        g_true = pheno['Y.G']
        corr = np.corrcoef(scores, g_true)[0, 1]
        assert corr > 0.90, (
            f"PGS-true_G correlation = {corr:.3f}, expected > 0.90"
        )

    def test_pgs_gwas_weights_correlation(self):
        """
        PGS using GWAS-estimated weights should correlate positively
        with the true genetic value.
        """
        hap, eff, pheno = _build_sim(n=N, m=M, h2=0.5, seed=51)

        # Get GWAS weights
        res = GWAS(hap, pheno).run(['Y'])
        beta_hat = res['Y'].beta

        # PGS from GWAS betas (raw, not standardized — since GWAS beta is on raw scale)
        pgs = PGS(hap, weights=beta_hat, standardized=False)
        scores = pgs.score()

        g_true = pheno['Y.G']
        corr = np.corrcoef(scores, g_true)[0, 1]
        assert corr > 0.2, (
            f"PGS(GWAS)-true_G correlation = {corr:.3f}, expected > 0.2"
        )


class TestPGSR2ApproximatesH2:
    """PGS r^2 should approximate h^2 when using true weights."""

    def test_r2_close_to_h2(self):
        """
        With true effect weights, PGS r^2 with phenotype should be
        close to h^2 (since PGS = G_true up to centering).
        """
        h2 = 0.5
        hap, eff, pheno = _build_sim(n=N, m=M, h2=h2, seed=60)

        pgs = PGS(hap, weights=eff.effects, standardized=True)
        scores = pgs.score()

        y = pheno['Y']
        r = np.corrcoef(scores, y)[0, 1]
        r2 = r ** 2

        # r^2 should be in the neighborhood of h2
        # With m=100 variants and stochastic effect sampling, there can be
        # substantial deviation — allow generous tolerance
        assert abs(r2 - h2) < 0.30, (
            f"PGS r^2 = {r2:.3f}, h2 target = {h2}, difference = {abs(r2 - h2):.3f}"
        )

    def test_higher_h2_higher_r2(self):
        """Higher h2 should yield higher PGS r^2 with true weights."""
        r2s = []
        for h2 in [0.2, 0.5, 0.8]:
            hap, eff, pheno = _build_sim(n=N, m=M, h2=h2, seed=61)
            pgs = PGS(hap, weights=eff.effects, standardized=True)
            scores = pgs.score()
            y = pheno['Y']
            r = np.corrcoef(scores, y)[0, 1]
            r2s.append(r ** 2)

        for i in range(len(r2s) - 1):
            assert r2s[i + 1] > r2s[i], (
                f"r^2 should increase with h2: {r2s}"
            )


class TestGWASNullCalibration:
    """Under the null, GWAS statistics should be calibrated."""

    def test_null_median_p_near_half(self):
        """With Y = noise (h2=0), median p-value should be near 0.5."""
        n, m = 2000, 50
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=70)
        rng = np.random.RandomState(71)
        pheno = NPhenotypeArray(samples=hap.samples)
        pheno['Y'] = rng.normal(0, 1, size=n)

        res = GWAS(hap, pheno).run()
        median_p = np.median(res['Y'].p_value)
        assert 0.25 < median_p < 0.75, (
            f"Null median p = {median_p:.3f}, expected near 0.5"
        )

    def test_null_type1_error_controlled(self):
        """
        Under the null, proportion of p < 0.05 should be roughly 5%.
        """
        n, m = 5000, 200
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=72)
        rng = np.random.RandomState(73)
        pheno = NPhenotypeArray(samples=hap.samples)
        pheno['Y'] = rng.normal(0, 1, size=n)

        res = GWAS(hap, pheno).run()
        frac_sig = np.mean(res['Y'].p_value < 0.05)
        # Should be approximately 0.05, allow generous tolerance
        assert frac_sig < 0.15, (
            f"Fraction p<0.05 = {frac_sig:.3f}, expected ~0.05"
        )
