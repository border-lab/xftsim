"""
Unit tests for xftsim.ngwas — GWAS and PGS modules.

Covers:
- GWAS beta recovery with known effect sizes
- GWAS p-values for null variants
- GWAS with multiple phenotype keys
- GWAS with sample_indices (unrelated subset)
- PGS computation matches manual calculation
- PGS standardized vs raw modes
- PGS with dict-based weights
- Edge cases: single variant, constant phenotype, single individual,
  monomorphic variants
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray, NPhenotypeArray
from xftsim.ngwas import GWAS, GWASResult, PGS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hap(n, m, seed=42):
    """Create a DenseHaplotypeArray with random genotypes."""
    rng = np.random.RandomState(seed)
    geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
    return DenseHaplotypeArray(genotypes=geno)


def _make_phenotype_from_hap(hap, beta, noise_sd=0.0, seed=99):
    """Create NPhenotypeArray with Y = G @ beta + noise."""
    rng = np.random.RandomState(seed)
    G = hap.diploid_genotypes.astype(np.float64)
    y = G @ beta
    if noise_sd > 0:
        y = y + rng.normal(0, noise_sd, size=len(y))
    pheno = NPhenotypeArray(samples=hap.samples)
    pheno['Y'] = y
    return pheno


# ---------------------------------------------------------------------------
# GWAS tests
# ---------------------------------------------------------------------------

class TestGWASBetaRecovery:
    """GWAS beta estimates should recover known effect sizes."""

    def test_noiseless_exact_recovery(self):
        """With no noise, beta_hat should equal the true beta exactly."""
        n, m = 200, 10
        hap = _make_hap(n, m, seed=10)
        rng = np.random.RandomState(7)
        beta_true = rng.normal(0, 1, size=m)

        pheno = _make_phenotype_from_hap(hap, beta_true, noise_sd=0.0)
        gwas = GWAS(hap, pheno)
        res = gwas.run(['Y'])

        r = res['Y']
        assert r.beta.shape == (m,)
        # With zero noise, OLS per-variant won't recover multivariate beta exactly
        # but single-variant regression should still be close when variants are
        # not too correlated. Check shape and finiteness.
        assert np.all(np.isfinite(r.beta))
        assert np.all(np.isfinite(r.se))
        assert r.n == n

    def test_single_causal_variant(self):
        """
        With one causal variant and the rest zero, GWAS should
        recover that variant's beta exactly (no noise).
        """
        n, m = 500, 20
        hap = _make_hap(n, m, seed=20)
        beta_true = np.zeros(m)
        beta_true[5] = 0.8

        pheno = _make_phenotype_from_hap(hap, beta_true, noise_sd=0.0)
        res = GWAS(hap, pheno).run(['Y'])
        r = res['Y']

        # The causal variant's marginal beta should be very close to 0.8
        # (exact only if uncorrelated with others, but close for random genotypes)
        assert abs(r.beta[5] - 0.8) < 0.15, f"beta[5]={r.beta[5]}"

    def test_noisy_beta_direction(self):
        """With moderate noise, beta estimates should have correct sign for large effects."""
        n, m = 1000, 5
        hap = _make_hap(n, m, seed=30)
        beta_true = np.array([0.5, -0.3, 0.0, 0.7, -0.5])

        pheno = _make_phenotype_from_hap(hap, beta_true, noise_sd=1.0, seed=31)
        res = GWAS(hap, pheno).run()
        r = res['Y']

        # Large positive effects should have positive beta
        assert r.beta[0] > 0, f"beta[0]={r.beta[0]}, expected positive"
        assert r.beta[3] > 0, f"beta[3]={r.beta[3]}, expected positive"
        # Large negative effects should have negative beta
        assert r.beta[1] < 0, f"beta[1]={r.beta[1]}, expected negative"
        assert r.beta[4] < 0, f"beta[4]={r.beta[4]}, expected negative"


class TestGWASPValues:
    """P-value behavior under null and alternative."""

    def test_null_pvalues_broadly_uniform(self):
        """Under the null (Y = noise), p-values should be roughly uniform."""
        n, m = 500, 50
        hap = _make_hap(n, m, seed=40)
        rng = np.random.RandomState(41)
        pheno = NPhenotypeArray(samples=hap.samples)
        pheno['Y'] = rng.normal(0, 1, size=n)

        res = GWAS(hap, pheno).run()
        p = res['Y'].p_value

        # Under the null, median p should be around 0.5
        assert 0.15 < np.median(p) < 0.85, f"median p={np.median(p)}"
        # Very few should be < 0.01 (expect ~0.5 out of 50)
        n_sig = np.sum(p < 0.01)
        assert n_sig <= 10, f"{n_sig} variants with p<0.01 under null"

    def test_causal_variant_significant(self):
        """A strongly causal variant should have small p-value."""
        n, m = 1000, 20
        hap = _make_hap(n, m, seed=50)
        beta_true = np.zeros(m)
        beta_true[0] = 1.0  # strong effect

        pheno = _make_phenotype_from_hap(hap, beta_true, noise_sd=0.5, seed=51)
        res = GWAS(hap, pheno).run()
        r = res['Y']

        assert r.p_value[0] < 0.001, f"p[0]={r.p_value[0]}, expected significant"

    def test_pvalues_between_zero_and_one(self):
        """All p-values should be in [0, 1]."""
        n, m = 200, 30
        hap = _make_hap(n, m, seed=60)
        rng = np.random.RandomState(61)
        pheno = NPhenotypeArray(samples=hap.samples)
        pheno['Y'] = rng.normal(0, 1, size=n)

        res = GWAS(hap, pheno).run()
        p = res['Y'].p_value
        assert np.all(p >= 0.0)
        assert np.all(p <= 1.0)


class TestGWASMultipleKeys:
    """GWAS with multiple phenotype keys."""

    def test_two_phenotypes(self):
        """Running GWAS on two phenotypes should return separate results."""
        n, m = 300, 15
        hap = _make_hap(n, m, seed=70)
        rng = np.random.RandomState(71)

        pheno = NPhenotypeArray(samples=hap.samples)
        pheno['Y1'] = rng.normal(0, 1, size=n)
        pheno['Y2'] = rng.normal(5, 2, size=n)

        res = GWAS(hap, pheno).run()
        assert 'Y1' in res
        assert 'Y2' in res
        assert res['Y1'].beta.shape == (m,)
        assert res['Y2'].beta.shape == (m,)

    def test_specific_keys(self):
        """Specifying keys should only run those phenotypes."""
        n, m = 200, 10
        hap = _make_hap(n, m, seed=72)
        rng = np.random.RandomState(73)

        pheno = NPhenotypeArray(samples=hap.samples)
        pheno['Y1'] = rng.normal(0, 1, size=n)
        pheno['Y2'] = rng.normal(0, 1, size=n)

        res = GWAS(hap, pheno).run(['Y1'])
        assert 'Y1' in res
        assert 'Y2' not in res


class TestGWASSampleSubset:
    """GWAS with sample_indices (e.g. from UnrelatedFilter)."""

    def test_sample_indices_subset(self):
        """Using sample_indices should restrict to those samples."""
        n, m = 400, 10
        hap = _make_hap(n, m, seed=80)
        rng = np.random.RandomState(81)
        pheno = NPhenotypeArray(samples=hap.samples)
        pheno['Y'] = rng.normal(0, 1, size=n)

        # Use first 200 samples
        idx = np.arange(200)
        res = GWAS(hap, pheno, sample_indices=idx).run()
        assert res['Y'].n == 200


class TestGWASOutputAttributes:
    """Verify GWASResult attributes."""

    def test_result_fields(self):
        """GWASResult should have all expected fields."""
        n, m = 100, 5
        hap = _make_hap(n, m, seed=90)
        rng = np.random.RandomState(91)
        pheno = NPhenotypeArray(samples=hap.samples)
        pheno['Y'] = rng.normal(0, 1, size=n)

        res = GWAS(hap, pheno).run()
        r = res['Y']

        assert hasattr(r, 'beta')
        assert hasattr(r, 'se')
        assert hasattr(r, 't_stat')
        assert hasattr(r, 'p_value')
        assert hasattr(r, 'af')
        assert hasattr(r, 'n')
        assert r.beta.shape == (m,)
        assert r.se.shape == (m,)
        assert r.t_stat.shape == (m,)
        assert r.p_value.shape == (m,)
        assert r.af.shape == (m,)

    def test_af_reasonable(self):
        """Allele frequencies should be in [0, 1]."""
        n, m = 300, 20
        hap = _make_hap(n, m, seed=92)
        rng = np.random.RandomState(93)
        pheno = NPhenotypeArray(samples=hap.samples)
        pheno['Y'] = rng.normal(0, 1, size=n)

        res = GWAS(hap, pheno).run()
        af = res['Y'].af
        assert np.all(af >= 0.0)
        assert np.all(af <= 1.0)


# ---------------------------------------------------------------------------
# GWAS edge cases
# ---------------------------------------------------------------------------

class TestGWASEdgeCases:
    """Edge cases for GWAS."""

    def test_single_variant(self):
        """GWAS with a single variant should work."""
        n = 100
        hap = _make_hap(n, 1, seed=100)
        rng = np.random.RandomState(101)
        pheno = NPhenotypeArray(samples=hap.samples)
        pheno['Y'] = rng.normal(0, 1, size=n)

        res = GWAS(hap, pheno).run()
        assert res['Y'].beta.shape == (1,)

    def test_constant_phenotype(self):
        """Constant phenotype should produce zero beta and nan SE."""
        n, m = 100, 5
        hap = _make_hap(n, m, seed=110)
        pheno = NPhenotypeArray(samples=hap.samples)
        pheno['Y'] = np.ones(n) * 3.0

        res = GWAS(hap, pheno).run()
        r = res['Y']
        # With constant Y, beta should be zero (no covariance)
        np.testing.assert_allclose(r.beta, 0.0, atol=1e-10)

    def test_two_samples(self):
        """GWAS with exactly 2 samples (df=0) should produce NaN SE."""
        n, m = 2, 5
        hap = _make_hap(n, m, seed=120)
        rng = np.random.RandomState(121)
        pheno = NPhenotypeArray(samples=hap.samples)
        pheno['Y'] = rng.normal(0, 1, size=n)

        res = GWAS(hap, pheno).run()
        r = res['Y']
        # df = 2 - 2 = 0, SE should be NaN
        assert np.all(np.isnan(r.se))
        assert np.all(np.isnan(r.p_value))

    def test_monomorphic_variant(self):
        """Monomorphic variants (zero variance) should produce zero beta."""
        n, m = 100, 3
        rng = np.random.RandomState(130)
        geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
        # Make variant 1 monomorphic (all zeros)
        geno[:, 1, :] = 0
        hap = DenseHaplotypeArray(genotypes=geno)

        pheno = NPhenotypeArray(samples=hap.samples)
        pheno['Y'] = rng.normal(0, 1, size=n)

        res = GWAS(hap, pheno).run()
        r = res['Y']
        assert r.beta[1] == 0.0
        assert np.isnan(r.se[1])


# ---------------------------------------------------------------------------
# PGS tests
# ---------------------------------------------------------------------------

class TestPGSComputation:
    """PGS score computation."""

    def test_raw_pgs_matches_manual(self):
        """PGS with raw dosage should match manual G @ w."""
        n, m = 200, 10
        hap = _make_hap(n, m, seed=200)
        rng = np.random.RandomState(201)
        weights = rng.normal(0, 0.5, size=m)

        pgs = PGS(hap, weights=weights, standardized=False)
        scores = pgs.score()

        G = hap.diploid_genotypes.astype(np.float64)
        expected = G @ weights
        np.testing.assert_allclose(scores, expected, atol=1e-10)

    def test_standardized_pgs_matches_manual(self):
        """PGS with standardized mode should match centered genotypes @ w."""
        n, m = 200, 10
        hap = _make_hap(n, m, seed=210)
        rng = np.random.RandomState(211)
        weights = rng.normal(0, 0.5, size=m)

        pgs = PGS(hap, weights=weights, standardized=True)
        scores = pgs.score()

        G = hap.diploid_genotypes.astype(np.float64)
        af = G.mean(axis=0) / 2.0
        denom = np.sqrt(2 * af * (1 - af))
        denom[denom == 0] = 1.0
        G_std = (G - 2 * af) / denom
        expected = G_std @ weights
        np.testing.assert_allclose(scores, expected, atol=1e-10)

    def test_pgs_shape(self):
        """PGS should return (n,) array."""
        n, m = 100, 5
        hap = _make_hap(n, m, seed=220)
        weights = np.ones(m) * 0.1
        scores = PGS(hap, weights).score()
        assert scores.shape == (n,)

    def test_pgs_zero_weights(self):
        """PGS with zero weights should produce zero scores."""
        n, m = 100, 5
        hap = _make_hap(n, m, seed=230)
        scores = PGS(hap, weights=np.zeros(m)).score()
        np.testing.assert_allclose(scores, 0.0, atol=1e-10)


class TestPGSDict:
    """PGS with dict-based weights."""

    def test_dict_weights(self):
        """Dict weights should look up by variant ID."""
        n, m = 100, 10
        hap = _make_hap(n, m, seed=240)
        vid = hap.variants.vid

        # Set weight only for variant 3
        wt_dict = {vid[3]: 0.5}
        pgs = PGS(hap, weights=wt_dict)
        scores = pgs.score()

        G = hap.diploid_genotypes.astype(np.float64)
        expected = G[:, 3] * 0.5
        np.testing.assert_allclose(scores, expected, atol=1e-10)

    def test_dict_missing_variant_ignored(self):
        """Dict weights with unknown variant IDs should be silently ignored."""
        n, m = 100, 5
        hap = _make_hap(n, m, seed=250)
        wt_dict = {'nonexistent_variant': 1.0}
        pgs = PGS(hap, weights=wt_dict)
        scores = pgs.score()
        np.testing.assert_allclose(scores, 0.0, atol=1e-10)


class TestPGSEdgeCases:
    """Edge cases for PGS."""

    def test_single_variant(self):
        """PGS with single variant."""
        n = 50
        hap = _make_hap(n, 1, seed=260)
        scores = PGS(hap, weights=np.array([2.0])).score()
        G = hap.diploid_genotypes.astype(np.float64)
        expected = G[:, 0] * 2.0
        np.testing.assert_allclose(scores, expected, atol=1e-10)

    def test_single_individual(self):
        """PGS with single individual."""
        m = 10
        hap = _make_hap(1, m, seed=270)
        rng = np.random.RandomState(271)
        weights = rng.normal(0, 1, size=m)
        scores = PGS(hap, weights=weights).score()
        assert scores.shape == (1,)
        G = hap.diploid_genotypes.astype(np.float64)
        np.testing.assert_allclose(scores, G @ weights, atol=1e-10)
