"""
Numerical tests for covariance and variance properties.

Stochastic protocol: tolerance ~ 4/sqrt(N), N=10000.
Seeds are logged on failure for reproducibility.
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray, NPhenotypeArray
from xftsim.narch import (
    Architecture, GeneticComponent, MVGeneticComponent, NoiseComponent,
    CNoiseComponent, AggregationComponent,
)
from xftsim.neffect import AdditiveEffects, MultivariateEffects

N = 10000
M = 100
TOL = 4.0 / np.sqrt(N)  # ~0.04


def _make_haplotypes(n=N, m=M, seed=42):
    rng = np.random.RandomState(seed)
    af = np.full(m, 0.5)
    geno = np.zeros((n, m, 2), dtype=np.int8)
    for j in range(m):
        geno[:, j, 0] = rng.binomial(1, af[j], size=n)
        geno[:, j, 1] = rng.binomial(1, af[j], size=n)
    sex = np.tile([0, 1], (n + 1) // 2)[:n]
    samples = SampleMeta(iid=np.arange(n), sex=sex)
    variants = VariantMeta(vid=np.arange(m), af=af)
    return DenseHaplotypeArray(genotypes=geno, samples=samples, variants=variants)


class TestIndependentComponents:
    def test_independent_noise_zero_cross_cov(self, stochastic_seed):
        """Two independent noise components should have ~0 cross-covariance."""
        rng = stochastic_seed.rng
        n = N
        e1 = rng.normal(0, 1.0, size=n)
        e2 = rng.normal(0, 1.0, size=n)
        cross_cov = np.abs(np.cov(e1, e2)[0, 1])
        assert cross_cov < TOL, f"seed={stochastic_seed.seed}, cross_cov={cross_cov}"

    def test_noise_variance_matches(self, stochastic_seed):
        """Noise(var) should produce samples with variance ≈ var."""
        hap = _make_haplotypes(seed=stochastic_seed.seed)
        target_var = 0.7
        arch = Architecture()
        arch.add('E', NoiseComponent(variance=target_var))
        pheno = arch.compute(hap, rng=stochastic_seed.rng)
        obs_var = np.var(pheno['E'])
        assert abs(obs_var - target_var) < TOL, (
            f"seed={stochastic_seed.seed}, obs={obs_var}, target={target_var}"
        )


class TestCNoiseCovariance:
    def test_cnoise_cov_matches(self, stochastic_seed):
        """Correlated noise covariance should match target."""
        target_cov = np.array([[1.0, 0.3], [0.3, 0.5]])
        hap = _make_haplotypes(seed=stochastic_seed.seed)
        arch = Architecture()
        arch.add(['E1', 'E2'], CNoiseComponent(cov=target_cov))
        pheno = arch.compute(hap, rng=stochastic_seed.rng)
        data = np.column_stack([pheno['E1'], pheno['E2']])
        obs_cov = np.cov(data, rowvar=False)
        np.testing.assert_allclose(obs_cov, target_cov, atol=3 * TOL)


class TestVarianceDecomposition:
    def test_var_sum_rule(self, stochastic_seed):
        """Var(A+B) ≈ Var(A) + Var(B) when A,B independent."""
        hap = _make_haplotypes(seed=stochastic_seed.seed)
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=0.5))
        arch.add('B', NoiseComponent(variance=0.3))
        arch.add('Y', AggregationComponent('A + B'))
        pheno = arch.compute(hap, rng=stochastic_seed.rng)
        var_y = np.var(pheno['Y'])
        expected = 0.5 + 0.3
        assert abs(var_y - expected) < 3 * TOL

    def test_genetic_variance_h2(self, stochastic_seed):
        """Genetic variance should approximate h2 under non-standardized effects.

        With uniform AF=0.5, diploid genotype var per SNP is 2*p*q = 0.5.
        So Var(G) ≈ 2*p*q * m * E[beta^2] = 0.5 * sum(beta^2).
        We use non-standardized effects so the relation is clean.
        """
        h2 = 0.5
        hap = _make_haplotypes(seed=stochastic_seed.seed)
        eff = AdditiveEffects.from_h2(h2=h2, m=M, seed=stochastic_seed.seed,
                                       standardized=False)
        arch = Architecture()
        arch.add('G', GeneticComponent(eff))
        pheno = arch.compute(hap, rng=stochastic_seed.rng)
        var_g = np.var(pheno['G'])
        # Expected: sum(2*p*q * beta^2) where p=q=0.5 → 0.5 * sum(beta^2)
        expected = 0.5 * np.sum(eff.effects**2)
        # Wider tolerance: genetic variance has higher sampling error
        assert abs(var_g - expected) / max(expected, 0.01) < 0.5


class TestGroupedNoise:
    def test_within_group_zero_variance(self):
        """Grouped noise should give identical values within each group (exact)."""
        n = 100
        fid = np.repeat(np.arange(20), 5)
        sex = np.tile([0, 1], 50)
        samples = SampleMeta(iid=np.arange(n), fid=fid, sex=sex)
        m = 10
        geno = np.zeros((n, m, 2), dtype=np.int8)
        variants = VariantMeta(vid=np.arange(m), af=np.full(m, 0.5))
        hap = DenseHaplotypeArray(genotypes=geno, samples=samples, variants=variants)
        arch = Architecture()
        arch.add('E', NoiseComponent(variance=1.0), grouping='FID')
        pheno = arch.compute(hap, rng=np.random.RandomState(42))
        # All members of same family should have identical noise
        for f in range(20):
            mask = fid == f
            vals = pheno['E'][mask]
            assert np.all(vals == vals[0])

    def test_between_group_variance(self, stochastic_seed):
        """Between-group variance should match the noise variance parameter."""
        n = 10000
        n_groups = 5000
        fid = np.repeat(np.arange(n_groups), n // n_groups)[:n]
        sex = np.tile([0, 1], n // 2)[:n]
        samples = SampleMeta(iid=np.arange(n), fid=fid, sex=sex)
        m = 5
        geno = np.zeros((n, m, 2), dtype=np.int8)
        variants = VariantMeta(vid=np.arange(m), af=np.full(m, 0.5))
        hap = DenseHaplotypeArray(genotypes=geno, samples=samples, variants=variants)
        target_var = 0.7
        arch = Architecture()
        arch.add('E', NoiseComponent(variance=target_var), grouping='FID')
        pheno = arch.compute(hap, rng=stochastic_seed.rng)
        # Get one value per group
        group_vals = pheno['E'][::2]  # each group has 2 members
        obs_var = np.var(group_vals)
        assert abs(obs_var - target_var) < 3 * TOL


class TestMVGeneticCovariance:
    def test_mvgenetic_cross_covariance(self, stochastic_seed):
        """mvGenetic cross-covariance should approximate the target rg."""
        h2 = [0.5, 0.3]
        rg = 0.3
        hap = _make_haplotypes(seed=stochastic_seed.seed)
        eff = MultivariateEffects.from_h2_rg(h2=h2, rg=rg, m=M, seed=stochastic_seed.seed)
        arch = Architecture()
        arch.add(['G1', 'G2'], MVGeneticComponent(eff))
        pheno = arch.compute(hap, rng=stochastic_seed.rng)
        data = np.column_stack([pheno['G1'], pheno['G2']])
        obs_cov = np.cov(data, rowvar=False)
        # Target: genetic covariance = rg * sqrt(h2_1 * h2_2)
        target_cov12 = rg * np.sqrt(h2[0] * h2[1])
        assert abs(obs_cov[0, 1] - target_cov12) < 5 * TOL
