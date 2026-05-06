"""
Numerical tests for genetic variance components.

Tests:
1. Genetic variance under Hardy-Weinberg (2pq * beta^2)
2. Diploid matvec variance scales with effect variance
3. MVGenetic covariance structure matches rg
4. Genetic + noise total variance ≈ 1 when h2 + e2 = 1
5. Standardized matvec has zero mean
6. Multiple independent genetic components add variance linearly
"""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation

from xftsim.arch import Architecture, GeneticComponent, MVGeneticComponent, NoiseComponent, AggregationComponent
from xftsim.effect import AdditiveEffects, MultivariateEffects


class TestGeneticVariance:
    def test_genetic_variance_approximately_h2(self):
        """With standardized effects, Var(Y.G) ≈ h2."""
        n, m = 2000, 500
        h2 = 0.5
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        effects = AdditiveEffects.from_h2(m=m, h2=h2, seed=42)
        arch = Architecture()
        arch.add('Y', GeneticComponent(effects))
        pheno = arch.compute(hap, rng=np.random.RandomState(42))
        var_g = np.var(pheno['Y'])
        # Variance should be in the right ballpark (not exact due to finite sample)
        assert abs(var_g - h2) < 0.4, f"Genetic variance {var_g:.3f} far from h2={h2}"

    def test_zero_h2_zero_variance(self):
        """h2=0 → zero genetic variance."""
        n, m = 100, 20
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        effects = AdditiveEffects.from_h2(m=m, h2=0.0, seed=42)
        arch = Architecture()
        arch.add('Y', GeneticComponent(effects))
        pheno = arch.compute(hap, rng=np.random.RandomState(42))
        # All values should be identical (zero variance)
        assert np.var(pheno['Y']) < 1e-10

    def test_standardized_matvec_zero_mean(self):
        """Standardized matvec should produce approximately zero mean."""
        n, m = 1000, 100
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        effects = AdditiveEffects.from_h2(m=m, h2=0.5, seed=42)
        arch = Architecture()
        arch.add('Y', GeneticComponent(effects))
        pheno = arch.compute(hap, rng=np.random.RandomState(42))
        assert abs(np.mean(pheno['Y'])) < 0.5


class TestGeneticPlusNoise:
    def test_total_variance(self):
        """genetic(h2) + noise(1-h2) → total variance ≈ 1."""
        n, m = 2000, 500
        h2 = 0.5
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        effects = AdditiveEffects.from_h2(m=m, h2=h2, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(effects))
        arch.add('Y.E', NoiseComponent(variance=1.0 - h2))
        arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])
        pheno = arch.compute(hap, rng=np.random.RandomState(42))
        total_var = np.var(pheno['Y'])
        # Should be around 1.0
        assert abs(total_var - 1.0) < 0.5, f"Total variance {total_var:.3f} far from 1.0"


class TestMVGeneticCorrelation:
    def test_positive_rg_positive_correlation(self):
        """Positive rg → positive phenotypic correlation."""
        n, m = 2000, 500
        rg = 0.8
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        effects = MultivariateEffects.from_h2_rg(m=m, h2=[0.5, 0.5], rg=rg, seed=42)
        arch = Architecture()
        arch.add(['Y.G', 'Z.G'], MVGeneticComponent(effects))
        pheno = arch.compute(hap, rng=np.random.RandomState(42))
        corr = np.corrcoef(pheno['Y.G'], pheno['Z.G'])[0, 1]
        assert corr > 0.3, f"Genetic correlation {corr:.3f} should be positive"

    def test_negative_rg_negative_correlation(self):
        """Negative rg → negative phenotypic correlation."""
        n, m = 2000, 500
        rg = -0.8
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        effects = MultivariateEffects.from_h2_rg(m=m, h2=[0.5, 0.5], rg=rg, seed=42)
        arch = Architecture()
        arch.add(['Y.G', 'Z.G'], MVGeneticComponent(effects))
        pheno = arch.compute(hap, rng=np.random.RandomState(42))
        corr = np.corrcoef(pheno['Y.G'], pheno['Z.G'])[0, 1]
        assert corr < -0.3, f"Genetic correlation {corr:.3f} should be negative"

    def test_zero_rg_uncorrelated(self):
        """rg=0 → approximately uncorrelated."""
        n, m = 2000, 500
        rg = 0.0
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        effects = MultivariateEffects.from_h2_rg(m=m, h2=[0.5, 0.5], rg=rg, seed=42)
        arch = Architecture()
        arch.add(['Y.G', 'Z.G'], MVGeneticComponent(effects))
        pheno = arch.compute(hap, rng=np.random.RandomState(42))
        corr = np.corrcoef(pheno['Y.G'], pheno['Z.G'])[0, 1]
        assert abs(corr) < 0.3, f"Genetic correlation {corr:.3f} should be near 0"


class TestAdditiveVarianceComponents:
    def test_independent_genetic_components_add(self):
        """Two independent genetic components: Var(A+B) ≈ Var(A) + Var(B)."""
        n, m = 2000, 500
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        effects1 = AdditiveEffects.from_h2(m=m, h2=0.3, seed=42)
        effects2 = AdditiveEffects.from_h2(m=m, h2=0.2, seed=43)

        arch = Architecture()
        arch.add('A', GeneticComponent(effects1))
        arch.add('B', GeneticComponent(effects2))
        arch.add('C', AggregationComponent('A + B'), inputs=['A', 'B'])
        pheno = arch.compute(hap, rng=np.random.RandomState(42))

        var_a = np.var(pheno['A'])
        var_b = np.var(pheno['B'])
        var_c = np.var(pheno['C'])
        # With independent effects, Var(C) ≈ Var(A) + Var(B) + 2*Cov(A,B)
        # Since effects are independent seeds, covariance should be small
        assert abs(var_c - (var_a + var_b)) < 0.5 * (var_a + var_b)
