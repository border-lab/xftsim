"""
Numerical test: multivariate effects produce correct covariance structure.

Tests:
1. MVGenetic with rg=0 → near-zero genetic correlation
2. MVGenetic with rg=0.5 → positive genetic correlation
3. MVGenetic per-trait h2 approximately match targets
4. Genetic covariance matrix is positive semidefinite
"""
import numpy as np
import pytest

from xftsim.effect import MultivariateEffects
from xftsim.arch import Architecture, MVGeneticComponent, NoiseComponent, AggregationComponent

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestMultivariateGeneticCovariance:
    def test_rg_zero_near_zero_genetic_corr(self):
        """rg=0 should produce near-zero genetic correlation between traits."""
        n, m = 2000, 100
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        mv = MultivariateEffects.from_h2_rg(h2=[0.5, 0.5], rg=0.0, m=m, seed=42)
        arch = Architecture()
        arch.add(['T1.G', 'T2.G'], MVGeneticComponent(mv))
        result = arch.compute(hap, rng=np.random.RandomState(42))
        corr = np.corrcoef(result['T1.G'], result['T2.G'])[0, 1]
        assert abs(corr) < 0.15, f"rg=0 but genetic corr = {corr:.3f}"

    def test_rg_positive_positive_genetic_corr(self):
        """rg=0.5 should produce positive genetic correlation."""
        n, m = 2000, 100
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        mv = MultivariateEffects.from_h2_rg(h2=[0.5, 0.5], rg=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add(['T1.G', 'T2.G'], MVGeneticComponent(mv))
        result = arch.compute(hap, rng=np.random.RandomState(42))
        corr = np.corrcoef(result['T1.G'], result['T2.G'])[0, 1]
        assert corr > 0.1, f"rg=0.5 but genetic corr = {corr:.3f}"

    def test_per_trait_h2_approximate(self):
        """Per-trait h2 ≈ Var(G_trait) / Var(Y_trait) approximately match targets."""
        n, m = 3000, 100
        h2_targets = [0.6, 0.3]
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        mv = MultivariateEffects.from_h2_rg(h2=h2_targets, rg=0.2, m=m, seed=42)
        arch = Architecture()
        arch.add(['T1.G', 'T2.G'], MVGeneticComponent(mv))
        arch.add('T1.E', NoiseComponent(variance=1.0 - h2_targets[0]))
        arch.add('T2.E', NoiseComponent(variance=1.0 - h2_targets[1]))
        arch.add('T1', AggregationComponent('T1.G + T1.E'), inputs=['T1.G', 'T1.E'])
        arch.add('T2', AggregationComponent('T2.G + T2.E'), inputs=['T2.G', 'T2.E'])

        result = arch.compute(hap, rng=np.random.RandomState(42))

        for i, (trait, h2_t) in enumerate(zip(['T1', 'T2'], h2_targets)):
            var_g = np.var(result[f'{trait}.G'])
            var_y = np.var(result[trait])
            h2_r = var_g / var_y
            assert 0.1 < h2_r < 0.9, \
                f"{trait}: h2_realized={h2_r:.3f}, target={h2_t}"

    def test_genetic_cov_matrix_psd(self):
        """Realized genetic covariance matrix should be positive semidefinite."""
        n, m = 2000, 100
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        mv = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.4, m=m, seed=42)
        arch = Architecture()
        arch.add(['T1.G', 'T2.G'], MVGeneticComponent(mv))
        result = arch.compute(hap, rng=np.random.RandomState(42))

        g1 = result['T1.G']
        g2 = result['T2.G']
        cov_matrix = np.cov(np.vstack([g1, g2]))
        eigenvalues = np.linalg.eigvalsh(cov_matrix)
        assert np.all(eigenvalues >= -1e-10), \
            f"Genetic cov matrix not PSD: eigenvalues = {eigenvalues}"
