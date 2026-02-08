"""
Numerical test: NoiseComponent and CNoiseComponent produce correct variance.

Tests:
1. NoiseComponent(variance=v) produces Var(E) ≈ v
2. CNoiseComponent produces correct covariance structure
3. Noise is independent across calls (different RNG states)
4. Large sample noise mean ≈ 0
"""
import numpy as np
import pytest

from xftsim.narch import (
    Architecture, NoiseComponent, CNoiseComponent,
    GeneticComponent, AggregationComponent, ArchNode,
)
from xftsim.neffect import AdditiveEffects
from xftsim.struct import NPhenotypeArray

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestNoiseComponentVariance:
    def test_univariate_variance(self):
        """NoiseComponent(variance=v) should produce Var(E) ≈ v."""
        n = 5000
        hap = TestSimulation.founder_haplotypes(n=n, m=10, seed=42)

        comp = NoiseComponent(variance=2.0)
        node = ArchNode(outputs=['E'], component=comp, inputs=[], grouping=None)
        pheno = NPhenotypeArray(hap.samples)
        rng = np.random.RandomState(42)
        result = comp.compute(node, hap, pheno, rng=rng)

        var = np.var(result)
        np.testing.assert_allclose(var, 2.0, atol=0.15)

    def test_univariate_mean_zero(self):
        """NoiseComponent should have mean ≈ 0."""
        n = 5000
        hap = TestSimulation.founder_haplotypes(n=n, m=10, seed=42)

        comp = NoiseComponent(variance=1.0)
        node = ArchNode(outputs=['E'], component=comp, inputs=[], grouping=None)
        pheno = NPhenotypeArray(hap.samples)
        rng = np.random.RandomState(42)
        result = comp.compute(node, hap, pheno, rng=rng)

        np.testing.assert_allclose(np.mean(result), 0.0, atol=0.05)

    def test_multivariate_covariance(self):
        """CNoiseComponent should produce correct covariance structure."""
        n = 5000
        hap = TestSimulation.founder_haplotypes(n=n, m=10, seed=42)
        target_cov = np.array([[1.0, 0.5], [0.5, 2.0]])

        comp = CNoiseComponent(cov=target_cov)
        node = ArchNode(outputs=['E1', 'E2'], component=comp, inputs=[], grouping=None)
        pheno = NPhenotypeArray(hap.samples)
        rng = np.random.RandomState(42)
        result = comp.compute(node, hap, pheno, rng=rng)

        # Result should be (n, 2)
        assert result.shape == (n, 2)
        empirical_cov = np.cov(result.T)
        np.testing.assert_allclose(empirical_cov, target_cov, atol=0.15)

    def test_different_rng_different_noise(self):
        """Different RNG states should produce different noise."""
        n = 100
        hap = TestSimulation.founder_haplotypes(n=n, m=5, seed=42)

        comp = NoiseComponent(variance=1.0)
        node = ArchNode(outputs=['E'], component=comp, inputs=[], grouping=None)
        pheno = NPhenotypeArray(hap.samples)

        rng1 = np.random.RandomState(42)
        result1 = comp.compute(node, hap, pheno, rng=rng1)

        rng2 = np.random.RandomState(99)
        result2 = comp.compute(node, hap, pheno, rng=rng2)

        assert not np.allclose(result1, result2)
