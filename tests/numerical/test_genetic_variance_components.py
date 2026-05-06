"""
Numerical tests for genetic variance components.

Tests:
1. Non-standardized effects: Var(G) depends on allele frequencies
2. Standardized effects: Var(G) ≈ h2
3. MVGenetic: per-trait variance matches h2
4. HaplotypeGenetic: maternal variance ≈ 0.5 * diploid variance
"""
import numpy as np
import pytest

from xftsim.effect import AdditiveEffects, MultivariateEffects
from xftsim.arch import (
    Architecture, GeneticComponent, MVGeneticComponent,
    HaplotypeGeneticComponent, NoiseComponent, AggregationComponent,
)
from xftsim.mate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.sim import NSimulation

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestStandardizedVariance:
    def test_genetic_variance_approximates_h2(self):
        """With standardized effects, Var(G) ≈ h2."""
        n, m = 1000, 100
        h2 = 0.5
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=h2, m=m, standardized=True, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        pheno = arch.compute(hap)
        var_g = np.var(pheno['Y.G'], ddof=1)
        # Should be in ballpark of h2 (wide tolerance for finite samples)
        assert 0.1 < var_g < 1.5, f"Var(G) = {var_g:.3f}, expected ≈ {h2}"


class TestNonStandardizedVariance:
    def test_non_standardized_variance_positive(self):
        """Non-standardized effects should produce positive genetic variance."""
        n, m = 500, 50
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, standardized=False, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        pheno = arch.compute(hap)
        var_g = np.var(pheno['Y.G'], ddof=1)
        assert var_g > 0.01


class TestMVGeneticVariance:
    def test_per_trait_variance(self):
        """Each trait's genetic variance should be approximately its h2."""
        n, m = 1000, 100
        h2 = [0.5, 0.3]
        mv_eff = MultivariateEffects.from_h2_rg(h2=h2, rg=0.2, m=m, seed=42)
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        arch = Architecture()
        arch.add(['T1.G', 'T2.G'], MVGeneticComponent(mv_eff))
        pheno = arch.compute(hap)
        var1 = np.var(pheno['T1.G'], ddof=1)
        var2 = np.var(pheno['T2.G'], ddof=1)
        # Wide tolerance
        assert 0.1 < var1 < 1.5, f"Var(T1.G) = {var1:.3f}, expected ≈ {h2[0]}"
        assert 0.05 < var2 < 1.0, f"Var(T2.G) = {var2:.3f}, expected ≈ {h2[1]}"


class TestHaplotypeVariancePartitioning:
    def test_maternal_half_diploid(self):
        """Maternal genetic variance ≈ 0.5 * diploid genetic variance."""
        n, m = 1000, 100
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, standardized=False, seed=42)
        mat_val = hap.matvec_maternal(eff.effects)
        diploid_val = hap.matvec(eff.effects)
        var_mat = np.var(mat_val, ddof=1)
        var_dip = np.var(diploid_val, ddof=1)
        # Maternal should be about half of diploid
        # (with some sampling variance and LD effects)
        ratio = var_mat / var_dip
        assert 0.15 < ratio < 0.85, \
            f"Maternal/Diploid ratio = {ratio:.3f}, expected ≈ 0.5"

    def test_maternal_paternal_covariance_small(self):
        """Maternal and paternal genetic values should have ~0 covariance in founders."""
        n, m = 1000, 100
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, standardized=False, seed=42)
        mat_val = hap.matvec_maternal(eff.effects)
        pat_val = hap.matvec_paternal(eff.effects)
        cov = np.cov(mat_val, pat_val)[0, 1]
        var_mat = np.var(mat_val, ddof=1)
        # Correlation should be near zero
        corr = cov / var_mat if var_mat > 0 else 0
        assert abs(corr) < 0.3, f"Maternal-paternal corr = {corr:.3f}, expected ≈ 0"
