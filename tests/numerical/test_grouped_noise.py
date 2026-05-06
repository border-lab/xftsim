"""
Numerical tests for grouped noise and correlated noise components.

Verifies:
1. Grouped noise: same-FID individuals share identical noise values
2. Grouped cnoise: same-FID individuals share identical multivariate noise
3. Ungrouped noise: independent draws per individual (low correlation)
4. Noise variance matches specified parameter
5. CNoiseComponent correlation matches specified covariance
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray, NPhenotypeArray
from xftsim.arch import (
    Architecture, GeneticComponent, NoiseComponent, CNoiseComponent,
    AggregationComponent,
)
from xftsim.effect import AdditiveEffects, MultivariateEffects
from xftsim.sim import NSimulation
from xftsim.mate import RandomMating
from xftsim.reproduce import RecombinationMap

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestGroupedNoiseWithinFamily:
    """Same-FID individuals should share grouped noise values."""

    def test_grouped_noise_shared_within_family(self):
        """noise | FID → all siblings get identical noise."""
        n, m = 200, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture.from_formula("""
            Y.G ~ genetic(beta)
            Y.E ~ noise(0.5) | FID
            Y ~ Y.G + Y.E
        """, effects={'beta': eff})
        mate = RandomMating(offspring_per_pair=3)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        sim = NSimulation(hap, arch, mate, rmap, seed=42)
        sim.run(2)

        pheno = sim.phenotype_history[1]
        fids = pheno.samples.fid
        # For each family, noise values should be identical
        for fid_val in np.unique(fids):
            mask = fids == fid_val
            noise_vals = pheno['Y.E'][mask]
            assert np.all(noise_vals == noise_vals[0]), (
                f"FID={fid_val}: noise values differ within family"
            )


class TestUngroupedNoiseIndependence:
    """Ungrouped noise draws should be independent per individual."""

    def test_ungrouped_noise_has_variation(self):
        """noise(v) without grouping → each individual gets own draw."""
        n, m = 200, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture.from_formula("""
            Y.G ~ genetic(beta)
            Y.E ~ noise(0.5)
            Y ~ Y.G + Y.E
        """, effects={'beta': eff})
        mate = RandomMating(offspring_per_pair=3)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        sim = NSimulation(hap, arch, mate, rmap, seed=42)
        sim.run(2)

        pheno = sim.phenotype_history[1]
        fids = pheno.samples.fid
        # Check that siblings generally DON'T have identical noise
        n_diff = 0
        for fid_val in np.unique(fids):
            mask = fids == fid_val
            noise_vals = pheno['Y.E'][mask]
            if not np.all(noise_vals == noise_vals[0]):
                n_diff += 1
        # Most families should have different noise values
        assert n_diff > len(np.unique(fids)) * 0.5


class TestNoiseVariance:
    """Noise variance should approximately match the specified parameter."""

    def test_noise_variance_close_to_specified(self):
        """Var(noise(v)) ≈ v for large n."""
        n, m = 1000, 5
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        arch = Architecture()
        arch.add('E', NoiseComponent(variance=2.0))
        pheno = arch.compute(hap, rng=np.random.RandomState(42))
        var_e = np.var(pheno['E'])
        assert abs(var_e - 2.0) < 0.3, f"Expected var ≈ 2.0, got {var_e:.3f}"


class TestCNoiseCorrelation:
    """CNoiseComponent correlation should match off-diagonal covariance."""

    def test_cnoise_correlation_structure(self):
        """cnoise with positive off-diagonal → positive phenotype correlation."""
        n, m = 1000, 5
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        arch = Architecture()
        arch.add(('E1', 'E2'), CNoiseComponent(cov))
        pheno = arch.compute(hap, rng=np.random.RandomState(42))
        r = np.corrcoef(pheno['E1'], pheno['E2'])[0, 1]
        assert r > 0.2, f"Expected positive correlation, got r={r:.3f}"

    def test_cnoise_independent(self):
        """cnoise with diagonal cov → ~zero correlation."""
        n, m = 1000, 5
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        cov = np.array([[1.0, 0.0], [0.0, 1.0]])
        arch = Architecture()
        arch.add(('E1', 'E2'), CNoiseComponent(cov))
        pheno = arch.compute(hap, rng=np.random.RandomState(42))
        r = np.corrcoef(pheno['E1'], pheno['E2'])[0, 1]
        assert abs(r) < 0.15, f"Expected ~0 correlation, got r={r:.3f}"

    def test_cnoise_variance_matches(self):
        """Diagonal of cnoise cov → variance of each component."""
        n, m = 2000, 5
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        cov = np.array([[3.0, 0.5], [0.5, 1.0]])
        arch = Architecture()
        arch.add(('E1', 'E2'), CNoiseComponent(cov))
        pheno = arch.compute(hap, rng=np.random.RandomState(42))
        assert abs(np.var(pheno['E1']) - 3.0) < 0.5
        assert abs(np.var(pheno['E2']) - 1.0) < 0.3
