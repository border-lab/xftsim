"""
Integration test: formula with all function types produces valid simulation.

Tests:
1. Formula with genetic + noise + aggregation runs
2. Formula with haplotypeGenetic maternal/paternal
3. Formula with sibling_mean and FID grouping
4. Formula with parent/mother/father vertical transmission
5. Formula with cnoise multivariate
"""
import numpy as np
import pytest

from xftsim.arch import Architecture
from xftsim.effect import AdditiveEffects, MultivariateEffects
from xftsim.sim import Simulation
from xftsim.mate import RandomMating
from xftsim.reproduce import RecombinationMap

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestFormulaAllFunctions:
    def test_genetic_noise_agg(self):
        """Basic formula: genetic + noise + aggregation."""
        n, m = 100, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture(
            formula="Y.G ~ genetic(eff)\nY.E ~ noise(0.5)\nY ~ Y.G + Y.E",
            effects={'eff': eff},
        )
        mate = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        sim = Simulation(hap, arch, mate, rmap, seed=42)
        sim.run(3)
        assert sim.generation == 2
        assert np.all(np.isfinite(sim.phenotype_history[2]['Y']))

    def test_haplotype_genetic(self):
        """Formula with haplotypeGenetic."""
        n, m = 100, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture(
            formula="Y.mat ~ haplotypeGenetic(eff, haplotype='maternal')\nY.pat ~ haplotypeGenetic(eff, haplotype='paternal')\nY ~ Y.mat + Y.pat",
            effects={'eff': eff},
        )
        mate = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        sim = Simulation(hap, arch, mate, rmap, seed=42)
        sim.run(2)
        assert np.all(np.isfinite(sim.phenotype_history[1]['Y']))

    def test_sibling_mean_formula(self):
        """Formula with sibling_mean grouped by FID."""
        n, m = 100, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture(
            formula="Y.G ~ genetic(eff)\nY.E ~ noise(0.5)\nY ~ Y.G + Y.E\nY.sm ~ sibling_mean(Y) | FID",
            effects={'eff': eff},
        )
        mate = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        sim = Simulation(hap, arch, mate, rmap, seed=42)
        sim.run(2)
        pheno = sim.phenotype_history[1]
        assert 'Y.sm' in pheno
        assert np.all(np.isfinite(pheno['Y.sm']))

    def test_vertical_transmission_formula(self):
        """Formula with mother/father components (multi-gen)."""
        n, m = 100, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture(
            formula="Y.G ~ genetic(eff)\nY.E ~ noise(0.5)\nY ~ Y.G + Y.E\nY.m ~ mother(Y)\nY.f ~ father(Y)",
            effects={'eff': eff},
        )
        mate = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        sim = Simulation(hap, arch, mate, rmap, seed=42)
        sim.run(3)
        # At gen 2, mother/father values should be from gen 1
        pheno = sim.phenotype_history[2]
        assert 'Y.m' in pheno
        assert 'Y.f' in pheno
        assert np.all(np.isfinite(pheno['Y.m']))
        assert np.all(np.isfinite(pheno['Y.f']))

    def test_cnoise_formula(self):
        """Formula with cnoise multivariate."""
        n, m = 100, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        arch = Architecture(
            formula="(E1, E2) ~ cnoise(cov=[[1.0, 0.5], [0.5, 1.0]])",
        )
        mate = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        sim = Simulation(hap, arch, mate, rmap, seed=42)
        sim.run(2)
        pheno = sim.phenotype_history[1]
        assert 'E1' in pheno
        assert 'E2' in pheno
        assert np.all(np.isfinite(pheno['E1']))
        assert np.all(np.isfinite(pheno['E2']))
