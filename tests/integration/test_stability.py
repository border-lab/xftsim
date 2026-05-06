"""Long-run stability tests: verify simulations remain numerically sane over many generations."""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation

from xftsim.sim import NSimulation
from xftsim.mate import RandomMating, LinearAssortativeMating
from xftsim.arch import (
    Architecture, GeneticComponent, MVGeneticComponent, NoiseComponent,
    CNoiseComponent, AggregationComponent, ParentComponent,
)
from xftsim.effect import AdditiveEffects, MultivariateEffects
from xftsim.reproduce import RecombinationMap


class TestLongRunStability:
    """Run simulations for 20+ generations and verify numerical stability."""

    def test_single_trait_20_gen(self):
        """Single-trait sim for 20 generations: no NaN, bounded variance."""
        m, n = 50, 200
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        arch = TestSimulation.simple_architecture(m=m, h2=0.5)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        mate = RandomMating(offspring_per_pair=2)
        sim = NSimulation(
            hap, arch, mate, rmap, seed=42,
            retain_haplotypes=2, retain_phenotypes=3,
        )
        sim.run(20)
        assert sim.generation == 19
        pheno = sim.phenotype_history[19]
        Y = pheno['Y']
        assert np.all(np.isfinite(Y))
        # Variance should be bounded (not diverging)
        assert Y.var() < 100.0

    def test_single_trait_retention_works(self):
        """Retention policy should keep memory bounded over 20 generations."""
        m, n = 50, 200
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        arch = TestSimulation.simple_architecture(m=m, h2=0.5)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        mate = RandomMating(offspring_per_pair=2)
        sim = NSimulation(
            hap, arch, mate, rmap, seed=42,
            retain_haplotypes=1, retain_phenotypes=2,
        )
        sim.run(20)
        # Should only have recent generations in history
        assert len(sim.haplotype_history) <= 2
        assert len(sim.phenotype_history) <= 3
        # Latest generation should be present
        assert 19 in sim.phenotype_history
        assert 19 in sim.haplotype_history

    def test_bivariate_20_gen(self):
        """Bivariate sim for 20 generations: both traits finite and bounded."""
        m, n = 50, 200
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        arch = TestSimulation.bivariate_architecture(m=m, h2=[0.5, 0.3], rg=0.3)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        mate = RandomMating(offspring_per_pair=2)
        sim = NSimulation(
            hap, arch, mate, rmap, seed=42,
            retain_haplotypes=2, retain_phenotypes=3,
        )
        sim.run(20)
        pheno = sim.phenotype_history[19]
        for name in ['trait1', 'trait2']:
            vals = pheno[name]
            assert np.all(np.isfinite(vals)), f"{name} has NaN/Inf at gen 19"
            assert vals.var() < 100.0, f"{name} variance diverged at gen 19"

    def test_vt_20_gen(self):
        """Vertical transmission sim for 20 generations stays finite."""
        m, n = 50, 200
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        arch = TestSimulation.vt_architecture(m=m, h2=0.5, vt_weight=0.3)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        mate = RandomMating(offspring_per_pair=2)
        sim = NSimulation(
            hap, arch, mate, rmap, seed=42,
            retain_haplotypes=2, retain_phenotypes=3,
        )
        sim.run(20)
        pheno = sim.phenotype_history[19]
        assert np.all(np.isfinite(pheno['Y']))
        assert pheno['Y'].var() < 100.0

    def test_assortative_20_gen(self):
        """Assortative mating for 20 generations: finite, bounded."""
        m, n = 50, 200
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        arch = TestSimulation.simple_architecture(m=m, h2=0.5)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        mate = LinearAssortativeMating(
            component_names=['Y'], r=0.5, offspring_per_pair=2,
        )
        sim = NSimulation(
            hap, arch, mate, rmap, seed=42,
            retain_haplotypes=2, retain_phenotypes=3,
        )
        sim.run(20)
        pheno = sim.phenotype_history[19]
        assert np.all(np.isfinite(pheno['Y']))
        # Assortative mating can inflate variance, but shouldn't diverge
        assert pheno['Y'].var() < 200.0

    def test_disassortative_20_gen(self):
        """Disassortative mating for 20 generations: finite, bounded."""
        m, n = 50, 200
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        arch = TestSimulation.simple_architecture(m=m, h2=0.5)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        mate = LinearAssortativeMating(
            component_names=['Y'], r=-0.5, offspring_per_pair=2,
        )
        sim = NSimulation(
            hap, arch, mate, rmap, seed=42,
            retain_haplotypes=2, retain_phenotypes=3,
        )
        sim.run(20)
        pheno = sim.phenotype_history[19]
        assert np.all(np.isfinite(pheno['Y']))
        assert pheno['Y'].var() < 200.0

    def test_population_size_stable(self):
        """With opp=2 and balanced sex, population size should be constant."""
        m, n = 20, 100
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        arch = TestSimulation.simple_architecture(m=m, h2=0.5)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        mate = RandomMating(offspring_per_pair=2)
        sim = NSimulation(
            hap, arch, mate, rmap, seed=42,
            retain_haplotypes=2, retain_phenotypes=10,
        )
        sim.run(10)
        # All generations should have the same population size
        for gen in range(10):
            if gen in sim.phenotype_history:
                assert sim.phenotype_history[gen].samples.n == n

    def test_early_stopping_callback(self):
        """Early stopping callback should terminate correctly."""
        m, n = 20, 100
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        arch = TestSimulation.simple_architecture(m=m, h2=0.5)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        mate = RandomMating(offspring_per_pair=2)

        stop_gen = 5
        def stopper(sim):
            if sim.generation >= stop_gen:
                sim.stop = True

        sim = NSimulation(
            hap, arch, mate, rmap, seed=42,
            retain_haplotypes=2, retain_phenotypes=10,
            callbacks=[stopper],
        )
        sim.run(20)
        assert sim.generation == stop_gen
        assert sim.stop is True


class TestComplexArchitectureStability:
    """Test stability with more complex architecture configurations."""

    def test_correlated_noise_10_gen(self):
        """Correlated noise component should stay finite over 10 generations."""
        m, n = 30, 200
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.3, m=m, seed=42)
        cov = np.array([[0.3, 0.1], [0.1, 0.4]])
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add(['Y.E1', 'Y.E2'], CNoiseComponent(cov=cov))
        arch.add('Y', AggregationComponent('Y.G + Y.E1'))
        arch.add('Z', AggregationComponent('Y.E2'))
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        mate = RandomMating(offspring_per_pair=2)
        sim = NSimulation(
            hap, arch, mate, rmap, seed=42,
            retain_haplotypes=2, retain_phenotypes=3,
        )
        sim.run(10)
        for name in ['Y', 'Z']:
            vals = sim.phenotype_history[9][name]
            assert np.all(np.isfinite(vals))

    def test_vt_plus_assortative_10_gen(self):
        """VT + assortative mating should not cause numerical explosion."""
        m, n = 30, 200
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        arch = TestSimulation.vt_architecture(m=m, h2=0.4, vt_weight=0.2)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        mate = LinearAssortativeMating(
            component_names=['Y'], r=0.3, offspring_per_pair=2,
        )
        sim = NSimulation(
            hap, arch, mate, rmap, seed=42,
            retain_haplotypes=2, retain_phenotypes=3,
        )
        sim.run(10)
        pheno = sim.phenotype_history[9]
        assert np.all(np.isfinite(pheno['Y']))
        assert pheno['Y'].var() < 200.0
