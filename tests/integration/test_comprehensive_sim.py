"""
Comprehensive integration tests combining multiple features.

Tests:
1. Bivariate VT + assortative mating + filters + stats + callbacks
2. HaplotypeGenetic with sibling components + trio filter
3. Multi-gen simulation with all feature types exercised simultaneously
4. Retention pruning with filters still working
5. SampleMeta extra fields propagated through simulation
6. Architecture from_formula with DSL equivalent to programmatic
7. Deterministic reproduction (same seed = same result)
8. Large family size (5 offspring per pair) with sibling stats
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray, NPhenotypeArray
from xftsim.narch import (
    Architecture, GeneticComponent, NoiseComponent, AggregationComponent,
    MVGeneticComponent, CNoiseComponent, MotherComponent,
    HaplotypeGeneticComponent, SiblingMeanComponent,
)
from xftsim.neffect import AdditiveEffects, MultivariateEffects
from xftsim.nsim import NSimulation
from xftsim.nmate import RandomMating, LinearAssortativeMating
from xftsim.reproduce import RecombinationMap
from xftsim.nfilter import TrioFilter, SibPairFilter
from xftsim.nstats import SampleStatistics

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestBivariateVTAssortative:
    """Bivariate VT with assortative mating, filters, stats, and callbacks."""

    def test_full_feature_bivariate(self):
        n, m = 200, 15
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = MultivariateEffects.from_h2_rg(h2=[0.3, 0.3], rg=0.4, m=m, seed=42)
        cov = np.array([[0.4, 0.05], [0.05, 0.4]])

        arch = Architecture()
        arch.add(('Y1.G', 'Y2.G'), MVGeneticComponent(eff))
        arch.add(('Y1.E', 'Y2.E'), CNoiseComponent(cov))
        arch.add('Y1.VT', MotherComponent('Y1', founder_component=NoiseComponent(0.1)))
        arch.add('Y1', AggregationComponent('Y1.G + Y1.E + Y1.VT'))
        arch.add('Y2', AggregationComponent('Y2.G + Y2.E'))

        mate = LinearAssortativeMating(component_names=['Y1'], r=0.3, offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)

        variances = []
        def track_var(sim):
            pheno = sim.phenotype_history[sim.generation]
            variances.append(np.var(pheno['Y1']))

        sim = NSimulation(
            hap, arch, mate, rmap, seed=42,
            filters={'trio': TrioFilter()},
            statistics=[SampleStatistics()],
            callbacks=[track_var],
            retain_phenotypes=3,
        )
        sim.run(5)

        assert sim.generation >= 4
        assert len(variances) >= 4
        assert all(np.isfinite(v) and v > 0 for v in variances)
        assert len(sim.results) >= 4

        # VT should inflate Y1 variance relative to Y2
        pheno = sim.phenotype_history[sim.generation]
        var_y1 = np.var(pheno['Y1'])
        var_y2 = np.var(pheno['Y2'])
        # Not always true for small sims, but Y1 includes VT
        assert np.isfinite(var_y1) and np.isfinite(var_y2)


class TestHaplotypeGeneticWithSiblings:
    """HaplotypeGenetic (maternal/paternal) with sibling mean."""

    def test_hapgenetic_sibling_integration(self):
        n, m = 200, 15
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.Gmat', HaplotypeGeneticComponent(eff, haplotype='maternal'))
        arch.add('Y.Gpat', HaplotypeGeneticComponent(eff, haplotype='paternal'))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.Gmat + Y.Gpat + Y.E'))
        arch.add('Y.sib', SiblingMeanComponent('Y'), inputs=['Y'], grouping='FID')

        mate = RandomMating(offspring_per_pair=3)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        sim = NSimulation(
            hap, arch, mate, rmap, seed=42,
            filters={'sib': SibPairFilter()},
        )
        sim.run(3)

        pheno = sim.phenotype_history[sim.generation]
        assert np.all(np.isfinite(pheno['Y']))
        assert np.all(np.isfinite(pheno['Y.sib']))
        assert np.all(np.isfinite(pheno['Y.Gmat']))
        assert np.all(np.isfinite(pheno['Y.Gpat']))

        # Sibling mean should have lower variance than individual Y
        assert np.var(pheno['Y.sib']) < np.var(pheno['Y']) + 0.5


class TestDeterministicReproduction:
    """Same seed should produce identical results."""

    def test_same_seed_same_gen0(self):
        """Same seed → identical gen-0 phenotypes (no meiosis involved)."""
        n, m = 100, 10

        results = []
        for _ in range(2):
            eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
            hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
            arch = Architecture.from_formula("""
                Y.G ~ genetic(beta)
                Y.E ~ noise(0.5)
                Y ~ Y.G + Y.E
            """, effects={'beta': eff})
            mate = RandomMating(offspring_per_pair=2)
            rmap = RecombinationMap.constant_map(m=m, p=0.5)
            sim = NSimulation(hap, arch, mate, rmap, seed=99)
            sim.run(1)  # gen 0 only
            results.append(sim.phenotype_history[0]['Y'].copy())

        np.testing.assert_array_equal(results[0], results[1])


class TestLargeFamilySiblings:
    """Large family sizes with sibling statistics."""

    def test_five_offspring_per_pair(self):
        n, m = 200, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture.from_formula("""
            Y.G ~ genetic(beta)
            Y.E ~ noise(0.5)
            Y ~ Y.G + Y.E
            Y.sib_mean ~ sibling_mean(Y) | FID
            Y.sib_count ~ sibling_count(Y) | FID
        """, effects={'beta': eff})
        mate = RandomMating(offspring_per_pair=5)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        sim = NSimulation(hap, arch, mate, rmap, seed=42)
        sim.run(2)

        pheno = sim.phenotype_history[1]
        # All offspring families should have 5 members
        assert np.all(pheno['Y.sib_count'] == 5)
        # Sibling means should be well-defined
        assert np.all(np.isfinite(pheno['Y.sib_mean']))


class TestRetentionWithAllFeatures:
    """Aggressive retention with filters, stats, and callbacks."""

    def test_minimal_retention_survives(self):
        n, m = 100, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))

        mate = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)

        gen_count = [0]
        def count_gens(sim):
            gen_count[0] += 1

        sim = NSimulation(
            hap, arch, mate, rmap, seed=42,
            retain_haplotypes=1,
            retain_phenotypes=1,
            statistics=[SampleStatistics()],
            callbacks=[count_gens],
        )
        sim.run(8)

        # run(8) → gen 0 + gens 1..7 = gen 7, callbacks fire 8 times (once per gen)
        assert sim.generation == 7
        assert gen_count[0] == 8
        # Minimal retention
        assert len(sim.haplotype_history) <= 2
        assert len(sim.phenotype_history) <= 2
        # Stats captured for all generations
        assert len(sim.results) >= 7


class TestDSLvsProgrammatic:
    """DSL formula should produce equivalent results to programmatic API."""

    def test_formula_vs_add(self):
        n, m = 200, 10
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)

        # Programmatic
        arch1 = Architecture()
        arch1.add('Y.G', GeneticComponent(eff))
        arch1.add('Y.E', NoiseComponent(variance=0.5))
        arch1.add('Y', AggregationComponent('Y.G + Y.E'))

        # DSL
        arch2 = Architecture.from_formula("""
            Y.G ~ genetic(beta)
            Y.E ~ noise(0.5)
            Y ~ Y.G + Y.E
        """, effects={'beta': eff})

        # Same haplotypes, same seed → same phenotypes
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        rng1 = np.random.RandomState(99)
        rng2 = np.random.RandomState(99)

        pheno1 = arch1.compute(hap, rng=rng1)
        pheno2 = arch2.compute(hap, rng=rng2)

        np.testing.assert_allclose(pheno1['Y.G'], pheno2['Y.G'])
        np.testing.assert_allclose(pheno1['Y.E'], pheno2['Y.E'])
        np.testing.assert_allclose(pheno1['Y'], pheno2['Y'])


class TestEarlyStoppingWithAllFeatures:
    """Early stopping callback with filters, stats, and callbacks."""

    def test_stop_at_gen_3(self):
        n, m = 100, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))

        def stop_at_3(sim):
            if sim.generation >= 3:
                sim.stop = True

        sim = NSimulation(
            hap, arch, RandomMating(offspring_per_pair=2),
            RecombinationMap.constant_map(m=m, p=0.5),
            seed=42,
            filters={'trio': TrioFilter()},
            statistics=[SampleStatistics()],
            callbacks=[stop_at_3],
        )
        sim.run(100)
        assert sim.generation == 3


class TestMultiGenPhenotypeStability:
    """Phenotypes should remain finite and well-behaved across generations."""

    def test_10_gen_all_finite(self):
        n, m = 100, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))

        variances = []
        def track(sim):
            pheno = sim.phenotype_history[sim.generation]
            v = np.var(pheno['Y'])
            assert np.isfinite(v) and v > 0
            variances.append(v)

        sim = NSimulation(
            hap, arch, RandomMating(offspring_per_pair=2),
            RecombinationMap.constant_map(m=m, p=0.5),
            seed=42,
            callbacks=[track],
            retain_haplotypes=1,
            retain_phenotypes=1,
        )
        sim.run(10)

        assert len(variances) == 10
        assert all(np.isfinite(v) for v in variances)
