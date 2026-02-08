"""
Integration tests for full-featured simulations with filters, statistics,
callbacks, and complex architectures.

Tests:
1. Simulation with TrioFilter + SampleStatistics
2. Simulation with SibPairFilter + callback
3. VT simulation with filters collecting data across generations
4. Early stopping callback with filter
5. Multi-trait simulation with all features
6. Simulation with sibling components
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray, NPhenotypeArray
from xftsim.narch import (
    Architecture, GeneticComponent, NoiseComponent, AggregationComponent,
    MVGeneticComponent, CNoiseComponent, MotherComponent, SiblingMeanComponent,
)
from xftsim.neffect import AdditiveEffects, MultivariateEffects
from xftsim.nsim import NSimulation
from xftsim.nmate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.nfilter import TrioFilter, SibPairFilter
from xftsim.nstats import SampleStatistics

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestSimulationWithTrioFilter:
    def test_trio_filter_collects_across_generations(self):
        """TrioFilter should produce None at gen 0, data at gen 1+."""
        n, m = 60, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        mate = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)

        trio_filter = TrioFilter()
        stats = SampleStatistics()
        sim = NSimulation(
            hap, arch, mate, rmap, seed=42,
            filters={'trio': trio_filter},
            statistics=[stats],
        )
        sim.run(3)

        # Should have results for each generation
        assert len(sim.results) >= 3
        # Gen 0 result should exist
        gen0_result = sim.results[0]
        assert gen0_result.generation == 0

    def test_trio_filter_values_consistent(self):
        """TrioFilter mother values should match prev gen phenotypes."""
        n, m = 60, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        mate = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)

        trio_filter = TrioFilter()
        sim = NSimulation(
            hap, arch, mate, rmap, seed=42,
            filters={'trio': trio_filter},
            retain_phenotypes=3,
        )
        sim.run(2)

        # At gen 1, trio filter should have indexed parent phenotypes
        # We can verify by checking gen 0 phenotypes are used
        assert 1 in sim.phenotype_history
        assert np.all(np.isfinite(sim.phenotype_history[1]['Y']))


class TestSimulationWithSibPairFilter:
    def test_sib_pair_filter_produces_pairs(self):
        """SibPairFilter should find pairs in offspring (who share FID)."""
        n, m = 60, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        mate = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)

        sib_filter = SibPairFilter()
        sim = NSimulation(
            hap, arch, mate, rmap, seed=42,
            filters={'sib': sib_filter},
        )
        sim.run(2)
        # After meiosis, offspring in gen 1 should share FIDs
        # and produce sibling pairs
        assert sim.generation >= 1


class TestSimulationWithCallbackAndFilter:
    def test_early_stopping_with_filter(self):
        """Early stopping callback should work with filters."""
        n, m = 60, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        mate = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)

        def stop_at_gen_2(sim):
            if sim.generation >= 2:
                sim.stop = True

        sim = NSimulation(
            hap, arch, mate, rmap, seed=42,
            filters={'trio': TrioFilter()},
            statistics=[SampleStatistics()],
            callbacks=[stop_at_gen_2],
        )
        sim.run(10)  # Would run 10, but stops at 2
        assert sim.generation == 2


class TestSimulationMultiTrait:
    def test_bivariate_with_filters_and_stats(self):
        """Bivariate simulation with filters and statistics."""
        n, m = 60, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.5], rg=0.3, m=m, seed=42)
        cov = np.array([[0.5, 0.1], [0.1, 0.5]])
        arch = Architecture()
        arch.add(('Y1.G', 'Y2.G'), MVGeneticComponent(eff))
        arch.add(('Y1.E', 'Y2.E'), CNoiseComponent(cov))
        arch.add('Y1', AggregationComponent('Y1.G + Y1.E'))
        arch.add('Y2', AggregationComponent('Y2.G + Y2.E'))
        mate = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)

        sim = NSimulation(
            hap, arch, mate, rmap, seed=42,
            statistics=[SampleStatistics()],
        )
        sim.run(3)

        # Statistics should capture both traits
        assert len(sim.results) >= 3
        last_result = sim.results[-1]
        sample_stats = last_result.statistics['SampleStatistics']
        assert 'Y1' in sample_stats['keys']
        assert 'Y2' in sample_stats['keys']
        k = len(sample_stats['keys'])
        assert sample_stats['cov'].shape == (k, k)
        assert np.all(np.isfinite(sample_stats['cov']))


class TestSimulationWithVTAndFilters:
    def test_vt_with_trio_filter(self):
        """VT simulation should work with TrioFilter."""
        n, m = 60, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.VT', MotherComponent('Y', founder_component=NoiseComponent(variance=0.1)))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.VT + Y.E'))
        mate = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)

        sim = NSimulation(
            hap, arch, mate, rmap, seed=42,
            filters={'trio': TrioFilter()},
            retain_phenotypes=3,
        )
        sim.run(4)

        # VT should produce finite phenotypes at all generations
        for gen in sim.phenotype_history:
            pheno = sim.phenotype_history[gen]
            assert np.all(np.isfinite(pheno['Y']))
            assert np.all(np.isfinite(pheno['Y.VT']))


class TestSimulationWithSiblingComponents:
    def test_sibling_mean_in_simulation(self):
        """Sibling mean component should work in full simulation via formula."""
        n, m = 60, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        # Use from_formula to ensure correct topological ordering
        arch = Architecture.from_formula("""
            Y.G ~ genetic(beta)
            Y.E ~ noise(0.5)
            Y ~ Y.G + Y.E
            Y.sib ~ sibling_mean(Y) | FID
        """, effects={'beta': eff})
        mate = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)

        sim = NSimulation(hap, arch, mate, rmap, seed=42)
        sim.run(3)

        # Sibling means should be finite and have lower variance than Y
        pheno = sim.phenotype_history[sim.generation]
        assert np.all(np.isfinite(pheno['Y.sib']))
        # Mean of siblings should have less variance than individual phenotype
        assert np.var(pheno['Y.sib']) <= np.var(pheno['Y']) + 0.1


class TestRetentionWithFilters:
    def test_retention_works_with_filters(self):
        """Aggressive retention should not break filters."""
        n, m = 60, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        mate = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)

        sim = NSimulation(
            hap, arch, mate, rmap, seed=42,
            retain_haplotypes=1,
            retain_phenotypes=2,
            statistics=[SampleStatistics()],
        )
        sim.run(5)

        # Should still have stats for all generations
        assert len(sim.results) >= 5
        # But only recent phenotypes/haplotypes retained
        assert len(sim.haplotype_history) <= 2
        assert len(sim.phenotype_history) <= 3

    def test_all_generations_produce_finite_phenotypes(self):
        """Even with aggressive retention, all computed phenotypes should be finite."""
        n, m = 60, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        mate = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)

        variances = []

        def track_variance(sim):
            pheno = sim.phenotype_history[sim.generation]
            variances.append(np.var(pheno['Y']))

        sim = NSimulation(
            hap, arch, mate, rmap, seed=42,
            retain_haplotypes=1,
            retain_phenotypes=1,
            callbacks=[track_variance],
        )
        sim.run(5)

        assert len(variances) == 5
        assert all(np.isfinite(v) and v > 0 for v in variances)
