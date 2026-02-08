"""
Unit tests for NSimulation retention policy and edge cases.

Tests:
1. _enforce_retention: haplotype pruning, phenotype pruning, pedigree pruning,
   mate_assignment pruning
2. Retention with retain_haplotypes=0 (keep only current)
3. Multiple statistics with same type (naming collision)
4. NSimulation repr
5. continue_run after aggressive pruning
6. Validation: effect dimension mismatch
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray, NPhenotypeArray
from xftsim.narch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.neffect import AdditiveEffects
from xftsim.nsim import NSimulation
from xftsim.nmate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.nstats import SampleStatistics

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestRetentionPolicy:
    def _setup_sim(self, n=50, m=5, retain_haplotypes=1, retain_phenotypes=2, seed=42):
        """Create a simple simulation for retention testing."""
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=seed)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        mate = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        return NSimulation(
            hap, arch, mate, rmap,
            retain_haplotypes=retain_haplotypes,
            retain_phenotypes=retain_phenotypes,
            seed=seed,
        )

    def test_haplotype_pruning(self):
        """Old haplotypes should be pruned per retain_haplotypes."""
        sim = self._setup_sim(retain_haplotypes=1)
        sim.run(5)
        assert sim.generation == 4
        # retain_haplotypes=1 means keep current + 1 back = 2 max
        assert len(sim.haplotype_history) <= 2

    def test_phenotype_pruning(self):
        """Old phenotypes should be pruned per retain_phenotypes."""
        sim = self._setup_sim(retain_phenotypes=1)
        sim.run(5)
        assert sim.generation == 4
        assert len(sim.phenotype_history) <= 2

    def test_pedigree_pruning(self):
        """Pedigree history should be pruned with phenotypes."""
        sim = self._setup_sim(retain_phenotypes=1)
        sim.run(5)
        assert len(sim.pedigree_history) <= 2

    def test_mate_assignment_pruning(self):
        """Only most recent mate assignment should be kept."""
        sim = self._setup_sim()
        sim.run(5)
        assert len(sim._mate_assignments) <= 2

    def test_aggressive_retention(self):
        """retain_haplotypes=0 should keep only current gen."""
        sim = self._setup_sim(retain_haplotypes=0, retain_phenotypes=0)
        sim.run(5)
        assert sim.generation == 4
        assert len(sim.haplotype_history) == 1
        assert sim.generation in sim.haplotype_history

    def test_large_retention_keeps_all(self):
        """Large retention values should keep everything."""
        sim = self._setup_sim(retain_haplotypes=100, retain_phenotypes=100)
        sim.run(5)
        assert len(sim.haplotype_history) == 5  # gen 0..4

    def test_current_gen_always_available(self):
        """Current generation should always be in history regardless of retention."""
        for retain in [0, 1, 2]:
            sim = self._setup_sim(retain_haplotypes=retain, retain_phenotypes=retain)
            sim.run(4)
            assert sim.generation in sim.haplotype_history
            assert sim.generation in sim.phenotype_history


class TestMultipleStatistics:
    def test_same_stat_type_naming(self):
        """Multiple SampleStatistics should get unique names."""
        n, m = 50, 5
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
            statistics=[SampleStatistics(), SampleStatistics()],
        )
        sim.run(2)
        # Should have 2 generations of results
        assert len(sim.results) >= 1
        # Check naming: first is "SampleStatistics", second "SampleStatistics_1"
        first_result = sim.results[0]
        assert 'SampleStatistics' in first_result.statistics
        assert 'SampleStatistics_1' in first_result.statistics


class TestNSimulationRepr:
    def test_repr(self):
        """NSimulation repr should show generation, n, m."""
        n, m = 50, 5
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        mate = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        sim = NSimulation(hap, arch, mate, rmap, seed=42)
        r = repr(sim)
        assert 'generation=0' in r


class TestValidation:
    def test_dimension_mismatch_raises(self):
        """Effects with wrong m should raise ValueError."""
        n, m = 50, 5
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)  # m=10 != 5
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        mate = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        sim = NSimulation(hap, arch, mate, rmap, seed=42)
        with pytest.raises(ValueError, match="dimension mismatch"):
            sim.run(1)


class TestContinueRunAfterPruning:
    def test_continue_after_pruning(self):
        """continue_run should work even after aggressive pruning."""
        n, m = 50, 5
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
            retain_haplotypes=1, retain_phenotypes=1,
        )
        sim.run(3)
        assert sim.generation == 2
        sim.continue_run(3)
        assert sim.generation == 5
        assert np.all(np.isfinite(sim.phenotype_history[5]['Y']))
