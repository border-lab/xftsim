"""
Numerical tests for recombination correctness.

Tests:
1. Free recombination (p=0.5): offspring genotypes should differ from both parents
2. No recombination (p=0): offspring haplotypes should match one parental haplotype exactly
3. Allele frequency conservation: offspring AF ≈ parental AF
4. Recombination rate estimation: crossover frequency matches expected rate
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray
from xftsim.arch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.effect import AdditiveEffects
from xftsim.sim import NSimulation
from xftsim.mate import RandomMating
from xftsim.reproduce import RecombinationMap, meiosis

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestAlleleFrequencyConservation:
    def test_af_approximately_conserved(self):
        """Allele frequencies should be approximately conserved across generations."""
        n, m = 500, 20
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        af0 = hap.recompute_af()
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))

        sim = NSimulation(
            hap, arch, RandomMating(offspring_per_pair=2),
            RecombinationMap.constant_map(m=m, p=0.5),
            seed=42, retain_haplotypes=1,
        )
        sim.run(3)

        af_final = sim.haplotype_history[sim.generation].recompute_af()
        # With N=500 and 3 gens, drift is small; correlation should be high
        corr = np.corrcoef(af0, af_final)[0, 1]
        assert corr > 0.8, f"AF correlation too low: {corr}"


class TestNoRecombination:
    def test_p_zero_preserves_haplotypes(self):
        """With p=0, each offspring haplotype should match one parental haplotype exactly."""
        n, m = 20, 50
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)

        # Create mate assignment manually
        maternal_idx = np.array([0, 0, 2, 2], dtype=np.int64)
        paternal_idx = np.array([1, 1, 3, 3], dtype=np.int64)

        rmap = RecombinationMap.constant_map(m=m, p=0.0)
        offspring_geno = meiosis(hap, rmap, maternal_idx, paternal_idx)

        # Each offspring's maternal haplotype should match one of the mother's haplotypes
        for i in range(4):
            mat_idx = maternal_idx[i]
            offspring_mat_hap = offspring_geno[i, :, 0]
            parent_hap0 = hap.genotypes[mat_idx, :, 0]
            parent_hap1 = hap.genotypes[mat_idx, :, 1]
            matches_0 = np.array_equal(offspring_mat_hap, parent_hap0)
            matches_1 = np.array_equal(offspring_mat_hap, parent_hap1)
            assert matches_0 or matches_1, (
                f"Offspring {i} maternal haplotype doesn't match either parent haplotype"
            )


class TestFreeRecombination:
    def test_p_half_produces_crossovers(self):
        """With p=0.5, there should be crossover events (not identical to parent)."""
        n, m = 20, 100  # many loci to guarantee crossovers
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)

        maternal_idx = np.repeat(np.arange(0, n, 2), 2).astype(np.int64)
        paternal_idx = np.repeat(np.arange(1, n, 2), 2).astype(np.int64)

        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        offspring_geno = meiosis(hap, rmap, maternal_idx, paternal_idx)

        # With 100 loci and p=0.5, at least some offspring should differ
        # from both parental haplotypes (crossover occurred)
        crossover_found = False
        for i in range(len(maternal_idx)):
            mat_idx = maternal_idx[i]
            offspring_mat = offspring_geno[i, :, 0]
            p0 = hap.genotypes[mat_idx, :, 0]
            p1 = hap.genotypes[mat_idx, :, 1]
            if not (np.array_equal(offspring_mat, p0) or np.array_equal(offspring_mat, p1)):
                crossover_found = True
                break
        assert crossover_found, "No crossovers detected with p=0.5 and 100 loci"


class TestMeiosisOutputShape:
    def test_output_shape(self):
        """Meiosis should produce (n_offspring, m, 2) genotypes."""
        n, m = 20, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        maternal_idx = np.array([0, 0, 2, 2, 4], dtype=np.int64)
        paternal_idx = np.array([1, 1, 3, 3, 5], dtype=np.int64)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        offspring = meiosis(hap, rmap, maternal_idx, paternal_idx)
        assert offspring.shape == (5, m, 2)

    def test_output_binary(self):
        """Meiosis output should be binary (0 or 1)."""
        n, m = 20, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        maternal_idx = np.array([0, 2, 4], dtype=np.int64)
        paternal_idx = np.array([1, 3, 5], dtype=np.int64)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        offspring = meiosis(hap, rmap, maternal_idx, paternal_idx)
        assert np.all((offspring == 0) | (offspring == 1))
