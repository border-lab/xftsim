"""
Unit tests for meiosis special cases and edge conditions.

Tests special cases that complement existing meiosis test coverage:
1. Multi-chromosome boundary forcing (p=0.5 at chromosome boundaries)
2. Very high recombination rate (p close to 1): frequent alternation
3. Very low recombination rate (p close to 0): minimal recombination
4. Single variant per chromosome scenarios
5. All-homozygous parents: offspring invariant to crossovers
6. Binary output validation (0/1 haplotype values)
7. Maternal vs paternal haplotype selection randomness
8. Offspring count matches assignment
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from testdata import TestSimulation
from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray
from xftsim.reproduce import RecombinationMap, meiosis
from xftsim.mate import NMateAssignment


def _make_hap(n, m, seed=42, chrom=None):
    """Helper to create test haplotypes."""
    rng = np.random.RandomState(seed)
    genotypes = rng.binomial(1, 0.5, size=(n, m, 2)).astype(np.int8)
    sm = SampleMeta(iid=np.arange(n), sex=np.tile([0, 1], (n + 1) // 2)[:n])
    vm = VariantMeta(
        vid=np.array([f'v{i}' for i in range(m)]),
        chrom=chrom if chrom is not None else np.zeros(m, dtype=np.int64)
    )
    return DenseHaplotypeArray(genotypes=genotypes, samples=sm, variants=vm)


def _make_assignment(n_offspring, maternal_idx, paternal_idx, generation=1):
    """Helper to create NMateAssignment."""
    sm = SampleMeta(iid=np.arange(n_offspring), generation=generation)
    return NMateAssignment(
        offspring_samples=sm,
        maternal_idx=np.array(maternal_idx, dtype=np.int64),
        paternal_idx=np.array(paternal_idx, dtype=np.int64),
    )


class TestMultiChromosomeBoundaryForcing:
    """Tests for chromosome boundary forcing to p=0.5."""

    def test_boundary_forced_to_half(self):
        """Chromosome boundaries should be forced to p=0.5 regardless of input p."""
        m = 60
        chrom = np.array([1]*20 + [2]*20 + [3]*20)
        # Try to set low recombination everywhere
        p = np.ones(m) * 0.01
        rmap = RecombinationMap(p=p, m=m, chrom=chrom)

        # Check boundaries are forced to 0.5
        assert rmap._probabilities[0] == 0.5  # start of chrom 1
        assert rmap._probabilities[20] == 0.5  # start of chrom 2
        assert rmap._probabilities[40] == 0.5  # start of chrom 3
        # Non-boundaries should preserve input
        assert rmap._probabilities[10] == 0.01
        assert rmap._probabilities[30] == 0.01

    def test_independent_segregation_across_chromosomes(self):
        """Different chromosomes should segregate independently (boundary effect)."""
        n = 100
        m_per_chrom = 30
        chrom = np.array([1]*m_per_chrom + [2]*m_per_chrom)
        m = len(chrom)

        # Create heterozygous parents: hap0=all 0, hap1=all 1
        geno = np.zeros((n, m, 2), dtype=np.int8)
        geno[:, :, 1] = 1
        sm = SampleMeta(iid=np.arange(n), sex=np.tile([0, 1], n // 2))
        vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]), chrom=chrom)
        hap = DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)

        # Use p=0 within chromosomes, but boundaries will be forced to 0.5
        rmap = RecombinationMap(p=0.0, m=m, chrom=chrom)

        mat_idx = np.arange(0, 50, dtype=np.int64)
        pat_idx = np.arange(50, 100, dtype=np.int64)

        offspring = meiosis(hap, rmap, mat_idx, pat_idx)

        # Within each chromosome, with p=0, offspring should have one uniform value
        # But across chromosomes, values should be independent
        n_mixed_chromosomes = 0
        for i in range(50):
            chrom1_vals = offspring[i, :m_per_chrom, 0]
            chrom2_vals = offspring[i, m_per_chrom:, 0]
            # Each chromosome should be uniform (all 0 or all 1)
            chrom1_uniform = np.all(chrom1_vals == 0) or np.all(chrom1_vals == 1)
            chrom2_uniform = np.all(chrom2_vals == 0) or np.all(chrom2_vals == 1)
            assert chrom1_uniform, f"Offspring {i} chrom1 not uniform with p=0"
            assert chrom2_uniform, f"Offspring {i} chrom2 not uniform with p=0"
            # Check if chromosomes differ
            if not np.array_equal(chrom1_vals, chrom2_vals):
                n_mixed_chromosomes += 1

        # With forced p=0.5 at boundaries, roughly half should have different
        # chromosome states
        assert n_mixed_chromosomes > 10, \
            f"Only {n_mixed_chromosomes}/50 show independent chromosome segregation"


class TestVeryHighRecombinationRate:
    """Tests for very high recombination rates (p close to 1)."""

    def test_p_near_one_frequent_alternation(self):
        """With p~0.99, haplotypes should alternate very frequently."""
        n, m = 100, 100
        # Create het parents: hap0=all 0, hap1=all 1
        geno = np.zeros((n, m, 2), dtype=np.int8)
        geno[:, :, 1] = 1
        sm = SampleMeta(iid=np.arange(n), sex=np.tile([0, 1], n // 2))
        vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
        hap = DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)

        rmap = RecombinationMap.constant_map(m=m, p=0.99)
        mat_idx = np.arange(0, 50, dtype=np.int64)
        pat_idx = np.arange(50, 100, dtype=np.int64)

        offspring = meiosis(hap, rmap, mat_idx, pat_idx)

        # Count alternations in each offspring's maternal haplotype
        total_switches = 0
        for i in range(50):
            mat_hap = offspring[i, :, 0]
            # Count switches between consecutive positions
            switches = np.sum(mat_hap[1:] != mat_hap[:-1])
            total_switches += switches

        # With p=0.99 and m=100, expect ~99 switches per offspring
        # Total across 50 offspring should be ~4950, but use conservative bound
        assert total_switches > 3000, \
            f"With p=0.99, expected >3000 switches, got {total_switches}"

    def test_p_near_one_alternating_pattern(self):
        """With p very high, offspring haplotypes should show alternating pattern."""
        n, m = 20, 50
        # Het parents
        geno = np.zeros((n, m, 2), dtype=np.int8)
        geno[:, :, 1] = 1
        sm = SampleMeta(iid=np.arange(n), sex=np.tile([0, 1], n // 2))
        vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
        hap = DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)

        rmap = RecombinationMap.constant_map(m=m, p=0.95)
        mat_idx = np.array([0], dtype=np.int64)
        pat_idx = np.array([1], dtype=np.int64)

        # Multiple runs to check pattern
        np.random.seed(42)
        switch_counts = []
        for _ in range(20):
            offspring = meiosis(hap, rmap, mat_idx, pat_idx)
            mat_hap = offspring[0, :, 0]
            switches = np.sum(mat_hap[1:] != mat_hap[:-1])
            switch_counts.append(switches)

        mean_switches = np.mean(switch_counts)
        # With p=0.95 and m=50, expect ~47.5 switches on average
        assert mean_switches > 35, f"Mean switches = {mean_switches}, expected >35"


class TestVeryLowRecombinationRate:
    """Tests for very low recombination rates (p close to 0)."""

    def test_p_near_zero_minimal_recombination(self):
        """With p~0.001, should see very few crossovers."""
        n, m = 100, 200
        # Het parents
        geno = np.zeros((n, m, 2), dtype=np.int8)
        geno[:, :, 1] = 1
        sm = SampleMeta(iid=np.arange(n), sex=np.tile([0, 1], n // 2))
        vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
        hap = DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)

        rmap = RecombinationMap.constant_map(m=m, p=0.001)
        mat_idx = np.arange(0, 50, dtype=np.int64)
        pat_idx = np.arange(50, 100, dtype=np.int64)

        offspring = meiosis(hap, rmap, mat_idx, pat_idx)

        # Count total switches across all offspring
        total_switches = 0
        for i in range(50):
            mat_hap = offspring[i, :, 0]
            switches = np.sum(mat_hap[1:] != mat_hap[:-1])
            total_switches += switches

        # With p=0.001, m=200, 50 offspring: expect ~50*199*0.001 ≈ 10 switches
        assert total_switches < 50, \
            f"With p=0.001, expected <50 switches, got {total_switches}"

    def test_p_near_zero_mostly_intact_haplotypes(self):
        """With very low p, most offspring should inherit nearly intact haplotypes."""
        n, m = 100, 100
        rng = np.random.RandomState(42)
        geno = rng.binomial(1, 0.5, size=(n, m, 2)).astype(np.int8)
        sm = SampleMeta(iid=np.arange(n), sex=np.tile([0, 1], n // 2))
        vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
        hap = DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)

        rmap = RecombinationMap.constant_map(m=m, p=0.005)
        mat_idx = np.arange(0, 50, dtype=np.int64)
        pat_idx = np.arange(50, 100, dtype=np.int64)

        offspring = meiosis(hap, rmap, mat_idx, pat_idx)

        # Count offspring with 0 or 1 switches (nearly intact)
        nearly_intact = 0
        for i in range(50):
            mat_idx_i = mat_idx[i]
            child_mat = offspring[i, :, 0]
            parent_h0 = geno[mat_idx_i, :, 0]
            parent_h1 = geno[mat_idx_i, :, 1]

            # Check which parent haplotype matches better
            match0 = np.sum(child_mat == parent_h0)
            match1 = np.sum(child_mat == parent_h1)

            # If one matches >95%, consider it nearly intact
            if max(match0, match1) > 0.95 * m:
                nearly_intact += 1

        # Most offspring should be nearly intact
        assert nearly_intact > 20, \
            f"Only {nearly_intact}/50 offspring are nearly intact with p=0.005"


class TestSingleVariantPerChromosome:
    """Tests for edge case of single variant per chromosome."""

    def test_single_variant_per_chrom_two_chromosomes(self):
        """Two chromosomes with 1 variant each."""
        n = 20
        m = 2
        chrom = np.array([1, 2])

        geno = np.zeros((n, m, 2), dtype=np.int8)
        geno[:, :, 1] = 1  # het everywhere
        sm = SampleMeta(iid=np.arange(n), sex=np.tile([0, 1], n // 2))
        vm = VariantMeta(vid=np.array(['v0', 'v1']), chrom=chrom)
        hap = DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)

        rmap = RecombinationMap(p=0.3, m=m, chrom=chrom)
        mat_idx = np.arange(0, 10, dtype=np.int64)
        pat_idx = np.arange(10, 20, dtype=np.int64)

        offspring = meiosis(hap, rmap, mat_idx, pat_idx)

        assert offspring.shape == (10, 2, 2)
        assert np.all((offspring == 0) | (offspring == 1))

        # Each variant should show 0 or 1 (from het parent)
        # With boundaries forced to 0.5, should see both values
        v0_maternal = offspring[:, 0, 0]
        v1_maternal = offspring[:, 1, 0]
        assert 0 in v0_maternal and 1 in v0_maternal
        assert 0 in v1_maternal and 1 in v1_maternal

    def test_many_chroms_single_variant_each(self):
        """Multiple chromosomes with single variant each."""
        n = 50
        n_chrom = 10
        m = n_chrom
        chrom = np.arange(n_chrom)

        geno = np.zeros((n, m, 2), dtype=np.int8)
        geno[:, :, 1] = 1
        sm = SampleMeta(iid=np.arange(n), sex=np.tile([0, 1], n // 2))
        vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]), chrom=chrom)
        hap = DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)

        rmap = RecombinationMap(p=0.2, m=m, chrom=chrom)
        mat_idx = np.arange(0, 25, dtype=np.int64)
        pat_idx = np.arange(25, 50, dtype=np.int64)

        offspring = meiosis(hap, rmap, mat_idx, pat_idx)

        assert offspring.shape == (25, 10, 2)
        assert np.all((offspring == 0) | (offspring == 1))


class TestAllHomozygousParent:
    """Tests for parents with no heterozygous sites."""

    def test_hom_00_parent_invariant_to_crossovers(self):
        """Homozygous 0/0 parent produces same offspring regardless of crossovers."""
        n = 20
        m = 50
        geno = np.zeros((n, m, 2), dtype=np.int8)
        # Parent 0: all homozygous 0/0 (mother)
        # Parent 1-9: all homozygous 0/0 (mothers)
        # Parent 10-19: random heterozygous (fathers)
        rng = np.random.RandomState(42)
        geno[10:, :, :] = rng.binomial(1, 0.5, size=(10, m, 2)).astype(np.int8)

        sm = SampleMeta(iid=np.arange(n), sex=np.concatenate([np.zeros(10), np.ones(10)]))
        vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
        hap = DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)

        # Test with different recombination rates
        for p in [0.0, 0.5, 0.99]:
            rmap = RecombinationMap.constant_map(m=m, p=p)
            mat_idx = np.array([0] * 10, dtype=np.int64)  # all from parent 0 (hom 0/0)
            pat_idx = np.arange(10, 20, dtype=np.int64)

            offspring = meiosis(hap, rmap, mat_idx, pat_idx)

            # All maternal haplotypes should be all 0s
            assert np.all(offspring[:, :, 0] == 0), \
                f"Hom 0/0 parent should always pass 0 allele (p={p})"

    def test_hom_11_parent_invariant_to_crossovers(self):
        """Homozygous 1/1 parent produces same offspring regardless of crossovers."""
        n = 20
        m = 50
        geno = np.zeros((n, m, 2), dtype=np.int8)
        # Parent 0-9: all homozygous 1/1 (mothers)
        geno[:10, :, :] = 1
        # Parent 10-19: random heterozygous (fathers)
        rng = np.random.RandomState(42)
        geno[10:, :, :] = rng.binomial(1, 0.5, size=(10, m, 2)).astype(np.int8)

        sm = SampleMeta(iid=np.arange(n), sex=np.concatenate([np.zeros(10), np.ones(10)]))
        vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
        hap = DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)

        # Test with different recombination rates
        for p in [0.0, 0.5, 0.99]:
            rmap = RecombinationMap.constant_map(m=m, p=p)
            mat_idx = np.array([0] * 10, dtype=np.int64)  # all from parent 0 (hom 1/1)
            pat_idx = np.arange(10, 20, dtype=np.int64)

            offspring = meiosis(hap, rmap, mat_idx, pat_idx)

            # All maternal haplotypes should be all 1s
            assert np.all(offspring[:, :, 0] == 1), \
                f"Hom 1/1 parent should always pass 1 allele (p={p})"

    def test_all_homozygous_population(self):
        """Population where all individuals are homozygous (mix of 0/0 and 1/1)."""
        n = 100
        m = 30
        rng = np.random.RandomState(42)
        geno = np.zeros((n, m, 2), dtype=np.int8)
        # Random assignment of homozygous states per individual
        for i in range(n):
            for j in range(m):
                allele = rng.binomial(1, 0.5)
                geno[i, j, :] = allele

        sm = SampleMeta(iid=np.arange(n), sex=np.tile([0, 1], n // 2))
        vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
        hap = DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)

        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        mat_idx = np.arange(0, 50, dtype=np.int64)
        pat_idx = np.arange(50, 100, dtype=np.int64)

        offspring = meiosis(hap, rmap, mat_idx, pat_idx)

        # Each offspring should match parent exactly (no het sites → no crossover effect)
        for i in range(50):
            mat_parent = geno[mat_idx[i], :, 0]  # both haps same for hom parent
            offspring_mat = offspring[i, :, 0]
            assert np.array_equal(offspring_mat, mat_parent), \
                f"Offspring {i} maternal doesn't match homozygous parent"


class TestBinaryOutputValidation:
    """Comprehensive tests that outputs are strictly binary (0 or 1)."""

    def test_binary_output_various_scenarios(self):
        """Test binary output across various meiosis scenarios."""
        scenarios = [
            (10, 5, 0.0),    # small, no recombination
            (100, 50, 0.5),  # medium, high recombination
            (50, 200, 0.99), # long genome, very high recombination
            (200, 10, 0.01), # many offspring, low recombination
        ]

        for n, m, p in scenarios:
            hap = _make_hap(n, m, seed=42)
            rmap = RecombinationMap.constant_map(m=m, p=p)
            mat_idx = np.arange(0, n // 2, dtype=np.int64)
            pat_idx = np.arange(n // 2, n, dtype=np.int64)

            offspring = meiosis(hap, rmap, mat_idx, pat_idx)

            assert offspring.dtype == np.int8
            assert np.all((offspring == 0) | (offspring == 1)), \
                f"Non-binary output for n={n}, m={m}, p={p}"
            # Check no NaNs or invalid values
            assert not np.any(np.isnan(offspring.astype(float)))

    def test_binary_output_extreme_parents(self):
        """Test binary output with extreme parent genotypes."""
        n, m = 50, 30
        # Test with all-zero parents
        geno_zero = np.zeros((n, m, 2), dtype=np.int8)
        sm = SampleMeta(iid=np.arange(n), sex=np.tile([0, 1], n // 2))
        vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
        hap_zero = DenseHaplotypeArray(genotypes=geno_zero, samples=sm, variants=vm)

        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        mat_idx = np.arange(0, 25, dtype=np.int64)
        pat_idx = np.arange(25, 50, dtype=np.int64)

        offspring = meiosis(hap_zero, rmap, mat_idx, pat_idx)
        assert np.all(offspring == 0)

        # Test with all-one parents
        geno_one = np.ones((n, m, 2), dtype=np.int8)
        hap_one = DenseHaplotypeArray(genotypes=geno_one, samples=sm, variants=vm)
        offspring = meiosis(hap_one, rmap, mat_idx, pat_idx)
        assert np.all(offspring == 1)


class TestHaplotypeSelectionRandomness:
    """Tests for randomness in maternal vs paternal haplotype selection."""

    def test_maternal_selection_approximately_50_50(self):
        """With het parent and p=0, should select each haplotype ~50% of time."""
        n = 4
        m = 1
        geno = np.zeros((n, m, 2), dtype=np.int8)
        # Mother 0: het 0/1
        geno[0, 0, 1] = 1
        # Mother 1: het 0/1
        geno[1, 0, 1] = 1
        # Father 2: hom 0/0
        # Father 3: hom 0/0

        sm = SampleMeta(iid=np.arange(n), sex=np.array([0, 0, 1, 1]))
        vm = VariantMeta(vid=np.array(['v0']))
        hap = DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)

        rmap = RecombinationMap.constant_map(m=m, p=0.0)

        # Many offspring from same mother
        n_offspring = 1000
        mat_idx = np.zeros(n_offspring, dtype=np.int64)
        pat_idx = np.ones(n_offspring, dtype=np.int64) * 2

        np.random.seed(42)
        offspring = meiosis(hap, rmap, mat_idx, pat_idx)

        # Count how many got allele 0 vs 1 from mother
        maternal_alleles = offspring[:, 0, 0]
        frac_one = np.mean(maternal_alleles)

        # Should be approximately 0.5 (binomial, n=1000, p=0.5)
        assert 0.45 < frac_one < 0.55, \
            f"Maternal haplotype selection not random: frac_one={frac_one}"

    def test_paternal_selection_approximately_50_50(self):
        """Paternal haplotype selection should also be ~50/50."""
        n = 4
        m = 1
        geno = np.zeros((n, m, 2), dtype=np.int8)
        # Mother 0: hom 0/0
        # Mother 1: hom 0/0
        # Father 2: het 0/1
        geno[2, 0, 1] = 1
        # Father 3: het 0/1
        geno[3, 0, 1] = 1

        sm = SampleMeta(iid=np.arange(n), sex=np.array([0, 0, 1, 1]))
        vm = VariantMeta(vid=np.array(['v0']))
        hap = DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)

        rmap = RecombinationMap.constant_map(m=m, p=0.0)

        # Many offspring from same father
        n_offspring = 1000
        mat_idx = np.zeros(n_offspring, dtype=np.int64)
        pat_idx = np.ones(n_offspring, dtype=np.int64) * 2

        np.random.seed(42)
        offspring = meiosis(hap, rmap, mat_idx, pat_idx)

        # Count how many got allele 0 vs 1 from father
        paternal_alleles = offspring[:, 0, 1]
        frac_one = np.mean(paternal_alleles)

        # Should be approximately 0.5
        assert 0.45 < frac_one < 0.55, \
            f"Paternal haplotype selection not random: frac_one={frac_one}"

    def test_independent_selection_per_offspring(self):
        """Each offspring should independently select haplotypes."""
        n = 10
        m = 20
        # Create het parents
        geno = np.zeros((n, m, 2), dtype=np.int8)
        geno[:, :, 1] = 1
        sm = SampleMeta(iid=np.arange(n), sex=np.tile([0, 1], n // 2))
        vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
        hap = DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)

        rmap = RecombinationMap.constant_map(m=m, p=0.0)

        # Multiple offspring from same parents
        n_offspring = 100
        mat_idx = np.zeros(n_offspring, dtype=np.int64)
        pat_idx = np.ones(n_offspring, dtype=np.int64)

        np.random.seed(42)
        offspring = meiosis(hap, rmap, mat_idx, pat_idx)

        # Each offspring's maternal haplotype should be all 0 or all 1
        # But not all offspring should be identical
        all_zero_count = 0
        all_one_count = 0
        for i in range(n_offspring):
            mat_hap = offspring[i, :, 0]
            if np.all(mat_hap == 0):
                all_zero_count += 1
            elif np.all(mat_hap == 1):
                all_one_count += 1

        # Should have mix of both
        assert all_zero_count > 20 and all_one_count > 20, \
            f"Not enough variation: {all_zero_count} zeros, {all_one_count} ones"


class TestOffspringCountMatching:
    """Tests that offspring count matches assignment."""

    def test_offspring_count_matches_assignment(self):
        """Output shape should match number of mating pairs."""
        test_counts = [1, 5, 10, 50, 100, 237]

        for n_offspring in test_counts:
            hap = _make_hap(n=200, m=30, seed=42)
            rmap = RecombinationMap.constant_map(m=30, p=0.5)

            rng = np.random.RandomState(42)
            mat_idx = rng.randint(0, 100, size=n_offspring).astype(np.int64)
            pat_idx = rng.randint(100, 200, size=n_offspring).astype(np.int64)

            offspring = meiosis(hap, rmap, mat_idx, pat_idx)

            assert offspring.shape[0] == n_offspring, \
                f"Expected {n_offspring} offspring, got {offspring.shape[0]}"

    def test_zero_offspring_edge_case(self):
        """Empty mate assignment raises ValueError (np.max on empty array)."""
        hap = _make_hap(n=10, m=5, seed=42)
        rmap = RecombinationMap.constant_map(m=5, p=0.5)
        mat_idx = np.array([], dtype=np.int64)
        pat_idx = np.array([], dtype=np.int64)

        # Empty arrays cause ValueError in assertion check (np.max on empty)
        # This is expected behavior - meiosis requires at least one offspring
        with pytest.raises(ValueError, match="zero-size array"):
            offspring = meiosis(hap, rmap, mat_idx, pat_idx)

    def test_variable_offspring_per_pair(self):
        """Different pairs can have different offspring counts via index repetition."""
        hap = _make_hap(n=10, m=5, seed=42)
        rmap = RecombinationMap.constant_map(m=5, p=0.5)

        # Pair (0,5) has 1 offspring, (1,6) has 3, (2,7) has 2
        mat_idx = np.array([0, 1, 1, 1, 2, 2], dtype=np.int64)
        pat_idx = np.array([5, 6, 6, 6, 7, 7], dtype=np.int64)

        offspring = meiosis(hap, rmap, mat_idx, pat_idx)

        assert offspring.shape[0] == 6
        # Offspring 1-3 should all have mat_idx[i]==1, pat_idx[i]==6
        assert np.all((offspring[1:4, :, 0] >= 0) & (offspring[1:4, :, 0] <= 1))


class TestMeiosisWithTestSimulation:
    """Integration tests using TestSimulation helper."""

    def test_founder_haplotypes_meiosis(self):
        """Meiosis on TestSimulation founder haplotypes."""
        hap = TestSimulation.founder_haplotypes(n=100, m=50, seed=42)
        rmap = TestSimulation.recombination_map(m=50, p=0.5)

        mat_idx = np.arange(0, 50, dtype=np.int64)
        pat_idx = np.arange(50, 100, dtype=np.int64)

        offspring = meiosis(hap, rmap, mat_idx, pat_idx)

        assert offspring.shape == (50, 50, 2)
        assert offspring.dtype == np.int8
        assert np.all((offspring == 0) | (offspring == 1))

    def test_multi_generation_via_meiosis(self):
        """Multiple generations of meiosis preserve binary property."""
        hap_g0 = TestSimulation.founder_haplotypes(n=50, m=20, seed=42)
        rmap = TestSimulation.recombination_map(m=20, p=0.3)

        # Generation 1
        mat_idx = np.arange(0, 25, dtype=np.int64)
        pat_idx = np.arange(25, 50, dtype=np.int64)
        g1_geno = meiosis(hap_g0, rmap, mat_idx, pat_idx)
        sm_g1 = SampleMeta(iid=np.arange(25), sex=np.tile([0, 1], 13)[:25])
        hap_g1 = DenseHaplotypeArray(genotypes=g1_geno, samples=sm_g1,
                                      variants=hap_g0.variants)

        # Generation 2
        mat_idx_g2 = np.arange(0, 12, dtype=np.int64)
        pat_idx_g2 = np.arange(13, 25, dtype=np.int64)
        g2_geno = meiosis(hap_g1, rmap, mat_idx_g2, pat_idx_g2)

        assert np.all((g2_geno == 0) | (g2_geno == 1))
        assert g2_geno.dtype == np.int8
