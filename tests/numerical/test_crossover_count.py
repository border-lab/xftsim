"""
Numerical test: meiosis crossover counts match expected rates.

Tests:
1. p=0.5 → ~m/2 crossovers per meiosis (haplotype switches)
2. p=0.0 → 0 crossovers (perfect inheritance)
3. p=0.01 → few crossovers
4. Crossover positions are distributed across loci (not clustered)
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray
from xftsim.reproduce import RecombinationMap, meiosis
from xftsim.nmate import NMateAssignment


def _count_switches(offspring_geno, parent_geno, parent_idx, hap_col):
    """Count haplotype switches in offspring relative to parent."""
    n_offspring, m, _ = offspring_geno.shape
    total_switches = 0
    for i in range(n_offspring):
        pidx = parent_idx[i]
        # Offspring haplotype at hap_col
        child_hap = offspring_geno[i, :, hap_col]
        # Parent has two haplotypes at (pidx, :, 0) and (pidx, :, 1)
        # Determine which parent hap matches at each locus
        # A switch is when the source haplotype changes
        match_0 = (child_hap == parent_geno[pidx, :, 0])
        match_1 = (child_hap == parent_geno[pidx, :, 1])
        # At heterozygous sites, we can detect switches
        het_sites = parent_geno[pidx, :, 0] != parent_geno[pidx, :, 1]
        if np.sum(het_sites) < 2:
            continue
        het_idx = np.where(het_sites)[0]
        from_0 = match_0[het_idx]
        # Count transitions between consecutive het sites
        switches = np.sum(from_0[1:] != from_0[:-1])
        total_switches += switches
    return total_switches


class TestCrossoverCount:
    def test_p_zero_no_crossovers(self):
        """With p=0 recombination, offspring inherit intact haplotypes."""
        n, m = 100, 50
        rng = np.random.RandomState(42)
        sm = SampleMeta(iid=np.arange(n), sex=np.tile([0, 1], n // 2))
        vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
        geno = rng.binomial(1, 0.5, (n, m, 2)).astype(np.int8)
        hap = DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)

        rmap = RecombinationMap.constant_map(m=m, p=0.0)
        mat_idx = np.arange(0, 50, dtype=np.int64)  # all females
        pat_idx = np.arange(50, 100, dtype=np.int64)  # all males

        offspring = meiosis(hap, rmap, mat_idx, pat_idx)

        # Each offspring should inherit exactly one parent haplotype
        for i in range(50):
            # Maternal haplotype should match parent's hap 0 OR hap 1 exactly
            child_mat = offspring[i, :, 0]
            parent_hap0 = geno[mat_idx[i], :, 0]
            parent_hap1 = geno[mat_idx[i], :, 1]
            assert np.all(child_mat == parent_hap0) or np.all(child_mat == parent_hap1), \
                f"Offspring {i} maternal haplotype doesn't match either parent haplotype with p=0"

    def test_p_half_many_crossovers(self):
        """With p=0.5, there should be many crossovers (haplotype switches)."""
        n, m = 200, 100
        rng = np.random.RandomState(42)
        sm = SampleMeta(iid=np.arange(n), sex=np.tile([0, 1], n // 2))
        vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
        geno = rng.binomial(1, 0.5, (n, m, 2)).astype(np.int8)
        hap = DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)

        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        mat_idx = np.arange(0, 100, dtype=np.int64)
        pat_idx = np.arange(100, 200, dtype=np.int64)

        offspring = meiosis(hap, rmap, mat_idx, pat_idx)

        total_switches = _count_switches(offspring, geno, mat_idx, 0)
        # With p=0.5 and m=100 het sites, expect ~50 switches per meiosis
        # This is very rough — many switches expected
        assert total_switches > 100, \
            f"Expected many crossovers with p=0.5, got {total_switches} total"

    def test_low_recombination_few_crossovers(self):
        """With p=0.01, there should be very few crossovers."""
        n, m = 200, 100
        rng = np.random.RandomState(42)
        sm = SampleMeta(iid=np.arange(n), sex=np.tile([0, 1], n // 2))
        vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
        geno = rng.binomial(1, 0.5, (n, m, 2)).astype(np.int8)
        hap = DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)

        rmap = RecombinationMap.constant_map(m=m, p=0.01)
        mat_idx = np.arange(0, 100, dtype=np.int64)
        pat_idx = np.arange(100, 200, dtype=np.int64)

        offspring = meiosis(hap, rmap, mat_idx, pat_idx)

        total_switches = _count_switches(offspring, geno, mat_idx, 0)
        # With p=0.01 and 100 offspring, expect ~1 switch per meiosis
        # So total ~100, but could be lower
        assert total_switches < 500, \
            f"Expected few crossovers with p=0.01, got {total_switches} total"

    def test_crossovers_distributed_across_genome(self):
        """With p=0.5, offspring haplotypes should not be identical to either parent haplotype."""
        n, m = 100, 200
        rng = np.random.RandomState(42)
        sm = SampleMeta(iid=np.arange(n), sex=np.tile([0, 1], n // 2))
        vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
        # Use deterministic heterozygous parents to detect crossovers clearly:
        # Parent hap0 = all 0, hap1 = all 1 → any crossover changes source
        geno = np.zeros((n, m, 2), dtype=np.int8)
        geno[:, :, 1] = 1  # hap1 = all 1s

        hap = DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)

        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        mat_idx = np.arange(0, 50, dtype=np.int64)
        pat_idx = np.arange(50, 100, dtype=np.int64)

        offspring = meiosis(hap, rmap, mat_idx, pat_idx)

        # Each offspring's maternal haplotype should be a mix of 0s and 1s
        n_mixed = 0
        for i in range(50):
            child_mat = offspring[i, :, 0]
            frac_ones = child_mat.mean()
            # With crossovers, it shouldn't be all 0 or all 1
            if 0.1 < frac_ones < 0.9:
                n_mixed += 1
        # Most offspring should show mixing
        assert n_mixed > 30, f"Only {n_mixed}/50 offspring show cross-genome variation"
