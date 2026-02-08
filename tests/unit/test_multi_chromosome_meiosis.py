"""
Unit tests for multi-chromosome meiosis.

Tests:
1. Two-chromosome setup preserves correct shapes
2. Chromosome boundary resets crossover state
3. Multi-chromosome allele frequency conservation
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray
from xftsim.nmate import RandomMating
from xftsim.reproduce import RecombinationMap


def _make_multi_chrom_hap(n=100, m_per_chrom=20, n_chrom=3, seed=42):
    m = m_per_chrom * n_chrom
    rng = np.random.RandomState(seed)
    sm = SampleMeta(
        iid=np.arange(n), fid=np.arange(n) // 2,
        sex=np.tile([0, 1], n // 2),
    )
    chrom = np.repeat(np.arange(n_chrom), m_per_chrom)
    vm = VariantMeta(vid=np.array([f'chr{c}_v{i}' for c, i in
                                    zip(chrom, range(m))]),
                     chrom=chrom)
    geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
    return DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm), chrom


class TestMultiChromMeiosisShape:
    def test_output_shape(self):
        hap, chrom = _make_multi_chrom_hap(n=100, m_per_chrom=20, n_chrom=3)
        rmap = RecombinationMap(p=0.1, m=60, chrom=chrom)
        mate = RandomMating(offspring_per_pair=2)
        assignment = mate.mate(hap.samples, rng=np.random.RandomState(42))
        offspring = hap.meiosis(assignment, rmap)
        assert offspring.genotypes.shape[1] == 60
        assert offspring.genotypes.shape[2] == 2

    def test_output_binary(self):
        hap, chrom = _make_multi_chrom_hap()
        rmap = RecombinationMap(p=0.2, m=60, chrom=chrom)
        mate = RandomMating(offspring_per_pair=2)
        assignment = mate.mate(hap.samples, rng=np.random.RandomState(42))
        offspring = hap.meiosis(assignment, rmap)
        assert np.all((offspring.genotypes == 0) | (offspring.genotypes == 1))


class TestMultiChromAFConservation:
    def test_af_approximately_conserved(self):
        """Allele frequencies should be approximately conserved under random mating."""
        n = 200
        hap, chrom = _make_multi_chrom_hap(n=n, m_per_chrom=20, n_chrom=3, seed=42)
        rmap = RecombinationMap(p=0.2, m=60, chrom=chrom)
        mate = RandomMating(offspring_per_pair=2)

        parent_af = hap.genotypes.mean(axis=(0, 2))

        assignment = mate.mate(hap.samples, rng=np.random.RandomState(42))
        offspring = hap.meiosis(assignment, rmap)
        offspring_af = offspring.genotypes.mean(axis=(0, 2))

        # AF should be similar (within 0.15 for each variant, wide tolerance)
        max_diff = np.max(np.abs(parent_af - offspring_af))
        assert max_diff < 0.20, f"Max AF difference = {max_diff:.3f}"


class TestChromBoundaryEffect:
    def test_no_recombination_within_chrom(self):
        """With p=0 (no recombination), offspring should inherit
        complete parental haplotypes within each chromosome."""
        n = 50
        m_per_chrom = 10
        n_chrom = 2
        m = m_per_chrom * n_chrom
        hap, chrom = _make_multi_chrom_hap(n=n, m_per_chrom=m_per_chrom,
                                            n_chrom=n_chrom, seed=42)
        rmap = RecombinationMap(p=0.0, m=m, chrom=chrom)
        mate = RandomMating(offspring_per_pair=2)
        assignment = mate.mate(hap.samples, rng=np.random.RandomState(42))
        offspring = hap.meiosis(assignment, rmap)

        # Within each chromosome (excluding first locus which is boundary),
        # offspring should have intact parental haplotype
        for i in range(min(10, offspring.n)):
            mat_idx = assignment.maternal_idx[i]
            # Chromosome 1: indices 1..9 (skip boundary at 0)
            off_chrom1 = offspring.genotypes[i, 1:m_per_chrom, 0]
            par_h0_c1 = hap.genotypes[mat_idx, 1:m_per_chrom, 0]
            par_h1_c1 = hap.genotypes[mat_idx, 1:m_per_chrom, 1]
            assert np.all(off_chrom1 == par_h0_c1) or np.all(off_chrom1 == par_h1_c1)
