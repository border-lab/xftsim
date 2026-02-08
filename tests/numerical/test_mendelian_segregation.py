"""
Numerical test: Mendelian segregation properties.

Tests:
1. Het parent (0/1) passes each allele ~50% of the time
2. Hom parent (0/0) always passes 0 allele
3. Hom parent (1/1) always passes 1 allele
4. Maternal haplotype comes from mother, paternal from father
5. Offspring genotype values are binary (0 or 1)
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray
from xftsim.reproduce import RecombinationMap, meiosis


class TestMendelianSegregation:
    def test_het_parent_transmits_50_50(self):
        """Heterozygous parent should transmit each allele ~50% of time."""
        n, m = 2, 100
        # Parent 0: all heterozygous (0/1)
        # Parent 1: all heterozygous (0/1)
        geno = np.zeros((n, m, 2), dtype=np.int8)
        geno[:, :, 1] = 1  # All het: hap0=0, hap1=1
        sm = SampleMeta(iid=np.arange(n), sex=np.array([0, 1]))
        vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
        hap = DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)

        rmap = RecombinationMap.constant_map(m=m, p=0.0)  # no recombination
        # Many offspring from same parents
        n_offspring = 1000
        mat_idx = np.zeros(n_offspring, dtype=np.int64)  # all from mother 0
        pat_idx = np.ones(n_offspring, dtype=np.int64)    # all from father 1

        offspring = meiosis(hap, rmap, mat_idx, pat_idx)

        # For maternal haplotype: each offspring gets hap0 (=0) or hap1 (=1)
        # At first locus, fraction getting allele 1 should be ~0.5
        maternal_alleles = offspring[:, 0, 0]  # first locus, maternal hap
        frac_one = np.mean(maternal_alleles)
        assert 0.3 < frac_one < 0.7, \
            f"Het parent transmits allele 1 {frac_one:.3f} of time, expected ~0.5"

    def test_hom_00_always_passes_0(self):
        """Homozygous 0/0 parent should always pass allele 0."""
        n, m = 2, 20
        geno = np.zeros((n, m, 2), dtype=np.int8)
        # Parent 0: hom 0/0 (mother)
        # Parent 1: hom 1/1 (father)
        geno[1, :, :] = 1

        sm = SampleMeta(iid=np.arange(n), sex=np.array([0, 1]))
        vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
        hap = DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)

        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        n_offspring = 100
        mat_idx = np.zeros(n_offspring, dtype=np.int64)
        pat_idx = np.ones(n_offspring, dtype=np.int64)

        offspring = meiosis(hap, rmap, mat_idx, pat_idx)

        # Maternal haplotype should always be 0 (mother is 0/0)
        assert np.all(offspring[:, :, 0] == 0), \
            "Hom 0/0 mother should always pass allele 0"
        # Paternal haplotype should always be 1 (father is 1/1)
        assert np.all(offspring[:, :, 1] == 1), \
            "Hom 1/1 father should always pass allele 1"

    def test_offspring_genotype_binary(self):
        """All offspring genotype values should be 0 or 1."""
        n, m = 50, 30
        rng = np.random.RandomState(42)
        geno = rng.binomial(1, 0.5, (n, m, 2)).astype(np.int8)
        sm = SampleMeta(iid=np.arange(n), sex=np.tile([0, 1], n // 2))
        vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
        hap = DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)

        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        mat_idx = np.arange(0, 25, dtype=np.int64)
        pat_idx = np.arange(25, 50, dtype=np.int64)

        offspring = meiosis(hap, rmap, mat_idx, pat_idx)
        assert np.all((offspring == 0) | (offspring == 1)), \
            "Offspring genotypes should be binary (0 or 1)"

    def test_maternal_from_mother_paternal_from_father(self):
        """Offspring hap0 comes from mother, hap1 from father."""
        n, m = 4, 10
        geno = np.zeros((n, m, 2), dtype=np.int8)
        # Mother 0: all 1s on both haplotypes
        geno[0, :, :] = 1
        # Mother 1: all 0s
        # Father 2: all 0s
        # Father 3: all 1s on both haplotypes
        geno[3, :, :] = 1

        sm = SampleMeta(iid=np.arange(n), sex=np.array([0, 0, 1, 1]))
        vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
        hap = DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)

        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        # Offspring 0: mother=0 (all 1), father=2 (all 0)
        # Offspring 1: mother=1 (all 0), father=3 (all 1)
        mat_idx = np.array([0, 1], dtype=np.int64)
        pat_idx = np.array([2, 3], dtype=np.int64)

        offspring = meiosis(hap, rmap, mat_idx, pat_idx)

        # Offspring 0: maternal = 1, paternal = 0
        assert np.all(offspring[0, :, 0] == 1), "Child 0 maternal should be all 1"
        assert np.all(offspring[0, :, 1] == 0), "Child 0 paternal should be all 0"
        # Offspring 1: maternal = 0, paternal = 1
        assert np.all(offspring[1, :, 0] == 0), "Child 1 maternal should be all 0"
        assert np.all(offspring[1, :, 1] == 1), "Child 1 paternal should be all 1"
