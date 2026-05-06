"""
Unit tests for meiosis edge cases.

Tests:
1. Single variant (m=1) meiosis
2. Single offspring meiosis
3. p=0 (no recombination) preserves parent haplotypes exactly
4. Self-mating (maternal == paternal)
5. Output shape correctness
6. Output values are binary (0 or 1)
7. Large family (many offspring from same parents)
8. Meiosis preserves diploid allele count distribution
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray
from xftsim.mate import NMateAssignment, RandomMating
from xftsim.reproduce import RecombinationMap


def _make_hap(n, m, seed=42):
    sm = SampleMeta(iid=np.arange(n))
    vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
    rng = np.random.RandomState(seed)
    geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
    return DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)


def _make_assignment(n_offspring, maternal_idx, paternal_idx, generation=1):
    sm = SampleMeta(iid=np.arange(n_offspring), generation=generation)
    return NMateAssignment(
        offspring_samples=sm,
        maternal_idx=np.array(maternal_idx, dtype=np.int64),
        paternal_idx=np.array(paternal_idx, dtype=np.int64),
    )


class TestMeiosisSingleVariant:
    def test_single_variant(self):
        """Meiosis with m=1 should produce valid offspring."""
        hap = _make_hap(n=10, m=1, seed=42)
        rmap = RecombinationMap.constant_map(m=1, p=0.5)
        assignment = _make_assignment(
            n_offspring=4,
            maternal_idx=[0, 0, 2, 4],
            paternal_idx=[1, 3, 5, 7],
        )
        offspring = hap.meiosis(assignment, rmap)
        assert offspring.genotypes.shape == (4, 1, 2)
        assert np.all((offspring.genotypes == 0) | (offspring.genotypes == 1))


class TestMeiosisSingleOffspring:
    def test_single_offspring(self):
        """Meiosis producing exactly 1 offspring."""
        hap = _make_hap(n=10, m=5, seed=42)
        rmap = RecombinationMap.constant_map(m=5, p=0.5)
        assignment = _make_assignment(
            n_offspring=1,
            maternal_idx=[0],
            paternal_idx=[1],
        )
        offspring = hap.meiosis(assignment, rmap)
        assert offspring.genotypes.shape == (1, 5, 2)
        assert offspring.n == 1


class TestMeiosisNoRecombination:
    def test_p_zero_preserves_haplotypes(self):
        """With p=0, each offspring haplotype should be an exact copy of one parent haplotype."""
        n, m = 10, 20
        hap = _make_hap(n=n, m=m, seed=42)
        rmap = RecombinationMap.constant_map(m=m, p=0.0)
        assignment = _make_assignment(
            n_offspring=4,
            maternal_idx=[0, 0, 2, 4],
            paternal_idx=[1, 3, 5, 7],
        )
        offspring = hap.meiosis(assignment, rmap)

        for i in range(4):
            mat_idx = assignment.maternal_idx[i]
            # Offspring maternal haplotype should be one of mother's two haplotypes
            off_mat = offspring.genotypes[i, :, 0]
            mom_h0 = hap.genotypes[mat_idx, :, 0]
            mom_h1 = hap.genotypes[mat_idx, :, 1]
            assert np.array_equal(off_mat, mom_h0) or np.array_equal(off_mat, mom_h1)


class TestMeiosisSelfMating:
    def test_self_mating(self):
        """Meiosis where maternal_idx == paternal_idx (self-mating)."""
        hap = _make_hap(n=10, m=5, seed=42)
        rmap = RecombinationMap.constant_map(m=5, p=0.5)
        assignment = _make_assignment(
            n_offspring=3,
            maternal_idx=[0, 0, 0],
            paternal_idx=[0, 0, 0],  # same parent
        )
        offspring = hap.meiosis(assignment, rmap)
        assert offspring.genotypes.shape == (3, 5, 2)
        assert np.all((offspring.genotypes == 0) | (offspring.genotypes == 1))


class TestMeiosisOutputProperties:
    def test_output_shape(self):
        hap = _make_hap(n=20, m=10, seed=42)
        rmap = RecombinationMap.constant_map(m=10, p=0.5)
        rm = RandomMating(offspring_per_pair=3)
        assignment = rm.mate(hap.samples, rng=np.random.RandomState(42))
        offspring = hap.meiosis(assignment, rmap)
        assert offspring.genotypes.shape[1] == 10  # m preserved
        assert offspring.genotypes.shape[2] == 2   # diploid
        assert offspring.n == assignment.n_offspring

    def test_output_binary(self):
        hap = _make_hap(n=20, m=10, seed=42)
        rmap = RecombinationMap.constant_map(m=10, p=0.5)
        rm = RandomMating(offspring_per_pair=2)
        assignment = rm.mate(hap.samples, rng=np.random.RandomState(42))
        offspring = hap.meiosis(assignment, rmap)
        assert np.all((offspring.genotypes == 0) | (offspring.genotypes == 1))

    def test_output_metadata(self):
        """Offspring should have correct generation and sample count."""
        hap = _make_hap(n=20, m=5, seed=42)
        rmap = RecombinationMap.constant_map(m=5, p=0.5)
        rm = RandomMating(offspring_per_pair=2)
        assignment = rm.mate(hap.samples, rng=np.random.RandomState(42))
        offspring = hap.meiosis(assignment, rmap)
        assert offspring.generation == assignment.offspring_samples.generation
        assert offspring.m == hap.m


class TestMeiosisLargeFamily:
    def test_many_offspring_same_parents(self):
        """Large family: many offspring from same two parents."""
        hap = _make_hap(n=10, m=10, seed=42)
        rmap = RecombinationMap.constant_map(m=10, p=0.5)
        n_kids = 50
        assignment = _make_assignment(
            n_offspring=n_kids,
            maternal_idx=[0] * n_kids,
            paternal_idx=[1] * n_kids,
        )
        offspring = hap.meiosis(assignment, rmap)
        assert offspring.n == n_kids
        # With 50 offspring and p=0.5, offspring should show variation
        diploid = offspring.genotypes.sum(axis=2)  # (50, 10)
        # Not all rows should be identical
        assert not np.all(diploid == diploid[0])


class TestMeiosisAlleleFrequency:
    def test_af_approximately_preserved(self):
        """Allele frequencies should be approximately preserved across meiosis."""
        n, m = 200, 50
        hap = _make_hap(n=n, m=m, seed=42)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        rm = RandomMating(offspring_per_pair=2)
        assignment = rm.mate(hap.samples, rng=np.random.RandomState(42))
        offspring = hap.meiosis(assignment, rmap)

        parent_af = hap.af_empirical
        offspring_af = offspring.af_empirical

        # Allele frequencies should be correlated
        corr = np.corrcoef(parent_af, offspring_af)[0, 1]
        assert corr > 0.7, f"AF correlation {corr} too low"
