"""Tests for mating edge cases."""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray
from xftsim.nmate import RandomMating, NMateAssignment


def _make_haplotypes(n, seed=42):
    rng = np.random.RandomState(seed)
    m = 10
    geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
    sex = np.tile([0, 1], (n + 1) // 2)[:n]
    samples = SampleMeta(iid=np.arange(n), sex=sex)
    variants = VariantMeta(vid=np.arange(m), af=np.full(m, 0.5))
    return DenseHaplotypeArray(genotypes=geno, samples=samples, variants=variants)


class TestMateEdgeCases:
    def test_unbalanced_sex_3_offspring(self):
        """Unbalanced sex ratio with 3 offspring per pair."""
        # 6 females, 4 males → 4 pairs → 12 offspring
        n = 10
        m = 5
        rng = np.random.RandomState(0)
        geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
        sex = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
        samples = SampleMeta(iid=np.arange(n), sex=sex)
        variants = VariantMeta(vid=np.arange(m), af=np.full(m, 0.5))
        hap = DenseHaplotypeArray(genotypes=geno, samples=samples, variants=variants)
        mate = RandomMating(offspring_per_pair=3)
        assignment = mate.mate(hap.samples, rng=np.random.RandomState(1))
        assert assignment.n_offspring == 4 * 3  # 4 pairs * 3

    def test_large_population(self):
        """Large population should mate without error."""
        hap = _make_haplotypes(n=10000, seed=99)
        mate = RandomMating()
        assignment = mate.mate(hap.samples, rng=np.random.RandomState(0))
        assert assignment.n_offspring == 5000 * 2  # 5000 pairs * 2

    def test_single_pair(self):
        """Minimum population: 1 female + 1 male."""
        n = 2
        m = 5
        geno = np.zeros((n, m, 2), dtype=np.int8)
        sex = np.array([0, 1])
        samples = SampleMeta(iid=np.arange(n), sex=sex)
        variants = VariantMeta(vid=np.arange(m), af=np.full(m, 0.5))
        hap = DenseHaplotypeArray(genotypes=geno, samples=samples, variants=variants)
        mate = RandomMating()
        assignment = mate.mate(hap.samples, rng=np.random.RandomState(0))
        assert assignment.n_offspring == 2
        assert assignment.maternal_idx[0] == 0
        assert assignment.paternal_idx[0] == 1
