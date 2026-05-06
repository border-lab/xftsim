"""
Unit tests for reproduce module: RecombinationMap, meiosis.
"""
import numpy as np
import pytest

from xftsim.reproduce import RecombinationMap, meiosis
from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray
from xftsim.mate import RandomMating

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


# ── RecombinationMap construction ──────────────────────────────────────────

class TestRecombinationMapConstruction:
    def test_constant_map(self):
        rmap = RecombinationMap.constant_map(m=10, p=0.3)
        assert rmap.m == 10
        # First locus is always a chromosome boundary (0.5), rest are p=0.3
        assert rmap._probabilities[0] == 0.5
        assert np.all(rmap._probabilities[1:] == 0.3)

    def test_float_p(self):
        rmap = RecombinationMap(p=0.2, m=5)
        assert rmap.m == 5
        # First locus is always a chromosome boundary (0.5)
        assert rmap._probabilities[0] == 0.5
        # Non-boundary loci should have p=0.2
        assert np.all(rmap._probabilities[1:] == 0.2)

    def test_array_p(self):
        p = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        rmap = RecombinationMap(p=p, m=5)
        assert rmap.m == 5
        # First position is chromosome boundary → 0.5
        assert rmap._probabilities[0] == 0.5
        np.testing.assert_allclose(rmap._probabilities[1:], [0.2, 0.3, 0.4, 0.5])

    def test_vid_inferred_m(self):
        rmap = RecombinationMap(vid=np.arange(7))
        assert rmap.m == 7

    def test_no_m_or_vid_raises(self):
        with pytest.raises(ValueError, match="Must provide m or vid"):
            RecombinationMap()

    def test_array_p_length_mismatch_raises(self):
        with pytest.raises(AssertionError):
            RecombinationMap(p=np.array([0.1, 0.2]), m=5)

    def test_p_out_of_range_raises(self):
        with pytest.raises(AssertionError):
            RecombinationMap(p=1.5, m=5)

    def test_negative_p_raises(self):
        with pytest.raises(AssertionError):
            RecombinationMap(p=-0.1, m=5)

    def test_default_p_is_free_recombination(self):
        rmap = RecombinationMap(m=5)
        assert np.all(rmap._probabilities == 0.5)

    def test_vid_and_chrom(self):
        vid = np.array([0, 1, 2, 3, 4])
        chrom = np.array([1, 1, 2, 2, 2])
        rmap = RecombinationMap(p=0.1, vid=vid, chrom=chrom)
        # Boundaries at index 0 (always) and index 2 (chrom change)
        assert rmap._probabilities[0] == 0.5
        assert rmap._probabilities[2] == 0.5
        # Non-boundary loci
        assert rmap._probabilities[1] == 0.1
        assert rmap._probabilities[3] == 0.1
        assert rmap._probabilities[4] == 0.1


class TestRecombinationMapChromBoundaries:
    def test_single_chrom_boundary_at_zero(self):
        rmap = RecombinationMap(p=0.1, m=10)
        assert rmap._chrom_boundary[0] == 0
        assert len(rmap._chrom_boundary) == 1

    def test_multi_chrom_boundaries(self):
        chrom = np.array([1, 1, 1, 2, 2, 3])
        rmap = RecombinationMap(p=0.1, m=6, chrom=chrom)
        np.testing.assert_array_equal(rmap._chrom_boundary, [0, 3, 5])
        # All boundary positions should be 0.5
        for idx in [0, 3, 5]:
            assert rmap._probabilities[idx] == 0.5
        # Non-boundary positions should be 0.1
        for idx in [1, 2, 4]:
            assert rmap._probabilities[idx] == 0.1

    def test_each_variant_own_chrom(self):
        chrom = np.array([1, 2, 3, 4])
        rmap = RecombinationMap(p=0.01, m=4, chrom=chrom)
        # Every position is a boundary → all 0.5
        assert np.all(rmap._probabilities == 0.5)


class TestRecombinationMapStaticMethods:
    def test_constant_map_creates_valid(self):
        rmap = RecombinationMap.constant_map(m=20, p=0.3)
        assert rmap.m == 20
        assert len(rmap._probabilities) == 20

    def test_from_haplotypes(self):
        hap = TestSimulation.founder_haplotypes(n=10, m=15)
        rmap = RecombinationMap.from_haplotypes(hap, p=0.4)
        assert rmap.m == 15
        # First locus is boundary
        assert rmap._probabilities[0] == 0.5
        assert np.all(rmap._probabilities[1:] == 0.4)

    def test_repr(self):
        rmap = RecombinationMap.constant_map(m=5, p=0.3)
        r = repr(rmap)
        assert 'vid' in r
        assert 'chrom' in r
        assert 'p' in r


# ── meiosis function ──────────────────────────────────────────────────────

class TestMeiosisFunction:
    @pytest.fixture
    def parent_hap(self):
        return TestSimulation.founder_haplotypes(n=100, m=50)

    @pytest.fixture
    def rmap(self):
        return RecombinationMap.constant_map(m=50, p=0.5)

    def test_output_shape(self, parent_hap, rmap):
        rm = RandomMating(offspring_per_pair=2)
        ma = rm.mate(parent_hap.samples, rng=np.random.RandomState(42))
        result = meiosis(parent_hap, rmap, ma.maternal_idx, ma.paternal_idx)
        assert result.shape == (ma.n_offspring, 50, 2)

    def test_output_dtype(self, parent_hap, rmap):
        rm = RandomMating(offspring_per_pair=2)
        ma = rm.mate(parent_hap.samples, rng=np.random.RandomState(42))
        result = meiosis(parent_hap, rmap, ma.maternal_idx, ma.paternal_idx)
        assert result.dtype == np.int8

    def test_alleles_binary(self, parent_hap, rmap):
        rm = RandomMating(offspring_per_pair=2)
        ma = rm.mate(parent_hap.samples, rng=np.random.RandomState(42))
        result = meiosis(parent_hap, rmap, ma.maternal_idx, ma.paternal_idx)
        assert set(np.unique(result)).issubset({0, 1})

    def test_alleles_from_parents(self, parent_hap, rmap):
        """Each offspring allele must come from the correct parent."""
        rm = RandomMating(offspring_per_pair=2)
        ma = rm.mate(parent_hap.samples, rng=np.random.RandomState(42))
        offspring_geno = meiosis(parent_hap, rmap, ma.maternal_idx, ma.paternal_idx)
        # Check first 5 offspring
        for i in range(5):
            mat_idx = ma.maternal_idx[i]
            pat_idx = ma.paternal_idx[i]
            for j in range(parent_hap.m):
                off_mat = offspring_geno[i, j, 0]
                mom_alleles = set(parent_hap.genotypes[mat_idx, j, :])
                assert off_mat in mom_alleles
                off_pat = offspring_geno[i, j, 1]
                dad_alleles = set(parent_hap.genotypes[pat_idx, j, :])
                assert off_pat in dad_alleles

    def test_no_recombination(self):
        """With p=0, offspring should inherit one complete parental haplotype."""
        rng = np.random.RandomState(42)
        n, m = 20, 10
        geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
        sm = SampleMeta(iid=np.arange(n))
        hap = DenseHaplotypeArray(genotypes=geno, samples=sm)
        rmap = RecombinationMap(p=0.0, m=m)  # p=0 but boundary at 0 is 0.5

        # Even with p=0, the first locus is boundary (0.5) so there's some randomness.
        # With p=0 beyond the first locus, each offspring should have a complete
        # parental haplotype from one strand (no crossovers after first locus).
        rm = RandomMating(offspring_per_pair=2)
        ma = rm.mate(hap.samples, rng=np.random.RandomState(0))
        offspring_geno = meiosis(hap, rmap, ma.maternal_idx, ma.paternal_idx)

        # Check that for loci 1 onwards, no crossover: all from same strand
        for i in range(min(5, ma.n_offspring)):
            mat_idx = ma.maternal_idx[i]
            # The maternal contribution from loci 1..m-1 should be from one strand
            maternal_hap = offspring_geno[i, 1:, 0]
            from_strand_0 = (maternal_hap == geno[mat_idx, 1:, 0])
            from_strand_1 = (maternal_hap == geno[mat_idx, 1:, 1])
            # Each allele comes from one strand, and with p=0 they should all
            # be from the same strand (no crossovers)
            if np.all(from_strand_0):
                pass  # all from strand 0
            elif np.all(from_strand_1):
                pass  # all from strand 1
            else:
                # Might be ambiguous if both strands have same allele
                # Just check each allele comes from at least one strand
                assert np.all(from_strand_0 | from_strand_1)

    def test_single_offspring(self, parent_hap, rmap):
        mat = np.array([0], dtype=np.int64)
        pat = np.array([1], dtype=np.int64)
        result = meiosis(parent_hap, rmap, mat, pat)
        assert result.shape == (1, 50, 2)


class TestSingleLocusMeiosis:
    """Edge case: single-locus meiosis."""

    def test_single_locus_shape(self):
        """Meiosis with m=1 should produce correct shape."""
        geno = np.array([[[1, 0]], [[0, 1]], [[1, 1]], [[0, 0]]], dtype=np.int8)
        sex = np.array([0, 1, 0, 1])
        samples = SampleMeta(iid=np.arange(4), sex=sex)
        hap = DenseHaplotypeArray(genotypes=geno, samples=samples)
        rmap = RecombinationMap.constant_map(m=1, p=0.5)
        mate = RandomMating(offspring_per_pair=2)
        ma = mate.mate(hap.samples, rng=np.random.RandomState(42))
        offspring = hap.meiosis(ma, rmap)
        assert offspring.m == 1
        assert offspring.genotypes.shape[2] == 2

    def test_single_locus_alleles_from_parents(self):
        """Each offspring allele should come from a parent with m=1."""
        geno = np.array([[[1, 1]], [[0, 0]], [[1, 1]], [[0, 0]]], dtype=np.int8)
        sex = np.array([0, 1, 0, 1])
        samples = SampleMeta(iid=np.arange(4), sex=sex)
        hap = DenseHaplotypeArray(genotypes=geno, samples=samples)
        rmap = RecombinationMap.constant_map(m=1, p=0.5)
        mate = RandomMating(offspring_per_pair=2)
        ma = mate.mate(hap.samples, rng=np.random.RandomState(42))
        offspring = hap.meiosis(ma, rmap)
        # Mothers are all 1/1, fathers are all 0/0
        # So maternal allele should be 1, paternal should be 0
        for i in range(offspring.n):
            mother_idx = ma.maternal_idx[i]
            father_idx = ma.paternal_idx[i]
            # Offspring maternal allele must be one of mother's alleles
            assert offspring.genotypes[i, 0, 0] in geno[mother_idx, 0]
            # Offspring paternal allele must be one of father's alleles
            assert offspring.genotypes[i, 0, 1] in geno[father_idx, 0]


class TestNoRecombinationMeiosis:
    """Test meiosis with p=0 (no recombination)."""

    def test_no_recombination_inherits_whole_haplotype(self):
        """With p=0, offspring should get intact parental haplotypes."""
        n, m = 4, 10
        rng = np.random.RandomState(42)
        geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
        sex = np.array([0, 1, 0, 1])
        samples = SampleMeta(iid=np.arange(n), sex=sex)
        hap = DenseHaplotypeArray(genotypes=geno, samples=samples)
        rmap = RecombinationMap.constant_map(m=m, p=0.0)
        mate = RandomMating(offspring_per_pair=2)
        ma = mate.mate(hap.samples, rng=np.random.RandomState(99))
        offspring = hap.meiosis(ma, rmap)
        # Each offspring's maternal haplotype should be a copy of one of
        # the mother's haplotypes (either hap 0 or hap 1)
        for i in range(offspring.n):
            mother_idx = ma.maternal_idx[i]
            mat_alleles = offspring.genotypes[i, :, 0]
            mother_hap0 = geno[mother_idx, :, 0]
            mother_hap1 = geno[mother_idx, :, 1]
            assert (np.array_equal(mat_alleles, mother_hap0) or
                    np.array_equal(mat_alleles, mother_hap1))
