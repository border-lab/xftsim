"""
Unit tests for meiosis assertion paths and RecombinationMap edge cases.

Tests:
1. meiosis assertions: dimension mismatch, index OOB, dtype checks
2. RecombinationMap: constructor edge cases, chromosome boundary handling
3. RecombinationMap.from_haplotypes
4. Meiosis class: basic operation
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray
from xftsim.reproduce import RecombinationMap, meiosis, Meiosis


def _make_hap(n=10, m=5, seed=42):
    rng = np.random.RandomState(seed)
    genotypes = rng.binomial(1, 0.3, size=(n, m, 2)).astype(np.int8)
    sm = SampleMeta(iid=np.arange(n))
    vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
    return DenseHaplotypeArray(genotypes=genotypes, generation=0, samples=sm, variants=vm)


class TestMeiosisAssertions:
    def test_dtype_int64_maternal(self):
        """Maternal indices must be int64."""
        hap = _make_hap(n=10, m=5)
        rmap = RecombinationMap.constant_map(m=5, p=0.5)
        mat = np.array([0, 1], dtype=np.int32)  # wrong dtype
        pat = np.array([2, 3], dtype=np.int64)
        with pytest.raises(AssertionError):
            meiosis(hap, rmap, mat, pat)

    def test_dtype_int64_paternal(self):
        """Paternal indices must be int64."""
        hap = _make_hap(n=10, m=5)
        rmap = RecombinationMap.constant_map(m=5, p=0.5)
        mat = np.array([0, 1], dtype=np.int64)
        pat = np.array([2, 3], dtype=np.int32)  # wrong dtype
        with pytest.raises(AssertionError):
            meiosis(hap, rmap, mat, pat)

    def test_mismatched_index_lengths(self):
        """Maternal and paternal indices must have same length."""
        hap = _make_hap(n=10, m=5)
        rmap = RecombinationMap.constant_map(m=5, p=0.5)
        mat = np.array([0, 1, 2], dtype=np.int64)
        pat = np.array([3, 4], dtype=np.int64)
        with pytest.raises(AssertionError):
            meiosis(hap, rmap, mat, pat)

    def test_maternal_index_oob(self):
        """Maternal index >= n should fail assertion."""
        hap = _make_hap(n=10, m=5)
        rmap = RecombinationMap.constant_map(m=5, p=0.5)
        mat = np.array([10], dtype=np.int64)  # out of bounds
        pat = np.array([0], dtype=np.int64)
        with pytest.raises(AssertionError):
            meiosis(hap, rmap, mat, pat)

    def test_paternal_index_oob(self):
        """Paternal index >= n should fail assertion."""
        hap = _make_hap(n=10, m=5)
        rmap = RecombinationMap.constant_map(m=5, p=0.5)
        mat = np.array([0], dtype=np.int64)
        pat = np.array([10], dtype=np.int64)  # out of bounds
        with pytest.raises(AssertionError):
            meiosis(hap, rmap, mat, pat)

    def test_genotype_dtype_not_int8(self):
        """Genotypes must be int8."""
        n, m = 10, 5
        genotypes = np.zeros((n, m, 2), dtype=np.float64)  # wrong dtype
        sm = SampleMeta(iid=np.arange(n))
        vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
        hap = DenseHaplotypeArray(genotypes=genotypes.astype(np.int8), generation=0,
                                   samples=sm, variants=vm)
        # Force wrong dtype after construction
        hap.genotypes = genotypes
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        mat = np.array([0], dtype=np.int64)
        pat = np.array([1], dtype=np.int64)
        with pytest.raises(AssertionError):
            meiosis(hap, rmap, mat, pat)

    def test_valid_meiosis(self):
        """Valid inputs should produce correct output shape."""
        hap = _make_hap(n=10, m=5)
        rmap = RecombinationMap.constant_map(m=5, p=0.5)
        mat = np.array([0, 1, 2], dtype=np.int64)
        pat = np.array([3, 4, 5], dtype=np.int64)
        result = meiosis(hap, rmap, mat, pat)
        assert result.shape == (3, 5, 2)
        assert result.dtype == np.int8


class TestRecombinationMapEdges:
    def test_no_m_no_vid_raises(self):
        """Must provide m or vid."""
        with pytest.raises(ValueError, match="Must provide m or vid"):
            RecombinationMap()

    def test_vid_sets_m(self):
        """VID array sets m automatically."""
        vid = np.array(['a', 'b', 'c'])
        rmap = RecombinationMap(p=0.5, vid=vid)
        assert rmap.m == 3

    def test_invalid_probability(self):
        """Probability > 1 should raise."""
        with pytest.raises(AssertionError):
            RecombinationMap(p=1.5, m=5)

    def test_negative_probability(self):
        """Probability < 0 should raise."""
        with pytest.raises(AssertionError):
            RecombinationMap(p=-0.1, m=5)

    def test_array_probability_length_mismatch(self):
        """Array probability with wrong length should fail."""
        with pytest.raises(AssertionError):
            RecombinationMap(p=np.array([0.1, 0.2, 0.3]), m=5)

    def test_array_probability(self):
        """Array probability should be stored correctly."""
        p = np.array([0.5, 0.1, 0.2, 0.3, 0.4])
        rmap = RecombinationMap(p=p, m=5)
        assert rmap._probabilities.shape == (5,)
        # First position (boundary) forced to 0.5
        assert rmap._probabilities[0] == 0.5

    def test_chromosome_boundaries(self):
        """Chromosome boundaries should force p=0.5."""
        p = np.array([0.1, 0.1, 0.1, 0.1, 0.1])
        chrom = np.array([1, 1, 2, 2, 2])
        rmap = RecombinationMap(p=p, m=5, chrom=chrom)
        # Boundaries: positions 0 (start of chrom 1) and 2 (start of chrom 2)
        assert rmap._probabilities[0] == 0.5
        assert rmap._probabilities[2] == 0.5
        assert rmap._probabilities[1] == 0.1

    def test_default_probability(self):
        """No p provided → default 0.5 everywhere."""
        rmap = RecombinationMap(m=5)
        np.testing.assert_array_equal(rmap._probabilities, np.ones(5) * 0.5)

    def test_from_haplotypes(self):
        """from_haplotypes should create a map matching haplotype dimensions."""
        hap = _make_hap(n=10, m=7)
        rmap = RecombinationMap.from_haplotypes(hap, p=0.3)
        assert rmap.m == 7

    def test_repr(self):
        """RecombinationMap should have a repr."""
        rmap = RecombinationMap.constant_map(m=3, p=0.5)
        r = repr(rmap)
        assert '0.5' in r


class TestMeiosisClass:
    def test_constructor_xor(self):
        """Meiosis requires exactly one of p or rmap."""
        rmap = RecombinationMap.constant_map(m=5, p=0.5)
        # Both provided should fail
        with pytest.raises(AssertionError):
            Meiosis(rmap=rmap, p=0.5)
        # Neither provided should fail
        with pytest.raises(AssertionError):
            Meiosis()
        # Just rmap should work
        m_obj = Meiosis(rmap=rmap)
        assert m_obj.recombination_map is rmap
        # Just p should work
        m_obj2 = Meiosis(p=0.3)
        assert m_obj2._p == 0.3

    def test_get_recombination_map_with_rmap(self):
        """get_recombination_map returns stored map when rmap provided."""
        rmap = RecombinationMap.constant_map(m=5, p=0.5)
        m_obj = Meiosis(rmap=rmap)
        hap = _make_hap(n=10, m=5)
        result = m_obj.get_recombination_map(hap)
        assert result is rmap

    def test_get_recombination_map_with_p(self):
        """get_recombination_map generates map from haplotypes when p provided."""
        m_obj = Meiosis(p=0.3)
        hap = _make_hap(n=10, m=7)
        result = m_obj.get_recombination_map(hap)
        assert isinstance(result, RecombinationMap)
        assert result.m == 7
