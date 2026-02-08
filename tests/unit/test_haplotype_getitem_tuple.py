"""
Unit tests for DenseHaplotypeArray.__getitem__ with tuple indexing.

Tests:
1. Single index returns 2D haplotype array
2. Slice returns subset
3. Tuple (sample_idx, variant_idx)
4. Boolean mask indexing
5. Integer array indexing
6. subset() with sample_idx only
7. subset() with variant_idx only
8. subset() with both indices
9. subset() copy=True vs copy=False
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray


def _make_hap(n=10, m=5, seed=42):
    rng = np.random.RandomState(seed)
    sm = SampleMeta(iid=np.arange(n), fid=np.arange(n) // 2,
                    sex=np.tile([0, 1], n // 2))
    vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
    geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
    return DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)


class TestGetitemSingleIndex:
    def test_list_single_index(self):
        """hap[[0]] should return DenseHaplotypeArray with n=1."""
        hap = _make_hap()
        sub = hap[[0]]
        assert isinstance(sub, DenseHaplotypeArray)
        assert sub.n == 1
        assert sub.m == 5

    def test_slice_index(self):
        """hap[2:5] should return DenseHaplotypeArray with n=3."""
        hap = _make_hap()
        sub = hap[2:5]
        assert isinstance(sub, DenseHaplotypeArray)
        assert sub.n == 3
        assert sub.m == 5

    def test_array_index(self):
        """hap[np.array([1,3,5])] should work."""
        hap = _make_hap()
        sub = hap[np.array([1, 3, 5])]
        assert sub.n == 3


class TestGetitemTuple:
    def test_tuple_sample_variant(self):
        """hap[sample_idx, variant_idx] should subset both."""
        hap = _make_hap()
        sub = hap[np.array([0, 1, 2]), np.array([0, 2])]
        assert isinstance(sub, DenseHaplotypeArray)
        assert sub.n == 3
        assert sub.m == 2

    def test_tuple_slice_both(self):
        """hap[:3, :2] should work."""
        hap = _make_hap()
        sub = hap[:3, :2]
        assert sub.n == 3
        assert sub.m == 2


class TestSubsetMethod:
    def test_subset_samples_only(self):
        """subset(sample_idx=...) keeps all variants."""
        hap = _make_hap()
        sub = hap.subset(sample_idx=np.array([0, 1, 2]))
        assert sub.n == 3
        assert sub.m == 5

    def test_subset_variants_only(self):
        """subset(variant_idx=...) keeps all samples."""
        hap = _make_hap()
        sub = hap.subset(variant_idx=np.array([0, 2, 4]))
        assert sub.n == 10
        assert sub.m == 3

    def test_subset_both(self):
        """subset(sample_idx, variant_idx) keeps both."""
        hap = _make_hap()
        sub = hap.subset(
            sample_idx=np.array([0, 1]),
            variant_idx=np.array([0, 1]),
        )
        assert sub.n == 2
        assert sub.m == 2

    def test_subset_copy_true(self):
        """subset with copy=True should not share memory."""
        hap = _make_hap()
        sub = hap.subset(sample_idx=np.array([0, 1]), copy=True)
        sub.genotypes[0, 0, 0] = 99
        assert hap.genotypes[0, 0, 0] != 99

    def test_subset_copy_false(self):
        """subset with copy=False may share memory."""
        hap = _make_hap()
        sub = hap.subset(sample_idx=slice(0, 3), copy=False)
        # This might share memory (depends on contiguity)
        assert sub.n == 3

    def test_subset_preserves_metadata(self):
        """Subset should preserve correct sample/variant metadata."""
        hap = _make_hap()
        sub = hap.subset(sample_idx=np.array([0, 1, 2]))
        np.testing.assert_array_equal(sub.iid, np.array([0, 1, 2]))

    def test_subset_preserves_variant_metadata(self):
        """Subset should preserve correct variant metadata."""
        hap = _make_hap()
        sub = hap.subset(variant_idx=np.array([1, 3]))
        assert list(sub.vid) == ['v1', 'v3']
