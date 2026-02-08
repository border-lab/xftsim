"""
Unit tests for DenseHaplotypeArray.drop_isel() method.

Tests:
1. Drop samples by index
2. Drop variants by index
3. Drop both samples and variants
4. Drop no indices (identity)
5. Drop all but one sample
6. Drop all but one variant
7. Metadata preserved after drop
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray


def _make_hap(n=10, m=5, seed=42):
    rng = np.random.RandomState(seed)
    sm = SampleMeta(iid=np.arange(n), fid=np.arange(n) // 2)
    vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
    geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
    return DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)


class TestDropSamples:
    def test_drop_single_sample(self):
        hap = _make_hap()
        result = hap.drop_isel(sample=[0])
        assert result.n == 9
        assert result.m == 5

    def test_drop_multiple_samples(self):
        hap = _make_hap()
        result = hap.drop_isel(sample=[0, 2, 4])
        assert result.n == 7
        assert result.m == 5

    def test_drop_samples_preserves_genotypes(self):
        hap = _make_hap()
        result = hap.drop_isel(sample=[0])
        np.testing.assert_array_equal(result.genotypes[0], hap.genotypes[1])

    def test_drop_all_but_one_sample(self):
        hap = _make_hap()
        result = hap.drop_isel(sample=list(range(1, 10)))
        assert result.n == 1
        np.testing.assert_array_equal(result.genotypes[0], hap.genotypes[0])


class TestDropVariants:
    def test_drop_single_variant(self):
        hap = _make_hap()
        result = hap.drop_isel(variant=[0])
        assert result.n == 10
        assert result.m == 4

    def test_drop_multiple_variants(self):
        hap = _make_hap()
        result = hap.drop_isel(variant=[1, 3])
        assert result.n == 10
        assert result.m == 3

    def test_drop_all_but_one_variant(self):
        hap = _make_hap()
        result = hap.drop_isel(variant=list(range(1, 5)))
        assert result.m == 1


class TestDropBoth:
    def test_drop_samples_and_variants(self):
        hap = _make_hap()
        result = hap.drop_isel(sample=[0, 1], variant=[0])
        assert result.n == 8
        assert result.m == 4

    def test_drop_none_is_identity(self):
        hap = _make_hap()
        result = hap.drop_isel()
        assert result.n == 10
        assert result.m == 5
        np.testing.assert_array_equal(result.genotypes, hap.genotypes)


class TestDropMetadata:
    def test_sample_meta_updated(self):
        hap = _make_hap()
        result = hap.drop_isel(sample=[0, 1])
        assert len(result.samples.iid) == 8
        assert result.samples.iid[0] == 2

    def test_variant_meta_updated(self):
        hap = _make_hap()
        result = hap.drop_isel(variant=[0])
        assert len(result.variants.vid) == 4
        assert result.variants.vid[0] == 'v1'

    def test_returns_copy(self):
        """drop_isel should return a copy, not a view."""
        hap = _make_hap()
        result = hap.drop_isel(sample=[0])
        result.genotypes[0, 0, 0] = 99
        assert hap.genotypes[1, 0, 0] != 99
