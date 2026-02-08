"""
Unit tests for DenseHaplotypeArray compatibility properties.

Tests:
1. shape property (n, 2*m)
2. data property (interleaved format)
3. values property (alias for data)
4. attrs property
5. af_empirical property
6. Indexing with __getitem__ edge cases
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray


def _make_hap(n=6, m=3, seed=42):
    rng = np.random.RandomState(seed)
    sm = SampleMeta(iid=np.arange(n), fid=np.arange(n) // 2)
    vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
    geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
    return DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm, generation=3)


class TestShapeProperty:
    def test_shape_matches_interleaved(self):
        hap = _make_hap(n=6, m=3)
        assert hap.shape == (6, 6)

    def test_shape_single_variant(self):
        hap = _make_hap(n=4, m=1)
        assert hap.shape == (4, 2)


class TestDataProperty:
    def test_data_shape(self):
        hap = _make_hap(n=6, m=3)
        assert hap.data.shape == (6, 6)

    def test_data_interleaving(self):
        """Even columns = hap 0, odd columns = hap 1."""
        hap = _make_hap()
        data = hap.data
        np.testing.assert_array_equal(data[:, 0::2], hap.genotypes[:, :, 0])
        np.testing.assert_array_equal(data[:, 1::2], hap.genotypes[:, :, 1])

    def test_values_is_data(self):
        hap = _make_hap()
        np.testing.assert_array_equal(hap.values, hap.data)


class TestAttrsProperty:
    def test_attrs_has_generation(self):
        hap = _make_hap()
        assert hap.attrs == {'generation': 3}


class TestAFEmpirical:
    def test_af_shape(self):
        hap = _make_hap(n=100, m=5)
        af = hap.af_empirical
        assert af.shape == (5,)

    def test_af_range(self):
        hap = _make_hap(n=100, m=5)
        af = hap.af_empirical
        assert np.all(af >= 0.0)
        assert np.all(af <= 1.0)

    def test_af_all_zeros(self):
        """All-zero genotypes → AF = 0."""
        sm = SampleMeta(iid=np.arange(4), fid=np.arange(4))
        vm = VariantMeta(vid=np.array(['v0', 'v1']))
        geno = np.zeros((4, 2, 2), dtype=np.int8)
        hap = DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)
        np.testing.assert_allclose(hap.af_empirical, [0.0, 0.0])

    def test_af_all_ones(self):
        """All-one genotypes → AF = 1."""
        sm = SampleMeta(iid=np.arange(4), fid=np.arange(4))
        vm = VariantMeta(vid=np.array(['v0', 'v1']))
        geno = np.ones((4, 2, 2), dtype=np.int8)
        hap = DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)
        np.testing.assert_allclose(hap.af_empirical, [1.0, 1.0])

    def test_af_matches_manual(self):
        """Hand-computed allele frequency check."""
        sm = SampleMeta(iid=np.arange(2), fid=np.arange(2))
        vm = VariantMeta(vid=np.array(['v0']))
        # Sample 0: hap0=1, hap1=0
        # Sample 1: hap0=0, hap1=1
        geno = np.array([[[1, 0]], [[0, 1]]], dtype=np.int8)
        hap = DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)
        # hap0 mean = 0.5, hap1 mean = 0.5, average = 0.5
        np.testing.assert_allclose(hap.af_empirical, [0.5])


class TestGetitemEdgeCases:
    def test_single_sample_list(self):
        """hap[[0]] should work for single individual."""
        hap = _make_hap(n=6, m=3)
        result = hap[[0]]
        assert result.n == 1
        assert result.m == 3

    def test_slice_indexing(self):
        hap = _make_hap(n=6, m=3)
        result = hap[1:4]
        assert result.n == 3

    def test_boolean_indexing(self):
        hap = _make_hap(n=6, m=3)
        mask = np.array([True, False, True, False, True, False])
        result = hap[mask]
        assert result.n == 3

    def test_two_dim_indexing(self):
        hap = _make_hap(n=6, m=3)
        result = hap[[0, 1], [0, 2]]
        assert result.n == 2
        assert result.m == 2

    def test_too_many_indices_raises(self):
        hap = _make_hap()
        with pytest.raises(IndexError, match="Too many indices"):
            hap[0, 1, 2]
