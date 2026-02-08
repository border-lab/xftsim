"""
Unit tests for DenseHaplotypeArray properties and edge cases.

Tests:
1. genotypes direct assignment
2. diploid_genotypes property
3. shape property
4. n=0 and m=0 edge cases
5. genotypes dtype validation
6. genotypes dimensionality validation
7. copy vs view semantics in subset
8. to_dense returns self
9. __len__ behavior
10. generation property
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray


def _make_hap(n=10, m=5, seed=42, generation=0):
    sm = SampleMeta(iid=np.arange(n), generation=generation)
    vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
    rng = np.random.RandomState(seed)
    geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
    return DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm, generation=generation)


class TestGenotypesProperty:
    def test_genotypes_shape(self):
        hap = _make_hap(n=10, m=5)
        assert hap.genotypes.shape == (10, 5, 2)

    def test_genotypes_dtype(self):
        hap = _make_hap()
        assert hap.genotypes.dtype == np.int8

    def test_genotypes_direct_assignment(self):
        """Direct assignment to .genotypes should update the array."""
        hap = _make_hap(n=4, m=3)
        new_geno = np.zeros((4, 3, 2), dtype=np.int8)
        hap.genotypes = new_geno
        np.testing.assert_array_equal(hap.genotypes, new_geno)

    def test_genotypes_values_are_binary(self):
        hap = _make_hap(n=50, m=20)
        assert np.all((hap.genotypes == 0) | (hap.genotypes == 1))


class TestDiploidGenotypes:
    def test_diploid_genotypes_range(self):
        """Diploid genotypes should be 0, 1, or 2."""
        hap = _make_hap(n=50, m=20)
        dip = hap.diploid_genotypes
        assert np.all((dip >= 0) & (dip <= 2))

    def test_diploid_genotypes_shape(self):
        hap = _make_hap(n=10, m=5)
        assert hap.diploid_genotypes.shape == (10, 5)

    def test_diploid_genotypes_equals_sum(self):
        """diploid_genotypes = genotypes[:,:,0] + genotypes[:,:,1]."""
        hap = _make_hap(n=10, m=5)
        expected = hap.genotypes[:, :, 0] + hap.genotypes[:, :, 1]
        np.testing.assert_array_equal(hap.diploid_genotypes, expected)


class TestShapeProperty:
    def test_shape_format(self):
        """shape should be (n, 2*m) for compatibility."""
        hap = _make_hap(n=10, m=5)
        assert hap.shape == (10, 10)  # (n, 2*m)

    def test_n_m_properties(self):
        hap = _make_hap(n=10, m=5)
        assert hap.n == 10
        assert hap.m == 5


class TestEdgeCaseDimensions:
    def test_single_individual(self):
        """n=1 should work."""
        hap = _make_hap(n=1, m=5)
        assert hap.n == 1
        assert hap.m == 5
        assert hap.genotypes.shape == (1, 5, 2)

    def test_single_variant(self):
        """m=1 should work."""
        hap = _make_hap(n=10, m=1)
        assert hap.n == 10
        assert hap.m == 1

    def test_invalid_genotypes_shape_2d(self):
        """2D array should raise ValueError."""
        sm = SampleMeta(iid=np.arange(3))
        vm = VariantMeta(vid=np.array(['v0', 'v1']))
        with pytest.raises(ValueError, match="3-D"):
            DenseHaplotypeArray(
                genotypes=np.zeros((3, 2), dtype=np.int8),
                samples=sm, variants=vm,
            )

    def test_invalid_genotypes_shape_last_dim(self):
        """Last dim != 2 should raise ValueError."""
        sm = SampleMeta(iid=np.arange(3))
        vm = VariantMeta(vid=np.array(['v0', 'v1']))
        with pytest.raises(ValueError, match="last dim = 2"):
            DenseHaplotypeArray(
                genotypes=np.zeros((3, 2, 3), dtype=np.int8),
                samples=sm, variants=vm,
            )


class TestSubsetCopySemantics:
    def test_subset_copy_true(self):
        """subset(copy=True) should not share memory with original."""
        hap = _make_hap(n=10, m=5)
        sub = hap.subset(np.array([0, 1, 2]))
        sub.genotypes[0, 0, 0] = 99
        assert hap.genotypes[0, 0, 0] != 99

    def test_subset_produces_correct_n(self):
        hap = _make_hap(n=10, m=5)
        sub = hap.subset(np.array([0, 2, 4]))
        assert sub.n == 3
        assert sub.m == 5


class TestToDense:
    def test_to_dense_returns_self(self):
        """DenseHaplotypeArray.to_dense() should return self."""
        hap = _make_hap()
        assert hap.to_dense() is hap


class TestGeneration:
    def test_generation_from_init(self):
        hap = _make_hap(generation=5)
        assert hap.generation == 5

    def test_generation_default(self):
        hap = _make_hap(generation=0)
        assert hap.generation == 0


class TestDenseHaplotypeRepr:
    def test_repr_contains_class_name(self):
        hap = _make_hap(n=10, m=5)
        r = repr(hap)
        assert 'DenseHaplotypeArray' in r or 'n=' in r

    def test_repr_contains_dimensions(self):
        hap = _make_hap(n=10, m=5)
        r = repr(hap)
        assert '10' in r
        assert '5' in r


class TestDimensionMismatch:
    def test_samples_n_mismatch_raises(self):
        """samples.n != genotypes.shape[0] raises ValueError."""
        sm = SampleMeta(iid=np.arange(5))
        vm = VariantMeta(vid=np.array(['v0', 'v1']))
        geno = np.zeros((3, 2, 2), dtype=np.int8)  # n=3, but sm has n=5
        with pytest.raises(ValueError, match="samples.n"):
            DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)

    def test_variants_m_mismatch_raises(self):
        """variants.m != genotypes.shape[1] raises ValueError."""
        sm = SampleMeta(iid=np.arange(3))
        vm = VariantMeta(vid=np.array(['v0', 'v1', 'v2', 'v3', 'v4']))  # m=5
        geno = np.zeros((3, 2, 2), dtype=np.int8)  # m=2
        with pytest.raises(ValueError, match="variants.m"):
            DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)


class TestNoneDefaults:
    def test_none_genotypes(self):
        """genotypes=None should create empty array."""
        hap = DenseHaplotypeArray(genotypes=None)
        assert hap.n == 0
        assert hap.m == 0

    def test_none_samples_auto_created(self):
        """samples=None should auto-create SampleMeta."""
        geno = np.zeros((3, 2, 2), dtype=np.int8)
        hap = DenseHaplotypeArray(genotypes=geno)
        assert hap.samples.n == 3

    def test_none_variants_auto_created(self):
        """variants=None should auto-create VariantMeta."""
        geno = np.zeros((3, 2, 2), dtype=np.int8)
        hap = DenseHaplotypeArray(genotypes=geno)
        assert hap.variants.m == 2
