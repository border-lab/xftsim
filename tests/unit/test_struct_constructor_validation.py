"""
Unit tests for struct constructor validation error paths.

Tests:
1. DenseHaplotypeArray with mismatched samples.n raises
2. DenseHaplotypeArray with mismatched variants.m raises
3. NPhenotypeArray setitem with wrong shape raises
4. NPhenotypeArray setitem with 2-D array raises
5. VariantMeta __getitem__ on None core field raises KeyError
6. VariantMeta __getitem__ on existing core field returns array
7. VariantMeta __getitem__ on extras returns array
8. VariantMeta __getitem__ on missing extra raises KeyError
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray, NPhenotypeArray


class TestDenseHaplotypeArrayValidation:
    def test_samples_n_mismatch(self):
        """Providing samples with wrong n should raise ValueError."""
        geno = np.zeros((10, 5, 2), dtype=np.int8)
        sm = SampleMeta(iid=np.arange(7))  # n=7, but geno has 10
        with pytest.raises(ValueError, match="samples.n.*must match"):
            DenseHaplotypeArray(genotypes=geno, samples=sm)

    def test_variants_m_mismatch(self):
        """Providing variants with wrong m should raise ValueError."""
        geno = np.zeros((10, 5, 2), dtype=np.int8)
        vm = VariantMeta(vid=np.arange(3))  # m=3, but geno has 5
        with pytest.raises(ValueError, match="variants.m.*must match"):
            DenseHaplotypeArray(genotypes=geno, variants=vm)


class TestNPhenotypeArrayShapeValidation:
    def test_setitem_wrong_length(self):
        """Setting phenotype with wrong length raises ValueError."""
        sm = SampleMeta(iid=np.arange(10))
        pheno = NPhenotypeArray(samples=sm)
        with pytest.raises(ValueError, match="has shape.*expected"):
            pheno['Y'] = np.zeros(5)  # n=10, but providing 5

    def test_setitem_2d_array(self):
        """Setting phenotype with 2-D array raises ValueError."""
        sm = SampleMeta(iid=np.arange(10))
        pheno = NPhenotypeArray(samples=sm)
        with pytest.raises(ValueError, match="has shape.*expected"):
            pheno['Y'] = np.zeros((10, 2))  # wrong shape: (10,2) vs (10,)

    def test_setitem_scalar_broadcasts(self):
        """Setting phenotype with scalar should raise (not broadcast)."""
        sm = SampleMeta(iid=np.arange(10))
        pheno = NPhenotypeArray(samples=sm)
        with pytest.raises(ValueError, match="has shape.*expected"):
            pheno['Y'] = np.float64(1.0)  # scalar, shape ()


class TestVariantMetaGetitem:
    def test_access_existing_core_field(self):
        """Accessing a set core field returns the array."""
        vid = np.array(['v0', 'v1', 'v2'])
        vm = VariantMeta(vid=vid, pos_bp=np.array([100, 200, 300]))
        result = vm['pos_bp']
        np.testing.assert_array_equal(result, [100, 200, 300])

    def test_access_none_core_field(self):
        """Accessing a None core field raises KeyError."""
        vm = VariantMeta(vid=np.array(['v0', 'v1']))
        with pytest.raises(KeyError, match="Field.*is None"):
            vm['chrom']

    def test_access_vid_always_works(self):
        """vid is always set, so __getitem__ should return it."""
        vm = VariantMeta(vid=np.array(['v0', 'v1']))
        result = vm['vid']
        np.testing.assert_array_equal(result, ['v0', 'v1'])

    def test_access_extra_field(self):
        """Extra fields should be accessible via __getitem__."""
        vm = VariantMeta(
            vid=np.array(['v0', 'v1']),
            extra={'coding': np.array([True, False])}
        )
        result = vm['coding']
        np.testing.assert_array_equal(result, [True, False])

    def test_access_missing_extra_raises(self):
        """Accessing a nonexistent extra field raises KeyError."""
        vm = VariantMeta(vid=np.array(['v0', 'v1']))
        with pytest.raises(KeyError):
            vm['nonexistent']

    def test_access_af_when_none(self):
        """Accessing af when None raises KeyError."""
        vm = VariantMeta(vid=np.array(['v0']))
        with pytest.raises(KeyError, match="Field 'af' is None"):
            vm['af']
