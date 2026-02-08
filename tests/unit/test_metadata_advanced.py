"""
Unit tests for SampleMeta and VariantMeta advanced methods.

Tests:
1. SampleMeta: unique_identifier, with_generation, n_fam/n_female/n_male, extra fields subset
2. VariantMeta: subset with all optional fields, __getitem__ on extras, extra validation
3. VariantMeta: repr with chrom/af, __getitem__ on None core field
4. NPhenotypeArray: __contains__, repr, keys
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, NPhenotypeArray


class TestSampleMetaUniqueIdentifier:
    def test_format(self):
        """unique_identifier should be '{generation}.{iid}.{fid}'."""
        sm = SampleMeta(iid=np.array([10, 20]), fid=np.array([1, 2]), generation=3)
        uid = sm.unique_identifier
        assert uid[0] == '3.10.1'
        assert uid[1] == '3.20.2'

    def test_shape(self):
        """unique_identifier shape should match n."""
        sm = SampleMeta(iid=np.arange(5))
        assert sm.unique_identifier.shape == (5,)

    def test_uniqueness(self):
        """Identifiers should be unique when iid are unique."""
        sm = SampleMeta(iid=np.arange(10))
        uid = sm.unique_identifier
        assert len(set(uid)) == 10


class TestSampleMetaWithGeneration:
    def test_generation_changed(self):
        """with_generation should return new SampleMeta with new generation."""
        sm = SampleMeta(iid=np.arange(5), generation=0)
        sm2 = sm.with_generation(7)
        assert sm2.generation == 7
        assert sm.generation == 0  # original unchanged

    def test_data_preserved(self):
        """with_generation should preserve all other fields."""
        sm = SampleMeta(
            iid=np.array([1, 2, 3]),
            fid=np.array([10, 10, 20]),
            sex=np.array([0, 1, 0]),
            extra={'group': np.array([1, 2, 3])},
        )
        sm2 = sm.with_generation(5)
        np.testing.assert_array_equal(sm2.iid, sm.iid)
        np.testing.assert_array_equal(sm2.fid, sm.fid)
        np.testing.assert_array_equal(sm2.sex, sm.sex)
        np.testing.assert_array_equal(sm2.extra['group'], sm.extra['group'])


class TestSampleMetaProperties:
    def test_n_fam(self):
        """n_fam should count unique families."""
        sm = SampleMeta(iid=np.arange(6), fid=np.array([1, 1, 2, 2, 3, 3]))
        assert sm.n_fam == 3

    def test_n_female_n_male(self):
        """n_female/n_male should count by sex."""
        sm = SampleMeta(iid=np.arange(5), sex=np.array([0, 0, 0, 1, 1]))
        assert sm.n_female == 3
        assert sm.n_male == 2

    def test_all_same_sex(self):
        """All female or all male."""
        sm_f = SampleMeta(iid=np.arange(3), sex=np.zeros(3, dtype=int))
        assert sm_f.n_female == 3
        assert sm_f.n_male == 0

        sm_m = SampleMeta(iid=np.arange(3), sex=np.ones(3, dtype=int))
        assert sm_m.n_female == 0
        assert sm_m.n_male == 3


class TestSampleMetaExtraSubset:
    def test_extra_preserved_on_subset(self):
        """Extra fields should be subsetted along with core fields."""
        sm = SampleMeta(
            iid=np.arange(5),
            extra={'score': np.array([10.0, 20.0, 30.0, 40.0, 50.0])},
        )
        sub = sm.subset(np.array([0, 2, 4]))
        assert sub.n == 3
        np.testing.assert_array_equal(sub.extra['score'], [10.0, 30.0, 50.0])

    def test_extra_length_mismatch_raises(self):
        """Extra field with wrong length should raise."""
        with pytest.raises(ValueError, match="extra.*length"):
            SampleMeta(
                iid=np.arange(5),
                extra={'bad': np.array([1, 2, 3])},  # length 3 != 5
            )


class TestVariantMetaSubsetFull:
    def test_subset_all_fields(self):
        """subset should preserve all optional fields."""
        vm = VariantMeta(
            vid=np.array(['v0', 'v1', 'v2']),
            chrom=np.array([1, 1, 2]),
            pos_bp=np.array([100, 200, 300]),
            pos_cM=np.array([0.1, 0.2, 0.3]),
            af=np.array([0.1, 0.5, 0.9]),
            zero_allele=np.array(['A', 'C', 'G']),
            one_allele=np.array(['T', 'G', 'A']),
            extra={'maf_bin': np.array([0, 1, 2])},
        )
        sub = vm.subset(np.array([0, 2]))
        assert sub.m == 2
        np.testing.assert_array_equal(sub.vid, ['v0', 'v2'])
        np.testing.assert_array_equal(sub.chrom, [1, 2])
        np.testing.assert_array_equal(sub.pos_bp, [100, 300])
        np.testing.assert_allclose(sub.pos_cM, [0.1, 0.3])
        np.testing.assert_allclose(sub.af, [0.1, 0.9])
        np.testing.assert_array_equal(sub.zero_allele, ['A', 'G'])
        np.testing.assert_array_equal(sub.one_allele, ['T', 'A'])
        np.testing.assert_array_equal(sub.extra['maf_bin'], [0, 2])

    def test_subset_none_fields(self):
        """subset should handle None optional fields."""
        vm = VariantMeta(vid=np.array(['v0', 'v1']))
        sub = vm.subset(np.array([0]))
        assert sub.m == 1
        assert sub.chrom is None
        assert sub.pos_bp is None
        assert sub.af is None


class TestVariantMetaGetitem:
    def test_getitem_core_field(self):
        """__getitem__ should access core fields."""
        vm = VariantMeta(vid=np.array(['v0', 'v1']), af=np.array([0.1, 0.5]))
        np.testing.assert_array_equal(vm['vid'], ['v0', 'v1'])
        np.testing.assert_allclose(vm['af'], [0.1, 0.5])

    def test_getitem_none_field_raises(self):
        """Accessing a None core field should raise KeyError."""
        vm = VariantMeta(vid=np.array(['v0']))
        with pytest.raises(KeyError, match="None"):
            vm['chrom']

    def test_getitem_extra(self):
        """__getitem__ should access extra fields."""
        vm = VariantMeta(
            vid=np.array(['v0', 'v1']),
            extra={'coding': np.array([True, False])},
        )
        np.testing.assert_array_equal(vm['coding'], [True, False])

    def test_getitem_missing_extra_raises(self):
        """Accessing missing extra field should raise KeyError."""
        vm = VariantMeta(vid=np.array(['v0']))
        with pytest.raises(KeyError):
            vm['nonexistent']


class TestVariantMetaRepr:
    def test_repr_minimal(self):
        """Repr with only vid."""
        vm = VariantMeta(vid=np.array(['v0', 'v1']))
        r = repr(vm)
        assert 'm=2' in r

    def test_repr_with_chrom(self):
        """Repr should show n_chrom when chrom is set."""
        vm = VariantMeta(
            vid=np.array(['v0', 'v1', 'v2']),
            chrom=np.array([1, 1, 2]),
        )
        r = repr(vm)
        assert 'n_chrom=2' in r

    def test_repr_with_af(self):
        """Repr should show af=True when af is set."""
        vm = VariantMeta(
            vid=np.array(['v0', 'v1']),
            af=np.array([0.1, 0.9]),
        )
        r = repr(vm)
        assert 'af=True' in r


class TestVariantMetaExtraValidation:
    def test_extra_length_mismatch_raises(self):
        """Extra field with wrong length should raise."""
        with pytest.raises(ValueError, match="extra.*length"):
            VariantMeta(
                vid=np.array(['v0', 'v1']),
                extra={'bad': np.array([1, 2, 3])},
            )


class TestNPhenotypeArrayMethods:
    def test_contains(self):
        """__contains__ should check key existence."""
        sm = SampleMeta(iid=np.arange(5))
        pa = NPhenotypeArray(samples=sm, values={'x': np.zeros(5)})
        assert 'x' in pa
        assert 'y' not in pa

    def test_repr(self):
        """repr should show n and keys."""
        sm = SampleMeta(iid=np.arange(5))
        pa = NPhenotypeArray(samples=sm, values={'x': np.zeros(5), 'y': np.ones(5)})
        r = repr(pa)
        assert 'n=5' in r
        assert 'x' in r
        assert 'y' in r

    def test_keys(self):
        """keys should return dict_keys."""
        sm = SampleMeta(iid=np.arange(5))
        pa = NPhenotypeArray(samples=sm, values={'a': np.zeros(5), 'b': np.ones(5)})
        assert set(pa.keys) == {'a', 'b'}

    def test_subset_multiple_keys(self):
        """subset should preserve all keys."""
        sm = SampleMeta(iid=np.arange(5))
        pa = NPhenotypeArray(samples=sm, values={
            'x': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            'y': np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
        })
        sub = pa.subset(np.array([1, 3]))
        assert sub.samples.n == 2
        np.testing.assert_array_equal(sub['x'], [2.0, 4.0])
        np.testing.assert_array_equal(sub['y'], [20.0, 40.0])

    def test_empty_init(self):
        """NPhenotypeArray with no values should have 0 keys."""
        sm = SampleMeta(iid=np.arange(5))
        pa = NPhenotypeArray(samples=sm)
        assert len(pa.keys) == 0
