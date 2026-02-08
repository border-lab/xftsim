"""
Unit tests for SampleMeta and VariantMeta construction, properties, and methods.

Tests:
1. SampleMeta: n, n_fam, n_female, n_male, unique_identifier
2. SampleMeta: subset, with_generation, extra dict
3. SampleMeta: __repr__
4. VariantMeta: m, __getitem__ core/extra, missing field raises
5. VariantMeta: subset with optionals, extra, __repr__
6. VariantMeta: all optional fields None
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta


class TestSampleMetaProperties:
    def test_n(self):
        sm = SampleMeta(iid=np.arange(10))
        assert sm.n == 10

    def test_n_fam(self):
        sm = SampleMeta(iid=np.arange(6), fid=np.array([0, 0, 1, 1, 2, 2]))
        assert sm.n_fam == 3

    def test_n_fam_default_all_unique(self):
        """Default FID = IID → each sample is its own family."""
        sm = SampleMeta(iid=np.arange(5))
        assert sm.n_fam == 5

    def test_n_female(self):
        sm = SampleMeta(iid=np.arange(4), sex=np.array([0, 0, 1, 1]))
        assert sm.n_female == 2

    def test_n_male(self):
        sm = SampleMeta(iid=np.arange(4), sex=np.array([0, 0, 1, 1]))
        assert sm.n_male == 2

    def test_unique_identifier(self):
        sm = SampleMeta(iid=np.array([10, 20]), fid=np.array([1, 2]), generation=3)
        uid = sm.unique_identifier
        assert uid[0] == '3.10.1'
        assert uid[1] == '3.20.2'

    def test_default_sex_alternates(self):
        """Default sex should alternate 0, 1, 0, 1, ..."""
        sm = SampleMeta(iid=np.arange(6))
        expected = np.array([0, 1, 0, 1, 0, 1])
        np.testing.assert_array_equal(sm.sex, expected)

    def test_generation_default_zero(self):
        sm = SampleMeta(iid=np.arange(5))
        assert sm.generation == 0


class TestSampleMetaSubset:
    def test_subset_correct_n(self):
        sm = SampleMeta(iid=np.arange(10))
        sub = sm.subset(np.array([0, 3, 7]))
        assert sub.n == 3

    def test_subset_preserves_iid(self):
        sm = SampleMeta(iid=np.array([10, 20, 30, 40, 50]))
        sub = sm.subset(np.array([1, 3]))
        np.testing.assert_array_equal(sub.iid, [20, 40])

    def test_subset_preserves_fid(self):
        sm = SampleMeta(iid=np.arange(4), fid=np.array([0, 0, 1, 1]))
        sub = sm.subset(np.array([0, 2]))
        np.testing.assert_array_equal(sub.fid, [0, 1])

    def test_subset_preserves_extra(self):
        sm = SampleMeta(iid=np.arange(5), extra={'age': np.array([20, 30, 40, 50, 60])})
        sub = sm.subset(np.array([0, 4]))
        np.testing.assert_array_equal(sub.extra['age'], [20, 60])

    def test_subset_preserves_generation(self):
        sm = SampleMeta(iid=np.arange(5), generation=7)
        sub = sm.subset(np.array([0, 1]))
        assert sub.generation == 7


class TestSampleMetaWithGeneration:
    def test_changes_generation(self):
        sm = SampleMeta(iid=np.arange(5), generation=0)
        sm2 = sm.with_generation(3)
        assert sm2.generation == 3
        assert sm.generation == 0  # original unchanged

    def test_preserves_data(self):
        sm = SampleMeta(iid=np.arange(5), fid=np.array([0, 0, 1, 1, 2]))
        sm2 = sm.with_generation(5)
        np.testing.assert_array_equal(sm2.iid, sm.iid)
        np.testing.assert_array_equal(sm2.fid, sm.fid)


class TestSampleMetaExtra:
    def test_extra_validation_length(self):
        """Extra arrays with wrong length should raise."""
        with pytest.raises(ValueError, match="extra"):
            SampleMeta(iid=np.arange(5), extra={'x': np.ones(3)})

    def test_extra_valid(self):
        sm = SampleMeta(iid=np.arange(5), extra={'weight': np.ones(5)})
        np.testing.assert_array_equal(sm.extra['weight'], np.ones(5))


class TestSampleMetaRepr:
    def test_repr(self):
        sm = SampleMeta(iid=np.arange(10))
        r = repr(sm)
        assert 'SampleMeta' in r
        assert 'n=10' in r
        assert 'generation=' in r


class TestVariantMetaProperties:
    def test_m(self):
        vm = VariantMeta(vid=np.array(['a', 'b', 'c']))
        assert vm.m == 3

    def test_getitem_core(self):
        vm = VariantMeta(vid=np.array(['v0', 'v1']),
                         chrom=np.array([1, 2]))
        np.testing.assert_array_equal(vm['chrom'], [1, 2])

    def test_getitem_none_raises(self):
        vm = VariantMeta(vid=np.array(['v0']))
        with pytest.raises(KeyError, match="None"):
            _ = vm['chrom']

    def test_getitem_extra(self):
        vm = VariantMeta(vid=np.array(['v0']),
                         extra={'coding': np.array([True])})
        np.testing.assert_array_equal(vm['coding'], [True])

    def test_getitem_missing_extra_raises(self):
        vm = VariantMeta(vid=np.array(['v0']))
        with pytest.raises(KeyError):
            _ = vm['nonexistent']


class TestVariantMetaSubset:
    def test_subset_core(self):
        vm = VariantMeta(vid=np.array(['a', 'b', 'c']),
                         chrom=np.array([1, 1, 2]))
        sub = vm.subset(np.array([0, 2]))
        assert sub.m == 2
        np.testing.assert_array_equal(sub.vid, ['a', 'c'])
        np.testing.assert_array_equal(sub.chrom, [1, 2])

    def test_subset_none_optionals(self):
        vm = VariantMeta(vid=np.array(['a', 'b']))
        sub = vm.subset(np.array([0]))
        assert sub.m == 1
        assert sub.chrom is None
        assert sub.pos_bp is None

    def test_subset_extra(self):
        vm = VariantMeta(vid=np.array(['a', 'b', 'c']),
                         extra={'flag': np.array([1, 0, 1])})
        sub = vm.subset(np.array([0, 2]))
        np.testing.assert_array_equal(sub.extra['flag'], [1, 1])

    def test_subset_all_optionals(self):
        vm = VariantMeta(
            vid=np.array(['v0', 'v1', 'v2']),
            chrom=np.array([1, 1, 2]),
            pos_bp=np.array([100, 200, 300]),
            pos_cM=np.array([0.1, 0.2, 0.3]),
            af=np.array([0.1, 0.2, 0.3]),
            zero_allele=np.array(['A', 'C', 'G']),
            one_allele=np.array(['T', 'G', 'A']),
        )
        sub = vm.subset(np.array([1]))
        assert sub.m == 1
        np.testing.assert_array_equal(sub.chrom, [1])
        np.testing.assert_array_equal(sub.pos_bp, [200])
        np.testing.assert_array_equal(sub.af, [0.2])


class TestVariantMetaRepr:
    def test_repr_minimal(self):
        vm = VariantMeta(vid=np.array(['v0', 'v1']))
        r = repr(vm)
        assert 'VariantMeta' in r
        assert 'm=2' in r

    def test_repr_with_chrom(self):
        vm = VariantMeta(vid=np.array(['v0', 'v1']),
                         chrom=np.array([1, 2]))
        r = repr(vm)
        assert 'n_chrom=2' in r

    def test_repr_with_af(self):
        vm = VariantMeta(vid=np.array(['v0']), af=np.array([0.5]))
        r = repr(vm)
        assert 'af=True' in r


class TestVariantMetaExtraValidation:
    def test_extra_wrong_length(self):
        with pytest.raises(ValueError, match="extra"):
            VariantMeta(vid=np.array(['v0', 'v1']),
                        extra={'x': np.ones(3)})

    def test_extra_valid(self):
        vm = VariantMeta(vid=np.array(['v0', 'v1']),
                         extra={'x': np.array([1, 2])})
        np.testing.assert_array_equal(vm.extra['x'], [1, 2])
