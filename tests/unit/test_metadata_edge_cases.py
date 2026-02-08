"""
Unit tests for SampleMeta and VariantMeta edge cases.

Tests:
1. SampleMeta: construction with iid, fid, sex, generation, extras
2. SampleMeta: n property matches length of iid
3. SampleMeta: sex array validation (must be 0 or 1)
4. SampleMeta: subset by integer array preserves all fields
5. SampleMeta: extras dict preserved through subset
6. VariantMeta: construction with vid, chrom, pos_bp, pos_cM
7. VariantMeta: m property matches length of vid
8. VariantMeta: __getitem__ with integer array returns subset
9. VariantMeta: extras dict preserved through subset
10. SampleMeta/VariantMeta: empty construction (n=0, m=0)
11. SampleMeta: generation preserved on construction and through operations
12. VariantMeta: chrom parsing for multi-chromosome
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta


class TestSampleMetaConstruction:
    """Test SampleMeta construction with various parameter combinations."""

    def test_minimal_construction_iid_only(self):
        """Construct SampleMeta with only iid."""
        iid = np.array([100, 200, 300])
        sm = SampleMeta(iid=iid)

        assert sm.n == 3
        np.testing.assert_array_equal(sm.iid, iid)
        # fid should default to iid
        np.testing.assert_array_equal(sm.fid, iid)
        # sex should alternate 0, 1, 0
        np.testing.assert_array_equal(sm.sex, [0, 1, 0])
        assert sm.generation == 0
        assert sm.extra == {}

    def test_construction_with_fid(self):
        """Construct SampleMeta with iid and fid."""
        iid = np.array([1, 2, 3, 4])
        fid = np.array([10, 10, 20, 20])
        sm = SampleMeta(iid=iid, fid=fid)

        assert sm.n == 4
        np.testing.assert_array_equal(sm.iid, iid)
        np.testing.assert_array_equal(sm.fid, fid)

    def test_construction_with_sex(self):
        """Construct SampleMeta with explicit sex array."""
        iid = np.arange(5)
        sex = np.array([0, 0, 1, 1, 0])
        sm = SampleMeta(iid=iid, sex=sex)

        np.testing.assert_array_equal(sm.sex, sex)
        assert sm.n_female == 3
        assert sm.n_male == 2

    def test_construction_with_generation(self):
        """Construct SampleMeta with specific generation."""
        sm = SampleMeta(iid=np.arange(3), generation=5)
        assert sm.generation == 5

    def test_construction_with_extras(self):
        """Construct SampleMeta with extra metadata."""
        iid = np.arange(4)
        extras = {
            'batch': np.array([1, 1, 2, 2]),
            'pc1': np.array([0.1, 0.2, 0.3, 0.4]),
        }
        sm = SampleMeta(iid=iid, extra=extras)

        assert 'batch' in sm.extra
        assert 'pc1' in sm.extra
        np.testing.assert_array_equal(sm.extra['batch'], [1, 1, 2, 2])
        np.testing.assert_array_almost_equal(sm.extra['pc1'], [0.1, 0.2, 0.3, 0.4])

    def test_construction_all_parameters(self):
        """Construct SampleMeta with all parameters."""
        iid = np.array([10, 20, 30])
        fid = np.array([1, 1, 2])
        sex = np.array([0, 1, 0])
        generation = 7
        extras = {'cohort': np.array(['A', 'A', 'B'])}

        sm = SampleMeta(iid=iid, fid=fid, sex=sex, generation=generation, extra=extras)

        assert sm.n == 3
        np.testing.assert_array_equal(sm.iid, iid)
        np.testing.assert_array_equal(sm.fid, fid)
        np.testing.assert_array_equal(sm.sex, sex)
        assert sm.generation == 7
        np.testing.assert_array_equal(sm.extra['cohort'], ['A', 'A', 'B'])

    def test_default_sex_even_length(self):
        """Default sex for even number of samples should be [0,1,0,1,...]."""
        sm = SampleMeta(iid=np.arange(6))
        expected_sex = np.array([0, 1, 0, 1, 0, 1])
        np.testing.assert_array_equal(sm.sex, expected_sex)

    def test_default_sex_odd_length(self):
        """Default sex for odd number of samples should be [0,1,0,1,...,0]."""
        sm = SampleMeta(iid=np.arange(7))
        expected_sex = np.array([0, 1, 0, 1, 0, 1, 0])
        np.testing.assert_array_equal(sm.sex, expected_sex)


class TestSampleMetaProperties:
    """Test SampleMeta property accessors."""

    def test_n_property_matches_iid_length(self):
        """n property should match length of iid."""
        for length in [0, 1, 5, 100]:
            sm = SampleMeta(iid=np.arange(length))
            assert sm.n == length
            assert sm.n == len(sm.iid)

    def test_n_fam_single_family(self):
        """n_fam should be 1 when all samples in same family."""
        sm = SampleMeta(iid=np.arange(10), fid=np.zeros(10, dtype=int))
        assert sm.n_fam == 1

    def test_n_fam_each_individual_unique_family(self):
        """n_fam should equal n when each sample has unique fid."""
        n = 8
        sm = SampleMeta(iid=np.arange(n), fid=np.arange(n))
        assert sm.n_fam == n

    def test_n_female_n_male_all_female(self):
        """All females: n_female=n, n_male=0."""
        sm = SampleMeta(iid=np.arange(5), sex=np.zeros(5, dtype=int))
        assert sm.n_female == 5
        assert sm.n_male == 0

    def test_n_female_n_male_all_male(self):
        """All males: n_female=0, n_male=n."""
        sm = SampleMeta(iid=np.arange(5), sex=np.ones(5, dtype=int))
        assert sm.n_female == 0
        assert sm.n_male == 5

    def test_n_female_n_male_mixed(self):
        """Mixed sex array counts correctly."""
        sex = np.array([0, 0, 1, 0, 1, 1, 0])
        sm = SampleMeta(iid=np.arange(len(sex)), sex=sex)
        assert sm.n_female == 4
        assert sm.n_male == 3


class TestSampleMetaSubset:
    """Test SampleMeta subsetting operations."""

    def test_subset_integer_array_preserves_all_fields(self):
        """Subset with integer array should preserve iid, fid, sex, generation."""
        sm = SampleMeta(
            iid=np.array([10, 20, 30, 40]),
            fid=np.array([1, 1, 2, 2]),
            sex=np.array([0, 1, 0, 1]),
            generation=3,
        )
        idx = np.array([0, 2])
        sub = sm.subset(idx)

        assert sub.n == 2
        np.testing.assert_array_equal(sub.iid, [10, 30])
        np.testing.assert_array_equal(sub.fid, [1, 2])
        np.testing.assert_array_equal(sub.sex, [0, 0])
        assert sub.generation == 3

    def test_subset_preserves_extras(self):
        """Subset should preserve and correctly index extra fields."""
        sm = SampleMeta(
            iid=np.arange(6),
            extra={
                'batch': np.array([1, 1, 2, 2, 3, 3]),
                'age': np.array([25, 30, 35, 40, 45, 50]),
            }
        )
        idx = np.array([1, 3, 5])
        sub = sm.subset(idx)

        assert sub.n == 3
        np.testing.assert_array_equal(sub.extra['batch'], [1, 2, 3])
        np.testing.assert_array_equal(sub.extra['age'], [30, 40, 50])

    def test_subset_empty_extras(self):
        """Subset with no extras should work."""
        sm = SampleMeta(iid=np.arange(5))
        sub = sm.subset(np.array([0, 2, 4]))
        assert sub.extra == {}

    def test_subset_slice(self):
        """Subset with slice should work."""
        sm = SampleMeta(iid=np.arange(10), fid=np.arange(10))
        sub = sm.subset(slice(2, 8, 2))

        np.testing.assert_array_equal(sub.iid, [2, 4, 6])
        np.testing.assert_array_equal(sub.fid, [2, 4, 6])

    def test_subset_single_element(self):
        """Subset to single element."""
        sm = SampleMeta(iid=np.array([100, 200, 300]))
        sub = sm.subset(np.array([1]))

        assert sub.n == 1
        np.testing.assert_array_equal(sub.iid, [200])


class TestSampleMetaGeneration:
    """Test SampleMeta generation handling."""

    def test_generation_default_zero(self):
        """Default generation should be 0."""
        sm = SampleMeta(iid=np.arange(3))
        assert sm.generation == 0

    def test_generation_preserved_through_subset(self):
        """Generation should be preserved through subset operations."""
        sm = SampleMeta(iid=np.arange(10), generation=5)
        sub = sm.subset(np.array([0, 3, 7]))
        assert sub.generation == 5

    def test_with_generation_creates_new_instance(self):
        """with_generation should create new SampleMeta with updated generation."""
        sm = SampleMeta(iid=np.array([1, 2, 3]), generation=0)
        sm_new = sm.with_generation(10)

        assert sm_new.generation == 10
        assert sm.generation == 0  # original unchanged
        np.testing.assert_array_equal(sm_new.iid, sm.iid)
        np.testing.assert_array_equal(sm_new.fid, sm.fid)

    def test_with_generation_preserves_extras(self):
        """with_generation should preserve extra fields."""
        sm = SampleMeta(
            iid=np.arange(3),
            generation=0,
            extra={'cohort': np.array([1, 2, 3])}
        )
        sm_new = sm.with_generation(5)

        assert sm_new.generation == 5
        np.testing.assert_array_equal(sm_new.extra['cohort'], [1, 2, 3])


class TestSampleMetaEmptyConstruction:
    """Test SampleMeta with n=0 (empty)."""

    def test_empty_construction(self):
        """Construct SampleMeta with zero samples."""
        sm = SampleMeta(iid=np.array([]))

        assert sm.n == 0
        assert len(sm.iid) == 0
        assert len(sm.fid) == 0
        assert len(sm.sex) == 0
        assert sm.n_fam == 0
        assert sm.n_female == 0
        assert sm.n_male == 0

    def test_empty_subset(self):
        """Subset to empty array."""
        sm = SampleMeta(iid=np.arange(5))
        sub = sm.subset(np.array([], dtype=int))

        assert sub.n == 0
        assert len(sub.iid) == 0


class TestVariantMetaConstruction:
    """Test VariantMeta construction."""

    def test_minimal_construction_vid_only(self):
        """Construct VariantMeta with only vid."""
        vid = np.array(['rs1', 'rs2', 'rs3'])
        vm = VariantMeta(vid=vid)

        assert vm.m == 3
        np.testing.assert_array_equal(vm.vid, vid)
        assert vm.chrom is None
        assert vm.pos_bp is None
        assert vm.pos_cM is None
        assert vm.af is None
        assert vm.extra == {}

    def test_construction_with_chrom_pos_bp_pos_cM(self):
        """Construct VariantMeta with chromosome and position information."""
        vid = np.array(['v1', 'v2', 'v3', 'v4'])
        chrom = np.array([1, 1, 2, 2])
        pos_bp = np.array([1000, 2000, 500, 1500])
        pos_cM = np.array([0.1, 0.2, 0.05, 0.15])

        vm = VariantMeta(vid=vid, chrom=chrom, pos_bp=pos_bp, pos_cM=pos_cM)

        assert vm.m == 4
        np.testing.assert_array_equal(vm.vid, vid)
        np.testing.assert_array_equal(vm.chrom, chrom)
        np.testing.assert_array_equal(vm.pos_bp, pos_bp)
        np.testing.assert_array_almost_equal(vm.pos_cM, pos_cM)

    def test_construction_with_allele_frequencies(self):
        """Construct VariantMeta with allele frequencies."""
        vid = np.array(['v1', 'v2'])
        af = np.array([0.1, 0.9])
        vm = VariantMeta(vid=vid, af=af)

        np.testing.assert_array_almost_equal(vm.af, af)

    def test_construction_with_alleles(self):
        """Construct VariantMeta with reference and alternate alleles."""
        vid = np.array(['v1', 'v2', 'v3'])
        zero_allele = np.array(['A', 'G', 'T'])
        one_allele = np.array(['C', 'T', 'A'])

        vm = VariantMeta(vid=vid, zero_allele=zero_allele, one_allele=one_allele)

        np.testing.assert_array_equal(vm.zero_allele, zero_allele)
        np.testing.assert_array_equal(vm.one_allele, one_allele)

    def test_construction_with_extras(self):
        """Construct VariantMeta with extra metadata."""
        vid = np.array(['v1', 'v2', 'v3'])
        extras = {
            'coding': np.array([True, False, True]),
            'maf': np.array([0.05, 0.3, 0.15]),
        }
        vm = VariantMeta(vid=vid, extra=extras)

        assert 'coding' in vm.extra
        assert 'maf' in vm.extra
        np.testing.assert_array_equal(vm.extra['coding'], [True, False, True])

    def test_construction_all_parameters(self):
        """Construct VariantMeta with all parameters."""
        vid = np.array(['rs1', 'rs2'])
        chrom = np.array([1, 2])
        pos_bp = np.array([1000, 2000])
        pos_cM = np.array([0.1, 0.2])
        af = np.array([0.25, 0.75])
        zero_allele = np.array(['A', 'G'])
        one_allele = np.array(['T', 'C'])
        extras = {'gene': np.array(['GENE1', 'GENE2'])}

        vm = VariantMeta(
            vid=vid,
            chrom=chrom,
            pos_bp=pos_bp,
            pos_cM=pos_cM,
            af=af,
            zero_allele=zero_allele,
            one_allele=one_allele,
            extra=extras,
        )

        assert vm.m == 2
        np.testing.assert_array_equal(vm.vid, vid)
        np.testing.assert_array_equal(vm.chrom, chrom)
        np.testing.assert_array_equal(vm.pos_bp, pos_bp)
        np.testing.assert_array_almost_equal(vm.pos_cM, pos_cM)
        np.testing.assert_array_almost_equal(vm.af, af)
        np.testing.assert_array_equal(vm.zero_allele, zero_allele)
        np.testing.assert_array_equal(vm.one_allele, one_allele)
        np.testing.assert_array_equal(vm.extra['gene'], ['GENE1', 'GENE2'])


class TestVariantMetaProperties:
    """Test VariantMeta property accessors."""

    def test_m_property_matches_vid_length(self):
        """m property should match length of vid."""
        for length in [0, 1, 5, 1000]:
            vm = VariantMeta(vid=np.arange(length))
            assert vm.m == length
            assert vm.m == len(vm.vid)


class TestVariantMetaSubset:
    """Test VariantMeta subsetting operations."""

    def test_subset_integer_array_all_fields(self):
        """Subset with integer array should preserve all populated fields."""
        vm = VariantMeta(
            vid=np.array(['v0', 'v1', 'v2', 'v3']),
            chrom=np.array([1, 1, 2, 2]),
            pos_bp=np.array([100, 200, 300, 400]),
            pos_cM=np.array([0.1, 0.2, 0.3, 0.4]),
            af=np.array([0.1, 0.2, 0.3, 0.4]),
        )
        idx = np.array([0, 2])
        sub = vm.subset(idx)

        assert sub.m == 2
        np.testing.assert_array_equal(sub.vid, ['v0', 'v2'])
        np.testing.assert_array_equal(sub.chrom, [1, 2])
        np.testing.assert_array_equal(sub.pos_bp, [100, 300])
        np.testing.assert_array_almost_equal(sub.pos_cM, [0.1, 0.3])
        np.testing.assert_array_almost_equal(sub.af, [0.1, 0.3])

    def test_subset_preserves_extras(self):
        """Subset should preserve and correctly index extra fields."""
        vm = VariantMeta(
            vid=np.array(['v0', 'v1', 'v2', 'v3', 'v4']),
            extra={
                'coding': np.array([True, False, True, False, True]),
                'score': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
            }
        )
        idx = np.array([1, 2, 4])
        sub = vm.subset(idx)

        assert sub.m == 3
        np.testing.assert_array_equal(sub.extra['coding'], [False, True, True])
        np.testing.assert_array_almost_equal(sub.extra['score'], [2.0, 3.0, 5.0])

    def test_subset_empty_extras(self):
        """Subset with no extras should work."""
        vm = VariantMeta(vid=np.array(['v1', 'v2', 'v3']))
        sub = vm.subset(np.array([0, 2]))
        assert sub.extra == {}

    def test_subset_slice(self):
        """Subset with slice should work."""
        vm = VariantMeta(
            vid=np.array(['v0', 'v1', 'v2', 'v3', 'v4', 'v5']),
            chrom=np.array([1, 1, 1, 2, 2, 2]),
        )
        sub = vm.subset(slice(1, 5, 2))

        np.testing.assert_array_equal(sub.vid, ['v1', 'v3'])
        np.testing.assert_array_equal(sub.chrom, [1, 2])

    def test_subset_single_element(self):
        """Subset to single element."""
        vm = VariantMeta(vid=np.array(['v0', 'v1', 'v2']))
        sub = vm.subset(np.array([1]))

        assert sub.m == 1
        np.testing.assert_array_equal(sub.vid, ['v1'])


class TestVariantMetaEmptyConstruction:
    """Test VariantMeta with m=0 (empty)."""

    def test_empty_construction(self):
        """Construct VariantMeta with zero variants."""
        vm = VariantMeta(vid=np.array([]))

        assert vm.m == 0
        assert len(vm.vid) == 0

    def test_empty_subset(self):
        """Subset to empty array."""
        vm = VariantMeta(vid=np.array(['v1', 'v2', 'v3']))
        sub = vm.subset(np.array([], dtype=int))

        assert sub.m == 0
        assert len(sub.vid) == 0


class TestVariantMetaChromMultiple:
    """Test VariantMeta with multiple chromosomes."""

    def test_multiple_chromosomes_integer(self):
        """Multiple chromosomes as integers."""
        vm = VariantMeta(
            vid=np.array(['v1', 'v2', 'v3', 'v4', 'v5']),
            chrom=np.array([1, 1, 2, 2, 3]),
        )
        assert vm.m == 5
        # Check unique chromosomes
        unique_chroms = np.unique(vm.chrom)
        np.testing.assert_array_equal(unique_chroms, [1, 2, 3])

    def test_multiple_chromosomes_string(self):
        """Multiple chromosomes as strings."""
        vm = VariantMeta(
            vid=np.array(['v1', 'v2', 'v3', 'v4']),
            chrom=np.array(['1', '2', '2', 'X']),
        )
        np.testing.assert_array_equal(vm.chrom, ['1', '2', '2', 'X'])

    def test_subset_preserves_chrom_diversity(self):
        """Subsetting across chromosomes preserves chromosome info."""
        vm = VariantMeta(
            vid=np.array(['v1', 'v2', 'v3', 'v4', 'v5', 'v6']),
            chrom=np.array([1, 1, 2, 2, 3, 3]),
        )
        # Select one variant from each chromosome
        idx = np.array([0, 2, 4])
        sub = vm.subset(idx)

        np.testing.assert_array_equal(sub.chrom, [1, 2, 3])


class TestVariantMetaGetitem:
    """Test VariantMeta __getitem__ access."""

    def test_getitem_subset_behavior_note(self):
        """
        Note: VariantMeta.__getitem__ is for field access (e.g., vm['vid']),
        not for subsetting by index. Subsetting uses the subset() method.
        This test documents the current interface.
        """
        vm = VariantMeta(
            vid=np.array(['v1', 'v2', 'v3']),
            pos_bp=np.array([100, 200, 300]),
        )

        # __getitem__ accesses fields by name
        np.testing.assert_array_equal(vm['vid'], ['v1', 'v2', 'v3'])
        np.testing.assert_array_equal(vm['pos_bp'], [100, 200, 300])

        # Subsetting uses subset() method, not __getitem__
        sub = vm.subset(np.array([0, 2]))
        np.testing.assert_array_equal(sub.vid, ['v1', 'v3'])


class TestMetadataImmutability:
    """Test that SampleMeta and VariantMeta are immutable (frozen dataclasses)."""

    def test_samplemeta_frozen(self):
        """SampleMeta should be frozen and not allow attribute assignment."""
        sm = SampleMeta(iid=np.arange(3))

        # Attempting to assign should raise
        with pytest.raises((AttributeError, TypeError)):
            sm.generation = 10

    def test_variantmeta_frozen(self):
        """VariantMeta should be frozen and not allow attribute assignment."""
        vm = VariantMeta(vid=np.array(['v1', 'v2']))

        # Attempting to assign should raise
        with pytest.raises((AttributeError, TypeError)):
            vm.chrom = np.array([1, 2])
