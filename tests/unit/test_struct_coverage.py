"""
Tests covering struct.py gaps for the active classes.

Targets uncovered lines in struct.py for:
- SampleMeta (extra validation, unique_identifier, with_generation, repr)
- VariantMeta (extra validation, __getitem__, subset, to_variant_index, repr)
- DenseHaplotypeArray (chrom, pos_bp, pos_cM, af properties, standardized methods,
  deprecation warnings, repr, HaplotypeArrayAccessor)
- PhenotypeArray (subset, repr)
- PedigreeArray (validation, bounds checking)
"""
import numpy as np
import pytest
import warnings

from xftsim.struct import (
    SampleMeta, VariantMeta, DenseHaplotypeArray, PhenotypeArray,
    PedigreeArray, HaplotypeArrayAccessor,
)


# ---------------------------------------------------------------------------
# SampleMeta
# ---------------------------------------------------------------------------

class TestSampleMetaCoverage:
    def test_extra_field_validated(self):
        sm = SampleMeta(iid=np.array([0, 1, 2]), extra={"batch": [10, 20, 30]})
        assert np.array_equal(sm.extra["batch"], [10, 20, 30])

    def test_extra_field_wrong_length_raises(self):
        with pytest.raises(ValueError, match="extra\\['batch'\\] has length"):
            SampleMeta(iid=np.array([0, 1, 2]), extra={"batch": [10, 20]})

    def test_unique_identifier(self):
        sm = SampleMeta(iid=np.array(["a", "b"]), fid=np.array(["f1", "f2"]),
                        generation=3)
        uid = sm.unique_identifier
        assert uid[0] == "3.a.f1"
        assert uid[1] == "3.b.f2"

    def test_with_generation(self):
        sm = SampleMeta(iid=np.arange(5), generation=0)
        sm2 = sm.with_generation(7)
        assert sm2.generation == 7
        assert np.array_equal(sm2.iid, sm.iid)

    def test_repr(self):
        sm = SampleMeta(iid=np.arange(10))
        r = repr(sm)
        assert "n=10" in r
        assert "SampleMeta" in r

    def test_to_sample_index(self):
        sm = SampleMeta(iid=np.array(["a", "b"]), generation=2)
        idx = sm.to_sample_index()
        assert idx.n == 2
        assert idx.generation == 2

    def test_subset_with_extra(self):
        sm = SampleMeta(
            iid=np.array([10, 20, 30, 40]),
            extra={"batch": np.array([1, 2, 3, 4])},
        )
        sub = sm.subset([0, 2])
        assert len(sub.iid) == 2
        assert np.array_equal(sub.extra["batch"], [1, 3])

    def test_default_sex_alternates(self):
        sm = SampleMeta(iid=np.arange(4))
        assert np.array_equal(sm.sex, [0, 1, 0, 1])

    def test_default_fid_equals_iid(self):
        sm = SampleMeta(iid=np.array(["x", "y"]))
        assert np.array_equal(sm.fid, ["x", "y"])


# ---------------------------------------------------------------------------
# VariantMeta
# ---------------------------------------------------------------------------

class TestVariantMetaCoverage:
    def test_extra_field(self):
        vm = VariantMeta(
            vid=np.array(["v1", "v2"]),
            extra={"coding": np.array([True, False])},
        )
        assert vm.extra["coding"][0] == True

    def test_extra_wrong_length_raises(self):
        with pytest.raises(ValueError, match="extra\\['flag'\\] has length"):
            VariantMeta(
                vid=np.array(["v1", "v2"]),
                extra={"flag": np.array([True])},
            )

    def test_getitem_core_field(self):
        vm = VariantMeta(vid=np.array(["v1"]), chrom=np.array(["1"]))
        assert vm["chrom"][0] == "1"

    def test_getitem_none_field_raises(self):
        vm = VariantMeta(vid=np.array(["v1"]))
        with pytest.raises(KeyError, match="is None"):
            vm["chrom"]

    def test_getitem_extra(self):
        vm = VariantMeta(
            vid=np.array(["v1"]),
            extra={"ann": np.array([99])},
        )
        assert vm["ann"][0] == 99

    def test_getitem_missing_extra_raises(self):
        vm = VariantMeta(vid=np.array(["v1"]))
        with pytest.raises(KeyError):
            vm["no_such_field"]

    def test_subset_with_all_fields(self):
        vm = VariantMeta(
            vid=np.array(["a", "b", "c"]),
            chrom=np.array([1, 2, 3]),
            pos_bp=np.array([100, 200, 300]),
            pos_cM=np.array([1.0, 2.0, 3.0]),
            af=np.array([0.1, 0.2, 0.3]),
            zero_allele=np.array(["A", "C", "G"]),
            one_allele=np.array(["T", "G", "C"]),
            extra={"ann": np.array([10, 20, 30])},
        )
        sub = vm.subset([0, 2])
        assert sub.m == 2
        assert np.array_equal(sub.vid, ["a", "c"])
        assert np.array_equal(sub.chrom, [1, 3])
        assert sub.extra["ann"][1] == 30

    def test_subset_none_fields(self):
        vm = VariantMeta(vid=np.array(["a", "b"]))
        sub = vm.subset([1])
        assert sub.m == 1
        assert sub.chrom is None

    def test_to_variant_index(self):
        vm = VariantMeta(
            vid=np.array(["v1", "v2"]),
            chrom=np.array([1, 2]),
            pos_bp=np.array([100, 200]),
            af=np.array([0.3, 0.7]),
        )
        idx = vm.to_variant_index()
        assert idx.m == 2

    def test_to_variant_index_custom_af(self):
        vm = VariantMeta(vid=np.array(["v1"]))
        custom_af = np.array([0.5])
        idx = vm.to_variant_index(af=custom_af)
        assert idx.m == 1

    def test_repr_with_chrom_and_af(self):
        vm = VariantMeta(
            vid=np.array(["v1", "v2"]),
            chrom=np.array([1, 2]),
            af=np.array([0.3, 0.7]),
        )
        r = repr(vm)
        assert "m=2" in r
        assert "n_chrom=2" in r
        assert "af=True" in r

    def test_repr_minimal(self):
        vm = VariantMeta(vid=np.array(["v1"]))
        r = repr(vm)
        assert "m=1" in r
        assert "af" not in r


# ---------------------------------------------------------------------------
# DenseHaplotypeArray property accessors
# ---------------------------------------------------------------------------

class TestDenseHaplotypeArrayCoverage:
    @pytest.fixture
    def hap_with_metadata(self):
        geno = np.random.RandomState(42).randint(0, 2, (10, 5, 2)).astype(np.int8)
        samples = SampleMeta(iid=np.arange(10))
        variants = VariantMeta(
            vid=np.arange(5),
            chrom=np.array([1, 1, 2, 2, 2]),
            pos_bp=np.array([100, 200, 300, 400, 500]),
            pos_cM=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            af=np.array([0.3, 0.4, 0.5, 0.6, 0.7]),
        )
        return DenseHaplotypeArray(genotypes=geno, samples=samples, variants=variants)

    def test_chrom_property(self, hap_with_metadata):
        assert len(hap_with_metadata.chrom) == 5

    def test_pos_bp_property(self, hap_with_metadata):
        assert hap_with_metadata.pos_bp[0] == 100

    def test_pos_cM_property(self, hap_with_metadata):
        assert hap_with_metadata.pos_cM[0] == pytest.approx(0.1)

    def test_af_property(self, hap_with_metadata):
        assert hap_with_metadata.af[0] == pytest.approx(0.3)

    def test_chrom_none(self):
        geno = np.zeros((5, 3, 2), dtype=np.int8)
        hap = DenseHaplotypeArray(genotypes=geno)
        assert hap.chrom is None

    def test_pos_bp_none(self):
        geno = np.zeros((5, 3, 2), dtype=np.int8)
        hap = DenseHaplotypeArray(genotypes=geno)
        assert hap.pos_bp is None

    def test_pos_cM_none(self):
        geno = np.zeros((5, 3, 2), dtype=np.int8)
        hap = DenseHaplotypeArray(genotypes=geno)
        assert hap.pos_cM is None

    def test_standardized_haploid_matvec(self, hap_with_metadata):
        v = np.ones(5)
        result = hap_with_metadata.standardized_haploid_matvec(v, haploid=0)
        assert result.shape == (10,)
        # Mean-centered, so sum should be close to 0
        assert abs(np.mean(result)) < 1.0

    def test_diploid_matvec(self, hap_with_metadata):
        v = np.ones(5)
        result = hap_with_metadata.diploid_matvec(v)
        assert result.shape == (10,)

    def test_data_interleaved(self, hap_with_metadata):
        data = hap_with_metadata.data
        assert data.shape == (10, 10)  # n x 2m

    def test_values_alias(self, hap_with_metadata):
        vals = hap_with_metadata.values
        data = hap_with_metadata.data
        assert np.array_equal(vals, data)

    def test_attrs(self, hap_with_metadata):
        a = hap_with_metadata.attrs
        assert a["generation"] == 0

    def test_shape_2d(self, hap_with_metadata):
        assert hap_with_metadata.shape == (10, 10)  # n x 2m

    def test_get_sample_indexer_deprecated(self, hap_with_metadata):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            idx = hap_with_metadata.get_sample_indexer()
            assert idx.n == 10

    def test_get_variant_indexer_deprecated(self, hap_with_metadata):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            idx = hap_with_metadata.get_variant_indexer()
            assert idx.m == 5

    def test_xft_accessor(self, hap_with_metadata):
        acc = hap_with_metadata.xft
        assert isinstance(acc, HaplotypeArrayAccessor)
        assert acc.n == 10
        assert acc.m == 5
        assert acc.generation == 0

    def test_xft_accessor_methods(self, hap_with_metadata):
        acc = hap_with_metadata.xft
        af = acc.af_empirical
        assert af.shape == (5,)
        diploid = acc.to_diploid()
        assert diploid.shape == (10, 5)
        std = acc.to_diploid_standardized()
        assert std.shape == (10, 5)

    def test_repr(self, hap_with_metadata):
        r = repr(hap_with_metadata)
        assert "DenseHaplotypeArray" in r
        assert "n=10" in r
        assert "m=5" in r

    def test_repr_with_families(self):
        samples = SampleMeta(
            iid=np.arange(6),
            fid=np.array([0, 0, 1, 1, 2, 2]),
        )
        geno = np.zeros((6, 3, 2), dtype=np.int8)
        hap = DenseHaplotypeArray(genotypes=geno, samples=samples)
        r = repr(hap)
        assert "n_fam=3" in r

    def test_to_diploid_standardized_with_scale(self, hap_with_metadata):
        G = hap_with_metadata.to_diploid_standardized(scale=True)
        assert G.shape == (10, 5)

    def test_drop_isel(self, hap_with_metadata):
        dropped = hap_with_metadata.drop_isel(sample=[0, 1])
        assert dropped.n == 8

    def test_drop_isel_variant(self, hap_with_metadata):
        dropped = hap_with_metadata.drop_isel(variant=[0])
        assert dropped.m == 4

    def test_getitem_tuple_1(self, hap_with_metadata):
        sub = hap_with_metadata[(slice(0, 3),)]
        assert sub.n == 3

    def test_getitem_tuple_2(self, hap_with_metadata):
        sub = hap_with_metadata[np.array([0, 1]), np.array([0, 1])]
        assert sub.n == 2
        assert sub.m == 2

    def test_getitem_tuple_too_many_raises(self, hap_with_metadata):
        with pytest.raises(IndexError, match="Too many indices"):
            hap_with_metadata[(0, 1, 2)]

    def test_generation_setter(self, hap_with_metadata):
        hap_with_metadata.generation = 5
        assert hap_with_metadata.generation == 5


# ---------------------------------------------------------------------------
# HaplotypeArray base class
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# DenseHaplotypeArray constructor validation
# ---------------------------------------------------------------------------

class TestDenseHaplotypeArrayValidation:
    def test_none_genotypes(self):
        hap = DenseHaplotypeArray(genotypes=None)
        assert hap.n == 0
        assert hap.m == 0

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="must be 3-D"):
            DenseHaplotypeArray(genotypes=np.zeros((5, 3)))

    def test_wrong_last_dim_raises(self):
        with pytest.raises(ValueError, match="must be 3-D"):
            DenseHaplotypeArray(genotypes=np.zeros((5, 3, 3)))

    def test_samples_n_mismatch_raises(self):
        samples = SampleMeta(iid=np.arange(3))
        with pytest.raises(ValueError, match="samples.n.*must match"):
            DenseHaplotypeArray(
                genotypes=np.zeros((5, 3, 2), dtype=np.int8),
                samples=samples,
            )

    def test_variants_m_mismatch_raises(self):
        variants = VariantMeta(vid=np.arange(10))
        with pytest.raises(ValueError, match="variants.m.*must match"):
            DenseHaplotypeArray(
                genotypes=np.zeros((5, 3, 2), dtype=np.int8),
                variants=variants,
            )

    def test_generation_mismatch_corrected(self):
        samples = SampleMeta(iid=np.arange(5), generation=0)
        hap = DenseHaplotypeArray(
            genotypes=np.zeros((5, 3, 2), dtype=np.int8),
            generation=2,
            samples=samples,
        )
        assert hap.samples.generation == 2

# ---------------------------------------------------------------------------
# PhenotypeArray
# ---------------------------------------------------------------------------

class TestNPhenotypeArrayCoverage:
    def test_subset(self):
        sm = SampleMeta(iid=np.arange(5))
        pheno = PhenotypeArray(samples=sm, values={"Y": np.arange(5, dtype=float)})
        sub = pheno.subset([1, 3])
        assert sub.samples.n == 2
        assert np.array_equal(sub["Y"], [1.0, 3.0])

    def test_repr(self):
        sm = SampleMeta(iid=np.arange(3))
        pheno = PhenotypeArray(samples=sm, values={"Y": np.zeros(3)})
        r = repr(pheno)
        assert "PhenotypeArray" in r
        assert "n=3" in r
        assert "'Y'" in r


# ---------------------------------------------------------------------------
# PedigreeArray
# ---------------------------------------------------------------------------

class TestPedigreeArrayCoverage:
    def test_maternal_idx_length_mismatch(self):
        sm = SampleMeta(iid=np.arange(4))
        with pytest.raises(ValueError, match="maternal_idx length"):
            PedigreeArray(
                offspring_samples=sm,
                maternal_idx=np.array([0, 1]),  # too short
                paternal_idx=np.array([0, 1, 0, 1]),
                parent_n=5,
            )

    def test_paternal_idx_length_mismatch(self):
        sm = SampleMeta(iid=np.arange(4))
        with pytest.raises(ValueError, match="paternal_idx length"):
            PedigreeArray(
                offspring_samples=sm,
                maternal_idx=np.array([0, 1, 0, 1]),
                paternal_idx=np.array([0, 1]),  # too short
                parent_n=5,
            )

    def test_maternal_idx_out_of_bounds(self):
        sm = SampleMeta(iid=np.arange(2))
        with pytest.raises(ValueError, match="maternal_idx out of bounds"):
            PedigreeArray(
                offspring_samples=sm,
                maternal_idx=np.array([0, 10]),  # 10 >= parent_n=5
                paternal_idx=np.array([0, 1]),
                parent_n=5,
            )

    def test_paternal_idx_out_of_bounds(self):
        sm = SampleMeta(iid=np.arange(2))
        with pytest.raises(ValueError, match="paternal_idx out of bounds"):
            PedigreeArray(
                offspring_samples=sm,
                maternal_idx=np.array([0, 1]),
                paternal_idx=np.array([0, 10]),  # 10 >= parent_n=5
                parent_n=5,
            )

    def test_empty_pedigree_ok(self):
        sm = SampleMeta(iid=np.array([], dtype=int))
        ped = PedigreeArray(
            offspring_samples=sm,
            maternal_idx=np.array([], dtype=int),
            paternal_idx=np.array([], dtype=int),
            parent_n=0,
        )
        assert ped.offspring_samples.n == 0

    def test_negative_maternal_idx(self):
        sm = SampleMeta(iid=np.arange(2))
        with pytest.raises(ValueError, match="maternal_idx out of bounds"):
            PedigreeArray(
                offspring_samples=sm,
                maternal_idx=np.array([-1, 0]),
                paternal_idx=np.array([0, 1]),
                parent_n=5,
            )
