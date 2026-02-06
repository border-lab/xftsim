"""
Unit tests for SampleMeta, VariantMeta, NPhenotypeArray, PedigreeArray.
"""
import numpy as np
import pytest
import warnings
from xftsim.struct import SampleMeta, VariantMeta, NPhenotypeArray, PedigreeArray


# ── SampleMeta ──────────────────────────────────────────────────────────────

class TestSampleMeta:
    def test_construction_all_fields(self):
        sm = SampleMeta(
            iid=np.array([10, 20, 30]),
            fid=np.array([1, 1, 2]),
            sex=np.array([0, 1, 0]),
            generation=3,
        )
        assert sm.n == 3
        assert sm.generation == 3
        assert sm.n_fam == 2
        assert sm.n_female == 2
        assert sm.n_male == 1

    def test_default_fid(self):
        sm = SampleMeta(iid=np.array([0, 1, 2]))
        np.testing.assert_array_equal(sm.fid, sm.iid)

    def test_default_sex(self):
        sm = SampleMeta(iid=np.arange(5))
        np.testing.assert_array_equal(sm.sex, [0, 1, 0, 1, 0])

    def test_extra_field_access(self):
        sm = SampleMeta(
            iid=np.arange(3),
            extra={'batch': np.array([1, 1, 2]), 'ancestry': np.array([0.1, 0.2, 0.3])},
        )
        np.testing.assert_array_equal(sm.extra['batch'], [1, 1, 2])
        assert sm.extra['ancestry'][2] == pytest.approx(0.3)

    def test_extra_length_validation(self):
        with pytest.raises(ValueError, match="extra.*length"):
            SampleMeta(iid=np.arange(3), extra={'bad': np.array([1, 2])})

    def test_subset_preserves_alignment(self):
        sm = SampleMeta(
            iid=np.array([10, 20, 30, 40]),
            fid=np.array([1, 1, 2, 2]),
            sex=np.array([0, 1, 0, 1]),
            generation=2,
            extra={'batch': np.array([1, 2, 3, 4])},
        )
        idx = np.array([False, True, False, True])
        sub = sm.subset(idx)
        assert sub.n == 2
        np.testing.assert_array_equal(sub.iid, [20, 40])
        np.testing.assert_array_equal(sub.fid, [1, 2])
        np.testing.assert_array_equal(sub.sex, [1, 1])
        np.testing.assert_array_equal(sub.extra['batch'], [2, 4])
        assert sub.generation == 2

    def test_with_generation(self):
        sm = SampleMeta(iid=np.arange(3), generation=0, extra={'x': np.ones(3)})
        sm2 = sm.with_generation(5)
        assert sm2.generation == 5
        np.testing.assert_array_equal(sm2.iid, sm.iid)
        assert 'x' in sm2.extra


# ── VariantMeta ─────────────────────────────────────────────────────────────

class TestVariantMeta:
    def test_construction(self):
        vm = VariantMeta(vid=np.arange(5), chrom=np.array([1, 1, 2, 2, 2]))
        assert vm.m == 5

    def test_bracket_access_core(self):
        vm = VariantMeta(vid=np.array([10, 20, 30]))
        np.testing.assert_array_equal(vm['vid'], [10, 20, 30])

    def test_bracket_access_extras(self):
        vm = VariantMeta(
            vid=np.arange(3),
            extra={'coding': np.array([True, False, True])},
        )
        np.testing.assert_array_equal(vm['coding'], [True, False, True])

    def test_bracket_access_none_field(self):
        vm = VariantMeta(vid=np.arange(3))
        with pytest.raises(KeyError):
            vm['chrom']

    def test_bracket_access_missing_extra(self):
        vm = VariantMeta(vid=np.arange(3))
        with pytest.raises(KeyError):
            vm['nonexistent']

    def test_extra_length_validation(self):
        with pytest.raises(ValueError, match="extra.*length"):
            VariantMeta(vid=np.arange(3), extra={'bad': np.array([1, 2])})

    def test_subset_preserves_extras(self):
        vm = VariantMeta(
            vid=np.arange(4),
            chrom=np.array([1, 1, 2, 2]),
            extra={'ld_score': np.array([0.1, 0.2, 0.3, 0.4])},
        )
        sub = vm.subset(np.array([1, 3]))
        assert sub.m == 2
        np.testing.assert_array_equal(sub.vid, [1, 3])
        np.testing.assert_array_equal(sub.chrom, [1, 2])
        np.testing.assert_array_almost_equal(sub.extra['ld_score'], [0.2, 0.4])


# ── NPhenotypeArray ────────────────────────────────────────────────────────

class TestNPhenotypeArray:
    def test_get_set(self):
        sm = SampleMeta(iid=np.arange(5))
        pa = NPhenotypeArray(samples=sm)
        pa['height.G'] = np.ones(5) * 3.0
        np.testing.assert_array_equal(pa['height.G'], np.ones(5) * 3.0)

    def test_contains(self):
        sm = SampleMeta(iid=np.arange(5))
        pa = NPhenotypeArray(samples=sm)
        pa['x'] = np.zeros(5)
        assert 'x' in pa
        assert 'y' not in pa

    def test_keys(self):
        sm = SampleMeta(iid=np.arange(3))
        pa = NPhenotypeArray(samples=sm, values={
            'a': np.zeros(3),
            'b': np.ones(3),
        })
        assert set(pa.keys) == {'a', 'b'}

    def test_shape_validation(self):
        sm = SampleMeta(iid=np.arange(5))
        pa = NPhenotypeArray(samples=sm)
        with pytest.raises(ValueError, match="shape"):
            pa['bad'] = np.zeros(3)

    def test_overwrite_warning(self):
        sm = SampleMeta(iid=np.arange(5))
        pa = NPhenotypeArray(samples=sm)
        pa['x'] = np.zeros(5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pa['x'] = np.ones(5)
            assert len(w) == 1
            assert "Overwriting" in str(w[0].message)

    def test_subset(self):
        sm = SampleMeta(iid=np.arange(5))
        pa = NPhenotypeArray(samples=sm, values={
            'x': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        })
        sub = pa.subset(np.array([0, 2, 4]))
        assert sub.samples.n == 3
        np.testing.assert_array_equal(sub['x'], [1.0, 3.0, 5.0])

    def test_samples_travel_with_data(self):
        sm = SampleMeta(iid=np.array([10, 20, 30]))
        pa = NPhenotypeArray(samples=sm)
        pa['x'] = np.zeros(3)
        sub = pa.subset(np.array([1, 2]))
        np.testing.assert_array_equal(sub.samples.iid, [20, 30])


# ── PedigreeArray ───────────────────────────────────────────────────────────

class TestPedigreeArray:
    def test_valid_construction(self):
        offspring = SampleMeta(iid=np.arange(4))
        ped = PedigreeArray(
            offspring_samples=offspring,
            maternal_idx=np.array([0, 0, 1, 1]),
            paternal_idx=np.array([2, 2, 3, 3]),
            parent_n=5,
        )
        assert len(ped.maternal_idx) == 4
        assert len(ped.paternal_idx) == 4

    def test_length_mismatch_maternal(self):
        offspring = SampleMeta(iid=np.arange(4))
        with pytest.raises(ValueError, match="maternal_idx length"):
            PedigreeArray(
                offspring_samples=offspring,
                maternal_idx=np.array([0, 0]),
                paternal_idx=np.array([2, 2, 3, 3]),
                parent_n=5,
            )

    def test_length_mismatch_paternal(self):
        offspring = SampleMeta(iid=np.arange(4))
        with pytest.raises(ValueError, match="paternal_idx length"):
            PedigreeArray(
                offspring_samples=offspring,
                maternal_idx=np.array([0, 0, 1, 1]),
                paternal_idx=np.array([2, 2]),
                parent_n=5,
            )

    def test_bounds_maternal(self):
        offspring = SampleMeta(iid=np.arange(2))
        with pytest.raises(ValueError, match="maternal_idx out of bounds"):
            PedigreeArray(
                offspring_samples=offspring,
                maternal_idx=np.array([0, 10]),
                paternal_idx=np.array([1, 1]),
                parent_n=5,
            )

    def test_bounds_paternal(self):
        offspring = SampleMeta(iid=np.arange(2))
        with pytest.raises(ValueError, match="paternal_idx out of bounds"):
            PedigreeArray(
                offspring_samples=offspring,
                maternal_idx=np.array([0, 0]),
                paternal_idx=np.array([1, 5]),
                parent_n=5,
            )

    def test_empty_pedigree(self):
        offspring = SampleMeta(iid=np.array([]))
        ped = PedigreeArray(
            offspring_samples=offspring,
            maternal_idx=np.array([], dtype=np.intp),
            paternal_idx=np.array([], dtype=np.intp),
            parent_n=10,
        )
        assert len(ped.maternal_idx) == 0
