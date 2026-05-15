"""
Unit tests for SampleMeta, VariantMeta, PhenotypeArray, PedigreeArray.
"""
import numpy as np
import pytest
import warnings
from xftsim.struct import SampleMeta, VariantMeta, PhenotypeArray, PedigreeArray


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


# ── PhenotypeArray ────────────────────────────────────────────────────────

class TestNPhenotypeArray:
    def test_get_set(self):
        sm = SampleMeta(iid=np.arange(5))
        pa = PhenotypeArray(samples=sm)
        pa['height.G'] = np.ones(5) * 3.0
        np.testing.assert_array_equal(pa['height.G'], np.ones(5) * 3.0)

    def test_contains(self):
        sm = SampleMeta(iid=np.arange(5))
        pa = PhenotypeArray(samples=sm)
        pa['x'] = np.zeros(5)
        assert 'x' in pa
        assert 'y' not in pa

    def test_keys(self):
        sm = SampleMeta(iid=np.arange(3))
        pa = PhenotypeArray(samples=sm, values={
            'a': np.zeros(3),
            'b': np.ones(3),
        })
        assert set(pa.keys) == {'a', 'b'}

    def test_shape_validation(self):
        sm = SampleMeta(iid=np.arange(5))
        pa = PhenotypeArray(samples=sm)
        with pytest.raises(ValueError, match="shape"):
            pa['bad'] = np.zeros(3)

    def test_overwrite_warning(self):
        sm = SampleMeta(iid=np.arange(5))
        pa = PhenotypeArray(samples=sm)
        pa['x'] = np.zeros(5)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pa['x'] = np.ones(5)
            assert len(w) == 1
            assert "Overwriting" in str(w[0].message)

    def test_subset(self):
        sm = SampleMeta(iid=np.arange(5))
        pa = PhenotypeArray(samples=sm, values={
            'x': np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        })
        sub = pa.subset(np.array([0, 2, 4]))
        assert sub.samples.n == 3
        np.testing.assert_array_equal(sub['x'], [1.0, 3.0, 5.0])

    def test_samples_travel_with_data(self):
        sm = SampleMeta(iid=np.array([10, 20, 30]))
        pa = PhenotypeArray(samples=sm)
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


# ── Additional edge case tests ─────────────────────────────────────────────

class TestSampleMetaEdgeCases:
    """Edge case tests for SampleMeta."""

    def test_single_individual(self):
        sm = SampleMeta(iid=np.array([0]))
        assert sm.n == 1
        assert sm.n_fam == 1

    def test_odd_n_sex_default(self):
        """Odd n should still get alternating sex with correct length."""
        sm = SampleMeta(iid=np.arange(7))
        assert len(sm.sex) == 7
        assert sm.sex[0] == 0 and sm.sex[1] == 1

    def test_all_same_fid(self):
        """All individuals in same family."""
        sm = SampleMeta(iid=np.arange(5), fid=np.zeros(5, dtype=np.int64))
        assert sm.n_fam == 1

    def test_large_extras(self):
        """Many extra fields should work."""
        n = 10
        extras = {f'pc{i}': np.random.randn(n) for i in range(10)}
        sm = SampleMeta(iid=np.arange(n), extra=extras)
        assert len(sm.extra) == 10
        for k, v in sm.extra.items():
            assert len(v) == n

    def test_subset_integer_array(self):
        """Subset with integer array indices."""
        sm = SampleMeta(iid=np.arange(10), fid=np.repeat(np.arange(5), 2))
        sub = sm.subset(np.array([0, 2, 4]))
        assert sub.n == 3
        np.testing.assert_array_equal(sub.iid, [0, 2, 4])

    def test_repr(self):
        sm = SampleMeta(iid=np.arange(10))
        r = repr(sm)
        assert 'SampleMeta' in r
        assert 'n=10' in r


class TestVariantMetaEdgeCases:
    """Edge case tests for VariantMeta."""

    def test_single_variant(self):
        vm = VariantMeta(vid=np.array([0]))
        assert vm.m == 1

    def test_all_optional_fields(self):
        """Construct with all optional fields."""
        vm = VariantMeta(
            vid=np.arange(3),
            chrom=np.array([1, 1, 2]),
            pos_bp=np.array([100, 200, 300]),
            pos_cM=np.array([0.1, 0.2, 0.3]),
            af=np.array([0.1, 0.5, 0.9]),
            zero_allele=np.array(['A', 'C', 'G']),
            one_allele=np.array(['T', 'A', 'C']),
        )
        assert vm.m == 3
        np.testing.assert_array_equal(vm['chrom'], [1, 1, 2])
        np.testing.assert_array_equal(vm['pos_bp'], [100, 200, 300])

    def test_subset_preserves_af(self):
        vm = VariantMeta(vid=np.arange(5), af=np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
        sub = vm.subset(np.array([1, 3]))
        np.testing.assert_array_equal(sub.af, [0.2, 0.4])

    def test_repr(self):
        vm = VariantMeta(vid=np.arange(10), chrom=np.array([1]*5 + [2]*5))
        r = repr(vm)
        assert 'VariantMeta' in r
        assert 'm=10' in r


class TestNPhenotypeArrayEdgeCases:
    """Edge case tests for PhenotypeArray."""

    def test_initial_values(self):
        """Passing values dict at construction should populate immediately."""
        sm = SampleMeta(iid=np.arange(3))
        vals = {'x': np.array([1.0, 2.0, 3.0]), 'y': np.array([4.0, 5.0, 6.0])}
        pa = PhenotypeArray(samples=sm, values=vals)
        assert 'x' in pa
        assert 'y' in pa
        np.testing.assert_array_equal(pa['x'], [1.0, 2.0, 3.0])

    def test_missing_key_raises(self):
        sm = SampleMeta(iid=np.arange(3))
        pa = PhenotypeArray(samples=sm)
        with pytest.raises(KeyError):
            pa['nonexistent']

    def test_wrong_dtype_coerced(self):
        """Integer values should be coerced to float64."""
        sm = SampleMeta(iid=np.arange(3))
        pa = PhenotypeArray(samples=sm)
        pa['x'] = np.array([1, 2, 3])  # int
        assert pa['x'].dtype == np.float64

    def test_repr(self):
        sm = SampleMeta(iid=np.arange(3))
        pa = PhenotypeArray(samples=sm, values={'a': np.zeros(3)})
        r = repr(pa)
        assert 'PhenotypeArray' in r
        assert 'n=3' in r

    def test_empty_keys(self):
        sm = SampleMeta(iid=np.arange(5))
        pa = PhenotypeArray(samples=sm)
        assert len(list(pa.keys)) == 0


class TestPedigreeArrayEdgeCases:
    """Edge case tests for PedigreeArray."""

    def test_negative_maternal_idx(self):
        offspring = SampleMeta(iid=np.arange(2))
        with pytest.raises(ValueError, match="maternal_idx out of bounds"):
            PedigreeArray(
                offspring_samples=offspring,
                maternal_idx=np.array([-1, 0]),
                paternal_idx=np.array([1, 1]),
                parent_n=5,
            )

    def test_large_parent_n(self):
        """parent_n can be larger than actual indices — that's fine."""
        offspring = SampleMeta(iid=np.arange(2))
        ped = PedigreeArray(
            offspring_samples=offspring,
            maternal_idx=np.array([0, 1]),
            paternal_idx=np.array([2, 3]),
            parent_n=1000,
        )
        assert ped.parent_n == 1000

    def test_same_parent_multiple_offspring(self):
        """Same parent can appear for multiple offspring."""
        offspring = SampleMeta(iid=np.arange(4))
        ped = PedigreeArray(
            offspring_samples=offspring,
            maternal_idx=np.array([0, 0, 0, 0]),
            paternal_idx=np.array([1, 1, 1, 1]),
            parent_n=2,
        )
        assert len(ped.maternal_idx) == 4
        assert np.all(ped.maternal_idx == 0)
