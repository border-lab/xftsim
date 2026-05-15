"""
Tests for UnrelatedFilter, AscertainmentFilter, and SubsampleFilter.
"""
import numpy as np
import pytest

from xftsim.filters import (
    Filter, FilteredView,
    UnrelatedFilter, UnrelatedView,
    AscertainmentFilter, AscertainedView,
    SubsampleFilter, SubsampleView,
)
from xftsim.struct import SampleMeta, PhenotypeArray


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pheno(n, fids=None, seed=42, keys=('Y',)):
    """Build an PhenotypeArray with given FIDs and random phenotypes."""
    rng = np.random.RandomState(seed)
    iid = np.arange(n)
    if fids is None:
        fids = iid
    samples = SampleMeta(
        iid=iid,
        fid=np.asarray(fids),
        sex=np.tile([0, 1], (n + 1) // 2)[:n],
    )
    pheno = PhenotypeArray(samples=samples)
    for key in keys:
        pheno[key] = rng.randn(n)
    return pheno


def _history(pheno, gen=0):
    """Wrap a phenotype array as a single-generation history dict."""
    return {gen: pheno}


# ===========================================================================
# UnrelatedFilter tests
# ===========================================================================

class TestUnrelatedFilter:

    def test_returns_unrelated_view(self):
        pheno = _make_pheno(20, fids=np.repeat(np.arange(10), 2))
        view = UnrelatedFilter().apply(0, _history(pheno), {})
        assert isinstance(view, UnrelatedView)
        assert isinstance(view, FilteredView)

    def test_one_per_family(self):
        """Should select exactly one individual per unique FID."""
        n_fam = 15
        fids = np.repeat(np.arange(n_fam), 3)
        pheno = _make_pheno(n_fam * 3, fids=fids)
        view = UnrelatedFilter().apply(0, _history(pheno), {})
        assert len(view.indices) == n_fam
        # All selected FIDs should be unique
        selected_fids = pheno.samples.fid[view.indices]
        assert len(np.unique(selected_fids)) == n_fam

    def test_all_singletons(self):
        """When every individual is their own family, return all."""
        pheno = _make_pheno(10)  # default: fid == iid
        view = UnrelatedFilter().apply(0, _history(pheno), {})
        assert len(view.indices) == 10
        np.testing.assert_array_equal(view.indices, np.arange(10))

    def test_one_big_family(self):
        """One family of N -> exactly 1 individual selected."""
        pheno = _make_pheno(50, fids=np.zeros(50, dtype=int))
        view = UnrelatedFilter().apply(0, _history(pheno), {})
        assert len(view.indices) == 1

    def test_phenotypes_subset_matches_indices(self):
        """The subset phenotypes should match indexing the original."""
        fids = np.repeat(np.arange(5), 4)
        pheno = _make_pheno(20, fids=fids, keys=('Y', 'Y.G'))
        view = UnrelatedFilter().apply(0, _history(pheno), {})
        for key in ('Y', 'Y.G'):
            np.testing.assert_array_equal(
                view.phenotypes[key],
                pheno[key][view.indices],
            )

    def test_phenotype_samples_meta_correct(self):
        """Subset SampleMeta should have the right n and FIDs."""
        fids = np.repeat(np.arange(8), 2)
        pheno = _make_pheno(16, fids=fids)
        view = UnrelatedFilter().apply(0, _history(pheno), {})
        assert view.phenotypes.samples.n == 8
        assert len(np.unique(view.phenotypes.samples.fid)) == 8

    def test_missing_generation_returns_none(self):
        pheno = _make_pheno(10)
        view = UnrelatedFilter().apply(5, _history(pheno, gen=0), {})
        assert view is None

    def test_indices_are_sorted(self):
        """Returned indices should be in ascending order."""
        fids = np.repeat(np.arange(20), 2)
        pheno = _make_pheno(40, fids=fids)
        view = UnrelatedFilter().apply(0, _history(pheno), {})
        assert np.all(np.diff(view.indices) >= 0)

    def test_selects_first_occurrence(self):
        """Should select the first individual per FID."""
        fids = np.array([2, 2, 1, 1, 0, 0])
        pheno = _make_pheno(6, fids=fids)
        view = UnrelatedFilter().apply(0, _history(pheno), {})
        # np.unique return_index gives first occurrence of sorted unique values.
        # For fids=[2,2,1,1,0,0], unique sorted = [0,1,2], first idx = [4,2,0]
        expected = np.sort(np.array([4, 2, 0]))
        np.testing.assert_array_equal(view.indices, expected)

    def test_mixed_family_sizes(self):
        """Families of varying size: one per family."""
        fids = np.array([0, 0, 0, 1, 2, 2, 3, 3, 3, 3])
        pheno = _make_pheno(10, fids=fids)
        view = UnrelatedFilter().apply(0, _history(pheno), {})
        assert len(view.indices) == 4  # 4 unique FIDs


# ===========================================================================
# AscertainmentFilter tests
# ===========================================================================

class TestAscertainmentFilter:

    def test_returns_ascertained_view(self):
        pheno = _make_pheno(100)
        af = AscertainmentFilter('Y', quantile=0.1, tail='upper')
        view = af.apply(0, _history(pheno), {})
        assert isinstance(view, AscertainedView)
        assert isinstance(view, FilteredView)

    def test_upper_tail(self):
        """Upper tail should select individuals with high values."""
        rng = np.random.RandomState(123)
        vals = rng.randn(1000)
        pheno = _make_pheno(1000, seed=123)
        pheno._values['Y'] = vals

        af = AscertainmentFilter('Y', quantile=0.1, tail='upper')
        view = af.apply(0, _history(pheno), {})

        threshold = np.quantile(vals, 0.9)
        assert all(view.phenotypes['Y'] >= threshold)
        # Approximately 10% of 1000
        assert 80 <= len(view.indices) <= 120

    def test_lower_tail(self):
        """Lower tail should select individuals with low values."""
        rng = np.random.RandomState(456)
        vals = rng.randn(1000)
        pheno = _make_pheno(1000, seed=456)
        pheno._values['Y'] = vals

        af = AscertainmentFilter('Y', quantile=0.1, tail='lower')
        view = af.apply(0, _history(pheno), {})

        threshold = np.quantile(vals, 0.1)
        assert all(view.phenotypes['Y'] <= threshold)
        assert 80 <= len(view.indices) <= 120

    def test_both_tails(self):
        """Both tails should select from upper and lower."""
        rng = np.random.RandomState(789)
        vals = rng.randn(1000)
        pheno = _make_pheno(1000, seed=789)
        pheno._values['Y'] = vals

        af = AscertainmentFilter('Y', quantile=0.1, tail='both')
        view = af.apply(0, _history(pheno), {})

        lower_thresh = np.quantile(vals, 0.1)
        upper_thresh = np.quantile(vals, 0.9)
        for v in view.phenotypes['Y']:
            assert v <= lower_thresh or v >= upper_thresh
        # Approximately 20% of 1000
        assert 160 <= len(view.indices) <= 240

    def test_ascertainment_key_stored(self):
        pheno = _make_pheno(100)
        af = AscertainmentFilter('Y', quantile=0.2, tail='upper')
        view = af.apply(0, _history(pheno), {})
        assert view.ascertainment_key == 'Y'

    def test_threshold_stored_upper(self):
        """For upper tail, threshold should be the (1-q) percentile."""
        pheno = _make_pheno(1000, seed=11)
        af = AscertainmentFilter('Y', quantile=0.25, tail='upper')
        view = af.apply(0, _history(pheno), {})
        expected_thresh = np.quantile(pheno['Y'], 0.75)
        assert abs(view.threshold - expected_thresh) < 1e-10

    def test_threshold_stored_lower(self):
        """For lower tail, threshold should be the q percentile."""
        pheno = _make_pheno(1000, seed=22)
        af = AscertainmentFilter('Y', quantile=0.25, tail='lower')
        view = af.apply(0, _history(pheno), {})
        expected_thresh = np.quantile(pheno['Y'], 0.25)
        assert abs(view.threshold - expected_thresh) < 1e-10

    def test_threshold_stored_both(self):
        """For both tails, threshold stores the quantile value itself."""
        pheno = _make_pheno(1000, seed=33)
        af = AscertainmentFilter('Y', quantile=0.15, tail='both')
        view = af.apply(0, _history(pheno), {})
        assert abs(view.threshold - 0.15) < 1e-10

    def test_phenotypes_subset_matches_indices(self):
        pheno = _make_pheno(200, keys=('Y', 'Y.G'))
        af = AscertainmentFilter('Y', quantile=0.2, tail='upper')
        view = af.apply(0, _history(pheno), {})
        for key in ('Y', 'Y.G'):
            np.testing.assert_array_equal(
                view.phenotypes[key],
                pheno[key][view.indices],
            )

    def test_missing_generation_returns_none(self):
        pheno = _make_pheno(100)
        af = AscertainmentFilter('Y', quantile=0.1)
        view = af.apply(5, _history(pheno, gen=0), {})
        assert view is None

    def test_missing_key_returns_none(self):
        pheno = _make_pheno(100, keys=('Z',))
        af = AscertainmentFilter('Y', quantile=0.1)
        view = af.apply(0, _history(pheno), {})
        assert view is None

    def test_invalid_quantile_zero(self):
        with pytest.raises(ValueError, match="quantile must be in"):
            AscertainmentFilter('Y', quantile=0.0)

    def test_invalid_quantile_one(self):
        with pytest.raises(ValueError, match="quantile must be in"):
            AscertainmentFilter('Y', quantile=1.0)

    def test_invalid_quantile_negative(self):
        with pytest.raises(ValueError, match="quantile must be in"):
            AscertainmentFilter('Y', quantile=-0.1)

    def test_invalid_tail(self):
        with pytest.raises(ValueError, match="tail must be"):
            AscertainmentFilter('Y', quantile=0.1, tail='middle')

    def test_extreme_quantile_upper_nearly_all(self):
        """quantile=0.99 with upper should select ~99% of samples."""
        pheno = _make_pheno(1000, seed=44)
        af = AscertainmentFilter('Y', quantile=0.99, tail='upper')
        view = af.apply(0, _history(pheno), {})
        assert len(view.indices) >= 980

    def test_extreme_quantile_lower_nearly_all(self):
        """quantile=0.99 with lower should select ~99% of samples."""
        pheno = _make_pheno(1000, seed=55)
        af = AscertainmentFilter('Y', quantile=0.99, tail='lower')
        view = af.apply(0, _history(pheno), {})
        assert len(view.indices) >= 980

    def test_small_sample(self):
        """Ascertainment on very small sample should not crash."""
        pheno = _make_pheno(5, seed=66)
        af = AscertainmentFilter('Y', quantile=0.5, tail='upper')
        view = af.apply(0, _history(pheno), {})
        assert isinstance(view, AscertainedView)
        assert len(view.indices) > 0

    def test_ascertainment_on_specific_key(self):
        """Can ascertain on a key other than 'Y'."""
        pheno = _make_pheno(200, keys=('Y', 'Y.G', 'Y.E'), seed=77)
        af = AscertainmentFilter('Y.G', quantile=0.2, tail='upper')
        view = af.apply(0, _history(pheno), {})
        threshold = np.quantile(pheno['Y.G'], 0.8)
        assert all(view.phenotypes['Y.G'] >= threshold)

    def test_indices_into_original(self):
        """Indices should be valid into the original phenotype array."""
        pheno = _make_pheno(500, seed=88)
        af = AscertainmentFilter('Y', quantile=0.1, tail='both')
        view = af.apply(0, _history(pheno), {})
        assert np.all(view.indices >= 0)
        assert np.all(view.indices < 500)
        # Indices should be sorted (from np.where)
        assert np.all(np.diff(view.indices) > 0)

    def test_constant_values_selects_all(self):
        """If all values are identical, all pass any threshold."""
        n = 50
        pheno = _make_pheno(n, seed=99)
        pheno._values['Y'] = np.ones(n) * 3.14
        af = AscertainmentFilter('Y', quantile=0.1, tail='upper')
        view = af.apply(0, _history(pheno), {})
        # All values equal the quantile threshold, so all pass >= threshold
        assert len(view.indices) == n


# ===========================================================================
# SubsampleFilter tests
# ===========================================================================

class TestSubsampleFilter:

    def test_returns_subsample_view(self):
        pheno = _make_pheno(100)
        sf = SubsampleFilter(n=10, seed=42)
        view = sf.apply(0, _history(pheno), {})
        assert isinstance(view, SubsampleView)
        assert isinstance(view, FilteredView)

    def test_n_exact(self):
        """Requesting n=25 should produce exactly 25."""
        pheno = _make_pheno(100)
        sf = SubsampleFilter(n=25, seed=42)
        view = sf.apply(0, _history(pheno), {})
        assert len(view.indices) == 25
        assert view.n_subsample == 25

    def test_n_exceeds_population(self):
        """If n > population size, should return all."""
        pheno = _make_pheno(10)
        sf = SubsampleFilter(n=100, seed=42)
        view = sf.apply(0, _history(pheno), {})
        assert len(view.indices) == 10
        assert view.n_subsample == 10

    def test_fraction_mode(self):
        """fraction=0.5 of 100 should produce 50."""
        pheno = _make_pheno(100)
        sf = SubsampleFilter(fraction=0.5, seed=42)
        view = sf.apply(0, _history(pheno), {})
        assert view.n_subsample == 50
        assert len(view.indices) == 50

    def test_fraction_full(self):
        """fraction=1.0 should return all individuals."""
        pheno = _make_pheno(50)
        sf = SubsampleFilter(fraction=1.0, seed=42)
        view = sf.apply(0, _history(pheno), {})
        assert view.n_subsample == 50

    def test_fraction_small(self):
        """Very small fraction still produces at least 1."""
        pheno = _make_pheno(100)
        sf = SubsampleFilter(fraction=0.001, seed=42)
        view = sf.apply(0, _history(pheno), {})
        assert view.n_subsample >= 1
        assert len(view.indices) >= 1

    def test_phenotypes_subset_matches_indices(self):
        pheno = _make_pheno(200, keys=('Y', 'Y.G'))
        sf = SubsampleFilter(n=30, seed=42)
        view = sf.apply(0, _history(pheno), {})
        for key in ('Y', 'Y.G'):
            np.testing.assert_array_equal(
                view.phenotypes[key],
                pheno[key][view.indices],
            )

    def test_no_replacement(self):
        """All selected indices should be unique."""
        pheno = _make_pheno(100)
        sf = SubsampleFilter(n=50, seed=42)
        view = sf.apply(0, _history(pheno), {})
        assert len(np.unique(view.indices)) == 50

    def test_indices_sorted(self):
        pheno = _make_pheno(100)
        sf = SubsampleFilter(n=30, seed=42)
        view = sf.apply(0, _history(pheno), {})
        assert np.all(np.diff(view.indices) > 0)

    def test_seed_reproducibility(self):
        """Same seed should produce identical results."""
        pheno = _make_pheno(200)
        sf1 = SubsampleFilter(n=50, seed=99)
        sf2 = SubsampleFilter(n=50, seed=99)
        v1 = sf1.apply(0, _history(pheno), {})
        v2 = sf2.apply(0, _history(pheno), {})
        np.testing.assert_array_equal(v1.indices, v2.indices)

    def test_different_seeds_differ(self):
        """Different seeds should (almost certainly) differ."""
        pheno = _make_pheno(1000)
        sf1 = SubsampleFilter(n=100, seed=1)
        sf2 = SubsampleFilter(n=100, seed=2)
        v1 = sf1.apply(0, _history(pheno), {})
        v2 = sf2.apply(0, _history(pheno), {})
        assert not np.array_equal(v1.indices, v2.indices)

    def test_missing_generation_returns_none(self):
        pheno = _make_pheno(100)
        sf = SubsampleFilter(n=10, seed=42)
        view = sf.apply(5, _history(pheno, gen=0), {})
        assert view is None

    def test_samples_meta_correct(self):
        """Subset SampleMeta should reflect the subsample size."""
        pheno = _make_pheno(100)
        sf = SubsampleFilter(n=20, seed=42)
        view = sf.apply(0, _history(pheno), {})
        assert view.phenotypes.samples.n == 20

    def test_invalid_both_n_and_fraction(self):
        with pytest.raises(ValueError, match="exactly one"):
            SubsampleFilter(n=10, fraction=0.5)

    def test_invalid_neither_n_nor_fraction(self):
        with pytest.raises(ValueError, match="Must specify one"):
            SubsampleFilter()

    def test_invalid_n_zero(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            SubsampleFilter(n=0)

    def test_invalid_n_negative(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            SubsampleFilter(n=-5)

    def test_invalid_fraction_zero(self):
        with pytest.raises(ValueError, match="fraction must be in"):
            SubsampleFilter(fraction=0.0)

    def test_invalid_fraction_negative(self):
        with pytest.raises(ValueError, match="fraction must be in"):
            SubsampleFilter(fraction=-0.1)

    def test_invalid_fraction_greater_than_one(self):
        with pytest.raises(ValueError, match="fraction must be in"):
            SubsampleFilter(fraction=1.5)

    def test_no_seed_runs(self):
        """Without seed, filter should still work (non-deterministic)."""
        pheno = _make_pheno(100)
        sf = SubsampleFilter(n=10)
        view = sf.apply(0, _history(pheno), {})
        assert isinstance(view, SubsampleView)
        assert len(view.indices) == 10


# ===========================================================================
# Cross-filter composition tests
# ===========================================================================

class TestFilterComposition:
    """Test combining filters in sequence (manual composition)."""

    def test_unrelated_then_ascertained(self):
        """Select unrelated, then ascertain on the unrelated subset."""
        n_fam = 100
        fids = np.repeat(np.arange(n_fam), 3)
        pheno = _make_pheno(n_fam * 3, fids=fids, seed=42)

        # Step 1: unrelated
        uf = UnrelatedFilter()
        uview = uf.apply(0, _history(pheno), {})
        assert uview.phenotypes.samples.n == n_fam

        # Step 2: ascertain on the unrelated subset
        af = AscertainmentFilter('Y', quantile=0.1, tail='upper')
        unrel_history = {0: uview.phenotypes}
        aview = af.apply(0, unrel_history, {})
        assert isinstance(aview, AscertainedView)
        # Should be approximately 10% of 100 = 10
        assert 5 <= len(aview.indices) <= 15

    def test_subsample_then_ascertained(self):
        """Subsample, then ascertain on the subsample."""
        pheno = _make_pheno(1000, seed=42)

        sf = SubsampleFilter(n=200, seed=42)
        sview = sf.apply(0, _history(pheno), {})
        assert sview.n_subsample == 200

        af = AscertainmentFilter('Y', quantile=0.2, tail='both')
        sub_history = {0: sview.phenotypes}
        aview = af.apply(0, sub_history, {})
        # ~40% of 200 = ~80
        assert 60 <= len(aview.indices) <= 100

    def test_unrelated_then_subsample(self):
        """Unrelated filter then random subsample."""
        n_fam = 50
        fids = np.repeat(np.arange(n_fam), 4)
        pheno = _make_pheno(n_fam * 4, fids=fids, seed=42)

        uview = UnrelatedFilter().apply(0, _history(pheno), {})
        assert uview.phenotypes.samples.n == n_fam

        sf = SubsampleFilter(n=20, seed=7)
        sub_history = {0: uview.phenotypes}
        sview = sf.apply(0, sub_history, {})
        assert sview.n_subsample == 20


# ===========================================================================
# Edge case tests
# ===========================================================================

class TestEdgeCases:

    def test_single_individual_unrelated(self):
        """One individual should work for UnrelatedFilter."""
        pheno = _make_pheno(1)
        view = UnrelatedFilter().apply(0, _history(pheno), {})
        assert len(view.indices) == 1

    def test_single_individual_ascertainment(self):
        """One individual should work for AscertainmentFilter."""
        pheno = _make_pheno(1)
        af = AscertainmentFilter('Y', quantile=0.5, tail='upper')
        view = af.apply(0, _history(pheno), {})
        assert isinstance(view, AscertainedView)
        assert len(view.indices) == 1

    def test_single_individual_subsample(self):
        """One individual subsampled to 1 should work."""
        pheno = _make_pheno(1)
        sf = SubsampleFilter(n=1, seed=42)
        view = sf.apply(0, _history(pheno), {})
        assert len(view.indices) == 1

    def test_two_individuals_both_tails(self):
        """Two individuals, both tails q=0.5 should select both."""
        pheno = _make_pheno(2, seed=42)
        af = AscertainmentFilter('Y', quantile=0.5, tail='both')
        view = af.apply(0, _history(pheno), {})
        assert len(view.indices) == 2

    def test_multiple_keys_preserved(self):
        """All phenotype keys should be present in the filtered view."""
        keys = ('Y', 'Y.G', 'Y.E', 'X')
        pheno = _make_pheno(100, keys=keys, seed=42)
        for FilterCls, kwargs in [
            (UnrelatedFilter, {}),
            (AscertainmentFilter, {'phenotype_key': 'Y', 'quantile': 0.2, 'tail': 'upper'}),
            (SubsampleFilter, {'n': 30, 'seed': 42}),
        ]:
            f = FilterCls(**kwargs)
            view = f.apply(0, _history(pheno), {})
            for key in keys:
                assert key in view.phenotypes.keys, f"{FilterCls.__name__} missing key {key}"

    def test_empty_history(self):
        """All filters should return None for empty history."""
        for FilterCls, kwargs in [
            (UnrelatedFilter, {}),
            (AscertainmentFilter, {'phenotype_key': 'Y', 'quantile': 0.2}),
            (SubsampleFilter, {'n': 10, 'seed': 42}),
        ]:
            f = FilterCls(**kwargs)
            assert f.apply(0, {}, {}) is None

    def test_filter_abc_enforcement(self):
        """Cannot instantiate Filter directly."""
        with pytest.raises(TypeError):
            Filter()

    def test_all_views_are_filtered_view(self):
        """All view types should inherit from FilteredView."""
        assert issubclass(UnrelatedView, FilteredView)
        assert issubclass(AscertainedView, FilteredView)
        assert issubclass(SubsampleView, FilteredView)

    def test_all_filters_are_filter(self):
        """All filter types should inherit from Filter."""
        assert issubclass(UnrelatedFilter, Filter)
        assert issubclass(AscertainmentFilter, Filter)
        assert issubclass(SubsampleFilter, Filter)

    def test_string_fids(self):
        """FIDs can be strings -- UnrelatedFilter should still work."""
        n = 12
        fids = np.array(['fam_A'] * 4 + ['fam_B'] * 4 + ['fam_C'] * 4)
        samples = SampleMeta(
            iid=np.arange(n),
            fid=fids,
            sex=np.tile([0, 1], 6),
        )
        pheno = PhenotypeArray(samples=samples)
        pheno['Y'] = np.random.RandomState(42).randn(n)

        view = UnrelatedFilter().apply(0, {0: pheno}, {})
        assert len(view.indices) == 3
        selected_fids = set(pheno.samples.fid[view.indices])
        assert selected_fids == {'fam_A', 'fam_B', 'fam_C'}
