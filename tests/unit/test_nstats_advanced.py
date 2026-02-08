"""
Unit tests for HasemanElstonEstimator, ParentOffspringRegression, and MatingStatistics.

Tests cover:
- HE estimator: basic estimation, zero variance, empty views, known correlation
- PO regression: basic estimation, perfect heritability, zero variance, no trios
- Mating stats: pair counts, offspring counts, spouse correlations, missing view
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, NPhenotypeArray
from xftsim.nfilter import TrioView, SibPairView
from xftsim.nstats import (
    HasemanElstonEstimator,
    ParentOffspringRegression,
    MatingStatistics,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pheno(n, **kwargs):
    """Quick NPhenotypeArray from keyword arrays."""
    sm = SampleMeta(iid=np.arange(n))
    return NPhenotypeArray(samples=sm, values=kwargs)


def _make_pheno_with_fid(n, fid, **kwargs):
    """NPhenotypeArray with explicit FIDs."""
    sm = SampleMeta(iid=np.arange(n), fid=fid)
    return NPhenotypeArray(samples=sm, values=kwargs)


def _make_sibpair_view(y1, y2, keys=None):
    """Construct a SibPairView from paired arrays."""
    if keys is None:
        keys = ['Y']
    n = len(y1)
    sib1 = {k: np.asarray(y1, dtype=np.float64) for k in keys}
    sib2 = {k: np.asarray(y2, dtype=np.float64) for k in keys}
    return SibPairView(
        sib1_phenotypes=sib1,
        sib2_phenotypes=sib2,
        n_pairs=n,
        sib1_idx=np.arange(n),
        sib2_idx=np.arange(n, 2 * n),
    )


def _make_trio_view(y_off, y_mom, y_dad, keys=None):
    """Construct a TrioView from offspring/mother/father arrays."""
    if keys is None:
        keys = ['Y']
    n = len(y_off)
    off_d = {k: np.asarray(y_off, dtype=np.float64) for k in keys}
    mom_d = {k: np.asarray(y_mom, dtype=np.float64) for k in keys}
    dad_d = {k: np.asarray(y_dad, dtype=np.float64) for k in keys}
    return TrioView(
        offspring_phenotypes=off_d,
        mother_phenotypes=mom_d,
        father_phenotypes=dad_d,
        n_trios=n,
    )


# ===========================================================================
# HasemanElstonEstimator
# ===========================================================================

class TestHasemanElstonEstimator:

    def test_returns_none_without_sibpair_filter(self):
        """Returns None when no SibPairView is in filtered_views."""
        he = HasemanElstonEstimator(filter_name='sibpair')
        result = he.estimate({0: _make_pheno(10, Y=np.ones(10))}, {}, 0)
        assert result is None

    def test_returns_none_for_empty_view(self):
        """Returns None when SibPairView has zero pairs."""
        he = HasemanElstonEstimator()
        view = SibPairView(
            sib1_phenotypes={'Y': np.array([])},
            sib2_phenotypes={'Y': np.array([])},
            n_pairs=0,
        )
        result = he.estimate({}, {'sibpair': view}, 0)
        assert result is None

    def test_returns_none_wrong_filter_type(self):
        """Returns None when the filter name maps to a TrioView."""
        he = HasemanElstonEstimator()
        trio = _make_trio_view(np.ones(5), np.ones(5), np.ones(5))
        result = he.estimate({}, {'sibpair': trio}, 0)
        assert result is None

    def test_known_perfect_correlation(self):
        """Identical sibs (r=1) should give h2=2 (perfect, capped by theory)."""
        rng = np.random.RandomState(42)
        y = rng.randn(200)
        view = _make_sibpair_view(y, y)
        he = HasemanElstonEstimator()
        result = he.estimate({}, {'sibpair': view}, 0)
        assert result is not None
        assert result['Y']['h2'] == pytest.approx(2.0, abs=0.01)
        assert result['Y']['sib_r'] == pytest.approx(1.0, abs=0.01)

    def test_known_zero_correlation(self):
        """Independent sibs should give h2 near 0."""
        rng = np.random.RandomState(42)
        y1 = rng.randn(5000)
        y2 = rng.randn(5000)
        view = _make_sibpair_view(y1, y2)
        he = HasemanElstonEstimator()
        result = he.estimate({}, {'sibpair': view}, 0)
        assert result is not None
        assert abs(result['Y']['h2']) < 0.1

    def test_known_moderate_correlation(self):
        """Sibs with known r=0.25 should give h2 near 0.5."""
        rng = np.random.RandomState(42)
        n = 10000
        shared = rng.randn(n)
        y1 = shared * 0.5 + rng.randn(n) * np.sqrt(0.75)
        y2 = shared * 0.5 + rng.randn(n) * np.sqrt(0.75)
        # True Cov(y1,y2) = 0.25, Var(y1) = Var(y2) = 1.0, r = 0.25
        view = _make_sibpair_view(y1, y2)
        he = HasemanElstonEstimator()
        result = he.estimate({}, {'sibpair': view}, 0)
        assert abs(result['Y']['h2'] - 0.5) < 0.1

    def test_constant_phenotype(self):
        """Constant sibling values should give sib_r=0 (zero variance)."""
        y = np.ones(50)
        view = _make_sibpair_view(y, y)
        he = HasemanElstonEstimator()
        result = he.estimate({}, {'sibpair': view}, 0)
        assert result['Y']['sib_r'] == 0.0

    def test_multiple_keys(self):
        """Should compute h2 for each phenotype key independently."""
        rng = np.random.RandomState(42)
        n = 1000
        view = SibPairView(
            sib1_phenotypes={'A': rng.randn(n), 'B': rng.randn(n)},
            sib2_phenotypes={'A': rng.randn(n), 'B': rng.randn(n)},
            n_pairs=n,
        )
        he = HasemanElstonEstimator()
        result = he.estimate({}, {'sibpair': view}, 0)
        assert 'A' in result
        assert 'B' in result
        assert 'h2' in result['A']
        assert 'h2' in result['B']

    def test_n_pairs_reported(self):
        """n_pairs should be reported correctly."""
        rng = np.random.RandomState(42)
        n = 77
        view = _make_sibpair_view(rng.randn(n), rng.randn(n))
        he = HasemanElstonEstimator()
        result = he.estimate({}, {'sibpair': view}, 0)
        assert result['Y']['n_pairs'] == 77

    def test_single_pair(self):
        """Single pair (n=1) should return nan (not enough data)."""
        view = _make_sibpair_view(np.array([1.0]), np.array([2.0]))
        he = HasemanElstonEstimator()
        result = he.estimate({}, {'sibpair': view}, 0)
        assert np.isnan(result['Y']['h2'])

    def test_custom_filter_name(self):
        """Custom filter name should be respected."""
        rng = np.random.RandomState(42)
        view = _make_sibpair_view(rng.randn(100), rng.randn(100))
        he = HasemanElstonEstimator(filter_name='my_sibs')
        # Wrong name → None
        result = he.estimate({}, {'sibpair': view}, 0)
        assert result is None
        # Right name → result
        result = he.estimate({}, {'my_sibs': view}, 0)
        assert result is not None


# ===========================================================================
# ParentOffspringRegression
# ===========================================================================

class TestParentOffspringRegression:

    def test_returns_none_without_trio_filter(self):
        """Returns None when no TrioView is available."""
        por = ParentOffspringRegression()
        result = por.estimate({0: _make_pheno(10, Y=np.ones(10))}, {}, 0)
        assert result is None

    def test_returns_none_for_empty_view(self):
        """Returns None when TrioView has zero trios."""
        por = ParentOffspringRegression()
        view = TrioView(
            offspring_phenotypes={'Y': np.array([])},
            mother_phenotypes={'Y': np.array([])},
            father_phenotypes={'Y': np.array([])},
            n_trios=0,
        )
        result = por.estimate({}, {'trio': view}, 0)
        assert result is None

    def test_returns_none_wrong_filter_type(self):
        """Returns None when filter maps to SibPairView instead."""
        por = ParentOffspringRegression()
        sib = _make_sibpair_view(np.ones(5), np.ones(5))
        result = por.estimate({}, {'trio': sib}, 0)
        assert result is None

    def test_perfect_heritability(self):
        """When offspring = midparent exactly, slope should be 1.0."""
        rng = np.random.RandomState(42)
        n = 500
        y_mom = rng.randn(n)
        y_dad = rng.randn(n)
        y_off = 0.5 * (y_mom + y_dad)  # perfect h2=1
        view = _make_trio_view(y_off, y_mom, y_dad)
        por = ParentOffspringRegression()
        result = por.estimate({}, {'trio': view}, 0)
        assert result['Y']['h2'] == pytest.approx(1.0, abs=1e-10)

    def test_zero_heritability(self):
        """When offspring is pure noise, slope should be near 0."""
        rng = np.random.RandomState(42)
        n = 5000
        y_mom = rng.randn(n)
        y_dad = rng.randn(n)
        y_off = rng.randn(n)  # no relation to parents
        view = _make_trio_view(y_off, y_mom, y_dad)
        por = ParentOffspringRegression()
        result = por.estimate({}, {'trio': view}, 0)
        assert abs(result['Y']['h2']) < 0.1

    def test_known_h2(self):
        """Offspring = h2 * midparent + noise should recover h2."""
        rng = np.random.RandomState(42)
        n = 10000
        h2_true = 0.6
        y_mom = rng.randn(n)
        y_dad = rng.randn(n)
        midparent = 0.5 * (y_mom + y_dad)
        y_off = h2_true * midparent + rng.randn(n) * np.sqrt(1 - h2_true * np.var(midparent))
        view = _make_trio_view(y_off, y_mom, y_dad)
        por = ParentOffspringRegression()
        result = por.estimate({}, {'trio': view}, 0)
        assert abs(result['Y']['h2'] - h2_true) < 0.05

    def test_se_is_positive(self):
        """Standard error should be positive for non-degenerate data."""
        rng = np.random.RandomState(42)
        n = 200
        y_mom = rng.randn(n)
        y_dad = rng.randn(n)
        y_off = 0.5 * (y_mom + y_dad) + rng.randn(n)
        view = _make_trio_view(y_off, y_mom, y_dad)
        por = ParentOffspringRegression()
        result = por.estimate({}, {'trio': view}, 0)
        assert result['Y']['se'] > 0

    def test_constant_midparent(self):
        """Constant midparent (zero variance) should give nan."""
        n = 50
        y_mom = np.ones(n) * 3
        y_dad = np.ones(n) * 3
        y_off = np.random.randn(n)
        view = _make_trio_view(y_off, y_mom, y_dad)
        por = ParentOffspringRegression()
        result = por.estimate({}, {'trio': view}, 0)
        assert np.isnan(result['Y']['h2'])

    def test_n_trios_reported(self):
        """n_trios should be reported correctly."""
        rng = np.random.RandomState(42)
        n = 123
        view = _make_trio_view(rng.randn(n), rng.randn(n), rng.randn(n))
        por = ParentOffspringRegression()
        result = por.estimate({}, {'trio': view}, 0)
        assert result['Y']['n_trios'] == 123

    def test_multiple_keys(self):
        """Should compute h2 for each phenotype key."""
        rng = np.random.RandomState(42)
        n = 200
        view = TrioView(
            offspring_phenotypes={'A': rng.randn(n), 'B': rng.randn(n)},
            mother_phenotypes={'A': rng.randn(n), 'B': rng.randn(n)},
            father_phenotypes={'A': rng.randn(n), 'B': rng.randn(n)},
            n_trios=n,
        )
        por = ParentOffspringRegression()
        result = por.estimate({}, {'trio': view}, 0)
        assert 'A' in result
        assert 'B' in result

    def test_key_in_offspring_but_not_parents(self):
        """Key only in offspring should be skipped."""
        rng = np.random.RandomState(42)
        n = 50
        view = TrioView(
            offspring_phenotypes={'A': rng.randn(n), 'B': rng.randn(n)},
            mother_phenotypes={'A': rng.randn(n)},
            father_phenotypes={'A': rng.randn(n)},
            n_trios=n,
        )
        por = ParentOffspringRegression()
        result = por.estimate({}, {'trio': view}, 0)
        assert 'A' in result
        assert 'B' not in result

    def test_single_trio(self):
        """Single trio (n=1) should return nan."""
        view = _make_trio_view(np.array([1.0]), np.array([2.0]), np.array([3.0]))
        por = ParentOffspringRegression()
        result = por.estimate({}, {'trio': view}, 0)
        assert np.isnan(result['Y']['h2'])

    def test_custom_filter_name(self):
        """Custom filter name should be respected."""
        rng = np.random.RandomState(42)
        view = _make_trio_view(rng.randn(50), rng.randn(50), rng.randn(50))
        por = ParentOffspringRegression(filter_name='my_trio')
        # Wrong name → None
        assert por.estimate({}, {'trio': view}, 0) is None
        # Right name → result
        assert por.estimate({}, {'my_trio': view}, 0) is not None


# ===========================================================================
# MatingStatistics
# ===========================================================================

class TestMatingStatistics:

    def test_returns_none_missing_generation(self):
        """Returns None when generation not in phenotype_history."""
        ms = MatingStatistics()
        result = ms.estimate({}, {}, 5)
        assert result is None

    def test_pair_count_from_fid(self):
        """n_mating_pairs should equal number of unique FIDs."""
        fid = np.array([0, 0, 1, 1, 2, 2])
        pheno = _make_pheno_with_fid(6, fid, Y=np.random.randn(6))
        ms = MatingStatistics()
        result = ms.estimate({0: pheno}, {}, 0)
        assert result['n_mating_pairs'] == 3

    def test_mean_offspring_count(self):
        """mean_offspring_count should reflect family sizes."""
        fid = np.array([0, 0, 0, 1, 1, 2])
        pheno = _make_pheno_with_fid(6, fid, Y=np.random.randn(6))
        ms = MatingStatistics()
        result = ms.estimate({0: pheno}, {}, 0)
        # Families of size 3, 2, 1 → mean = 2.0
        assert result['mean_offspring_count'] == pytest.approx(2.0)

    def test_uniform_offspring(self):
        """All pairs have same number of offspring."""
        fid = np.repeat(np.arange(50), 2)
        pheno = _make_pheno_with_fid(100, fid, Y=np.random.randn(100))
        ms = MatingStatistics()
        result = ms.estimate({0: pheno}, {}, 0)
        assert result['n_mating_pairs'] == 50
        assert result['mean_offspring_count'] == pytest.approx(2.0)

    def test_spouse_correlations_no_trio(self):
        """Without TrioView, spouse_correlations should be empty dict."""
        fid = np.repeat(np.arange(10), 2)
        pheno = _make_pheno_with_fid(20, fid, Y=np.random.randn(20))
        ms = MatingStatistics()
        result = ms.estimate({0: pheno}, {}, 0)
        assert result['spouse_correlations'] == {}

    def test_spouse_correlations_with_trio(self):
        """With TrioView, spouse correlations should be computed."""
        rng = np.random.RandomState(42)
        n = 200
        y_mom = rng.randn(n)
        y_dad = rng.randn(n)
        y_off = rng.randn(n)
        view = _make_trio_view(y_off, y_mom, y_dad)
        fid = np.repeat(np.arange(n), 1)
        pheno = _make_pheno_with_fid(n, fid, Y=y_off)
        ms = MatingStatistics()
        result = ms.estimate({0: pheno}, {'trio': view}, 0)
        assert 'Y' in result['spouse_correlations']

    def test_spouse_correlation_high(self):
        """Correlated spouses should produce high spouse correlation."""
        rng = np.random.RandomState(42)
        n = 500
        shared = rng.randn(n)
        y_mom = shared + rng.randn(n) * 0.1
        y_dad = shared + rng.randn(n) * 0.1
        y_off = rng.randn(n)
        view = _make_trio_view(y_off, y_mom, y_dad)
        fid = np.arange(n)
        pheno = _make_pheno_with_fid(n, fid, Y=y_off)
        ms = MatingStatistics()
        result = ms.estimate({0: pheno}, {'trio': view}, 0)
        assert result['spouse_correlations']['Y'] > 0.8

    def test_spouse_correlation_near_zero_random(self):
        """Independent spouses should produce near-zero correlation."""
        rng = np.random.RandomState(42)
        n = 2000
        y_mom = rng.randn(n)
        y_dad = rng.randn(n)
        y_off = rng.randn(n)
        view = _make_trio_view(y_off, y_mom, y_dad)
        fid = np.arange(n)
        pheno = _make_pheno_with_fid(n, fid, Y=y_off)
        ms = MatingStatistics()
        result = ms.estimate({0: pheno}, {'trio': view}, 0)
        assert abs(result['spouse_correlations']['Y']) < 0.1

    def test_custom_filter_name(self):
        """Custom filter name should be respected for spouse correlations."""
        rng = np.random.RandomState(42)
        n = 100
        view = _make_trio_view(rng.randn(n), rng.randn(n), rng.randn(n))
        fid = np.arange(n)
        pheno = _make_pheno_with_fid(n, fid, Y=rng.randn(n))
        ms = MatingStatistics(filter_name='my_trio')
        # Wrong name → empty correlations
        result = ms.estimate({0: pheno}, {'trio': view}, 0)
        assert result['spouse_correlations'] == {}
        # Right name → non-empty
        result = ms.estimate({0: pheno}, {'my_trio': view}, 0)
        assert 'Y' in result['spouse_correlations']

    def test_single_family(self):
        """Single family should report 1 mating pair."""
        pheno = _make_pheno_with_fid(5, np.zeros(5, dtype=int), Y=np.ones(5))
        ms = MatingStatistics()
        result = ms.estimate({0: pheno}, {}, 0)
        assert result['n_mating_pairs'] == 1
        assert result['mean_offspring_count'] == pytest.approx(5.0)

    def test_multiple_phenotype_keys_spouse_corr(self):
        """Spouse correlations computed independently for each phenotype key."""
        rng = np.random.RandomState(42)
        n = 300
        # Create TrioView with two keys: A (correlated spouses), B (independent)
        shared = rng.randn(n)
        mom_a = shared + 0.1 * rng.randn(n)
        dad_a = shared + 0.1 * rng.randn(n)
        mom_b = rng.randn(n)
        dad_b = rng.randn(n)
        off = rng.randn(n)

        view = TrioView(
            offspring_phenotypes={'A': off, 'B': off.copy()},
            mother_phenotypes={'A': mom_a, 'B': mom_b},
            father_phenotypes={'A': dad_a, 'B': dad_b},
            n_trios=n,
        )
        fid = np.arange(n)
        pheno = _make_pheno_with_fid(n, fid, A=off, B=off.copy())
        ms = MatingStatistics()
        result = ms.estimate({0: pheno}, {'trio': view}, 0)
        assert result['spouse_correlations']['A'] > 0.5
        assert abs(result['spouse_correlations']['B']) < 0.2
