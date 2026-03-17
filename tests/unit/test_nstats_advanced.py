"""
Unit tests for HasemanElstonEstimator, ParentOffspringRegression, and MatingStatistics.

Tests cover:
- HE estimator: GRM-based h2 estimation, requires haplotype_history
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

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


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


def _make_he_sim_data(n=500, m=100, h2=0.5, seed=42):
    """Create haplotypes + phenotypes for HE unit tests."""
    from xftsim.neffect import AdditiveEffects
    from xftsim.narch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent

    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
    eff = AdditiveEffects.from_h2(h2=h2, m=m, seed=seed + 1)
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(eff))
    arch.add('Y.E', NoiseComponent(variance=1.0 - h2))
    arch.add('Y', AggregationComponent('Y.G + Y.E'))

    rng = np.random.RandomState(seed)
    pheno = arch.compute(hap, rng=rng, phenotype_history={}, pedigree_history={}, generation=0)

    return hap, pheno


# ===========================================================================
# HasemanElstonEstimator
# ===========================================================================

class TestHasemanElstonEstimator:

    def test_returns_none_without_haplotypes(self):
        """Returns None when no haplotype_history is passed."""
        he = HasemanElstonEstimator(phenotype_keys=['Y'])
        result = he.estimate({0: _make_pheno(10, Y=np.ones(10))}, {}, 0)
        assert result is None

    def test_returns_none_missing_generation(self):
        """Returns None when generation not in haplotype_history."""
        he = HasemanElstonEstimator(phenotype_keys=['Y'])
        hap, pheno = _make_he_sim_data(n=100, m=20)
        result = he.estimate(
            {0: pheno}, {}, 5,
            haplotype_history={0: hap},
        )
        assert result is None

    def test_recovers_h2_moderate(self):
        """GRM-based HE should recover h2≈0.5."""
        hap, pheno = _make_he_sim_data(n=2000, m=200, h2=0.5, seed=42)
        he = HasemanElstonEstimator(phenotype_keys=['Y'])
        result = he.estimate(
            {0: pheno}, {}, 0,
            haplotype_history={0: hap},
        )
        assert result is not None
        assert 'Y' in result
        assert abs(result['Y']['h2'] - 0.5) < 0.15, \
            f"HE h2={result['Y']['h2']:.3f}, expected ~0.5"

    def test_recovers_h2_high(self):
        """HE should recover h2≈0.8."""
        hap, pheno = _make_he_sim_data(n=2000, m=200, h2=0.8, seed=42)
        he = HasemanElstonEstimator(phenotype_keys=['Y'])
        result = he.estimate(
            {0: pheno}, {}, 0,
            haplotype_history={0: hap},
        )
        assert abs(result['Y']['h2'] - 0.8) < 0.15, \
            f"HE h2={result['Y']['h2']:.3f}, expected ~0.8"

    def test_recovers_h2_low(self):
        """HE should recover h2≈0.1."""
        hap, pheno = _make_he_sim_data(n=2000, m=200, h2=0.1, seed=42)
        he = HasemanElstonEstimator(phenotype_keys=['Y'])
        result = he.estimate(
            {0: pheno}, {}, 0,
            haplotype_history={0: hap},
        )
        assert abs(result['Y']['h2'] - 0.1) < 0.15, \
            f"HE h2={result['Y']['h2']:.3f}, expected ~0.1"

    def test_ordering_across_h2(self):
        """Higher true h2 should give higher estimated h2."""
        estimates = {}
        for h2 in [0.2, 0.7]:
            hap, pheno = _make_he_sim_data(n=2000, m=200, h2=h2, seed=42)
            he = HasemanElstonEstimator(phenotype_keys=['Y'])
            result = he.estimate({0: pheno}, {}, 0, haplotype_history={0: hap})
            estimates[h2] = result['Y']['h2']
        assert estimates[0.7] > estimates[0.2]

    def test_auto_selects_toplevel_keys(self):
        """With phenotype_keys=None, should select keys without dots."""
        hap, pheno = _make_he_sim_data(n=500, m=50, h2=0.5, seed=42)
        he = HasemanElstonEstimator()  # phenotype_keys=None
        result = he.estimate({0: pheno}, {}, 0, haplotype_history={0: hap})
        assert result is not None
        assert 'Y' in result
        # Subcomponents like Y.G, Y.E should be excluded
        assert 'Y.G' not in result
        assert 'Y.E' not in result

    def test_reports_n(self):
        """n should be reported in results."""
        hap, pheno = _make_he_sim_data(n=500, m=50, h2=0.5, seed=42)
        he = HasemanElstonEstimator(phenotype_keys=['Y'])
        result = he.estimate({0: pheno}, {}, 0, haplotype_history={0: hap})
        assert result['Y']['n'] == 500

    def test_stores_cov_g_matrix(self):
        """Should store the full genetic covariance matrix."""
        hap, pheno = _make_he_sim_data(n=500, m=50, h2=0.5, seed=42)
        he = HasemanElstonEstimator(phenotype_keys=['Y'])
        result = he.estimate({0: pheno}, {}, 0, haplotype_history={0: hap})
        assert '_cov_g' in result
        assert result['_cov_g'].shape == (1, 1)


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
        sib = SibPairView(
            sib1_phenotypes={'Y': np.ones(5)},
            sib2_phenotypes={'Y': np.ones(5)},
            n_pairs=5,
        )
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
