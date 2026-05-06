"""
Unit tests for SampleStatistics and GenerationResult.

Tests:
1. SampleStatistics estimate with single key
2. SampleStatistics estimate with multiple keys
3. SampleStatistics returns dict with cov, var, keys
4. SampleStatistics with TrioView
5. SampleStatistics with SibPairView
6. GenerationResult structure
7. Statistics with constant values (zero variance)
8. Statistics covariance matrix symmetry
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, NPhenotypeArray
from xftsim.stats import SampleStatistics, GenerationResult
from xftsim.filters import TrioView, SibPairView


def _make_pheno(n, values_dict, generation=0):
    sm = SampleMeta(iid=np.arange(n), generation=generation)
    pheno = NPhenotypeArray(samples=sm)
    for k, v in values_dict.items():
        pheno._values[k] = np.asarray(v, dtype=np.float64)
    return pheno


class TestSampleStatisticsBasic:
    def test_single_key(self):
        """Statistics for single phenotype key returns cov, var, keys."""
        stat = SampleStatistics()
        pheno = _make_pheno(100, {'Y': np.random.RandomState(42).normal(0, 1, 100)})
        result = stat.estimate({0: pheno}, {}, 0)

        assert 'cov' in result
        assert 'var' in result
        assert 'keys' in result
        assert result['keys'] == ['Y']
        assert result['cov'].shape == (1, 1)
        assert result['var'].shape == (1,)

    def test_multiple_keys(self):
        """Statistics for multiple phenotype keys returns k×k covariance."""
        stat = SampleStatistics()
        rng = np.random.RandomState(42)
        pheno = _make_pheno(50, {
            'A': rng.normal(0, 1, 50),
            'B': rng.normal(5, 2, 50),
        })
        result = stat.estimate({0: pheno}, {}, 0)
        assert len(result['keys']) == 2
        assert result['cov'].shape == (2, 2)
        assert result['var'].shape == (2,)

    def test_variance_matches_np(self):
        """Variance should match np.cov diagonal."""
        stat = SampleStatistics()
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pheno = _make_pheno(5, {'Y': vals})
        result = stat.estimate({0: pheno}, {}, 0)
        expected_var = np.var(vals, ddof=1)  # np.cov uses ddof=1
        np.testing.assert_allclose(result['var'][0], expected_var)

    def test_covariance_symmetry(self):
        """Covariance matrix should be symmetric."""
        stat = SampleStatistics()
        rng = np.random.RandomState(42)
        pheno = _make_pheno(50, {
            'A': rng.normal(0, 1, 50),
            'B': rng.normal(0, 1, 50),
            'C': rng.normal(0, 1, 50),
        })
        result = stat.estimate({0: pheno}, {}, 0)
        np.testing.assert_allclose(result['cov'], result['cov'].T)


class TestSampleStatisticsEdgeCases:
    def test_constant_values(self):
        """Constant values → variance = 0."""
        stat = SampleStatistics()
        pheno = _make_pheno(10, {'Y': np.ones(10)})
        result = stat.estimate({0: pheno}, {}, 0)
        assert result['var'][0] == 0.0

    def test_missing_generation(self):
        """Missing generation → returns None."""
        stat = SampleStatistics()
        result = stat.estimate({0: _make_pheno(5, {'Y': np.zeros(5)})}, {}, 5)
        assert result is None

    def test_empty_keys(self):
        """Empty phenotype → empty covariance."""
        stat = SampleStatistics()
        pheno = _make_pheno(5, {})
        result = stat.estimate({0: pheno}, {}, 0)
        assert result['keys'] == []
        assert result['var'].shape == (0,)


class TestSampleStatisticsWithFilters:
    def test_with_trio_view(self):
        """Statistics with TrioView present does not crash."""
        stat = SampleStatistics()
        pheno = _make_pheno(10, {'Y': np.arange(10, dtype=float)})
        trio = TrioView(
            offspring_phenotypes={'Y': np.array([1.0, 2.0, 3.0])},
            mother_phenotypes={'Y': np.array([4.0, 5.0, 6.0])},
            father_phenotypes={'Y': np.array([7.0, 8.0, 9.0])},
            n_trios=3,
        )
        result = stat.estimate({0: pheno}, {'trio': trio}, 0)
        assert result is not None

    def test_with_sib_pair_view(self):
        """Statistics with SibPairView present does not crash."""
        stat = SampleStatistics()
        pheno = _make_pheno(10, {'Y': np.arange(10, dtype=float)})
        spv = SibPairView(
            sib1_phenotypes={'Y': np.array([1.0, 2.0])},
            sib2_phenotypes={'Y': np.array([3.0, 4.0])},
            n_pairs=2,
            sib1_idx=np.array([0, 2]),
            sib2_idx=np.array([1, 3]),
        )
        result = stat.estimate({0: pheno}, {'sib': spv}, 0)
        assert result is not None


class TestGenerationResult:
    def test_structure(self):
        gr = GenerationResult(generation=5, statistics={'SampleStatistics': {'cov': np.eye(1)}})
        assert gr.generation == 5
        assert 'SampleStatistics' in gr.statistics

    def test_empty_statistics(self):
        gr = GenerationResult(generation=0, statistics={})
        assert gr.generation == 0
        assert len(gr.statistics) == 0

    def test_multiple_statistics(self):
        gr = GenerationResult(
            generation=3,
            statistics={
                'SampleStatistics': {'cov': np.eye(2)},
                'SampleStatistics_1': {'cov': np.eye(2)},
            },
        )
        assert len(gr.statistics) == 2
