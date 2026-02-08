"""
Unit tests for SampleStatistics edge cases.

Tests:
1. Empty phenotype array (no keys)
2. Single phenotype → scalar covariance
3. Multiple phenotypes → covariance matrix properties
4. Missing generation → None result
5. Large k (many phenotypes)
6. GenerationResult dataclass
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, NPhenotypeArray
from xftsim.nstats import SampleStatistics, GenerationResult


def _make_pheno(n, **kwargs):
    sm = SampleMeta(iid=np.arange(n))
    return NPhenotypeArray(samples=sm, values=kwargs)


class TestSampleStatisticsEdgeCases:
    def test_empty_keys(self):
        """No phenotype keys → empty cov/var."""
        stat = SampleStatistics()
        pheno = _make_pheno(5)
        result = stat.estimate({0: pheno}, {}, 0)
        assert result['cov'].shape == (1, 0)
        assert result['var'].shape == (0,)
        assert result['keys'] == []

    def test_single_key_scalar_cov(self):
        """Single phenotype key → 1x1 covariance matrix."""
        stat = SampleStatistics()
        pheno = _make_pheno(100, x=np.random.randn(100))
        result = stat.estimate({0: pheno}, {}, 0)
        assert result['cov'].shape == (1, 1)
        assert result['var'].shape == (1,)
        assert len(result['keys']) == 1

    def test_two_keys_cov_shape(self):
        """Two phenotype keys → 2x2 covariance matrix."""
        stat = SampleStatistics()
        rng = np.random.RandomState(42)
        pheno = _make_pheno(100, a=rng.randn(100), b=rng.randn(100))
        result = stat.estimate({0: pheno}, {}, 0)
        assert result['cov'].shape == (2, 2)
        assert result['var'].shape == (2,)

    def test_cov_symmetric(self):
        """Covariance matrix should be symmetric."""
        stat = SampleStatistics()
        rng = np.random.RandomState(42)
        pheno = _make_pheno(200, x=rng.randn(200), y=rng.randn(200))
        result = stat.estimate({0: pheno}, {}, 0)
        np.testing.assert_allclose(result['cov'], result['cov'].T)

    def test_cov_diagonal_equals_var(self):
        """Diagonal of covariance should equal variance array."""
        stat = SampleStatistics()
        rng = np.random.RandomState(42)
        pheno = _make_pheno(200, a=rng.randn(200), b=rng.randn(200),
                           c=rng.randn(200))
        result = stat.estimate({0: pheno}, {}, 0)
        np.testing.assert_allclose(np.diag(result['cov']), result['var'])

    def test_missing_generation(self):
        """Generation not in history → None."""
        stat = SampleStatistics()
        result = stat.estimate({}, {}, 5)
        assert result is None

    def test_large_k(self):
        """Many phenotype keys should work."""
        stat = SampleStatistics()
        rng = np.random.RandomState(42)
        values = {f'Y{i}': rng.randn(50) for i in range(10)}
        pheno = _make_pheno(50, **values)
        result = stat.estimate({0: pheno}, {}, 0)
        assert result['cov'].shape == (10, 10)
        assert len(result['keys']) == 10

    def test_constant_phenotype(self):
        """Constant phenotype → zero variance."""
        stat = SampleStatistics()
        pheno = _make_pheno(50, x=np.ones(50))
        result = stat.estimate({0: pheno}, {}, 0)
        assert result['var'][0] == pytest.approx(0.0, abs=1e-15)


class TestGenerationResult:
    def test_creation(self):
        """GenerationResult should store generation and statistics."""
        gr = GenerationResult(generation=5, statistics={'mean': 1.0, 'var': 2.0})
        assert gr.generation == 5
        assert gr.statistics['mean'] == 1.0
        assert gr.statistics['var'] == 2.0

    def test_default_statistics(self):
        """Default statistics should be empty dict."""
        gr = GenerationResult(generation=0)
        assert gr.statistics == {}
