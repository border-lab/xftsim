"""
Unit tests for NoiseComponent and CNoiseComponent with grouping.

Tests:
1. Ungrouped noise: all values independent
2. Grouped noise: same value within group (FID grouping)
3. CNoiseComponent ungrouped: k outputs
4. CNoiseComponent grouped: same multivariate value within group
5. Grouped noise variance matches spec
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray, NPhenotypeArray
from xftsim.arch import NoiseComponent, CNoiseComponent, ArchNode


def _make_hap(n=12, m=3, fids=None, generation=0):
    if fids is None:
        fids = np.repeat(np.arange(n // 2), 2)
    sm = SampleMeta(iid=np.arange(n), fid=fids, generation=generation)
    vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
    geno = np.zeros((n, m, 2), dtype=np.int8)
    return DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm, generation=generation)


class TestNoiseGrouped:
    def test_ungrouped_noise_independent(self):
        """Without grouping, all values should be independent draws."""
        comp = NoiseComponent(variance=1.0)
        node = ArchNode(outputs=['Y.E'], component=comp, inputs=[], grouping=None)
        hap = _make_hap()
        pheno = NPhenotypeArray(samples=hap.samples)
        rng = np.random.RandomState(42)
        result = comp.compute(node, hap, pheno, rng=rng, generation=0, pedigree_history={})
        assert result.shape == (12,)
        # Very unlikely all 12 are the same
        assert len(np.unique(result)) > 1

    def test_grouped_noise_fid(self):
        """With FID grouping, siblings in same family get same noise."""
        comp = NoiseComponent(variance=1.0)
        node = ArchNode(outputs=['Y.E'], component=comp, inputs=[], grouping='FID')
        fids = np.array([0, 0, 1, 1, 2, 2])
        hap = _make_hap(n=6, fids=fids)
        pheno = NPhenotypeArray(samples=hap.samples)
        rng = np.random.RandomState(42)
        result = comp.compute(node, hap, pheno, rng=rng, generation=0, pedigree_history={})
        # Siblings should have the same noise value
        assert result[0] == result[1]  # family 0
        assert result[2] == result[3]  # family 1
        assert result[4] == result[5]  # family 2
        # Different families should differ
        assert result[0] != result[2]

    def test_grouped_noise_variance(self):
        """Grouped noise should have approximately the specified variance across groups."""
        comp = NoiseComponent(variance=2.0)
        node = ArchNode(outputs=['Y.E'], component=comp, inputs=[], grouping='FID')
        n = 200
        fids = np.repeat(np.arange(100), 2)
        hap = _make_hap(n=n, fids=fids)
        pheno = NPhenotypeArray(samples=hap.samples)
        rng = np.random.RandomState(42)
        result = comp.compute(node, hap, pheno, rng=rng, generation=0, pedigree_history={})
        # Extract one value per family
        family_vals = result[::2]
        var = np.var(family_vals, ddof=1)
        assert 0.5 < var < 5.0, f"Variance = {var}, expected ≈ 2.0"


class TestCNoiseGrouped:
    def test_ungrouped_cnoise(self):
        """CNoiseComponent without grouping produces k independent arrays."""
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        comp = CNoiseComponent(cov=cov)
        node = ArchNode(outputs=['A', 'B'], component=comp, inputs=[], grouping=None)
        hap = _make_hap(n=10)
        pheno = NPhenotypeArray(samples=hap.samples)
        rng = np.random.RandomState(42)
        result = comp.compute(node, hap, pheno, rng=rng, generation=0, pedigree_history={})
        assert result.shape == (10, 2)

    def test_grouped_cnoise_fid(self):
        """CNoiseComponent with FID grouping: siblings get same multivariate noise."""
        cov = np.array([[1.0, 0.3], [0.3, 1.0]])
        comp = CNoiseComponent(cov=cov)
        node = ArchNode(outputs=['A', 'B'], component=comp, inputs=[], grouping='FID')
        fids = np.array([0, 0, 1, 1, 2, 2])
        hap = _make_hap(n=6, fids=fids)
        pheno = NPhenotypeArray(samples=hap.samples)
        rng = np.random.RandomState(42)
        result = comp.compute(node, hap, pheno, rng=rng, generation=0, pedigree_history={})
        assert result.shape == (6, 2)
        # Siblings should have same values
        np.testing.assert_array_equal(result[0], result[1])
        np.testing.assert_array_equal(result[2], result[3])
        np.testing.assert_array_equal(result[4], result[5])

    def test_cnoise_cov_validation_non_square(self):
        """Non-square covariance should raise."""
        with pytest.raises((ValueError, Exception)):
            CNoiseComponent(cov=np.array([[1, 2, 3], [4, 5, 6]]))
