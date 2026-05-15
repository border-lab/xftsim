"""
Unit tests for CNoiseComponent edge cases.

Tests:
1. CNoiseComponent k property
2. CNoiseComponent covariance validation
3. CNoiseComponent 1x1 covariance (univariate fallback)
4. CNoiseComponent 3x3 covariance
5. CNoiseComponent repr
6. CNoiseComponent compute produces correct shape
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray, PhenotypeArray
from xftsim.arch import CNoiseComponent, ArchNode


def _make_hap(n=10, m=3):
    sm = SampleMeta(iid=np.arange(n), fid=np.arange(n) // 2)
    vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
    geno = np.zeros((n, m, 2), dtype=np.int8)
    return DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)


class TestCNoiseProperties:
    def test_k_property_2x2(self):
        cov = np.array([[1.0, 0.3], [0.3, 1.0]])
        comp = CNoiseComponent(cov=cov)
        assert comp.k == 2

    def test_k_property_3x3(self):
        cov = np.eye(3)
        comp = CNoiseComponent(cov=cov)
        assert comp.k == 3

    def test_k_property_1x1(self):
        cov = np.array([[2.0]])
        comp = CNoiseComponent(cov=cov)
        assert comp.k == 1


class TestCNoiseValidation:
    def test_non_square_raises(self):
        with pytest.raises(ValueError, match="square"):
            CNoiseComponent(cov=np.array([[1, 2, 3], [4, 5, 6]]))

    def test_1d_raises(self):
        with pytest.raises(ValueError, match="square"):
            CNoiseComponent(cov=np.array([1, 2, 3]))


class TestCNoiseCompute:
    def test_2d_output_shape(self):
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        comp = CNoiseComponent(cov=cov)
        node = ArchNode(outputs=['A', 'B'], component=comp, inputs=[], grouping=None)
        hap = _make_hap(n=10)
        pheno = PhenotypeArray(samples=hap.samples)
        rng = np.random.RandomState(42)
        result = comp.compute(node, hap, pheno, rng=rng, generation=0, pedigree_history={})
        assert result.shape == (10, 2)

    def test_3d_output_shape(self):
        cov = np.eye(3)
        comp = CNoiseComponent(cov=cov)
        node = ArchNode(outputs=['A', 'B', 'C'], component=comp, inputs=[], grouping=None)
        hap = _make_hap(n=8)
        pheno = PhenotypeArray(samples=hap.samples)
        rng = np.random.RandomState(42)
        result = comp.compute(node, hap, pheno, rng=rng, generation=0, pedigree_history={})
        assert result.shape == (8, 3)

    def test_reproducibility(self):
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        comp = CNoiseComponent(cov=cov)
        node = ArchNode(outputs=['A', 'B'], component=comp, inputs=[])
        hap = _make_hap(n=10)
        pheno = PhenotypeArray(samples=hap.samples)
        r1 = comp.compute(node, hap, pheno, rng=np.random.RandomState(42),
                          generation=0, pedigree_history={})
        r2 = comp.compute(node, hap, pheno, rng=np.random.RandomState(42),
                          generation=0, pedigree_history={})
        np.testing.assert_array_equal(r1, r2)


class TestCNoiseRepr:
    def test_repr(self):
        cov = np.eye(2)
        comp = CNoiseComponent(cov=cov)
        r = repr(comp)
        assert 'CNoiseComponent' in r
        assert '(2, 2)' in r
