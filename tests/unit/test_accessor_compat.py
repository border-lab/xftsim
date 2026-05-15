"""
Unit tests for HaplotypeArrayAccessor and DenseHaplotypeArray compatibility.

Tests:
1. .xft accessor properties: n, m, generation, samples, variants, af_empirical
2. .xft.to_diploid: matches diploid_genotypes
3. .xft.to_diploid_standardized: centered
4. DenseHaplotypeArray matvec_maternal/paternal: correct haplotype access
5. DenseHaplotypeArray to_dense: returns self
6. DenseHaplotypeArray __repr__
7. DenseHaplotypeArray alias
"""
import numpy as np
import pytest

from xftsim.struct import (
    SampleMeta, VariantMeta, DenseHaplotypeArray, DenseHaplotypeArray,
)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


def _make_hap(n=10, m=5, seed=42):
    return TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)


class TestXftAccessor:
    def test_n(self):
        """xft.n should match n."""
        hap = _make_hap()
        assert hap.xft.n == hap.n

    def test_m(self):
        """xft.m should match m."""
        hap = _make_hap()
        assert hap.xft.m == hap.m

    def test_generation(self):
        """xft.generation should match generation."""
        hap = _make_hap()
        assert hap.xft.generation == hap.generation

    def test_samples(self):
        """xft.samples should be same object."""
        hap = _make_hap()
        assert hap.xft.samples is hap.samples

    def test_variants(self):
        """xft.variants should be same object."""
        hap = _make_hap()
        assert hap.xft.variants is hap.variants

    def test_af_empirical(self):
        """xft.af_empirical should match direct access."""
        hap = _make_hap()
        np.testing.assert_array_equal(hap.xft.af_empirical, hap.af_empirical)


class TestXftDiploidMethods:
    def test_to_diploid(self):
        """xft.to_diploid should match diploid_genotypes."""
        hap = _make_hap(n=10, m=5)
        np.testing.assert_array_equal(hap.xft.to_diploid(), hap.diploid_genotypes)

    def test_to_diploid_standardized_centered(self):
        """xft.to_diploid_standardized should have zero column means."""
        hap = _make_hap(n=100, m=10, seed=42)
        G = hap.xft.to_diploid_standardized()
        np.testing.assert_allclose(G.mean(axis=0), 0.0, atol=1e-10)

    def test_to_diploid_standardized_with_scale(self):
        """xft.to_diploid_standardized(scale=True) should have unit variance."""
        hap = _make_hap(n=200, m=10, seed=42)
        G = hap.xft.to_diploid_standardized(scale=True)
        col_vars = G.var(axis=0)
        nonzero = col_vars > 0
        if np.any(nonzero):
            np.testing.assert_allclose(col_vars[nonzero], 1.0, atol=0.15)


class TestMatvecMaternalPaternal:
    def test_maternal_matches(self):
        """matvec_maternal should equal hap[:,:,0] @ v."""
        hap = _make_hap(n=10, m=5)
        v = np.random.RandomState(0).randn(5)
        result = hap.matvec_maternal(v)
        expected = hap.genotypes[:, :, 0] @ v
        np.testing.assert_allclose(result, expected)

    def test_paternal_matches(self):
        """matvec_paternal should equal hap[:,:,1] @ v."""
        hap = _make_hap(n=10, m=5)
        v = np.random.RandomState(0).randn(5)
        result = hap.matvec_paternal(v)
        expected = hap.genotypes[:, :, 1] @ v
        np.testing.assert_allclose(result, expected)

    def test_maternal_plus_paternal_equals_matvec(self):
        """maternal + paternal should equal matvec."""
        hap = _make_hap(n=10, m=5)
        v = np.random.RandomState(0).randn(5)
        mat = hap.matvec_maternal(v)
        pat = hap.matvec_paternal(v)
        full = hap.matvec(v)
        np.testing.assert_allclose(mat + pat, full)


class TestToDense:
    def test_returns_self(self):
        """to_dense() on DenseHaplotypeArray should return self."""
        hap = _make_hap()
        assert hap.to_dense() is hap


class TestDenseHaplotypeArrayRepr:
    def test_repr_basic(self):
        """Repr should include class name, n, m, generation."""
        hap = _make_hap(n=10, m=5)
        r = repr(hap)
        assert 'DenseHaplotypeArray' in r
        assert 'n=10' in r
        assert 'm=5' in r
        assert 'generation=' in r

    def test_repr_with_families(self):
        """If n_fam != n, repr should show n_fam."""
        sm = SampleMeta(
            iid=np.arange(10),
            fid=np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4]),
        )
        vm = VariantMeta(vid=np.array(['v0']))
        g = np.zeros((10, 1, 2), dtype=np.int8)
        hap = DenseHaplotypeArray(genotypes=g, generation=0, samples=sm, variants=vm)
        r = repr(hap)
        assert 'n_fam=5' in r


class TestNHaplotypeArrayAlias:
    def test_alias(self):
        """DenseHaplotypeArray should be DenseHaplotypeArray."""
        assert DenseHaplotypeArray is DenseHaplotypeArray
