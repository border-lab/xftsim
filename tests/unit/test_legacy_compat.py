"""
Unit tests for backward-compatibility methods and deprecated interfaces.

Tests:
1. DenseHaplotypeArray.get_sample_indexer() emits DeprecationWarning
2. DenseHaplotypeArray.get_variant_indexer() emits DeprecationWarning
3. SampleMeta.to_sample_index() returns SampleIndex
4. VariantMeta.to_variant_index() returns DiploidVariantIndex
5. NHaplotypeArrayAccessor.get_sample_indexer delegates
6. NHaplotypeArrayAccessor.get_variant_indexer delegates
7. NHaplotypeArray alias == DenseHaplotypeArray
"""
import numpy as np
import pytest
import warnings

from xftsim.struct import (
    SampleMeta, VariantMeta, DenseHaplotypeArray,
    NHaplotypeArray, NHaplotypeArrayAccessor,
)
import xftsim.index as xft_index

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestDeprecatedIndexers:
    def test_get_sample_indexer_warns(self):
        hap = TestSimulation.founder_haplotypes(n=10, m=5, seed=42)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            idx = hap.get_sample_indexer()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
        assert isinstance(idx, xft_index.SampleIndex)

    def test_get_variant_indexer_warns(self):
        hap = TestSimulation.founder_haplotypes(n=10, m=5, seed=42)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            idx = hap.get_variant_indexer()
            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()
        assert isinstance(idx, xft_index.DiploidVariantIndex)


class TestSampleMetaToIndex:
    def test_to_sample_index_type(self):
        sm = SampleMeta(iid=np.arange(5), generation=2)
        idx = sm.to_sample_index()
        assert isinstance(idx, xft_index.SampleIndex)

    def test_to_sample_index_n(self):
        sm = SampleMeta(iid=np.arange(5), generation=2)
        idx = sm.to_sample_index()
        assert idx.n == 5

    def test_to_sample_index_generation(self):
        sm = SampleMeta(iid=np.arange(5), generation=3)
        idx = sm.to_sample_index()
        assert idx.generation == 3

    def test_to_sample_index_iid_strings(self):
        sm = SampleMeta(iid=np.array([10, 20, 30]))
        idx = sm.to_sample_index()
        # IIDs should be converted to strings
        assert idx.iid.dtype.kind in ('U', 'O')  # string or object


class TestVariantMetaToIndex:
    def test_to_variant_index_type(self):
        vm = VariantMeta(vid=np.array(['v1', 'v2', 'v3']))
        idx = vm.to_variant_index()
        assert isinstance(idx, xft_index.DiploidVariantIndex)

    def test_to_variant_index_with_af(self):
        vm = VariantMeta(vid=np.array(['v1', 'v2']))
        af = np.array([0.1, 0.3])
        idx = vm.to_variant_index(af=af)
        np.testing.assert_array_equal(idx.af, af)

    def test_to_variant_index_uses_stored_af(self):
        vm = VariantMeta(
            vid=np.array(['v1', 'v2']),
            af=np.array([0.2, 0.4]),
        )
        idx = vm.to_variant_index()
        np.testing.assert_array_equal(idx.af, np.array([0.2, 0.4]))

    def test_to_variant_index_with_chrom(self):
        vm = VariantMeta(
            vid=np.array(['v1', 'v2']),
            chrom=np.array([1, 2]),
        )
        idx = vm.to_variant_index()
        np.testing.assert_array_equal(idx.chrom, np.array([1, 2]))


class TestAccessorDelegation:
    def test_accessor_get_sample_indexer(self):
        hap = TestSimulation.founder_haplotypes(n=10, m=5, seed=42)
        acc = NHaplotypeArrayAccessor(hap)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            idx = acc.get_sample_indexer()
        assert isinstance(idx, xft_index.SampleIndex)

    def test_accessor_get_variant_indexer(self):
        hap = TestSimulation.founder_haplotypes(n=10, m=5, seed=42)
        acc = NHaplotypeArrayAccessor(hap)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            idx = acc.get_variant_indexer()
        assert isinstance(idx, xft_index.DiploidVariantIndex)


class TestAlias:
    def test_nhaplotype_array_is_dense(self):
        assert NHaplotypeArray is DenseHaplotypeArray
