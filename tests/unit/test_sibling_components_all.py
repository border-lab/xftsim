"""
Unit tests for all sibling aggregation components.

Tests SiblingAnyComponent, SiblingCountComponent,
SiblingEldestComponent, SiblingYoungestComponent.
"""
import numpy as np
import pytest

from xftsim.arch import (
    SiblingAnyComponent, SiblingCountComponent,
    SiblingEldestComponent, SiblingYoungestComponent,
    ArchNode,
)
from xftsim.struct import SampleMeta, PhenotypeArray, DenseHaplotypeArray, VariantMeta


def _make_grouped_scenario(fids, values):
    """Create haplotype + phenotype with specified FIDs and Y values."""
    n = len(fids)
    sm = SampleMeta(iid=np.arange(n), fid=np.array(fids))
    vm = VariantMeta(vid=np.array([f'v{i}' for i in range(3)]))
    hap = DenseHaplotypeArray(np.zeros((n, 3, 2), dtype=np.int8), samples=sm, variants=vm)
    pheno = PhenotypeArray(sm)
    pheno['Y'] = np.array(values, dtype=np.float64)
    return hap, pheno


class TestSiblingAnyComponent:
    def test_any_with_positive_values(self):
        """Any returns 1.0 for all members if any member > 0."""
        fids = [0, 0, 0, 1, 1]
        vals = [0.0, 0.0, 1.0, 0.0, 0.0]  # family 0 has one positive, family 1 has none
        hap, pheno = _make_grouped_scenario(fids, vals)

        comp = SiblingAnyComponent('Y')
        node = ArchNode(outputs=['Y.any'], component=comp, inputs=['Y'], grouping='FID')
        result = comp.compute(node, hap, pheno)

        # Family 0: any > 0 → 1.0 for all; Family 1: none > 0 → 0.0 for all
        expected = np.array([1.0, 1.0, 1.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_any_all_zeros(self):
        """Any returns 0.0 when all values are zero."""
        fids = [0, 0, 1, 1]
        vals = [0.0, 0.0, 0.0, 0.0]
        hap, pheno = _make_grouped_scenario(fids, vals)

        comp = SiblingAnyComponent('Y')
        node = ArchNode(outputs=['Y.any'], component=comp, inputs=['Y'], grouping='FID')
        result = comp.compute(node, hap, pheno)
        np.testing.assert_array_equal(result, np.zeros(4))

    def test_any_all_positive(self):
        """Any returns 1.0 when all values are positive."""
        fids = [0, 0]
        vals = [5.0, 3.0]
        hap, pheno = _make_grouped_scenario(fids, vals)

        comp = SiblingAnyComponent('Y')
        node = ArchNode(outputs=['Y.any'], component=comp, inputs=['Y'], grouping='FID')
        result = comp.compute(node, hap, pheno)
        np.testing.assert_array_equal(result, np.array([1.0, 1.0]))


class TestSiblingCountComponent:
    def test_count_varying_sizes(self):
        """Count returns group sizes."""
        fids = [0, 0, 0, 1, 1, 2]
        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        hap, pheno = _make_grouped_scenario(fids, vals)

        comp = SiblingCountComponent('Y')
        node = ArchNode(outputs=['Y.cnt'], component=comp, inputs=['Y'], grouping='FID')
        result = comp.compute(node, hap, pheno)

        expected = np.array([3.0, 3.0, 3.0, 2.0, 2.0, 1.0])
        np.testing.assert_array_equal(result, expected)

    def test_count_single_family(self):
        """All members in one family."""
        fids = [0, 0, 0, 0]
        vals = [1.0, 2.0, 3.0, 4.0]
        hap, pheno = _make_grouped_scenario(fids, vals)

        comp = SiblingCountComponent('Y')
        node = ArchNode(outputs=['Y.cnt'], component=comp, inputs=['Y'], grouping='FID')
        result = comp.compute(node, hap, pheno)
        np.testing.assert_array_equal(result, np.array([4.0, 4.0, 4.0, 4.0]))


class TestSiblingEldestComponent:
    def test_eldest_picks_first_member(self):
        """Eldest returns value of first (lowest index) member."""
        fids = [0, 0, 0, 1, 1]
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        hap, pheno = _make_grouped_scenario(fids, vals)

        comp = SiblingEldestComponent('Y')
        node = ArchNode(outputs=['Y.eld'], component=comp, inputs=['Y'], grouping='FID')
        result = comp.compute(node, hap, pheno)

        # Family 0: eldest is index 0 (10.0); Family 1: eldest is index 3 (40.0)
        expected = np.array([10.0, 10.0, 10.0, 40.0, 40.0])
        np.testing.assert_array_equal(result, expected)

    def test_eldest_single_member(self):
        """Eldest of singleton group is that individual's value."""
        fids = [0, 1, 2]
        vals = [5.0, 10.0, 15.0]
        hap, pheno = _make_grouped_scenario(fids, vals)

        comp = SiblingEldestComponent('Y')
        node = ArchNode(outputs=['Y.eld'], component=comp, inputs=['Y'], grouping='FID')
        result = comp.compute(node, hap, pheno)
        np.testing.assert_array_equal(result, np.array([5.0, 10.0, 15.0]))


class TestSiblingYoungestComponent:
    def test_youngest_picks_last_member(self):
        """Youngest returns value of last (highest index) member."""
        fids = [0, 0, 0, 1, 1]
        vals = [10.0, 20.0, 30.0, 40.0, 50.0]
        hap, pheno = _make_grouped_scenario(fids, vals)

        comp = SiblingYoungestComponent('Y')
        node = ArchNode(outputs=['Y.yng'], component=comp, inputs=['Y'], grouping='FID')
        result = comp.compute(node, hap, pheno)

        # Family 0: youngest is index 2 (30.0); Family 1: youngest is index 4 (50.0)
        expected = np.array([30.0, 30.0, 30.0, 50.0, 50.0])
        np.testing.assert_array_equal(result, expected)

    def test_youngest_single_member(self):
        """Youngest of singleton group is that individual's value."""
        fids = [0, 1]
        vals = [7.0, 3.0]
        hap, pheno = _make_grouped_scenario(fids, vals)

        comp = SiblingYoungestComponent('Y')
        node = ArchNode(outputs=['Y.yng'], component=comp, inputs=['Y'], grouping='FID')
        result = comp.compute(node, hap, pheno)
        np.testing.assert_array_equal(result, np.array([7.0, 3.0]))
