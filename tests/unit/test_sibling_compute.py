"""
Unit tests for sibling component _aggregate_groups methods directly.

Tests each sibling component with known inputs and groups.
"""
import numpy as np
import pytest

from xftsim.arch import (
    SiblingMeanComponent, SiblingSumComponent, SiblingAnyComponent,
    SiblingCountComponent, SiblingEldestComponent, SiblingYoungestComponent,
)


# groups: [A, A, B, B, B, C]
LABELS = np.array([0, 0, 1, 1, 1, 2])
VALUES = np.array([1.0, 3.0, 10.0, 20.0, 30.0, 5.0])


class TestSiblingMean:
    def test_basic(self):
        comp = SiblingMeanComponent('Y')
        result = comp._aggregate_groups(VALUES, LABELS)
        # Group 0: mean(1,3)=2, Group 1: mean(10,20,30)=20, Group 2: 5
        np.testing.assert_allclose(result, [2.0, 2.0, 20.0, 20.0, 20.0, 5.0])

    def test_single_member_groups(self):
        labels = np.array([0, 1, 2])
        values = np.array([10.0, 20.0, 30.0])
        comp = SiblingMeanComponent('Y')
        result = comp._aggregate_groups(values, labels)
        np.testing.assert_allclose(result, [10.0, 20.0, 30.0])

    def test_all_same_group(self):
        labels = np.zeros(4, dtype=int)
        values = np.array([1.0, 2.0, 3.0, 4.0])
        comp = SiblingMeanComponent('Y')
        result = comp._aggregate_groups(values, labels)
        np.testing.assert_allclose(result, [2.5, 2.5, 2.5, 2.5])


class TestSiblingSum:
    def test_basic(self):
        comp = SiblingSumComponent('Y')
        result = comp._aggregate_groups(VALUES, LABELS)
        # Group 0: 1+3=4, Group 1: 10+20+30=60, Group 2: 5
        np.testing.assert_allclose(result, [4.0, 4.0, 60.0, 60.0, 60.0, 5.0])

    def test_zeros(self):
        labels = np.array([0, 0, 1, 1])
        values = np.zeros(4)
        comp = SiblingSumComponent('Y')
        result = comp._aggregate_groups(values, labels)
        np.testing.assert_array_equal(result, np.zeros(4))


class TestSiblingAny:
    def test_basic(self):
        comp = SiblingAnyComponent('Y')
        labels = np.array([0, 0, 1, 1, 1, 2])
        values = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        result = comp._aggregate_groups(values, labels)
        # Group 0: no positive → 0, Group 1: has 1.0 → 1, Group 2: 0
        np.testing.assert_array_equal(result, [0.0, 0.0, 1.0, 1.0, 1.0, 0.0])

    def test_all_positive(self):
        labels = np.array([0, 0])
        values = np.array([5.0, 3.0])
        comp = SiblingAnyComponent('Y')
        result = comp._aggregate_groups(values, labels)
        np.testing.assert_array_equal(result, [1.0, 1.0])

    def test_all_zero(self):
        labels = np.array([0, 0])
        values = np.zeros(2)
        comp = SiblingAnyComponent('Y')
        result = comp._aggregate_groups(values, labels)
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_negative_not_positive(self):
        """Negative values should not count as positive."""
        labels = np.array([0, 0])
        values = np.array([-1.0, -2.0])
        comp = SiblingAnyComponent('Y')
        result = comp._aggregate_groups(values, labels)
        np.testing.assert_array_equal(result, [0.0, 0.0])


class TestSiblingCount:
    def test_basic(self):
        comp = SiblingCountComponent('Y')
        result = comp._aggregate_groups(VALUES, LABELS)
        np.testing.assert_array_equal(result, [2.0, 2.0, 3.0, 3.0, 3.0, 1.0])

    def test_all_same_group(self):
        labels = np.zeros(5, dtype=int)
        values = np.ones(5)
        comp = SiblingCountComponent('Y')
        result = comp._aggregate_groups(values, labels)
        np.testing.assert_array_equal(result, [5.0, 5.0, 5.0, 5.0, 5.0])


class TestSiblingEldest:
    def test_basic(self):
        comp = SiblingEldestComponent('Y')
        result = comp._aggregate_groups(VALUES, LABELS)
        # Eldest = first in array: Group 0: 1.0, Group 1: 10.0, Group 2: 5.0
        np.testing.assert_array_equal(result, [1.0, 1.0, 10.0, 10.0, 10.0, 5.0])

    def test_single_member(self):
        labels = np.array([0, 1])
        values = np.array([7.0, 3.0])
        comp = SiblingEldestComponent('Y')
        result = comp._aggregate_groups(values, labels)
        np.testing.assert_array_equal(result, [7.0, 3.0])


class TestSiblingYoungest:
    def test_basic(self):
        comp = SiblingYoungestComponent('Y')
        result = comp._aggregate_groups(VALUES, LABELS)
        # Youngest = last in array: Group 0: 3.0, Group 1: 30.0, Group 2: 5.0
        np.testing.assert_array_equal(result, [3.0, 3.0, 30.0, 30.0, 30.0, 5.0])

    def test_single_member(self):
        labels = np.array([0, 1])
        values = np.array([7.0, 3.0])
        comp = SiblingYoungestComponent('Y')
        result = comp._aggregate_groups(values, labels)
        np.testing.assert_array_equal(result, [7.0, 3.0])


class TestSiblingRepr:
    def test_mean_repr(self):
        comp = SiblingMeanComponent('height')
        assert 'SiblingMeanComponent' in repr(comp)
        assert "'height'" in repr(comp)

    def test_count_repr(self):
        comp = SiblingCountComponent('Y')
        assert 'SiblingCountComponent' in repr(comp)

    def test_any_repr(self):
        comp = SiblingAnyComponent('affected')
        assert "'affected'" in repr(comp)


class TestSiblingMissingSource:
    def test_missing_source_raises(self):
        """Source not in phenotypes should raise ValueError."""
        from xftsim.struct import SampleMeta, NPhenotypeArray
        from xftsim.arch import ArchNode
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from testdata import TestSimulation

        hap = TestSimulation.founder_haplotypes(n=10, m=3, seed=42)
        comp = SiblingMeanComponent('NONEXISTENT')
        node = ArchNode(outputs=['Y.sib'], component=comp, inputs=['NONEXISTENT'],
                        grouping='FID')
        pheno = NPhenotypeArray(samples=hap.samples)
        with pytest.raises(ValueError, match="not found"):
            comp.compute(node, hap, pheno)
