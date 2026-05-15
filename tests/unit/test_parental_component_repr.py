"""
Unit tests for parental component repr and construction edge cases.

Tests:
1. MotherComponent repr
2. FatherComponent repr
3. ParentComponent repr (midparent)
4. MotherComponent with founder_component
5. MotherComponent kind is 'reference'
6. ParentComponent produces midparent average
"""
import numpy as np
import pytest

from xftsim.arch import MotherComponent, FatherComponent, ParentComponent
from xftsim.arch import NoiseComponent
from xftsim.struct import SampleMeta, PhenotypeArray, PedigreeArray


class TestParentalComponentRepr:
    def test_mother_repr(self):
        r = repr(MotherComponent('Y'))
        assert 'MotherComponent' in r
        assert "'Y'" in r

    def test_father_repr(self):
        r = repr(FatherComponent('Y'))
        assert 'FatherComponent' in r

    def test_parent_repr(self):
        r = repr(ParentComponent('Y'))
        assert 'ParentComponent' in r


class TestParentalComponentKind:
    def test_kind_is_reference(self):
        assert MotherComponent('Y').kind == 'reference'
        assert FatherComponent('Y').kind == 'reference'
        assert ParentComponent('Y').kind == 'reference'

    def test_accepts_grouping_false(self):
        assert MotherComponent('Y').accepts_grouping is False


class TestParentalComponentCompute:
    def _make_scenario(self):
        """Create parent and offspring phenotypes + pedigree."""
        parent_sm = SampleMeta(iid=np.arange(10), generation=0)
        parent_pheno = PhenotypeArray(parent_sm)
        parent_pheno['Y'] = np.arange(10, dtype=np.float64)

        offspring_sm = SampleMeta(iid=np.arange(4), generation=1)
        offspring_pheno = PhenotypeArray(offspring_sm)
        offspring_pheno['Y'] = np.zeros(4)

        # Offspring 0,1 from parents (0,5); offspring 2,3 from parents (1,6)
        ped = PedigreeArray(
            offspring_samples=offspring_sm,
            maternal_idx=np.array([0, 0, 1, 1]),
            paternal_idx=np.array([5, 5, 6, 6]),
            parent_n=10,
        )
        return parent_pheno, offspring_pheno, ped

    def test_mother_extracts_maternal(self):
        """MotherComponent should return mother's phenotype."""
        parent_pheno, offspring_pheno, ped = self._make_scenario()

        from xftsim.arch import ArchNode
        from xftsim.struct import DenseHaplotypeArray, VariantMeta
        hap = DenseHaplotypeArray(
            np.zeros((4, 5, 2), dtype=np.int8),
            samples=offspring_pheno.samples,
            variants=VariantMeta(vid=np.array([f'v{i}' for i in range(5)])),
        )

        comp = MotherComponent('Y')
        node = ArchNode(outputs=['Y.m'], component=comp, inputs=[], grouping=None)
        result = comp.compute(
            node, hap, offspring_pheno,
            phenotype_history={0: parent_pheno},
            pedigree_history={1: ped},
            generation=1,
        )
        # Parent values are 0-9, maternal indices are [0,0,1,1]
        expected = np.array([0.0, 0.0, 1.0, 1.0])
        np.testing.assert_array_equal(result, expected)

    def test_parent_midparent_average(self):
        """ParentComponent should return (mother + father) / 2."""
        parent_pheno, offspring_pheno, ped = self._make_scenario()

        from xftsim.arch import ArchNode
        from xftsim.struct import DenseHaplotypeArray, VariantMeta
        hap = DenseHaplotypeArray(
            np.zeros((4, 5, 2), dtype=np.int8),
            samples=offspring_pheno.samples,
            variants=VariantMeta(vid=np.array([f'v{i}' for i in range(5)])),
        )

        comp = ParentComponent('Y')
        node = ArchNode(outputs=['Y.p'], component=comp, inputs=[], grouping=None)
        result = comp.compute(
            node, hap, offspring_pheno,
            phenotype_history={0: parent_pheno},
            pedigree_history={1: ped},
            generation=1,
        )
        # maternal: [0,0,1,1], paternal: [5,5,6,6]
        # midparent: [(0+5)/2, (0+5)/2, (1+6)/2, (1+6)/2] = [2.5, 2.5, 3.5, 3.5]
        expected = np.array([2.5, 2.5, 3.5, 3.5])
        np.testing.assert_array_equal(result, expected)
