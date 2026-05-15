"""
Unit tests for sibling component edge cases.

Tests:
1. SiblingMeanComponent with missing source raises ValueError
2. SiblingMeanComponent without grouping (per-individual) returns copy
3. SiblingMeanComponent repr
4. Multiple sibling types produce different results
"""
import numpy as np
import pytest

from xftsim.arch import (
    Architecture, GeneticComponent, NoiseComponent, AggregationComponent,
    SiblingMeanComponent, SiblingSumComponent,
    ArchNode,
)
from xftsim.struct import SampleMeta, PhenotypeArray
from xftsim.effect import AdditiveEffects

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestSiblingComponentEdges:
    def test_missing_source_raises(self):
        """Sibling component with nonexistent source should raise ValueError."""
        n = 10
        hap = TestSimulation.founder_haplotypes(n=n, m=5, seed=42)
        pheno = PhenotypeArray(hap.samples)
        pheno['Y'] = np.random.normal(0, 1, n)

        comp = SiblingMeanComponent('NONEXISTENT')
        node = ArchNode(outputs=['Y.sib'], component=comp, inputs=[], grouping='FID')

        with pytest.raises(ValueError, match="source.*not found"):
            comp.compute(node, hap, pheno)

    def test_no_grouping_returns_copy(self):
        """Without grouping (labels=None), should return values copy."""
        n = 10
        hap = TestSimulation.founder_haplotypes(n=n, m=5, seed=42)
        pheno = PhenotypeArray(hap.samples)
        pheno['Y'] = np.arange(n, dtype=np.float64)

        comp = SiblingMeanComponent('Y')
        # grouping=None → per-individual (IID), labels=None
        node = ArchNode(outputs=['Y.sib'], component=comp, inputs=[], grouping=None)

        result = comp.compute(node, hap, pheno)
        np.testing.assert_array_equal(result, pheno['Y'])

    def test_repr(self):
        """SiblingMeanComponent repr should show source name."""
        comp = SiblingMeanComponent('Y')
        assert "SiblingMeanComponent" in repr(comp)
        assert "'Y'" in repr(comp)

    def test_mean_vs_sum_different(self):
        """SiblingMean and SiblingSum should produce different results for families > 1."""
        n = 12
        fids = np.repeat(np.arange(3), 4)
        sm = SampleMeta(iid=np.arange(n), fid=fids)
        pheno = PhenotypeArray(sm)
        pheno['Y'] = np.array([1, 2, 3, 4, 10, 20, 30, 40, 100, 200, 300, 400],
                              dtype=np.float64)

        # Create haplotypes with matching samples (same fids)
        geno = np.random.RandomState(42).randint(0, 2, (n, 5, 2)).astype(np.int8)
        from xftsim.struct import DenseHaplotypeArray, VariantMeta
        vm = VariantMeta(vid=np.array([f'v{i}' for i in range(5)]))
        hap = DenseHaplotypeArray(geno, samples=sm, variants=vm)

        mean_comp = SiblingMeanComponent('Y')
        sum_comp = SiblingSumComponent('Y')

        node_mean = ArchNode(outputs=['Y.mean'], component=mean_comp,
                             inputs=[], grouping='FID')
        node_sum = ArchNode(outputs=['Y.sum'], component=sum_comp,
                            inputs=[], grouping='FID')

        result_mean = mean_comp.compute(
            node_mean, hap, pheno,
            phenotype_history={0: pheno}, generation=0,
        )
        result_sum = sum_comp.compute(
            node_sum, hap, pheno,
            phenotype_history={0: pheno}, generation=0,
        )

        # For families of size 4, sum = 4 * mean
        assert not np.allclose(result_mean, result_sum)

    def test_sibling_mean_correct_values(self):
        """SiblingMean should return family mean for each individual."""
        n = 6
        fids = np.array([0, 0, 0, 1, 1, 1])
        sm = SampleMeta(iid=np.arange(n), fid=fids)
        pheno = PhenotypeArray(sm)
        pheno['Y'] = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])

        # Create haplotypes with matching samples (same fids)
        geno = np.random.RandomState(42).randint(0, 2, (n, 5, 2)).astype(np.int8)
        from xftsim.struct import DenseHaplotypeArray, VariantMeta
        vm = VariantMeta(vid=np.array([f'v{i}' for i in range(5)]))
        hap = DenseHaplotypeArray(geno, samples=sm, variants=vm)

        comp = SiblingMeanComponent('Y')
        node = ArchNode(outputs=['Y.sib_mean'], component=comp,
                        inputs=[], grouping='FID')

        result = comp.compute(
            node, hap, pheno,
            phenotype_history={0: pheno}, generation=0,
        )

        # Family 0: mean = 2.0, Family 1: mean = 20.0
        expected = np.array([2.0, 2.0, 2.0, 20.0, 20.0, 20.0])
        np.testing.assert_allclose(result, expected)
