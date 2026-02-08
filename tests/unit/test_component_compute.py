"""
Unit tests for individual ArchComponent.compute() methods.

Tests:
1. GeneticComponent: standardized vs raw, 2D effects (multivariate)
2. NoiseComponent: correct shape, rng determinism
3. CNoiseComponent: shape for k>1, non-square raises
4. AggregationComponent: expression parsing, _extract_names
5. MVGeneticComponent: multi-output shape
6. HaplotypeGeneticComponent: maternal vs paternal
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray, NPhenotypeArray
from xftsim.narch import (
    GeneticComponent, MVGeneticComponent, HaplotypeGeneticComponent,
    NoiseComponent, CNoiseComponent, AggregationComponent, ArchNode,
    Architecture,
)
from xftsim.neffect import AdditiveEffects, MultivariateEffects

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


def _make_hap(n=20, m=10, seed=42):
    return TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)


class TestGeneticComponentCompute:
    def test_standardized_zero_mean(self):
        """Standardized genetic component should produce ~zero mean."""
        hap = _make_hap(n=200, m=20)
        eff = AdditiveEffects.from_h2(h2=0.5, m=20, standardized=True, seed=42)
        comp = GeneticComponent(eff)
        node = ArchNode(outputs=['Y.G'], component=comp, inputs=[])
        pheno = NPhenotypeArray(samples=hap.samples)
        result = comp.compute(node, hap, pheno, rng=np.random.RandomState(42))
        assert abs(np.mean(result)) < 0.5  # Approximately centered

    def test_raw_effects(self):
        """Non-standardized effects should use raw diploid genotypes."""
        hap = _make_hap(n=20, m=5)
        eff = AdditiveEffects.from_h2(h2=0.5, m=5, standardized=False, seed=42)
        comp = GeneticComponent(eff)
        node = ArchNode(outputs=['Y.G'], component=comp, inputs=[])
        pheno = NPhenotypeArray(samples=hap.samples)
        result = comp.compute(node, hap, pheno, rng=np.random.RandomState(42))
        expected = hap.matvec(eff.effects)
        np.testing.assert_allclose(result, expected)

    def test_shape(self):
        """Output should be (n,) for univariate effects."""
        hap = _make_hap(n=20, m=10)
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        comp = GeneticComponent(eff)
        node = ArchNode(outputs=['Y.G'], component=comp, inputs=[])
        pheno = NPhenotypeArray(samples=hap.samples)
        result = comp.compute(node, hap, pheno, rng=np.random.RandomState(42))
        assert result.shape == (20,)


class TestMVGeneticComponentCompute:
    def test_multi_output_shape(self):
        """MVGenetic with k=2 should return (n, 2)."""
        hap = _make_hap(n=20, m=10)
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.4, m=10, seed=42)
        comp = MVGeneticComponent(eff)
        node = ArchNode(outputs=['Y1.G', 'Y2.G'], component=comp, inputs=[])
        pheno = NPhenotypeArray(samples=hap.samples)
        result = comp.compute(node, hap, pheno, rng=np.random.RandomState(42))
        assert result.shape == (20, 2)


class TestHaplotypeGeneticComponentCompute:
    def test_maternal_vs_paternal_differ(self):
        """Maternal and paternal components should produce different values."""
        hap = _make_hap(n=20, m=10)
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        mat = HaplotypeGeneticComponent(eff, haplotype='maternal')
        pat = HaplotypeGeneticComponent(eff, haplotype='paternal')
        node_m = ArchNode(outputs=['Y.Gmat'], component=mat, inputs=[])
        node_p = ArchNode(outputs=['Y.Gpat'], component=pat, inputs=[])
        pheno = NPhenotypeArray(samples=hap.samples)
        rng = np.random.RandomState(42)
        result_m = mat.compute(node_m, hap, pheno, rng=rng)
        result_p = pat.compute(node_p, hap, pheno, rng=rng)
        # They should differ (different haplotype columns)
        assert not np.allclose(result_m, result_p)

    def test_maternal_shape(self):
        """Maternal component should return (n,)."""
        hap = _make_hap(n=20, m=10)
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        comp = HaplotypeGeneticComponent(eff, haplotype='maternal')
        node = ArchNode(outputs=['Y.Gmat'], component=comp, inputs=[])
        pheno = NPhenotypeArray(samples=hap.samples)
        result = comp.compute(node, hap, pheno, rng=np.random.RandomState(42))
        assert result.shape == (20,)


class TestNoiseComponentCompute:
    def test_shape(self):
        """Noise should return (n,)."""
        hap = _make_hap(n=20, m=10)
        comp = NoiseComponent(variance=1.0)
        node = ArchNode(outputs=['Y.E'], component=comp, inputs=[])
        pheno = NPhenotypeArray(samples=hap.samples)
        result = comp.compute(node, hap, pheno, rng=np.random.RandomState(42))
        assert result.shape == (20,)

    def test_deterministic_with_seed(self):
        """Same rng should produce same noise."""
        hap = _make_hap(n=20, m=10)
        comp = NoiseComponent(variance=1.0)
        node = ArchNode(outputs=['Y.E'], component=comp, inputs=[])
        pheno = NPhenotypeArray(samples=hap.samples)
        r1 = comp.compute(node, hap, pheno, rng=np.random.RandomState(42))
        r2 = comp.compute(node, hap, pheno, rng=np.random.RandomState(42))
        np.testing.assert_array_equal(r1, r2)

    def test_repr(self):
        """NoiseComponent repr should show variance."""
        comp = NoiseComponent(variance=0.5)
        assert 'variance=0.5' in repr(comp)


class TestCNoiseComponentCompute:
    def test_shape(self):
        """CNoiseComponent with k=3 should return (n, 3)."""
        hap = _make_hap(n=20, m=10)
        cov = np.eye(3)
        comp = CNoiseComponent(cov=cov)
        node = ArchNode(outputs=['A', 'B', 'C'], component=comp, inputs=[])
        pheno = NPhenotypeArray(samples=hap.samples)
        result = comp.compute(node, hap, pheno, rng=np.random.RandomState(42))
        assert result.shape == (20, 3)

    def test_non_square_raises(self):
        """Non-square cov should raise ValueError."""
        with pytest.raises(ValueError, match="square"):
            CNoiseComponent(cov=np.ones((2, 3)))

    def test_k_property(self):
        """k should match cov dimension."""
        comp = CNoiseComponent(cov=np.eye(4))
        assert comp.k == 4


class TestAggregationComponentCompute:
    def test_extract_names(self):
        """_extract_names should find all variable references."""
        comp = AggregationComponent('a + b.c * d')
        assert 'a' in comp._input_names
        assert 'b.c' in comp._input_names
        assert 'd' in comp._input_names

    def test_dedup_names(self):
        """_extract_names should deduplicate."""
        comp = AggregationComponent('a + a + b')
        assert comp._input_names.count('a') == 1
        assert comp._input_names.count('b') == 1

    def test_repr(self):
        """AggregationComponent repr should show expression."""
        comp = AggregationComponent('X + Y')
        r = repr(comp)
        assert 'X + Y' in r
