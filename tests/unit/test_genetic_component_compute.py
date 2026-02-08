"""
Unit tests for GeneticComponent, MVGeneticComponent, and HaplotypeGeneticComponent compute.

Tests:
1. GeneticComponent produces (n,) array
2. MVGeneticComponent produces (n, k) array
3. HaplotypeGeneticComponent maternal vs paternal
4. HaplotypeGeneticComponent invalid haplotype raises
5. GeneticComponent result matches manual matvec
6. HaplotypeGenetic maternal + paternal ≈ GeneticComponent
"""
import numpy as np
import pytest

from xftsim.narch import (
    GeneticComponent, MVGeneticComponent, HaplotypeGeneticComponent, ArchNode,
)
from xftsim.neffect import AdditiveEffects, MultivariateEffects
from xftsim.struct import NPhenotypeArray

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestGeneticComponentCompute:
    def test_produces_1d_array(self):
        """GeneticComponent.compute should return (n,) array."""
        n, m = 50, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        comp = GeneticComponent(eff)
        node = ArchNode(outputs=['Y.G'], component=comp, inputs=[], grouping=None)
        pheno = NPhenotypeArray(hap.samples)

        result = comp.compute(node, hap, pheno)
        assert result.shape == (n,)
        assert np.all(np.isfinite(result))

    def test_matches_manual_matvec(self):
        """GeneticComponent should match hap.matvec(effects)."""
        n, m = 50, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, standardized=False, seed=42)
        comp = GeneticComponent(eff)
        node = ArchNode(outputs=['Y.G'], component=comp, inputs=[], grouping=None)
        pheno = NPhenotypeArray(hap.samples)

        result = comp.compute(node, hap, pheno)
        expected = hap.matvec(eff.effects)
        np.testing.assert_allclose(result, expected, atol=1e-10)


class TestMVGeneticComponentCompute:
    def test_produces_2d_array(self):
        """MVGeneticComponent.compute should return (n, k) array."""
        n, m = 50, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        mv = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.2, m=m, seed=42)
        comp = MVGeneticComponent(mv)
        node = ArchNode(outputs=['Y1.G', 'Y2.G'], component=comp,
                        inputs=[], grouping=None)
        pheno = NPhenotypeArray(hap.samples)

        result = comp.compute(node, hap, pheno)
        assert result.shape == (n, 2)
        assert np.all(np.isfinite(result))


class TestHaplotypeGeneticComponent:
    def test_invalid_haplotype_raises(self):
        """Invalid haplotype string should raise ValueError."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        with pytest.raises(ValueError, match="maternal.*paternal"):
            HaplotypeGeneticComponent(eff, haplotype='both')

    def test_maternal_compute(self):
        """Maternal haplotype genetic = hap[:,:,0] @ effects."""
        n, m = 50, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, standardized=False, seed=42)

        comp = HaplotypeGeneticComponent(eff, haplotype='maternal')
        node = ArchNode(outputs=['Y.mat'], component=comp, inputs=[], grouping=None)
        pheno = NPhenotypeArray(hap.samples)

        result = comp.compute(node, hap, pheno)
        expected = hap.matvec_maternal(eff.effects)
        np.testing.assert_allclose(result, expected)

    def test_paternal_compute(self):
        """Paternal haplotype genetic = hap[:,:,1] @ effects."""
        n, m = 50, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, standardized=False, seed=42)

        comp = HaplotypeGeneticComponent(eff, haplotype='paternal')
        node = ArchNode(outputs=['Y.pat'], component=comp, inputs=[], grouping=None)
        pheno = NPhenotypeArray(hap.samples)

        result = comp.compute(node, hap, pheno)
        expected = hap.matvec_paternal(eff.effects)
        np.testing.assert_allclose(result, expected)

    def test_maternal_plus_paternal_equals_genetic(self):
        """Maternal + paternal haplotype genetic ≈ diploid genetic."""
        n, m = 50, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, standardized=False, seed=42)

        comp_mat = HaplotypeGeneticComponent(eff, haplotype='maternal')
        comp_pat = HaplotypeGeneticComponent(eff, haplotype='paternal')
        node = ArchNode(outputs=['tmp'], component=comp_mat, inputs=[], grouping=None)
        pheno = NPhenotypeArray(hap.samples)

        mat = comp_mat.compute(node, hap, pheno)
        pat = comp_pat.compute(node, hap, pheno)
        dip = hap.matvec(eff.effects)

        np.testing.assert_allclose(mat + pat, dip, atol=1e-10)
