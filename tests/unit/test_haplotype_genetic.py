"""Tests for HaplotypeGeneticComponent."""
import numpy as np
import pytest

from xftsim.struct import DenseHaplotypeArray, PhenotypeArray, SampleMeta, VariantMeta
from xftsim.arch import (
    HaplotypeGeneticComponent, GeneticComponent, Architecture, AggregationComponent,
    ArchNode,
)
from xftsim.effect import AdditiveEffects
from xftsim.parser import parse_formula


def _make_haplotypes(n=100, m=50, seed=42):
    rng = np.random.RandomState(seed)
    geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
    sex = np.tile([0, 1], (n + 1) // 2)[:n]
    samples = SampleMeta(iid=np.arange(n), sex=sex)
    variants = VariantMeta(vid=np.arange(m), af=np.full(m, 0.5))
    return DenseHaplotypeArray(genotypes=geno, samples=samples, variants=variants)


def _make_effects(m=50, seed=123):
    return AdditiveEffects.from_h2(h2=0.5, m=m, seed=seed, standardized=False)


class TestHaplotypeGeneticComponent:
    def test_maternal_matches_manual(self):
        hap = _make_haplotypes()
        eff = _make_effects(m=hap.m)
        comp = HaplotypeGeneticComponent(eff, haplotype='maternal')
        node = ArchNode(outputs=['Y.mat'], component=comp, inputs=[])
        pheno = PhenotypeArray(samples=hap.samples)
        result = comp.compute(node, hap, pheno)
        expected = hap.genotypes[:, :, 0].astype(np.float64) @ eff.effects
        np.testing.assert_allclose(result, expected)

    def test_paternal_matches_manual(self):
        hap = _make_haplotypes()
        eff = _make_effects(m=hap.m)
        comp = HaplotypeGeneticComponent(eff, haplotype='paternal')
        node = ArchNode(outputs=['Y.pat'], component=comp, inputs=[])
        pheno = PhenotypeArray(samples=hap.samples)
        result = comp.compute(node, hap, pheno)
        expected = hap.genotypes[:, :, 1].astype(np.float64) @ eff.effects
        np.testing.assert_allclose(result, expected)

    def test_maternal_plus_paternal_equals_diploid(self):
        """maternal + paternal should equal diploid for non-standardized effects."""
        hap = _make_haplotypes()
        eff = _make_effects(m=hap.m)
        mat_comp = HaplotypeGeneticComponent(eff, haplotype='maternal')
        pat_comp = HaplotypeGeneticComponent(eff, haplotype='paternal')
        dip_comp = GeneticComponent(eff)
        node = ArchNode(outputs=['tmp'], component=mat_comp, inputs=[])
        pheno = PhenotypeArray(samples=hap.samples)
        mat = mat_comp.compute(node, hap, pheno)
        pat = pat_comp.compute(node, hap, pheno)
        dip = dip_comp.compute(node, hap, pheno)
        np.testing.assert_allclose(mat + pat, dip)

    def test_invalid_haplotype_raises(self):
        eff = _make_effects()
        with pytest.raises(ValueError, match="'maternal' or 'paternal'"):
            HaplotypeGeneticComponent(eff, haplotype='both')

    def test_parser_maternal_default(self):
        eff = _make_effects()
        nodes = parse_formula("Y.mat ~ haplotypeGenetic(eff)", effects={'eff': eff})
        assert len(nodes) == 1
        assert isinstance(nodes[0].component, HaplotypeGeneticComponent)
        assert nodes[0].component.haplotype == 'maternal'

    def test_parser_paternal_explicit(self):
        eff = _make_effects()
        nodes = parse_formula(
            "Y.pat ~ haplotypeGenetic(eff, haplotype='paternal')",
            effects={'eff': eff},
        )
        assert len(nodes) == 1
        assert nodes[0].component.haplotype == 'paternal'

    def test_parser_maternal_explicit(self):
        eff = _make_effects()
        nodes = parse_formula(
            "Y.mat ~ haplotypeGenetic(eff, haplotype='maternal')",
            effects={'eff': eff},
        )
        assert nodes[0].component.haplotype == 'maternal'

    def test_architecture_e2e(self):
        """Full architecture with haplotypeGenetic computes correctly."""
        hap = _make_haplotypes(n=200, m=30)
        eff = AdditiveEffects.from_h2(h2=0.5, m=30, seed=99, standardized=False)
        arch = Architecture(
            formula="Y.mat ~ haplotypeGenetic(eff)\nY.pat ~ haplotypeGenetic(eff, haplotype='paternal')\nY ~ Y.mat + Y.pat",
            effects={'eff': eff},
        )
        pheno = arch.compute(hap, rng=np.random.RandomState(0))
        expected_mat = hap.genotypes[:, :, 0].astype(np.float64) @ eff.effects
        expected_pat = hap.genotypes[:, :, 1].astype(np.float64) @ eff.effects
        np.testing.assert_allclose(pheno['Y.mat'], expected_mat)
        np.testing.assert_allclose(pheno['Y.pat'], expected_pat)
        np.testing.assert_allclose(pheno['Y'], expected_mat + expected_pat)
