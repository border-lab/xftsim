"""
Integration tests for Architecture save/load with various component types.

Tests:
1. Simple G + E architecture roundtrip
2. Architecture with MVGenetic roundtrip
3. Architecture with HaplotypeGenetic roundtrip
4. Architecture with CNoiseComponent roundtrip
5. Architecture with parental (Mother/Father) roundtrip
6. Architecture with sibling components roundtrip
7. Loaded architecture produces same phenotypes as original
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray, PhenotypeArray
from xftsim.arch import (
    Architecture, GeneticComponent, MVGeneticComponent, HaplotypeGeneticComponent,
    NoiseComponent, CNoiseComponent, AggregationComponent,
    MotherComponent, FatherComponent, ParentComponent,
    SiblingMeanComponent, SiblingCountComponent,
)
from xftsim.effect import AdditiveEffects, MultivariateEffects
from xftsim.io import save_architecture, load_architecture

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


def _make_hap(n=20, m=10, seed=42):
    return TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)


class TestSimpleArchitectureRoundtrip:
    def test_genetic_noise_agg(self, tmp_path):
        """G + E → Y architecture roundtrip."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))

        save_architecture(arch, str(tmp_path / 'arch'))
        arch2 = load_architecture(str(tmp_path / 'arch'))

        assert len(arch2.nodes) == 3
        output_names = [n.outputs[0] for n in arch2.nodes]
        assert 'Y.G' in output_names
        assert 'Y.E' in output_names
        assert 'Y' in output_names

    def test_loaded_produces_same_phenotypes(self, tmp_path):
        """Loaded architecture should produce same phenotypes."""
        hap = _make_hap()
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))

        save_architecture(arch, str(tmp_path / 'arch'))
        arch2 = load_architecture(str(tmp_path / 'arch'))

        rng1 = np.random.RandomState(99)
        rng2 = np.random.RandomState(99)
        p1 = arch.compute(hap, rng=rng1)
        p2 = arch2.compute(hap, rng=rng2)

        np.testing.assert_allclose(p1['Y.G'], p2['Y.G'])
        np.testing.assert_allclose(p1['Y.E'], p2['Y.E'])
        np.testing.assert_allclose(p1['Y'], p2['Y'])


class TestMVGeneticRoundtrip:
    def test_mvgenetic(self, tmp_path):
        """Architecture with MVGeneticComponent roundtrip."""
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.4, m=10, seed=42)
        arch = Architecture()
        arch.add(['Y1.G', 'Y2.G'], MVGeneticComponent(eff))

        save_architecture(arch, str(tmp_path / 'arch'))
        arch2 = load_architecture(str(tmp_path / 'arch'))

        assert len(arch2.nodes) == 1
        assert arch2.nodes[0].outputs == ['Y1.G', 'Y2.G']


class TestHaplotypeGeneticRoundtrip:
    def test_haplotype_genetic(self, tmp_path):
        """Architecture with HaplotypeGeneticComponent roundtrip."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        arch = Architecture()
        arch.add('Y.Gmat', HaplotypeGeneticComponent(eff, haplotype='maternal'))
        arch.add('Y.Gpat', HaplotypeGeneticComponent(eff, haplotype='paternal'))

        save_architecture(arch, str(tmp_path / 'arch'))
        arch2 = load_architecture(str(tmp_path / 'arch'))

        assert len(arch2.nodes) == 2
        # Check haplotype attribute preserved
        comps = {n.outputs[0]: n.component for n in arch2.nodes}
        assert comps['Y.Gmat'].haplotype == 'maternal'
        assert comps['Y.Gpat'].haplotype == 'paternal'


class TestCNoiseRoundtrip:
    def test_cnoise(self, tmp_path):
        """Architecture with CNoiseComponent roundtrip."""
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        arch = Architecture()
        arch.add(['A', 'B'], CNoiseComponent(cov=cov))

        save_architecture(arch, str(tmp_path / 'arch'))
        arch2 = load_architecture(str(tmp_path / 'arch'))

        loaded_comp = arch2.nodes[0].component
        np.testing.assert_allclose(loaded_comp.cov, cov)


class TestParentalRoundtrip:
    def test_mother_father(self, tmp_path):
        """Architecture with Mother/Father components roundtrip."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=5, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        arch.add('Y.m', MotherComponent('Y'))
        arch.add('Y.f', FatherComponent('Y'))
        arch.add('Y.p', ParentComponent('Y'))

        save_architecture(arch, str(tmp_path / 'arch'))
        arch2 = load_architecture(str(tmp_path / 'arch'))

        comps = {n.outputs[0]: n.component for n in arch2._nodes}
        assert isinstance(comps['Y.m'], MotherComponent)
        assert isinstance(comps['Y.f'], FatherComponent)
        assert isinstance(comps['Y.p'], ParentComponent)
        assert comps['Y.m'].phenotype_name == 'Y'


class TestSiblingRoundtrip:
    def test_sibling_components(self, tmp_path):
        """Architecture with sibling components roundtrip."""
        arch = Architecture()
        arch.add('Y', NoiseComponent(variance=1.0))
        arch.add('Y.sib_mean', SiblingMeanComponent('Y'))
        arch.add('Y.sib_count', SiblingCountComponent('Y'))

        save_architecture(arch, str(tmp_path / 'arch'))
        arch2 = load_architecture(str(tmp_path / 'arch'))

        comps = {n.outputs[0]: n.component for n in arch2._nodes}
        assert isinstance(comps['Y.sib_mean'], SiblingMeanComponent)
        assert isinstance(comps['Y.sib_count'], SiblingCountComponent)
        assert comps['Y.sib_mean'].source_name == 'Y'
