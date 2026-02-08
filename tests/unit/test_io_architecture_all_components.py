"""
Unit tests for architecture I/O with all component types.

Tests save_architecture + load_architecture roundtrip for:
1. GeneticComponent
2. NoiseComponent
3. CNoiseComponent
4. AggregationComponent
5. HaplotypeGeneticComponent
6. MotherComponent
7. SiblingMeanComponent
8. Mixed architecture with multiple component types
"""
import numpy as np
import pytest
import tempfile
import os

from xftsim.narch import (
    Architecture, GeneticComponent, MVGeneticComponent,
    HaplotypeGeneticComponent, NoiseComponent, CNoiseComponent,
    AggregationComponent, MotherComponent, FatherComponent,
    SiblingMeanComponent,
)
from xftsim.neffect import AdditiveEffects, MultivariateEffects
from xftsim.io import save_architecture, load_architecture


class TestArchitectureIOAllComponents:
    def test_genetic_roundtrip(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))

        with tempfile.TemporaryDirectory() as d:
            save_architecture(arch, d)
            loaded = load_architecture(d)
            assert len(loaded._nodes) == 1
            assert loaded._nodes[0].outputs == ['Y.G']
            comp = loaded._nodes[0].component
            assert isinstance(comp, GeneticComponent)
            np.testing.assert_allclose(comp.effects.effects, eff.effects)

    def test_noise_roundtrip(self):
        arch = Architecture()
        arch.add('E', NoiseComponent(variance=2.5))

        with tempfile.TemporaryDirectory() as d:
            save_architecture(arch, d)
            loaded = load_architecture(d)
            comp = loaded._nodes[0].component
            assert isinstance(comp, NoiseComponent)
            assert comp.variance == 2.5

    def test_cnoise_roundtrip(self):
        cov = np.array([[1.0, 0.3], [0.3, 1.0]])
        arch = Architecture()
        arch.add(['E1', 'E2'], CNoiseComponent(cov=cov))

        with tempfile.TemporaryDirectory() as d:
            save_architecture(arch, d)
            loaded = load_architecture(d)
            comp = loaded._nodes[0].component
            assert isinstance(comp, CNoiseComponent)
            np.testing.assert_allclose(comp.cov, cov)

    def test_aggregation_roundtrip(self):
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('B', NoiseComponent(variance=1.0))
        arch.add('Y', AggregationComponent('0.5 * A + 0.5 * B'), inputs=['A', 'B'])

        with tempfile.TemporaryDirectory() as d:
            save_architecture(arch, d)
            loaded = load_architecture(d)
            agg_node = loaded._nodes[2]
            assert isinstance(agg_node.component, AggregationComponent)
            assert agg_node.component.expression == '0.5 * A + 0.5 * B'
            assert agg_node.inputs == ['A', 'B']

    def test_haplotype_genetic_roundtrip(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        arch = Architecture()
        arch.add('Y.mat', HaplotypeGeneticComponent(eff, haplotype='maternal'))
        arch.add('Y.pat', HaplotypeGeneticComponent(eff, haplotype='paternal'))

        with tempfile.TemporaryDirectory() as d:
            save_architecture(arch, d)
            loaded = load_architecture(d)
            comp0 = loaded._nodes[0].component
            comp1 = loaded._nodes[1].component
            assert isinstance(comp0, HaplotypeGeneticComponent)
            assert comp0.haplotype == 'maternal'
            assert comp1.haplotype == 'paternal'

    def test_mother_component_roundtrip(self):
        arch = Architecture()
        arch.add('Y.m', MotherComponent('Y'))

        with tempfile.TemporaryDirectory() as d:
            save_architecture(arch, d)
            loaded = load_architecture(d)
            comp = loaded._nodes[0].component
            assert isinstance(comp, MotherComponent)
            assert comp.phenotype_name == 'Y'

    def test_sibling_component_roundtrip(self):
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('Y.sib', SiblingMeanComponent('A'), inputs=['A'], grouping='FID')

        with tempfile.TemporaryDirectory() as d:
            save_architecture(arch, d)
            loaded = load_architecture(d)
            sib_node = loaded._nodes[1]
            assert isinstance(sib_node.component, SiblingMeanComponent)
            assert sib_node.component.source_name == 'A'
            assert sib_node.grouping == 'FID'

    def test_mixed_architecture(self):
        """Architecture with many component types survives roundtrip."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])
        arch.add('Y.m', MotherComponent('Y'))

        with tempfile.TemporaryDirectory() as d:
            save_architecture(arch, d)
            loaded = load_architecture(d)
            assert len(loaded._nodes) == 4
            types = [type(n.component).__name__ for n in loaded._nodes]
            assert 'GeneticComponent' in types
            assert 'NoiseComponent' in types
            assert 'AggregationComponent' in types
            assert 'MotherComponent' in types
