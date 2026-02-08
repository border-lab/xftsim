"""
Integration test: Architecture save/load roundtrip with ALL component types.

Tests that every component type can be serialized and deserialized correctly.
"""
import numpy as np
import pytest
import tempfile
import os

from xftsim.narch import (
    Architecture, GeneticComponent, MVGeneticComponent, HaplotypeGeneticComponent,
    NoiseComponent, CNoiseComponent, AggregationComponent,
    MotherComponent, FatherComponent, ParentComponent,
    SiblingMeanComponent, SiblingSumComponent, SiblingAnyComponent,
    SiblingCountComponent, SiblingEldestComponent, SiblingYoungestComponent,
)
from xftsim.neffect import AdditiveEffects, MultivariateEffects
from xftsim.io import save_architecture, load_architecture


class TestAllComponentTypesRoundtrip:
    def test_genetic_component(self):
        """GeneticComponent roundtrip."""
        arch = Architecture()
        effects = AdditiveEffects.from_h2(m=10, h2=0.5, seed=42)
        arch.add('Y.G', GeneticComponent(effects))

        with tempfile.TemporaryDirectory() as tmpdir:
            d = os.path.join(tmpdir, 'arch')
            save_architecture(arch, d)
            loaded = load_architecture(d)
            assert len(loaded._nodes) == 1
            assert loaded._nodes[0].outputs == ['Y.G']
            assert isinstance(loaded._nodes[0].component, GeneticComponent)

    def test_mvgenetic_component(self):
        """MVGeneticComponent roundtrip."""
        arch = Architecture()
        effects = MultivariateEffects.from_h2_rg(m=10, h2=[0.5, 0.5], rg=0.3, seed=42)
        arch.add(['Y.G', 'Z.G'], MVGeneticComponent(effects))

        with tempfile.TemporaryDirectory() as tmpdir:
            d = os.path.join(tmpdir, 'arch')
            save_architecture(arch, d)
            loaded = load_architecture(d)
            assert loaded._nodes[0].outputs == ['Y.G', 'Z.G']
            assert isinstance(loaded._nodes[0].component, MVGeneticComponent)

    def test_haplotype_genetic_component(self):
        """HaplotypeGeneticComponent roundtrip with haplotype kwarg."""
        arch = Architecture()
        effects = AdditiveEffects.from_h2(m=10, h2=0.3, seed=42)
        arch.add('Y.HG', HaplotypeGeneticComponent(effects, haplotype='paternal'))

        with tempfile.TemporaryDirectory() as tmpdir:
            d = os.path.join(tmpdir, 'arch')
            save_architecture(arch, d)
            loaded = load_architecture(d)
            comp = loaded._nodes[0].component
            assert isinstance(comp, HaplotypeGeneticComponent)
            assert comp.haplotype == 'paternal'

    def test_noise_component(self):
        """NoiseComponent roundtrip."""
        arch = Architecture()
        arch.add('Y.E', NoiseComponent(variance=0.7))

        with tempfile.TemporaryDirectory() as tmpdir:
            d = os.path.join(tmpdir, 'arch')
            save_architecture(arch, d)
            loaded = load_architecture(d)
            comp = loaded._nodes[0].component
            assert isinstance(comp, NoiseComponent)
            assert comp.variance == 0.7

    def test_cnoise_component(self):
        """CNoiseComponent roundtrip."""
        arch = Architecture()
        cov = np.array([[1.0, 0.5], [0.5, 2.0]])
        arch.add(['A', 'B'], CNoiseComponent(cov=cov))

        with tempfile.TemporaryDirectory() as tmpdir:
            d = os.path.join(tmpdir, 'arch')
            save_architecture(arch, d)
            loaded = load_architecture(d)
            comp = loaded._nodes[0].component
            assert isinstance(comp, CNoiseComponent)
            np.testing.assert_allclose(comp.cov, cov)

    def test_aggregation_component(self):
        """AggregationComponent roundtrip."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('B', NoiseComponent(variance=1.0))
        arch.add('C', AggregationComponent('A + B'), inputs=['A', 'B'])

        with tempfile.TemporaryDirectory() as tmpdir:
            d = os.path.join(tmpdir, 'arch')
            save_architecture(arch, d)
            loaded = load_architecture(d)
            agg = loaded._nodes[2].component
            assert isinstance(agg, AggregationComponent)
            assert agg.expression == 'A + B'

    def test_mother_component(self):
        """MotherComponent roundtrip."""
        arch = Architecture()
        arch.add('Y.VT', MotherComponent('Y'))

        with tempfile.TemporaryDirectory() as tmpdir:
            d = os.path.join(tmpdir, 'arch')
            save_architecture(arch, d)
            loaded = load_architecture(d)
            comp = loaded._nodes[0].component
            assert isinstance(comp, MotherComponent)
            assert comp.phenotype_name == 'Y'

    def test_father_component(self):
        """FatherComponent roundtrip."""
        arch = Architecture()
        arch.add('Y.FVT', FatherComponent('Y'))

        with tempfile.TemporaryDirectory() as tmpdir:
            d = os.path.join(tmpdir, 'arch')
            save_architecture(arch, d)
            loaded = load_architecture(d)
            comp = loaded._nodes[0].component
            assert isinstance(comp, FatherComponent)
            assert comp.phenotype_name == 'Y'

    def test_parent_component(self):
        """ParentComponent roundtrip."""
        arch = Architecture()
        arch.add('Y.PVT', ParentComponent('Y'))

        with tempfile.TemporaryDirectory() as tmpdir:
            d = os.path.join(tmpdir, 'arch')
            save_architecture(arch, d)
            loaded = load_architecture(d)
            comp = loaded._nodes[0].component
            assert isinstance(comp, ParentComponent)
            assert comp.phenotype_name == 'Y'

    def test_sibling_mean_component(self):
        arch = Architecture()
        arch.add('Y.sib_mean', SiblingMeanComponent('Y'), inputs=['Y'])

        with tempfile.TemporaryDirectory() as tmpdir:
            d = os.path.join(tmpdir, 'arch')
            save_architecture(arch, d)
            loaded = load_architecture(d)
            comp = loaded._nodes[0].component
            assert isinstance(comp, SiblingMeanComponent)
            assert comp.source_name == 'Y'

    def test_sibling_sum_component(self):
        arch = Architecture()
        arch.add('Y.sib_sum', SiblingSumComponent('Y'), inputs=['Y'])

        with tempfile.TemporaryDirectory() as tmpdir:
            d = os.path.join(tmpdir, 'arch')
            save_architecture(arch, d)
            loaded = load_architecture(d)
            assert isinstance(loaded._nodes[0].component, SiblingSumComponent)

    def test_sibling_any_component(self):
        arch = Architecture()
        arch.add('Y.sib_any', SiblingAnyComponent('Y'), inputs=['Y'])

        with tempfile.TemporaryDirectory() as tmpdir:
            d = os.path.join(tmpdir, 'arch')
            save_architecture(arch, d)
            loaded = load_architecture(d)
            assert isinstance(loaded._nodes[0].component, SiblingAnyComponent)

    def test_sibling_count_component(self):
        arch = Architecture()
        arch.add('Y.sib_count', SiblingCountComponent('Y'), inputs=['Y'])

        with tempfile.TemporaryDirectory() as tmpdir:
            d = os.path.join(tmpdir, 'arch')
            save_architecture(arch, d)
            loaded = load_architecture(d)
            assert isinstance(loaded._nodes[0].component, SiblingCountComponent)

    def test_sibling_eldest_component(self):
        arch = Architecture()
        arch.add('Y.eldest', SiblingEldestComponent('Y'), inputs=['Y'])

        with tempfile.TemporaryDirectory() as tmpdir:
            d = os.path.join(tmpdir, 'arch')
            save_architecture(arch, d)
            loaded = load_architecture(d)
            assert isinstance(loaded._nodes[0].component, SiblingEldestComponent)

    def test_sibling_youngest_component(self):
        arch = Architecture()
        arch.add('Y.youngest', SiblingYoungestComponent('Y'), inputs=['Y'])

        with tempfile.TemporaryDirectory() as tmpdir:
            d = os.path.join(tmpdir, 'arch')
            save_architecture(arch, d)
            loaded = load_architecture(d)
            assert isinstance(loaded._nodes[0].component, SiblingYoungestComponent)


class TestComplexArchitectureRoundtrip:
    def test_multi_component_architecture(self):
        """Save/load architecture with many different component types."""
        arch = Architecture()
        effects = AdditiveEffects.from_h2(m=10, h2=0.5, seed=42)
        mv_effects = MultivariateEffects.from_h2_rg(m=10, h2=[0.3, 0.3], rg=0.5, seed=42)

        arch.add('Y.G', GeneticComponent(effects))
        arch.add(['A.G', 'B.G'], MVGeneticComponent(mv_effects))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y.VT', MotherComponent('Y'))
        arch.add('Y', AggregationComponent('Y.G + Y.E + Y.VT'),
                 inputs=['Y.G', 'Y.E', 'Y.VT'])

        with tempfile.TemporaryDirectory() as tmpdir:
            d = os.path.join(tmpdir, 'arch')
            save_architecture(arch, d)
            loaded = load_architecture(d)

            assert len(loaded._nodes) == 5
            types = [type(n.component).__name__ for n in loaded._nodes]
            assert 'GeneticComponent' in types
            assert 'MVGeneticComponent' in types
            assert 'NoiseComponent' in types
            assert 'MotherComponent' in types
            assert 'AggregationComponent' in types

    def test_grouping_preserved(self):
        """Node grouping should survive roundtrip."""
        arch = Architecture()
        arch.add('Y.E', NoiseComponent(variance=1.0), grouping='FID')

        with tempfile.TemporaryDirectory() as tmpdir:
            d = os.path.join(tmpdir, 'arch')
            save_architecture(arch, d)
            loaded = load_architecture(d)
            assert loaded._nodes[0].grouping == 'FID'

    def test_inputs_preserved(self):
        """Node inputs should survive roundtrip."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('B', AggregationComponent('A * 2'), inputs=['A'])

        with tempfile.TemporaryDirectory() as tmpdir:
            d = os.path.join(tmpdir, 'arch')
            save_architecture(arch, d)
            loaded = load_architecture(d)
            assert loaded._nodes[1].inputs == ['A']
