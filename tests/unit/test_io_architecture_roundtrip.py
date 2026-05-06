"""
Unit tests for architecture serialization roundtrip with all component types.

Tests:
1. HaplotypeGeneticComponent roundtrip (maternal/paternal)
2. Architecture with all component types in one roundtrip
3. Sibling component roundtrips (all 6 types)
4. Unknown component type raises ValueError
5. Architecture with grouped noise roundtrip
6. _serialize/_deserialize mating regime roundtrip
"""
import numpy as np
import pytest
import json
import os

from xftsim.io import (
    save_architecture, load_architecture,
    _serialize_mating_regime, _deserialize_mating_regime,
)
from xftsim.arch import (
    Architecture, GeneticComponent, MVGeneticComponent,
    HaplotypeGeneticComponent, NoiseComponent, CNoiseComponent,
    AggregationComponent, MotherComponent, FatherComponent,
    SiblingMeanComponent, SiblingSumComponent, SiblingCountComponent,
    SiblingEldestComponent, SiblingYoungestComponent, SiblingAnyComponent,
)
from xftsim.effect import AdditiveEffects, MultivariateEffects
from xftsim.mate import RandomMating, LinearAssortativeMating


class TestHaplotypeGeneticRoundtrip:
    def test_maternal_roundtrip(self, tmp_path):
        """HaplotypeGeneticComponent(maternal) should survive save/load."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        arch = Architecture()
        arch.add('Y.Gmat', HaplotypeGeneticComponent(eff, haplotype='maternal'))
        dir_path = str(tmp_path / 'arch')
        save_architecture(arch, dir_path)
        loaded = load_architecture(dir_path)
        node = loaded._nodes[0]
        assert isinstance(node.component, HaplotypeGeneticComponent)
        assert node.component.haplotype == 'maternal'
        np.testing.assert_array_equal(node.component.effects.effects, eff.effects)

    def test_paternal_roundtrip(self, tmp_path):
        """HaplotypeGeneticComponent(paternal) should survive save/load."""
        eff = AdditiveEffects.from_h2(h2=0.3, m=15, seed=0)
        arch = Architecture()
        arch.add('Y.Gpat', HaplotypeGeneticComponent(eff, haplotype='paternal'))
        dir_path = str(tmp_path / 'arch')
        save_architecture(arch, dir_path)
        loaded = load_architecture(dir_path)
        node = loaded._nodes[0]
        assert isinstance(node.component, HaplotypeGeneticComponent)
        assert node.component.haplotype == 'paternal'

    def test_maternal_paternal_together(self, tmp_path):
        """Both maternal and paternal in same architecture."""
        eff = AdditiveEffects.from_h2(h2=0.4, m=8, seed=42)
        arch = Architecture()
        arch.add('Y.Gmat', HaplotypeGeneticComponent(eff, haplotype='maternal'))
        arch.add('Y.Gpat', HaplotypeGeneticComponent(eff, haplotype='paternal'))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.Gmat + Y.Gpat + Y.E'))
        dir_path = str(tmp_path / 'arch')
        save_architecture(arch, dir_path)
        loaded = load_architecture(dir_path)
        assert len(loaded._nodes) == 4
        mat_node = loaded._nodes[0]
        pat_node = loaded._nodes[1]
        assert mat_node.component.haplotype == 'maternal'
        assert pat_node.component.haplotype == 'paternal'


class TestAllSiblingComponentsRoundtrip:
    @pytest.mark.parametrize("cls,name", [
        (SiblingMeanComponent, 'SiblingMeanComponent'),
        (SiblingSumComponent, 'SiblingSumComponent'),
        (SiblingCountComponent, 'SiblingCountComponent'),
        (SiblingEldestComponent, 'SiblingEldestComponent'),
        (SiblingYoungestComponent, 'SiblingYoungestComponent'),
        (SiblingAnyComponent, 'SiblingAnyComponent'),
    ])
    def test_sibling_roundtrip(self, tmp_path, cls, name):
        """Each sibling component type should survive save/load."""
        arch = Architecture()
        arch.add('Y.sib', cls('Y'), inputs=['Y'], grouping='FID')
        dir_path = str(tmp_path / 'arch')
        save_architecture(arch, dir_path)
        loaded = load_architecture(dir_path)
        node = loaded._nodes[0]
        assert type(node.component).__name__ == name
        assert node.component.source_name == 'Y'
        assert node.grouping == 'FID'


class TestAllComponentTypesRoundtrip:
    def test_full_architecture_roundtrip(self, tmp_path):
        """Architecture with every component type should roundtrip."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        mv_eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.2, m=10, seed=42)
        cov = np.array([[0.5, 0.1], [0.1, 0.5]])

        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add(('T1.G', 'T2.G'), MVGeneticComponent(mv_eff))
        arch.add('Y.Gmat', HaplotypeGeneticComponent(eff, haplotype='maternal'))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add(('T1.E', 'T2.E'), CNoiseComponent(cov))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        arch.add('Y.mother', MotherComponent('Y'))
        arch.add('Y.father', FatherComponent('Y'))
        arch.add('Y.sib', SiblingMeanComponent('Y'), inputs=['Y'], grouping='FID')

        dir_path = str(tmp_path / 'arch')
        save_architecture(arch, dir_path)
        loaded = load_architecture(dir_path)
        assert len(loaded._nodes) == 9

        # Verify types preserved
        types = [type(n.component).__name__ for n in loaded._nodes]
        assert 'GeneticComponent' in types
        assert 'MVGeneticComponent' in types
        assert 'HaplotypeGeneticComponent' in types
        assert 'NoiseComponent' in types
        assert 'CNoiseComponent' in types
        assert 'AggregationComponent' in types
        assert 'MotherComponent' in types
        assert 'FatherComponent' in types
        assert 'SiblingMeanComponent' in types


class TestUnknownComponentType:
    def test_unknown_component_type_raises(self, tmp_path):
        """Loading architecture with unknown component type should raise."""
        dir_path = str(tmp_path / 'arch')
        os.makedirs(dir_path)
        node_specs = [{
            'outputs': ['X'],
            'inputs': [],
            'grouping': None,
            'component_type': 'FooBarComponent',
        }]
        with open(os.path.join(dir_path, 'architecture.json'), 'w') as f:
            json.dump(node_specs, f)
        with pytest.raises(ValueError, match="Unknown component type"):
            load_architecture(dir_path)


class TestGroupedComponentRoundtrip:
    def test_grouped_noise_roundtrip(self, tmp_path):
        """Noise with FID grouping should preserve grouping through save/load."""
        arch = Architecture()
        arch.add('Y.E', NoiseComponent(variance=1.0), grouping='FID')
        dir_path = str(tmp_path / 'arch')
        save_architecture(arch, dir_path)
        loaded = load_architecture(dir_path)
        assert loaded._nodes[0].grouping == 'FID'

    def test_no_grouping_roundtrip(self, tmp_path):
        """None grouping should be preserved."""
        arch = Architecture()
        arch.add('Y.E', NoiseComponent(variance=1.0))
        dir_path = str(tmp_path / 'arch')
        save_architecture(arch, dir_path)
        loaded = load_architecture(dir_path)
        assert loaded._nodes[0].grouping is None


class TestMatingRegimeSerialize:
    def test_random_mating_roundtrip(self):
        """RandomMating serialize/deserialize."""
        mate = RandomMating(offspring_per_pair=3)
        d = _serialize_mating_regime(mate)
        loaded = _deserialize_mating_regime(d)
        assert isinstance(loaded, RandomMating)
        assert loaded.offspring_per_pair == 3

    def test_assortative_mating_roundtrip(self):
        """LinearAssortativeMating serialize/deserialize."""
        mate = LinearAssortativeMating(
            component_names=['Y', 'X'], r=0.6, offspring_per_pair=2)
        d = _serialize_mating_regime(mate)
        loaded = _deserialize_mating_regime(d)
        assert isinstance(loaded, LinearAssortativeMating)
        assert loaded.r == 0.6
        assert loaded.component_names == ['Y', 'X']
        assert loaded.offspring_per_pair == 2

    def test_unknown_mating_type_raises(self):
        """Unknown mating regime type should raise."""
        with pytest.raises(ValueError, match="Unknown mating regime"):
            _deserialize_mating_regime({'type': 'WeirdMating'})
