"""
Unit tests for ArchNode dataclass and its properties.

Tests:
1. ArchNode construction with defaults
2. ArchNode repr
3. ArchNode with grouping
4. ArchNode with multi-output
5. BUILTINS registry completeness
"""
import numpy as np
import pytest

from xftsim.arch import (
    ArchNode, NoiseComponent, GeneticComponent, AggregationComponent,
    CNoiseComponent, MVGeneticComponent, HaplotypeGeneticComponent,
    MotherComponent, FatherComponent, ParentComponent,
    SiblingMeanComponent, SiblingSumComponent, SiblingAnyComponent,
    SiblingCountComponent, SiblingEldestComponent, SiblingYoungestComponent,
    BUILTINS,
)
from xftsim.effect import AdditiveEffects


class TestArchNodeConstruction:
    def test_default_inputs(self):
        node = ArchNode(outputs=['Y'], component=NoiseComponent(1.0))
        assert node.inputs == []
        assert node.grouping is None

    def test_with_inputs(self):
        node = ArchNode(
            outputs=['Y'],
            component=AggregationComponent('A + B'),
            inputs=['A', 'B'],
        )
        assert node.inputs == ['A', 'B']

    def test_with_grouping(self):
        node = ArchNode(
            outputs=['Y.E'],
            component=NoiseComponent(1.0),
            grouping='FID',
        )
        assert node.grouping == 'FID'

    def test_multi_output(self):
        cov = np.eye(2)
        node = ArchNode(
            outputs=['A', 'B'],
            component=CNoiseComponent(cov),
        )
        assert len(node.outputs) == 2


class TestArchNodeRepr:
    def test_repr(self):
        node = ArchNode(
            outputs=['Y'],
            component=NoiseComponent(1.0),
            inputs=[],
            grouping='FID',
        )
        r = repr(node)
        assert 'ArchNode' in r
        assert 'Y' in r
        assert 'FID' in r


class TestBuiltinsRegistry:
    def test_all_components_registered(self):
        expected = {
            'genetic', 'mvGenetic', 'haplotypeGenetic',
            'noise', 'cnoise',
            'parent', 'mother', 'father',
            'sibling_mean', 'sibling_sum', 'sibling_any',
            'sibling_count', 'sibling_eldest', 'sibling_youngest',
        }
        assert set(BUILTINS.keys()) == expected

    def test_builtins_are_classes(self):
        for name, cls in BUILTINS.items():
            assert isinstance(cls, type), f"{name} is not a class"

    def test_builtins_match_component_name(self):
        """Each component's .name should match its BUILTINS key."""
        for key, cls in BUILTINS.items():
            assert cls.name == key, f"BUILTINS['{key}'].name = '{cls.name}'"


class TestComponentProperties:
    def test_genetic_kind(self):
        assert GeneticComponent.kind == 'genetic'
        assert MVGeneticComponent.kind == 'genetic'
        assert HaplotypeGeneticComponent.kind == 'genetic'

    def test_generative_kind(self):
        assert NoiseComponent.kind == 'generative'
        assert CNoiseComponent.kind == 'generative'

    def test_aggregating_kind(self):
        assert AggregationComponent.kind == 'aggregating'

    def test_reference_kind(self):
        assert MotherComponent.kind == 'reference'
        assert FatherComponent.kind == 'reference'
        assert ParentComponent.kind == 'reference'

    def test_grouping_acceptance(self):
        assert NoiseComponent.accepts_grouping is True
        assert CNoiseComponent.accepts_grouping is True
        assert GeneticComponent.accepts_grouping is False
        assert AggregationComponent.accepts_grouping is False

    def test_sibling_kind(self):
        assert SiblingMeanComponent.kind == 'aggregating'
        assert SiblingMeanComponent.accepts_grouping is True
