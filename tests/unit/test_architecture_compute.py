"""
Unit tests for Architecture.compute() and related methods.

Tests:
1. compute auto-creates NPhenotypeArray
2. compute with explicit phenotypes
3. compute with multi-output (MVGenetic)
4. compute propagates values through DAG
5. compute with None rng still works
6. Architecture repr
7. ArchNode repr
8. from_formula + compute end-to-end
9. BUILTINS registry contains expected keys
10. Architecture with grouping parameter
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray, NPhenotypeArray
from xftsim.narch import (
    Architecture, ArchNode, GeneticComponent, MVGeneticComponent,
    NoiseComponent, CNoiseComponent, AggregationComponent,
    BUILTINS, HaplotypeGeneticComponent,
    MotherComponent, FatherComponent, ParentComponent,
    SiblingMeanComponent, SiblingSumComponent, SiblingAnyComponent,
    SiblingCountComponent, SiblingEldestComponent, SiblingYoungestComponent,
)
from xftsim.neffect import AdditiveEffects, MultivariateEffects

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


def _make_hap(n=20, m=10, seed=42):
    return TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)


class TestArchitectureCompute:
    def test_auto_creates_phenotypes(self):
        """compute with phenotypes=None should create NPhenotypeArray."""
        hap = _make_hap()
        arch = Architecture()
        arch.add('Y.E', NoiseComponent(variance=1.0))
        result = arch.compute(hap, rng=np.random.RandomState(42))
        assert isinstance(result, NPhenotypeArray)
        assert 'Y.E' in result._values

    def test_with_explicit_phenotypes(self):
        """compute with explicit phenotypes should write to that object."""
        hap = _make_hap()
        arch = Architecture()
        arch.add('Y.E', NoiseComponent(variance=1.0))
        pheno = NPhenotypeArray(samples=hap.samples)
        result = arch.compute(hap, phenotypes=pheno, rng=np.random.RandomState(42))
        assert result is pheno
        assert 'Y.E' in pheno._values

    def test_none_rng(self):
        """compute with rng=None should still work."""
        hap = _make_hap()
        arch = Architecture()
        arch.add('Y.E', NoiseComponent(variance=1.0))
        result = arch.compute(hap)
        assert 'Y.E' in result._values

    def test_dag_propagation(self):
        """Values should propagate through the DAG: G + E → Y."""
        hap = _make_hap(n=20, m=10)
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        result = arch.compute(hap, rng=np.random.RandomState(42))
        expected = result['Y.G'] + result['Y.E']
        np.testing.assert_allclose(result['Y'], expected)

    def test_multi_output_node(self):
        """MVGenetic with k=2 should write two output columns."""
        hap = _make_hap(n=20, m=10)
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.4, m=10, seed=42)
        arch = Architecture()
        arch.add(['Y1.G', 'Y2.G'], MVGeneticComponent(eff))
        result = arch.compute(hap, rng=np.random.RandomState(42))
        assert 'Y1.G' in result._values
        assert 'Y2.G' in result._values
        assert result['Y1.G'].shape == (20,)
        assert result['Y2.G'].shape == (20,)

    def test_deterministic_with_same_rng(self):
        """Same rng seed should produce identical results."""
        hap = _make_hap()
        arch = Architecture()
        arch.add('Y.E', NoiseComponent(variance=1.0))
        r1 = arch.compute(hap, rng=np.random.RandomState(42))
        r2 = arch.compute(hap, rng=np.random.RandomState(42))
        np.testing.assert_array_equal(r1['Y.E'], r2['Y.E'])

    def test_compute_all_values_finite(self):
        """All computed values should be finite."""
        hap = _make_hap(n=50, m=10)
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        result = arch.compute(hap, rng=np.random.RandomState(42))
        for key in ['Y.G', 'Y.E', 'Y']:
            assert np.all(np.isfinite(result[key]))


class TestFromFormulaCompute:
    def test_from_formula_compute(self):
        """Architecture built from formula should compute correctly."""
        hap = _make_hap(n=50, m=10)
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        arch = Architecture.from_formula("""
            Y.G ~ genetic(eff1)
            Y.E ~ noise(0.5)
            Y ~ Y.G + Y.E
        """, effects={'eff1': eff})
        result = arch.compute(hap, rng=np.random.RandomState(42))
        assert 'Y' in result._values
        assert np.all(np.isfinite(result['Y']))

    def test_from_formula_multi_trait(self):
        """Multi-trait formula should compute correctly."""
        hap = _make_hap(n=50, m=10)
        eff1 = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        eff2 = AdditiveEffects.from_h2(h2=0.3, m=10, seed=43)
        arch = Architecture.from_formula("""
            Y1.G ~ genetic(e1)
            Y1.E ~ noise(0.5)
            Y1 ~ Y1.G + Y1.E
            Y2.G ~ genetic(e2)
            Y2.E ~ noise(0.7)
            Y2 ~ Y2.G + Y2.E
        """, effects={'e1': eff1, 'e2': eff2})
        result = arch.compute(hap, rng=np.random.RandomState(42))
        assert 'Y1' in result._values
        assert 'Y2' in result._values


class TestArchitectureRepr:
    def test_repr_empty(self):
        """Empty architecture repr."""
        arch = Architecture()
        r = repr(arch)
        assert 'Architecture' in r
        assert 'nodes=0' in r

    def test_repr_with_nodes(self):
        """Architecture with nodes shows outputs."""
        arch = Architecture()
        arch.add('Y.G', NoiseComponent(variance=1.0))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        r = repr(arch)
        assert 'nodes=2' in r
        assert 'Y.G' in r
        assert 'Y.E' in r


class TestArchNodeRepr:
    def test_archnode_repr(self):
        """ArchNode repr shows all fields."""
        node = ArchNode(outputs=['Y'], component=NoiseComponent(variance=1.0),
                        inputs=[], grouping=None)
        r = repr(node)
        assert 'ArchNode' in r
        assert 'Y' in r

    def test_archnode_repr_with_grouping(self):
        """ArchNode with grouping shows it in repr."""
        node = ArchNode(outputs=['Y.sib'], component=SiblingMeanComponent('Y'),
                        inputs=['Y'], grouping='FID')
        r = repr(node)
        assert 'FID' in r


class TestBuiltinsRegistry:
    def test_expected_keys(self):
        """BUILTINS should contain all expected component types."""
        expected = [
            'genetic', 'mvGenetic', 'haplotypeGenetic',
            'noise', 'cnoise',
            'parent', 'mother', 'father',
            'sibling_mean', 'sibling_sum', 'sibling_any',
            'sibling_count', 'sibling_eldest', 'sibling_youngest',
        ]
        for key in expected:
            assert key in BUILTINS, f"Missing BUILTIN: {key}"

    def test_builtin_values_are_classes(self):
        """All BUILTIN values should be classes."""
        for key, val in BUILTINS.items():
            assert isinstance(val, type), f"{key} is not a class"

    def test_builtin_count(self):
        """BUILTINS should have exactly 14 entries."""
        assert len(BUILTINS) == 14


class TestArchitectureAdd:
    def test_add_with_list_outputs(self):
        """add() should accept list of output names."""
        arch = Architecture()
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.4, m=5, seed=42)
        arch.add(['Y1.G', 'Y2.G'], MVGeneticComponent(eff))
        assert len(arch.nodes) == 1
        assert arch.nodes[0].outputs == ['Y1.G', 'Y2.G']

    def test_add_with_grouping(self):
        """add() with grouping parameter."""
        arch = Architecture()
        arch.add('Y', NoiseComponent(variance=1.0), grouping='FID')
        assert arch.nodes[0].grouping == 'FID'

    def test_add_auto_detects_agg_inputs(self):
        """add() with AggregationComponent auto-detects inputs."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('B', NoiseComponent(variance=1.0))
        arch.add('Y', AggregationComponent('A + B'))
        y_node = [n for n in arch.nodes if n.outputs == ['Y']][0]
        assert 'A' in y_node.inputs
        assert 'B' in y_node.inputs

    def test_add_explicit_inputs(self):
        """add() with explicit inputs overrides auto-detection."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        # Pass explicit inputs
        arch.add('Y', AggregationComponent('A'), inputs=['A'])
        y_node = [n for n in arch.nodes if n.outputs == ['Y']][0]
        assert y_node.inputs == ['A']
