"""
Unit tests for Architecture, ArchNode, toposort, and execution.
"""
import numpy as np
import pytest
from xftsim.narch import (
    Architecture, ArchNode, GeneticComponent, NoiseComponent,
    AggregationComponent,
)
from xftsim.neffect import AdditiveEffects
from xftsim.struct import DenseHaplotypeArray, NPhenotypeArray, SampleMeta


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def haplotypes(rng):
    geno = rng.randint(0, 2, size=(50, 20, 2)).astype(np.int8)
    return DenseHaplotypeArray(genotypes=geno)


@pytest.fixture
def effects():
    return AdditiveEffects.from_h2(h2=0.5, m=20, seed=42)


# ── Programmatic API ────────────────────────────────────────────────────────

class TestProgrammaticAPI:
    def test_add_single_node(self, effects):
        arch = Architecture()
        arch.add('height.G', GeneticComponent(effects=effects))
        assert len(arch._nodes) == 1
        assert arch._nodes[0].outputs == ['height.G']

    def test_add_multiple_nodes(self, effects):
        arch = Architecture()
        arch.add('height.G', GeneticComponent(effects=effects))
        arch.add('height.E', NoiseComponent(variance=0.5))
        arch.add('height', AggregationComponent('height.G + height.E'))
        assert len(arch._nodes) == 3

    def test_duplicate_output_error(self, effects):
        arch = Architecture()
        arch.add('x', GeneticComponent(effects=effects))
        with pytest.raises(ValueError, match="Duplicate output"):
            arch.add('x', NoiseComponent(variance=0.5))

    def test_aggregation_auto_inputs(self):
        arch = Architecture()
        arch.add('a', NoiseComponent(variance=1.0))
        arch.add('b', NoiseComponent(variance=1.0))
        arch.add('c', AggregationComponent('a + b'))
        node_c = arch._nodes[2]
        assert set(node_c.inputs) == {'a', 'b'}

    def test_list_outputs(self, effects):
        arch = Architecture()
        arch.add(['x', 'y'], NoiseComponent(variance=1.0))
        assert arch._nodes[0].outputs == ['x', 'y']


# ── Parsed API ──────────────────────────────────────────────────────────────

class TestParsedAPI:
    def test_parsed_produces_correct_nodes(self, effects):
        arch = Architecture("""
            height.G ~ genetic(eff)
            height.E ~ noise(0.5)
            height ~ height.G + height.E
        """, effects={'eff': effects})
        assert len(arch._nodes) == 3
        assert isinstance(arch._nodes[0].component, GeneticComponent)
        assert isinstance(arch._nodes[1].component, NoiseComponent)
        assert isinstance(arch._nodes[2].component, AggregationComponent)

    def test_parsed_matches_programmatic(self, effects, haplotypes):
        """Programmatic and parsed should produce same results with same rng."""
        # Programmatic
        arch1 = Architecture()
        arch1.add('height.G', GeneticComponent(effects=effects))
        arch1.add('height.E', NoiseComponent(variance=0.5))
        arch1.add('height', AggregationComponent('height.G + height.E'))
        p1 = arch1.compute(haplotypes, rng=np.random.RandomState(99))

        # Parsed
        arch2 = Architecture("""
            height.G ~ genetic(eff)
            height.E ~ noise(0.5)
            height ~ height.G + height.E
        """, effects={'eff': effects})
        p2 = arch2.compute(haplotypes, rng=np.random.RandomState(99))

        np.testing.assert_allclose(p1['height.G'], p2['height.G'])
        np.testing.assert_allclose(p1['height.E'], p2['height.E'])
        np.testing.assert_allclose(p1['height'], p2['height'])


# ── Toposort ────────────────────────────────────────────────────────────────

class TestToposort:
    def test_dependencies_before_dependents(self, effects):
        arch = Architecture()
        # Deliberately add aggregation first
        arch.add('height', AggregationComponent('height.G + height.E'))
        arch.add('height.G', GeneticComponent(effects=effects))
        arch.add('height.E', NoiseComponent(variance=0.5))
        sorted_nodes = arch.nodes
        # height.G and height.E must come before height
        output_order = [n.outputs[0] for n in sorted_nodes]
        assert output_order.index('height.G') < output_order.index('height')
        assert output_order.index('height.E') < output_order.index('height')

    def test_diamond_dependency(self):
        arch = Architecture()
        arch.add('a', NoiseComponent(variance=1.0))
        arch.add('b', AggregationComponent('a'))
        arch.add('c', AggregationComponent('a'))
        arch.add('d', AggregationComponent('b + c'))
        sorted_nodes = arch.nodes
        outputs = [n.outputs[0] for n in sorted_nodes]
        assert outputs.index('a') < outputs.index('b')
        assert outputs.index('a') < outputs.index('c')
        assert outputs.index('b') < outputs.index('d')
        assert outputs.index('c') < outputs.index('d')

    def test_undefined_reference(self):
        arch = Architecture()
        arch.add('x', AggregationComponent('undefined_var'))
        with pytest.raises(ValueError, match="Undefined reference"):
            arch.nodes

    def test_cycle_detection(self):
        """Cycles should be detected and raise ValueError."""
        arch = Architecture()
        # We can't really create a cycle through AggregationComponent
        # because inputs are extracted at creation time, but we can
        # force a cycle by manually creating nodes.
        from xftsim.narch import ArchNode
        node_a = ArchNode(outputs=['a'], component=NoiseComponent(variance=1.0),
                          inputs=['b'])
        node_b = ArchNode(outputs=['b'], component=NoiseComponent(variance=1.0),
                          inputs=['a'])
        arch._register_node(node_a)
        arch._register_node(node_b)
        with pytest.raises(ValueError, match="Cycle detected"):
            arch.nodes


# ── Execution ───────────────────────────────────────────────────────────────

class TestExecution:
    def test_compute_writes_phenotypes(self, effects, haplotypes):
        arch = Architecture()
        arch.add('height.G', GeneticComponent(effects=effects))
        arch.add('height.E', NoiseComponent(variance=0.5))
        arch.add('height', AggregationComponent('height.G + height.E'))
        result = arch.compute(haplotypes, rng=np.random.RandomState(42))
        assert isinstance(result, NPhenotypeArray)
        assert 'height.G' in result
        assert 'height.E' in result
        assert 'height' in result

    def test_aggregation_addition(self, haplotypes):
        arch = Architecture()
        arch.add('a', NoiseComponent(variance=1.0))
        arch.add('b', NoiseComponent(variance=1.0))
        arch.add('c', AggregationComponent('a + b'))
        result = arch.compute(haplotypes, rng=np.random.RandomState(42))
        np.testing.assert_allclose(result['c'], result['a'] + result['b'])

    def test_aggregation_subtraction(self, haplotypes):
        arch = Architecture()
        arch.add('a', NoiseComponent(variance=1.0))
        arch.add('b', NoiseComponent(variance=1.0))
        arch.add('c', AggregationComponent('a - b'))
        result = arch.compute(haplotypes, rng=np.random.RandomState(42))
        np.testing.assert_allclose(result['c'], result['a'] - result['b'])

    def test_aggregation_multiplication(self, haplotypes):
        arch = Architecture()
        arch.add('a', NoiseComponent(variance=1.0))
        arch.add('b', NoiseComponent(variance=1.0))
        arch.add('c', AggregationComponent('a * b'))
        result = arch.compute(haplotypes, rng=np.random.RandomState(42))
        np.testing.assert_allclose(result['c'], result['a'] * result['b'])

    def test_aggregation_division(self, haplotypes):
        arch = Architecture()
        arch.add('a', NoiseComponent(variance=1.0))
        arch.add('b', NoiseComponent(variance=1.0))
        arch.add('c', AggregationComponent('a / b'))
        result = arch.compute(haplotypes, rng=np.random.RandomState(42))
        np.testing.assert_allclose(result['c'], result['a'] / result['b'])

    def test_scalar_multiplication(self, haplotypes):
        arch = Architecture()
        arch.add('a', NoiseComponent(variance=1.0))
        arch.add('b', AggregationComponent('0.5 * a'))
        result = arch.compute(haplotypes, rng=np.random.RandomState(42))
        np.testing.assert_allclose(result['b'], 0.5 * result['a'])

    def test_genetic_values_correct(self, effects, haplotypes):
        arch = Architecture()
        arch.add('g', GeneticComponent(effects=effects))
        result = arch.compute(haplotypes)
        expected = haplotypes.standardized_matvec(effects.effects)
        np.testing.assert_allclose(result['g'], expected)

    def test_compute_returns_same_phenotype_array(self, haplotypes):
        """When passed an existing NPhenotypeArray, writes into it."""
        sm = haplotypes.samples
        pheno = NPhenotypeArray(samples=sm)
        arch = Architecture()
        arch.add('x', NoiseComponent(variance=1.0))
        result = arch.compute(haplotypes, phenotypes=pheno, rng=np.random.RandomState(0))
        assert result is pheno
        assert 'x' in pheno

    def test_chained_aggregation(self, haplotypes):
        arch = Architecture()
        arch.add('a', NoiseComponent(variance=1.0))
        arch.add('b', NoiseComponent(variance=1.0))
        arch.add('c', AggregationComponent('a + b'))
        arch.add('d', AggregationComponent('c * 2.0'))
        result = arch.compute(haplotypes, rng=np.random.RandomState(42))
        np.testing.assert_allclose(result['d'], 2.0 * (result['a'] + result['b']))
