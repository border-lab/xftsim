"""
Unit tests for Architecture DAG operations.

Tests:
1. add() with string output (auto-wraps to list)
2. add() auto-detects AggregationComponent inputs
3. nodes property caching behavior
4. _toposort with large DAG (50+ nodes)
5. _toposort with diamond dependency pattern
6. _toposort with wide DAG (many independent nodes)
7. compute with multi-output node (mvGenetic)
8. compute creates phenotypes if None
9. compute creates rng if None
10. __repr__ with various node counts
11. from_formula returns Architecture
"""
import numpy as np
import pytest

from xftsim.arch import (
    Architecture, ArchNode, GeneticComponent, MVGeneticComponent,
    NoiseComponent, CNoiseComponent, AggregationComponent,
)
from xftsim.effect import AdditiveEffects, MultivariateEffects
from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray


def _make_hap(n=10, m=5, seed=42):
    sm = SampleMeta(iid=np.arange(n))
    vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
    rng = np.random.RandomState(seed)
    geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
    return DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)


class TestArchitectureAdd:
    def test_string_output_wrapped(self):
        """add('Y', ...) should wrap output to ['Y']."""
        arch = Architecture()
        arch.add('Y', NoiseComponent(variance=1.0))
        assert arch._nodes[0].outputs == ['Y']

    def test_list_output_preserved(self):
        """add(['Y.G'], ...) keeps list."""
        arch = Architecture()
        arch.add(['Y.G'], NoiseComponent(variance=1.0))
        assert arch._nodes[0].outputs == ['Y.G']

    def test_auto_detect_aggregation_inputs(self):
        """AggregationComponent inputs auto-detected."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('B', NoiseComponent(variance=1.0))
        comp = AggregationComponent('A + B')
        arch.add('C', comp)
        # Inputs should be auto-detected from expression
        node = arch._nodes[2]
        assert 'A' in node.inputs
        assert 'B' in node.inputs

    def test_duplicate_output_raises(self):
        """Adding duplicate output name raises ValueError."""
        arch = Architecture()
        arch.add('Y', NoiseComponent(variance=1.0))
        with pytest.raises(ValueError, match="Duplicate output"):
            arch.add('Y', NoiseComponent(variance=0.5))


class TestArchitectureNodes:
    def test_nodes_property_caching(self):
        """Accessing .nodes twice returns same list (cached)."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        nodes1 = arch.nodes
        nodes2 = arch.nodes
        assert nodes1 is nodes2

    def test_add_invalidates_cache(self):
        """Adding a node invalidates the toposort cache."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        nodes1 = arch.nodes
        arch.add('B', NoiseComponent(variance=0.5))
        nodes2 = arch.nodes
        assert nodes1 is not nodes2
        assert len(nodes2) == 2


class TestArchitectureToposort:
    def test_large_linear_dag(self):
        """50-node linear chain: each depends on previous."""
        arch = Architecture()
        # First node: noise
        arch.add('N0', NoiseComponent(variance=1.0))
        for i in range(1, 50):
            arch.add(f'N{i}', AggregationComponent(f'N{i-1} + 0'),
                     inputs=[f'N{i-1}'])
        nodes = arch.nodes
        assert len(nodes) == 50
        # First node should be N0
        assert nodes[0].outputs == ['N0']
        # Last should be N49
        assert nodes[-1].outputs == ['N49']

    def test_wide_independent_dag(self):
        """30 independent noise nodes: all should appear."""
        arch = Architecture()
        for i in range(30):
            arch.add(f'N{i}', NoiseComponent(variance=1.0))
        nodes = arch.nodes
        assert len(nodes) == 30

    def test_diamond_dag(self):
        """Diamond: A → B, A → C, B+C → D."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('B', AggregationComponent('A * 2'), inputs=['A'])
        arch.add('C', AggregationComponent('A * 3'), inputs=['A'])
        arch.add('D', AggregationComponent('B + C'), inputs=['B', 'C'])
        nodes = arch.nodes
        assert len(nodes) == 4
        # A must come before B and C, which must come before D
        order = {n.outputs[0]: i for i, n in enumerate(nodes)}
        assert order['A'] < order['B']
        assert order['A'] < order['C']
        assert order['B'] < order['D']
        assert order['C'] < order['D']

    def test_undefined_reference_raises(self):
        """Reference to non-existent output raises ValueError."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('B', AggregationComponent('A + MISSING'), inputs=['A', 'MISSING'])
        with pytest.raises(ValueError, match="Undefined reference"):
            arch.nodes


class TestArchitectureCompute:
    def test_compute_creates_phenotypes(self):
        """compute(hap, phenotypes=None) creates new phenotype array."""
        hap = _make_hap(n=10, m=5)
        effects = AdditiveEffects.from_h2(m=5, h2=0.5, seed=42)
        arch = Architecture()
        arch.add('Y', GeneticComponent(effects))
        result = arch.compute(hap)
        assert 'Y' in result
        assert result['Y'].shape == (10,)

    def test_compute_creates_rng(self):
        """compute(hap, rng=None) creates new rng."""
        hap = _make_hap(n=10, m=5)
        arch = Architecture()
        arch.add('Y', NoiseComponent(variance=1.0))
        result = arch.compute(hap)
        assert 'Y' in result

    def test_compute_writes_to_existing_phenotypes(self):
        """Passing existing phenotype array adds keys to it."""
        hap = _make_hap(n=10, m=5)
        from xftsim.struct import PhenotypeArray
        pheno = PhenotypeArray(samples=hap.samples)
        pheno._values['EXISTING'] = np.ones(10)

        arch = Architecture()
        arch.add('NEW', NoiseComponent(variance=1.0))
        result = arch.compute(hap, phenotypes=pheno, rng=np.random.RandomState(42))
        assert 'EXISTING' in result
        assert 'NEW' in result

    def test_compute_multi_output(self):
        """MVGeneticComponent produces multiple outputs."""
        hap = _make_hap(n=10, m=5)
        effects = MultivariateEffects.from_h2_rg(m=5, h2=[0.5, 0.5], rg=0.3, seed=42)
        arch = Architecture()
        arch.add(['Y.G', 'Z.G'], MVGeneticComponent(effects))
        result = arch.compute(hap, rng=np.random.RandomState(42))
        assert 'Y.G' in result
        assert 'Z.G' in result
        assert result['Y.G'].shape == (10,)
        assert result['Z.G'].shape == (10,)


class TestArchitectureRepr:
    def test_empty_repr(self):
        arch = Architecture()
        r = repr(arch)
        assert "Architecture" in r
        assert "nodes=0" in r

    def test_repr_with_nodes(self):
        arch = Architecture()
        arch.add('Y.G', NoiseComponent(variance=1.0))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        r = repr(arch)
        assert "nodes=2" in r
        assert "Y.G" in r
        assert "Y.E" in r

    def test_from_formula(self):
        """from_formula should return an Architecture instance."""
        effects = AdditiveEffects.from_h2(m=5, h2=0.5, seed=42)
        arch = Architecture.from_formula(
            "Y ~ genetic(eff)",
            effects={'eff': effects},
        )
        assert isinstance(arch, Architecture)
        assert len(arch._nodes) >= 1
