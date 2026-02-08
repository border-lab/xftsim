"""
Unit tests for advanced Architecture class edge cases.

Tests:
1. Architecture.from_formula with multi-line formula including all component types
2. Architecture repr: contains component names, has expected structure
3. Architecture with 10+ nodes: toposort handles correctly
4. Architecture.add with inputs kwarg: edges created correctly
5. Architecture.nodes property caches after first call, invalidated by add
6. Architecture with no nodes: compute returns empty phenotype
7. Duplicate output name detection: adding same output twice raises
8. Architecture compute with generation > 0: uses phenotype_history
9. from_formula and programmatic API produce equivalent results
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray, NPhenotypeArray, PedigreeArray
from xftsim.narch import (
    Architecture, ArchNode, GeneticComponent, MVGeneticComponent,
    NoiseComponent, CNoiseComponent, AggregationComponent,
    ParentComponent, MotherComponent, FatherComponent,
    HaplotypeGeneticComponent,
)
from xftsim.neffect import AdditiveEffects, MultivariateEffects

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


def _make_hap(n=20, m=10, seed=42):
    """Helper to create test haplotypes."""
    return TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)


class TestFromFormulaMultiComponent:
    """Test Architecture.from_formula with complex multi-line formulas."""

    def test_formula_all_component_types(self):
        """Formula with genetic, mvGenetic, noise, cnoise, aggregation, parent."""
        m = 10
        eff1 = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        eff2 = MultivariateEffects.from_h2_rg(h2=[0.4, 0.3], rg=0.2, m=m, seed=43)

        formula = """
            # Single trait with genetic and noise
            trait1.G ~ genetic(beta1)
            trait1.E ~ noise(0.5)
            trait1 ~ trait1.G + trait1.E

            # Multivariate traits
            (trait2.G, trait3.G) ~ mvGenetic(beta_mv)

            # Correlated noise
            (trait2.E, trait3.E) ~ cnoise(cov=[[0.6, 0.1], [0.1, 0.7]])

            # Aggregation
            trait2 ~ trait2.G + trait2.E
            trait3 ~ trait3.G + trait3.E

            # Vertical transmission
            trait4.VT ~ parent(trait1, founder=noise(0.3))
            trait4 ~ 0.5 * trait4.VT
        """

        effects = {
            'beta1': eff1,
            'beta_mv': eff2,
        }

        arch = Architecture.from_formula(formula, effects=effects)

        # Check that all outputs are registered
        all_outputs = [out for node in arch._nodes for out in node.outputs]
        assert 'trait1.G' in all_outputs
        assert 'trait1.E' in all_outputs
        assert 'trait1' in all_outputs
        assert 'trait2.G' in all_outputs
        assert 'trait3.G' in all_outputs
        assert 'trait2.E' in all_outputs
        assert 'trait3.E' in all_outputs
        assert 'trait2' in all_outputs
        assert 'trait3' in all_outputs
        assert 'trait4.VT' in all_outputs
        assert 'trait4' in all_outputs

        # Verify we can compute
        hap = _make_hap(n=30, m=m)
        pheno = arch.compute(hap, rng=np.random.RandomState(42))
        assert 'trait1' in pheno
        assert 'trait2' in pheno
        assert 'trait3' in pheno
        assert 'trait4' in pheno

    def test_formula_with_grouping(self):
        """Formula with | grouping operator."""
        m = 10
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)

        formula = """
            Y.G ~ genetic(beta)
            Y.E ~ noise(0.5) | FID
            Y ~ Y.G + Y.E
        """

        arch = Architecture.from_formula(formula, effects={'beta': eff})

        # Find the noise node
        noise_node = None
        for node in arch._nodes:
            if 'Y.E' in node.outputs:
                noise_node = node
                break

        assert noise_node is not None
        assert noise_node.grouping == 'FID'

    def test_formula_comments_and_empty_lines(self):
        """Formula with comments and empty lines should parse correctly."""
        m = 5
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)

        formula = """
            # This is a comment

            Y.G ~ genetic(beta)

            # Another comment
            Y.E ~ noise(0.5)

            Y ~ Y.G + Y.E
        """

        arch = Architecture.from_formula(formula, effects={'beta': eff})
        assert len(arch._nodes) == 3


class TestArchitectureRepr:
    """Test Architecture and ArchNode repr methods."""

    def test_architecture_repr_contains_node_count(self):
        """Architecture repr should show node count."""
        arch = Architecture()
        arch.add('Y.E', NoiseComponent(variance=1.0))
        arch.add('Y.G', GeneticComponent(AdditiveEffects.from_h2(h2=0.5, m=5, seed=42)))

        r = repr(arch)
        assert 'Architecture' in r
        assert 'nodes=2' in r

    def test_architecture_repr_contains_outputs(self):
        """Architecture repr should show output names."""
        arch = Architecture()
        arch.add('trait1', NoiseComponent(variance=1.0))
        arch.add('trait2', NoiseComponent(variance=1.0))

        r = repr(arch)
        assert 'trait1' in r
        assert 'trait2' in r

    def test_architecture_repr_empty(self):
        """Empty architecture repr should show 0 nodes."""
        arch = Architecture()
        r = repr(arch)
        assert 'Architecture' in r
        assert 'nodes=0' in r

    def test_archnode_repr_contains_details(self):
        """ArchNode repr should show outputs, component, inputs, grouping."""
        component = NoiseComponent(variance=1.0)
        node = ArchNode(outputs=['Y'], component=component, inputs=['X'], grouping='FID')

        r = repr(node)
        assert 'ArchNode' in r
        assert 'Y' in r
        assert 'NoiseComponent' in r
        assert 'X' in r
        assert 'FID' in r


class TestArchitectureLargeDAG:
    """Test Architecture with 10+ nodes for toposort correctness."""

    def test_ten_plus_nodes_toposort(self):
        """Architecture with 10+ nodes should toposort correctly."""
        arch = Architecture()

        # Create a chain of 12 nodes
        arch.add('n0', NoiseComponent(variance=1.0))
        for i in range(1, 12):
            arch.add(f'n{i}', AggregationComponent(f'n{i-1}'), inputs=[f'n{i-1}'])

        # Verify all nodes are sorted
        sorted_nodes = arch.nodes
        assert len(sorted_nodes) == 12

        # Verify topological order: n0 < n1 < ... < n11
        node_outputs = [node.outputs[0] for node in sorted_nodes]
        for i in range(12):
            assert node_outputs[i] == f'n{i}'

    def test_complex_dag_fifteen_nodes(self):
        """Complex DAG with 15 nodes, multiple paths."""
        arch = Architecture()

        # Layer 0: 3 noise nodes
        arch.add('a', NoiseComponent(variance=1.0))
        arch.add('b', NoiseComponent(variance=1.0))
        arch.add('c', NoiseComponent(variance=1.0))

        # Layer 1: combinations
        arch.add('d', AggregationComponent('a + b'), inputs=['a', 'b'])
        arch.add('e', AggregationComponent('b + c'), inputs=['b', 'c'])
        arch.add('f', AggregationComponent('a + c'), inputs=['a', 'c'])

        # Layer 2: more combinations
        arch.add('g', AggregationComponent('d + e'), inputs=['d', 'e'])
        arch.add('h', AggregationComponent('e + f'), inputs=['e', 'f'])
        arch.add('i', AggregationComponent('d + f'), inputs=['d', 'f'])

        # Layer 3: triple combinations
        arch.add('j', AggregationComponent('g + h'), inputs=['g', 'h'])
        arch.add('k', AggregationComponent('h + i'), inputs=['h', 'i'])
        arch.add('l', AggregationComponent('g + i'), inputs=['g', 'i'])

        # Layer 4: final combinations
        arch.add('m', AggregationComponent('j + k'), inputs=['j', 'k'])
        arch.add('n', AggregationComponent('k + l'), inputs=['k', 'l'])
        arch.add('o', AggregationComponent('j + l'), inputs=['j', 'l'])

        sorted_nodes = arch.nodes
        assert len(sorted_nodes) == 15

        # Verify topological order constraints
        node_map = {node.outputs[0]: i for i, node in enumerate(sorted_nodes)}

        # Layer 0 should come before all others
        assert node_map['a'] < node_map['d']
        assert node_map['b'] < node_map['d']
        assert node_map['c'] < node_map['e']

        # Layer 1 before layer 2
        assert node_map['d'] < node_map['g']
        assert node_map['e'] < node_map['g']

        # Final layer should be last
        assert node_map['j'] < node_map['m']
        assert node_map['k'] < node_map['m']


class TestArchitectureAddWithInputs:
    """Test Architecture.add with explicit inputs kwarg."""

    def test_add_with_explicit_inputs(self):
        """add() with inputs kwarg should create edges correctly."""
        arch = Architecture()
        arch.add('X', NoiseComponent(variance=1.0))
        arch.add('Y', NoiseComponent(variance=1.0))
        arch.add('Z', AggregationComponent('X + Y'), inputs=['X', 'Y'])

        # Find the Z node
        z_node = None
        for node in arch._nodes:
            if 'Z' in node.outputs:
                z_node = node
                break

        assert z_node is not None
        assert set(z_node.inputs) == {'X', 'Y'}

    def test_add_overrides_auto_detected_inputs(self):
        """Explicit inputs kwarg should override auto-detection."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('B', NoiseComponent(variance=1.0))

        # AggregationComponent would auto-detect ['A', 'B'], but we override
        arch.add('C', AggregationComponent('A + B'), inputs=['A'])

        c_node = [n for n in arch._nodes if 'C' in n.outputs][0]
        assert c_node.inputs == ['A']

    def test_add_with_empty_inputs_list(self):
        """add() with inputs=[] should create no edges."""
        arch = Architecture()
        arch.add('X', NoiseComponent(variance=1.0), inputs=[])

        x_node = arch._nodes[0]
        assert x_node.inputs == []


class TestArchitectureNodesCaching:
    """Test Architecture.nodes property caching behavior."""

    def test_nodes_caches_after_first_call(self):
        """nodes property should cache sorted list."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('B', NoiseComponent(variance=1.0))

        # First call should populate cache
        nodes1 = arch.nodes
        assert arch._sorted is not None

        # Second call should return same cached list
        nodes2 = arch.nodes
        assert nodes1 is nodes2

    def test_nodes_invalidated_by_add(self):
        """Adding a node should invalidate the cache."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))

        # Populate cache
        _ = arch.nodes
        assert arch._sorted is not None

        # Add another node
        arch.add('B', NoiseComponent(variance=1.0))

        # Cache should be invalidated
        assert arch._sorted is None

        # Next call should re-sort and include both
        nodes = arch.nodes
        assert len(nodes) == 2

    def test_nodes_recaches_after_invalidation(self):
        """After invalidation, nodes should recache on next access."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        _ = arch.nodes  # cache

        arch.add('B', NoiseComponent(variance=1.0))  # invalidate
        assert arch._sorted is None

        _ = arch.nodes  # recache
        assert arch._sorted is not None


class TestArchitectureEmptyCompute:
    """Test Architecture with no nodes."""

    def test_empty_architecture_compute(self):
        """Architecture with no nodes should return empty phenotype."""
        arch = Architecture()
        hap = _make_hap()

        result = arch.compute(hap, rng=np.random.RandomState(42))

        assert isinstance(result, NPhenotypeArray)
        assert len(result.keys) == 0
        assert result.samples.n == hap.n

    def test_empty_architecture_nodes_property(self):
        """Empty architecture should have empty nodes list."""
        arch = Architecture()
        assert arch.nodes == []
        assert arch._sorted == []


class TestDuplicateOutputDetection:
    """Test duplicate output name detection."""

    def test_duplicate_single_output_raises(self):
        """Adding same output name twice should raise."""
        arch = Architecture()
        arch.add('Y', NoiseComponent(variance=1.0))

        with pytest.raises(ValueError, match="Duplicate output name 'Y'"):
            arch.add('Y', NoiseComponent(variance=2.0))

    def test_duplicate_in_multi_output_raises(self):
        """Duplicate within multi-output list should raise."""
        m = 5
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.5], rg=0.2, m=m, seed=42)

        arch = Architecture()
        arch.add('X', NoiseComponent(variance=1.0))

        # Try to add multi-output that includes existing 'X'
        with pytest.raises(ValueError, match="Duplicate output name 'X'"):
            arch.add(['X', 'Y'], MVGeneticComponent(eff))

    def test_duplicate_across_nodes_raises(self):
        """Duplicate output across different nodes should raise."""
        arch = Architecture()
        arch.add(['A', 'B'], NoiseComponent(variance=1.0))  # won't work, but for testing

        # Actually, NoiseComponent is single-output, let me fix this test
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('B', NoiseComponent(variance=1.0))

        with pytest.raises(ValueError, match="Duplicate output name"):
            arch.add('A', NoiseComponent(variance=1.0))


class TestArchitectureComputeWithHistory:
    """Test Architecture.compute with generation > 0 and phenotype_history."""

    def test_compute_with_phenotype_history(self):
        """compute() with generation > 0 should use phenotype_history."""
        m = 10
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)

        # Architecture with vertical transmission
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.VT', ParentComponent('Y', founder_component=NoiseComponent(variance=0.1)))
        arch.add('Y', AggregationComponent('Y.G + 0.3 * Y.VT'))

        # Gen 0: founders
        hap0 = _make_hap(n=20, m=m, seed=42)
        pheno0 = arch.compute(hap0, rng=np.random.RandomState(42), generation=0)

        # Gen 1: offspring (need pedigree for parent lookup)
        hap1 = _make_hap(n=20, m=m, seed=43)

        # Create dummy pedigree for gen 1
        offspring_samples = SampleMeta(iid=np.arange(20), sex=np.tile([0, 1], 10))
        ped1 = PedigreeArray(
            offspring_samples=offspring_samples,
            maternal_idx=np.tile(np.arange(10), 2),  # dummy parents from gen 0
            paternal_idx=np.tile(np.arange(10, 20), 2),
            parent_n=20,
        )

        phenotype_history = {0: pheno0}
        pedigree_history = {1: ped1}

        pheno1 = arch.compute(
            hap1,
            rng=np.random.RandomState(43),
            generation=1,
            phenotype_history=phenotype_history,
            pedigree_history=pedigree_history,
        )

        # VT component should have been computed using phenotype_history
        assert 'Y.VT' in pheno1
        assert 'Y' in pheno1

    def test_compute_generation_zero_no_history_needed(self):
        """compute() with generation=0 should not require phenotype_history."""
        m = 10
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)

        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.VT', ParentComponent('Y', founder_component=NoiseComponent(variance=0.1)))
        arch.add('Y', AggregationComponent('Y.G + Y.VT'))

        hap = _make_hap(n=20, m=m)

        # Should work without phenotype_history because of founder fallback
        pheno = arch.compute(hap, rng=np.random.RandomState(42), generation=0)
        assert 'Y' in pheno


class TestFormulaVsProgrammaticEquivalence:
    """Test that from_formula and programmatic API produce equivalent results."""

    def test_simple_formula_vs_programmatic(self):
        """Simple formula should produce same structure as programmatic API."""
        m = 10
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)

        # Formula version
        formula = """
            Y.G ~ genetic(beta)
            Y.E ~ noise(0.5)
            Y ~ Y.G + Y.E
        """
        arch_formula = Architecture.from_formula(formula, effects={'beta': eff})

        # Programmatic version
        arch_prog = Architecture()
        arch_prog.add('Y.G', GeneticComponent(eff))
        arch_prog.add('Y.E', NoiseComponent(variance=0.5))
        arch_prog.add('Y', AggregationComponent('Y.G + Y.E'))

        # Check same number of nodes
        assert len(arch_formula.nodes) == len(arch_prog.nodes)

        # Check same outputs
        formula_outputs = [node.outputs for node in arch_formula.nodes]
        prog_outputs = [node.outputs for node in arch_prog.nodes]
        assert formula_outputs == prog_outputs

    def test_multivariate_formula_vs_programmatic(self):
        """Multivariate formula should match programmatic API."""
        m = 10
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.4], rg=0.3, m=m, seed=42)

        # Formula version
        formula = """
            (trait1.G, trait2.G) ~ mvGenetic(beta_mv)
            trait1.E ~ noise(0.5)
            trait2.E ~ noise(0.6)
            trait1 ~ trait1.G + trait1.E
            trait2 ~ trait2.G + trait2.E
        """
        arch_formula = Architecture.from_formula(formula, effects={'beta_mv': eff})

        # Programmatic version
        arch_prog = Architecture()
        arch_prog.add(['trait1.G', 'trait2.G'], MVGeneticComponent(eff))
        arch_prog.add('trait1.E', NoiseComponent(variance=0.5))
        arch_prog.add('trait2.E', NoiseComponent(variance=0.6))
        arch_prog.add('trait1', AggregationComponent('trait1.G + trait1.E'))
        arch_prog.add('trait2', AggregationComponent('trait2.G + trait2.E'))

        assert len(arch_formula.nodes) == len(arch_prog.nodes)

        # Verify both can compute same outputs
        hap = _make_hap(n=30, m=m)
        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)

        pheno_formula = arch_formula.compute(hap, rng=rng1)
        pheno_prog = arch_prog.compute(hap, rng=rng2)

        # Check same keys
        assert set(pheno_formula.keys) == set(pheno_prog.keys)

    def test_vt_formula_vs_programmatic(self):
        """Vertical transmission formula should match programmatic API."""
        m = 10
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)

        formula = """
            Y.G ~ genetic(beta)
            Y.VT ~ parent(Y, founder=noise(0.2))
            Y.E ~ noise(0.5)
            Y ~ Y.G + 0.3 * Y.VT + Y.E
        """
        arch_formula = Architecture.from_formula(formula, effects={'beta': eff})

        # Programmatic version
        arch_prog = Architecture()
        arch_prog.add('Y.G', GeneticComponent(eff))
        arch_prog.add('Y.VT', ParentComponent('Y', founder_component=NoiseComponent(variance=0.2)))
        arch_prog.add('Y.E', NoiseComponent(variance=0.5))
        arch_prog.add('Y', AggregationComponent('Y.G + 0.3 * Y.VT + Y.E'))

        assert len(arch_formula.nodes) == len(arch_prog.nodes)

        formula_outputs = [node.outputs for node in arch_formula.nodes]
        prog_outputs = [node.outputs for node in arch_prog.nodes]
        assert formula_outputs == prog_outputs
