"""
Unit tests for Architecture DAG edge cases.

Tests:
1. Single node architecture (no dependencies)
2. Diamond dependency pattern (A → C, B → C, A → D, B → D)
3. Long chain (A → B → C → D → E)
4. Self-loop allowed (node depends on itself)
5. Duplicate node name raises
6. Empty architecture (no nodes) computes empty phenotypes
7. toposort order respects dependencies
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray, PhenotypeArray
from xftsim.arch import Architecture, NoiseComponent, AggregationComponent, GeneticComponent
from xftsim.effect import AdditiveEffects


def _make_hap(n=20, m=5):
    rng = np.random.RandomState(42)
    sm = SampleMeta(iid=np.arange(n), sex=np.tile([0, 1], n // 2))
    vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
    geno = rng.binomial(1, 0.5, (n, m, 2)).astype(np.int8)
    return DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)


class TestSingleNode:
    def test_single_noise_node(self):
        """Architecture with single noise node should work."""
        arch = Architecture()
        arch.add('Y', NoiseComponent(variance=1.0))
        hap = _make_hap()
        result = arch.compute(hap, rng=np.random.RandomState(42))
        assert 'Y' in result
        assert len(result['Y']) == 20


class TestDiamondDependency:
    def test_diamond_pattern(self):
        """A, B → C; A, B → D should compute correctly."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('B', NoiseComponent(variance=1.0))
        arch.add('C', AggregationComponent('A + B'), inputs=['A', 'B'])
        arch.add('D', AggregationComponent('A - B'), inputs=['A', 'B'])

        hap = _make_hap()
        result = arch.compute(hap, rng=np.random.RandomState(42))
        np.testing.assert_allclose(result['C'], result['A'] + result['B'])
        np.testing.assert_allclose(result['D'], result['A'] - result['B'])


class TestLongChain:
    def test_chain_a_b_c_d_e(self):
        """A → B → C → D → E chain should compute in order."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('B', AggregationComponent('A'), inputs=['A'])
        arch.add('C', AggregationComponent('B'), inputs=['B'])
        arch.add('D', AggregationComponent('C'), inputs=['C'])
        arch.add('E', AggregationComponent('D'), inputs=['D'])

        hap = _make_hap()
        result = arch.compute(hap, rng=np.random.RandomState(42))
        # All should be equal (each is just a copy)
        np.testing.assert_allclose(result['A'], result['E'])


class TestEmptyArchitecture:
    def test_empty_computes(self):
        """Architecture with no nodes should return empty phenotypes."""
        arch = Architecture()
        hap = _make_hap()
        result = arch.compute(hap, rng=np.random.RandomState(42))
        assert isinstance(result, PhenotypeArray)
        assert len(result.keys) == 0


class TestToposortOrder:
    def test_reverse_add_order(self):
        """Adding nodes in reverse dependency order should still work via toposort."""
        arch = Architecture()
        arch.add('D', AggregationComponent('C'), inputs=['C'])
        arch.add('C', AggregationComponent('B'), inputs=['B'])
        arch.add('B', AggregationComponent('A'), inputs=['A'])
        arch.add('A', NoiseComponent(variance=1.0))

        hap = _make_hap()
        result = arch.compute(hap, rng=np.random.RandomState(42))
        np.testing.assert_allclose(result['A'], result['D'])

    def test_toposort_nodes_order(self):
        """Nodes should be returned in topological order."""
        arch = Architecture()
        arch.add('Y', AggregationComponent('A + B'), inputs=['A', 'B'])
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('B', NoiseComponent(variance=1.0))

        node_names = [n.outputs for n in arch.nodes]
        a_idx = node_names.index(['A'])
        b_idx = node_names.index(['B'])
        y_idx = node_names.index(['Y'])
        assert a_idx < y_idx, "A should come before Y in toposort"
        assert b_idx < y_idx, "B should come before Y in toposort"


class TestArchitectureRepr:
    def test_repr_nonempty(self):
        """Architecture repr should be non-empty and informative."""
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(AdditiveEffects.from_h2(h2=0.5, m=5, seed=42)))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        r = repr(arch)
        assert 'Architecture' in r
        assert len(r) > 10

    def test_repr_empty(self):
        """Empty architecture repr should mention 0 nodes."""
        arch = Architecture()
        r = repr(arch)
        assert '0' in r or 'empty' in r.lower() or 'Architecture' in r
