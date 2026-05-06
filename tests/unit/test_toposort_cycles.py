"""
Unit tests for Architecture._toposort cycle detection and ordering.

Tests:
1. Self-referential cycle (A depends on A's output)
2. Two-node cycle (A→B→A)
3. Three-node cycle (A→B→C→A)
4. Diamond DAG (no cycle, valid toposort)
5. Linear chain (correct order)
6. Independent nodes (any order valid)
7. Duplicate output name raises in _register_node
"""
import numpy as np
import pytest

from xftsim.arch import (
    Architecture, ArchNode, NoiseComponent, AggregationComponent, GeneticComponent,
)
from xftsim.effect import AdditiveEffects


class TestCycleDetection:
    def test_self_referential_no_cycle(self):
        """Self-loop is explicitly allowed (skipped in Kahn's algorithm)."""
        arch = Architecture()
        arch.add('X', AggregationComponent('X + 1'))
        # Self-loops are filtered out — this should NOT raise
        nodes = arch.nodes
        assert len(nodes) == 1

    def test_two_node_cycle(self):
        """A depends on B, B depends on A → cycle."""
        arch = Architecture()
        # First add creates the 'A' output, second creates 'B' output
        # But B depends on A, A depends on B → cycle at toposort
        arch.add('A', AggregationComponent('B'))
        arch.add('B', AggregationComponent('A'))
        with pytest.raises(ValueError, match="Cycle"):
            _ = arch.nodes

    def test_three_node_cycle(self):
        """A→B→C→A → cycle."""
        arch = Architecture()
        # All three nodes are registered, but form a cycle
        arch.add('A', AggregationComponent('C'))
        arch.add('B', AggregationComponent('A'))
        arch.add('C', AggregationComponent('B'))
        with pytest.raises(ValueError, match="Cycle"):
            _ = arch.nodes


class TestToposortOrdering:
    def test_linear_chain(self):
        """A → B → C should be sorted [A, B, C]."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('B', AggregationComponent('A'))
        arch.add('C', AggregationComponent('B'))
        ordered = arch.nodes
        output_order = [n.outputs[0] for n in ordered]
        assert output_order.index('A') < output_order.index('B')
        assert output_order.index('B') < output_order.index('C')

    def test_diamond_dag(self):
        """Diamond: A → B, A → C, B,C → D."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('B', AggregationComponent('A'))
        arch.add('C', AggregationComponent('A'))
        arch.add('D', AggregationComponent('B + C'))
        ordered = arch.nodes
        output_order = [n.outputs[0] for n in ordered]
        assert output_order.index('A') < output_order.index('B')
        assert output_order.index('A') < output_order.index('C')
        assert output_order.index('B') < output_order.index('D')
        assert output_order.index('C') < output_order.index('D')

    def test_independent_nodes(self):
        """Independent nodes should all appear (order doesn't matter)."""
        arch = Architecture()
        arch.add('X', NoiseComponent(variance=1.0))
        arch.add('Y', NoiseComponent(variance=2.0))
        arch.add('Z', NoiseComponent(variance=3.0))
        ordered = arch.nodes
        output_set = {n.outputs[0] for n in ordered}
        assert output_set == {'X', 'Y', 'Z'}

    def test_five_node_dag(self):
        """Complex 5-node DAG with multiple dependencies."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('B', NoiseComponent(variance=1.0))
        arch.add('C', AggregationComponent('A + B'))
        arch.add('D', AggregationComponent('A'))
        arch.add('E', AggregationComponent('C + D'))
        ordered = arch.nodes
        output_order = [n.outputs[0] for n in ordered]
        # A, B before C, D; C, D before E
        assert output_order.index('A') < output_order.index('C')
        assert output_order.index('B') < output_order.index('C')
        assert output_order.index('A') < output_order.index('D')
        assert output_order.index('C') < output_order.index('E')
        assert output_order.index('D') < output_order.index('E')

    def test_toposort_caching(self):
        """Accessing .nodes twice should return same object (cached)."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('B', AggregationComponent('A'))
        first = arch.nodes
        second = arch.nodes
        assert first is second

    def test_cache_invalidation(self):
        """Adding a node should invalidate the toposort cache."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        first = arch.nodes
        arch.add('B', AggregationComponent('A'))
        second = arch.nodes
        assert len(second) == 2
        assert len(first) == 1  # first was a snapshot


class TestDuplicateOutput:
    def test_duplicate_output_raises(self):
        """Adding a node with a duplicate output name should raise."""
        arch = Architecture()
        arch.add('Y', NoiseComponent(variance=1.0))
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            arch.add('Y', NoiseComponent(variance=2.0))


class TestUndefinedReference:
    def test_undefined_input(self):
        """Referencing undefined output should raise ValueError at toposort."""
        arch = Architecture()
        arch.add('Y', AggregationComponent('NONEXISTENT'))
        with pytest.raises(ValueError, match="Undefined"):
            _ = arch.nodes

    def test_partial_undefined(self):
        """One defined + one undefined input."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('Y', AggregationComponent('A + MISSING'))
        with pytest.raises(ValueError, match="Undefined"):
            _ = arch.nodes
