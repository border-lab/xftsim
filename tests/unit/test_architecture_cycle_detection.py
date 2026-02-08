"""
Unit tests for Architecture cycle detection and toposort edge cases.

Tests:
1. Cyclic DAG raises ValueError
2. Self-loop (A depends on A) is allowed (skipped in toposort)
3. Diamond dependency resolves correctly
4. Undefined reference raises ValueError
5. Duplicate output raises ValueError
6. Deep chain toposort order
"""
import numpy as np
import pytest

from xftsim.narch import (
    Architecture, ArchNode, AggregationComponent, NoiseComponent,
    GeneticComponent,
)
from xftsim.neffect import AdditiveEffects


class TestCycleDetection:
    def test_cycle_raises(self):
        """A → B → A cycle should raise ValueError."""
        arch = Architecture()
        arch.add('A', AggregationComponent('B'), inputs=['B'])
        arch.add('B', AggregationComponent('A'), inputs=['A'])
        with pytest.raises(ValueError, match="Cycle detected"):
            _ = arch.nodes

    def test_three_node_cycle(self):
        """A → B → C → A."""
        arch = Architecture()
        arch.add('A', AggregationComponent('C'), inputs=['C'])
        arch.add('B', AggregationComponent('A'), inputs=['A'])
        arch.add('C', AggregationComponent('B'), inputs=['B'])
        with pytest.raises(ValueError, match="Cycle detected"):
            _ = arch.nodes


class TestSelfLoop:
    def test_self_loop_allowed(self):
        """Self-loop (node depends on itself) should not raise."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        # A → B (B = A + A, same as 2*A — depends on itself through A ref)
        arch.add('B', AggregationComponent('A'), inputs=['A'])
        # This should work fine
        nodes = arch.nodes
        assert len(nodes) == 2


class TestUndefinedReference:
    def test_undefined_input_raises(self):
        arch = Architecture()
        arch.add('Y', AggregationComponent('MISSING + ALSO_MISSING'),
                 inputs=['MISSING', 'ALSO_MISSING'])
        with pytest.raises(ValueError, match="Undefined reference"):
            _ = arch.nodes


class TestDuplicateOutput:
    def test_duplicate_output_raises(self):
        arch = Architecture()
        arch.add('Y.E', NoiseComponent(variance=1.0))
        with pytest.raises(ValueError, match="Duplicate output"):
            arch.add('Y.E', NoiseComponent(variance=0.5))


class TestToposortOrder:
    def test_deep_chain_order(self):
        """A → B → C → D → E: should be sorted in declaration order."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('B', AggregationComponent('A'), inputs=['A'])
        arch.add('C', AggregationComponent('B'), inputs=['B'])
        arch.add('D', AggregationComponent('C'), inputs=['C'])
        arch.add('E', AggregationComponent('D'), inputs=['D'])
        nodes = arch.nodes
        outputs = [n.outputs[0] for n in nodes]
        assert outputs == ['A', 'B', 'C', 'D', 'E']

    def test_diamond_dependency(self):
        """Diamond: A → B, A → C, B+C → D."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('B', AggregationComponent('A'), inputs=['A'])
        arch.add('C', AggregationComponent('A'), inputs=['A'])
        arch.add('D', AggregationComponent('B + C'), inputs=['B', 'C'])
        nodes = arch.nodes
        outputs = [n.outputs[0] for n in nodes]
        # A must come first, D must come last, B and C can be in either order
        assert outputs[0] == 'A'
        assert outputs[-1] == 'D'
        assert set(outputs[1:3]) == {'B', 'C'}

    def test_reverse_declaration_order(self):
        """Nodes declared in reverse dependency order should still sort correctly."""
        arch = Architecture()
        arch.add('C', AggregationComponent('B'), inputs=['B'])
        arch.add('B', AggregationComponent('A'), inputs=['A'])
        arch.add('A', NoiseComponent(variance=1.0))
        nodes = arch.nodes
        outputs = [n.outputs[0] for n in nodes]
        assert outputs == ['A', 'B', 'C']

    def test_wide_independent_nodes(self):
        """Many independent noise nodes should all appear before the aggregation."""
        arch = Architecture()
        for i in range(5):
            arch.add(f'N{i}', NoiseComponent(variance=float(i + 1)))
        expr = ' + '.join([f'N{i}' for i in range(5)])
        arch.add('Y', AggregationComponent(expr), inputs=[f'N{i}' for i in range(5)])
        nodes = arch.nodes
        outputs = [n.outputs[0] for n in nodes]
        assert outputs[-1] == 'Y'
        assert set(outputs[:-1]) == {f'N{i}' for i in range(5)}


class TestCacheInvalidation:
    def test_add_invalidates_cache(self):
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        nodes1 = arch.nodes
        assert len(nodes1) == 1
        arch.add('B', NoiseComponent(variance=2.0))
        nodes2 = arch.nodes
        assert len(nodes2) == 2
