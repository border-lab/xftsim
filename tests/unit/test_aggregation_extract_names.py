"""
Unit tests for AggregationComponent._extract_names and auto-detection of inputs.

Tests:
1. Simple variable extraction
2. Dotted names
3. Numeric constants not extracted
4. Duplicates removed
5. Order preserved
6. Complex expression
7. Architecture.add auto-detects inputs from AggregationComponent
"""
import numpy as np
import pytest

from xftsim.arch import AggregationComponent, Architecture, NoiseComponent


class TestExtractNames:
    def test_simple_two_vars(self):
        comp = AggregationComponent('A + B')
        assert comp._input_names == ['A', 'B']

    def test_dotted_names(self):
        comp = AggregationComponent('Y.G + Y.E')
        assert comp._input_names == ['Y.G', 'Y.E']

    def test_numeric_not_extracted(self):
        comp = AggregationComponent('0.5 * A + 2 * B')
        assert comp._input_names == ['A', 'B']

    def test_duplicates_removed(self):
        comp = AggregationComponent('A + A')
        assert comp._input_names == ['A']

    def test_order_preserved(self):
        comp = AggregationComponent('C + A + B')
        assert comp._input_names == ['C', 'A', 'B']

    def test_complex_expression(self):
        comp = AggregationComponent('0.5 * (X.a + X.b) - 0.1 * Z')
        assert comp._input_names == ['X.a', 'X.b', 'Z']

    def test_single_variable(self):
        comp = AggregationComponent('X')
        assert comp._input_names == ['X']


class TestAutoDetectedInputs:
    def test_add_auto_detects_aggregation_inputs(self):
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('B', NoiseComponent(variance=1.0))
        arch.add('Y', AggregationComponent('A + B'))
        # Should have auto-detected inputs=['A', 'B']
        y_node = [n for n in arch._nodes if 'Y' in n.outputs][0]
        assert 'A' in y_node.inputs
        assert 'B' in y_node.inputs

    def test_add_explicit_inputs_override(self):
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('B', NoiseComponent(variance=1.0))
        arch.add('Y', AggregationComponent('A + B'), inputs=['A'])
        y_node = [n for n in arch._nodes if 'Y' in n.outputs][0]
        # Explicit inputs should be used, not auto-detected
        assert y_node.inputs == ['A']

    def test_non_aggregation_no_auto_inputs(self):
        arch = Architecture()
        arch.add('Y', NoiseComponent(variance=1.0))
        y_node = [n for n in arch._nodes if 'Y' in n.outputs][0]
        assert y_node.inputs == []
