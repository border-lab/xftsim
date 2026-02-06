"""
Unit tests for the formula parser (Phase 1: minimal grammar).
"""
import numpy as np
import pytest
from xftsim.parser import parse_formula
from xftsim.narch import (
    ArchNode, GeneticComponent, NoiseComponent, AggregationComponent,
)
from xftsim.neffect import AdditiveEffects


@pytest.fixture
def simple_effects():
    """A simple AdditiveEffects for testing."""
    return {'eff': AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)}


# ── Valid formulas ──────────────────────────────────────────────────────────

class TestValidFormulas:
    def test_genetic(self, simple_effects):
        nodes = parse_formula("height.G ~ genetic(eff)", simple_effects)
        assert len(nodes) == 1
        assert nodes[0].outputs == ['height.G']
        assert isinstance(nodes[0].component, GeneticComponent)
        assert nodes[0].inputs == []

    def test_noise(self):
        nodes = parse_formula("height.E ~ noise(0.2)")
        assert len(nodes) == 1
        assert nodes[0].outputs == ['height.E']
        assert isinstance(nodes[0].component, NoiseComponent)
        assert nodes[0].component.variance == 0.2

    def test_aggregation(self):
        nodes = parse_formula("""
            height.G ~ noise(0.5)
            height.E ~ noise(0.5)
            height ~ height.G + height.E
        """)
        assert len(nodes) == 3
        agg = nodes[2]
        assert agg.outputs == ['height']
        assert isinstance(agg.component, AggregationComponent)
        assert set(agg.inputs) == {'height.G', 'height.E'}

    def test_scalar_multiplication(self):
        nodes = parse_formula("""
            a ~ noise(1.0)
            b ~ noise(1.0)
            c ~ 0.3 * a + b
        """)
        assert len(nodes) == 3
        agg = nodes[2]
        assert 'a' in agg.inputs
        assert 'b' in agg.inputs

    def test_multi_statement(self, simple_effects):
        nodes = parse_formula("""
            height.G ~ genetic(eff)
            height.E ~ noise(0.2)
            height ~ height.G + height.E
        """, simple_effects)
        assert len(nodes) == 3
        assert nodes[0].outputs == ['height.G']
        assert nodes[1].outputs == ['height.E']
        assert nodes[2].outputs == ['height']

    def test_comment_and_blank_lines(self):
        nodes = parse_formula("""
            # This is a comment
            x ~ noise(1.0)

            # Another comment
            y ~ noise(2.0)
        """)
        assert len(nodes) == 2

    def test_complex_expression(self):
        nodes = parse_formula("""
            a ~ noise(1.0)
            b ~ noise(1.0)
            c ~ noise(1.0)
            d ~ a + b * c
        """)
        assert len(nodes) == 4
        agg = nodes[3]
        assert set(agg.inputs) == {'a', 'b', 'c'}

    def test_subtraction(self):
        nodes = parse_formula("""
            a ~ noise(1.0)
            b ~ noise(1.0)
            c ~ a - b
        """)
        agg = nodes[2]
        assert set(agg.inputs) == {'a', 'b'}

    def test_division(self):
        nodes = parse_formula("""
            a ~ noise(1.0)
            b ~ noise(1.0)
            c ~ a / b
        """)
        agg = nodes[2]
        assert set(agg.inputs) == {'a', 'b'}

    def test_effect_name_resolution(self, simple_effects):
        """Effect names are resolved from the effects dict."""
        nodes = parse_formula("x ~ genetic(eff)", simple_effects)
        assert isinstance(nodes[0].component, GeneticComponent)
        assert nodes[0].component.effects is simple_effects['eff']


# ── Error cases ─────────────────────────────────────────────────────────────

class TestErrors:
    def test_unknown_function(self):
        with pytest.raises(ValueError, match="unknown function"):
            parse_formula("x ~ foobar(1.0)")

    def test_missing_effect(self):
        with pytest.raises(ValueError, match="not found in effects"):
            parse_formula("x ~ genetic(missing)", {})

    def test_missing_lhs(self):
        with pytest.raises(ValueError, match="missing LHS"):
            parse_formula("~ noise(1.0)")

    def test_missing_rhs(self):
        with pytest.raises(ValueError, match="missing RHS"):
            parse_formula("x ~")

    def test_missing_tilde(self):
        with pytest.raises(ValueError, match="missing '~'"):
            parse_formula("x = noise(1.0)")

    def test_duplicate_output(self):
        with pytest.raises(ValueError, match="duplicate output"):
            parse_formula("""
                x ~ noise(1.0)
                x ~ noise(2.0)
            """)

    def test_genetic_no_effect_spec(self):
        with pytest.raises(ValueError, match="not found in effects"):
            parse_formula("x ~ genetic(eff)", {})

    def test_noise_non_numeric(self):
        with pytest.raises(ValueError, match="numeric variance"):
            parse_formula("x ~ noise(abc)")
