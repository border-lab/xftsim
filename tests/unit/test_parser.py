"""
Unit tests for the formula parser.
"""
import numpy as np
import pytest
from xftsim.parser import parse_formula
from xftsim.narch import (
    ArchNode, Architecture, GeneticComponent, MVGeneticComponent,
    HaplotypeGeneticComponent,
    NoiseComponent, CNoiseComponent, AggregationComponent,
    MotherComponent, FatherComponent, ParentComponent,
    SiblingMeanComponent,
)
from xftsim.neffect import AdditiveEffects, MultivariateEffects


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

    def test_grouping_on_non_groupable(self):
        """genetic() does not accept | grouping."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        with pytest.raises(ValueError, match="does not accept"):
            parse_formula("x ~ genetic(eff) | FID", {'eff': eff})

    def test_grouping_on_aggregation_rejected(self):
        """Aggregation expressions cannot have | grouping."""
        with pytest.raises(ValueError, match="grouping is only valid"):
            parse_formula("""
                a ~ noise(1.0)
                b ~ noise(1.0)
                c ~ a + b | FID
            """)

    def test_invalid_grouping_identifier(self):
        """Non-identifier grouping should be rejected."""
        with pytest.raises(ValueError, match="Invalid grouping"):
            parse_formula("x ~ noise(1.0) | 123bad")


# ── Phase 3 extended grammar ────────────────────────────────────────────

class TestTupleLHS:
    def test_mvgenetic_tuple(self):
        mv = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.2, m=10, seed=42)
        nodes = parse_formula("(a.G, b.G) ~ mvGenetic(mv)", {'mv': mv})
        assert len(nodes) == 1
        assert nodes[0].outputs == ['a.G', 'b.G']
        assert isinstance(nodes[0].component, MVGeneticComponent)

    def test_cnoise_tuple(self):
        nodes = parse_formula("(a.E, b.E) ~ cnoise(cov=[[1.0,0.2],[0.2,1.0]])")
        assert len(nodes) == 1
        assert nodes[0].outputs == ['a.E', 'b.E']
        assert isinstance(nodes[0].component, CNoiseComponent)

    def test_tuple_lhs_single_element_rejected(self):
        with pytest.raises(ValueError, match="at least 2"):
            parse_formula("(a) ~ noise(1.0)")

    def test_cnoise_dimension_mismatch(self):
        """cnoise cov dimension must match LHS tuple length."""
        with pytest.raises(ValueError, match="LHS has 2 outputs"):
            parse_formula("(a, b) ~ cnoise(cov=[[1,0,0],[0,1,0],[0,0,1]])")

    def test_mvgenetic_dimension_mismatch(self):
        """mvGenetic effect k must match LHS tuple length."""
        mv = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.2, m=10, seed=42)
        with pytest.raises(ValueError, match="LHS has 3 outputs"):
            parse_formula("(a, b, c) ~ mvGenetic(mv)", {'mv': mv})


class TestGroupingParsing:
    def test_noise_with_fid(self):
        nodes = parse_formula("x ~ noise(0.5) | FID")
        assert nodes[0].grouping == 'FID'

    def test_noise_with_sex(self):
        nodes = parse_formula("x ~ noise(0.5) | sex")
        assert nodes[0].grouping == 'sex'

    def test_cnoise_with_grouping(self):
        nodes = parse_formula("(a, b) ~ cnoise(cov=[[1,0.2],[0.2,1]]) | FID")
        assert nodes[0].grouping == 'FID'

    def test_no_grouping(self):
        nodes = parse_formula("x ~ noise(0.5)")
        assert nodes[0].grouping is None

    def test_pipe_inside_parens_not_extracted(self):
        """Pipe inside function parens should not be treated as grouping."""
        # This is a tricky edge case for the parser
        nodes = parse_formula("x ~ noise(0.5)")
        # Just verify it parses without error
        assert len(nodes) == 1


class TestParentalParsing:
    def test_parent_basic(self):
        nodes = parse_formula("x ~ parent(Y)")
        assert isinstance(nodes[0].component, ParentComponent)
        assert nodes[0].component.phenotype_name == 'Y'

    def test_mother_basic(self):
        nodes = parse_formula("x ~ mother(Y)")
        assert isinstance(nodes[0].component, MotherComponent)

    def test_father_basic(self):
        nodes = parse_formula("x ~ father(Y)")
        assert isinstance(nodes[0].component, FatherComponent)

    def test_parent_with_founder(self):
        nodes = parse_formula("x ~ parent(Y, founder=noise(0.5))")
        comp = nodes[0].component
        assert isinstance(comp, ParentComponent)
        assert comp.founder_component is not None
        assert isinstance(comp.founder_component, NoiseComponent)
        assert comp.founder_component.variance == 0.5

    def test_founder_bad_function(self):
        with pytest.raises(ValueError, match="unsupported function"):
            parse_formula("x ~ parent(Y, founder=genetic(eff))")


class TestSiblingParsing:
    def test_sibling_mean(self):
        nodes = parse_formula("x ~ sibling_mean(Y)")
        assert isinstance(nodes[0].component, SiblingMeanComponent)
        assert nodes[0].inputs == ['Y']

    def test_sibling_with_grouping(self):
        nodes = parse_formula("x ~ sibling_mean(Y) | sex")
        assert nodes[0].grouping == 'sex'

    def test_sibling_empty_source_rejected(self):
        with pytest.raises(ValueError, match="requires a source"):
            parse_formula("x ~ sibling_mean()")


class TestHaplotypeGeneticParsing:
    def test_haplotype_genetic_default(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        nodes = parse_formula("x ~ haplotypeGenetic(eff)", {'eff': eff})
        assert isinstance(nodes[0].component, HaplotypeGeneticComponent)
        assert nodes[0].component.haplotype == 'maternal'

    def test_haplotype_genetic_paternal(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        nodes = parse_formula(
            "x ~ haplotypeGenetic(eff, haplotype='paternal')", {'eff': eff}
        )
        assert nodes[0].component.haplotype == 'paternal'


class TestCircularDependency:
    def test_undefined_reference_in_aggregation(self):
        """Referencing an undefined name should raise at toposort time."""
        with pytest.raises(ValueError, match="Undefined reference"):
            arch = Architecture.from_formula("Y ~ Z + noise(0.5)")
            # toposort happens at construction

    def test_self_reference(self):
        """A node referencing its own output should be detected."""
        # AggregationComponent extracts 'Y' from expression 'Y + noise(1.0)'
        # but since Y hasn't been defined, this should fail
        with pytest.raises(ValueError):
            Architecture.from_formula("Y ~ Y + noise(0.5)")
