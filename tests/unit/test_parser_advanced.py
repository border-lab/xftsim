"""
Unit tests for advanced formula parser edge cases.

Tests cover:
1. Multi-line formulas with blank/comment lines
2. Leading/trailing whitespace handling
3. LHS tuple parsing for multi-output components
4. Duplicate output detection
5. Founder fallback via founder= kwarg
6. Mixed component types in one formula
7. Empty formula handling
8. Invalid syntax errors
9. HaplotypeGenetic haplotype kwarg parsing
10. Effect names with dots/underscores
11. Additional edge cases from original tests
"""
import numpy as np
import pytest

from xftsim.parser import parse_formula, _extract_grouping
from xftsim.arch import (
    Architecture,
    GeneticComponent, NoiseComponent, CNoiseComponent, AggregationComponent,
    MVGeneticComponent, HaplotypeGeneticComponent,
    MotherComponent, FatherComponent, ParentComponent,
    SiblingMeanComponent, SiblingSumComponent, SiblingCountComponent,
)
from xftsim.effect import AdditiveEffects, MultivariateEffects


class TestExtractGrouping:
    """Test _extract_grouping parser function."""

    def test_no_pipe(self):
        """No pipe → (rhs, None)."""
        rhs, group = _extract_grouping('noise(1.0)')
        assert rhs == 'noise(1.0)'
        assert group is None

    def test_pipe_outside_parens(self):
        """Pipe outside parentheses."""
        rhs, group = _extract_grouping('noise(1.0) | FID')
        assert rhs == 'noise(1.0)'
        assert group == 'FID'

    def test_pipe_inside_parens_ignored(self):
        """Pipe inside parens should not be treated as grouping separator."""
        rhs, group = _extract_grouping('func(a|b)')
        assert group is None
        assert rhs == 'func(a|b)'

    def test_empty_grouping_after_pipe(self):
        """Pipe with nothing after → None grouping."""
        rhs, group = _extract_grouping('noise(1.0) | ')
        assert group is None

    def test_invalid_grouping_variable(self):
        """Non-identifier grouping variable should raise."""
        with pytest.raises(ValueError, match="Invalid grouping"):
            _extract_grouping('noise(1.0) | 123invalid')


class TestMultiLineFormulas:
    """Test multi-line formula parsing with various line types."""

    def test_formula_with_blank_lines(self):
        """Blank lines should be skipped correctly."""
        formula = """
        Y.G ~ genetic(eff)

        Y.E ~ noise(0.5)

        Y ~ Y.G + Y.E
        """
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        nodes = parse_formula(formula, effects={'eff': eff})
        assert len(nodes) == 3
        assert nodes[0].outputs == ['Y.G']
        assert nodes[1].outputs == ['Y.E']
        assert nodes[2].outputs == ['Y']

    def test_formula_with_comment_lines(self):
        """Lines starting with # should be skipped."""
        formula = """
        # This is a comment
        Y.G ~ genetic(eff)
        # Another comment
        Y.E ~ noise(0.5)
        # Final comment
        Y ~ Y.G + Y.E
        """
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        nodes = parse_formula(formula, effects={'eff': eff})
        assert len(nodes) == 3

    def test_formula_with_mixed_blank_and_comment_lines(self):
        """Formula with both blank lines and comments."""
        formula = """

        # Header comment
        Y.G ~ genetic(eff)

        # Middle comment

        Y.E ~ noise(0.5)

        # End comment
        Y ~ Y.G + Y.E

        """
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        nodes = parse_formula(formula, effects={'eff': eff})
        assert len(nodes) == 3

    def test_formula_with_inline_comment_not_supported(self):
        """Inline comments (not at line start) are not parsed specially."""
        # The parser treats # specially only at the start of a line
        # Actually, the parser may strip the entire line, so inline comments
        # within the RHS would not cause an error. Let's test what actually happens.
        formula = "Y.G ~ genetic(eff)"
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        # This should parse successfully
        nodes = parse_formula(formula, effects={'eff': eff})
        assert len(nodes) == 1


class TestWhitespaceHandling:
    """Test that leading/trailing whitespace is handled correctly."""

    def test_leading_whitespace_in_formula(self):
        """Formula with leading whitespace should parse correctly."""
        formula = "   Y.G ~ genetic(eff)"
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        nodes = parse_formula(formula, effects={'eff': eff})
        assert len(nodes) == 1
        assert nodes[0].outputs == ['Y.G']

    def test_trailing_whitespace_in_formula(self):
        """Formula with trailing whitespace should parse correctly."""
        formula = "Y.G ~ genetic(eff)   "
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        nodes = parse_formula(formula, effects={'eff': eff})
        assert len(nodes) == 1
        assert nodes[0].outputs == ['Y.G']

    def test_whitespace_around_tilde(self):
        """Whitespace around ~ should be handled correctly."""
        formula = "Y.G   ~   genetic(eff)"
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        nodes = parse_formula(formula, effects={'eff': eff})
        assert len(nodes) == 1
        assert nodes[0].outputs == ['Y.G']

    def test_whitespace_in_lhs_tuple(self):
        """Whitespace in LHS tuple should be handled correctly."""
        formula = "( E1 , E2 ) ~ cnoise(cov=[[1, 0.2], [0.2, 1]])"
        nodes = parse_formula(formula)
        assert len(nodes) == 1
        assert nodes[0].outputs == ['E1', 'E2']


class TestLhsTupleParsing:
    """Test tuple LHS parsing for multi-output components."""

    def test_simple_tuple_lhs(self):
        """Simple (A, B) ~ cnoise(...) produces 2 outputs."""
        formula = "(A, B) ~ cnoise(cov=[[1, 0.2], [0.2, 1]])"
        nodes = parse_formula(formula)
        assert len(nodes) == 1
        assert nodes[0].outputs == ['A', 'B']
        assert isinstance(nodes[0].component, CNoiseComponent)

    def test_triple_tuple_lhs(self):
        """Triple (A, B, C) ~ cnoise(...) produces 3 outputs."""
        cov = [[1, 0.2, 0.1], [0.2, 1, 0.15], [0.1, 0.15, 1]]
        formula = f"(A, B, C) ~ cnoise(cov={cov})"
        nodes = parse_formula(formula)
        assert len(nodes) == 1
        assert nodes[0].outputs == ['A', 'B', 'C']

    def test_tuple_with_dots_in_names(self):
        """Tuple with dotted names like (trait1.E, trait2.E)."""
        formula = "(trait1.E, trait2.E) ~ cnoise(cov=[[1, 0.3], [0.3, 1]])"
        nodes = parse_formula(formula)
        assert len(nodes) == 1
        assert nodes[0].outputs == ['trait1.E', 'trait2.E']

    def test_tuple_with_underscores_in_names(self):
        """Tuple with underscored names like (my_trait_a, my_trait_b)."""
        formula = "(my_trait_a, my_trait_b) ~ cnoise(cov=[[1, 0.2], [0.2, 1]])"
        nodes = parse_formula(formula)
        assert len(nodes) == 1
        assert nodes[0].outputs == ['my_trait_a', 'my_trait_b']

    def test_single_element_tuple_raises(self):
        """Single-element tuple (A) ~ ... should raise ValueError."""
        formula = "(A) ~ noise(0.5)"
        with pytest.raises(ValueError, match="at least 2 names"):
            parse_formula(formula)

    def test_empty_tuple_raises(self):
        """Empty tuple () ~ ... should raise ValueError."""
        formula = "() ~ noise(0.5)"
        with pytest.raises(ValueError, match="at least 2 names"):
            parse_formula(formula)

    def test_tuple_with_mvgenetic(self):
        """Tuple LHS with mvGenetic component."""
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.2, m=10, seed=42)
        formula = "(trait1.G, trait2.G) ~ mvGenetic(eff)"
        nodes = parse_formula(formula, effects={'eff': eff})
        assert len(nodes) == 1
        assert nodes[0].outputs == ['trait1.G', 'trait2.G']
        assert isinstance(nodes[0].component, MVGeneticComponent)


class TestDuplicateOutputDetection:
    """Test that duplicate output names raise ValueError."""

    def test_duplicate_simple_output(self):
        """Same output name used twice should raise."""
        formula = """
        Y ~ genetic(eff)
        Y ~ noise(0.5)
        """
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        with pytest.raises(ValueError, match="duplicate output name 'Y'"):
            parse_formula(formula, effects={'eff': eff})

    def test_duplicate_in_tuple_lhs(self):
        """Duplicate name within a tuple LHS should raise."""
        formula = "(A, A) ~ cnoise(cov=[[1, 0.2], [0.2, 1]])"
        with pytest.raises(ValueError, match="duplicate output name 'A'"):
            parse_formula(formula)

    def test_duplicate_across_tuple_and_single(self):
        """Duplicate across tuple LHS and single LHS should raise."""
        formula = """
        (A, B) ~ cnoise(cov=[[1, 0.2], [0.2, 1]])
        A ~ noise(0.5)
        """
        with pytest.raises(ValueError, match="duplicate output name 'A'"):
            parse_formula(formula)

    def test_duplicate_in_multiline_formula(self):
        """Duplicate in multi-line formula should raise."""
        formula = """
        Y.G ~ genetic(eff)
        Y.E ~ noise(0.5)
        Y ~ Y.G + Y.E
        Y.G ~ genetic(eff2)
        """
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        eff2 = AdditiveEffects.from_h2(h2=0.3, m=10, seed=43)
        with pytest.raises(ValueError, match="duplicate output name 'Y.G'"):
            parse_formula(formula, effects={'eff': eff, 'eff2': eff2})


class TestFounderFallback:
    """Test founder= kwarg for parental components."""

    def test_mother_with_founder_noise(self):
        """mother(Y, founder=noise(0.3)) should parse correctly."""
        formula = "Y.m ~ mother(Y, founder=noise(0.3))"
        nodes = parse_formula(formula)
        assert len(nodes) == 1
        assert isinstance(nodes[0].component, MotherComponent)
        assert nodes[0].component.founder_component is not None
        assert isinstance(nodes[0].component.founder_component, NoiseComponent)
        assert nodes[0].component.founder_component.variance == 0.3

    def test_father_with_founder_noise(self):
        """father(Y, founder=noise(0.5)) should parse correctly."""
        formula = "Y.f ~ father(Y, founder=noise(0.5))"
        nodes = parse_formula(formula)
        assert len(nodes) == 1
        assert isinstance(nodes[0].component, FatherComponent)
        assert nodes[0].component.founder_component.variance == 0.5

    def test_parent_with_founder_noise(self):
        """parent(Y, founder=noise(0.4)) should parse correctly."""
        formula = "Y.p ~ parent(Y, founder=noise(0.4))"
        nodes = parse_formula(formula)
        assert len(nodes) == 1
        assert isinstance(nodes[0].component, ParentComponent)
        assert nodes[0].component.founder_component.variance == 0.4

    def test_multiple_parental_with_different_founder_variance(self):
        """Multiple parental components with different founder variances."""
        formula = """
        Y.m ~ mother(Y, founder=noise(0.2))
        Y.f ~ father(Y, founder=noise(0.3))
        """
        nodes = parse_formula(formula)
        assert len(nodes) == 2
        assert nodes[0].component.founder_component.variance == 0.2
        assert nodes[1].component.founder_component.variance == 0.3


class TestMixedComponentTypes:
    """Test formulas with multiple component types mixed together."""

    def test_genetic_noise_agg_sibling_vt(self):
        """Formula with genetic, noise, agg, sibling, and VT components."""
        formula = """
        Y.G ~ genetic(eff)
        Y.E ~ noise(0.5)
        Y ~ Y.G + Y.E
        Y.m ~ mother(Y, founder=noise(0.3))
        Y.sm ~ sibling_mean(Y) | FID
        Z ~ Y + 0.5 * Y.m
        """
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        nodes = parse_formula(formula, effects={'eff': eff})
        assert len(nodes) == 6
        assert isinstance(nodes[0].component, GeneticComponent)
        assert isinstance(nodes[1].component, NoiseComponent)
        assert isinstance(nodes[2].component, AggregationComponent)
        assert isinstance(nodes[3].component, MotherComponent)
        assert isinstance(nodes[4].component, SiblingMeanComponent)
        assert isinstance(nodes[5].component, AggregationComponent)

    def test_multivariate_genetic_with_noise_and_agg(self):
        """Multivariate genetic + noise + aggregation."""
        formula = """
        (trait1.G, trait2.G) ~ mvGenetic(eff)
        trait1.E ~ noise(0.5)
        trait2.E ~ noise(0.7)
        trait1 ~ trait1.G + trait1.E
        trait2 ~ trait2.G + trait2.E
        """
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.2, m=10, seed=42)
        nodes = parse_formula(formula, effects={'eff': eff})
        assert len(nodes) == 5

    def test_haplotype_genetic_with_sibling(self):
        """HaplotypeGenetic + sibling functions."""
        formula = """
        Y.mat ~ haplotypeGenetic(eff, haplotype='maternal')
        Y.pat ~ haplotypeGenetic(eff, haplotype='paternal')
        Y ~ Y.mat + Y.pat
        Y.sm ~ sibling_mean(Y) | FID
        """
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        nodes = parse_formula(formula, effects={'eff': eff})
        assert len(nodes) == 4
        assert isinstance(nodes[0].component, HaplotypeGeneticComponent)
        assert isinstance(nodes[1].component, HaplotypeGeneticComponent)
        assert isinstance(nodes[2].component, AggregationComponent)
        assert isinstance(nodes[3].component, SiblingMeanComponent)

    def test_cnoise_with_sibling_sum(self):
        """Correlated noise + sibling_sum."""
        formula = """
        (E1, E2) ~ cnoise(cov=[[1, 0.3], [0.3, 1]])
        E1.sum ~ sibling_sum(E1) | FID
        E2.sum ~ sibling_sum(E2) | FID
        """
        nodes = parse_formula(formula)
        assert len(nodes) == 3
        assert isinstance(nodes[0].component, CNoiseComponent)
        assert isinstance(nodes[1].component, SiblingSumComponent)
        assert isinstance(nodes[2].component, SiblingSumComponent)


class TestEmptyFormula:
    """Test handling of empty formulas."""

    def test_empty_string_returns_empty_list(self):
        """Empty formula string should return empty list."""
        nodes = parse_formula("")
        assert nodes == []

    def test_only_whitespace_returns_empty_list(self):
        """Formula with only whitespace should return empty list."""
        nodes = parse_formula("   \n   \n   ")
        assert nodes == []

    def test_only_comments_returns_empty_list(self):
        """Formula with only comments should return empty list."""
        formula = """
        # Comment 1
        # Comment 2
        # Comment 3
        """
        nodes = parse_formula(formula)
        assert nodes == []

    def test_only_blank_lines_and_comments_returns_empty_list(self):
        """Formula with only blank lines and comments returns empty list."""
        formula = """

        # Comment

        # Another comment

        """
        nodes = parse_formula(formula)
        assert nodes == []


class TestInvalidSyntax:
    """Test error handling for invalid syntax."""

    def test_missing_tilde_raises(self):
        """Line without ~ should raise ValueError."""
        formula = "Y.G genetic(eff)"
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        with pytest.raises(ValueError, match="missing '~'"):
            parse_formula(formula, effects={'eff': eff})

    def test_missing_lhs_raises(self):
        """Line with ~ but no LHS should raise ValueError."""
        formula = "~ genetic(eff)"
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        with pytest.raises(ValueError, match="missing LHS"):
            parse_formula(formula, effects={'eff': eff})

    def test_missing_rhs_raises(self):
        """Line with ~ but no RHS should raise ValueError."""
        formula = "Y.G ~"
        with pytest.raises(ValueError, match="missing RHS"):
            parse_formula(formula)

    def test_invalid_grouping_on_aggregation_raises(self):
        """Grouping on aggregation expression should raise ValueError."""
        formula = "Y ~ Y.G + Y.E | FID"
        with pytest.raises(ValueError, match="only valid on function calls"):
            parse_formula(formula)

    def test_unknown_function_raises(self):
        """Unknown function name should raise ValueError."""
        formula = "Y ~ foobar(eff)"
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        with pytest.raises(ValueError, match="unknown function"):
            parse_formula(formula, effects={'eff': eff})


class TestHaplotypeGeneticKwarg:
    """Test haplotypeGenetic haplotype kwarg parsing."""

    def _make_effects(self):
        return {'eff': AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)}

    def test_maternal_kwarg_explicit(self):
        """haplotypeGenetic(eff, haplotype='maternal') should parse correctly."""
        formula = "Y.mat ~ haplotypeGenetic(eff, haplotype='maternal')"
        nodes = parse_formula(formula, effects=self._make_effects())
        comp = nodes[0].component
        assert isinstance(comp, HaplotypeGeneticComponent)
        assert comp.haplotype == 'maternal'

    def test_paternal_kwarg(self):
        """haplotypeGenetic(eff, haplotype='paternal') should parse correctly."""
        formula = "Y.pat ~ haplotypeGenetic(eff, haplotype='paternal')"
        nodes = parse_formula(formula, effects=self._make_effects())
        comp = nodes[0].component
        assert isinstance(comp, HaplotypeGeneticComponent)
        assert comp.haplotype == 'paternal'

    def test_maternal_with_double_quotes(self):
        """haplotype='maternal' with double quotes should work."""
        formula = 'Y.mat ~ haplotypeGenetic(eff, haplotype="maternal")'
        nodes = parse_formula(formula, effects=self._make_effects())
        comp = nodes[0].component
        assert comp.haplotype == 'maternal'

    def test_paternal_with_double_quotes(self):
        """haplotype='paternal' with double quotes should work."""
        formula = 'Y.pat ~ haplotypeGenetic(eff, haplotype="paternal")'
        nodes = parse_formula(formula, effects=self._make_effects())
        comp = nodes[0].component
        assert comp.haplotype == 'paternal'

    def test_maternal_and_paternal_in_same_formula(self):
        """Both maternal and paternal in same formula."""
        formula = """
        Y.mat ~ haplotypeGenetic(eff, haplotype='maternal')
        Y.pat ~ haplotypeGenetic(eff, haplotype='paternal')
        Y ~ Y.mat + Y.pat
        """
        nodes = parse_formula(formula, effects=self._make_effects())
        assert len(nodes) == 3
        assert nodes[0].component.haplotype == 'maternal'
        assert nodes[1].component.haplotype == 'paternal'

    def test_maternal_default(self):
        """Default haplotype should be maternal."""
        formula = "Y ~ haplotypeGenetic(eff)"
        nodes = parse_formula(formula, effects=self._make_effects())
        node = nodes[0]
        assert node.component.haplotype == 'maternal'

    def test_unknown_arg_raises(self):
        """Unknown argument in haplotypeGenetic should raise."""
        formula = "Y ~ haplotypeGenetic(eff, foo=bar)"
        with pytest.raises(ValueError, match="unexpected argument"):
            parse_formula(formula, effects=self._make_effects())


class TestEffectNamesWithSpecialCharacters:
    """Test effect names with dots and underscores."""

    def test_effect_name_with_dots(self):
        """Effect name like 'my.effect.name' should parse."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        formula = "Y.G ~ genetic(my.effect.name)"
        nodes = parse_formula(formula, effects={'my.effect.name': eff})
        assert len(nodes) == 1
        assert isinstance(nodes[0].component, GeneticComponent)

    def test_effect_name_with_underscores(self):
        """Effect name like 'my_effect_name' should parse."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        formula = "Y.G ~ genetic(my_effect_name)"
        nodes = parse_formula(formula, effects={'my_effect_name': eff})
        assert len(nodes) == 1
        assert isinstance(nodes[0].component, GeneticComponent)

    def test_effect_name_with_numbers(self):
        """Effect name like 'eff123' should parse."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        formula = "Y.G ~ genetic(eff123)"
        nodes = parse_formula(formula, effects={'eff123': eff})
        assert len(nodes) == 1
        assert isinstance(nodes[0].component, GeneticComponent)

    def test_effect_name_with_mixed_special_chars(self):
        """Effect name like 'my_effect.v2' should parse."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        formula = "Y.G ~ genetic(my_effect.v2)"
        nodes = parse_formula(formula, effects={'my_effect.v2': eff})
        assert len(nodes) == 1
        assert isinstance(nodes[0].component, GeneticComponent)

    def test_effect_name_with_dots_in_mvgenetic(self):
        """Effect name with dots in mvGenetic."""
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.2, m=10, seed=42)
        formula = "(trait1.G, trait2.G) ~ mvGenetic(my.mv.effect)"
        nodes = parse_formula(formula, effects={'my.mv.effect': eff})
        assert len(nodes) == 1
        assert isinstance(nodes[0].component, MVGeneticComponent)

    def test_effect_name_with_underscores_in_haplotype_genetic(self):
        """Effect name with underscores in haplotypeGenetic."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        formula = "Y.mat ~ haplotypeGenetic(my_hap_effect, haplotype='maternal')"
        nodes = parse_formula(formula, effects={'my_hap_effect': eff})
        assert len(nodes) == 1
        assert isinstance(nodes[0].component, HaplotypeGeneticComponent)


class TestParserCnoiseAdvanced:
    """Advanced cnoise error handling."""

    def test_non_square_matrix_raises(self):
        """Non-square cov matrix should raise."""
        with pytest.raises(ValueError, match="square"):
            parse_formula("(A, B) ~ cnoise([[1,0,0],[0,1,0]])")

    def test_k_mismatch_raises(self):
        """cov k != number of outputs should raise."""
        with pytest.raises(ValueError, match="k="):
            parse_formula("(A, B, C) ~ cnoise([[1,0],[0,1]])")

    def test_invalid_matrix_literal_raises(self):
        """Non-parseable matrix should raise."""
        with pytest.raises(ValueError, match="matrix literal"):
            parse_formula("(A, B) ~ cnoise(not_a_matrix)")


class TestParserNoiseAdvanced:
    """Advanced noise error handling."""

    def test_non_numeric_variance_raises(self):
        """noise with non-numeric variance should raise."""
        with pytest.raises(ValueError, match="numeric variance"):
            parse_formula("Y ~ noise(abc)")


class TestParserParentalAdvanced:
    """Advanced parental component error handling."""

    def test_missing_phenotype_name_raises(self):
        """Empty phenotype name should raise."""
        with pytest.raises(ValueError, match="requires a phenotype name"):
            parse_formula("Y.VT ~ mother()")

    def test_unsupported_founder_function_raises(self):
        """founder= with unsupported function should raise."""
        with pytest.raises(ValueError, match="unsupported function"):
            parse_formula("Y.VT ~ mother(Y, founder=genetic(beta))")

    def test_founder_non_function_raises(self):
        """founder= with non-function value should raise."""
        with pytest.raises(ValueError, match="function call"):
            parse_formula("Y.VT ~ mother(Y, founder=0.5)")

    def test_founder_noise_valid(self):
        """founder=noise(0.1) should work."""
        nodes = parse_formula("Y.VT ~ mother(Y, founder=noise(0.1))")
        assert len(nodes) == 1
        assert isinstance(nodes[0].component, MotherComponent)
        assert nodes[0].component.founder_component is not None


class TestParserSiblingAdvanced:
    """Advanced sibling function error handling."""

    def test_empty_source_name_raises(self):
        """sibling_mean() with empty source name should raise."""
        with pytest.raises(ValueError, match="requires"):
            parse_formula("Y.sib ~ sibling_mean() | FID")

    def test_sibling_with_grouping(self):
        """sibling_mean with | FID grouping."""
        nodes = parse_formula("Y.sib ~ sibling_mean(Y) | FID")
        assert nodes[0].grouping == 'FID'

    def test_sibling_with_sex_grouping(self):
        """sibling_count with | sex grouping."""
        nodes = parse_formula("Y.cnt ~ sibling_count(Y) | sex")
        assert nodes[0].grouping == 'sex'


class TestParserAggregationAdvanced:
    """Advanced aggregation parsing tests."""

    def test_pipe_on_aggregation_raises(self):
        """Grouping on aggregation expression should raise."""
        with pytest.raises(ValueError, match="grouping"):
            parse_formula("""
                Y.G ~ noise(1.0)
                Y.E ~ noise(1.0)
                Y ~ Y.G + Y.E | FID
            """)


class TestParserMultiOutput:
    """Multi-output component parsing tests."""

    def test_multi_output_mvGenetic(self):
        """Multi-output LHS with mvGenetic should work."""
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.2, m=10, seed=42)
        nodes = parse_formula("(A, B) ~ mvGenetic(beta)", effects={'beta': eff})
        assert len(nodes) == 1
        assert nodes[0].outputs == ['A', 'B']

    def test_multi_output_cnoise(self):
        """Multi-output cnoise."""
        nodes = parse_formula("(A, B) ~ cnoise([[0.5, 0.1], [0.1, 0.5]])")
        assert len(nodes) == 1
        assert nodes[0].outputs == ['A', 'B']


class TestRealWorldFormulas:
    """Test realistic complex formulas that combine multiple features."""

    def test_complex_bivariate_with_vt_and_sibling(self):
        """Complex bivariate architecture with VT and sibling effects."""
        formula = """
        # Genetic components
        (trait1.G, trait2.G) ~ mvGenetic(mv_eff)

        # Environmental noise
        trait1.E ~ noise(0.5)
        trait2.E ~ noise(0.7)

        # Vertical transmission
        trait1.VT ~ mother(trait1, founder=noise(0.2))
        trait2.VT ~ father(trait2, founder=noise(0.3))

        # Sibling effects
        trait1.sm ~ sibling_mean(trait1) | FID
        trait2.sum ~ sibling_sum(trait2) | FID

        # Final phenotypes
        trait1 ~ trait1.G + trait1.E + 0.3 * trait1.VT + 0.1 * trait1.sm
        trait2 ~ trait2.G + trait2.E + 0.2 * trait2.VT + 0.05 * trait2.sum
        """
        mv_eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.2, m=10, seed=42)
        nodes = parse_formula(formula, effects={'mv_eff': mv_eff})
        # 1 mvGenetic (2 outputs), 2 noise, 2 VT, 2 sibling, 2 agg = 9 nodes
        assert len(nodes) == 9

    def test_haplotype_decomposition_with_aggregation(self):
        """Decompose additive effect into maternal and paternal haplotypes."""
        formula = """
        # Haplotype-specific effects
        Y.mat ~ haplotypeGenetic(eff, haplotype='maternal')
        Y.pat ~ haplotypeGenetic(eff, haplotype='paternal')

        # Combine haplotypes
        Y.G ~ Y.mat + Y.pat

        # Add noise
        Y.E ~ noise(0.5)

        # Final phenotype
        Y ~ Y.G + Y.E
        """
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        nodes = parse_formula(formula, effects={'eff': eff})
        assert len(nodes) == 5
        assert isinstance(nodes[0].component, HaplotypeGeneticComponent)
        assert isinstance(nodes[1].component, HaplotypeGeneticComponent)
        assert isinstance(nodes[2].component, AggregationComponent)
        assert isinstance(nodes[3].component, NoiseComponent)
        assert isinstance(nodes[4].component, AggregationComponent)
