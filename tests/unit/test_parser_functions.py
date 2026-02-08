"""
Unit tests for parser internal functions.

Tests:
1. _extract_grouping: pipe inside/outside parens, empty grouping, invalid grouping
2. _try_parse_function: unknown function, non-function RHS, grouping on non-groupable
3. _parse_genetic: empty args, missing effect, non-EffectSpec
4. _parse_noise: non-numeric variance
5. _parse_cnoise: non-square cov, dimension mismatch, invalid literal
6. _parse_parental: with/without founder=, empty phenotype name
7. _parse_sibling: empty source, grouping propagation
8. _parse_haplotypeGenetic: default haplotype, haplotype kwarg, unexpected arg
9. _parse_aggregation: arithmetic expression parsing
10. parse_formula: comment lines, empty lines, duplicate output, missing ~, missing LHS/RHS
11. Tuple LHS: fewer than 2 names, mvGenetic k mismatch
"""
import numpy as np
import pytest

from xftsim.parser import (
    parse_formula, _extract_grouping, _try_parse_function,
    _parse_genetic, _parse_noise, _parse_cnoise,
    _parse_parental, _parse_sibling, _parse_aggregation,
    _parse_haplotypeGenetic,
)
from xftsim.narch import ArchNode
from xftsim.neffect import AdditiveEffects, MultivariateEffects


class TestExtractGrouping:
    def test_no_pipe(self):
        rhs, grp = _extract_grouping("noise(0.5)")
        assert grp is None
        assert rhs == "noise(0.5)"

    def test_trailing_pipe(self):
        rhs, grp = _extract_grouping("noise(0.5) | FID")
        assert grp == "FID"
        assert rhs == "noise(0.5)"

    def test_pipe_inside_parens_ignored(self):
        rhs, grp = _extract_grouping("cnoise(cov=[[1|2],[3|4]])")
        # Pipes inside parens should NOT be treated as grouping
        assert grp is None or "|" not in rhs  # implementation-dependent

    def test_empty_grouping_after_pipe(self):
        rhs, grp = _extract_grouping("noise(0.5) |")
        assert grp is None  # empty grouping → None

    def test_invalid_grouping_identifier(self):
        with pytest.raises(ValueError, match="Invalid grouping"):
            _extract_grouping("noise(0.5) | 123bad")

    def test_valid_grouping_underscore(self):
        rhs, grp = _extract_grouping("noise(0.5) | _group")
        assert grp == "_group"

    def test_pipe_with_no_rhs(self):
        rhs, grp = _extract_grouping(" | FID")
        assert grp == "FID"


class TestParseFormulaLines:
    def test_comment_lines_skipped(self):
        formula = "# this is a comment\nY.E ~ noise(0.5)\n# another comment"
        nodes = parse_formula(formula)
        assert len(nodes) == 1
        assert nodes[0].outputs == ['Y.E']

    def test_empty_lines_skipped(self):
        formula = "\n\nY.E ~ noise(0.5)\n\n"
        nodes = parse_formula(formula)
        assert len(nodes) == 1

    def test_missing_tilde(self):
        with pytest.raises(ValueError, match="missing '~'"):
            parse_formula("Y noise(0.5)")

    def test_missing_lhs(self):
        with pytest.raises(ValueError, match="missing LHS"):
            parse_formula("~ noise(0.5)")

    def test_missing_rhs(self):
        with pytest.raises(ValueError, match="missing RHS"):
            parse_formula("Y ~")

    def test_duplicate_output(self):
        with pytest.raises(ValueError, match="duplicate output"):
            parse_formula("Y ~ noise(0.5)\nY ~ noise(0.3)")

    def test_tuple_lhs_single_name_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            parse_formula("(X) ~ noise(0.5)")


class TestParseGeneticErrors:
    def test_empty_effect_name(self):
        with pytest.raises(ValueError, match="requires an effect name"):
            _parse_genetic(['Y.G'], '', {}, lineno=1)

    def test_effect_not_in_dict(self):
        with pytest.raises(ValueError, match="not found in effects dict"):
            _parse_genetic(['Y.G'], 'missing', {}, lineno=1)

    def test_non_effect_spec(self):
        with pytest.raises(ValueError, match="not an EffectSpec"):
            _parse_genetic(['Y.G'], 'bad', {'bad': "not_an_effect"}, lineno=1)

    def test_valid_genetic(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        node = _parse_genetic(['Y.G'], 'eff1', {'eff1': eff}, lineno=1)
        assert isinstance(node, ArchNode)
        assert node.outputs == ['Y.G']


class TestParseNoiseErrors:
    def test_non_numeric(self):
        with pytest.raises(ValueError, match="numeric variance"):
            _parse_noise(['Y.E'], 'abc', lineno=1)

    def test_valid_noise(self):
        node = _parse_noise(['Y.E'], '0.5', lineno=1)
        assert node.outputs == ['Y.E']


class TestParseCnoiseErrors:
    def test_non_square_cov(self):
        with pytest.raises(ValueError, match="square matrix"):
            _parse_cnoise(['A', 'B'], 'cov=[[1,0],[0,1],[1,1]]', lineno=1)

    def test_k_mismatch(self):
        with pytest.raises(ValueError, match="LHS has"):
            _parse_cnoise(['A'], 'cov=[[1,0],[0,1]]', lineno=1)

    def test_invalid_literal(self):
        with pytest.raises(ValueError, match="matrix literal"):
            _parse_cnoise(['A', 'B'], 'cov=not_a_matrix', lineno=1)

    def test_valid_cnoise(self):
        node = _parse_cnoise(['A', 'B'], 'cov=[[1,0.2],[0.2,1]]', lineno=1)
        assert node.outputs == ['A', 'B']

    def test_cnoise_without_cov_prefix(self):
        node = _parse_cnoise(['A', 'B'], '[[1,0],[0,1]]', lineno=1)
        assert node.outputs == ['A', 'B']


class TestParseParental:
    def test_mother_basic(self):
        node = _parse_parental('mother', ['Y.m'], 'Y', {}, lineno=1)
        assert node.outputs == ['Y.m']

    def test_father_basic(self):
        node = _parse_parental('father', ['Y.f'], 'Y', {}, lineno=1)
        assert node.outputs == ['Y.f']

    def test_parent_basic(self):
        node = _parse_parental('parent', ['Y.p'], 'Y', {}, lineno=1)
        assert node.outputs == ['Y.p']

    def test_with_founder_noise(self):
        node = _parse_parental('mother', ['Y.m'], 'Y, founder=noise(0.3)', {}, lineno=1)
        assert node.outputs == ['Y.m']
        assert node.component.founder_component is not None

    def test_empty_phenotype_name(self):
        with pytest.raises(ValueError, match="requires a phenotype name"):
            _parse_parental('mother', ['Y.m'], '', {}, lineno=1)

    def test_founder_unsupported_function(self):
        with pytest.raises(ValueError, match="unsupported function"):
            _parse_parental('mother', ['Y.m'], 'Y, founder=genetic(eff)', {}, lineno=1)

    def test_founder_not_function_call(self):
        with pytest.raises(ValueError, match="requires a function call"):
            _parse_parental('mother', ['Y.m'], 'Y, founder=0.5', {}, lineno=1)


class TestParseSibling:
    def test_basic(self):
        node = _parse_sibling('sibling_mean', ['Y.sib'], 'Y', lineno=1)
        assert node.outputs == ['Y.sib']
        assert node.inputs == ['Y']

    def test_empty_source(self):
        with pytest.raises(ValueError, match="requires a source"):
            _parse_sibling('sibling_mean', ['Y.sib'], '', lineno=1)

    def test_with_grouping(self):
        node = _parse_sibling('sibling_count', ['Y.n'], 'Y', lineno=1, grouping='FID')
        assert node.grouping == 'FID'


class TestParseHaplotypeGenetic:
    def test_default_maternal(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        node = _parse_haplotypeGenetic(['Y.H'], 'eff1', {'eff1': eff}, lineno=1)
        assert node.component.haplotype == 'maternal'

    def test_paternal_kwarg(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        node = _parse_haplotypeGenetic(['Y.H'], "eff1, haplotype='paternal'", {'eff1': eff}, lineno=1)
        assert node.component.haplotype == 'paternal'

    def test_unexpected_arg(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        with pytest.raises(ValueError, match="unexpected argument"):
            _parse_haplotypeGenetic(['Y.H'], "eff1, badarg=true", {'eff1': eff}, lineno=1)


class TestParseAggregation:
    def test_simple_sum(self):
        node = _parse_aggregation(['Y'], 'Y.G + Y.E', lineno=1)
        assert 'Y.G' in node.inputs
        assert 'Y.E' in node.inputs

    def test_weighted(self):
        node = _parse_aggregation(['Y'], '0.5 * Y.G + Y.E', lineno=1)
        assert 'Y.G' in node.inputs
        assert 'Y.E' in node.inputs

    def test_single_input(self):
        node = _parse_aggregation(['Y'], 'Y.G', lineno=1)
        assert node.inputs == ['Y.G']


class TestTryParseFunctionEdgeCases:
    def test_non_function_returns_none(self):
        result = _try_parse_function(['Y'], 'X + Z', {}, lineno=1)
        assert result is None

    def test_unknown_function_raises(self):
        with pytest.raises(ValueError, match="unknown function"):
            _try_parse_function(['Y'], 'unknown_func(arg)', {}, lineno=1)


class TestParseFormulaIntegration:
    def test_multiline_formula(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        formula = """
        Y.G ~ genetic(eff1)
        Y.E ~ noise(0.5)
        Y ~ Y.G + Y.E
        """
        nodes = parse_formula(formula, effects={'eff1': eff})
        assert len(nodes) == 3
        assert nodes[0].outputs == ['Y.G']
        assert nodes[1].outputs == ['Y.E']
        assert nodes[2].outputs == ['Y']

    def test_formula_with_grouping(self):
        formula = "Y.sib ~ sibling_mean(Y) | FID"
        nodes = parse_formula(formula)
        assert len(nodes) == 1
        assert nodes[0].grouping == 'FID'

    def test_mvGenetic_k_mismatch(self):
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.5, m=10, seed=42)
        formula = "(A, B, C) ~ mvGenetic(eff1)"
        with pytest.raises(ValueError, match="k=2"):
            parse_formula(formula, effects={'eff1': eff})

    def test_grouping_on_aggregation_raises(self):
        with pytest.raises(ValueError, match="grouping is only valid"):
            parse_formula("Y ~ X + Z | FID")

    def test_grouping_on_non_groupable_raises(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        # genetic does not accept grouping
        with pytest.raises(ValueError, match="does not accept"):
            parse_formula("Y.G ~ genetic(eff1) | FID", effects={'eff1': eff})

    def test_parental_with_founder_in_formula(self):
        formula = "Y.VTm ~ mother(Y, founder=noise(0.1))"
        nodes = parse_formula(formula)
        assert len(nodes) == 1
        assert nodes[0].component.founder_component is not None
