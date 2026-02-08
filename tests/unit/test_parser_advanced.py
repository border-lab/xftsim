"""
Unit tests for parser.py advanced edge cases.

Tests:
1. _extract_grouping: pipe inside parens vs outside, empty grouping, invalid grouping
2. haplotypeGenetic: unknown arg raises, paternal haplotype, missing effect
3. cnoise: non-square matrix, k mismatch, invalid matrix literal
4. noise: non-numeric variance
5. parental: missing name, unsupported founder function, empty founder
6. sibling: empty source name
7. aggregation: pipe on aggregation raises
8. unknown function
9. multi-output genetic/mvGenetic
"""
import numpy as np
import pytest

from xftsim.narch import Architecture, HaplotypeGeneticComponent
from xftsim.neffect import AdditiveEffects, MultivariateEffects
from xftsim.parser import parse_formula, _extract_grouping


class TestExtractGrouping:
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


class TestParserHaplotypeGeneticAdvanced:
    def test_unknown_arg_raises(self):
        """Unknown argument in haplotypeGenetic should raise."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        with pytest.raises(ValueError, match="unexpected argument"):
            Architecture.from_formula("""
                Y ~ haplotypeGenetic(beta, foo=bar)
            """, effects={'beta': eff})

    def test_paternal_haplotype(self):
        """haplotype='paternal' should work."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        arch = Architecture.from_formula("""
            Y ~ haplotypeGenetic(beta, haplotype='paternal')
        """, effects={'beta': eff})
        node = arch._nodes[0]
        assert isinstance(node.component, HaplotypeGeneticComponent)
        assert node.component.haplotype == 'paternal'

    def test_maternal_haplotype_default(self):
        """Default haplotype should be maternal."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        arch = Architecture.from_formula("""
            Y ~ haplotypeGenetic(beta)
        """, effects={'beta': eff})
        node = arch._nodes[0]
        assert node.component.haplotype == 'maternal'


class TestParserCnoiseAdvanced:
    def test_non_square_matrix_raises(self):
        """Non-square cov matrix should raise."""
        with pytest.raises(ValueError, match="square"):
            Architecture.from_formula("""
                (A, B) ~ cnoise([[1,0,0],[0,1,0]])
            """)

    def test_k_mismatch_raises(self):
        """cov k != number of outputs should raise."""
        with pytest.raises(ValueError, match="k="):
            Architecture.from_formula("""
                (A, B, C) ~ cnoise([[1,0],[0,1]])
            """)

    def test_invalid_matrix_literal_raises(self):
        """Non-parseable matrix should raise."""
        with pytest.raises(ValueError, match="matrix literal"):
            Architecture.from_formula("""
                (A, B) ~ cnoise(not_a_matrix)
            """)


class TestParserNoiseAdvanced:
    def test_non_numeric_variance_raises(self):
        """noise with non-numeric variance should raise."""
        with pytest.raises(ValueError, match="numeric variance"):
            Architecture.from_formula("""
                Y ~ noise(abc)
            """)


class TestParserParentalAdvanced:
    def test_missing_phenotype_name_raises(self):
        """Empty phenotype name should raise."""
        with pytest.raises(ValueError, match="requires a phenotype name"):
            Architecture.from_formula("""
                Y.VT ~ mother()
            """)

    def test_unsupported_founder_function_raises(self):
        """founder= with unsupported function should raise."""
        with pytest.raises(ValueError, match="unsupported function"):
            Architecture.from_formula("""
                Y.VT ~ mother(Y, founder=genetic(beta))
            """)

    def test_founder_non_function_raises(self):
        """founder= with non-function value should raise."""
        with pytest.raises(ValueError, match="function call"):
            Architecture.from_formula("""
                Y.VT ~ mother(Y, founder=0.5)
            """)

    def test_founder_noise_valid(self):
        """founder=noise(0.1) should work."""
        arch = Architecture.from_formula("""
            Y.VT ~ mother(Y, founder=noise(0.1))
        """)
        assert len(arch._nodes) == 1
        from xftsim.narch import MotherComponent
        assert isinstance(arch._nodes[0].component, MotherComponent)
        assert arch._nodes[0].component.founder_component is not None


class TestParserSiblingAdvanced:
    def test_empty_source_name_raises(self):
        """sibling_mean() with empty source name should raise."""
        with pytest.raises(ValueError, match="requires"):
            Architecture.from_formula("""
                Y.sib ~ sibling_mean() | FID
            """)

    def test_sibling_with_grouping(self):
        """sibling_mean with | FID grouping."""
        arch = Architecture.from_formula("""
            Y ~ noise(1.0)
            Y.sib ~ sibling_mean(Y) | FID
        """)
        sib_node = [n for n in arch._nodes if 'Y.sib' in n.outputs][0]
        assert sib_node.grouping == 'FID'

    def test_sibling_with_sex_grouping(self):
        """sibling_count with | sex grouping."""
        arch = Architecture.from_formula("""
            Y ~ noise(1.0)
            Y.cnt ~ sibling_count(Y) | sex
        """)
        cnt_node = [n for n in arch._nodes if 'Y.cnt' in n.outputs][0]
        assert cnt_node.grouping == 'sex'


class TestParserAggregationAdvanced:
    def test_pipe_on_aggregation_raises(self):
        """Grouping on aggregation expression should raise."""
        with pytest.raises(ValueError, match="grouping"):
            Architecture.from_formula("""
                Y.G ~ noise(1.0)
                Y.E ~ noise(1.0)
                Y ~ Y.G + Y.E | FID
            """)


class TestParserUnknownFunctionAdvanced:
    def test_unknown_function_raises(self):
        """Unknown function name should raise."""
        with pytest.raises(ValueError, match="unknown function"):
            Architecture.from_formula("""
                Y ~ foobar(1.0)
            """)


class TestParserMultiOutput:
    def test_multi_output_mvGenetic(self):
        """Multi-output LHS with mvGenetic should work."""
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.2, m=10, seed=42)
        arch = Architecture.from_formula("""
            (A, B) ~ mvGenetic(beta)
        """, effects={'beta': eff})
        assert len(arch._nodes) == 1
        assert arch._nodes[0].outputs == ['A', 'B']

    def test_multi_output_cnoise(self):
        """Multi-output cnoise."""
        arch = Architecture.from_formula("""
            (A, B) ~ cnoise([[0.5, 0.1], [0.1, 0.5]])
        """)
        assert len(arch._nodes) == 1
        assert arch._nodes[0].outputs == ['A', 'B']
