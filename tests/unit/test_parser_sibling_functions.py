"""
Unit tests for parser handling of ALL sibling functions.

Tests cover:
1. Each sibling function parses correctly (6 tests - one per function)
2. Empty source name validation
3. Grouping with | FID
4. No grouping (defaults to FID)
5. Unknown sibling function error handling
"""
import pytest

from xftsim.parser import parse_formula
from xftsim.arch import (
    SiblingMeanComponent,
    SiblingSumComponent,
    SiblingAnyComponent,
    SiblingCountComponent,
    SiblingEldestComponent,
    SiblingYoungestComponent,
)


class TestSiblingMeanParsing:
    """Test parsing of sibling_mean()."""

    def test_sibling_mean_parses(self):
        nodes = parse_formula("Y.sm ~ sibling_mean(Y)")
        assert len(nodes) == 1
        assert isinstance(nodes[0].component, SiblingMeanComponent)
        assert nodes[0].component.source_name == 'Y'
        assert nodes[0].inputs == ['Y']
        assert nodes[0].outputs == ['Y.sm']

    def test_sibling_mean_with_grouping(self):
        nodes = parse_formula("Y.sm ~ sibling_mean(Y) | FID")
        assert len(nodes) == 1
        assert isinstance(nodes[0].component, SiblingMeanComponent)
        assert nodes[0].component.source_name == 'Y'
        assert nodes[0].grouping == 'FID'

    def test_sibling_mean_no_grouping_defaults_to_fid(self):
        """When no grouping is specified, component should still work (default FID)."""
        nodes = parse_formula("Y.sm ~ sibling_mean(Y)")
        assert len(nodes) == 1
        assert isinstance(nodes[0].component, SiblingMeanComponent)
        # No explicit grouping in the ArchNode, but component handles default
        assert nodes[0].grouping is None  # Parser doesn't set default, component does


class TestSiblingSumParsing:
    """Test parsing of sibling_sum()."""

    def test_sibling_sum_parses(self):
        nodes = parse_formula("Y.sum ~ sibling_sum(Y)")
        assert len(nodes) == 1
        assert isinstance(nodes[0].component, SiblingSumComponent)
        assert nodes[0].component.source_name == 'Y'
        assert nodes[0].inputs == ['Y']
        assert nodes[0].outputs == ['Y.sum']

    def test_sibling_sum_with_grouping(self):
        nodes = parse_formula("Y.sum ~ sibling_sum(Y) | FID")
        assert len(nodes) == 1
        assert isinstance(nodes[0].component, SiblingSumComponent)
        assert nodes[0].component.source_name == 'Y'
        assert nodes[0].grouping == 'FID'


class TestSiblingAnyParsing:
    """Test parsing of sibling_any()."""

    def test_sibling_any_parses(self):
        nodes = parse_formula("Y.any ~ sibling_any(Y)")
        assert len(nodes) == 1
        assert isinstance(nodes[0].component, SiblingAnyComponent)
        assert nodes[0].component.source_name == 'Y'
        assert nodes[0].inputs == ['Y']
        assert nodes[0].outputs == ['Y.any']

    def test_sibling_any_with_grouping(self):
        nodes = parse_formula("Y.any ~ sibling_any(Y) | FID")
        assert len(nodes) == 1
        assert isinstance(nodes[0].component, SiblingAnyComponent)
        assert nodes[0].component.source_name == 'Y'
        assert nodes[0].grouping == 'FID'


class TestSiblingCountParsing:
    """Test parsing of sibling_count()."""

    def test_sibling_count_parses(self):
        nodes = parse_formula("Y.cnt ~ sibling_count(Y)")
        assert len(nodes) == 1
        assert isinstance(nodes[0].component, SiblingCountComponent)
        assert nodes[0].component.source_name == 'Y'
        assert nodes[0].inputs == ['Y']
        assert nodes[0].outputs == ['Y.cnt']

    def test_sibling_count_with_grouping(self):
        nodes = parse_formula("Y.cnt ~ sibling_count(Y) | FID")
        assert len(nodes) == 1
        assert isinstance(nodes[0].component, SiblingCountComponent)
        assert nodes[0].component.source_name == 'Y'
        assert nodes[0].grouping == 'FID'


class TestSiblingEldestParsing:
    """Test parsing of sibling_eldest()."""

    def test_sibling_eldest_parses(self):
        nodes = parse_formula("Y.eldest ~ sibling_eldest(Y)")
        assert len(nodes) == 1
        assert isinstance(nodes[0].component, SiblingEldestComponent)
        assert nodes[0].component.source_name == 'Y'
        assert nodes[0].inputs == ['Y']
        assert nodes[0].outputs == ['Y.eldest']

    def test_sibling_eldest_with_grouping(self):
        nodes = parse_formula("Y.eldest ~ sibling_eldest(Y) | FID")
        assert len(nodes) == 1
        assert isinstance(nodes[0].component, SiblingEldestComponent)
        assert nodes[0].component.source_name == 'Y'
        assert nodes[0].grouping == 'FID'


class TestSiblingYoungestParsing:
    """Test parsing of sibling_youngest()."""

    def test_sibling_youngest_parses(self):
        nodes = parse_formula("Y.youngest ~ sibling_youngest(Y)")
        assert len(nodes) == 1
        assert isinstance(nodes[0].component, SiblingYoungestComponent)
        assert nodes[0].component.source_name == 'Y'
        assert nodes[0].inputs == ['Y']
        assert nodes[0].outputs == ['Y.youngest']

    def test_sibling_youngest_with_grouping(self):
        nodes = parse_formula("Y.youngest ~ sibling_youngest(Y) | FID")
        assert len(nodes) == 1
        assert isinstance(nodes[0].component, SiblingYoungestComponent)
        assert nodes[0].component.source_name == 'Y'
        assert nodes[0].grouping == 'FID'


class TestSiblingFunctionErrors:
    """Test error handling for sibling functions."""

    def test_empty_source_name_sibling_mean(self):
        """Empty source name should raise ValueError."""
        with pytest.raises(ValueError, match="requires a source component name"):
            parse_formula("Y.sm ~ sibling_mean()")

    def test_empty_source_name_sibling_sum(self):
        """Empty source name should raise ValueError."""
        with pytest.raises(ValueError, match="requires a source component name"):
            parse_formula("Y.sum ~ sibling_sum()")

    def test_empty_source_name_sibling_any(self):
        """Empty source name should raise ValueError."""
        with pytest.raises(ValueError, match="requires a source component name"):
            parse_formula("Y.any ~ sibling_any()")

    def test_empty_source_name_sibling_count(self):
        """Empty source name should raise ValueError."""
        with pytest.raises(ValueError, match="requires a source component name"):
            parse_formula("Y.cnt ~ sibling_count()")

    def test_empty_source_name_sibling_eldest(self):
        """Empty source name should raise ValueError."""
        with pytest.raises(ValueError, match="requires a source component name"):
            parse_formula("Y.eldest ~ sibling_eldest()")

    def test_empty_source_name_sibling_youngest(self):
        """Empty source name should raise ValueError."""
        with pytest.raises(ValueError, match="requires a source component name"):
            parse_formula("Y.youngest ~ sibling_youngest()")

    def test_unknown_sibling_function(self):
        """Unknown sibling function should raise unknown function error."""
        with pytest.raises(ValueError, match="unknown function"):
            parse_formula("Y.out ~ sibling_median(Y)")


class TestSiblingGroupingVariations:
    """Test various grouping patterns with sibling functions."""

    def test_sibling_mean_grouping_with_spaces(self):
        """Grouping with spaces should parse correctly."""
        nodes = parse_formula("Y.sm ~ sibling_mean(Y) | FID")
        assert len(nodes) == 1
        assert isinstance(nodes[0].component, SiblingMeanComponent)
        assert nodes[0].grouping == 'FID'

    def test_sibling_sum_grouping_mother(self):
        """Grouping by mother should parse correctly."""
        nodes = parse_formula("Y.sum ~ sibling_sum(Y) | mother")
        assert len(nodes) == 1
        assert isinstance(nodes[0].component, SiblingSumComponent)
        assert nodes[0].grouping == 'mother'

    def test_sibling_any_grouping_father(self):
        """Grouping by father should parse correctly."""
        nodes = parse_formula("Y.any ~ sibling_any(Y) | father")
        assert len(nodes) == 1
        assert isinstance(nodes[0].component, SiblingAnyComponent)
        assert nodes[0].grouping == 'father'

    def test_sibling_count_grouping_custom(self):
        """Grouping by custom field should parse correctly."""
        nodes = parse_formula("Y.cnt ~ sibling_count(Y) | school_id")
        assert len(nodes) == 1
        assert isinstance(nodes[0].component, SiblingCountComponent)
        assert nodes[0].grouping == 'school_id'


class TestSiblingFunctionSourceNames:
    """Test that source names are correctly captured."""

    def test_source_name_with_dots(self):
        """Source names with dots should be captured correctly."""
        nodes = parse_formula("Y.result ~ sibling_mean(Y.phenotype)")
        assert len(nodes) == 1
        assert nodes[0].component.source_name == 'Y.phenotype'
        assert nodes[0].inputs == ['Y.phenotype']

    def test_source_name_with_underscores(self):
        """Source names with underscores should be captured correctly."""
        nodes = parse_formula("result ~ sibling_sum(some_phenotype)")
        assert len(nodes) == 1
        assert nodes[0].component.source_name == 'some_phenotype'
        assert nodes[0].inputs == ['some_phenotype']

    def test_source_name_numeric_suffix(self):
        """Source names with numeric suffixes should be captured correctly."""
        nodes = parse_formula("out ~ sibling_eldest(trait123)")
        assert len(nodes) == 1
        assert nodes[0].component.source_name == 'trait123'
        assert nodes[0].inputs == ['trait123']


class TestMultipleSiblingFunctions:
    """Test parsing multiple sibling functions in one formula."""

    def test_two_different_sibling_functions(self):
        """Two different sibling functions should both parse."""
        formula = """
        Y.mean ~ sibling_mean(Y)
        Y.sum ~ sibling_sum(Y)
        """
        nodes = parse_formula(formula)
        assert len(nodes) == 2
        assert isinstance(nodes[0].component, SiblingMeanComponent)
        assert isinstance(nodes[1].component, SiblingSumComponent)

    def test_all_six_sibling_functions(self):
        """All six sibling functions in one formula."""
        formula = """
        Y.mean ~ sibling_mean(Y)
        Y.sum ~ sibling_sum(Y)
        Y.any ~ sibling_any(Y)
        Y.cnt ~ sibling_count(Y)
        Y.eldest ~ sibling_eldest(Y)
        Y.youngest ~ sibling_youngest(Y)
        """
        nodes = parse_formula(formula)
        assert len(nodes) == 6
        assert isinstance(nodes[0].component, SiblingMeanComponent)
        assert isinstance(nodes[1].component, SiblingSumComponent)
        assert isinstance(nodes[2].component, SiblingAnyComponent)
        assert isinstance(nodes[3].component, SiblingCountComponent)
        assert isinstance(nodes[4].component, SiblingEldestComponent)
        assert isinstance(nodes[5].component, SiblingYoungestComponent)

    def test_chained_sibling_functions(self):
        """Sibling function output can be input to another sibling function."""
        formula = """
        Y.mean ~ sibling_mean(Y)
        Y.mean.sum ~ sibling_sum(Y.mean)
        """
        nodes = parse_formula(formula)
        assert len(nodes) == 2
        assert isinstance(nodes[0].component, SiblingMeanComponent)
        assert nodes[0].component.source_name == 'Y'
        assert isinstance(nodes[1].component, SiblingSumComponent)
        assert nodes[1].component.source_name == 'Y.mean'
