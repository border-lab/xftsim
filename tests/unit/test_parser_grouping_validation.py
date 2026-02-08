"""
Unit tests for parser grouping validation edge cases.

Tests:
1. _extract_grouping with invalid variable name (starts with digit)
2. _extract_grouping with special chars
3. _extract_grouping with empty trailing pipe
4. _extract_grouping respects parentheses depth
5. Grouping on aggregation expression raises
6. Grouping on non-accepting component raises
"""
import pytest

from xftsim.narch import Architecture
from xftsim.neffect import AdditiveEffects
from xftsim.parser import _extract_grouping, parse_formula


class TestExtractGroupingValidation:
    def test_invalid_grouping_starts_with_digit(self):
        """Grouping variable starting with digit should raise."""
        with pytest.raises(ValueError, match="Invalid grouping"):
            _extract_grouping("noise(1.0) | 123abc")

    def test_invalid_grouping_special_chars(self):
        """Grouping variable with special chars should raise."""
        with pytest.raises(ValueError, match="Invalid grouping"):
            _extract_grouping("noise(1.0) | a+b")

    def test_invalid_grouping_space_in_name(self):
        """Grouping variable with spaces should raise."""
        with pytest.raises(ValueError, match="Invalid grouping"):
            _extract_grouping("noise(1.0) | hello world")

    def test_empty_trailing_pipe(self):
        """Trailing pipe with nothing after → no grouping."""
        rhs, grouping = _extract_grouping("noise(1.0) |")
        assert grouping is None

    def test_valid_grouping_underscore(self):
        """Grouping variable with underscores should work."""
        rhs, grouping = _extract_grouping("noise(1.0) | my_field")
        assert grouping == "my_field"

    def test_pipe_inside_parens_not_extracted(self):
        """Pipe inside parentheses should NOT be treated as grouping."""
        rhs, grouping = _extract_grouping("cnoise(cov=[[1|0],[0|1]])")
        assert grouping is None

    def test_valid_grouping_returns_stripped_rhs(self):
        """Grouping extraction should strip whitespace from RHS."""
        rhs, grouping = _extract_grouping("  noise(1.0)  | FID  ")
        assert rhs == "noise(1.0)"
        assert grouping == "FID"


class TestGroupingOnNonAcceptingComponent:
    def test_grouping_on_genetic_raises(self):
        """genetic() does not accept grouping."""
        m = 10
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        with pytest.raises(ValueError, match="does not accept"):
            Architecture.from_formula(
                'Y ~ genetic(beta) | FID',
                effects={'beta': eff},
            )

    def test_grouping_on_parent_raises(self):
        """parent() does not accept grouping — it uses pedigree."""
        with pytest.raises(ValueError, match="does not accept"):
            Architecture.from_formula('Y.VT ~ parent(Y) | FID')

    def test_grouping_on_mother_raises(self):
        """mother() does not accept grouping."""
        with pytest.raises(ValueError, match="does not accept"):
            Architecture.from_formula('Y.VT ~ mother(Y) | FID')


class TestGroupingOnAggregation:
    def test_pipe_on_aggregation_raises_clear_error(self):
        """Y ~ Y.G + Y.E | FID should raise (grouping on aggregation)."""
        m = 10
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        with pytest.raises(ValueError, match="grouping.*only valid on function"):
            Architecture.from_formula("""
                Y.G ~ genetic(beta)
                Y.E ~ noise(0.5)
                Y ~ Y.G + Y.E | FID
            """, effects={'beta': eff})


class TestUnknownFunctionInFormula:
    def test_unknown_function_raises(self):
        """Unknown function should raise with available list."""
        with pytest.raises(ValueError, match="unknown function.*'bogus'"):
            Architecture.from_formula('Y ~ bogus(1.0)')

    def test_unknown_function_message_lists_available(self):
        """Error message should list available functions."""
        with pytest.raises(ValueError, match="Available"):
            Architecture.from_formula('Y ~ bogus(1.0)')
