"""
Unit tests for _extract_grouping function from xftsim.parser.

The function finds the last `|` not inside parentheses in a string
and splits the string at that point. The part after `|` is the grouping variable.
"""
import pytest
from xftsim.parser import _extract_grouping


class TestExtractGrouping:
    """Test suite for _extract_grouping parser function."""

    def test_no_pipe_returns_rhs_and_none(self):
        """Test that strings without pipe return (rhs, None)."""
        rhs, grouping = _extract_grouping("noise(0.3)")
        assert rhs == "noise(0.3)"
        assert grouping is None

    def test_simple_pipe_splits_correctly(self):
        """Test simple case: 'noise(0.3) | FID' returns ('noise(0.3)', 'FID')."""
        rhs, grouping = _extract_grouping("noise(0.3) | FID")
        assert rhs == "noise(0.3)"
        assert grouping == "FID"

    def test_pipe_inside_parens_ignored(self):
        """Test that pipes inside parentheses are ignored."""
        # Pipe inside function args should be ignored
        rhs, grouping = _extract_grouping("func(a|b) | FID")
        assert rhs == "func(a|b)"
        assert grouping == "FID"

    def test_pipe_inside_nested_parens(self):
        """Test pipes inside nested parentheses are ignored."""
        rhs, grouping = _extract_grouping("cnoise(cov=[[1,2],[3,4]]) | FID")
        assert rhs == "cnoise(cov=[[1,2],[3,4]])"
        assert grouping == "FID"

    def test_empty_grouping_after_pipe_returns_none(self):
        """Test that empty grouping after pipe returns (rhs, None)."""
        rhs, grouping = _extract_grouping("noise(0.3) |")
        assert rhs == "noise(0.3)"
        assert grouping is None

    def test_empty_grouping_with_spaces_returns_none(self):
        """Test that whitespace-only grouping after pipe returns (rhs, None)."""
        rhs, grouping = _extract_grouping("noise(0.3) |   ")
        assert rhs == "noise(0.3)"
        assert grouping is None

    def test_invalid_grouping_identifier_raises_valueerror(self):
        """Test that invalid grouping identifiers raise ValueError."""
        with pytest.raises(ValueError, match="Invalid grouping variable"):
            _extract_grouping("noise(0.3) | 123abc")

    def test_invalid_grouping_with_special_chars_raises_valueerror(self):
        """Test that grouping with special characters raises ValueError."""
        with pytest.raises(ValueError, match="Invalid grouping variable"):
            _extract_grouping("noise(0.3) | FID-123")

    def test_invalid_grouping_with_spaces_raises_valueerror(self):
        """Test that grouping with spaces raises ValueError."""
        with pytest.raises(ValueError, match="Invalid grouping variable"):
            _extract_grouping("noise(0.3) | FID IID")

    def test_multiple_pipes_outside_parens_uses_last(self):
        """Test that multiple pipes outside parens uses the last one."""
        rhs, grouping = _extract_grouping("A | B | C")
        assert rhs == "A | B"
        assert grouping == "C"

    def test_multiple_pipes_complex_expression(self):
        """Test multiple pipes with complex expression."""
        rhs, grouping = _extract_grouping("func(x|y) | intermediate | FINAL")
        assert rhs == "func(x|y) | intermediate"
        assert grouping == "FINAL"

    def test_grouping_with_underscore(self):
        """Test that grouping identifiers can contain underscores."""
        rhs, grouping = _extract_grouping("noise(0.3) | family_id")
        assert rhs == "noise(0.3)"
        assert grouping == "family_id"

    def test_grouping_starting_with_underscore(self):
        """Test that grouping identifiers can start with underscore."""
        rhs, grouping = _extract_grouping("noise(0.3) | _private")
        assert rhs == "noise(0.3)"
        assert grouping == "_private"

    def test_grouping_with_numbers_after_letter(self):
        """Test that grouping identifiers can contain numbers after initial letter."""
        rhs, grouping = _extract_grouping("noise(0.3) | FID123")
        assert rhs == "noise(0.3)"
        assert grouping == "FID123"

    def test_whitespace_handling(self):
        """Test that whitespace around pipe is properly stripped."""
        rhs, grouping = _extract_grouping("noise(0.3)   |   FID")
        assert rhs == "noise(0.3)"
        assert grouping == "FID"

    def test_no_whitespace_around_pipe(self):
        """Test that function works without whitespace around pipe."""
        rhs, grouping = _extract_grouping("noise(0.3)|FID")
        assert rhs == "noise(0.3)"
        assert grouping == "FID"

    def test_complex_nested_expression(self):
        """Test complex nested expression with multiple parentheses."""
        rhs, grouping = _extract_grouping("func(a, func2(b|c, d|e)) | GROUPING")
        assert rhs == "func(a, func2(b|c, d|e))"
        assert grouping == "GROUPING"

    def test_unbalanced_parens_with_pipe_after(self):
        """Test that unbalanced parens are handled (depth tracking)."""
        # This has unbalanced parens but pipe is after, should still extract
        rhs, grouping = _extract_grouping("func(a | GROUP")
        # The pipe is inside unclosed parens, so no grouping extracted
        assert rhs == "func(a | GROUP"
        assert grouping is None

    def test_pipe_at_beginning(self):
        """Test pipe at the beginning of string."""
        rhs, grouping = _extract_grouping("| FID")
        assert rhs == ""
        assert grouping == "FID"

    def test_only_pipe(self):
        """Test string with only pipe returns empty rhs and None."""
        rhs, grouping = _extract_grouping("|")
        assert rhs == ""
        assert grouping is None

    def test_realistic_noise_expression(self):
        """Test realistic noise expression from DSL."""
        rhs, grouping = _extract_grouping("noise(0.3) | FID")
        assert rhs == "noise(0.3)"
        assert grouping == "FID"

    def test_realistic_genetic_expression(self):
        """Test realistic genetic expression from DSL."""
        rhs, grouping = _extract_grouping("genetic(eff1)")
        assert rhs == "genetic(eff1)"
        assert grouping is None

    def test_realistic_cnoise_expression(self):
        """Test realistic correlated noise expression from DSL."""
        rhs, grouping = _extract_grouping("cnoise(cov=[[1,0.2],[0.2,1]]) | FID")
        assert rhs == "cnoise(cov=[[1,0.2],[0.2,1]])"
        assert grouping == "FID"
