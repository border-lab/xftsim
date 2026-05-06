"""
Unit tests for parser cnoise error handling and haplotypeGenetic parsing.

Tests:
1. cnoise with non-square matrix → ValueError
2. cnoise with dimension mismatch (LHS count ≠ cov dimension) → ValueError
3. cnoise with invalid matrix literal → ValueError
4. haplotypeGenetic parses maternal default
5. haplotypeGenetic parses paternal kwarg
6. haplotypeGenetic with invalid argument raises
7. cnoise with cov= prefix works
8. cnoise without cov= prefix works
"""
import numpy as np
import pytest

from xftsim.parser import parse_formula
from xftsim.arch import HaplotypeGeneticComponent, CNoiseComponent
from xftsim.effect import AdditiveEffects


class TestParserCnoiseErrors:
    def test_non_square_matrix(self):
        """cnoise with rectangular matrix → ValueError."""
        formula = "(E1, E2) ~ cnoise(cov=[[1, 0.5, 0.3], [0.5, 1, 0.2]])"
        with pytest.raises(ValueError, match="square matrix"):
            parse_formula(formula)

    def test_dimension_mismatch(self):
        """cnoise with 3x3 cov but 2 outputs → ValueError."""
        formula = "(E1, E2) ~ cnoise(cov=[[1, 0.5, 0.1], [0.5, 1, 0.2], [0.1, 0.2, 1]])"
        with pytest.raises(ValueError, match="LHS has 2"):
            parse_formula(formula)

    def test_invalid_matrix_literal(self):
        """cnoise with malformed matrix → ValueError."""
        formula = "(E1, E2) ~ cnoise(cov=not_a_matrix)"
        with pytest.raises(ValueError, match="matrix literal"):
            parse_formula(formula)

    def test_cnoise_with_cov_prefix(self):
        """cnoise(cov=[[1, 0.2], [0.2, 1]]) works."""
        formula = "(E1, E2) ~ cnoise(cov=[[1, 0.2], [0.2, 1]])"
        nodes = parse_formula(formula)
        assert len(nodes) == 1
        assert isinstance(nodes[0].component, CNoiseComponent)

    def test_cnoise_without_cov_prefix(self):
        """cnoise([[1, 0.2], [0.2, 1]]) works."""
        formula = "(E1, E2) ~ cnoise([[1, 0.2], [0.2, 1]])"
        nodes = parse_formula(formula)
        assert len(nodes) == 1
        assert isinstance(nodes[0].component, CNoiseComponent)


class TestParserHaplotypeGenetic:
    def _make_effects(self):
        return {'eff': AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)}

    def test_maternal_default(self):
        """haplotypeGenetic(eff) defaults to maternal."""
        formula = "Y.mat ~ haplotypeGenetic(eff)"
        effects = self._make_effects()
        nodes = parse_formula(formula, effects=effects)
        assert len(nodes) == 1
        comp = nodes[0].component
        assert isinstance(comp, HaplotypeGeneticComponent)
        assert comp.haplotype == 'maternal'

    def test_paternal_kwarg(self):
        """haplotypeGenetic(eff, haplotype='paternal')."""
        formula = "Y.pat ~ haplotypeGenetic(eff, haplotype='paternal')"
        effects = self._make_effects()
        nodes = parse_formula(formula, effects=effects)
        comp = nodes[0].component
        assert comp.haplotype == 'paternal'

    def test_invalid_argument_raises(self):
        """haplotypeGenetic(eff, foo=bar) → ValueError."""
        formula = "Y.mat ~ haplotypeGenetic(eff, foo=bar)"
        effects = self._make_effects()
        with pytest.raises(ValueError, match="unexpected argument"):
            parse_formula(formula, effects=effects)

    def test_parental_no_grouping(self):
        """Parental components reject | grouping."""
        formula = "Y.m ~ mother(Y) | FID"
        with pytest.raises(ValueError, match="does not accept"):
            parse_formula(formula)
