"""
Unit tests for parser edge cases and error paths.

Tests:
1. haplotypeGenetic: invalid haplotype, missing effect, extra args
2. cnoise: non-square matrix, output count mismatch
3. parental: empty phenotype name, founder with unsupported function
4. sibling: empty source name
5. _extract_grouping edge cases
6. _parse_mvGenetic k mismatch
"""
import numpy as np
import pytest

from xftsim.arch import Architecture, HaplotypeGeneticComponent
from xftsim.effect import AdditiveEffects, MultivariateEffects
from xftsim.parser import parse_formula


class TestHaplotypeGeneticParser:
    def test_haplotypeGenetic_default_maternal(self):
        """haplotypeGenetic(eff) defaults to maternal."""
        m = 10
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture.from_formula(
            'Y ~ haplotypeGenetic(beta)',
            effects={'beta': eff},
        )
        nodes = arch._nodes
        assert len(nodes) == 1
        assert isinstance(nodes[0].component, HaplotypeGeneticComponent)
        assert nodes[0].component.haplotype == 'maternal'

    def test_haplotypeGenetic_paternal(self):
        """haplotypeGenetic(eff, haplotype='paternal')."""
        m = 10
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture.from_formula(
            "Y ~ haplotypeGenetic(beta, haplotype='paternal')",
            effects={'beta': eff},
        )
        assert arch._nodes[0].component.haplotype == 'paternal'

    def test_haplotypeGenetic_missing_effect_raises(self):
        """haplotypeGenetic with missing effect name should raise."""
        with pytest.raises(ValueError, match="not found in effects"):
            Architecture.from_formula(
                'Y ~ haplotypeGenetic(nonexistent)',
                effects={},
            )

    def test_haplotypeGenetic_extra_args_raises(self):
        """haplotypeGenetic with unknown kwarg should raise."""
        m = 10
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        with pytest.raises(ValueError, match="unexpected argument"):
            Architecture.from_formula(
                'Y ~ haplotypeGenetic(beta, bad_arg=1)',
                effects={'beta': eff},
            )

    def test_haplotypeGenetic_empty_effect_name_raises(self):
        """haplotypeGenetic with empty effect name should raise."""
        with pytest.raises(ValueError, match="requires an effect name"):
            Architecture.from_formula(
                'Y ~ haplotypeGenetic()',
                effects={},
            )


class TestCNoiseParser:
    def test_cnoise_non_square_raises(self):
        """cnoise with non-square matrix should raise."""
        with pytest.raises(ValueError, match="square matrix"):
            Architecture.from_formula(
                '(Y1, Y2) ~ cnoise(cov=[[1, 0, 0], [0, 1, 0]])',
            )

    def test_cnoise_output_count_mismatch_raises(self):
        """cnoise with wrong number of outputs should raise."""
        with pytest.raises(ValueError, match="LHS has .* outputs"):
            Architecture.from_formula(
                '(Y1, Y2, Y3) ~ cnoise(cov=[[1, 0], [0, 1]])',
            )

    def test_cnoise_invalid_literal_raises(self):
        """cnoise with unparseable matrix should raise."""
        with pytest.raises(ValueError, match="matrix literal"):
            Architecture.from_formula(
                '(Y1, Y2) ~ cnoise(cov=not_a_matrix)',
            )


class TestParentalParser:
    def test_mother_basic(self):
        """mother(Y) should parse to MotherComponent."""
        from xftsim.arch import MotherComponent
        arch = Architecture.from_formula('Y.VT ~ mother(Y)')
        assert len(arch._nodes) == 1
        assert isinstance(arch._nodes[0].component, MotherComponent)
        assert arch._nodes[0].component.phenotype_name == 'Y'

    def test_father_basic(self):
        """father(Y) should parse to FatherComponent."""
        from xftsim.arch import FatherComponent
        arch = Architecture.from_formula('Y.VT ~ father(Y)')
        assert isinstance(arch._nodes[0].component, FatherComponent)

    def test_parent_basic(self):
        """parent(Y) should parse to ParentComponent."""
        from xftsim.arch import ParentComponent
        arch = Architecture.from_formula('Y.VT ~ parent(Y)')
        assert isinstance(arch._nodes[0].component, ParentComponent)

    def test_mother_with_founder_noise(self):
        """mother(Y, founder=noise(0.3)) should set founder_component."""
        from xftsim.arch import MotherComponent, NoiseComponent
        arch = Architecture.from_formula('Y.VT ~ mother(Y, founder=noise(0.3))')
        comp = arch._nodes[0].component
        assert isinstance(comp, MotherComponent)
        assert comp.founder_component is not None
        assert isinstance(comp.founder_component, NoiseComponent)

    def test_founder_unsupported_function_raises(self):
        """founder= with unsupported function should raise."""
        with pytest.raises(ValueError, match="unsupported function"):
            Architecture.from_formula('Y.VT ~ mother(Y, founder=genetic(beta))')

    def test_founder_not_function_raises(self):
        """founder= with non-function should raise."""
        with pytest.raises(ValueError, match="function call"):
            Architecture.from_formula('Y.VT ~ mother(Y, founder=0.3)')

    def test_empty_phenotype_name_raises(self):
        """mother() with no phenotype name should raise."""
        with pytest.raises(ValueError, match="requires a phenotype name"):
            Architecture.from_formula('Y.VT ~ mother()')


class TestSiblingParser:
    def test_sibling_mean_basic(self):
        """sibling_mean(Y) should parse when Y is defined."""
        from xftsim.arch import SiblingMeanComponent
        arch = Architecture.from_formula("""
            Y ~ noise(1.0)
            Y.sib ~ sibling_mean(Y)
        """)
        sib_nodes = [n for n in arch._nodes
                     if isinstance(n.component, SiblingMeanComponent)]
        assert len(sib_nodes) == 1
        assert sib_nodes[0].component.source_name == 'Y'

    def test_sibling_empty_source_raises(self):
        """sibling_mean() with empty source should raise."""
        with pytest.raises(ValueError, match="requires a source"):
            Architecture.from_formula('Y.sib ~ sibling_mean()')

    def test_all_sibling_functions(self):
        """All 6 sibling functions should parse."""
        from xftsim.arch import (
            SiblingMeanComponent, SiblingSumComponent, SiblingAnyComponent,
            SiblingCountComponent, SiblingEldestComponent, SiblingYoungestComponent,
        )
        expected = {
            'sibling_mean': SiblingMeanComponent,
            'sibling_sum': SiblingSumComponent,
            'sibling_any': SiblingAnyComponent,
            'sibling_count': SiblingCountComponent,
            'sibling_eldest': SiblingEldestComponent,
            'sibling_youngest': SiblingYoungestComponent,
        }
        for func_name, cls in expected.items():
            arch = Architecture.from_formula(f"""
                Y ~ noise(1.0)
                Y.sib ~ {func_name}(Y)
            """)
            sib_nodes = [n for n in arch._nodes if isinstance(n.component, cls)]
            assert len(sib_nodes) == 1


class TestMVGeneticParser:
    def test_mvGenetic_k_mismatch_raises(self):
        """mvGenetic with wrong k should raise."""
        m = 10
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.5], rg=0.3, m=m, seed=42)
        with pytest.raises(ValueError, match="LHS has .* outputs"):
            Architecture.from_formula(
                '(Y1, Y2, Y3) ~ mvGenetic(beta)',
                effects={'beta': eff},
            )


class TestGroupingParser:
    def test_grouping_with_fid(self):
        """noise with | FID should set grouping."""
        arch = Architecture.from_formula('Y ~ noise(1.0) | FID')
        assert arch._nodes[0].grouping == 'FID'

    def test_grouping_with_sex(self):
        """noise with | sex should set grouping."""
        arch = Architecture.from_formula('Y ~ noise(1.0) | sex')
        assert arch._nodes[0].grouping == 'sex'

    def test_sibling_with_grouping(self):
        """sibling_mean with | FID should set grouping."""
        arch = Architecture.from_formula("""
            Y ~ noise(1.0)
            Y.sib ~ sibling_mean(Y) | FID
        """)
        from xftsim.arch import SiblingMeanComponent
        sib_nodes = [n for n in arch._nodes
                     if isinstance(n.component, SiblingMeanComponent)]
        assert len(sib_nodes) == 1
        assert sib_nodes[0].grouping == 'FID'

    def test_no_grouping_by_default(self):
        """Without |, grouping should be None."""
        arch = Architecture.from_formula('Y ~ noise(1.0)')
        assert arch._nodes[0].grouping is None


class TestMultilineFormula:
    def test_multiline_with_aggregation(self):
        """Multi-line formula with aggregation."""
        m = 10
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture.from_formula("""
            Y.G ~ genetic(beta)
            Y.E ~ noise(0.5)
            Y ~ Y.G + Y.E
        """, effects={'beta': eff})
        outputs = sorted(o for n in arch._nodes for o in n.outputs)
        assert 'Y' in outputs
        assert 'Y.G' in outputs
        assert 'Y.E' in outputs

    def test_multiline_with_vt_and_founder(self):
        """Multi-line formula with VT and founder."""
        m = 10
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture.from_formula("""
            Y.G ~ genetic(beta)
            Y.VT ~ mother(Y, founder=noise(0.1))
            Y.E ~ noise(0.5)
            Y ~ Y.G + Y.VT + Y.E
        """, effects={'beta': eff})
        assert len(arch._nodes) == 4

    def test_multiline_bivariate(self):
        """Multi-line bivariate formula."""
        m = 10
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.5], rg=0.3, m=m, seed=42)
        arch = Architecture.from_formula("""
            (Y1.G, Y2.G) ~ mvGenetic(beta)
            (Y1.E, Y2.E) ~ cnoise(cov=[[0.5, 0.1], [0.1, 0.5]])
            Y1 ~ Y1.G + Y1.E
            Y2 ~ Y2.G + Y2.E
        """, effects={'beta': eff})
        assert len(arch._nodes) == 4
