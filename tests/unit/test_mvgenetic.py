"""
Tests for MVGeneticComponent, tuple LHS parsing, and bivariate architectures.
"""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation, TestEffects

from xftsim.arch import (
    Architecture, MVGeneticComponent, NoiseComponent, AggregationComponent,
)
from xftsim.effect import MultivariateEffects
from xftsim.parser import parse_formula
from xftsim.sim import NSimulation


class TestMVGeneticComponent:
    @pytest.fixture
    def hap(self):
        return TestSimulation.founder_haplotypes(n=200, m=50)

    @pytest.fixture
    def mv_effects(self):
        return TestEffects.multivariate(m=50, h2=[0.5, 0.3], rg=0.2)

    def test_shape(self, hap, mv_effects):
        """MVGenetic should return (n, k) array."""
        arch = Architecture()
        arch.add(['t1.G', 't2.G'], MVGeneticComponent(mv_effects))
        pheno = arch.compute(hap, rng=np.random.RandomState(42))
        assert 't1.G' in pheno
        assert 't2.G' in pheno
        assert pheno['t1.G'].shape == (200,)
        assert pheno['t2.G'].shape == (200,)

    def test_programmatic_api(self, hap, mv_effects):
        """Programmatic construction works with multi-output add()."""
        arch = Architecture()
        arch.add(['a', 'b'], MVGeneticComponent(mv_effects))
        arch.add('a.E', NoiseComponent(0.5))
        arch.add('b.E', NoiseComponent(0.7))
        arch.add('A', AggregationComponent('a + a.E'))
        arch.add('B', AggregationComponent('b + b.E'))
        pheno = arch.compute(hap, rng=np.random.RandomState(42))
        assert len(list(pheno.keys)) == 6

    def test_output_write(self, hap, mv_effects):
        """Both output columns should be written to phenotype array."""
        arch = Architecture()
        arch.add(['x', 'y'], MVGeneticComponent(mv_effects))
        pheno = arch.compute(hap, rng=np.random.RandomState(42))
        # They should be different arrays (different traits)
        assert not np.allclose(pheno['x'], pheno['y'])

    def test_values_correct(self, hap, mv_effects):
        """MVGenetic values should match direct standardized_matvec."""
        comp = MVGeneticComponent(mv_effects)
        from xftsim.arch import ArchNode
        node = ArchNode(outputs=['a', 'b'], component=comp)
        result = comp.compute(node, hap, None, rng=np.random.RandomState(42))
        expected = hap.standardized_matvec(mv_effects.effects)
        np.testing.assert_allclose(result, expected)

    def test_parser_tuple_lhs(self, mv_effects):
        """Parser should handle (a, b) ~ mvGenetic(eff)."""
        formula = "(t1.G, t2.G) ~ mvGenetic(eff)"
        nodes = parse_formula(formula, effects={'eff': mv_effects})
        assert len(nodes) == 1
        assert nodes[0].outputs == ['t1.G', 't2.G']
        assert isinstance(nodes[0].component, MVGeneticComponent)

    def test_wrong_k_error(self, mv_effects):
        """Parser should error if k != len(outputs)."""
        formula = "(a, b, c) ~ mvGenetic(eff)"
        with pytest.raises(ValueError, match="k=2"):
            parse_formula(formula, effects={'eff': mv_effects})

    def test_bivariate_formula_e2e(self, hap):
        """Full bivariate architecture via formula string."""
        mv_eff = TestEffects.multivariate(m=50, h2=[0.5, 0.3])
        formula = """
        (t1.G, t2.G) ~ mvGenetic(eff)
        t1.E ~ noise(0.5)
        t2.E ~ noise(0.7)
        t1 ~ t1.G + t1.E
        t2 ~ t2.G + t2.E
        """
        arch = Architecture(formula=formula, effects={'eff': mv_eff})
        pheno = arch.compute(hap, rng=np.random.RandomState(42))
        assert 't1' in pheno
        assert 't2' in pheno
        assert pheno['t1'].shape == (200,)

    def test_kwargs_propagation(self, hap, mv_effects):
        """Architecture.compute should propagate kwargs to components."""
        arch = Architecture()
        arch.add(['a', 'b'], MVGeneticComponent(mv_effects))
        pheno = arch.compute(
            hap, rng=np.random.RandomState(42),
            phenotype_history={}, pedigree_history={}, generation=0,
        )
        assert 'a' in pheno
