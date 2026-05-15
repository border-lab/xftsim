"""
Unit tests for Architecture, ArchNode, toposort, and execution.
"""
import warnings
import numpy as np
import pytest
from xftsim.arch import (
    Architecture, ArchNode, GeneticComponent, NoiseComponent,
    AggregationComponent, MVGeneticComponent, CNoiseComponent,
    _resolve_grouping,
)
from xftsim.effect import AdditiveEffects, MultivariateEffects
from xftsim.struct import (
    DenseHaplotypeArray, PhenotypeArray, SampleMeta, PedigreeArray,
)


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def haplotypes(rng):
    geno = rng.randint(0, 2, size=(50, 20, 2)).astype(np.int8)
    return DenseHaplotypeArray(genotypes=geno)


@pytest.fixture
def effects():
    return AdditiveEffects.from_h2(h2=0.5, m=20, seed=42)


# ── Programmatic API ────────────────────────────────────────────────────────

class TestProgrammaticAPI:
    def test_add_single_node(self, effects):
        arch = Architecture()
        arch.add('height.G', GeneticComponent(effects=effects))
        assert len(arch._nodes) == 1
        assert arch._nodes[0].outputs == ['height.G']

    def test_add_multiple_nodes(self, effects):
        arch = Architecture()
        arch.add('height.G', GeneticComponent(effects=effects))
        arch.add('height.E', NoiseComponent(variance=0.5))
        arch.add('height', AggregationComponent('height.G + height.E'))
        assert len(arch._nodes) == 3

    def test_duplicate_output_error(self, effects):
        arch = Architecture()
        arch.add('x', GeneticComponent(effects=effects))
        with pytest.raises(ValueError, match="Duplicate output"):
            arch.add('x', NoiseComponent(variance=0.5))

    def test_aggregation_auto_inputs(self):
        arch = Architecture()
        arch.add('a', NoiseComponent(variance=1.0))
        arch.add('b', NoiseComponent(variance=1.0))
        arch.add('c', AggregationComponent('a + b'))
        node_c = arch._nodes[2]
        assert set(node_c.inputs) == {'a', 'b'}

    def test_list_outputs(self, effects):
        arch = Architecture()
        arch.add(['x', 'y'], NoiseComponent(variance=1.0))
        assert arch._nodes[0].outputs == ['x', 'y']


# ── Parsed API ──────────────────────────────────────────────────────────────

class TestParsedAPI:
    def test_parsed_produces_correct_nodes(self, effects):
        arch = Architecture("""
            height.G ~ genetic(eff)
            height.E ~ noise(0.5)
            height ~ height.G + height.E
        """, effects={'eff': effects})
        assert len(arch._nodes) == 3
        assert isinstance(arch._nodes[0].component, GeneticComponent)
        assert isinstance(arch._nodes[1].component, NoiseComponent)
        assert isinstance(arch._nodes[2].component, AggregationComponent)

    def test_parsed_matches_programmatic(self, effects, haplotypes):
        """Programmatic and parsed should produce same results with same rng."""
        # Programmatic
        arch1 = Architecture()
        arch1.add('height.G', GeneticComponent(effects=effects))
        arch1.add('height.E', NoiseComponent(variance=0.5))
        arch1.add('height', AggregationComponent('height.G + height.E'))
        p1 = arch1.compute(haplotypes, rng=np.random.RandomState(99))

        # Parsed
        arch2 = Architecture("""
            height.G ~ genetic(eff)
            height.E ~ noise(0.5)
            height ~ height.G + height.E
        """, effects={'eff': effects})
        p2 = arch2.compute(haplotypes, rng=np.random.RandomState(99))

        np.testing.assert_allclose(p1['height.G'], p2['height.G'])
        np.testing.assert_allclose(p1['height.E'], p2['height.E'])
        np.testing.assert_allclose(p1['height'], p2['height'])


# ── Toposort ────────────────────────────────────────────────────────────────

class TestToposort:
    def test_dependencies_before_dependents(self, effects):
        arch = Architecture()
        # Deliberately add aggregation first
        arch.add('height', AggregationComponent('height.G + height.E'))
        arch.add('height.G', GeneticComponent(effects=effects))
        arch.add('height.E', NoiseComponent(variance=0.5))
        sorted_nodes = arch.nodes
        # height.G and height.E must come before height
        output_order = [n.outputs[0] for n in sorted_nodes]
        assert output_order.index('height.G') < output_order.index('height')
        assert output_order.index('height.E') < output_order.index('height')

    def test_diamond_dependency(self):
        arch = Architecture()
        arch.add('a', NoiseComponent(variance=1.0))
        arch.add('b', AggregationComponent('a'))
        arch.add('c', AggregationComponent('a'))
        arch.add('d', AggregationComponent('b + c'))
        sorted_nodes = arch.nodes
        outputs = [n.outputs[0] for n in sorted_nodes]
        assert outputs.index('a') < outputs.index('b')
        assert outputs.index('a') < outputs.index('c')
        assert outputs.index('b') < outputs.index('d')
        assert outputs.index('c') < outputs.index('d')

    def test_undefined_reference(self):
        arch = Architecture()
        arch.add('x', AggregationComponent('undefined_var'))
        with pytest.raises(ValueError, match="Undefined reference"):
            arch.nodes

    def test_cycle_detection(self):
        """Cycles should be detected and raise ValueError."""
        arch = Architecture()
        # We can't really create a cycle through AggregationComponent
        # because inputs are extracted at creation time, but we can
        # force a cycle by manually creating nodes.
        from xftsim.arch import ArchNode
        node_a = ArchNode(outputs=['a'], component=NoiseComponent(variance=1.0),
                          inputs=['b'])
        node_b = ArchNode(outputs=['b'], component=NoiseComponent(variance=1.0),
                          inputs=['a'])
        arch._register_node(node_a)
        arch._register_node(node_b)
        with pytest.raises(ValueError, match="Cycle detected"):
            arch.nodes


# ── Execution ───────────────────────────────────────────────────────────────

class TestExecution:
    def test_compute_writes_phenotypes(self, effects, haplotypes):
        arch = Architecture()
        arch.add('height.G', GeneticComponent(effects=effects))
        arch.add('height.E', NoiseComponent(variance=0.5))
        arch.add('height', AggregationComponent('height.G + height.E'))
        result = arch.compute(haplotypes, rng=np.random.RandomState(42))
        assert isinstance(result, PhenotypeArray)
        assert 'height.G' in result
        assert 'height.E' in result
        assert 'height' in result

    def test_aggregation_addition(self, haplotypes):
        arch = Architecture()
        arch.add('a', NoiseComponent(variance=1.0))
        arch.add('b', NoiseComponent(variance=1.0))
        arch.add('c', AggregationComponent('a + b'))
        result = arch.compute(haplotypes, rng=np.random.RandomState(42))
        np.testing.assert_allclose(result['c'], result['a'] + result['b'])

    def test_aggregation_subtraction(self, haplotypes):
        arch = Architecture()
        arch.add('a', NoiseComponent(variance=1.0))
        arch.add('b', NoiseComponent(variance=1.0))
        arch.add('c', AggregationComponent('a - b'))
        result = arch.compute(haplotypes, rng=np.random.RandomState(42))
        np.testing.assert_allclose(result['c'], result['a'] - result['b'])

    def test_aggregation_multiplication(self, haplotypes):
        arch = Architecture()
        arch.add('a', NoiseComponent(variance=1.0))
        arch.add('b', NoiseComponent(variance=1.0))
        arch.add('c', AggregationComponent('a * b'))
        result = arch.compute(haplotypes, rng=np.random.RandomState(42))
        np.testing.assert_allclose(result['c'], result['a'] * result['b'])

    def test_aggregation_division(self, haplotypes):
        arch = Architecture()
        arch.add('a', NoiseComponent(variance=1.0))
        arch.add('b', NoiseComponent(variance=1.0))
        arch.add('c', AggregationComponent('a / b'))
        result = arch.compute(haplotypes, rng=np.random.RandomState(42))
        np.testing.assert_allclose(result['c'], result['a'] / result['b'])

    def test_scalar_multiplication(self, haplotypes):
        arch = Architecture()
        arch.add('a', NoiseComponent(variance=1.0))
        arch.add('b', AggregationComponent('0.5 * a'))
        result = arch.compute(haplotypes, rng=np.random.RandomState(42))
        np.testing.assert_allclose(result['b'], 0.5 * result['a'])

    def test_genetic_values_correct(self, effects, haplotypes):
        arch = Architecture()
        arch.add('g', GeneticComponent(effects=effects))
        result = arch.compute(haplotypes)
        expected = haplotypes.standardized_matvec(effects.effects)
        np.testing.assert_allclose(result['g'], expected)

    def test_compute_returns_same_phenotype_array(self, haplotypes):
        """When passed an existing PhenotypeArray, writes into it."""
        sm = haplotypes.samples
        pheno = PhenotypeArray(samples=sm)
        arch = Architecture()
        arch.add('x', NoiseComponent(variance=1.0))
        result = arch.compute(haplotypes, phenotypes=pheno, rng=np.random.RandomState(0))
        assert result is pheno
        assert 'x' in pheno

    def test_chained_aggregation(self, haplotypes):
        arch = Architecture()
        arch.add('a', NoiseComponent(variance=1.0))
        arch.add('b', NoiseComponent(variance=1.0))
        arch.add('c', AggregationComponent('a + b'))
        arch.add('d', AggregationComponent('c * 2.0'))
        result = arch.compute(haplotypes, rng=np.random.RandomState(42))
        np.testing.assert_allclose(result['d'], 2.0 * (result['a'] + result['b']))


# ── Error handling ─────────────────────────────────────────────────────────

class TestErrorHandling:
    """Tests for architecture error conditions."""

    def test_undefined_aggregation_reference_at_compute(self, haplotypes):
        """Referencing a nonexistent variable in aggregation should raise at compute."""
        arch = Architecture()
        arch.add('a', NoiseComponent(variance=1.0))
        # 'missing' is not defined by any node → toposort should catch it
        arch.add('c', AggregationComponent('a + missing'))
        with pytest.raises(ValueError, match="Undefined reference"):
            arch.nodes

    def test_cnoise_non_square_cov(self):
        """CNoiseComponent with non-square cov should raise."""
        from xftsim.arch import CNoiseComponent
        with pytest.raises(ValueError, match="square matrix"):
            CNoiseComponent(cov=np.ones((2, 3)))

    def test_haplotype_genetic_invalid_haplotype(self):
        """HaplotypeGeneticComponent with invalid haplotype arg should raise."""
        from xftsim.arch import HaplotypeGeneticComponent
        eff = AdditiveEffects.from_h2(h2=0.5, m=20, seed=42)
        with pytest.raises(ValueError, match="maternal.*paternal"):
            HaplotypeGeneticComponent(effects=eff, haplotype='both')

    def test_assortative_r_out_of_range(self):
        """LinearAssortativeMating with |r| >= 1 should raise."""
        from xftsim.mate import LinearAssortativeMating
        with pytest.raises(ValueError, match="r must be"):
            LinearAssortativeMating(component_names=['Y'], r=1.0)
        with pytest.raises(ValueError, match="r must be"):
            LinearAssortativeMating(component_names=['Y'], r=-1.0)

    def test_aggregation_empty_expression(self, haplotypes):
        """Empty expression should raise during compute or tokenization."""
        arch = Architecture()
        arch.add('a', NoiseComponent(variance=1.0))
        arch.add('b', AggregationComponent(''))
        # Empty expression → empty token list → stack has 0 values
        with pytest.raises(ValueError, match="stack has 0 values"):
            arch.compute(haplotypes, rng=np.random.RandomState(0))

    def test_aggregation_unbalanced_parens(self):
        """Unbalanced parentheses in expression should raise."""
        arch = Architecture()
        arch.add('a', NoiseComponent(variance=1.0))
        arch.add('b', AggregationComponent('(a + a'))
        with pytest.raises(ValueError, match="parentheses"):
            arch.compute(
                DenseHaplotypeArray(genotypes=np.zeros((2, 1, 2), dtype=np.int8)),
                rng=np.random.RandomState(0),
            )

    def test_repr_all_components(self, effects, haplotypes):
        """repr() should not crash for any component type."""
        from xftsim.arch import (
            GeneticComponent, MVGeneticComponent, HaplotypeGeneticComponent,
            NoiseComponent, CNoiseComponent, AggregationComponent,
            MotherComponent, FatherComponent, ParentComponent,
            SiblingMeanComponent, SiblingSumComponent, SiblingAnyComponent,
            SiblingCountComponent, SiblingEldestComponent, SiblingYoungestComponent,
        )
        from xftsim.effect import MultivariateEffects
        mv_eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.2, m=effects.m, seed=42)
        components = [
            GeneticComponent(effects),
            MVGeneticComponent(mv_eff),
            HaplotypeGeneticComponent(effects, haplotype='maternal'),
            HaplotypeGeneticComponent(effects, haplotype='paternal'),
            NoiseComponent(variance=1.0),
            CNoiseComponent(cov=np.eye(2)),
            AggregationComponent('a + b'),
            MotherComponent('Y'),
            FatherComponent('Y'),
            ParentComponent('Y'),
            SiblingMeanComponent('Y'),
            SiblingSumComponent('Y'),
            SiblingAnyComponent('Y'),
            SiblingCountComponent('Y'),
            SiblingEldestComponent('Y'),
            SiblingYoungestComponent('Y'),
        ]
        for comp in components:
            r = repr(comp)
            assert isinstance(r, str) and len(r) > 0

    def test_architecture_repr(self, effects):
        """Architecture repr should work."""
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(effects))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        r = repr(arch)
        assert "Architecture" in r
        assert "nodes=3" in r

    def test_archnode_repr(self, effects):
        """ArchNode repr should work."""
        from xftsim.arch import ArchNode
        node = ArchNode(
            outputs=['Y.G'],
            component=GeneticComponent(effects),
            inputs=[],
        )
        r = repr(node)
        assert "ArchNode" in r
        assert "Y.G" in r


# ── Architecture.from_formula unit tests ───────────────────────────────────

class TestFromFormula:
    """Unit tests for Architecture.from_formula()."""

    def test_simple_formula(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=20, seed=42)
        arch = Architecture.from_formula("""
            Y.G ~ genetic(eff)
            Y.E ~ noise(0.5)
            Y ~ Y.G + Y.E
        """, effects={'eff': eff})
        assert len(arch.nodes) == 3
        outputs = [n.outputs[0] for n in arch.nodes]
        assert 'Y.G' in outputs
        assert 'Y.E' in outputs
        assert 'Y' in outputs

    def test_formula_toposort_order(self):
        """Nodes should be in topological order (dependencies before dependents)."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=20, seed=42)
        arch = Architecture.from_formula("""
            Y ~ Y.G + Y.E
            Y.E ~ noise(0.5)
            Y.G ~ genetic(eff)
        """, effects={'eff': eff})
        outputs = [n.outputs[0] for n in arch.nodes]
        assert outputs.index('Y.G') < outputs.index('Y')
        assert outputs.index('Y.E') < outputs.index('Y')

    def test_formula_computes_correctly(self, haplotypes, rng):
        """from_formula architecture should produce identical results to programmatic."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=20, seed=42)

        # Programmatic
        arch1 = Architecture()
        arch1.add('Y.G', GeneticComponent(eff))
        arch1.add('Y.E', NoiseComponent(variance=0.5))
        arch1.add('Y', AggregationComponent('Y.G + Y.E'))

        # Formula
        arch2 = Architecture.from_formula("""
            Y.G ~ genetic(eff)
            Y.E ~ noise(0.5)
            Y ~ Y.G + Y.E
        """, effects={'eff': eff})

        rng1 = np.random.RandomState(99)
        rng2 = np.random.RandomState(99)
        p1 = arch1.compute(haplotypes, rng=rng1)
        p2 = arch2.compute(haplotypes, rng=rng2)
        np.testing.assert_array_equal(p1['Y.G'], p2['Y.G'])
        np.testing.assert_array_equal(p1['Y.E'], p2['Y.E'])
        np.testing.assert_array_equal(p1['Y'], p2['Y'])

    def test_formula_with_vt(self):
        """Formula with vertical transmission should parse and have correct structure."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=20, seed=42)
        arch = Architecture.from_formula("""
            Y.G ~ genetic(eff)
            Y.VT ~ parent(Y, founder=noise(0.3))
            Y.E ~ noise(0.2)
            Y ~ Y.G + 0.3 * Y.VT + Y.E
        """, effects={'eff': eff})
        assert len(arch.nodes) == 4
        outputs = [n.outputs[0] for n in arch.nodes]
        assert 'Y.VT' in outputs

    def test_formula_with_grouping(self):
        """Formula with | grouping should preserve grouping on nodes."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=20, seed=42)
        arch = Architecture.from_formula("""
            Y.G ~ genetic(eff)
            Y.E ~ noise(0.5) | FID
            Y ~ Y.G + Y.E
        """, effects={'eff': eff})
        noise_node = [n for n in arch.nodes if n.outputs == ['Y.E']][0]
        assert noise_node.grouping == 'FID'

    def test_formula_constructor_alias(self):
        """Architecture(formula=...) should behave like from_formula."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=20, seed=42)
        arch1 = Architecture(formula="Y ~ noise(0.5)", effects={})
        arch2 = Architecture.from_formula("Y ~ noise(0.5)", effects={})
        assert len(arch1.nodes) == len(arch2.nodes)
        assert arch1.nodes[0].outputs == arch2.nodes[0].outputs

    def test_formula_no_effects(self):
        """Formula with only noise and aggregation needs no effects dict."""
        arch = Architecture.from_formula("""
            a ~ noise(1.0)
            b ~ noise(0.5)
            c ~ a + b
        """)
        assert len(arch.nodes) == 3


class TestArchitectureComputeEdgeCases:
    """Edge cases for Architecture.compute()."""

    def test_single_noise_component(self, haplotypes, rng):
        """Architecture with a single noise component should work."""
        arch = Architecture()
        arch.add('E', NoiseComponent(variance=1.0))
        pheno = arch.compute(haplotypes, rng=rng)
        assert 'E' in pheno.keys
        assert pheno['E'].shape == (haplotypes.n,)
        assert np.all(np.isfinite(pheno['E']))

    def test_single_genetic_component(self, haplotypes, effects, rng):
        """Architecture with a single genetic component should work."""
        arch = Architecture()
        arch.add('G', GeneticComponent(effects))
        pheno = arch.compute(haplotypes, rng=rng)
        assert 'G' in pheno.keys
        assert pheno['G'].shape == (haplotypes.n,)

    def test_many_independent_noises(self, haplotypes, rng):
        """Architecture with many independent noise components."""
        arch = Architecture()
        for i in range(10):
            arch.add(f'E{i}', NoiseComponent(variance=0.1 * (i + 1)))
        pheno = arch.compute(haplotypes, rng=rng)
        for i in range(10):
            assert f'E{i}' in pheno.keys
            assert np.all(np.isfinite(pheno[f'E{i}']))

    def test_deep_aggregation_chain(self, haplotypes, rng):
        """Deeply nested aggregation should work."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('B', NoiseComponent(variance=1.0))
        arch.add('AB', AggregationComponent('A + B'))
        arch.add('C', NoiseComponent(variance=1.0))
        arch.add('ABC', AggregationComponent('AB + C'))
        arch.add('D', NoiseComponent(variance=1.0))
        arch.add('ABCD', AggregationComponent('ABC + D'))
        pheno = arch.compute(haplotypes, rng=rng)
        # ABCD should equal A + B + C + D
        expected = pheno['A'] + pheno['B'] + pheno['C'] + pheno['D']
        np.testing.assert_allclose(pheno['ABCD'], expected, atol=1e-10)

    def test_aggregation_with_coefficients(self, haplotypes, effects, rng):
        """Aggregation with various coefficient patterns."""
        arch = Architecture()
        arch.add('G', GeneticComponent(effects))
        arch.add('E', NoiseComponent(variance=1.0))
        arch.add('Y', AggregationComponent('0.5 * G + 2.0 * E'))
        pheno = arch.compute(haplotypes, rng=rng)
        expected = 0.5 * pheno['G'] + 2.0 * pheno['E']
        np.testing.assert_allclose(pheno['Y'], expected, atol=1e-10)

    def test_compute_deterministic_with_seed(self, haplotypes, effects):
        """Same seed should produce identical results."""
        arch = Architecture()
        arch.add('G', GeneticComponent(effects))
        arch.add('E', NoiseComponent(variance=1.0))
        arch.add('Y', AggregationComponent('G + E'))

        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(42)
        p1 = arch.compute(haplotypes, rng=rng1)
        p2 = arch.compute(haplotypes, rng=rng2)
        np.testing.assert_array_equal(p1['Y'], p2['Y'])

    def test_compute_different_seeds_differ(self, haplotypes, effects):
        """Different seeds should produce different results (noise differs)."""
        arch = Architecture()
        arch.add('E', NoiseComponent(variance=1.0))

        rng1 = np.random.RandomState(42)
        rng2 = np.random.RandomState(99)
        p1 = arch.compute(haplotypes, rng=rng1)
        p2 = arch.compute(haplotypes, rng=rng2)
        assert not np.array_equal(p1['E'], p2['E'])

    def test_aggregation_subtraction(self, haplotypes, rng):
        """Subtraction in aggregation expression."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('B', NoiseComponent(variance=1.0))
        arch.add('diff', AggregationComponent('A - B'))
        pheno = arch.compute(haplotypes, rng=rng)
        expected = pheno['A'] - pheno['B']
        np.testing.assert_allclose(pheno['diff'], expected, atol=1e-10)


class TestExpressionEdgeCases:
    """Test _evaluate_expression / AggregationComponent edge cases."""

    @pytest.fixture
    def haplotypes(self):
        rng = np.random.RandomState(42)
        geno = rng.randint(0, 2, size=(20, 10, 2)).astype(np.int8)
        return DenseHaplotypeArray(genotypes=geno)

    def test_unary_minus_number(self, haplotypes):
        """Unary minus on a number: -1 * A."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('Y', AggregationComponent('-1 * A'))
        rng = np.random.RandomState(42)
        pheno = arch.compute(haplotypes, rng=rng)
        np.testing.assert_allclose(pheno['Y'], -pheno['A'], atol=1e-10)

    def test_unary_minus_name(self, haplotypes):
        """Unary minus on a name: Y = -A (parsed as -1 * A)."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('Y', AggregationComponent('-A'))
        rng = np.random.RandomState(42)
        pheno = arch.compute(haplotypes, rng=rng)
        np.testing.assert_allclose(pheno['Y'], -pheno['A'], atol=1e-10)

    def test_nested_parentheses(self, haplotypes):
        """Nested parentheses: (A + (B * 2))."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('B', NoiseComponent(variance=1.0))
        arch.add('Y', AggregationComponent('(A + (B * 2))'))
        rng = np.random.RandomState(42)
        pheno = arch.compute(haplotypes, rng=rng)
        expected = pheno['A'] + pheno['B'] * 2
        np.testing.assert_allclose(pheno['Y'], expected, atol=1e-10)

    def test_division_expression(self, haplotypes):
        """Division in expression: A / 2."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('Y', AggregationComponent('A / 2'))
        rng = np.random.RandomState(42)
        pheno = arch.compute(haplotypes, rng=rng)
        np.testing.assert_allclose(pheno['Y'], pheno['A'] / 2, atol=1e-10)

    def test_undefined_reference_raises(self, haplotypes):
        """Referencing an undefined name should raise ValueError."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('Y', AggregationComponent('A + UNDEFINED'))
        rng = np.random.RandomState(42)
        with pytest.raises(ValueError, match="Undefined reference"):
            arch.compute(haplotypes, rng=rng)

    def test_mismatched_parentheses_raises(self, haplotypes):
        """Mismatched parentheses in expression should raise at compute time."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('B', NoiseComponent(variance=1.0))
        arch.add('Y', AggregationComponent('(A + B'))
        rng = np.random.RandomState(42)
        with pytest.raises(ValueError, match="[Pp]arenthes"):
            arch.compute(haplotypes, rng=rng)

    def test_complex_expression_precedence(self, haplotypes):
        """Operator precedence: A + B * 2 should be A + (B*2) not (A+B)*2."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('B', NoiseComponent(variance=1.0))
        arch.add('Y', AggregationComponent('A + B * 2'))
        rng = np.random.RandomState(42)
        pheno = arch.compute(haplotypes, rng=rng)
        expected = pheno['A'] + pheno['B'] * 2
        np.testing.assert_allclose(pheno['Y'], expected, atol=1e-10)


class TestResolveGrouping:
    """Tests for _resolve_grouping() function."""

    def _make_hap(self, n=20, m=10, fid=None, extra=None):
        rng = np.random.RandomState(42)
        geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
        sex = np.tile([0, 1], (n + 1) // 2)[:n]
        if fid is None:
            fid = np.repeat(np.arange(n // 2), 2)
        samples = SampleMeta(iid=np.arange(n), fid=fid, sex=sex,
                             extra=extra or {})
        return DenseHaplotypeArray(genotypes=geno, samples=samples)

    def test_none_returns_none(self):
        """grouping=None should return None."""
        hap = self._make_hap()
        result = _resolve_grouping(None, hap)
        assert result is None

    def test_fid_grouping(self):
        """grouping='FID' should return FID labels."""
        fid = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4,
                        5, 5, 6, 6, 7, 7, 8, 8, 9, 9])
        hap = self._make_hap(fid=fid)
        result = _resolve_grouping('FID', hap)
        np.testing.assert_array_equal(result, fid)

    def test_sex_grouping(self):
        """grouping='sex' should return sex labels."""
        hap = self._make_hap()
        result = _resolve_grouping('sex', hap)
        expected = np.tile([0, 1], 10)
        np.testing.assert_array_equal(result, expected)

    def test_mother_grouping_gen0_warns(self):
        """grouping='mother' at gen 0 should warn and return None."""
        hap = self._make_hap()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _resolve_grouping('mother', hap, generation=0, pedigree_history={})
        assert result is None
        assert len(w) >= 1
        assert "mother" in str(w[0].message).lower()

    def test_father_grouping_gen0_warns(self):
        """grouping='father' at gen 0 should warn and return None."""
        hap = self._make_hap()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _resolve_grouping('father', hap, generation=0, pedigree_history={})
        assert result is None
        assert len(w) >= 1

    def test_mother_grouping_with_pedigree(self):
        """grouping='mother' with pedigree should return maternal_idx."""
        hap = self._make_hap()
        n = hap.n
        maternal_idx = np.repeat(np.arange(n // 2), 2)
        paternal_idx = np.tile(np.arange(n // 2, n), 2)[:n]
        ped = PedigreeArray(
            offspring_samples=hap.samples,
            maternal_idx=maternal_idx,
            paternal_idx=paternal_idx,
            parent_n=n,
        )
        result = _resolve_grouping('mother', hap, generation=1,
                                    pedigree_history={1: ped})
        np.testing.assert_array_equal(result, maternal_idx)

    def test_extra_field_grouping(self):
        """Custom extra field on SampleMeta should work as grouping."""
        cluster = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
                            2, 2, 2, 2, 2, 3, 3, 3, 3, 3])
        hap = self._make_hap(extra={'cluster': cluster})
        result = _resolve_grouping('cluster', hap)
        np.testing.assert_array_equal(result, cluster)

    def test_unknown_grouping_raises(self):
        """Unknown grouping variable should raise ValueError."""
        hap = self._make_hap()
        with pytest.raises(ValueError, match="Unknown grouping"):
            _resolve_grouping('nonexistent', hap)


class TestMultiOutputComponents:
    """Tests for MVGeneticComponent and CNoiseComponent multi-output handling."""

    @pytest.fixture
    def haplotypes(self):
        rng = np.random.RandomState(42)
        geno = rng.randint(0, 2, size=(50, 20, 2)).astype(np.int8)
        return DenseHaplotypeArray(genotypes=geno)

    def test_mvgenetic_produces_correct_shape(self, haplotypes):
        """MVGeneticComponent with k=2 should produce 2 output columns."""
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.2, m=20, seed=42)
        arch = Architecture()
        arch.add(['Y1.G', 'Y2.G'], MVGeneticComponent(eff))
        rng = np.random.RandomState(42)
        pheno = arch.compute(haplotypes, rng=rng)
        assert 'Y1.G' in pheno.keys
        assert 'Y2.G' in pheno.keys
        assert len(pheno['Y1.G']) == 50
        assert len(pheno['Y2.G']) == 50

    def test_mvgenetic_traits_differ(self, haplotypes):
        """Different traits from MVGeneticComponent should not be identical."""
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.2, m=20, seed=42)
        arch = Architecture()
        arch.add(['Y1.G', 'Y2.G'], MVGeneticComponent(eff))
        rng = np.random.RandomState(42)
        pheno = arch.compute(haplotypes, rng=rng)
        assert not np.array_equal(pheno['Y1.G'], pheno['Y2.G'])

    def test_cnoise_produces_correct_shape(self, haplotypes):
        """CNoiseComponent with k=2 should produce 2 output columns."""
        cov = np.array([[1.0, 0.3], [0.3, 1.0]])
        arch = Architecture()
        arch.add(['Y1.E', 'Y2.E'], CNoiseComponent(cov=cov))
        rng = np.random.RandomState(42)
        pheno = arch.compute(haplotypes, rng=rng)
        assert 'Y1.E' in pheno.keys
        assert 'Y2.E' in pheno.keys
        assert len(pheno['Y1.E']) == 50

    def test_cnoise_traits_correlated(self, haplotypes):
        """CNoiseComponent with positive off-diagonal should produce correlated traits."""
        cov = np.array([[1.0, 0.8], [0.8, 1.0]])
        arch = Architecture()
        arch.add(['Y1.E', 'Y2.E'], CNoiseComponent(cov=cov))
        rng = np.random.RandomState(42)
        # Use more samples for reliable correlation
        geno = rng.randint(0, 2, size=(500, 20, 2)).astype(np.int8)
        big_hap = DenseHaplotypeArray(genotypes=geno)
        pheno = arch.compute(big_hap, rng=np.random.RandomState(42))
        corr = np.corrcoef(pheno['Y1.E'], pheno['Y2.E'])[0, 1]
        assert corr > 0.5, f"Expected high correlation, got {corr}"

    def test_cnoise_non_square_cov_raises(self):
        """Non-square covariance matrix should raise."""
        with pytest.raises(ValueError, match="square"):
            CNoiseComponent(cov=np.array([[1.0, 0.3]]))

    def test_bivariate_full_architecture(self, haplotypes):
        """Full bivariate architecture: MVGenetic + CNoise + 2 Aggregations."""
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.2, m=20, seed=42)
        cov = np.array([[0.5, 0.1], [0.1, 0.7]])
        arch = Architecture()
        arch.add(['Y1.G', 'Y2.G'], MVGeneticComponent(eff))
        arch.add(['Y1.E', 'Y2.E'], CNoiseComponent(cov=cov))
        arch.add('Y1', AggregationComponent('Y1.G + Y1.E'))
        arch.add('Y2', AggregationComponent('Y2.G + Y2.E'))
        rng = np.random.RandomState(42)
        pheno = arch.compute(haplotypes, rng=rng)
        # Y1 should equal Y1.G + Y1.E
        np.testing.assert_allclose(
            pheno['Y1'], pheno['Y1.G'] + pheno['Y1.E'], atol=1e-10
        )
        np.testing.assert_allclose(
            pheno['Y2'], pheno['Y2.G'] + pheno['Y2.E'], atol=1e-10
        )


class TestGroupedNoise:
    """Test grouped noise components (noise with grouping variable)."""

    def test_fid_grouped_noise_siblings_share_value(self):
        """Noise grouped by FID: siblings should share the same noise value."""
        n, m = 20, 10
        rng = np.random.RandomState(42)
        geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
        fid = np.repeat(np.arange(n // 2), 2)  # pairs share FID
        sex = np.tile([0, 1], n // 2)
        samples = SampleMeta(iid=np.arange(n), fid=fid, sex=sex)
        hap = DenseHaplotypeArray(genotypes=geno, samples=samples)

        arch = Architecture()
        arch.add('Y.E', NoiseComponent(variance=1.0), grouping='FID')
        pheno = arch.compute(hap, rng=np.random.RandomState(42))

        # Each pair (i, i+1) should have the same noise value
        for i in range(0, n, 2):
            assert pheno['Y.E'][i] == pheno['Y.E'][i + 1], (
                f"Siblings at {i},{i+1} have different noise: "
                f"{pheno['Y.E'][i]} vs {pheno['Y.E'][i+1]}"
            )

    def test_sex_grouped_noise_same_sex_share(self):
        """Noise grouped by sex: same-sex individuals share value."""
        n, m = 20, 10
        rng = np.random.RandomState(42)
        geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
        sex = np.tile([0, 1], n // 2)
        samples = SampleMeta(iid=np.arange(n), sex=sex)
        hap = DenseHaplotypeArray(genotypes=geno, samples=samples)

        arch = Architecture()
        arch.add('Y.E', NoiseComponent(variance=1.0), grouping='sex')
        pheno = arch.compute(hap, rng=np.random.RandomState(42))

        females = pheno['Y.E'][sex == 0]
        males = pheno['Y.E'][sex == 1]
        # All females should have the same value, all males the same value
        assert np.all(females == females[0])
        assert np.all(males == males[0])
        # But males and females should differ
        assert females[0] != males[0]

    def test_grouped_cnoise_siblings_share(self):
        """CNoiseComponent grouped by FID: siblings share correlated noise."""
        n, m = 20, 10
        rng = np.random.RandomState(42)
        geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
        fid = np.repeat(np.arange(n // 2), 2)
        sex = np.tile([0, 1], n // 2)
        samples = SampleMeta(iid=np.arange(n), fid=fid, sex=sex)
        hap = DenseHaplotypeArray(genotypes=geno, samples=samples)

        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        arch = Architecture()
        arch.add(['E1', 'E2'], CNoiseComponent(cov=cov), grouping='FID')
        pheno = arch.compute(hap, rng=np.random.RandomState(42))

        # Each pair should share both E1 and E2
        for i in range(0, n, 2):
            assert pheno['E1'][i] == pheno['E1'][i + 1]
            assert pheno['E2'][i] == pheno['E2'][i + 1]
