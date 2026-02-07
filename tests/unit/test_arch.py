"""
Unit tests for Architecture, ArchNode, toposort, and execution.
"""
import numpy as np
import pytest
from xftsim.narch import (
    Architecture, ArchNode, GeneticComponent, NoiseComponent,
    AggregationComponent,
)
from xftsim.neffect import AdditiveEffects
from xftsim.struct import DenseHaplotypeArray, NPhenotypeArray, SampleMeta


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
        from xftsim.narch import ArchNode
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
        assert isinstance(result, NPhenotypeArray)
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
        """When passed an existing NPhenotypeArray, writes into it."""
        sm = haplotypes.samples
        pheno = NPhenotypeArray(samples=sm)
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
        from xftsim.narch import CNoiseComponent
        with pytest.raises(ValueError, match="square matrix"):
            CNoiseComponent(cov=np.ones((2, 3)))

    def test_haplotype_genetic_invalid_haplotype(self):
        """HaplotypeGeneticComponent with invalid haplotype arg should raise."""
        from xftsim.narch import HaplotypeGeneticComponent
        eff = AdditiveEffects.from_h2(h2=0.5, m=20, seed=42)
        with pytest.raises(ValueError, match="maternal.*paternal"):
            HaplotypeGeneticComponent(effects=eff, haplotype='both')

    def test_assortative_r_out_of_range(self):
        """LinearAssortativeMating with |r| >= 1 should raise."""
        from xftsim.nmate import LinearAssortativeMating
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
        from xftsim.narch import (
            GeneticComponent, MVGeneticComponent, HaplotypeGeneticComponent,
            NoiseComponent, CNoiseComponent, AggregationComponent,
            MotherComponent, FatherComponent, ParentComponent,
            SiblingMeanComponent, SiblingSumComponent, SiblingAnyComponent,
            SiblingCountComponent, SiblingEldestComponent, SiblingYoungestComponent,
        )
        from xftsim.neffect import MultivariateEffects
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
        from xftsim.narch import ArchNode
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
