"""
Unit tests for Architecture advanced paths.

Tests:
1. _toposort: undefined reference detection
2. Architecture.compute: empty arch, multi-output, None phenotypes/rng
3. Architecture.add: auto-detect inputs for AggregationComponent
4. ArchNode repr
5. Architecture repr
6. from_formula vs __init__ equivalence
7. _register_node invalidates sort cache
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray, NPhenotypeArray
from xftsim.narch import (
    Architecture, ArchNode, ArchComponent, GeneticComponent,
    NoiseComponent, AggregationComponent, CNoiseComponent,
    MVGeneticComponent,
)
from xftsim.neffect import AdditiveEffects, MultivariateEffects


def _make_hap(n=20, m=5, seed=42):
    rng = np.random.RandomState(seed)
    genotypes = rng.binomial(1, 0.3, size=(n, m, 2)).astype(np.int8)
    sm = SampleMeta(iid=np.arange(n))
    vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
    return DenseHaplotypeArray(genotypes=genotypes, generation=0, samples=sm, variants=vm)


class TestToposortUndefinedRef:
    def test_undefined_input_raises(self):
        """Referencing a name that no node produces should raise."""
        arch = Architecture()
        arch.add('Y', AggregationComponent('X + Z'))
        with pytest.raises(ValueError, match="Undefined reference"):
            arch.nodes  # triggers toposort

    def test_partial_undefined_raises(self):
        """One defined, one undefined input should raise."""
        arch = Architecture()
        arch.add('X', NoiseComponent(variance=1.0))
        arch.add('Y', AggregationComponent('X + Z'))
        with pytest.raises(ValueError, match="Undefined reference 'Z'"):
            arch.nodes


class TestArchitectureCompute:
    def test_empty_architecture(self):
        """Empty architecture produces phenotypes with no values."""
        arch = Architecture()
        hap = _make_hap()
        pheno = arch.compute(hap)
        assert isinstance(pheno, NPhenotypeArray)
        assert len(pheno.keys) == 0

    def test_compute_creates_phenotypes_if_none(self):
        """compute() with phenotypes=None should auto-create."""
        arch = Architecture()
        arch.add('E', NoiseComponent(variance=1.0))
        hap = _make_hap()
        pheno = arch.compute(hap, phenotypes=None, rng=np.random.RandomState(42))
        assert 'E' in pheno.keys
        assert pheno['E'].shape == (hap.n,)

    def test_compute_creates_rng_if_none(self):
        """compute() with rng=None should auto-create."""
        arch = Architecture()
        arch.add('E', NoiseComponent(variance=1.0))
        hap = _make_hap()
        pheno = arch.compute(hap, rng=None)
        assert 'E' in pheno.keys
        assert np.all(np.isfinite(pheno['E']))

    def test_multi_output_node(self):
        """Multi-output node (MVGeneticComponent) writes to each output."""
        m = 5
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.5], rg=0.3, m=m, seed=42)
        arch = Architecture()
        arch.add(('Y1.G', 'Y2.G'), MVGeneticComponent(eff))
        hap = _make_hap(n=20, m=m)
        pheno = arch.compute(hap, rng=np.random.RandomState(42))
        assert 'Y1.G' in pheno.keys
        assert 'Y2.G' in pheno.keys
        assert pheno['Y1.G'].shape == (20,)
        assert pheno['Y2.G'].shape == (20,)

    def test_compute_writes_to_existing_phenotypes(self):
        """compute() with existing phenotypes should add new keys."""
        arch = Architecture()
        arch.add('E', NoiseComponent(variance=1.0))
        hap = _make_hap()
        sm = hap.samples
        pheno = NPhenotypeArray(samples=sm, values={'prior': np.ones(hap.n)})
        result = arch.compute(hap, phenotypes=pheno, rng=np.random.RandomState(42))
        assert 'prior' in result.keys
        assert 'E' in result.keys
        np.testing.assert_array_equal(result['prior'], np.ones(hap.n))


class TestArchitectureAdd:
    def test_add_auto_detects_aggregation_inputs(self):
        """add() should extract input names from AggregationComponent."""
        arch = Architecture()
        arch.add('X', NoiseComponent(variance=1.0))
        arch.add('Z', NoiseComponent(variance=1.0))
        arch.add('Y', AggregationComponent('X + Z'))
        # Should be able to toposort without error
        nodes = arch.nodes
        assert len(nodes) == 3

    def test_add_invalidates_cache(self):
        """Adding a node should invalidate the sorted cache."""
        arch = Architecture()
        arch.add('X', NoiseComponent(variance=1.0))
        _ = arch.nodes  # populates cache
        arch.add('Y', NoiseComponent(variance=1.0))
        # Should re-sort and include both
        assert len(arch.nodes) == 2

    def test_add_string_output_normalized(self):
        """add('Y', ...) should be equivalent to add(['Y'], ...)."""
        arch = Architecture()
        arch.add('Y', NoiseComponent(variance=1.0))
        assert arch._nodes[0].outputs == ['Y']

    def test_add_tuple_output(self):
        """add(('A', 'B'), ...) should register two outputs."""
        m = 5
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.5], rg=0.3, m=m, seed=42)
        arch = Architecture()
        arch.add(('A', 'B'), MVGeneticComponent(eff))
        assert 'A' in arch._output_map
        assert 'B' in arch._output_map


class TestArchNodeRepr:
    def test_repr_contains_outputs(self):
        """ArchNode repr should show outputs."""
        node = ArchNode(outputs=['Y'], component=NoiseComponent(variance=1.0), inputs=[])
        r = repr(node)
        assert 'Y' in r
        assert 'NoiseComponent' in r

    def test_repr_with_grouping(self):
        """ArchNode repr should show grouping."""
        node = ArchNode(outputs=['Y'], component=NoiseComponent(variance=1.0),
                        inputs=[], grouping='FID')
        r = repr(node)
        assert 'FID' in r


class TestArchitectureRepr:
    def test_repr_empty(self):
        """Empty architecture repr."""
        arch = Architecture()
        r = repr(arch)
        assert 'Architecture' in r
        assert '0' in r

    def test_repr_with_nodes(self):
        """Architecture repr should list outputs."""
        arch = Architecture()
        arch.add('Y', NoiseComponent(variance=1.0))
        arch.add('X', NoiseComponent(variance=2.0))
        r = repr(arch)
        assert 'Y' in r
        assert 'X' in r


class TestFromFormula:
    def test_from_formula_equals_constructor(self):
        """from_formula() and Architecture(formula=...) should be equivalent."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=5, seed=42)
        formula = """
            Y.G ~ genetic(beta)
            Y.E ~ noise(0.5)
            Y ~ Y.G + Y.E
        """
        a1 = Architecture.from_formula(formula, effects={'beta': eff})
        a2 = Architecture(formula=formula, effects={'beta': eff})
        assert len(a1.nodes) == len(a2.nodes)
        assert [n.outputs for n in a1.nodes] == [n.outputs for n in a2.nodes]
