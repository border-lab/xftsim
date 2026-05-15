"""Tests for sibling aggregation components."""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray, PhenotypeArray
from xftsim.arch import (
    Architecture, NoiseComponent, AggregationComponent, ArchNode,
    SiblingMeanComponent, SiblingSumComponent, SiblingAnyComponent,
    SiblingCountComponent, SiblingEldestComponent, SiblingYoungestComponent,
    _SIBLING_COMPONENTS,
)
from xftsim.effect import AdditiveEffects
from xftsim.parser import parse_formula


def _make_family_haplotypes():
    """Create haplotypes with known family structure: 3 families, sizes 3,2,1."""
    n = 6
    m = 5
    rng = np.random.RandomState(0)
    geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
    fid = np.array([0, 0, 0, 1, 1, 2])
    iid = np.arange(n)
    sex = np.array([0, 1, 0, 1, 0, 1])
    samples = SampleMeta(iid=iid, fid=fid, sex=sex)
    variants = VariantMeta(vid=np.arange(m), af=np.full(m, 0.5))
    return DenseHaplotypeArray(genotypes=geno, samples=samples, variants=variants)


def _make_phenotypes_with_values(hap, vals, name='X'):
    """Create an PhenotypeArray with specific values."""
    pheno = PhenotypeArray(samples=hap.samples)
    pheno._values[name] = np.asarray(vals, dtype=np.float64)
    return pheno


class TestSiblingMean:
    def test_correctness(self):
        hap = _make_family_haplotypes()
        vals = np.array([3.0, 6.0, 9.0, 10.0, 20.0, 100.0])
        pheno = _make_phenotypes_with_values(hap, vals, 'X')
        comp = SiblingMeanComponent('X')
        node = ArchNode(outputs=['X.sib_mean'], component=comp, inputs=['X'])
        result = comp.compute(node, hap, pheno)
        # Family 0: mean(3,6,9) = 6; Family 1: mean(10,20) = 15; Family 2: 100
        expected = np.array([6.0, 6.0, 6.0, 15.0, 15.0, 100.0])
        np.testing.assert_allclose(result, expected)


class TestSiblingSum:
    def test_correctness(self):
        hap = _make_family_haplotypes()
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        pheno = _make_phenotypes_with_values(hap, vals, 'X')
        comp = SiblingSumComponent('X')
        node = ArchNode(outputs=['X.sib_sum'], component=comp, inputs=['X'])
        result = comp.compute(node, hap, pheno)
        # Family 0: 1+2+3=6; Family 1: 4+5=9; Family 2: 6
        expected = np.array([6.0, 6.0, 6.0, 9.0, 9.0, 6.0])
        np.testing.assert_allclose(result, expected)


class TestSiblingAny:
    def test_correctness(self):
        hap = _make_family_haplotypes()
        vals = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        pheno = _make_phenotypes_with_values(hap, vals, 'X')
        comp = SiblingAnyComponent('X')
        node = ArchNode(outputs=['X.any'], component=comp, inputs=['X'])
        result = comp.compute(node, hap, pheno)
        # Family 0: any positive → 1; Family 1: none → 0; Family 2: none → 0
        expected = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(result, expected)


class TestSiblingCount:
    def test_correctness(self):
        hap = _make_family_haplotypes()
        vals = np.zeros(6)  # values don't matter for count
        pheno = _make_phenotypes_with_values(hap, vals, 'X')
        comp = SiblingCountComponent('X')
        node = ArchNode(outputs=['X.count'], component=comp, inputs=['X'])
        result = comp.compute(node, hap, pheno)
        expected = np.array([3.0, 3.0, 3.0, 2.0, 2.0, 1.0])
        np.testing.assert_allclose(result, expected)


class TestSiblingEldest:
    def test_correctness(self):
        hap = _make_family_haplotypes()
        vals = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        pheno = _make_phenotypes_with_values(hap, vals, 'X')
        comp = SiblingEldestComponent('X')
        node = ArchNode(outputs=['X.eldest'], component=comp, inputs=['X'])
        result = comp.compute(node, hap, pheno)
        # Eldest = first in array: fam0→10, fam1→40, fam2→60
        expected = np.array([10.0, 10.0, 10.0, 40.0, 40.0, 60.0])
        np.testing.assert_allclose(result, expected)


class TestSiblingYoungest:
    def test_correctness(self):
        hap = _make_family_haplotypes()
        vals = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        pheno = _make_phenotypes_with_values(hap, vals, 'X')
        comp = SiblingYoungestComponent('X')
        node = ArchNode(outputs=['X.youngest'], component=comp, inputs=['X'])
        result = comp.compute(node, hap, pheno)
        # Youngest = last in array: fam0→30, fam1→50, fam2→60
        expected = np.array([30.0, 30.0, 30.0, 50.0, 50.0, 60.0])
        np.testing.assert_allclose(result, expected)


class TestSiblingGrouping:
    def test_default_fid_grouping(self):
        """Default grouping (None on node) should use FID."""
        hap = _make_family_haplotypes()
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        pheno = _make_phenotypes_with_values(hap, vals, 'X')
        comp = SiblingMeanComponent('X')
        # Explicit FID grouping
        node_fid = ArchNode(outputs=['X.mean1'], component=comp, inputs=['X'], grouping='FID')
        # Default grouping (None → falls back to FID in compute)
        node_default = ArchNode(outputs=['X.mean2'], component=comp, inputs=['X'], grouping=None)
        r1 = comp.compute(node_fid, hap, pheno)
        r2 = comp.compute(node_default, hap, pheno)
        np.testing.assert_allclose(r1, r2)

    def test_explicit_sex_grouping(self):
        """Explicit | sex grouping overrides FID."""
        hap = _make_family_haplotypes()
        vals = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
        pheno = _make_phenotypes_with_values(hap, vals, 'X')
        comp = SiblingMeanComponent('X')
        node = ArchNode(outputs=['X.mean_sex'], component=comp, inputs=['X'], grouping='sex')
        result = comp.compute(node, hap, pheno)
        # sex=0: indices 0,2,4 → values 2,6,10 → mean=6
        # sex=1: indices 1,3,5 → values 4,8,12 → mean=8
        expected = np.array([6.0, 8.0, 6.0, 8.0, 6.0, 8.0])
        np.testing.assert_allclose(result, expected)


class TestSiblingParser:
    def test_parse_sibling_mean(self):
        nodes = parse_formula(
            "X ~ noise(1.0)\nX.mean ~ sibling_mean(X)",
            effects={},
        )
        assert len(nodes) == 2
        assert isinstance(nodes[1].component, SiblingMeanComponent)
        assert nodes[1].inputs == ['X']

    def test_parse_with_grouping(self):
        nodes = parse_formula(
            "X ~ noise(1.0)\nX.mean ~ sibling_mean(X) | sex",
            effects={},
        )
        assert nodes[1].grouping == 'sex'

    def test_dag_ordering(self):
        """Sibling node must come after its source in topological order."""
        arch = Architecture()
        arch.add('X', NoiseComponent(variance=1.0))
        arch.add('X.mean', SiblingMeanComponent('X'), inputs=['X'])
        sorted_outputs = [n.outputs[0] for n in arch.nodes]
        assert sorted_outputs.index('X') < sorted_outputs.index('X.mean')


class TestSiblingErrors:
    def test_missing_source_raises(self):
        hap = _make_family_haplotypes()
        pheno = PhenotypeArray(samples=hap.samples)
        comp = SiblingMeanComponent('nonexistent')
        node = ArchNode(outputs=['out'], component=comp, inputs=['nonexistent'])
        with pytest.raises(ValueError, match="not found"):
            comp.compute(node, hap, pheno)

    def test_all_six_functions_registered(self):
        expected = {'sibling_mean', 'sibling_sum', 'sibling_any',
                    'sibling_count', 'sibling_eldest', 'sibling_youngest'}
        assert expected.issubset(set(_SIBLING_COMPONENTS.keys()))
        from xftsim.arch import BUILTINS
        assert expected.issubset(set(BUILTINS.keys()))


class TestSiblingE2E:
    def test_architecture_computes(self):
        hap = _make_family_haplotypes()
        arch = Architecture(
            formula="X ~ noise(1.0)\nX.mean ~ sibling_mean(X)\nX.count ~ sibling_count(X)",
            effects={},
        )
        pheno = arch.compute(hap, rng=np.random.RandomState(42))
        assert 'X.mean' in pheno
        assert 'X.count' in pheno
        # Count should match family sizes
        fids = hap.samples.fid
        for i in range(hap.n):
            expected_count = np.sum(fids == fids[i])
            assert pheno['X.count'][i] == expected_count
