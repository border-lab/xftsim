"""
Unit tests for Architecture.compute() method directly.

Tests:
1. Empty architecture produces empty phenotype
2. Single noise node
3. Multi-output node (mvGenetic)
4. compute() creates phenotype if None
5. compute() uses provided phenotype
6. compute() with explicit rng for reproducibility
7. Architecture repr
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray, PhenotypeArray
from xftsim.arch import (
    Architecture, GeneticComponent, MVGeneticComponent, NoiseComponent,
    AggregationComponent, ArchNode,
)
from xftsim.effect import AdditiveEffects, MultivariateEffects


def _make_hap(n=20, m=10, seed=42):
    rng = np.random.RandomState(seed)
    sm = SampleMeta(iid=np.arange(n), fid=np.arange(n) // 2)
    vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
    geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
    return DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)


class TestEmptyArchitecture:
    def test_empty_compute(self):
        arch = Architecture()
        hap = _make_hap()
        pheno = arch.compute(hap)
        assert isinstance(pheno, PhenotypeArray)
        assert len(pheno.keys) == 0

    def test_empty_nodes(self):
        arch = Architecture()
        assert len(arch.nodes) == 0

    def test_empty_repr(self):
        arch = Architecture()
        assert 'Architecture' in repr(arch)
        assert 'nodes=0' in repr(arch)


class TestSingleNodeCompute:
    def test_noise_only(self):
        arch = Architecture()
        arch.add('Y', NoiseComponent(variance=1.0))
        hap = _make_hap()
        pheno = arch.compute(hap, rng=np.random.RandomState(42))
        assert 'Y' in pheno.keys
        assert pheno['Y'].shape == (20,)

    def test_genetic_only(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        hap = _make_hap()
        pheno = arch.compute(hap)
        assert 'Y.G' in pheno.keys
        expected = hap.standardized_matvec(eff.effects)
        np.testing.assert_allclose(pheno['Y.G'], expected)


class TestMultiOutputCompute:
    def test_mv_genetic(self):
        mv_eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.2, m=10, seed=42)
        arch = Architecture()
        arch.add(['T1.G', 'T2.G'], MVGeneticComponent(mv_eff))
        hap = _make_hap()
        pheno = arch.compute(hap)
        assert 'T1.G' in pheno.keys
        assert 'T2.G' in pheno.keys
        assert pheno['T1.G'].shape == (20,)
        assert pheno['T2.G'].shape == (20,)


class TestComputeOptions:
    def test_creates_phenotype_if_none(self):
        arch = Architecture()
        arch.add('Y', NoiseComponent(variance=1.0))
        hap = _make_hap()
        pheno = arch.compute(hap, phenotypes=None, rng=np.random.RandomState(42))
        assert isinstance(pheno, PhenotypeArray)
        assert 'Y' in pheno.keys

    def test_uses_provided_phenotype(self):
        arch = Architecture()
        arch.add('Y', NoiseComponent(variance=1.0))
        hap = _make_hap()
        existing = PhenotypeArray(samples=hap.samples)
        existing._values['pre'] = np.ones(20)
        result = arch.compute(hap, phenotypes=existing, rng=np.random.RandomState(42))
        assert result is existing
        assert 'pre' in result.keys
        assert 'Y' in result.keys

    def test_rng_reproducibility(self):
        arch = Architecture()
        arch.add('Y', NoiseComponent(variance=1.0))
        hap = _make_hap()
        r1 = arch.compute(hap, rng=np.random.RandomState(42))
        r2 = arch.compute(hap, rng=np.random.RandomState(42))
        np.testing.assert_array_equal(r1['Y'], r2['Y'])

    def test_different_rng_different_result(self):
        arch = Architecture()
        arch.add('Y', NoiseComponent(variance=1.0))
        hap = _make_hap()
        r1 = arch.compute(hap, rng=np.random.RandomState(42))
        r2 = arch.compute(hap, rng=np.random.RandomState(99))
        assert not np.array_equal(r1['Y'], r2['Y'])


class TestArchitectureRepr:
    def test_repr_with_nodes(self):
        arch = Architecture()
        arch.add('Y.G', NoiseComponent(variance=1.0))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        r = repr(arch)
        assert 'nodes=2' in r
        assert 'Y.G' in r
        assert 'Y.E' in r
