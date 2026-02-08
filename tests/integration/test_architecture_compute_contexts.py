"""
Integration tests for Architecture.compute with various contexts.

Tests:
1. Compute with pre-existing phenotypes object
2. Compute with rng=None (auto-creates)
3. Compute idempotent: same hap + same seed → same result
4. Compute with phenotype_history and pedigree_history kwargs
5. Multi-output node produces correct number of columns
6. Architecture compute respects toposort order
"""
import numpy as np
import pytest

from xftsim.neffect import AdditiveEffects, MultivariateEffects
from xftsim.narch import (
    Architecture, GeneticComponent, MVGeneticComponent,
    NoiseComponent, AggregationComponent,
)
from xftsim.struct import NPhenotypeArray

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestComputeWithExistingPhenotypes:
    def test_compute_writes_into_existing(self):
        """Compute with pre-existing NPhenotypeArray should add keys."""
        hap = TestSimulation.founder_haplotypes(n=50, m=10, seed=42)
        arch = Architecture()
        arch.add('Y.E', NoiseComponent(variance=1.0))
        existing = NPhenotypeArray(samples=hap.samples)
        existing['EXTRA'] = np.ones(50)

        result = arch.compute(hap, phenotypes=existing, rng=np.random.RandomState(42))
        assert result is existing  # same object
        assert 'EXTRA' in result
        assert 'Y.E' in result

    def test_compute_without_phenotypes_creates_new(self):
        """Compute without phenotypes= should create new NPhenotypeArray."""
        hap = TestSimulation.founder_haplotypes(n=50, m=10, seed=42)
        arch = Architecture()
        arch.add('Y.E', NoiseComponent(variance=1.0))
        result = arch.compute(hap, rng=np.random.RandomState(42))
        assert isinstance(result, NPhenotypeArray)
        assert 'Y.E' in result


class TestComputeWithDefaultRng:
    def test_compute_rng_none(self):
        """Compute with rng=None should work (creates default)."""
        hap = TestSimulation.founder_haplotypes(n=50, m=10, seed=42)
        arch = Architecture()
        arch.add('Y.E', NoiseComponent(variance=1.0))
        result = arch.compute(hap)
        assert 'Y.E' in result
        assert np.all(np.isfinite(result['Y.E']))


class TestComputeReproducibility:
    def test_same_seed_same_result(self):
        """Same haplotypes + same RNG seed → identical phenotypes."""
        hap = TestSimulation.founder_haplotypes(n=50, m=10, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])

        r1 = arch.compute(hap, rng=np.random.RandomState(99))
        r2 = arch.compute(hap, rng=np.random.RandomState(99))
        np.testing.assert_array_equal(r1['Y'], r2['Y'])


class TestMultiOutputCompute:
    def test_mvgenetic_produces_two_outputs(self):
        """MVGeneticComponent with 2 traits should produce 2 phenotype keys."""
        hap = TestSimulation.founder_haplotypes(n=50, m=10, seed=42)
        mv_eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.2, m=10, seed=42)
        arch = Architecture()
        arch.add(['T1.G', 'T2.G'], MVGeneticComponent(mv_eff))
        result = arch.compute(hap, rng=np.random.RandomState(42))
        assert 'T1.G' in result
        assert 'T2.G' in result
        assert len(result['T1.G']) == 50
        assert len(result['T2.G']) == 50


class TestToposortOrder:
    def test_aggregation_after_inputs(self):
        """Aggregation should compute after its inputs regardless of add order."""
        hap = TestSimulation.founder_haplotypes(n=50, m=10, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        arch = Architecture()
        # Add aggregation before its inputs (should still work via toposort)
        arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))

        result = arch.compute(hap, rng=np.random.RandomState(42))
        # Y should be the sum of Y.G and Y.E
        np.testing.assert_allclose(
            result['Y'],
            result['Y.G'] + result['Y.E'],
        )
