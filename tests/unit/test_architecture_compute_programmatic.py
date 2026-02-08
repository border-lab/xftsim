"""
Unit tests for Architecture.compute() with programmatic (non-formula) construction.

Tests:
1. Single genetic node produces correct phenotype
2. Multi-node: genetic + noise + aggregation
3. compute creates phenotype if None
4. compute uses existing phenotype
5. toposort cycle detection
6. Undefined reference in input raises
7. Self-referencing node (self-loop) allowed
8. Diamond dependency
"""
import numpy as np
import pytest

from xftsim.narch import (
    Architecture, GeneticComponent, NoiseComponent,
    AggregationComponent, ArchNode,
)
from xftsim.neffect import AdditiveEffects
from xftsim.struct import NPhenotypeArray

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestArchitectureComputeProgrammatic:
    def test_single_genetic_node(self):
        """Programmatic single-node architecture produces correct result."""
        n, m = 50, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, standardized=False, seed=42)

        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))

        pheno = arch.compute(hap, rng=np.random.RandomState(42))
        assert 'Y.G' in pheno
        expected = hap.matvec(eff.effects)
        np.testing.assert_allclose(pheno['Y.G'], expected)

    def test_multi_node_genetic_noise_agg(self):
        """genetic + noise + aggregation computes correctly."""
        n, m = 50, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, standardized=False, seed=42)

        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])

        rng = np.random.RandomState(42)
        pheno = arch.compute(hap, rng=rng)
        assert 'Y' in pheno
        np.testing.assert_allclose(pheno['Y'], pheno['Y.G'] + pheno['Y.E'])

    def test_compute_creates_phenotype_if_none(self):
        """compute() with phenotypes=None creates a fresh NPhenotypeArray."""
        n, m = 20, 5
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)

        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))

        pheno = arch.compute(hap)
        assert isinstance(pheno, NPhenotypeArray)
        assert 'Y.G' in pheno

    def test_compute_uses_existing_phenotype(self):
        """compute() with existing phenotype writes into it."""
        n, m = 20, 5
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)

        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))

        existing = NPhenotypeArray(hap.samples)
        existing['pre'] = np.ones(n)
        result = arch.compute(hap, phenotypes=existing)
        assert result is existing  # same object
        assert 'pre' in result  # preserved
        assert 'Y.G' in result  # added


class TestToposortEdgeCases:
    def test_cycle_detection(self):
        """Circular dependency → ValueError."""
        arch = Architecture()
        arch.add('A', AggregationComponent('B'), inputs=['B'])
        arch.add('B', AggregationComponent('A'), inputs=['A'])

        with pytest.raises(ValueError, match="[Cc]ycle"):
            _ = arch.nodes

    def test_undefined_reference(self):
        """Reference to non-existent output → ValueError."""
        arch = Architecture()
        arch.add('Y', AggregationComponent('missing + 1'), inputs=['missing'])

        with pytest.raises(ValueError, match="Undefined reference"):
            _ = arch.nodes

    def test_self_loop_allowed(self):
        """A node referencing its own output is a self-loop — allowed by toposort."""
        arch = Architecture()
        arch.add('Y', AggregationComponent('Y'), inputs=['Y'])
        nodes = arch.nodes
        assert len(nodes) == 1

    def test_diamond_dependency(self):
        """Diamond DAG: A → B, A → C, B+C → D."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('B', AggregationComponent('A * 2'), inputs=['A'])
        arch.add('C', AggregationComponent('A * 3'), inputs=['A'])
        arch.add('D', AggregationComponent('B + C'), inputs=['B', 'C'])

        nodes = arch.nodes
        names = [n.outputs[0] for n in nodes]
        assert names.index('A') < names.index('B')
        assert names.index('A') < names.index('C')
        assert names.index('B') < names.index('D')
        assert names.index('C') < names.index('D')

    def test_toposort_invalidation_on_add(self):
        """Adding a node invalidates the cached sort."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        _ = arch.nodes  # trigger sort
        assert arch._sorted is not None

        arch.add('B', AggregationComponent('A'), inputs=['A'])
        assert arch._sorted is None  # invalidated

        nodes = arch.nodes
        assert len(nodes) == 2
