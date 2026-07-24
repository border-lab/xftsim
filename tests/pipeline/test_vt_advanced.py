"""
Advanced vertical transmission integration tests.

Tests:
1. VT with retention pruning: VT still works when old gens are pruned
2. VT with sibling components together
3. VT midparent variance check: should be bounded
4. Multi-trait VT: two traits with independent VT
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray
from xftsim.arch import (
    Architecture, GeneticComponent, NoiseComponent, AggregationComponent,
    MotherComponent, FatherComponent, ParentComponent,
    SiblingMeanComponent, SiblingCountComponent,
)
from xftsim.effect import AdditiveEffects
from xftsim.sim import Simulation
from xftsim.mate import RandomMating
from xftsim.reproduce import RecombinationMap

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestVTWithRetention:
    def test_vt_works_with_aggressive_retention(self):
        """VT should work even with retain_phenotypes=2 (only prev gen available)."""
        n, m = 100, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.3, m=m, seed=42)

        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.3))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        arch.add('Y.VTm', MotherComponent('Y', founder_component=NoiseComponent(variance=0.1)))

        sim = Simulation(
            hap, arch, RandomMating(offspring_per_pair=2),
            RecombinationMap.constant_map(m=m, p=0.5),
            seed=42, retain_haplotypes=1, retain_phenotypes=2,
        )
        sim.run(5)
        assert sim.generation == 4
        pheno = sim.phenotype_history[sim.generation]
        assert np.all(np.isfinite(pheno['Y.VTm']))


class TestVTWithSiblings:
    def test_vt_and_sibling_together(self):
        """Simulation with VT and sibling components should run."""
        n, m = 100, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.3, m=m, seed=42)

        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.3))
        arch.add('Y.VTm', MotherComponent('Y', founder_component=NoiseComponent(variance=0.1)))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        arch.add('Y.sib_mean', SiblingMeanComponent('Y'), inputs=['Y'])
        arch.add('Y.sib_count', SiblingCountComponent('Y'), inputs=['Y'])

        sim = Simulation(
            hap, arch, RandomMating(offspring_per_pair=3),
            RecombinationMap.constant_map(m=m, p=0.5),
            seed=42, retain_phenotypes=2,
        )
        sim.run(3)
        pheno = sim.phenotype_history[sim.generation]
        assert np.all(np.isfinite(pheno['Y.VTm']))
        assert np.all(np.isfinite(pheno['Y.sib_mean']))
        assert np.all(pheno['Y.sib_count'] >= 1)


class TestVTVarianceBounded:
    def test_midparent_variance_bounded(self):
        """VT midparent variance should not grow unbounded."""
        n, m = 200, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.3, m=m, seed=42)

        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.3))
        arch.add('Y.VTp', ParentComponent('Y', founder_component=NoiseComponent(variance=0.1)))
        arch.add('Y', AggregationComponent('Y.G + Y.E + 0.3 * Y.VTp'))

        variances = []
        def track_var(sim):
            pheno = sim.phenotype_history[sim.generation]
            variances.append(np.var(pheno['Y']))

        sim = Simulation(
            hap, arch, RandomMating(offspring_per_pair=2),
            RecombinationMap.constant_map(m=m, p=0.5),
            seed=42, callbacks=[track_var],
            retain_haplotypes=1, retain_phenotypes=2,
        )
        sim.run(8)
        assert all(np.isfinite(v) for v in variances)
        assert all(v > 0 for v in variances)
        # Variance shouldn't grow more than 10x from first gen
        assert max(variances) < variances[0] * 10


class TestMultiTraitVT:
    def test_two_trait_vt(self):
        """Two traits with independent VT should both propagate."""
        n, m = 100, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff1 = AdditiveEffects.from_h2(h2=0.3, m=m, seed=42)
        eff2 = AdditiveEffects.from_h2(h2=0.4, m=m, seed=43)

        arch = Architecture()
        arch.add('Y1.G', GeneticComponent(eff1))
        arch.add('Y1.E', NoiseComponent(variance=0.3))
        arch.add('Y1', AggregationComponent('Y1.G + Y1.E'))
        arch.add('Y2.G', GeneticComponent(eff2))
        arch.add('Y2.E', NoiseComponent(variance=0.2))
        arch.add('Y2', AggregationComponent('Y2.G + Y2.E'))
        arch.add('Y1.VTm', MotherComponent('Y1', founder_component=NoiseComponent(variance=0.1)))
        arch.add('Y2.VTm', MotherComponent('Y2', founder_component=NoiseComponent(variance=0.1)))

        sim = Simulation(
            hap, arch, RandomMating(offspring_per_pair=2),
            RecombinationMap.constant_map(m=m, p=0.5),
            seed=42, retain_phenotypes=2,
        )
        sim.run(3)
        pheno = sim.phenotype_history[sim.generation]
        assert np.all(np.isfinite(pheno['Y1.VTm']))
        assert np.all(np.isfinite(pheno['Y2.VTm']))
        # Y1.VTm and Y2.VTm should differ (different parent phenotypes)
        assert not np.allclose(pheno['Y1.VTm'], pheno['Y2.VTm'])
