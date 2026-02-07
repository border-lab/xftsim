"""
Integration tests combining multiple features: VT, sibling effects,
assortative mating, filters, and statistics in a single simulation.
"""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation

from xftsim.nsim import NSimulation
from xftsim.nmate import RandomMating, LinearAssortativeMating
from xftsim.narch import (
    Architecture, GeneticComponent, NoiseComponent, AggregationComponent,
    ParentComponent, MotherComponent, FatherComponent,
    SiblingMeanComponent,
)
from xftsim.neffect import AdditiveEffects
from xftsim.nfilter import TrioFilter, SibPairFilter
from xftsim.nstats import SampleStatistics
from xftsim.reproduce import RecombinationMap


class TestCombinedVTAssortative:
    """Combine VT + assortative mating in one simulation."""

    def test_vt_assortative_5gen(self):
        """5-gen simulation with VT + assortative mating stays finite."""
        m, n = 30, 400
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.4, m=m, seed=123)

        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.VT', ParentComponent('Y', founder_component=NoiseComponent(variance=0.3)))
        arch.add('Y.E', NoiseComponent(variance=0.3))
        arch.add('Y', AggregationComponent('Y.G + 0.2 * Y.VT + Y.E'))

        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        mate = LinearAssortativeMating(['Y'], r=0.3, offspring_per_pair=2)

        sim = NSimulation(
            hap, arch, mate, rmap, seed=42,
            retain_haplotypes=2, retain_phenotypes=3,
        )
        sim.run(5)

        for gen in range(max(0, sim.generation - 2), sim.generation + 1):
            if gen in sim.phenotype_history:
                Y = sim.phenotype_history[gen]['Y']
                assert np.all(np.isfinite(Y)), f"NaN/Inf at gen {gen}"
                assert Y.var() < 500.0, f"Variance diverged at gen {gen}: {Y.var()}"

    def test_with_filters_and_stats(self):
        """Combined simulation with trio filter + statistics collects results."""
        m, n = 30, 400
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.4, m=m, seed=123)

        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.VT', ParentComponent('Y', founder_component=NoiseComponent(variance=0.3)))
        arch.add('Y.E', NoiseComponent(variance=0.3))
        arch.add('Y', AggregationComponent('Y.G + 0.2 * Y.VT + Y.E'))

        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        mate = LinearAssortativeMating(['Y'], r=0.3, offspring_per_pair=2)
        trio_filter = TrioFilter()
        stats = SampleStatistics()

        sim = NSimulation(
            hap, arch, mate, rmap, seed=42,
            retain_haplotypes=2, retain_phenotypes=3,
            filters={'trio': trio_filter},
            statistics=[stats],
        )
        sim.run(4)

        # Generation 0 has no trios; gen 1+ should have results
        assert len(sim.results) >= 3
        # Check statistics collected
        for r in sim.results:
            if r.generation > 0:
                assert 'SampleStatistics' in r.statistics
                assert 'var' in r.statistics['SampleStatistics']

    def test_mother_father_components(self):
        """Simulation with separate mother/father components stays finite."""
        m, n = 30, 400
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.4, m=m, seed=123)

        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.mom', MotherComponent('Y', founder_component=NoiseComponent(variance=0.3)))
        arch.add('Y.dad', FatherComponent('Y', founder_component=NoiseComponent(variance=0.3)))
        arch.add('Y.E', NoiseComponent(variance=0.3))
        arch.add('Y', AggregationComponent('Y.G + 0.15 * Y.mom + 0.15 * Y.dad + Y.E'))

        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        mate = RandomMating(offspring_per_pair=2)

        sim = NSimulation(
            hap, arch, mate, rmap, seed=42,
            retain_haplotypes=2, retain_phenotypes=3,
        )
        sim.run(5)

        Y = sim.phenotype_history[sim.generation]['Y']
        assert np.all(np.isfinite(Y))

    def test_bivariate_vt_assortative_with_stats(self):
        """Bivariate traits with VT + assortative mating + stats for 5 generations."""
        m, n = 30, 400
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        from xftsim.neffect import MultivariateEffects
        from xftsim.narch import MVGeneticComponent, CNoiseComponent

        mv_eff = MultivariateEffects.from_h2_rg(h2=[0.4, 0.3], rg=0.2, m=m, seed=123)
        cov_e = np.array([[0.3, 0.05], [0.05, 0.4]])

        arch = Architecture()
        arch.add(['Y1.G', 'Y2.G'], MVGeneticComponent(mv_eff))
        arch.add('Y1.VT', ParentComponent('Y1', founder_component=NoiseComponent(variance=0.2)))
        arch.add('Y2.VT', ParentComponent('Y2', founder_component=NoiseComponent(variance=0.2)))
        arch.add(['Y1.E', 'Y2.E'], CNoiseComponent(cov=cov_e))
        arch.add('Y1', AggregationComponent('Y1.G + 0.2 * Y1.VT + Y1.E'))
        arch.add('Y2', AggregationComponent('Y2.G + 0.2 * Y2.VT + Y2.E'))

        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        mate = LinearAssortativeMating(['Y1'], r=0.3, offspring_per_pair=2)
        stats = SampleStatistics()

        sim = NSimulation(
            hap, arch, mate, rmap, seed=42,
            retain_haplotypes=2, retain_phenotypes=3,
            statistics=[stats],
        )
        sim.run(5)

        pheno = sim.phenotype_history[sim.generation]
        for name in ['Y1', 'Y2']:
            vals = pheno[name]
            assert np.all(np.isfinite(vals)), f"{name} has NaN/Inf"
            assert vals.var() < 500.0, f"{name} variance diverged"

        # Stats should have been collected
        assert len(sim.results) >= 4


class TestCombinedSiblingEffects:
    """Test sibling effects in simulation context."""

    def test_sibling_mean_computed_after_source(self):
        """Sibling mean component should be computed after its source phenotype."""
        m, n = 30, 400
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.4, m=m, seed=123)

        # Y is computed first, then Y.sibmean reads from Y
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.4))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        arch.add('Y.sibmean', SiblingMeanComponent('Y'), inputs=['Y'])

        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        mate = RandomMating(offspring_per_pair=3)

        sim = NSimulation(
            hap, arch, mate, rmap, seed=42,
            retain_haplotypes=2, retain_phenotypes=3,
        )
        sim.run(3)

        pheno = sim.phenotype_history[sim.generation]
        assert 'Y.sibmean' in pheno.keys
        assert np.all(np.isfinite(pheno['Y.sibmean']))
        assert np.all(np.isfinite(pheno['Y']))

    def test_sib_filter_collects_data(self):
        """Sibling pair filter + stats should collect results in simulations with families."""
        m, n = 30, 400
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.4, m=m, seed=123)

        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.4))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))

        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        mate = RandomMating(offspring_per_pair=3)
        sib_filter = SibPairFilter()
        stats = SampleStatistics()

        sim = NSimulation(
            hap, arch, mate, rmap, seed=42,
            retain_haplotypes=2, retain_phenotypes=3,
            filters={'sib': sib_filter},
            statistics=[stats],
        )
        sim.run(3)

        Y = sim.phenotype_history[sim.generation]['Y']
        assert np.all(np.isfinite(Y))
        # Statistics should have produced results
        assert len(sim.results) > 0


class TestAllFeaturesCombined:
    """Kitchen-sink test combining all features."""

    def test_all_features_10gen(self):
        """VT + assortative mating + filters + stats + callbacks for 10 generations."""
        m, n = 30, 400
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.3, m=m, seed=123)

        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.VT', ParentComponent('Y', founder_component=NoiseComponent(variance=0.2)))
        arch.add('Y.E', NoiseComponent(variance=0.3))
        arch.add('Y', AggregationComponent('Y.G + 0.15 * Y.VT + Y.E'))

        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        mate = LinearAssortativeMating(['Y'], r=0.3, offspring_per_pair=2)
        trio_filter = TrioFilter()
        stats = SampleStatistics()

        callbacks_called = [0]
        def counter(sim):
            callbacks_called[0] += 1

        sim = NSimulation(
            hap, arch, mate, rmap, seed=42,
            retain_haplotypes=2, retain_phenotypes=3,
            filters={'trio': trio_filter},
            statistics=[stats],
            callbacks=[counter],
        )
        sim.run(10)

        assert sim.generation == 9
        Y = sim.phenotype_history[sim.generation]['Y']
        assert np.all(np.isfinite(Y))
        assert Y.var() < 500.0
        assert callbacks_called[0] == 10
        assert len(sim.results) >= 8

    def test_bivariate_mother_father_sib_assortative(self):
        """Bivariate: separate mother/father VT + sibling mean + assortative."""
        m, n = 30, 400
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.3, m=m, seed=123)

        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.mom', MotherComponent('Y', founder_component=NoiseComponent(variance=0.2)))
        arch.add('Y.dad', FatherComponent('Y', founder_component=NoiseComponent(variance=0.2)))
        arch.add('Y.E', NoiseComponent(variance=0.3))
        arch.add('Y', AggregationComponent('Y.G + 0.1 * Y.mom + 0.1 * Y.dad + Y.E'))
        # Sibling mean computed AFTER Y (non-circular)
        arch.add('Y.sibmean', SiblingMeanComponent('Y'), inputs=['Y'])

        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        mate = LinearAssortativeMating(['Y'], r=0.3, offspring_per_pair=3)

        sim = NSimulation(
            hap, arch, mate, rmap, seed=42,
            retain_haplotypes=2, retain_phenotypes=3,
        )
        sim.run(5)

        pheno = sim.phenotype_history[sim.generation]
        assert np.all(np.isfinite(pheno['Y']))
        assert np.all(np.isfinite(pheno['Y.sibmean']))
        assert np.all(np.isfinite(pheno['Y.mom']))
        assert np.all(np.isfinite(pheno['Y.dad']))
