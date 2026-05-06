"""
Integration test: full pipeline over multiple generations with all features.

Tests:
1. 10-generation sim with AM + filters + stats + callbacks all work
2. Population size stays constant across generations
3. Phenotype keys consistent across generations
4. Statistics covariance matrix size matches trait count
5. Continue_run after checkpoint preserves phenotype structure
"""
import numpy as np
import pytest
import tempfile
import shutil

from xftsim.effect import AdditiveEffects, MultivariateEffects
from xftsim.arch import (
    Architecture, GeneticComponent, MVGeneticComponent,
    NoiseComponent, CNoiseComponent, AggregationComponent,
)
from xftsim.mate import RandomMating, LinearAssortativeMating
from xftsim.reproduce import RecombinationMap
from xftsim.sim import NSimulation
from xftsim.filters import TrioFilter, SibPairFilter
from xftsim.stats import SampleStatistics
from xftsim.io import save_simulation_checkpoint

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestFullPipelineMultigen:
    def test_10_gen_all_features(self):
        """10-generation run with AM, filters, stats, callbacks."""
        n, m = 200, 20
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))

        gen_log = []

        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=LinearAssortativeMating(
                component_names=['Y'], r=0.3, offspring_per_pair=2,
            ),
            recombination_map=RecombinationMap.constant_map(m=m),
            seed=42,
            statistics=[SampleStatistics()],
            filters={'trio': TrioFilter()},
            callbacks=[lambda s: gen_log.append(s.generation)],
            retain_phenotypes=10, retain_haplotypes=5,
        )
        sim.run(10)

        assert sim.generation == 9
        assert gen_log == list(range(10))
        assert len(sim.results) == 10

    def test_population_size_stable(self):
        """Population size should remain constant with offspring_per_pair=2."""
        n, m = 200, 20
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))

        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=RandomMating(offspring_per_pair=2),
            recombination_map=RecombinationMap.constant_map(m=m),
            seed=42, retain_phenotypes=10, retain_haplotypes=10,
        )
        sim.run(5)

        # All generations should have the same population size
        for g in sim.phenotype_history:
            pheno = sim.phenotype_history[g]
            assert pheno.samples.n == n, \
                f"Gen {g}: n={pheno.samples.n}, expected {n}"

    def test_phenotype_keys_consistent(self):
        """All generations should have the same phenotype keys."""
        n, m = 100, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))

        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=RandomMating(offspring_per_pair=2),
            recombination_map=RecombinationMap.constant_map(m=m),
            seed=42, retain_phenotypes=10,
        )
        sim.run(5)

        expected_keys = set(sim.phenotype_history[0].keys)
        for g in sim.phenotype_history:
            assert set(sim.phenotype_history[g].keys) == expected_keys, \
                f"Gen {g} keys don't match gen 0"

    def test_multitrait_stats_cov_size(self):
        """Statistics covariance matrix should match number of traits."""
        n, m = 200, 20
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        mv = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.2, m=m, seed=42)
        arch = Architecture()
        arch.add(['Y1.G', 'Y2.G'], MVGeneticComponent(mv))
        arch.add(['Y1.E', 'Y2.E'], CNoiseComponent(cov=np.diag([0.5, 0.7])))
        arch.add('Y1', AggregationComponent('Y1.G + Y1.E'))
        arch.add('Y2', AggregationComponent('Y2.G + Y2.E'))

        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=RandomMating(offspring_per_pair=2),
            recombination_map=RecombinationMap.constant_map(m=m),
            seed=42,
            statistics=[SampleStatistics()],
            retain_phenotypes=10,
        )
        sim.run(3)

        for result in sim.results:
            stats = result.statistics['SampleStatistics']
            n_keys = len(stats['keys'])
            assert stats['cov'].shape == (n_keys, n_keys)
            assert len(stats['var']) == n_keys

    def test_checkpoint_continue_preserves_structure(self):
        """Continue after checkpoint should produce consistent phenotype keys."""
        n, m = 100, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))

        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=RandomMating(offspring_per_pair=2),
            recombination_map=RecombinationMap.constant_map(m=m),
            seed=42, retain_phenotypes=10, retain_haplotypes=10,
        )
        sim.run(3)
        keys_before = set(sim.phenotypes.keys)

        tmpdir = tempfile.mkdtemp()
        try:
            save_simulation_checkpoint(sim, tmpdir)
            loaded = NSimulation.from_checkpoint(tmpdir)
            loaded.continue_run(2)
            keys_after = set(loaded.phenotypes.keys)
            assert keys_after == keys_before
        finally:
            shutil.rmtree(tmpdir)
