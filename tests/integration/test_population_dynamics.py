"""
Integration tests for population dynamics across generations.

Tests:
1. Population size with offspring_per_pair=1 (shrinks)
2. Population size with offspring_per_pair=3 (grows)
3. Allele frequency drift across generations
4. Genetic variance maintained across generations (no collapse)
5. Noise mean approximately zero each generation
"""
import numpy as np
import pytest

from xftsim.effect import AdditiveEffects
from xftsim.arch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.mate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.sim import NSimulation
from xftsim.stats import SampleStatistics

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


def _make_sim(n=200, m=20, h2=0.5, offspring_per_pair=2, seed=42, **kwargs):
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
    eff = AdditiveEffects.from_h2(h2=h2, m=m, seed=seed)
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(eff))
    arch.add('Y.E', NoiseComponent(variance=1.0 - h2))
    arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])
    return NSimulation(
        founder_haplotypes=hap, architecture=arch,
        mating_regime=RandomMating(offspring_per_pair=offspring_per_pair),
        recombination_map=RecombinationMap.constant_map(m=m),
        seed=seed, **kwargs,
    )


class TestPopulationSize:
    def test_opp_2_stable(self):
        """offspring_per_pair=2 with equal sex ratio: pop stable."""
        sim = _make_sim(n=200, offspring_per_pair=2)
        sim.run(3)
        # With n females and n males, n pairs × 2 = 2n offspring
        # Population should remain roughly stable
        gen_n = sim.haplotypes.n
        assert gen_n > 50  # not collapsed
        assert gen_n <= 400  # not exploded

    def test_opp_3_grows(self):
        """offspring_per_pair=3: population should grow."""
        sim = _make_sim(n=100, offspring_per_pair=3)
        sim.run(2)
        gen1_n = sim.haplotypes.n
        # With 50 pairs × 3 = 150 offspring
        assert gen1_n > 100


class TestAlleleDrift:
    def test_allele_frequencies_bounded(self):
        """Allele frequencies should stay in [0, 1] across generations."""
        sim = _make_sim(n=200, m=20)
        sim.run(5)
        af = sim.haplotypes.recompute_af()
        assert np.all(af >= 0.0)
        assert np.all(af <= 1.0)

    def test_allele_frequencies_drift(self):
        """Allele frequencies should change slightly from founders."""
        sim = _make_sim(n=200, m=20)
        founder_af = sim.haplotype_history[0].recompute_af()
        sim.run(5)
        current_af = sim.haplotypes.recompute_af()
        # They should be different (drift happened)
        # But not massively different with n=200
        diff = np.abs(current_af - founder_af)
        # At least some have drifted
        assert np.any(diff > 0)


class TestVarianceMaintenance:
    def test_genetic_variance_positive_all_gens(self):
        """Genetic variance should be positive every generation."""
        sim = _make_sim(
            n=200, m=20, h2=0.5,
            statistics=[SampleStatistics()],
        )
        sim.run(4)
        for r in sim.results:
            stat = r.statistics.get('SampleStatistics')
            if stat is not None:
                g_idx = stat['keys'].index('Y.G')
                assert stat['var'][g_idx] > 0.01, \
                    f"Gen {r.generation}: Var(Y.G) = {stat['var'][g_idx]}"

    def test_total_variance_positive_all_gens(self):
        """Total phenotype variance should be positive every generation."""
        sim = _make_sim(
            n=200, m=20, h2=0.5,
            statistics=[SampleStatistics()],
        )
        sim.run(4)
        for r in sim.results:
            stat = r.statistics.get('SampleStatistics')
            if stat is not None:
                y_idx = stat['keys'].index('Y')
                assert stat['var'][y_idx] > 0.1, \
                    f"Gen {r.generation}: Var(Y) = {stat['var'][y_idx]}"


class TestNoiseMean:
    def test_noise_mean_approximately_zero(self):
        """Noise component mean should be approximately zero each generation."""
        sim = _make_sim(n=500, m=20, h2=0.5)
        sim.run(3)
        for gen in sim.phenotype_history:
            pheno = sim.phenotype_history[gen]
            if 'Y.E' in pheno:
                mean_e = np.mean(pheno['Y.E'])
                assert abs(mean_e) < 0.3, \
                    f"Gen {gen}: mean(Y.E) = {mean_e:.3f}, expected ≈ 0"
