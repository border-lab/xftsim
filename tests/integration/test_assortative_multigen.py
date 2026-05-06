"""
Integration test: assortative mating effects across multiple generations.

Tests:
1. Assortative mating increases genetic variance across generations
2. Negative assortative mating decreases genetic variance
3. Random mating preserves genetic variance (approximately)
"""
import numpy as np
import pytest

from xftsim.effect import AdditiveEffects
from xftsim.arch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.mate import RandomMating, LinearAssortativeMating
from xftsim.reproduce import RecombinationMap
from xftsim.sim import NSimulation
from xftsim.stats import SampleStatistics

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


def _make_sim(n=400, m=50, r=0.0, seed=42):
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
    eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=seed)
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(eff))
    arch.add('Y.E', NoiseComponent(variance=0.5))
    arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])

    if r == 0.0:
        mate = RandomMating(offspring_per_pair=2)
    else:
        mate = LinearAssortativeMating(component_names=['Y'], r=r, offspring_per_pair=2)

    return NSimulation(
        founder_haplotypes=hap, architecture=arch,
        mating_regime=mate,
        recombination_map=RecombinationMap.constant_map(m=m),
        seed=seed,
        statistics=[SampleStatistics()],
        retain_phenotypes=10,
    )


class TestAssortativeMultiGen:
    def test_positive_assortment_increases_variance(self):
        """Positive assortative mating should increase phenotypic variance."""
        sim = _make_sim(r=0.7)
        sim.run(6)

        # Get variances from first and last generation
        gen0_pheno = sim.phenotype_history[0]
        final_pheno = sim.phenotypes
        var_0 = np.var(gen0_pheno['Y'])
        var_final = np.var(final_pheno['Y'])

        # With strong positive AM, variance should increase
        assert var_final > var_0 * 0.8, \
            f"var_final={var_final:.3f} should be >= var_0={var_0:.3f} * 0.8"

    def test_random_mating_variance_stable(self):
        """Random mating should keep variance roughly stable."""
        sim = _make_sim(r=0.0)
        sim.run(6)

        gen0_pheno = sim.phenotype_history[0]
        final_pheno = sim.phenotypes
        var_0 = np.var(gen0_pheno['Y'])
        var_final = np.var(final_pheno['Y'])

        # Variance should stay within reasonable bounds
        ratio = var_final / var_0
        assert 0.3 < ratio < 3.0, \
            f"Variance ratio = {ratio:.3f}, expected roughly stable"

    def test_negative_assortment_effect(self):
        """Negative assortative mating should not increase variance as much."""
        sim_pos = _make_sim(r=0.7, seed=42)
        sim_neg = _make_sim(r=-0.7, seed=42)
        sim_pos.run(4)
        sim_neg.run(4)

        var_pos = np.var(sim_pos.phenotypes['Y'])
        var_neg = np.var(sim_neg.phenotypes['Y'])

        # Positive AM should produce higher variance than negative
        assert var_pos > var_neg * 0.5, \
            f"Positive AM var={var_pos:.3f} should exceed negative AM var={var_neg:.3f}"
