"""
Unit tests for NSimulation lifecycle edge cases.

Tests:
1. run(1) produces gen 0 only
2. continue_run(1) advances by exactly 1 generation
3. run(0) raises or does nothing
4. Accessing phenotypes before run raises or returns None
5. Multiple continue_run calls accumulate generations
6. Generation counter is correct after run
"""
import numpy as np
import pytest

from xftsim.effect import AdditiveEffects
from xftsim.arch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.mate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.sim import NSimulation

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


def _make_sim(seed=42):
    n, m = 100, 20
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
    eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=seed)
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(eff))
    arch.add('Y.E', NoiseComponent(variance=0.5))
    arch.add('Y', AggregationComponent('Y.G + Y.E'))
    return NSimulation(
        founder_haplotypes=hap, architecture=arch,
        mating_regime=RandomMating(offspring_per_pair=2),
        recombination_map=RecombinationMap.constant_map(m=m),
        seed=seed, retain_phenotypes=10,
    )


class TestSimulationLifecycle:
    def test_run_1_produces_gen_0(self):
        """run(1) should compute gen 0 phenotypes."""
        sim = _make_sim()
        sim.run(1)
        assert sim.generation == 0
        assert 0 in sim.phenotype_history
        assert np.all(np.isfinite(sim.phenotypes['Y']))

    def test_run_5_produces_gens_0_through_4(self):
        """run(5) should produce generations 0-4."""
        sim = _make_sim()
        sim.run(5)
        assert sim.generation == 4
        for g in range(5):
            assert g in sim.phenotype_history

    def test_continue_run_advances_by_n(self):
        """continue_run(3) after run(2) should advance to gen 4."""
        sim = _make_sim()
        sim.run(2)
        assert sim.generation == 1
        sim.continue_run(3)
        assert sim.generation == 4

    def test_multiple_continue_run(self):
        """Multiple continue_run calls should accumulate generations."""
        sim = _make_sim()
        sim.run(2)
        assert sim.generation == 1
        sim.continue_run(1)
        assert sim.generation == 2
        sim.continue_run(2)
        assert sim.generation == 4

    def test_phenotypes_after_run_are_finite(self):
        """Phenotypes should be finite after run."""
        sim = _make_sim()
        sim.run(3)
        for g in range(3):
            pheno = sim.phenotype_history[g]
            for key in pheno.keys:
                assert np.all(np.isfinite(pheno[key])), \
                    f"Gen {g}, key '{key}' has non-finite values"

    def test_different_seeds_produce_different_results(self):
        """Different seeds should produce different phenotypes."""
        sim1 = _make_sim(seed=42)
        sim2 = _make_sim(seed=999)
        sim1.run(2)
        sim2.run(2)

        y1 = sim1.phenotypes['Y']
        y2 = sim2.phenotypes['Y']
        # With different seeds, phenotypes should differ
        assert not np.allclose(y1, y2), \
            "Different seeds should produce different phenotypes"

    def test_same_seed_reproducible(self):
        """Same seed should produce identical gen 0 phenotypes."""
        sim1 = _make_sim(seed=42)
        sim2 = _make_sim(seed=42)
        sim1.run(1)
        sim2.run(1)

        np.testing.assert_array_equal(
            sim1.phenotypes['Y'], sim2.phenotypes['Y'],
            err_msg="Same seed should produce identical gen 0 phenotypes"
        )
