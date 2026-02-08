"""
Unit tests for NSimulation validation error paths.

Tests:
1. Effect dimension mismatch (m) detected at run() time
2. continue_run(0) does nothing
3. Accessing phenotypes before run() raises KeyError
4. run(0) raises or does nothing
"""
import numpy as np
import pytest

from xftsim.neffect import AdditiveEffects
from xftsim.narch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.nmate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.nsim import NSimulation

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestNSimValidation:
    def test_effect_dimension_mismatch_raises(self):
        """Effects with wrong m should raise ValueError at run()."""
        n, m = 50, 20
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        # Effects with m=10 vs haplotypes with m=20
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))

        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=RandomMating(offspring_per_pair=2),
            recombination_map=RecombinationMap.constant_map(m=m),
            seed=42,
        )
        with pytest.raises(ValueError, match="[Ee]ffect dimension mismatch"):
            sim.run(1)

    def test_phenotypes_before_run_raises(self):
        """Accessing phenotypes before run should raise KeyError."""
        n, m = 50, 20
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
            seed=42,
        )
        with pytest.raises(KeyError):
            _ = sim.phenotypes

    def test_continue_run_0_noop(self):
        """continue_run(0) should not change generation."""
        n, m = 50, 20
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
            seed=42,
        )
        sim.run(2)
        gen_before = sim.generation
        sim.continue_run(0)
        assert sim.generation == gen_before
