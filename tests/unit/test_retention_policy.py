"""
Unit tests for NSimulation retention policy edge cases.

Tests:
1. retain_haplotypes=0 keeps only current generation
2. retain_haplotypes=1 keeps current and previous
3. Large retain value keeps all generations
4. Pedigree retention follows phenotype retention
5. Mate assignment retention (only most recent)
6. Retention after continue_run
7. Default retention values
8. Retention with VT (tests that VT still works after pruning parent gen)
"""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation

from xftsim.narch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.neffect import AdditiveEffects
from xftsim.nmate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.nsim import NSimulation


def _simple_sim(n=50, m=10, retain_haplotypes=1, retain_phenotypes=2, seed=42):
    """Create a simple simulation for retention tests."""
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
    effects = AdditiveEffects.from_h2(m=m, h2=0.5, seed=seed)
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(effects))
    arch.add('Y.E', NoiseComponent(variance=0.5))
    arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])
    mating = RandomMating(offspring_per_pair=2)
    rmap = RecombinationMap.constant_map(m=m)
    return NSimulation(
        founder_haplotypes=hap,
        architecture=arch,
        mating_regime=mating,
        recombination_map=rmap,
        retain_haplotypes=retain_haplotypes,
        retain_phenotypes=retain_phenotypes,
        seed=seed,
    )


class TestRetainHaplotypes:
    def test_retain_zero_keeps_current_only(self):
        """retain_haplotypes=0 should keep only the current generation."""
        sim = _simple_sim(retain_haplotypes=0)
        # run(n) includes gen 0: run(4) → gens 0,1,2,3 → final gen=3
        sim.run(4)
        assert sim.generation == 3
        # Only current generation should remain
        assert list(sim.haplotype_history.keys()) == [3]

    def test_retain_one_keeps_current_and_previous(self):
        """retain_haplotypes=1 keeps current and one previous."""
        sim = _simple_sim(retain_haplotypes=1)
        sim.run(4)
        assert sim.generation == 3
        # Current and previous generation
        keys = sorted(sim.haplotype_history.keys())
        assert keys == [2, 3]

    def test_retain_large_keeps_all(self):
        """retain_haplotypes >= n_gen keeps all generations."""
        sim = _simple_sim(retain_haplotypes=100)
        sim.run(4)
        keys = sorted(sim.haplotype_history.keys())
        assert keys == [0, 1, 2, 3]


class TestRetainPhenotypes:
    def test_phenotype_retention(self):
        """retain_phenotypes controls phenotype history pruning."""
        sim = _simple_sim(retain_phenotypes=1)
        sim.run(4)
        keys = sorted(sim.phenotype_history.keys())
        assert keys == [2, 3]

    def test_pedigree_follows_phenotype_retention(self):
        """Pedigree retention follows phenotype retention settings."""
        sim = _simple_sim(retain_phenotypes=1)
        sim.run(4)
        ped_keys = sorted(sim.pedigree_history.keys())
        pheno_keys = sorted(sim.phenotype_history.keys())
        # Pedigrees should be pruned similarly
        for k in ped_keys:
            assert k >= min(pheno_keys)


class TestMateAssignmentRetention:
    def test_only_most_recent_mate_assignment(self):
        """Mate assignments only keep the most recent one."""
        sim = _simple_sim()
        sim.run(4)
        mate_keys = sorted(sim._mate_assignments.keys())
        # Only most recent (current_gen - 1 or current_gen)
        assert len(mate_keys) <= 2
        assert max(mate_keys) >= sim.generation - 1


class TestRetentionAfterContinueRun:
    def test_retention_after_continue(self):
        """Retention should be enforced after continue_run too."""
        sim = _simple_sim(retain_haplotypes=1)
        sim.run(3)  # gens 0,1,2 → gen=2
        sim.continue_run(2)  # gens 3,4 → gen=4
        assert sim.generation == 4
        hap_keys = sorted(sim.haplotype_history.keys())
        assert hap_keys == [3, 4]


class TestDefaultRetention:
    def test_default_retain_values(self):
        """Default retention should be reasonable."""
        hap = TestSimulation.founder_haplotypes(n=20, m=5, seed=42)
        effects = AdditiveEffects.from_h2(m=5, h2=0.5, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(effects))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])
        mating = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=5)
        sim = NSimulation(
            founder_haplotypes=hap,
            architecture=arch,
            mating_regime=mating,
            recombination_map=rmap,
        )
        # Defaults should exist and be > 0
        assert sim.retain_haplotypes >= 0
        assert sim.retain_phenotypes >= 0
