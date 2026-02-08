"""
Integration tests for multi-generation phenotype properties.

Tests:
1. Phenotype keys consistent across generations
2. All phenotypes finite
3. Sample sizes stable across generations with opp=2
4. Generation attribute increments correctly
5. Pedigree connects adjacent generations
6. Phenotype mean drift is bounded
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


def _make_sim(n=200, m=20, seed=42, **kwargs):
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
    eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=seed)
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(eff))
    arch.add('Y.E', NoiseComponent(variance=0.5))
    arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])
    return NSimulation(
        founder_haplotypes=hap, architecture=arch,
        mating_regime=RandomMating(offspring_per_pair=2),
        recombination_map=RecombinationMap.constant_map(m=m),
        seed=seed, **kwargs,
    )


class TestPhenotypeKeysConsistency:
    def test_keys_same_across_generations(self):
        """All retained generations should have the same phenotype keys."""
        sim = _make_sim(retain_phenotypes=10)
        sim.run(5)
        gens = sorted(sim.phenotype_history.keys())
        keys_first = set(sim.phenotype_history[gens[0]].keys)
        for gen in gens[1:]:
            keys_gen = set(sim.phenotype_history[gen].keys)
            assert keys_gen == keys_first, \
                f"Gen {gen} keys {keys_gen} != gen {gens[0]} keys {keys_first}"

    def test_all_phenotypes_finite(self):
        """All phenotype values should be finite in every generation."""
        sim = _make_sim(n=200, retain_phenotypes=10)
        sim.run(5)
        for gen in sim.phenotype_history:
            pheno = sim.phenotype_history[gen]
            for key in pheno.keys:
                vals = pheno[key]
                assert np.all(np.isfinite(vals)), \
                    f"Gen {gen}, key {key}: contains non-finite values"


class TestPopulationSizeStability:
    def test_sample_sizes_stable(self):
        """With opp=2, population size should be roughly stable."""
        sim = _make_sim(n=200)
        sim.run(6)
        sizes = [sim.haplotype_history[g].n for g in sim.haplotype_history]
        for s in sizes:
            assert s >= 50, f"Population too small: {s}"
            assert s <= 500, f"Population too large: {s}"


class TestGenerationTracking:
    def test_generation_increments(self):
        """Simulation generation should increment correctly."""
        sim = _make_sim()
        sim.run(5)
        assert sim.generation == 4

    def test_haplotype_generations_match_keys(self):
        """Haplotype history keys should match actual generation metadata."""
        sim = _make_sim()
        sim.run(4)
        for gen, hap in sim.haplotype_history.items():
            assert hap.generation == gen, \
                f"History key {gen} but haplotype.generation={hap.generation}"


class TestPedigreeConnectivity:
    def test_pedigree_connects_adjacent_gens(self):
        """Pedigree should connect offspring to parents in previous generation."""
        sim = _make_sim(n=100, retain_haplotypes=10)
        sim.run(3)
        for gen in range(1, 3):
            if gen in sim.pedigree_history and (gen - 1) in sim.haplotype_history:
                ped = sim.pedigree_history[gen]
                parent_n = sim.haplotype_history[gen - 1].n
                assert np.all(ped.maternal_idx >= 0)
                assert np.all(ped.paternal_idx >= 0)
                assert np.all(ped.maternal_idx < parent_n), \
                    f"Gen {gen}: maternal_idx exceeds parent pop size {parent_n}"
                assert np.all(ped.paternal_idx < parent_n), \
                    f"Gen {gen}: paternal_idx exceeds parent pop size {parent_n}"


class TestPhenotypeMeanDrift:
    def test_mean_bounded(self):
        """Phenotype mean should not drift too far from zero."""
        sim = _make_sim(n=500, m=50)
        sim.run(10)
        for gen in sim.phenotype_history:
            pheno = sim.phenotype_history[gen]
            mean_y = np.mean(pheno['Y'])
            assert abs(mean_y) < 3.0, \
                f"Gen {gen}: mean(Y) = {mean_y:.3f}, expected near 0"
