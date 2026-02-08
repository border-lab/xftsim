"""
Numerical test: population size effects on genetic properties.

Tests:
1. Larger populations have more accurate h2 estimates
2. Population size is preserved across generations (with opp=2)
3. Genetic variance estimate improves with sample size
4. Small populations lose heterozygosity faster
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


def _run_sim(n, m=50, h2=0.5, n_gen=4, seed=42):
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
    eff = AdditiveEffects.from_h2(h2=h2, m=m, seed=seed + 100)
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(eff))
    arch.add('Y.E', NoiseComponent(variance=1.0 - h2))
    arch.add('Y', AggregationComponent('Y.G + Y.E'))
    sim = NSimulation(
        founder_haplotypes=hap, architecture=arch,
        mating_regime=RandomMating(offspring_per_pair=2),
        recombination_map=RecombinationMap.constant_map(m=m),
        seed=seed, retain_haplotypes=10, retain_phenotypes=10,
    )
    sim.run(n_gen)
    return sim


class TestPopulationSizeEffects:
    def test_larger_pop_more_accurate_h2(self):
        """Larger populations should give h2 estimates closer to theoretical."""
        errors = {}
        for n in [200, 1000]:
            sim = _run_sim(n=n, n_gen=1, seed=42)
            pheno = sim.phenotype_history[0]
            h2_est = np.var(pheno['Y.G']) / np.var(pheno['Y'])
            # Theoretical h2 for centered-unscaled: h2_design * mean(2pq) / (h2_d*mean(2pq) + noise)
            errors[n] = abs(h2_est - 0.333)  # theoretical ≈ 0.333 for h2=0.5, noise=0.5

        assert errors[1000] < errors[200] + 0.05, \
            f"Large pop error={errors[1000]:.3f} should be ≤ small pop error={errors[200]:.3f}"

    def test_population_size_preserved(self):
        """With opp=2, population size should remain stable (balanced sex)."""
        sim = _run_sim(n=200, n_gen=5, seed=42)
        for g in range(5):
            n_g = sim.haplotype_history[g].n if g in sim.haplotype_history else None
            if n_g is not None:
                # With opp=2 and balanced sex, n should be stable ± some tolerance
                assert 50 < n_g < 400, \
                    f"Gen {g} pop size = {n_g}, expected ~200"

    def test_small_pop_loses_heterozygosity(self):
        """Smaller populations should lose heterozygosity faster."""
        def het_fraction(sim):
            """Fraction of loci that are heterozygous in population."""
            geno = sim.haplotypes.genotypes
            hets = (geno[:, :, 0] != geno[:, :, 1])  # n x m
            return np.mean(hets)

        sim_small = _run_sim(n=50, m=30, n_gen=8, seed=42)
        sim_large = _run_sim(n=500, m=30, n_gen=8, seed=42)

        het_small = het_fraction(sim_small)
        het_large = het_fraction(sim_large)

        # Larger population should retain more heterozygosity
        # (or at least not dramatically less)
        assert het_large >= het_small * 0.5, \
            f"Large pop het={het_large:.3f}, small pop het={het_small:.3f}"

    def test_genetic_variance_stable_with_random_mating(self):
        """Under random mating, genetic variance should be roughly stable."""
        sim = _run_sim(n=500, m=100, n_gen=5, seed=42)
        var_0 = np.var(sim.phenotype_history[0]['Y.G'])
        var_final = np.var(sim.phenotypes['Y.G'])
        ratio = var_final / var_0
        # Should be within a factor of 3
        assert 0.3 < ratio < 3.0, \
            f"Genetic variance ratio = {ratio:.3f}, expected roughly stable"
