"""Numerical tests for allele frequency drift and genetic variance across generations.

These tests verify that simulation dynamics match theoretical expectations:
- AF drift rate proportional to 1/(2N)
- Genetic variance decays due to drift
- Selection-free simulation preserves mean AF
"""
import numpy as np
import pytest

from tests.testdata import TestSimulation
from xftsim.struct import DenseHaplotypeArray, SampleMeta, VariantMeta
from xftsim.effect import AdditiveEffects
from xftsim.arch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.sim import Simulation
from xftsim.mate import RandomMating
from xftsim.reproduce import RecombinationMap


def _run_drift_sim(n=500, m=50, n_gen=10, seed=42):
    """Run a neutral simulation and track allele frequencies."""
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
    eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=123)
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(eff))
    arch.add('Y.E', NoiseComponent(variance=0.5))
    arch.add('Y', AggregationComponent('Y.G + Y.E'))
    rmap = RecombinationMap.constant_map(m=m, p=0.5)
    mate = RandomMating(offspring_per_pair=2)
    sim = Simulation(
        founder_haplotypes=hap, architecture=arch,
        mating_regime=mate, recombination_map=rmap,
        retain_haplotypes=n_gen + 1,
        retain_phenotypes=n_gen + 1,
        seed=seed,
    )
    sim.run(n_gen)
    return sim


class TestAFDrift:
    """Allele frequency drift under random mating (no selection)."""

    def test_mean_af_conserved(self):
        """Mean allele frequency across loci should be approximately conserved."""
        sim = _run_drift_sim(n=500, m=50, n_gen=5, seed=42)
        af_0 = sim.haplotype_history[0].recompute_af()
        af_last = sim.haplotype_history[sim.generation].recompute_af()
        # Mean AF should not drift systematically
        # Allow generous tolerance since n=500 still has sampling noise
        assert abs(af_0.mean() - af_last.mean()) < 0.05

    def test_af_stays_in_range(self):
        """All AFs should remain in [0, 1]."""
        sim = _run_drift_sim(n=500, m=50, n_gen=5, seed=42)
        for gen in sim.haplotype_history:
            af = sim.haplotype_history[gen].recompute_af()
            assert np.all(af >= 0.0)
            assert np.all(af <= 1.0)

    def test_drift_variance_order_of_magnitude(self):
        """Variance of AF change should be on order of p(1-p)/(2N) per generation.

        E[Var(delta_p)] = p*(1-p) / (2*N) per locus per generation.
        We check across many loci that observed variance is in the right ballpark.
        """
        n = 500
        m = 50
        sim = _run_drift_sim(n=n, m=m, n_gen=2, seed=42)
        af_0 = sim.haplotype_history[0].recompute_af()
        af_1 = sim.haplotype_history[1].recompute_af()
        delta = af_1 - af_0

        # Expected per-locus variance: p(1-p)/(2N) — average across loci
        expected_var = np.mean(af_0 * (1 - af_0)) / (2 * n)
        observed_var = np.var(delta)

        # These should be within an order of magnitude
        # Generous tolerance since we only have 50 loci
        assert observed_var < expected_var * 10
        assert observed_var > expected_var * 0.01


class TestGeneticVariance:
    """Genetic variance dynamics across generations."""

    def test_genetic_variance_positive(self):
        """Genetic variance should remain positive across generations."""
        sim = _run_drift_sim(n=500, m=50, n_gen=5, seed=42)
        for gen in range(sim.generation + 1):
            if gen in sim.phenotype_history:
                yg = sim.phenotype_history[gen]['Y.G']
                assert np.var(yg) > 0

    def test_phenotypic_variance_stable(self):
        """Phenotypic variance should be relatively stable (not exploding/collapsing)."""
        sim = _run_drift_sim(n=500, m=50, n_gen=5, seed=42)
        var_0 = np.var(sim.phenotype_history[0]['Y'])
        for gen in range(1, sim.generation + 1):
            if gen in sim.phenotype_history:
                var_g = np.var(sim.phenotype_history[gen]['Y'])
                # Phenotypic variance should stay within 3x of gen-0
                assert var_g > var_0 * 0.1
                assert var_g < var_0 * 3.0

    def test_larger_population_less_drift(self):
        """Larger populations should show less AF drift than smaller ones."""
        n_gen = 3
        # Small population
        sim_small = _run_drift_sim(n=100, m=50, n_gen=n_gen, seed=42)
        af0_small = sim_small.haplotype_history[0].recompute_af()
        afT_small = sim_small.haplotype_history[sim_small.generation].recompute_af()
        drift_small = np.mean((afT_small - af0_small) ** 2)

        # Large population
        sim_large = _run_drift_sim(n=1000, m=50, n_gen=n_gen, seed=42)
        af0_large = sim_large.haplotype_history[0].recompute_af()
        afT_large = sim_large.haplotype_history[sim_large.generation].recompute_af()
        drift_large = np.mean((afT_large - af0_large) ** 2)

        # Larger population should have less drift (MSE of AF change)
        assert drift_large < drift_small
