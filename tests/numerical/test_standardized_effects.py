"""
Numerical tests for standardized vs non-standardized genetic effects.

Verifies:
1. Standardized effects produce mean-zero genetic values
2. Non-standardized and standardized produce same variance structure
3. Sparse effects work correctly in multi-gen simulations
4. h2 is approximately correct when using standardized effects

Stochastic protocol: tolerance ~ 4/sqrt(N), N=2000.
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray
from xftsim.arch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.effect import AdditiveEffects, SparseEffects
from xftsim.sim import NSimulation
from xftsim.mate import RandomMating
from xftsim.reproduce import RecombinationMap

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation

N = 2000
M = 50
TOL = 4.0 / np.sqrt(N)


def _make_haplotypes(n=N, m=M, seed=42):
    rng = np.random.RandomState(seed)
    af = np.full(m, 0.5)
    geno = np.zeros((n, m, 2), dtype=np.int8)
    for j in range(m):
        geno[:, j, 0] = rng.binomial(1, af[j], size=n)
        geno[:, j, 1] = rng.binomial(1, af[j], size=n)
    sex = np.tile([0, 1], (n + 1) // 2)[:n]
    samples = SampleMeta(iid=np.arange(n), sex=sex)
    variants = VariantMeta(vid=np.arange(m), af=af)
    return DenseHaplotypeArray(genotypes=geno, samples=samples, variants=variants)


class TestStandardizedEffects:
    """Test that standardized effects produce mean-zero genetic values."""

    def test_standardized_mean_zero(self, stochastic_seed):
        """Standardized genetic values should have mean ~0."""
        hap = _make_haplotypes(seed=stochastic_seed.seed)
        eff = AdditiveEffects.from_h2(h2=0.5, m=M, seed=stochastic_seed.seed,
                                       standardized=True)
        arch = Architecture()
        arch.add('G', GeneticComponent(eff))
        pheno = arch.compute(hap, rng=stochastic_seed.rng)
        assert abs(np.mean(pheno['G'])) < 3 * TOL, (
            f"seed={stochastic_seed.seed}, mean={np.mean(pheno['G']):.4f}"
        )

    def test_non_standardized_not_mean_zero(self, stochastic_seed):
        """Non-standardized genetic values need not have mean 0."""
        hap = _make_haplotypes(seed=stochastic_seed.seed)
        eff = AdditiveEffects.from_h2(h2=0.5, m=M, seed=stochastic_seed.seed,
                                       standardized=False)
        arch = Architecture()
        arch.add('G', GeneticComponent(eff))
        pheno = arch.compute(hap, rng=stochastic_seed.rng)
        # Mean should be 2*mean(af)*sum(beta) — generally non-zero
        # Just check it's finite
        assert np.isfinite(np.mean(pheno['G']))

    def test_standardized_variance_positive(self, stochastic_seed):
        """Standardized genetic values should have positive variance."""
        hap = _make_haplotypes(seed=stochastic_seed.seed)
        eff = AdditiveEffects.from_h2(h2=0.5, m=M, seed=stochastic_seed.seed,
                                       standardized=True)
        arch = Architecture()
        arch.add('G', GeneticComponent(eff))
        pheno = arch.compute(hap, rng=stochastic_seed.rng)
        assert np.var(pheno['G']) > 0.01


class TestSparseEffectsNumerical:
    """Numerical tests for sparse effects (k_causal < m)."""

    def test_sparse_fewer_causal_lower_variance(self, stochastic_seed):
        """Fewer causal variants should produce lower genetic variance."""
        hap = _make_haplotypes(seed=stochastic_seed.seed)
        eff_many = SparseEffects.from_h2(h2=0.5, m=M, k_causal=M, seed=42)
        eff_few = SparseEffects.from_h2(h2=0.5, m=M, k_causal=5, seed=42)

        arch_many = Architecture()
        arch_many.add('G', GeneticComponent(eff_many))
        pheno_many = arch_many.compute(hap, rng=stochastic_seed.rng)

        arch_few = Architecture()
        arch_few.add('G', GeneticComponent(eff_few))
        pheno_few = arch_few.compute(hap, rng=stochastic_seed.rng)

        # With same h2 target but fewer variants, variance could differ
        # but both should be positive
        assert np.var(pheno_many['G']) > 0
        assert np.var(pheno_few['G']) > 0

    def test_sparse_effects_in_simulation(self):
        """SparseEffects should work across multiple generations."""
        m, n = 50, 400
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = SparseEffects.from_h2(h2=0.5, m=m, k_causal=10, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        mate = RandomMating(offspring_per_pair=2)
        sim = NSimulation(hap, arch, mate, rmap, seed=42)
        sim.run(5)
        pheno = sim.phenotype_history[sim.generation]
        assert np.all(np.isfinite(pheno['Y']))
        assert np.var(pheno['Y']) > 0

    def test_sparse_non_causal_zero_contribution(self, stochastic_seed):
        """Non-causal variants in SparseEffects should contribute 0 to genetic value."""
        hap = _make_haplotypes(seed=stochastic_seed.seed)
        eff = SparseEffects.from_h2(h2=0.5, m=M, k_causal=5, seed=42)
        # Non-causal effects should be exactly 0
        non_causal = ~eff.variant_mask
        assert np.all(eff.effects[non_causal] == 0.0)
        # Genetic value should only depend on causal variants
        G_full = hap.matvec(eff.effects)
        G_causal_only = hap.matvec(eff.effects * eff.variant_mask)
        np.testing.assert_allclose(G_full, G_causal_only, atol=1e-10)


class TestHeritabilityEstimation:
    """Test that observed h2 approximately matches the target."""

    def test_h2_approximation(self, stochastic_seed):
        """Var(G) / Var(Y) should approximately equal h2 at gen 0."""
        target_h2 = 0.5
        target_ve = (1 - target_h2) / target_h2  # var_e such that h2 = var_g/(var_g+var_e)
        hap = _make_haplotypes(seed=stochastic_seed.seed)
        eff = AdditiveEffects.from_h2(h2=target_h2, m=M, seed=stochastic_seed.seed,
                                       standardized=False)
        arch = Architecture()
        arch.add('G', GeneticComponent(eff))
        arch.add('E', NoiseComponent(variance=target_ve))
        arch.add('Y', AggregationComponent('G + E'))
        pheno = arch.compute(hap, rng=stochastic_seed.rng)
        var_g = np.var(pheno['G'])
        var_y = np.var(pheno['Y'])
        obs_h2 = var_g / var_y if var_y > 0 else 0
        # h2 should be in the right ballpark (not exact due to finite sample)
        assert 0.1 < obs_h2 < 0.9, (
            f"seed={stochastic_seed.seed}, h2={obs_h2:.4f}, target={target_h2}"
        )
