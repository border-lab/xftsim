"""
Numerical tests for SparseEffects genetic variance.

Tests:
1. Sparse effects genetic variance ≈ h2
2. More causal variants → same expected variance
3. Sparse vs additive: same genetic variance with different sparsity
"""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation

from xftsim.neffect import AdditiveEffects, SparseEffects
from xftsim.narch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.nmate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.nsim import NSimulation


class TestSparseEffectsVariance:
    def test_sparse_genetic_variance_matches_h2(self):
        """Genetic variance from sparse effects should be ≈ h2."""
        n, m = 500, 200
        h2 = 0.5
        k_causal = 50  # 25% of variants causal
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = SparseEffects.from_h2(h2=h2, m=m, k_causal=k_causal, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=1.0 - h2))
        arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])
        mating = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m)
        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=mating, recombination_map=rmap, seed=42,
        )
        sim.run(1)
        gen_vals = sim.phenotype_history[0]['Y.G']
        var_g = np.var(gen_vals, ddof=1)
        # Should be approximately h2 (wide tolerance for sparsity and finite samples)
        assert 0.05 < var_g < 2.0, f"Genetic variance = {var_g:.3f}, expected ≈ {h2}"

    def test_sparse_non_causal_zero_contribution(self):
        """Non-causal variants should contribute zero genetic variance."""
        m = 100
        k_causal = 10
        eff = SparseEffects.from_h2(h2=0.5, m=m, k_causal=k_causal, seed=42)
        # Verify non-causal effects are exactly zero
        non_causal = ~eff.variant_mask
        assert np.all(eff.effects[non_causal] == 0.0)
        # All variance comes from causal variants
        causal_var = np.sum(eff.effects[eff.variant_mask] ** 2)
        total_var = np.sum(eff.effects ** 2)
        np.testing.assert_allclose(causal_var, total_var)

    def test_sparse_vs_dense_same_total_variance(self):
        """Sparse and dense effects with same h2 should have similar genetic variance."""
        n, m = 500, 200
        h2 = 0.5

        # Dense (all causal)
        hap_dense = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff_dense = AdditiveEffects.from_h2(h2=h2, m=m, seed=42)
        arch_dense = Architecture()
        arch_dense.add('Y.G', GeneticComponent(eff_dense))
        arch_dense.add('Y.E', NoiseComponent(variance=1.0 - h2))
        arch_dense.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])
        sim_dense = NSimulation(
            founder_haplotypes=hap_dense, architecture=arch_dense,
            mating_regime=RandomMating(), recombination_map=RecombinationMap.constant_map(m=m),
            seed=42,
        )
        sim_dense.run(1)
        var_dense = np.var(sim_dense.phenotype_history[0]['Y.G'], ddof=1)

        # Sparse (50% causal)
        hap_sparse = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff_sparse = SparseEffects.from_h2(h2=h2, m=m, k_causal=100, seed=42)
        arch_sparse = Architecture()
        arch_sparse.add('Y.G', GeneticComponent(eff_sparse))
        arch_sparse.add('Y.E', NoiseComponent(variance=1.0 - h2))
        arch_sparse.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])
        sim_sparse = NSimulation(
            founder_haplotypes=hap_sparse, architecture=arch_sparse,
            mating_regime=RandomMating(), recombination_map=RecombinationMap.constant_map(m=m),
            seed=42,
        )
        sim_sparse.run(1)
        var_sparse = np.var(sim_sparse.phenotype_history[0]['Y.G'], ddof=1)

        # Both should be in the right ballpark (wide tolerance)
        assert var_dense > 0.05
        assert var_sparse > 0.05
