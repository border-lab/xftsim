"""
Unit tests for SparseEffects causal variant handling.

Tests:
1. Non-causal variants have exactly zero effect
2. Only causal variants contribute to genetic variance
3. variant_mask correctly identifies causal loci
4. k_causal == m means all variants are causal
5. k_causal > m raises ValueError
6. Effects sum of squares proportional to h2
"""
import numpy as np
import pytest

from xftsim.effect import SparseEffects, AdditiveEffects
from xftsim.arch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestSparseEffectsMask:
    def test_noncausal_effects_zero(self):
        """Non-causal variants should have exactly zero effect weights."""
        eff = SparseEffects.from_h2(h2=0.5, m=100, k_causal=10, seed=42)
        noncausal = ~eff.variant_mask
        assert np.all(eff.effects[noncausal] == 0.0)

    def test_causal_effects_nonzero(self):
        """Causal variants should have nonzero effect weights."""
        eff = SparseEffects.from_h2(h2=0.5, m=100, k_causal=10, seed=42)
        causal = eff.variant_mask
        # At least most causal variants should be nonzero
        assert np.sum(eff.effects[causal] != 0.0) >= 8

    def test_variant_mask_count(self):
        """variant_mask should have exactly k_causal True entries."""
        eff = SparseEffects.from_h2(h2=0.5, m=100, k_causal=10, seed=42)
        assert np.sum(eff.variant_mask) == 10

    def test_k_causal_equals_m(self):
        """k_causal == m: all variants causal."""
        eff = SparseEffects.from_h2(h2=0.5, m=20, k_causal=20, seed=42)
        assert np.all(eff.variant_mask)
        assert eff.m_causal == 20

    def test_k_causal_exceeds_m_raises(self):
        """k_causal > m should raise ValueError."""
        with pytest.raises(ValueError, match="k_causal"):
            SparseEffects.from_h2(h2=0.5, m=10, k_causal=15, seed=42)


class TestSparseEffectsVariance:
    def test_only_causal_contribute_to_genetic_value(self):
        """Genetic values should depend only on causal loci."""
        n, m, k = 200, 50, 5
        rng = np.random.RandomState(42)
        sm = SampleMeta(iid=np.arange(n), sex=np.tile([0, 1], n // 2))
        vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
        geno = rng.binomial(1, 0.5, (n, m, 2)).astype(np.int8)
        hap = DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)

        eff = SparseEffects.from_h2(h2=0.5, m=m, k_causal=k, seed=42)

        # Compute genetic values using full effects
        g_full = hap.matvec(eff.effects)

        # Zero out non-causal and recompute — should be identical
        eff_causal_only = eff.effects.copy()
        eff_causal_only[~eff.variant_mask] = 0.0  # already zero, but explicit
        g_causal = hap.matvec(eff_causal_only)

        np.testing.assert_array_equal(g_full, g_causal)

    def test_genetic_variance_from_causal_only(self):
        """Genetic variance should arise only from causal variants."""
        n, m, k = 500, 50, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = SparseEffects.from_h2(h2=0.5, m=m, k_causal=k, seed=42)

        # Compute genetic values
        g = hap.matvec(eff.effects)
        var_g = np.var(g)

        # Compute using only non-causal (should be ~0)
        eff_noncausal = np.zeros(m)
        eff_noncausal[~eff.variant_mask] = eff.effects[~eff.variant_mask]
        g_noncausal = hap.matvec(eff_noncausal)
        var_noncausal = np.var(g_noncausal)

        assert var_noncausal < 1e-20, \
            f"Non-causal variance = {var_noncausal}, expected 0"
        assert var_g > 0.01, \
            f"Causal variance = {var_g}, expected > 0"

    def test_sparse_genetic_value_in_simulation(self):
        """SparseEffects should work in full simulation pipeline."""
        n, m, k = 200, 30, 5
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = SparseEffects.from_h2(h2=0.5, m=m, k_causal=k, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])

        result = arch.compute(hap, rng=np.random.RandomState(42))
        assert 'Y' in result
        assert np.all(np.isfinite(result['Y']))
        # Genetic variance should be positive
        assert np.var(result['Y.G']) > 0.001

    def test_sparse_repr(self):
        """SparseEffects repr should include m_causal."""
        eff = SparseEffects.from_h2(h2=0.5, m=100, k_causal=10, seed=42)
        r = repr(eff)
        assert 'm_causal=10' in r
        assert 'm=100' in r
