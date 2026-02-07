"""
Unit tests for EffectSpec classes.
"""
import numpy as np
import pytest
from xftsim.neffect import AdditiveEffects, MultivariateEffects, SparseEffects


class TestAdditiveEffects:
    def test_from_h2_shape(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=100, seed=42)
        assert eff.effects.shape == (100,)
        assert eff.m == 100
        assert eff.m_causal == 100
        assert eff.k == 1

    def test_from_h2_standardized_flag(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=100, standardized=True)
        assert eff.standardized is True
        eff2 = AdditiveEffects.from_h2(h2=0.5, m=100, standardized=False)
        assert eff2.standardized is False

    def test_from_h2_seed_deterministic(self):
        e1 = AdditiveEffects.from_h2(h2=0.5, m=50, seed=42)
        e2 = AdditiveEffects.from_h2(h2=0.5, m=50, seed=42)
        np.testing.assert_array_equal(e1.effects, e2.effects)

    def test_from_h2_variance(self):
        """sum(beta^2) should be approximately h2 for large m."""
        m = 10000
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        var_g = np.sum(eff.effects**2)
        # generous tolerance: SE ≈ h2 * sqrt(2/m) ~ 0.007
        assert abs(var_g - 0.5) < 0.05

    def test_from_array(self):
        arr = np.array([0.1, 0.2, 0.3])
        eff = AdditiveEffects.from_array(arr)
        np.testing.assert_array_equal(eff.effects, arr)
        assert eff.m == 3
        assert eff.m_causal == 3
        assert np.all(eff.variant_mask)

    def test_from_array_roundtrip(self):
        arr = np.random.RandomState(42).randn(20)
        eff = AdditiveEffects.from_array(arr, standardized=False)
        np.testing.assert_array_equal(eff.effects, arr)
        assert eff.standardized is False


class TestMultivariateEffects:
    def test_from_h2_rg_shape(self):
        eff = MultivariateEffects.from_h2_rg(
            h2=[0.5, 0.3], rg=0.2, m=100, seed=42
        )
        assert eff.effects.shape == (100, 2)
        assert eff.m == 100
        assert eff.k == 2

    def test_from_h2_rg_correlation_range(self):
        eff = MultivariateEffects.from_h2_rg(
            h2=[0.5, 0.3], rg=0.5, m=5000, seed=42
        )
        # Empirical genetic correlation
        covg = eff.effects.T @ eff.effects
        varg = np.diag(covg)
        rg_emp = covg[0, 1] / np.sqrt(varg[0] * varg[1])
        # Should be close to 0.5, generous tolerance
        assert abs(rg_emp - 0.5) < 0.15

    def test_from_covg_shape(self):
        covg = np.array([[0.5, 0.1], [0.1, 0.3]])
        eff = MultivariateEffects.from_covg(covg=covg, m=100, seed=42)
        assert eff.effects.shape == (100, 2)

    def test_from_array(self):
        arr = np.random.RandomState(42).randn(50, 3)
        eff = MultivariateEffects.from_array(arr)
        np.testing.assert_array_equal(eff.effects, arr)
        assert eff.k == 3

    def test_standardized_flag(self):
        eff = MultivariateEffects.from_h2_rg(
            h2=[0.5, 0.3], rg=0.2, m=50, standardized=True
        )
        assert eff.standardized is True


class TestSparseEffects:
    def test_from_h2_shape(self):
        eff = SparseEffects.from_h2(h2=0.5, m=100, k_causal=10, seed=42)
        assert eff.effects.shape == (100,)
        assert eff.m == 100
        assert eff.m_causal == 10
        assert eff.k == 1

    def test_correct_nonzero_count(self):
        eff = SparseEffects.from_h2(h2=0.5, m=100, k_causal=10, seed=42)
        assert np.sum(eff.effects != 0) == 10
        assert np.sum(eff.variant_mask) == 10

    def test_variant_mask_matches_effects(self):
        eff = SparseEffects.from_h2(h2=0.5, m=50, k_causal=5, seed=42)
        nonzero = eff.effects != 0
        np.testing.assert_array_equal(eff.variant_mask, nonzero)

    def test_k_causal_exceeds_m(self):
        with pytest.raises(ValueError, match="k_causal"):
            SparseEffects.from_h2(h2=0.5, m=10, k_causal=20)

    def test_seed_deterministic(self):
        e1 = SparseEffects.from_h2(h2=0.5, m=50, k_causal=5, seed=42)
        e2 = SparseEffects.from_h2(h2=0.5, m=50, k_causal=5, seed=42)
        np.testing.assert_array_equal(e1.effects, e2.effects)
        np.testing.assert_array_equal(e1.variant_mask, e2.variant_mask)


class TestSparseEffectsIntegration:
    """Verify SparseEffects works with GeneticComponent and simulations."""

    def test_sparse_with_genetic_component(self):
        """SparseEffects should work as a drop-in for GeneticComponent."""
        from xftsim.narch import GeneticComponent
        from tests.testdata import TestGenomes

        eff = SparseEffects.from_h2(h2=0.5, m=100, k_causal=10, seed=42)
        gc = GeneticComponent(effects=eff)
        hap = TestGenomes.simple(n=50, m=100, seed=42)

        from xftsim.narch import ArchNode
        from xftsim.struct import NPhenotypeArray
        node = ArchNode(outputs=['Y.G'], component=gc, inputs=[])
        pheno = NPhenotypeArray(samples=hap.samples)
        result = gc.compute(node, hap, pheno)
        assert result.shape == (50,)
        assert np.all(np.isfinite(result))

    def test_sparse_in_formula(self):
        """SparseEffects should work via formula-based Architecture."""
        from xftsim.narch import Architecture

        eff = SparseEffects.from_h2(h2=0.5, m=50, k_causal=5, seed=42)
        arch = Architecture.from_formula("""
            Y.G ~ genetic(eff)
            Y.E ~ noise(0.5)
            Y ~ Y.G + Y.E
        """, effects={'eff': eff})
        assert len(arch.nodes) == 3

    def test_sparse_sim_runs(self):
        """Full simulation with SparseEffects should complete."""
        from xftsim.narch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
        from xftsim.nsim import NSimulation
        from xftsim.nmate import RandomMating
        from xftsim.reproduce import RecombinationMap
        from tests.testdata import TestSimulation

        m = 50
        eff = SparseEffects.from_h2(h2=0.5, m=m, k_causal=10, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))

        hap = TestSimulation.founder_haplotypes(n=500, m=m, seed=42)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        mate = RandomMating(offspring_per_pair=2)
        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=mate, recombination_map=rmap, seed=42,
        )
        sim.run(2)
        assert np.all(np.isfinite(sim.phenotype_history[0]['Y']))
        assert np.all(np.isfinite(sim.phenotype_history[1]['Y']))


# ── Additional edge case tests ─────────────────────────────────────────────

class TestEffectSpecEdgeCases:
    """Edge cases and repr tests for all EffectSpec classes."""

    def test_additive_repr(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        r = repr(eff)
        assert 'AdditiveEffects' in r
        assert 'm=10' in r

    def test_multivariate_repr(self):
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.2, m=10, seed=42)
        r = repr(eff)
        assert 'MultivariateEffects' in r
        assert 'k=2' in r

    def test_sparse_repr(self):
        eff = SparseEffects.from_h2(h2=0.5, m=50, k_causal=5, seed=42)
        r = repr(eff)
        assert 'SparseEffects' in r
        assert 'm_causal=5' in r

    def test_additive_single_variant(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=1, seed=42)
        assert eff.m == 1
        assert eff.m_causal == 1
        assert eff.effects.shape == (1,)

    def test_multivariate_three_traits(self):
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3, 0.4], rg=0.1, m=20, seed=42)
        assert eff.k == 3
        assert eff.effects.shape == (20, 3)

    def test_multivariate_from_covg_matches_h2_rg(self):
        """from_covg with diagonal covg ≈ from_h2_rg with rg=0."""
        m = 5000
        h2 = [0.5, 0.3]
        covg = np.diag(h2)
        eff1 = MultivariateEffects.from_covg(covg=covg, m=m, seed=42)
        # Check trait variances are approximately correct
        var_g = np.sum(eff1.effects**2, axis=0)
        np.testing.assert_allclose(var_g, h2, atol=0.1)

    def test_sparse_all_causal(self):
        """SparseEffects with k_causal=m should have all variants causal."""
        eff = SparseEffects.from_h2(h2=0.5, m=10, k_causal=10, seed=42)
        assert eff.m_causal == 10
        assert np.all(eff.variant_mask)

    def test_sparse_single_causal(self):
        """SparseEffects with k_causal=1."""
        eff = SparseEffects.from_h2(h2=0.5, m=100, k_causal=1, seed=42)
        assert eff.m_causal == 1
        assert np.sum(eff.effects != 0) == 1

    def test_additive_different_seeds_differ(self):
        e1 = AdditiveEffects.from_h2(h2=0.5, m=50, seed=42)
        e2 = AdditiveEffects.from_h2(h2=0.5, m=50, seed=99)
        assert not np.array_equal(e1.effects, e2.effects)

    def test_sparse_different_seeds_differ(self):
        e1 = SparseEffects.from_h2(h2=0.5, m=50, k_causal=5, seed=42)
        e2 = SparseEffects.from_h2(h2=0.5, m=50, k_causal=5, seed=99)
        # Different causal variants
        assert not np.array_equal(e1.variant_mask, e2.variant_mask)

    def test_from_array_preserves_zeros(self):
        """from_array should keep exact zeros."""
        arr = np.array([0.0, 1.0, 0.0, 2.0, 0.0])
        eff = AdditiveEffects.from_array(arr)
        assert eff.effects[0] == 0.0
        assert eff.effects[2] == 0.0
        # variant_mask is all True for AdditiveEffects (all "causal")
        assert np.all(eff.variant_mask)

    def test_multivariate_from_array_preserves_shape(self):
        arr = np.random.RandomState(42).randn(30, 4)
        eff = MultivariateEffects.from_array(arr)
        assert eff.m == 30
        assert eff.k == 4
        np.testing.assert_array_equal(eff.effects, arr)
