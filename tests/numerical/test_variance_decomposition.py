"""
Numerical test: variance decomposition properties.

Tests:
1. Var(Y) ≈ Var(G) + Var(E) when G and E are independent
2. h2 ≈ Var(G) / Var(Y) matches target
3. Noise variance matches specification
4. Two independent genetic components: variance is additive
5. Genetic + noise variances sum to total (no cross-terms)
"""
import numpy as np
import pytest

from xftsim.effect import AdditiveEffects
from xftsim.arch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestVarianceDecomposition:
    def test_var_y_equals_var_g_plus_var_e(self):
        """Var(Y) ≈ Var(G) + Var(E) at founder generation."""
        n, m = 2000, 100
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])
        result = arch.compute(hap, rng=np.random.RandomState(42))

        var_g = np.var(result['Y.G'])
        var_e = np.var(result['Y.E'])
        var_y = np.var(result['Y'])

        # G and E independent → Cov(G,E) ≈ 0
        cov_ge = np.cov(result['Y.G'], result['Y.E'])[0, 1]
        assert abs(cov_ge) < 0.1, f"Cov(G,E) = {cov_ge:.3f}, expected ≈ 0"
        assert abs(var_y - (var_g + var_e)) < 0.2, \
            f"Var(Y)={var_y:.3f}, Var(G)+Var(E)={var_g + var_e:.3f}"

    def test_h2_matches_target(self):
        """Realized h2 = Var(G)/Var(Y) should be close to target."""
        n, m = 2000, 100
        h2_target = 0.5
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=h2_target, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=1.0 - h2_target))
        arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])
        result = arch.compute(hap, rng=np.random.RandomState(42))

        var_g = np.var(result['Y.G'])
        var_y = np.var(result['Y'])
        h2_realized = var_g / var_y

        assert 0.2 < h2_realized < 0.8, \
            f"h2_realized={h2_realized:.3f}, expected ≈ {h2_target}"

    def test_noise_variance_matches_spec(self):
        """Noise variance should match the specified variance parameter."""
        n = 5000
        hap = TestSimulation.founder_haplotypes(n=n, m=10, seed=42)
        arch = Architecture()
        arch.add('Y.E', NoiseComponent(variance=2.0))
        result = arch.compute(hap, rng=np.random.RandomState(42))
        var_e = np.var(result['Y.E'])
        assert 1.5 < var_e < 2.5, \
            f"Var(E)={var_e:.3f}, expected ≈ 2.0"

    def test_two_independent_genetic_additive_variance(self):
        """With two independent effects, total variance is roughly additive."""
        n, m = 2000, 100
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff1 = AdditiveEffects.from_h2(h2=0.3, m=m, seed=42)
        eff2 = AdditiveEffects.from_h2(h2=0.2, m=m, seed=99)

        arch = Architecture()
        arch.add('Y.G1', GeneticComponent(eff1))
        arch.add('Y.G2', GeneticComponent(eff2))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G1 + Y.G2 + Y.E'),
                 inputs=['Y.G1', 'Y.G2', 'Y.E'])
        result = arch.compute(hap, rng=np.random.RandomState(42))

        var_g1 = np.var(result['Y.G1'])
        var_g2 = np.var(result['Y.G2'])
        var_e = np.var(result['Y.E'])
        var_y = np.var(result['Y'])

        # Total variance should be positive and reasonable
        assert var_y > 0.1
        # Variance should be roughly sum of components (±covariance)
        component_sum = var_g1 + var_g2 + var_e
        ratio = var_y / component_sum
        assert 0.5 < ratio < 2.0, \
            f"Var(Y)/sum(components) = {ratio:.3f}"

    def test_genetic_noise_sum_to_total(self):
        """(Var(G) + Var(E)) / Var(Y) ≈ 1 for independent components."""
        n = 3000
        hap = TestSimulation.founder_haplotypes(n=n, m=50, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.6, m=50, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.4))
        arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])
        result = arch.compute(hap, rng=np.random.RandomState(42))

        var_g = np.var(result['Y.G'])
        var_e = np.var(result['Y.E'])
        var_y = np.var(result['Y'])

        ratio = (var_g + var_e) / var_y
        assert 0.85 < ratio < 1.15, \
            f"(Var(G)+Var(E))/Var(Y) = {ratio:.3f}, expected ≈ 1.0"
