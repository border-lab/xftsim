"""
Numerical test: heritability estimation with proper theoretical expectations.

Effects are calibrated for standardized genotypes (E[sum(beta^2)] = h2_design).
But standardized_matvec uses centered-but-unscaled genotypes, so:
  Var(G) = h2_design * mean(2*p*(1-p))
For p=0.5 founders: mean(2pq) ≈ 0.5, so h2_realized ≈ 0.5*h2/(0.5*h2 + noise_var).

Tests:
1. h2 ratio matches theoretical prediction
2. Multi-seed mean h2 stable
3. Genetic-environmental covariance near zero
4. Variance additivity: Var(Y) ≈ Var(G) + Var(E)
5. Higher h2_design produces higher h2_realized
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


def _estimate_h2(n=2000, m=200, h2_design=0.5, noise_var=None, seed=42):
    """Run sim and estimate h2 from variance components."""
    if noise_var is None:
        noise_var = 1.0 - h2_design
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
    eff = AdditiveEffects.from_h2(h2=h2_design, m=m, seed=seed + 1000)
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(eff))
    arch.add('Y.E', NoiseComponent(variance=noise_var))
    arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])

    sim = NSimulation(
        founder_haplotypes=hap, architecture=arch,
        mating_regime=RandomMating(offspring_per_pair=2),
        recombination_map=RecombinationMap.constant_map(m=m),
        seed=seed,
    )
    sim.run(1)

    pheno = sim.phenotype_history[0]
    var_g = np.var(pheno['Y.G'])
    var_e = np.var(pheno['Y.E'])
    var_y = np.var(pheno['Y'])
    cov_ge = np.cov(pheno['Y.G'], pheno['Y.E'])[0, 1]
    h2_est = var_g / var_y

    return h2_est, var_g, var_e, var_y, cov_ge


def _theoretical_h2(h2_design, noise_var, mean_2pq=0.5):
    """Theoretical realized h2 for centered-but-unscaled genotypes.

    Var(G) ≈ h2_design * mean_2pq
    h2_realized = Var(G) / (Var(G) + noise_var)
    """
    var_g = h2_design * mean_2pq
    return var_g / (var_g + noise_var)


class TestH2EstimationTight:
    def test_h2_050_matches_theory(self):
        """h2 estimate should match theoretical prediction for centered genotypes."""
        h2_d = 0.5
        noise_v = 0.5
        h2_est, _, _, _, _ = _estimate_h2(h2_design=h2_d, noise_var=noise_v, seed=42)
        h2_theory = _theoretical_h2(h2_d, noise_v)  # ≈ 0.333
        assert abs(h2_est - h2_theory) < 0.10, \
            f"h2 estimated={h2_est:.3f}, theory={h2_theory:.3f}"

    def test_h2_080_matches_theory(self):
        """h2=0.8 design should produce ~0.667 realized."""
        h2_d = 0.8
        noise_v = 0.2
        h2_est, _, _, _, _ = _estimate_h2(h2_design=h2_d, noise_var=noise_v, seed=42)
        h2_theory = _theoretical_h2(h2_d, noise_v)  # ≈ 0.667
        assert abs(h2_est - h2_theory) < 0.10, \
            f"h2 estimated={h2_est:.3f}, theory={h2_theory:.3f}"

    def test_h2_020_matches_theory(self):
        """h2=0.2 design should produce ~0.111 realized."""
        h2_d = 0.2
        noise_v = 0.8
        h2_est, _, _, _, _ = _estimate_h2(h2_design=h2_d, noise_var=noise_v, seed=42)
        h2_theory = _theoretical_h2(h2_d, noise_v)  # ≈ 0.111
        assert abs(h2_est - h2_theory) < 0.10, \
            f"h2 estimated={h2_est:.3f}, theory={h2_theory:.3f}"

    def test_h2_multi_seed_stable(self):
        """Mean h2 across multiple seeds should be close to theory."""
        h2_d = 0.5
        noise_v = 0.5
        estimates = []
        for seed in [42, 123, 456, 789, 1234]:
            h2_est, _, _, _, _ = _estimate_h2(h2_design=h2_d, noise_var=noise_v, seed=seed)
            estimates.append(h2_est)
        mean_h2 = np.mean(estimates)
        h2_theory = _theoretical_h2(h2_d, noise_v)
        assert abs(mean_h2 - h2_theory) < 0.06, \
            f"Mean h2={mean_h2:.3f}, theory={h2_theory:.3f}"

    def test_higher_design_h2_gives_higher_realized(self):
        """Larger h2_design should produce larger realized h2."""
        h2_low, _, _, _, _ = _estimate_h2(h2_design=0.2, noise_var=0.8, seed=42)
        h2_mid, _, _, _, _ = _estimate_h2(h2_design=0.5, noise_var=0.5, seed=42)
        h2_high, _, _, _, _ = _estimate_h2(h2_design=0.8, noise_var=0.2, seed=42)
        assert h2_low < h2_mid < h2_high, \
            f"h2 should be monotone: {h2_low:.3f} < {h2_mid:.3f} < {h2_high:.3f}"

    def test_genetic_environmental_covariance_zero(self):
        """Cov(G, E) should be near zero (independent components)."""
        _, _, _, _, cov_ge = _estimate_h2(h2_design=0.5, noise_var=0.5, seed=42)
        assert abs(cov_ge) < 0.10, \
            f"Cov(G, E) = {cov_ge:.4f}, expected near 0"

    def test_variance_additivity(self):
        """Var(Y) should approximately equal Var(G) + Var(E) + 2*Cov(G,E)."""
        _, var_g, var_e, var_y, cov_ge = _estimate_h2(h2_design=0.5, noise_var=0.5, seed=42)
        expected = var_g + var_e + 2 * cov_ge
        ratio = var_y / expected
        assert 0.95 < ratio < 1.05, \
            f"Var(Y)={var_y:.3f}, Var(G)+Var(E)+2Cov={expected:.3f}, ratio={ratio:.3f}"
