"""
Numerical test: heritability round-trip specification test.

The fundamental invariant: if you specify h2_design, the simulation should
produce phenotypes with realized h2 ≈ h2_design (within sampling noise).

This is the test that validates the contract between AdditiveEffects.from_h2()
and standardized_matvec(). Effects are drawn assuming standardized genotypes
(Var(G_j) = 1 per SNP), and standardized_matvec must actually standardize
(center AND scale by sqrt(2pq)) for the math to work out.

Tests:
1. Round-trip h2: design h2 ≈ realized h2
2. Multi-seed stability
3. Genetic-environmental covariance near zero
4. Variance additivity: Var(Y) ≈ Var(G) + Var(E)
5. Monotonicity: higher h2_design → higher h2_realized
6. HE estimator also recovers h2 at generation 0
"""
import numpy as np
import pytest

from xftsim.effect import AdditiveEffects
from xftsim.arch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.mate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.sim import NSimulation
from xftsim.stats import HasemanElstonEstimator, SampleStatistics

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


class TestH2RoundTrip:
    """The fundamental specification test: input h2 ≈ output h2."""

    def test_h2_050_round_trip(self):
        """h2_design=0.5 should produce h2_realized ≈ 0.5."""
        h2_est, _, _, _, _ = _estimate_h2(h2_design=0.5, noise_var=0.5, seed=42)
        assert abs(h2_est - 0.5) < 0.10, \
            f"h2 design=0.5, realized={h2_est:.3f} — should be ~0.5"

    def test_h2_080_round_trip(self):
        """h2_design=0.8 should produce h2_realized ≈ 0.8."""
        h2_est, _, _, _, _ = _estimate_h2(h2_design=0.8, noise_var=0.2, seed=42)
        assert abs(h2_est - 0.8) < 0.10, \
            f"h2 design=0.8, realized={h2_est:.3f} — should be ~0.8"

    def test_h2_020_round_trip(self):
        """h2_design=0.2 should produce h2_realized ≈ 0.2."""
        h2_est, _, _, _, _ = _estimate_h2(h2_design=0.2, noise_var=0.8, seed=42)
        assert abs(h2_est - 0.2) < 0.10, \
            f"h2 design=0.2, realized={h2_est:.3f} — should be ~0.2"

    def test_h2_multi_seed_stable(self):
        """Mean h2 across multiple seeds should be close to design h2."""
        h2_d = 0.5
        noise_v = 0.5
        estimates = []
        for seed in [42, 123, 456, 789, 1234]:
            h2_est, _, _, _, _ = _estimate_h2(h2_design=h2_d, noise_var=noise_v, seed=seed)
            estimates.append(h2_est)
        mean_h2 = np.mean(estimates)
        assert abs(mean_h2 - h2_d) < 0.06, \
            f"Mean h2={mean_h2:.3f}, design={h2_d} — should match"

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


class TestHEAtGeneration0:
    """HE estimator should work at generation 0 (founders) and recover h2."""

    def test_he_recovers_h2_gen0(self):
        """GRM-based HE should estimate h2 ≈ design at gen 0."""
        n, m, h2_design = 2000, 200, 0.5
        noise_var = 1.0 - h2_design
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=h2_design, m=m, seed=1042)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=noise_var))
        arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])

        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=RandomMating(offspring_per_pair=2),
            recombination_map=RecombinationMap.constant_map(m=m),
            statistics=[HasemanElstonEstimator(phenotype_keys=['Y'])],
            seed=42,
        )
        sim.run(1)

        # Should have results for gen 0
        assert len(sim.results) >= 1
        he_result = sim.results[0].statistics.get('HasemanElstonEstimator')
        assert he_result is not None, "HE should produce results at gen 0"
        assert 'Y' in he_result, f"HE should have 'Y' key, got {list(he_result.keys())}"
        h2_he = he_result['Y']['h2']
        assert abs(h2_he - h2_design) < 0.15, \
            f"HE h2={h2_he:.3f}, design={h2_design} — should be close"
