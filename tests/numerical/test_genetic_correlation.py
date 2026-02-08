"""
Numerical test: genetic correlation between traits in multivariate simulation.

Tests:
1. MVGenetic with known rg produces observable genetic correlation
2. Independent traits (rg=0) have near-zero genetic correlation
3. Negative rg produces negative genetic correlation
"""
import numpy as np
import pytest

from xftsim.neffect import MultivariateEffects
from xftsim.narch import Architecture, MVGeneticComponent, NoiseComponent, AggregationComponent
from xftsim.nmate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.nsim import NSimulation

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


def _run_bivariate(rg, n=1000, m=100, seed=42):
    """Run a bivariate simulation and return genetic values."""
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
    mv_eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.5], rg=rg, m=m, seed=seed)
    arch = Architecture()
    arch.add(['T1.G', 'T2.G'], MVGeneticComponent(mv_eff))
    arch.add('T1.E', NoiseComponent(variance=0.5))
    arch.add('T2.E', NoiseComponent(variance=0.5))
    arch.add('T1', AggregationComponent('T1.G + T1.E'), inputs=['T1.G', 'T1.E'])
    arch.add('T2', AggregationComponent('T2.G + T2.E'), inputs=['T2.G', 'T2.E'])

    sim = NSimulation(
        founder_haplotypes=hap, architecture=arch,
        mating_regime=RandomMating(offspring_per_pair=2),
        recombination_map=RecombinationMap.constant_map(m=m),
        seed=seed,
    )
    sim.run(1)
    pheno = sim.phenotypes
    return pheno['T1.G'], pheno['T2.G'], pheno['T1'], pheno['T2']


class TestGeneticCorrelation:
    def test_positive_rg_produces_positive_corr(self):
        """rg=0.5 should produce positive genetic correlation."""
        g1, g2, _, _ = _run_bivariate(rg=0.5)
        corr = np.corrcoef(g1, g2)[0, 1]
        assert corr > 0.1, f"Genetic corr = {corr:.3f}, expected > 0.1 with rg=0.5"

    def test_zero_rg_produces_near_zero_corr(self):
        """rg=0 should produce near-zero genetic correlation."""
        g1, g2, _, _ = _run_bivariate(rg=0.0)
        corr = np.corrcoef(g1, g2)[0, 1]
        assert abs(corr) < 0.3, f"Genetic corr = {corr:.3f}, expected ≈ 0 with rg=0"

    def test_negative_rg_produces_negative_corr(self):
        """rg=-0.5 should produce negative genetic correlation."""
        g1, g2, _, _ = _run_bivariate(rg=-0.5)
        corr = np.corrcoef(g1, g2)[0, 1]
        assert corr < -0.1, f"Genetic corr = {corr:.3f}, expected < -0.1 with rg=-0.5"

    def test_phenotypic_corr_attenuated(self):
        """Phenotypic correlation should be weaker than genetic correlation."""
        g1, g2, y1, y2 = _run_bivariate(rg=0.5)
        rg_obs = np.corrcoef(g1, g2)[0, 1]
        ry_obs = np.corrcoef(y1, y2)[0, 1]
        # Phenotypic corr should be attenuated by noise
        assert ry_obs < rg_obs or abs(ry_obs - rg_obs) < 0.2

    def test_monotone_rg(self):
        """Higher rg_design should produce higher rg_realized."""
        results = {}
        for rg in [0.0, 0.3, 0.6, 0.9]:
            g1, g2, _, _ = _run_bivariate(rg=rg, seed=42)
            results[rg] = np.corrcoef(g1, g2)[0, 1]
        assert results[0.0] < results[0.3] < results[0.6] < results[0.9], \
            f"Genetic correlation should be monotone in rg_design: {results}"

    def test_high_rg_produces_strong_corr(self):
        """rg=0.9 should produce high genetic correlation."""
        g1, g2, _, _ = _run_bivariate(rg=0.9)
        corr = np.corrcoef(g1, g2)[0, 1]
        assert corr > 0.4, f"Genetic corr = {corr:.3f}, expected > 0.4 with rg=0.9"
