"""
Numerical test: assortative mating produces expected spousal correlation.

Tests:
1. r > 0 → positive spousal correlation on mating composite
2. r = 0 → near-zero spousal correlation
3. r < 0 → negative spousal correlation
4. Higher |r| → stronger spousal correlation
"""
import numpy as np
import pytest

from xftsim.neffect import AdditiveEffects
from xftsim.narch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.nmate import RandomMating, LinearAssortativeMating
from xftsim.reproduce import RecombinationMap
from xftsim.nsim import NSimulation

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


def _run_and_get_spousal_corr(r, n=500, m=50, seed=42):
    """Run sim, extract spousal correlation on Y from gen 0 mating."""
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
    eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=seed)
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(eff))
    arch.add('Y.E', NoiseComponent(variance=0.5))
    arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])

    if r == 0:
        mate = RandomMating(offspring_per_pair=2)
    else:
        mate = LinearAssortativeMating(component_names=['Y'], r=r, offspring_per_pair=2)

    sim = NSimulation(
        founder_haplotypes=hap, architecture=arch,
        mating_regime=mate,
        recombination_map=RecombinationMap.constant_map(m=m),
        seed=seed,
    )
    sim.run(2)

    # Get mate assignment from gen 0 and phenotypes
    assignment = sim._mate_assignments[0]
    pheno = sim.phenotype_history[0]
    y = pheno['Y']
    mother_y = y[assignment.maternal_idx[::2]]  # one per pair
    father_y = y[assignment.paternal_idx[::2]]  # one per pair
    return np.corrcoef(mother_y, father_y)[0, 1]


class TestAssortativeSpousalCorrelation:
    def test_positive_r_positive_corr(self):
        """r > 0 should produce positive spousal correlation."""
        corr = _run_and_get_spousal_corr(r=0.5)
        assert corr > 0.05, f"Spousal corr = {corr:.3f}, expected > 0 with r=0.5"

    def test_zero_r_near_zero_corr(self):
        """r = 0 should produce near-zero spousal correlation."""
        corr = _run_and_get_spousal_corr(r=0)
        assert abs(corr) < 0.2, f"Spousal corr = {corr:.3f}, expected ≈ 0 with r=0"

    def test_negative_r_negative_corr(self):
        """r < 0 should produce negative spousal correlation."""
        corr = _run_and_get_spousal_corr(r=-0.5)
        assert corr < -0.05, f"Spousal corr = {corr:.3f}, expected < 0 with r=-0.5"

    def test_stronger_r_stronger_corr(self):
        """Higher |r| should produce stronger spousal correlation."""
        corr_low = _run_and_get_spousal_corr(r=0.2)
        corr_high = _run_and_get_spousal_corr(r=0.8)
        assert corr_high > corr_low, \
            f"r=0.8 corr ({corr_high:.3f}) should exceed r=0.2 corr ({corr_low:.3f})"
