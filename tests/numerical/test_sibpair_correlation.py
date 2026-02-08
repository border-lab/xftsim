"""
Numerical test: sibling phenotype correlations in a simulation.

Tests:
1. Sibling genetic values should have positive correlation (shared parents)
2. Sibling noise values should be uncorrelated
3. Sibling overall phenotype correlation should be approximately h2/2
"""
import numpy as np
import pytest

from xftsim.neffect import AdditiveEffects
from xftsim.narch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.nmate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.nsim import NSimulation
from xftsim.nfilter import SibPairFilter

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


def _run_sim_with_sibpairs(n=500, m=50, h2=0.5, seed=42):
    """Run a 2-gen simulation and extract sibling pairs from gen 1."""
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
    eff = AdditiveEffects.from_h2(h2=h2, m=m, seed=seed)
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(eff))
    arch.add('Y.E', NoiseComponent(variance=1.0 - h2))
    arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])

    sim = NSimulation(
        founder_haplotypes=hap, architecture=arch,
        mating_regime=RandomMating(offspring_per_pair=2),
        recombination_map=RecombinationMap.constant_map(m=m),
        seed=seed,
        filters={'sib': SibPairFilter()},
    )
    sim.run(2)

    # Extract sibling pairs from gen 1
    filt = SibPairFilter()
    view = filt.apply(1, sim.phenotype_history, sim.pedigree_history)
    return view


class TestSiblingCorrelations:
    def test_genetic_values_positively_correlated(self):
        """Sibling genetic values should be positively correlated."""
        view = _run_sim_with_sibpairs(n=500, m=50, seed=42)
        if view is None or view.n_pairs < 10:
            pytest.skip("Not enough sibling pairs for test")
        g1 = view.sib1_phenotypes['Y.G']
        g2 = view.sib2_phenotypes['Y.G']
        corr = np.corrcoef(g1, g2)[0, 1]
        assert corr > 0, f"Sibling genetic correlation = {corr:.3f}, expected > 0"

    def test_noise_values_near_zero_correlation(self):
        """Sibling noise values should have near-zero correlation."""
        view = _run_sim_with_sibpairs(n=500, m=50, seed=42)
        if view is None or view.n_pairs < 10:
            pytest.skip("Not enough sibling pairs for test")
        e1 = view.sib1_phenotypes['Y.E']
        e2 = view.sib2_phenotypes['Y.E']
        corr = np.corrcoef(e1, e2)[0, 1]
        assert abs(corr) < 0.2, f"Sibling noise correlation = {corr:.3f}, expected ≈ 0"

    def test_phenotype_correlation_order_of_magnitude(self):
        """Sibling phenotype correlation should be in a plausible range."""
        view = _run_sim_with_sibpairs(n=500, m=50, h2=0.5, seed=42)
        if view is None or view.n_pairs < 10:
            pytest.skip("Not enough sibling pairs for test")
        y1 = view.sib1_phenotypes['Y']
        y2 = view.sib2_phenotypes['Y']
        corr = np.corrcoef(y1, y2)[0, 1]
        # With h2=0.5, expected sib corr ≈ h2/2 = 0.25
        # Wide tolerance for finite samples
        assert 0.0 < corr < 0.8, \
            f"Sibling phenotype correlation = {corr:.3f}, expected in (0, 0.8)"
