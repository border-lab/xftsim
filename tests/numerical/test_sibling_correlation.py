"""
Numerical test: sibling phenotype correlation.

Under additive genetics, the expected sibling correlation is approximately h2/2
(since siblings share ~50% of their genome on average).

Tests:
1. Sibling correlation is positive under h2>0
2. Sibling correlation < parent-offspring correlation with same h2 (approximately equal)
3. h2=0 produces near-zero sibling correlation
"""
import numpy as np
import pytest

from xftsim.effect import AdditiveEffects
from xftsim.arch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.mate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.sim import NSimulation
from xftsim.filters import SibPairFilter

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


def _run_sib_sim(h2=0.5, n=1000, m=50, opp=2, seed=42):
    """Run a simulation and return the sibling view from gen 1."""
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
    eff = AdditiveEffects.from_h2(h2=h2, m=m, seed=seed)
    noise_var = 1.0 - h2
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(eff))
    arch.add('Y.E', NoiseComponent(variance=noise_var))
    arch.add('Y', AggregationComponent('Y.G + Y.E'))

    sim = NSimulation(
        founder_haplotypes=hap, architecture=arch,
        mating_regime=RandomMating(offspring_per_pair=opp),
        recombination_map=RecombinationMap.constant_map(m=m),
        seed=seed,
        retain_phenotypes=10, retain_haplotypes=10,
    )
    sim.run(2)

    sf = SibPairFilter()
    view = sf.apply(
        generation=1,
        phenotype_history=sim.phenotype_history,
        pedigree_history=sim.pedigree_history,
    )
    return view


class TestSiblingCorrelation:
    def test_sib_correlation_positive(self):
        """With h2=0.5, sibling correlation should be positive."""
        view = _run_sib_sim(h2=0.5, n=2000, m=50, seed=42)
        assert view is not None
        assert view.n_pairs > 0

        sib1 = view.sib1_phenotypes['Y']
        sib2 = view.sib2_phenotypes['Y']
        corr = np.corrcoef(sib1, sib2)[0, 1]
        assert corr > 0.0, f"Sibling correlation = {corr}, expected > 0"

    def test_sib_corr_near_zero_when_h2_zero(self):
        """With h2≈0, sibling correlation should be near zero."""
        view = _run_sib_sim(h2=0.01, n=2000, m=50, seed=42)
        assert view is not None

        sib1 = view.sib1_phenotypes['Y']
        sib2 = view.sib2_phenotypes['Y']
        corr = np.corrcoef(sib1, sib2)[0, 1]
        assert abs(corr) < 0.1, \
            f"Sibling correlation = {corr}, expected near 0"

    def test_sib_corr_increases_with_h2(self):
        """Higher h2 should produce higher sibling correlation."""
        corrs = []
        for h2 in [0.1, 0.5, 0.9]:
            view = _run_sib_sim(h2=h2, n=2000, m=50, seed=42)
            sib1 = view.sib1_phenotypes['Y']
            sib2 = view.sib2_phenotypes['Y']
            corrs.append(np.corrcoef(sib1, sib2)[0, 1])

        # Monotonically increasing
        assert corrs[0] < corrs[1], \
            f"h2=0.1 corr={corrs[0]} not < h2=0.5 corr={corrs[1]}"
        assert corrs[1] < corrs[2], \
            f"h2=0.5 corr={corrs[1]} not < h2=0.9 corr={corrs[2]}"
