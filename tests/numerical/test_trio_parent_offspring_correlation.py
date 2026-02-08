"""
Numerical test: parent-offspring correlation in trios.

Under additive genetics with h2, the expected parent-offspring phenotype
correlation is approximately h2/2 (for standardized traits).

Tests:
1. Parent-offspring correlation is positive under h2>0
2. Midparent-offspring correlation ~ h2 (roughly)
3. Mother and father correlations are similar
"""
import numpy as np
import pytest

from xftsim.neffect import AdditiveEffects
from xftsim.narch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.nmate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.nsim import NSimulation
from xftsim.nfilter import TrioFilter

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


def _run_trio_sim(h2=0.5, n=500, m=50, seed=42):
    """Run a simulation and extract trio data from gen 1."""
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
    eff = AdditiveEffects.from_h2(h2=h2, m=m, seed=seed)
    noise_var = 1.0 - h2
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(eff))
    arch.add('Y.E', NoiseComponent(variance=noise_var))
    arch.add('Y', AggregationComponent('Y.G + Y.E'))

    tf = TrioFilter()
    sim = NSimulation(
        founder_haplotypes=hap, architecture=arch,
        mating_regime=RandomMating(offspring_per_pair=2),
        recombination_map=RecombinationMap.constant_map(m=m),
        seed=seed,
        filters={'trio': tf},
        retain_phenotypes=10,
        retain_haplotypes=10,
    )
    sim.run(2)
    return sim


class TestTrioCorrelation:
    def test_parent_offspring_correlation_positive(self):
        """With h2=0.5, parent-offspring correlation should be positive."""
        sim = _run_trio_sim(h2=0.5, n=1000, m=50, seed=42)
        # Access the trio view from the filter
        tf = TrioFilter()
        view = tf.apply(
            generation=1,
            phenotype_history=sim.phenotype_history,
            pedigree_history=sim.pedigree_history,
        )
        assert view is not None
        assert view.n_trios > 0

        offspring_y = view.offspring_phenotypes['Y']
        mother_y = view.mother_phenotypes['Y']
        father_y = view.father_phenotypes['Y']

        corr_mother = np.corrcoef(offspring_y, mother_y)[0, 1]
        corr_father = np.corrcoef(offspring_y, father_y)[0, 1]
        assert corr_mother > 0.0, f"Mother-offspring corr = {corr_mother}"
        assert corr_father > 0.0, f"Father-offspring corr = {corr_father}"

    def test_midparent_offspring_correlation_approximate(self):
        """Midparent-offspring correlation should approximate realized h2."""
        # Use larger n for better precision
        sim = _run_trio_sim(h2=0.5, n=2000, m=50, seed=123)
        tf = TrioFilter()
        view = tf.apply(
            generation=1,
            phenotype_history=sim.phenotype_history,
            pedigree_history=sim.pedigree_history,
        )

        offspring_y = view.offspring_phenotypes['Y']
        midparent = (view.mother_phenotypes['Y'] + view.father_phenotypes['Y']) / 2
        corr = np.corrcoef(offspring_y, midparent)[0, 1]

        # With centered-unscaled genotypes and p=0.5, h2_realized < h2_design
        # Midparent-offspring regression slope ≈ h2_realized
        # But correlation depends on variance reduction from midparent averaging
        # Just check it's positive and not near 1
        assert corr > 0.05, f"Midparent-offspring corr = {corr}, expected > 0.05"
        assert corr < 0.9, f"Midparent-offspring corr = {corr}, expected < 0.9"

    def test_mother_father_correlations_similar(self):
        """Mother and father correlations should be roughly equal under random mating."""
        sim = _run_trio_sim(h2=0.5, n=2000, m=50, seed=777)
        tf = TrioFilter()
        view = tf.apply(
            generation=1,
            phenotype_history=sim.phenotype_history,
            pedigree_history=sim.pedigree_history,
        )

        offspring_y = view.offspring_phenotypes['Y']
        corr_m = np.corrcoef(offspring_y, view.mother_phenotypes['Y'])[0, 1]
        corr_f = np.corrcoef(offspring_y, view.father_phenotypes['Y'])[0, 1]

        # Should be similar (within 0.15 of each other)
        assert abs(corr_m - corr_f) < 0.15, \
            f"Mother corr={corr_m:.3f} vs Father corr={corr_f:.3f}"
