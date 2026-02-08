"""
Numerical tests for sibling aggregation accuracy.

Tests:
1. Sibling mean within family matches manual calculation
2. Sibling sum equals family-size * mean
3. Sibling count matches actual family sizes
4. Sibling eldest/youngest values are correct
"""
import numpy as np
import pytest

from xftsim.neffect import AdditiveEffects
from xftsim.narch import (
    Architecture, GeneticComponent, NoiseComponent, AggregationComponent,
    SiblingMeanComponent, SiblingSumComponent, SiblingCountComponent,
    SiblingEldestComponent, SiblingYoungestComponent,
)
from xftsim.nmate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.nsim import NSimulation

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


def _build_sim_with_sibling_components(n=200, m=20, seed=42,
                                        sibling_components=None):
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
    eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=seed)
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(eff))
    arch.add('Y.E', NoiseComponent(variance=0.5))
    arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])
    if sibling_components:
        for name, comp in sibling_components.items():
            arch.add(name, comp, inputs=['Y'])
    rmap = RecombinationMap.constant_map(m=m)
    mate = RandomMating(offspring_per_pair=2)
    sim = NSimulation(
        founder_haplotypes=hap, architecture=arch,
        mating_regime=mate, recombination_map=rmap, seed=seed,
    )
    return sim


class TestSiblingMeanAccuracy:
    def test_sibling_mean_matches_manual(self):
        """After gen 1 (offspring), sibling_mean should match manual FID-grouped mean."""
        sim = _build_sim_with_sibling_components(
            sibling_components={'Y.sibmean': SiblingMeanComponent('Y')}
        )
        sim.run(2)
        pheno = sim.phenotype_history[1]
        fids = pheno.samples.fid
        y_vals = pheno['Y']
        sibmean_vals = pheno['Y.sibmean']
        # Manual check: for each family, mean of Y should match Y.sibmean
        unique_fids = np.unique(fids)
        for fid in unique_fids[:10]:
            mask = fids == fid
            expected_mean = np.mean(y_vals[mask])
            actual = sibmean_vals[mask]
            np.testing.assert_allclose(actual, expected_mean, atol=1e-10)


class TestSiblingSumAccuracy:
    def test_sibling_sum_matches_manual(self):
        sim = _build_sim_with_sibling_components(
            sibling_components={'Y.sibsum': SiblingSumComponent('Y')}
        )
        sim.run(2)
        pheno = sim.phenotype_history[1]
        fids = pheno.samples.fid
        y_vals = pheno['Y']
        sibsum_vals = pheno['Y.sibsum']
        unique_fids = np.unique(fids)
        for fid in unique_fids[:10]:
            mask = fids == fid
            expected_sum = np.sum(y_vals[mask])
            actual = sibsum_vals[mask]
            np.testing.assert_allclose(actual, expected_sum, atol=1e-10)


class TestSiblingCountAccuracy:
    def test_sibling_count_matches_family_size(self):
        sim = _build_sim_with_sibling_components(
            sibling_components={'Y.sibcount': SiblingCountComponent('Y')}
        )
        sim.run(2)
        pheno = sim.phenotype_history[1]
        fids = pheno.samples.fid
        sibcount_vals = pheno['Y.sibcount']
        unique_fids = np.unique(fids)
        for fid in unique_fids[:10]:
            mask = fids == fid
            expected_count = float(np.sum(mask))
            actual = sibcount_vals[mask]
            np.testing.assert_allclose(actual, expected_count, atol=1e-10)


class TestSiblingEldestYoungestAccuracy:
    def test_eldest_is_first_in_family(self):
        sim = _build_sim_with_sibling_components(
            sibling_components={'Y.eldest': SiblingEldestComponent('Y')}
        )
        sim.run(2)
        pheno = sim.phenotype_history[1]
        fids = pheno.samples.fid
        y_vals = pheno['Y']
        eldest_vals = pheno['Y.eldest']
        unique_fids = np.unique(fids)
        for fid in unique_fids[:10]:
            mask = fids == fid
            indices = np.where(mask)[0]
            expected_eldest = y_vals[indices[0]]
            actual = eldest_vals[mask]
            np.testing.assert_allclose(actual, expected_eldest, atol=1e-10)

    def test_youngest_is_last_in_family(self):
        sim = _build_sim_with_sibling_components(
            sibling_components={'Y.youngest': SiblingYoungestComponent('Y')}
        )
        sim.run(2)
        pheno = sim.phenotype_history[1]
        fids = pheno.samples.fid
        y_vals = pheno['Y']
        youngest_vals = pheno['Y.youngest']
        unique_fids = np.unique(fids)
        for fid in unique_fids[:10]:
            mask = fids == fid
            indices = np.where(mask)[0]
            expected_youngest = y_vals[indices[-1]]
            actual = youngest_vals[mask]
            np.testing.assert_allclose(actual, expected_youngest, atol=1e-10)
