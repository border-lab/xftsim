"""
Integration tests combining vertical transmission with grouping.

Tests:
1. VT with FID-grouped noise → shared family noise persists
2. Mother VT + FID noise → family structure maintained
3. Sibling mean depends on grouped noise
"""
import numpy as np
import pytest

from xftsim.neffect import AdditiveEffects
from xftsim.narch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.narch import MotherComponent, SiblingMeanComponent
from xftsim.nmate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.nsim import NSimulation

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestVTWithGroupedNoise:
    def test_vt_with_fid_noise(self):
        """Vertical transmission + FID-grouped noise: sim should complete."""
        n, m = 200, 30
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.4, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.shared', NoiseComponent(variance=0.3), grouping='FID')
        arch.add('Y.E', NoiseComponent(variance=0.3))
        arch.add('Y', AggregationComponent('Y.G + Y.shared + Y.E'),
                 inputs=['Y.G', 'Y.shared', 'Y.E'])
        rmap = RecombinationMap.constant_map(m=m)
        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=RandomMating(offspring_per_pair=2),
            recombination_map=rmap, seed=42,
        )
        sim.run(3)
        assert np.all(np.isfinite(sim.phenotypes['Y']))

    def test_vt_with_fid_noise_siblings_share(self):
        """After meiosis, siblings should share the FID-grouped noise."""
        n, m = 200, 30
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.4, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.shared', NoiseComponent(variance=0.3), grouping='FID')
        arch.add('Y.E', NoiseComponent(variance=0.1))
        arch.add('Y', AggregationComponent('Y.G + Y.shared + Y.E'),
                 inputs=['Y.G', 'Y.shared', 'Y.E'])
        rmap = RecombinationMap.constant_map(m=m)
        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=RandomMating(offspring_per_pair=2),
            recombination_map=rmap, seed=42,
        )
        sim.run(2)
        # Gen 1: offspring should have siblings with same Y.shared
        pheno = sim.phenotype_history[1]
        fids = pheno.samples.fid
        unique_fids = np.unique(fids)
        for fid in unique_fids[:5]:
            mask = fids == fid
            shared = pheno['Y.shared'][mask]
            if len(shared) > 1:
                assert np.all(shared == shared[0])


class TestMotherVTWithGroupedNoise:
    def test_mother_vt_sim_completes(self):
        """Mother VT + FID noise: sim completes without errors."""
        n, m = 200, 30
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.3, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.VT', MotherComponent('Y', founder_component=NoiseComponent(0.3)))
        arch.add('Y.shared', NoiseComponent(variance=0.2), grouping='FID')
        arch.add('Y.E', NoiseComponent(variance=0.2))
        arch.add('Y', AggregationComponent('Y.G + 0.2 * Y.VT + Y.shared + Y.E'),
                 inputs=['Y.G', 'Y.VT', 'Y.shared', 'Y.E'])
        rmap = RecombinationMap.constant_map(m=m)
        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=RandomMating(offspring_per_pair=2),
            recombination_map=rmap, seed=42,
        )
        sim.run(4)
        assert np.all(np.isfinite(sim.phenotypes['Y']))
        assert sim.phenotypes['Y'].var() < 100


class TestSiblingMeanWithGroupedNoise:
    def test_sibling_mean_after_grouped_noise(self):
        """Sibling mean computed from phenotype that includes grouped noise."""
        n, m = 200, 30
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.4, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.shared', NoiseComponent(variance=0.3), grouping='FID')
        arch.add('Y.E', NoiseComponent(variance=0.3))
        arch.add('Y', AggregationComponent('Y.G + Y.shared + Y.E'),
                 inputs=['Y.G', 'Y.shared', 'Y.E'])
        arch.add('Y.sibmean', SiblingMeanComponent('Y'), inputs=['Y'])
        rmap = RecombinationMap.constant_map(m=m)
        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=RandomMating(offspring_per_pair=2),
            recombination_map=rmap, seed=42,
        )
        sim.run(2)
        assert 'Y.sibmean' in sim.phenotypes.keys
        assert np.all(np.isfinite(sim.phenotypes['Y.sibmean']))
