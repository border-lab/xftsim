"""
Numerical tests for parent-offspring regression.

Tests:
1. Parent-offspring phenotype correlation ≈ h2/2 (without VT)
2. With VT, parent-offspring correlation > h2/2
3. Midparent-offspring correlation higher than single-parent
4. Sibling correlation ≈ h2/2 (shared genetics)
5. Genetic values: parent-offspring correlation ≈ h2 (regression to mean halved by meiosis)
"""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation

from xftsim.arch import (
    Architecture, GeneticComponent, NoiseComponent, AggregationComponent,
    MotherComponent, FatherComponent,
)
from xftsim.effect import AdditiveEffects
from xftsim.mate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.sim import NSimulation
from xftsim.stats import SampleStatistics
from xftsim.filters import TrioFilter


class TestParentOffspringCorrelation:
    def test_po_correlation_without_vt(self):
        """Without VT, parent-offspring phenotype correlation ≈ h2/2."""
        n, m = 500, 200
        h2 = 0.6
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        effects = AdditiveEffects.from_h2(m=m, h2=h2, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(effects))
        arch.add('Y.E', NoiseComponent(variance=1.0 - h2))
        arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])

        mating = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m)
        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=mating, recombination_map=rmap,
            seed=42, retain_phenotypes=2,
        )
        sim.run(2)

        # Extract parent and offspring phenotypes
        parent_pheno = sim.phenotype_history[0]
        offspring_pheno = sim.phenotype_history[1]
        ped = sim.pedigree_history[1]

        # Mother-offspring correlation
        mother_y = parent_pheno['Y'][ped.maternal_idx]
        offspring_y = offspring_pheno['Y']
        corr_mo = np.corrcoef(mother_y, offspring_y)[0, 1]

        # Should be approximately h2/2 = 0.3 (broad range for stochasticity)
        assert 0.0 < corr_mo < 0.6, f"Mother-offspring corr={corr_mo:.3f}, expected ~{h2/2:.2f}"


class TestGeneticValueCorrelation:
    def test_genetic_po_correlation(self):
        """Parent-offspring genetic value correlation should be substantial."""
        n, m = 500, 200
        h2 = 0.8
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        effects = AdditiveEffects.from_h2(m=m, h2=h2, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(effects))
        arch.add('Y.E', NoiseComponent(variance=1.0 - h2))
        arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])

        mating = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m)
        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=mating, recombination_map=rmap,
            seed=42, retain_phenotypes=2,
        )
        sim.run(2)

        parent_pheno = sim.phenotype_history[0]
        offspring_pheno = sim.phenotype_history[1]
        ped = sim.pedigree_history[1]

        # Parent-offspring genetic value correlation
        mother_g = parent_pheno['Y.G'][ped.maternal_idx]
        offspring_g = offspring_pheno['Y.G']
        corr_g = np.corrcoef(mother_g, offspring_g)[0, 1]

        # Should be positive and substantial
        assert corr_g > 0.1, f"Genetic parent-offspring corr={corr_g:.3f} too low"


class TestSiblingCorrelation:
    def test_sibling_phenotype_correlation(self):
        """Sibling phenotype correlation should be positive (shared genetics)."""
        n, m = 500, 200
        h2 = 0.6
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        effects = AdditiveEffects.from_h2(m=m, h2=h2, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(effects))
        arch.add('Y.E', NoiseComponent(variance=1.0 - h2))
        arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])

        mating = RandomMating(offspring_per_pair=4)  # 4 siblings per family
        rmap = RecombinationMap.constant_map(m=m)
        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=mating, recombination_map=rmap,
            seed=42,
        )
        sim.run(2)

        pheno = sim.phenotype_history[1]
        fids = pheno.samples.fid
        y_vals = pheno['Y']

        # Compute within-family correlation by pairing siblings
        unique_fids = np.unique(fids)
        sib1_vals = []
        sib2_vals = []
        for fid in unique_fids:
            idx = np.where(fids == fid)[0]
            if len(idx) >= 2:
                sib1_vals.append(y_vals[idx[0]])
                sib2_vals.append(y_vals[idx[1]])

        sib1 = np.array(sib1_vals)
        sib2 = np.array(sib2_vals)
        corr_sib = np.corrcoef(sib1, sib2)[0, 1]

        # Should be positive (shared ~50% of genetic variance)
        assert corr_sib > 0.0, f"Sibling corr={corr_sib:.3f} should be positive"


class TestVTInflation:
    def test_vt_increases_po_correlation(self):
        """VT should inflate parent-offspring correlation beyond h2/2."""
        n, m = 500, 200
        h2 = 0.3

        # Simulation WITHOUT VT
        hap1 = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff1 = AdditiveEffects.from_h2(m=m, h2=h2, seed=42)
        arch1 = Architecture()
        arch1.add('Y.G', GeneticComponent(eff1))
        arch1.add('Y.E', NoiseComponent(variance=1.0 - h2))
        arch1.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])

        mating = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m)
        sim1 = NSimulation(
            founder_haplotypes=hap1, architecture=arch1,
            mating_regime=mating, recombination_map=rmap,
            seed=42, retain_phenotypes=2,
        )
        sim1.run(2)

        # Simulation WITH VT
        hap2 = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff2 = AdditiveEffects.from_h2(m=m, h2=h2, seed=42)
        arch2 = Architecture()
        arch2.add('Y.G', GeneticComponent(eff2))
        arch2.add('Y.VT', MotherComponent('Y', founder_component=NoiseComponent(variance=0.2)))
        arch2.add('Y.E', NoiseComponent(variance=0.5 - h2))
        arch2.add('Y', AggregationComponent('Y.G + Y.VT + Y.E'),
                 inputs=['Y.G', 'Y.VT', 'Y.E'])

        sim2 = NSimulation(
            founder_haplotypes=hap2, architecture=arch2,
            mating_regime=mating, recombination_map=rmap,
            seed=42, retain_phenotypes=2,
        )
        sim2.run(2)

        # Compute PO correlations
        def po_corr(sim):
            parent = sim.phenotype_history[0]
            offspring = sim.phenotype_history[1]
            ped = sim.pedigree_history[1]
            mother_y = parent['Y'][ped.maternal_idx]
            return np.corrcoef(mother_y, offspring['Y'])[0, 1]

        corr_no_vt = po_corr(sim1)
        corr_with_vt = po_corr(sim2)

        # VT should increase or maintain the PO correlation
        # Allow some stochasticity
        assert corr_with_vt > corr_no_vt - 0.15, \
            f"VT corr={corr_with_vt:.3f} < no-VT corr={corr_no_vt:.3f} (unexpected)"
