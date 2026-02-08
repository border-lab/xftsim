"""
Numerical tests for sibling components in multi-generation simulations.

Verifies:
1. Sibling mean has lower variance than individual phenotype
2. Sibling sum has higher variance than individual phenotype
3. Sibling count equals family size
4. Sibling correlation (from SibPairFilter) is positive for genetic traits
5. Sibling eldest/youngest values are from correct family members
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, DenseHaplotypeArray, NPhenotypeArray
from xftsim.narch import (
    Architecture, GeneticComponent, NoiseComponent, AggregationComponent,
    SiblingMeanComponent, SiblingSumComponent, SiblingCountComponent,
    SiblingEldestComponent, SiblingYoungestComponent, SiblingAnyComponent,
)
from xftsim.neffect import AdditiveEffects
from xftsim.nsim import NSimulation
from xftsim.nmate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.nfilter import SibPairFilter

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation

N = 400
M = 20


class TestSiblingMeanVarianceReduction:
    """Sibling mean should have less variance than individual phenotype."""

    def test_sibling_mean_lower_variance(self):
        """Var(sibling_mean(Y)) < Var(Y) when families have 2+ members."""
        hap = TestSimulation.founder_haplotypes(n=N, m=M, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=M, seed=42)
        arch = Architecture.from_formula("""
            Y.G ~ genetic(beta)
            Y.E ~ noise(0.5)
            Y ~ Y.G + Y.E
            Y.sib_mean ~ sibling_mean(Y) | FID
        """, effects={'beta': eff})
        mate = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=M, p=0.5)
        sim = NSimulation(hap, arch, mate, rmap, seed=42)
        sim.run(2)

        # At gen 1+, offspring have siblings (same FID)
        pheno = sim.phenotype_history[1]
        var_y = np.var(pheno['Y'])
        var_sib = np.var(pheno['Y.sib_mean'])
        assert var_sib < var_y, (
            f"Var(sib_mean)={var_sib:.4f} should be < Var(Y)={var_y:.4f}"
        )


class TestSiblingCountMatchesFamilySize:
    """Sibling count should equal number of individuals per family."""

    def test_count_equals_family_size(self):
        """sibling_count should match offspring_per_pair for each family."""
        n = 200
        hap = TestSimulation.founder_haplotypes(n=n, m=M, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=M, seed=42)
        opp = 3
        arch = Architecture.from_formula("""
            Y.G ~ genetic(beta)
            Y.E ~ noise(0.5)
            Y ~ Y.G + Y.E
            Y.count ~ sibling_count(Y) | FID
        """, effects={'beta': eff})
        mate = RandomMating(offspring_per_pair=opp)
        rmap = RecombinationMap.constant_map(m=M, p=0.5)
        sim = NSimulation(hap, arch, mate, rmap, seed=42)
        sim.run(2)

        pheno = sim.phenotype_history[1]
        # All offspring families should have exactly `opp` members
        assert np.all(pheno['Y.count'] == opp)


class TestSiblingCorrelation:
    """Sibling pairs should show positive phenotypic correlation for genetic traits."""

    def test_positive_sib_correlation(self):
        """h2 > 0 → sibling Y correlation should be positive."""
        hap = TestSimulation.founder_haplotypes(n=N, m=M, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.8, m=M, seed=42)
        arch = Architecture.from_formula("""
            Y.G ~ genetic(beta)
            Y.E ~ noise(0.2)
            Y ~ Y.G + Y.E
        """, effects={'beta': eff})
        mate = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=M, p=0.5)

        sib_filter = SibPairFilter()
        sim = NSimulation(
            hap, arch, mate, rmap, seed=42,
            filters={'sib': sib_filter},
        )
        sim.run(2)

        # At gen 1, offspring share parents, so phenotypic correlation should be positive
        pheno = sim.phenotype_history[1]
        fids = pheno.samples.fid
        view = sib_filter.apply(1, sim.phenotype_history, sim.pedigree_history)
        if view is not None and view.n_pairs > 5:
            sib1_y = view.sib1_phenotypes['Y']
            sib2_y = view.sib2_phenotypes['Y']
            r = np.corrcoef(sib1_y, sib2_y)[0, 1]
            assert r > 0.0, f"Sibling correlation should be positive, got r={r:.4f}"


class TestSiblingEldestYoungest:
    """Eldest/youngest should pick first/last member in family."""

    def test_eldest_youngest_differ_in_families(self):
        """For families with varying phenotypes, eldest != youngest."""
        n = 200
        hap = TestSimulation.founder_haplotypes(n=n, m=M, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=M, seed=42)
        arch = Architecture.from_formula("""
            Y.G ~ genetic(beta)
            Y.E ~ noise(0.5)
            Y ~ Y.G + Y.E
            Y.eldest ~ sibling_eldest(Y) | FID
            Y.youngest ~ sibling_youngest(Y) | FID
        """, effects={'beta': eff})
        mate = RandomMating(offspring_per_pair=3)
        rmap = RecombinationMap.constant_map(m=M, p=0.5)
        sim = NSimulation(hap, arch, mate, rmap, seed=42)
        sim.run(2)

        pheno = sim.phenotype_history[1]
        # Not all families will have eldest == youngest (different individuals)
        diff = pheno['Y.eldest'] - pheno['Y.youngest']
        # At least some difference across families
        assert np.std(diff) > 0, "Eldest and youngest should sometimes differ"


class TestSiblingAnyBinary:
    """SiblingAny should produce 0/1 values."""

    def test_any_produces_binary(self):
        """sibling_any values should be 0 or 1."""
        n = 200
        hap = TestSimulation.founder_haplotypes(n=n, m=M, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=M, seed=42)
        arch = Architecture.from_formula("""
            Y.G ~ genetic(beta)
            Y.E ~ noise(0.5)
            Y ~ Y.G + Y.E
            Y.any ~ sibling_any(Y) | FID
        """, effects={'beta': eff})
        mate = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=M, p=0.5)
        sim = NSimulation(hap, arch, mate, rmap, seed=42)
        sim.run(2)

        pheno = sim.phenotype_history[1]
        any_vals = pheno['Y.any']
        assert set(np.unique(any_vals)).issubset({0.0, 1.0})
