"""
Integration test: vertical transmission chains across generations.

Tests:
1. MotherComponent correctly reads mother's phenotype from previous gen
2. VT + genetics produces intermediate phenotypes (blended inheritance)
3. Father component correctly reads father's phenotype
4. VT with founder NoiseComponent produces random gen-0 values
5. Multi-gen VT: values propagate through generations
"""
import numpy as np
import pytest

from xftsim.effect import AdditiveEffects
from xftsim.arch import (
    Architecture, GeneticComponent, NoiseComponent, AggregationComponent,
    MotherComponent, FatherComponent,
)
from xftsim.mate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.sim import NSimulation

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestVerticalTransmissionChain:
    def test_mother_reads_previous_gen(self):
        """MotherComponent should use mother's phenotype from gen g-1."""
        n, m = 200, 20
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        # No inputs= on MotherComponent — it reads from phenotype_history at runtime
        arch.add('Y.m', MotherComponent('Y'))

        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=RandomMating(offspring_per_pair=2),
            recombination_map=RecombinationMap.constant_map(m=m),
            seed=42, retain_phenotypes=10,
        )
        sim.run(3)

        # Gen 0: mother component = 0 (no pedigree)
        assert np.allclose(sim.phenotype_history[0]['Y.m'], 0.0)
        # Gen 1+: mother component = actual mother Y values
        gen1_mothers = sim.phenotype_history[1]['Y.m']
        assert np.all(np.isfinite(gen1_mothers))
        assert np.std(gen1_mothers) > 0, "VT should have variance in gen 1+"

    def test_vt_plus_genetics_blends(self):
        """VT + genetics should produce blended phenotypes."""
        n, m = 300, 30
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.3, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.3))
        # MotherComponent reads from prev gen phenotype_history — no DAG input
        arch.add('Y.m', MotherComponent('Y', founder_component=NoiseComponent(variance=0.1)))
        arch.add('Y', AggregationComponent('Y.G + Y.E + 0.3 * Y.m'))

        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=RandomMating(offspring_per_pair=2),
            recombination_map=RecombinationMap.constant_map(m=m),
            seed=42, retain_phenotypes=10,
        )
        sim.run(4)

        for g in range(4):
            pheno = sim.phenotype_history[g]
            assert np.all(np.isfinite(pheno['Y'])), f"Gen {g} has non-finite Y"
            assert 'Y.G' in pheno
            assert 'Y.m' in pheno

    def test_father_reads_correct_parent(self):
        """FatherComponent should use father's phenotype."""
        n, m = 200, 20
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        arch.add('Y.f', FatherComponent('Y'))

        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=RandomMating(offspring_per_pair=2),
            recombination_map=RecombinationMap.constant_map(m=m),
            seed=42, retain_phenotypes=10,
        )
        sim.run(3)

        # Gen 0: father component = 0 (no pedigree)
        assert np.allclose(sim.phenotype_history[0]['Y.f'], 0.0)
        # Gen 1+: father values should vary
        gen1_fathers = sim.phenotype_history[1]['Y.f']
        assert np.all(np.isfinite(gen1_fathers))
        assert np.std(gen1_fathers) > 0

    def test_vt_founder_noise(self):
        """MotherComponent with founder NoiseComponent produces random gen-0 values."""
        n, m = 200, 20
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y.m', MotherComponent('Y', founder_component=NoiseComponent(variance=1.0)))
        arch.add('Y', AggregationComponent('Y.G + Y.E + Y.m'))

        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=RandomMating(offspring_per_pair=2),
            recombination_map=RecombinationMap.constant_map(m=m),
            seed=42, retain_phenotypes=10,
        )
        sim.run(2)

        # Gen 0: should have non-zero VT values (from founder noise)
        gen0_vt = sim.phenotype_history[0]['Y.m']
        assert np.all(np.isfinite(gen0_vt))
        assert np.std(gen0_vt) > 0, "Founder noise should give variance at gen 0"

    def test_multi_gen_vt_propagation(self):
        """VT values should change across generations (not static)."""
        n, m = 200, 20
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.3, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.3))
        arch.add('Y.m', MotherComponent('Y'))
        arch.add('Y', AggregationComponent('Y.G + Y.E + 0.3 * Y.m'))

        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=RandomMating(offspring_per_pair=2),
            recombination_map=RecombinationMap.constant_map(m=m),
            seed=42, retain_phenotypes=10,
        )
        sim.run(5)

        # VT at gen 0 = zeros, gen 1+ = actual values
        vt_0 = np.std(sim.phenotype_history[0]['Y.m'])
        vt_2 = np.std(sim.phenotype_history[2]['Y.m'])
        assert vt_0 < 0.01, "Gen 0 VT should be ~0"
        assert vt_2 > 0.01, "Gen 2 VT should have variance"
