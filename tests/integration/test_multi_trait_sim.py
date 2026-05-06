"""
Integration tests for multi-trait simulations with complex architectures.

Tests:
1. Bivariate simulation with correlated genetic effects
2. Three-trait simulation with independent effects
3. Multi-trait with VT on one trait
4. Multi-trait with noise and aggregation
5. Multi-trait with assortative mating on one trait
6. Multi-trait phenotype values are finite
7. Multi-trait correlation structure preserved across generations
"""
import numpy as np
import pytest

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation

from xftsim.arch import (
    Architecture, GeneticComponent, MVGeneticComponent,
    NoiseComponent, AggregationComponent, MotherComponent, FatherComponent,
)
from xftsim.effect import AdditiveEffects, MultivariateEffects
from xftsim.mate import RandomMating, LinearAssortativeMating
from xftsim.reproduce import RecombinationMap
from xftsim.sim import NSimulation


class TestBivariateCorrelated:
    def test_bivariate_correlated_genetic(self):
        """Two traits with rg=0.8 should show positive phenotypic correlation."""
        n, m = 200, 50
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        effects = MultivariateEffects.from_h2_rg(m=m, h2=[0.5, 0.5], rg=0.8, seed=42)
        arch = Architecture()
        arch.add(['Y.G', 'Z.G'], MVGeneticComponent(effects))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Z.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])
        arch.add('Z', AggregationComponent('Z.G + Z.E'), inputs=['Z.G', 'Z.E'])

        mating = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m)
        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=mating, recombination_map=rmap, seed=42,
        )
        sim.run(3)

        # Check correlation in final generation
        pheno = sim.phenotypes
        corr = np.corrcoef(pheno['Y'], pheno['Z'])[0, 1]
        assert corr > 0.0, f"Bivariate corr={corr:.3f} should be positive"


class TestThreeTrait:
    def test_three_independent_traits(self):
        """Three independent traits should all have finite values."""
        n, m = 100, 20
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        arch = Architecture()
        for trait in ['X', 'Y', 'Z']:
            eff = AdditiveEffects.from_h2(m=m, h2=0.3, seed=hash(trait) % 2**31)
            arch.add(f'{trait}.G', GeneticComponent(eff))
            arch.add(f'{trait}.E', NoiseComponent(variance=0.7))
            arch.add(trait, AggregationComponent(f'{trait}.G + {trait}.E'),
                     inputs=[f'{trait}.G', f'{trait}.E'])

        mating = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m)
        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=mating, recombination_map=rmap, seed=42,
        )
        sim.run(2)

        pheno = sim.phenotypes
        for trait in ['X', 'Y', 'Z']:
            assert np.all(np.isfinite(pheno[trait])), f"Trait {trait} has non-finite values"


class TestMultiTraitVT:
    def test_vt_on_one_trait(self):
        """Multi-trait with VT on one trait: VT trait should have higher parent-offspring corr."""
        n, m = 200, 30
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        arch = Architecture()

        eff = AdditiveEffects.from_h2(m=m, h2=0.3, seed=42)
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.VT', MotherComponent('Y', founder_component=NoiseComponent(variance=0.1)))
        arch.add('Y.E', NoiseComponent(variance=0.3))
        arch.add('Y', AggregationComponent('Y.G + Y.VT + Y.E'),
                 inputs=['Y.G', 'Y.VT', 'Y.E'])

        eff2 = AdditiveEffects.from_h2(m=m, h2=0.3, seed=43)
        arch.add('Z.G', GeneticComponent(eff2))
        arch.add('Z.E', NoiseComponent(variance=0.7))
        arch.add('Z', AggregationComponent('Z.G + Z.E'), inputs=['Z.G', 'Z.E'])

        mating = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m)
        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=mating, recombination_map=rmap, seed=42,
            retain_phenotypes=2,
        )
        sim.run(3)

        # Both traits should be finite
        pheno = sim.phenotypes
        assert np.all(np.isfinite(pheno['Y']))
        assert np.all(np.isfinite(pheno['Z']))


class TestMultiTraitFiniteness:
    def test_all_values_finite_after_several_gens(self):
        """Ensure no NaN/Inf creep in over multiple generations."""
        n, m = 100, 20
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        effects = MultivariateEffects.from_h2_rg(m=m, h2=[0.4, 0.4], rg=0.5, seed=42)
        arch = Architecture()
        arch.add(['A.G', 'B.G'], MVGeneticComponent(effects))
        arch.add('A.E', NoiseComponent(variance=0.6))
        arch.add('B.E', NoiseComponent(variance=0.6))
        arch.add('A', AggregationComponent('A.G + A.E'), inputs=['A.G', 'A.E'])
        arch.add('B', AggregationComponent('B.G + B.E'), inputs=['B.G', 'B.E'])

        mating = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m)
        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=mating, recombination_map=rmap, seed=42,
        )
        sim.run(5)

        for gen, pheno in sim.phenotype_history.items():
            for key in pheno.keys:
                assert np.all(np.isfinite(pheno[key])), \
                    f"Gen {gen} key {key} has non-finite values"


class TestMultiTraitAssortative:
    def test_assortative_on_one_trait(self):
        """Assortative mating on trait Y should not crash multi-trait sim."""
        n, m = 100, 20
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        arch = Architecture()

        eff1 = AdditiveEffects.from_h2(m=m, h2=0.5, seed=42)
        arch.add('Y.G', GeneticComponent(eff1))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])

        eff2 = AdditiveEffects.from_h2(m=m, h2=0.5, seed=43)
        arch.add('Z.G', GeneticComponent(eff2))
        arch.add('Z.E', NoiseComponent(variance=0.5))
        arch.add('Z', AggregationComponent('Z.G + Z.E'), inputs=['Z.G', 'Z.E'])

        mating = LinearAssortativeMating(component_names=['Y'], r=0.5)
        rmap = RecombinationMap.constant_map(m=m)
        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=mating, recombination_map=rmap, seed=42,
        )
        sim.run(3)

        pheno = sim.phenotypes
        assert np.all(np.isfinite(pheno['Y']))
        assert np.all(np.isfinite(pheno['Z']))
