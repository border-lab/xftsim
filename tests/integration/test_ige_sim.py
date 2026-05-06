"""Integration tests for indirect genetic effect (IGE) simulations using haplotypeGenetic."""
import numpy as np
import pytest

from tests.testdata import TestSimulation
from xftsim.sim import NSimulation
from xftsim.arch import (
    Architecture, GeneticComponent, HaplotypeGeneticComponent,
    NoiseComponent, AggregationComponent,
)
from xftsim.effect import AdditiveEffects


def _run_ige_sim(n_gen=3, seed=42, arch=None):
    hap = TestSimulation.founder_haplotypes(n=500, m=50, seed=seed)
    if arch is None:
        eff = AdditiveEffects.from_h2(h2=0.5, m=50, seed=123, standardized=False)
        arch = Architecture()
        arch.add('Y.mat', HaplotypeGeneticComponent(eff, haplotype='maternal'))
        arch.add('Y.pat', HaplotypeGeneticComponent(eff, haplotype='paternal'))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.mat + Y.pat + Y.E'))
    rmap = TestSimulation.recombination_map(m=50)
    mate = TestSimulation.mating_regime()
    sim = NSimulation(
        founder_haplotypes=hap, architecture=arch,
        mating_regime=mate, recombination_map=rmap,
        retain_phenotypes=5, retain_haplotypes=5, seed=seed,
    )
    sim.run(n_gen)
    return sim


class TestIGESimulation:
    def test_maternal_paternal_in_sim(self):
        """Both maternal and paternal haplotype components should be computed."""
        sim = _run_ige_sim(n_gen=2)
        for gen in range(2):
            assert 'Y.mat' in sim.phenotype_history[gen]
            assert 'Y.pat' in sim.phenotype_history[gen]

    def test_ige_formula_runs(self):
        """IGE formula should parse and run in a simulation."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=50, seed=123, standardized=False)
        arch = Architecture(
            formula=(
                "Y.mat ~ haplotypeGenetic(eff)\n"
                "Y.pat ~ haplotypeGenetic(eff, haplotype='paternal')\n"
                "Y.E ~ noise(0.5)\n"
                "Y ~ Y.mat + Y.pat + Y.E"
            ),
            effects={'eff': eff},
        )
        sim = _run_ige_sim(n_gen=2, arch=arch)
        assert sim.generation == 1

    def test_maternal_differs_from_diploid_in_offspring(self):
        """After meiosis, maternal haplotype value should differ from full diploid."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=50, seed=123, standardized=False)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.mat', HaplotypeGeneticComponent(eff, haplotype='maternal'))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        sim = _run_ige_sim(n_gen=2, arch=arch)
        gen1 = sim.phenotype_history[1]
        # Y.mat != Y.G in general (Y.G uses both haplotypes)
        assert not np.allclose(gen1['Y.mat'], gen1['Y.G'])

    def test_multi_gen_ige(self):
        """Multi-generation IGE simulation should complete."""
        sim = _run_ige_sim(n_gen=4)
        assert sim.generation == 3

    def test_mat_plus_pat_equals_diploid(self):
        """At gen 0, mat + pat should equal diploid (non-standardized)."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=50, seed=123, standardized=False)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.mat', HaplotypeGeneticComponent(eff, haplotype='maternal'))
        arch.add('Y.pat', HaplotypeGeneticComponent(eff, haplotype='paternal'))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        sim = _run_ige_sim(n_gen=1, arch=arch)
        gen0 = sim.phenotype_history[0]
        np.testing.assert_allclose(
            gen0['Y.mat'] + gen0['Y.pat'], gen0['Y.G'], atol=1e-10
        )
