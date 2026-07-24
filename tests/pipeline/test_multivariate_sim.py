"""Integration tests for multivariate simulations."""
import numpy as np
import pytest

from tests.testdata import TestSimulation
from xftsim.sim import Simulation
from xftsim.filters import TrioFilter, SibPairFilter
from xftsim.stats import SampleStatistics


def _run_biv_sim(n_gen=3, seed=42, filters=None, statistics=None):
    hap = TestSimulation.founder_haplotypes(n=500, m=50, seed=seed)
    arch = TestSimulation.bivariate_architecture(m=50, h2=[0.5, 0.3], rg=0.2, seed=123)
    rmap = TestSimulation.recombination_map(m=50)
    mate = TestSimulation.mating_regime()
    sim = Simulation(
        founder_haplotypes=hap, architecture=arch,
        mating_regime=mate, recombination_map=rmap,
        retain_phenotypes=5, seed=seed,
        filters=filters or {},
        statistics=statistics or [],
    )
    sim.run(n_gen)
    return sim


class TestMultivariateSim:
    def test_bivariate_3gen(self):
        sim = _run_biv_sim(n_gen=3)
        assert sim.generation == 2

    def test_both_traits_populated(self):
        sim = _run_biv_sim(n_gen=2)
        for gen in range(2):
            pheno = sim.phenotype_history[gen]
            assert 'trait1' in pheno
            assert 'trait2' in pheno
            assert 'trait1.G' in pheno
            assert 'trait2.G' in pheno

    def test_with_filters(self):
        filters = {
            'trios': TrioFilter(),
            'sib_pairs': SibPairFilter(),
        }
        sim = _run_biv_sim(n_gen=3, filters=filters)
        assert sim.generation == 2

    def test_with_statistics(self):
        stats = [SampleStatistics()]
        sim = _run_biv_sim(n_gen=3, statistics=stats)
        assert len(sim.results) == 3  # one per generation
        # Check that statistics contain both traits
        last_result = sim.results[-1].statistics['SampleStatistics']
        assert 'trait1' in last_result['keys']
        assert 'trait2' in last_result['keys']

    def test_cross_trait_from_same_node(self):
        """trait1.G and trait2.G come from the same mvGenetic node."""
        sim = _run_biv_sim(n_gen=1)
        pheno = sim.phenotype_history[0]
        # Both G components should exist and differ
        assert not np.allclose(pheno['trait1.G'], pheno['trait2.G'])
