"""
Integration test: stdpopsim founders -> GRG-native multi-generation simulation.

Runs the same simulation on GRG-backed and dense founders with one seed and
requires identical genotypes and genetic values every generation. stdpopsim's
finite-sites mutation model produces recurrent mutations, which used to
desynchronize the two paths after GRG meiosis.
"""
import numpy as np
import pytest

pygrgl = pytest.importorskip("pygrgl")
pytest.importorskip("msprime")
pytest.importorskip("stdpopsim")

from xftsim.arch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.effect import AdditiveEffects
from xftsim.founders import founder_haplotypes_from_stdpopsim_grg
from xftsim.mate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.sim import Simulation
from xftsim.struct import DenseHaplotypeArray, GraphHaplotypeOperator


N_GEN = 4
SEED = 3


def _make_founders():
    return founder_haplotypes_from_stdpopsim_grg(
        samples={"YRI": 60, "CEU": 60}, chromosome="chr22",
        left=20_000_000, right=21_000_000, seed=SEED,
    )


@pytest.fixture(scope="module")
def founders():
    return _make_founders()


def _sim(founder, rmap, standardize_with='founders'):
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(AdditiveEffects.from_h2(h2=0.5, m=founder.m, seed=SEED)))
    arch.add('Y.E', NoiseComponent(variance=0.5))
    arch.add('Y', AggregationComponent('Y.G + Y.E'))
    return Simulation(founder_haplotypes=founder, architecture=arch,
                      mating_regime=RandomMating(offspring_per_pair=2),
                      recombination_map=rmap, seed=SEED, retain_haplotypes=0,
                      retain_phenotypes=N_GEN, standardize_with=standardize_with)


class TestStdpopsimGRGSimulation:
    def test_one_mutation_per_position(self, founders):
        assert np.all(np.diff(founders.variants.pos_bp) > 0)

    def test_cM_map_from_stdpopsim_founders(self, founders):
        rmap = RecombinationMap.from_haplotypes_with_cM(founders)
        p = rmap._probabilities
        assert p.shape == (founders.m,)
        assert p[0] == 0.5
        assert np.all((p[1:] >= 0) & (p[1:] < 0.5))
        # 1 Mb of chr22 is ~1-2 cM: a few hundredths of a crossover per meiosis.
        assert 0.001 < p[1:].sum() < 0.1

    def test_old_grg_checkpoint_cannot_switch_to_founders(self, tmp_path):
        """A retained generation-0 GRG shares the graph meiosis mutated."""
        import json
        import os
        import xftsim as xft
        founders = _make_founders()
        rmap = RecombinationMap.from_haplotypes_with_cM(founders)
        sim = _sim(founders, rmap, 'current')
        sim.retain_haplotypes = N_GEN
        sim.run(3)
        assert 0 in sim.haplotype_history
        xft.io.save_simulation_checkpoint(sim, str(tmp_path))
        os.remove(os.path.join(str(tmp_path), 'founder_af.npz'))
        meta_path = os.path.join(str(tmp_path), 'meta.json')
        meta = json.load(open(meta_path))
        del meta['standardize_with']
        json.dump(meta, open(meta_path, 'w'))
        with pytest.raises(ValueError, match="founders' allele"):
            Simulation.from_checkpoint(str(tmp_path), standardize_with='founders')

    @pytest.mark.parametrize("standardize_with", ["founders", "current"])
    def test_grg_matches_dense_across_generations(self, standardize_with):
        # GRG meiosis mutates the founders' graph in place, so each run
        # needs its own founders.
        founders = _make_founders()
        rmap = RecombinationMap.from_haplotypes_with_cM(founders)
        dense = founders.to_dense()
        dense_founders = DenseHaplotypeArray(genotypes=dense.genotypes.copy(),
                                             samples=founders.samples,
                                             variants=founders.variants)
        sim_dense = _sim(dense_founders, rmap, standardize_with)
        sim_dense.run(N_GEN)
        sim_grg = _sim(founders, rmap, standardize_with)
        sim_grg.run(N_GEN)
        np.testing.assert_allclose(sim_grg.founder_af, sim_dense.founder_af, atol=1e-12)

        last = sim_grg.haplotypes
        assert isinstance(last, GraphHaplotypeOperator)
        np.testing.assert_array_equal(last.to_dense().genotypes,
                                      sim_dense.haplotypes.genotypes)
        for gen in range(N_GEN):
            np.testing.assert_allclose(sim_grg.phenotype_history[gen]['Y.G'],
                                       sim_dense.phenotype_history[gen]['Y.G'],
                                       atol=1e-10)
