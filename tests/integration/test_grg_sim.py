"""Integration tests: GRG-backed founders through multi-generation simulation."""
import numpy as np
import pytest

pygrgl = pytest.importorskip("pygrgl")

from xftsim.struct import GraphHaplotypeOperator, DenseHaplotypeArray, SampleMeta
from xftsim.nsim import NSimulation
from xftsim.neffect import AdditiveEffects
from xftsim.narch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.nmate import RandomMating
from xftsim.reproduce import RecombinationMap
from tests.testdata import TestGRG


def _make_sim(grg_op, seed=42):
    """Build a simple simulation from a GRG operator."""
    m = grg_op.m
    # Ensure balanced sex
    sex = np.tile([0, 1], (grg_op.n + 1) // 2)[:grg_op.n]
    samples = SampleMeta(iid=grg_op.samples.iid, sex=sex)
    founder = GraphHaplotypeOperator(grg_op._grg, samples=samples)

    effects = AdditiveEffects.from_h2(h2=0.5, m=m, seed=seed)
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(effects))
    arch.add('Y.E', NoiseComponent(variance=0.5))
    arch.add('Y', AggregationComponent('Y.G + Y.E'))

    rmap = RecombinationMap.constant_map(m=m, p=0.5)
    mate = RandomMating(offspring_per_pair=2)

    return NSimulation(
        founder_haplotypes=founder,
        architecture=arch,
        mating_regime=mate,
        recombination_map=rmap,
        seed=seed,
    )


class TestGRGSimulation:
    def test_gen0_phenotypes_finite(self):
        """Generation-0 phenotypes from GRG founders should be finite."""
        grg_op = TestGRG.tiny_grg()
        sim = _make_sim(grg_op)
        sim.run(1)
        pheno = sim.phenotype_history[0]
        assert np.all(np.isfinite(pheno['Y']))

    def test_multi_gen_uses_dense_after_meiosis(self):
        """After meiosis, offspring haplotypes should be DenseHaplotypeArray."""
        grg_op = TestGRG.tiny_grg()
        sim = _make_sim(grg_op)
        sim.run(2)
        assert isinstance(sim.haplotype_history[0], GraphHaplotypeOperator)
        assert isinstance(sim.haplotype_history[1], DenseHaplotypeArray)

    def test_gen0_phenotypes_match_dense(self):
        """Gen-0 Y.G from GRG founders should match dense equivalent."""
        grg_op = TestGRG.tiny_grg()
        seed = 42

        # GRG sim
        sim_grg = _make_sim(grg_op, seed=seed)
        sim_grg.run(1)

        # Dense sim with same genotypes
        dense = grg_op.to_dense()
        sex = np.tile([0, 1], (dense.n + 1) // 2)[:dense.n]
        samples = SampleMeta(iid=dense.samples.iid, sex=sex)
        dense_founder = DenseHaplotypeArray(
            genotypes=dense.genotypes, samples=samples, variants=dense.variants,
        )
        sim_dense = _make_sim.__wrapped__(dense_founder, seed=seed) if hasattr(_make_sim, '__wrapped__') else None
        # Build manually to use dense founder
        m = dense_founder.m
        effects = AdditiveEffects.from_h2(h2=0.5, m=m, seed=seed)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(effects))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        mate = RandomMating(offspring_per_pair=2)
        sim_dense = NSimulation(
            founder_haplotypes=dense_founder,
            architecture=arch, mating_regime=mate,
            recombination_map=rmap, seed=seed,
        )
        sim_dense.run(1)

        # Y.G should match (noise may differ unless RNG state is same,
        # but the genetic component is deterministic given same genotypes + effects)
        np.testing.assert_allclose(
            sim_grg.phenotype_history[0]['Y.G'],
            sim_dense.phenotype_history[0]['Y.G'],
            atol=1e-10,
        )

    def test_small_grg_two_generations(self):
        """Small GRG (100 individuals) runs 2 generations without error."""
        grg_op = TestGRG.small_grg()
        sim = _make_sim(grg_op)
        sim.run(2)
        assert sim.generation == 1
        assert np.all(np.isfinite(sim.phenotype_history[1]['Y']))
