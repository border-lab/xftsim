"""Numerical tests for GRG-backed simulations.

Verify that GRG founder simulations produce correct heritability estimates
and match dense equivalents.
"""
import numpy as np
import pytest

pygrgl = pytest.importorskip("pygrgl")

from xftsim.struct import GraphHaplotypeOperator, DenseHaplotypeArray, SampleMeta
from xftsim.effect import AdditiveEffects
from xftsim.arch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.sim import Simulation
from xftsim.mate import RandomMating
from xftsim.reproduce import RecombinationMap
from tests.testdata import TestGRG


def _make_grg_sim(grg_op, h2=0.5, seed=42):
    """Build a simulation from a GRG operator."""
    m = grg_op.m
    sex = np.tile([0, 1], (grg_op.n + 1) // 2)[:grg_op.n]
    samples = SampleMeta(iid=grg_op.samples.iid, sex=sex)
    founder = GraphHaplotypeOperator(grg_op._grg, samples=samples)

    effects = AdditiveEffects.from_h2(h2=h2, m=m, seed=seed)
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(effects))
    arch.add('Y.E', NoiseComponent(variance=1.0 - h2))
    arch.add('Y', AggregationComponent('Y.G + Y.E'))

    rmap = RecombinationMap.constant_map(m=m, p=0.5)
    mate = RandomMating(offspring_per_pair=2)

    return Simulation(
        founder_haplotypes=founder,
        architecture=arch,
        mating_regime=mate,
        recombination_map=rmap,
        retain_haplotypes=10,
        retain_phenotypes=10,
        seed=seed,
    ), effects


class TestGRGHeritability:
    """Verify GRG-backed simulations produce reasonable heritability."""

    def test_genetic_variance_positive(self):
        grg_op = TestGRG.small_grg()  # 100 individuals, 1000 variants
        sim, _ = _make_grg_sim(grg_op)
        sim.run(1)
        yg = sim.phenotype_history[0]['Y.G']
        assert np.var(yg) > 0

    def test_phenotypic_variance_reasonable(self):
        grg_op = TestGRG.small_grg()
        sim, _ = _make_grg_sim(grg_op, h2=0.5)
        sim.run(1)
        y = sim.phenotype_history[0]['Y']
        var_y = np.var(y)
        # Phenotypic variance should be in a reasonable range
        # E[Var(Y)] = Var(G) + Var(E) ≈ h2 + (1-h2) = 1
        # But with finite n and standardized effects, actual can vary
        assert var_y > 0.1
        assert var_y < 10.0

    def test_grg_dense_genetic_values_identical(self):
        """GRG and dense should produce identical genetic values."""
        grg_op = TestGRG.small_grg()
        m = grg_op.m
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)

        # GRG matvec
        grg_gv = grg_op.matvec(eff.effects)

        # Dense matvec
        dense = grg_op.to_dense()
        dense_gv = dense.matvec(eff.effects)

        np.testing.assert_allclose(grg_gv, dense_gv, atol=1e-10)


class TestGRGMultiGen:
    """Multi-generation GRG simulations."""

    def test_three_gen_completes(self):
        grg_op = TestGRG.small_grg()
        sim, _ = _make_grg_sim(grg_op)
        sim.run(3)
        assert sim.generation == 2

    def test_offspring_af_reasonable(self):
        """Offspring AFs should be in [0,1] and close to founder AFs."""
        grg_op = TestGRG.small_grg()
        sim, _ = _make_grg_sim(grg_op)
        sim.run(2)
        af_0 = sim.haplotype_history[0].recompute_af()
        af_1 = sim.haplotype_history[1].recompute_af()
        assert np.all(af_1 >= 0)
        assert np.all(af_1 <= 1)
        # Mean AF should not change drastically in one generation
        assert abs(af_0.mean() - af_1.mean()) < 0.1

    def test_gen1_is_graph(self):
        """After GRG-native meiosis, haplotypes stay as a GraphHaplotypeOperator."""
        grg_op = TestGRG.small_grg()
        sim, _ = _make_grg_sim(grg_op)
        sim.run(2)
        assert isinstance(sim.haplotype_history[1], GraphHaplotypeOperator)
