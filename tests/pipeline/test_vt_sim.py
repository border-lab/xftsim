"""Integration tests for vertical transmission simulations."""
import warnings
import numpy as np
import pytest

from tests.testdata import TestSimulation
from xftsim.sim import Simulation
from xftsim.arch import (
    Architecture, GeneticComponent, NoiseComponent, AggregationComponent,
    MotherComponent, FatherComponent, ParentComponent,
)
from xftsim.effect import AdditiveEffects


def _run_vt_sim(n_gen=3, seed=42, vt_weight=0.3, arch=None):
    hap = TestSimulation.founder_haplotypes(n=500, m=50, seed=seed)
    if arch is None:
        arch = TestSimulation.vt_architecture(m=50, h2=0.5, vt_weight=vt_weight, seed=123)
    rmap = TestSimulation.recombination_map(m=50)
    mate = TestSimulation.mating_regime()
    sim = Simulation(
        founder_haplotypes=hap, architecture=arch,
        mating_regime=mate, recombination_map=rmap,
        retain_phenotypes=5, seed=seed,
    )
    sim.run(n_gen)
    return sim


class TestVerticalTransmission:
    def test_founder_fallback(self):
        """Gen 0 should use founder fallback (noise) for VT component."""
        sim = _run_vt_sim(n_gen=1)
        pheno = sim.phenotype_history[0]
        # Y.VT at gen 0 comes from founder noise, should have nonzero variance
        vt_vals = pheno['Y.VT']
        assert np.std(vt_vals) > 0

    def test_parent_lookup_gen1(self):
        """Gen 1 VT should reflect actual parent phenotypes from gen 0."""
        sim = _run_vt_sim(n_gen=2)
        # Y.VT at gen 1 is midparent of gen-0 Y values
        ped = sim.pedigree_history[1]
        parent_y = sim.phenotype_history[0]['Y']
        expected_vt = 0.5 * (parent_y[ped.maternal_idx] + parent_y[ped.paternal_idx])
        np.testing.assert_allclose(sim.phenotype_history[1]['Y.VT'], expected_vt)

    def test_vt_coefficient_affects_phenotype(self):
        """VT weight should modulate the VT contribution."""
        sim_low = _run_vt_sim(n_gen=2, vt_weight=0.1, seed=99)
        sim_high = _run_vt_sim(n_gen=2, vt_weight=0.9, seed=99)
        # Higher VT weight should increase parent-offspring correlation
        # (checking variance of VT component as a proxy)
        var_low = np.var(sim_low.phenotype_history[1]['Y.VT'])
        var_high = np.var(sim_high.phenotype_history[1]['Y.VT'])
        # VT component is the same (midparent); difference is in the weight
        # applied during aggregation: Y = Y.G + weight * Y.VT + Y.E
        # We check that Y variance differs
        y_var_low = np.var(sim_low.phenotype_history[1]['Y'])
        y_var_high = np.var(sim_high.phenotype_history[1]['Y'])
        # High VT weight inflates Y variance
        assert y_var_high > y_var_low

    def test_mother_only_architecture(self):
        """Architecture with only mother(Y) should work."""
        m = 50
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=123)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.M', MotherComponent('Y', founder_component=NoiseComponent(variance=0.3)))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + 0.2 * Y.M + Y.E'))
        sim = _run_vt_sim(n_gen=3, arch=arch)
        assert 'Y.M' in sim.phenotype_history[2]

    def test_father_only_architecture(self):
        """Architecture with only father(Y) should work."""
        m = 50
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=123)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.F', FatherComponent('Y', founder_component=NoiseComponent(variance=0.3)))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + 0.2 * Y.F + Y.E'))
        sim = _run_vt_sim(n_gen=3, arch=arch)
        assert 'Y.F' in sim.phenotype_history[2]

    def test_multi_gen_vt_runs_without_error(self):
        """5-generation VT simulation should complete without error."""
        sim = _run_vt_sim(n_gen=5)
        assert sim.generation == 4
