"""Integration tests for simulation checkpoint/resume cycle."""
import numpy as np
import pytest

from tests.testdata import TestSimulation
from xftsim.sim import NSimulation
from xftsim.mate import RandomMating, LinearAssortativeMating
from xftsim.arch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.effect import AdditiveEffects
from xftsim.reproduce import RecombinationMap
from xftsim.io import save_simulation_checkpoint


def _make_sim(m=20, n=100, seed=42, mating_regime=None):
    eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=seed)
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(eff))
    arch.add('Y.E', NoiseComponent(variance=0.5))
    arch.add('Y', AggregationComponent('Y.G + Y.E'))

    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
    rmap = RecombinationMap.constant_map(m=m, p=0.5)
    if mating_regime is None:
        mating_regime = RandomMating(offspring_per_pair=2)
    return NSimulation(
        founder_haplotypes=hap, architecture=arch,
        mating_regime=mating_regime, recombination_map=rmap,
        retain_haplotypes=10, retain_phenotypes=10, seed=seed,
    )


class TestCheckpointResume:
    def test_resume_produces_valid_phenotypes(self, tmp_path):
        """Resumed simulation should produce finite phenotypes."""
        sim = _make_sim()
        sim.run(3)
        save_simulation_checkpoint(sim, str(tmp_path / "ckpt"))

        resumed = NSimulation.from_checkpoint(str(tmp_path / "ckpt"))
        resumed.continue_run(3)

        for gen in range(3, 6):
            assert gen in resumed.phenotype_history
            assert np.all(np.isfinite(resumed.phenotype_history[gen]['Y']))

    def test_resume_generation_counter(self, tmp_path):
        """Resume should pick up from the correct generation."""
        sim = _make_sim()
        sim.run(5)
        assert sim.generation == 4
        save_simulation_checkpoint(sim, str(tmp_path / "ckpt"))

        resumed = NSimulation.from_checkpoint(str(tmp_path / "ckpt"))
        assert resumed.generation == 4
        resumed.continue_run(3)
        assert resumed.generation == 7

    def test_resume_pedigree_valid(self, tmp_path):
        """Pedigrees from resumed simulation should have valid indices."""
        sim = _make_sim()
        sim.run(3)
        save_simulation_checkpoint(sim, str(tmp_path / "ckpt"))

        resumed = NSimulation.from_checkpoint(str(tmp_path / "ckpt"))
        resumed.continue_run(2)

        for gen in [3, 4]:
            ped = resumed.pedigree_history[gen]
            prev_hap = resumed.haplotype_history[gen - 1]
            assert np.all(ped.maternal_idx >= 0)
            assert np.all(ped.maternal_idx < prev_hap.n)
            assert np.all(ped.paternal_idx >= 0)
            assert np.all(ped.paternal_idx < prev_hap.n)


class TestAssortativeCheckpoint:
    def test_assortative_checkpoint_roundtrip(self, tmp_path):
        """LinearAssortativeMating should survive checkpoint."""
        mate = LinearAssortativeMating(
            component_names=['Y'], r=0.5, offspring_per_pair=2
        )
        sim = _make_sim(mating_regime=mate)
        sim.run(3)
        save_simulation_checkpoint(sim, str(tmp_path / "ckpt"))

        resumed = NSimulation.from_checkpoint(str(tmp_path / "ckpt"))
        assert isinstance(resumed.mating_regime, LinearAssortativeMating)
        assert resumed.mating_regime.r == 0.5
        assert resumed.mating_regime.component_names == ['Y']

    def test_assortative_resume_runs(self, tmp_path):
        """Resumed assortative mating simulation should run."""
        mate = LinearAssortativeMating(
            component_names=['Y'], r=0.5, offspring_per_pair=2
        )
        sim = _make_sim(mating_regime=mate)
        sim.run(2)
        save_simulation_checkpoint(sim, str(tmp_path / "ckpt"))

        resumed = NSimulation.from_checkpoint(str(tmp_path / "ckpt"))
        resumed.continue_run(2)
        assert resumed.generation == 3
        assert np.all(np.isfinite(resumed.phenotype_history[3]['Y']))


class TestCheckpointEdgeCases:
    def test_checkpoint_gen0_only(self, tmp_path):
        """Checkpointing after only gen 0 should work."""
        sim = _make_sim()
        sim.run(1)
        save_simulation_checkpoint(sim, str(tmp_path / "ckpt"))

        resumed = NSimulation.from_checkpoint(str(tmp_path / "ckpt"))
        assert resumed.generation == 0
        resumed.continue_run(2)
        assert resumed.generation == 2

    def test_continue_run_zero(self, tmp_path):
        """continue_run(0) should be a no-op."""
        sim = _make_sim()
        sim.run(3)
        save_simulation_checkpoint(sim, str(tmp_path / "ckpt"))

        resumed = NSimulation.from_checkpoint(str(tmp_path / "ckpt"))
        resumed.continue_run(0)
        assert resumed.generation == 2

    def test_retention_across_resume(self, tmp_path):
        """Retention policy should work across resume boundary."""
        sim = _make_sim()
        sim.run(3)
        save_simulation_checkpoint(sim, str(tmp_path / "ckpt"))

        resumed = NSimulation.from_checkpoint(str(tmp_path / "ckpt"))
        resumed.retain_phenotypes = 2
        resumed.retain_haplotypes = 2
        resumed.continue_run(5)

        # Should only have recent generations
        assert resumed.generation == 7
        assert 7 in resumed.phenotype_history
        assert 0 not in resumed.phenotype_history
