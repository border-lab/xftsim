"""
Integration tests for simulation checkpoint save/load roundtrips.

Tests:
1. Full checkpoint roundtrip preserves generation
2. Checkpoint roundtrip preserves haplotype dimensions
3. Checkpoint roundtrip preserves phenotype values
4. Checkpoint roundtrip preserves pedigree structure
5. Continue from checkpoint produces valid simulation
6. Checkpoint with retention policy
7. Checkpoint with assortative mating
8. Missing checkpoint directory raises FileNotFoundError
"""
import numpy as np
import pytest
import tempfile
import os

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation

from xftsim.arch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.effect import AdditiveEffects
from xftsim.mate import RandomMating, LinearAssortativeMating
from xftsim.reproduce import RecombinationMap
from xftsim.sim import NSimulation
from xftsim.io import save_simulation_checkpoint, load_simulation_checkpoint


def _make_sim(n=50, m=10, seed=42, mating=None):
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
    effects = AdditiveEffects.from_h2(m=m, h2=0.5, seed=seed)
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(effects))
    arch.add('Y.E', NoiseComponent(variance=0.5))
    arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])
    if mating is None:
        mating = RandomMating(offspring_per_pair=2)
    rmap = RecombinationMap.constant_map(m=m)
    return NSimulation(
        founder_haplotypes=hap,
        architecture=arch,
        mating_regime=mating,
        recombination_map=rmap,
        seed=seed,
    )


class TestCheckpointGeneration:
    def test_generation_preserved(self):
        """Checkpoint and reload preserves generation counter."""
        sim = _make_sim()
        sim.run(3)  # gens 0, 1, 2
        assert sim.generation == 2

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = os.path.join(tmpdir, 'ckpt')
            save_simulation_checkpoint(sim, checkpoint_dir)
            data = load_simulation_checkpoint(checkpoint_dir)
            assert data['generation'] == 2


class TestCheckpointHaplotypes:
    def test_haplotype_dimensions_preserved(self):
        """Haplotype shapes match after checkpoint roundtrip."""
        sim = _make_sim()
        sim.run(2)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = os.path.join(tmpdir, 'ckpt')
            save_simulation_checkpoint(sim, checkpoint_dir)
            data = load_simulation_checkpoint(checkpoint_dir)

            for gen in sim.haplotype_history:
                assert gen in data['haplotype_history']
                original = sim.haplotype_history[gen]
                loaded = data['haplotype_history'][gen]
                assert loaded.n == original.n
                assert loaded.m == original.m


class TestCheckpointPhenotypes:
    def test_phenotype_values_preserved(self):
        """Phenotype values match after checkpoint roundtrip."""
        sim = _make_sim()
        sim.run(2)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = os.path.join(tmpdir, 'ckpt')
            save_simulation_checkpoint(sim, checkpoint_dir)
            data = load_simulation_checkpoint(checkpoint_dir)

            for gen in sim.phenotype_history:
                original = sim.phenotype_history[gen]
                loaded = data['phenotype_history'][gen]
                for key in original.keys:
                    np.testing.assert_allclose(
                        original[key], loaded[key],
                        err_msg=f"Gen {gen}, key {key} mismatch"
                    )


class TestCheckpointPedigree:
    def test_pedigree_structure_preserved(self):
        """Pedigree indices match after checkpoint roundtrip."""
        sim = _make_sim()
        sim.run(3)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = os.path.join(tmpdir, 'ckpt')
            save_simulation_checkpoint(sim, checkpoint_dir)
            data = load_simulation_checkpoint(checkpoint_dir)

            for gen in sim.pedigree_history:
                original = sim.pedigree_history[gen]
                loaded = data['pedigree_history'][gen]
                np.testing.assert_array_equal(original.maternal_idx, loaded.maternal_idx)
                np.testing.assert_array_equal(original.paternal_idx, loaded.paternal_idx)
                assert original.parent_n == loaded.parent_n


class TestCheckpointContinue:
    def test_continue_from_checkpoint(self):
        """Simulation can be reconstructed and continued from checkpoint."""
        sim = _make_sim()
        sim.run(3)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = os.path.join(tmpdir, 'ckpt')
            save_simulation_checkpoint(sim, checkpoint_dir)
            restored = NSimulation.from_checkpoint(checkpoint_dir)
            restored.continue_run(2)
            assert restored.generation == 4  # 2 + 2 more


class TestCheckpointRetention:
    def test_checkpoint_with_retention(self):
        """Checkpoint after retention pruning: only retained gens saved."""
        sim = _make_sim()
        sim.retain_haplotypes = 1
        sim.retain_phenotypes = 1
        sim.run(4)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = os.path.join(tmpdir, 'ckpt')
            save_simulation_checkpoint(sim, checkpoint_dir)
            data = load_simulation_checkpoint(checkpoint_dir)
            # Only retained generations should be in checkpoint
            assert len(data['haplotype_history']) <= 2
            assert len(data['phenotype_history']) <= 2


class TestCheckpointAssortative:
    def test_assortative_mating_checkpoint(self):
        """Checkpoint preserves assortative mating configuration."""
        mating = LinearAssortativeMating(component_names=['Y'], r=0.5)
        sim = _make_sim(mating=mating)
        sim.run(2)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = os.path.join(tmpdir, 'ckpt')
            save_simulation_checkpoint(sim, checkpoint_dir)
            data = load_simulation_checkpoint(checkpoint_dir)
            restored_mating = data['mating_regime']
            assert isinstance(restored_mating, LinearAssortativeMating)
            assert restored_mating.r == 0.5
            assert restored_mating.component_names == ['Y']


class TestCheckpointErrors:
    def test_missing_directory_raises(self):
        """Loading from nonexistent directory raises error."""
        with pytest.raises((FileNotFoundError, OSError)):
            load_simulation_checkpoint('/tmp/nonexistent_xftsim_ckpt_12345')
