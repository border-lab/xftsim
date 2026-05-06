"""
Integration tests for the full checkpoint save/load/continue cycle.

Tests:
1. Save checkpoint, load, continue_run, verify phenotypes finite
2. Checkpoint with filters and statistics
3. Checkpoint preserves generation number
4. Multiple save/load cycles
"""
import numpy as np
import pytest
import os

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray
from xftsim.arch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.effect import AdditiveEffects
from xftsim.sim import NSimulation
from xftsim.mate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.stats import SampleStatistics
from xftsim.filters import TrioFilter
from xftsim.io import save_simulation_checkpoint

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestCheckpointCycle:
    def _make_sim(self, seed=42):
        """Create a basic simulation."""
        n, m = 50, 5
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=seed)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        mate = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        return NSimulation(hap, arch, mate, rmap, seed=seed)

    def test_save_load_continue(self, tmp_path):
        """Full cycle: run → save → load → continue → verify."""
        sim = self._make_sim()
        sim.run(3)
        gen_before = sim.generation
        assert gen_before == 2

        # Save checkpoint
        ckpt_dir = str(tmp_path / 'checkpoint')
        save_simulation_checkpoint(sim, ckpt_dir)

        # Load and continue
        sim2 = NSimulation.from_checkpoint(
            ckpt_dir,
            mating_regime=RandomMating(offspring_per_pair=2),
            recombination_map=RecombinationMap.constant_map(m=5, p=0.5),
        )
        assert sim2.generation == 2
        sim2.continue_run(3)
        assert sim2.generation == 5
        pheno = sim2.phenotype_history[5]
        assert np.all(np.isfinite(pheno['Y']))

    def test_checkpoint_preserves_generation(self, tmp_path):
        """Generation number should survive checkpoint."""
        sim = self._make_sim()
        sim.run(5)
        assert sim.generation == 4

        ckpt_dir = str(tmp_path / 'ckpt2')
        save_simulation_checkpoint(sim, ckpt_dir)

        sim2 = NSimulation.from_checkpoint(
            ckpt_dir,
            mating_regime=RandomMating(offspring_per_pair=2),
            recombination_map=RecombinationMap.constant_map(m=5, p=0.5),
        )
        assert sim2.generation == 4

    def test_checkpoint_with_stats(self, tmp_path):
        """Checkpoint cycle with statistics should work."""
        sim = self._make_sim()
        sim.statistics = [SampleStatistics()]
        sim.run(3)

        ckpt_dir = str(tmp_path / 'ckpt_stats')
        save_simulation_checkpoint(sim, ckpt_dir)

        sim2 = NSimulation.from_checkpoint(
            ckpt_dir,
            mating_regime=RandomMating(offspring_per_pair=2),
            recombination_map=RecombinationMap.constant_map(m=5, p=0.5),
            statistics=[SampleStatistics()],
        )
        sim2.continue_run(2)
        assert sim2.generation == 4
        assert len(sim2.results) >= 2

    def test_multiple_checkpoint_cycles(self, tmp_path):
        """Multiple save/load cycles should work."""
        sim = self._make_sim()
        sim.run(2)

        for i in range(3):
            ckpt_dir = str(tmp_path / f'ckpt_{i}')
            save_simulation_checkpoint(sim, ckpt_dir)
            sim = NSimulation.from_checkpoint(
                ckpt_dir,
                mating_regime=RandomMating(offspring_per_pair=2),
                recombination_map=RecombinationMap.constant_map(m=5, p=0.5),
            )
            sim.continue_run(2)

        assert sim.generation >= 6
        pheno = sim.phenotype_history[sim.generation]
        assert np.all(np.isfinite(pheno['Y']))
