"""
Unit tests for NSimulation.from_checkpoint error paths.

Tests:
1. Nonexistent directory raises
2. Valid checkpoint roundtrip restores generation
3. Override mating_regime from checkpoint
"""
import numpy as np
import pytest
import tempfile
import shutil

from xftsim.neffect import AdditiveEffects
from xftsim.narch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.nmate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.nsim import NSimulation
from xftsim.io import save_simulation_checkpoint

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


def _make_sim(seed=42):
    n, m = 50, 10
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
    eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=seed)
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(eff))
    arch.add('Y.E', NoiseComponent(variance=0.5))
    arch.add('Y', AggregationComponent('Y.G + Y.E'))
    return NSimulation(
        founder_haplotypes=hap, architecture=arch,
        mating_regime=RandomMating(offspring_per_pair=2),
        recombination_map=RecombinationMap.constant_map(m=m),
        seed=seed, retain_phenotypes=10, retain_haplotypes=10,
    )


class TestFromCheckpointErrors:
    def test_nonexistent_dir_raises(self):
        """from_checkpoint with nonexistent directory should raise."""
        with pytest.raises((FileNotFoundError, OSError)):
            NSimulation.from_checkpoint('/tmp/nonexistent_xftsim_checkpoint_xyz')

    def test_checkpoint_roundtrip_restores_generation(self):
        """Save and reload should preserve generation counter."""
        sim = _make_sim()
        sim.run(3)
        assert sim.generation == 2

        tmpdir = tempfile.mkdtemp()
        try:
            save_simulation_checkpoint(sim, tmpdir)
            loaded = NSimulation.from_checkpoint(tmpdir)
            assert loaded.generation == 2
        finally:
            shutil.rmtree(tmpdir)

    def test_checkpoint_override_mating_regime(self):
        """from_checkpoint with explicit mating_regime should use the override."""
        sim = _make_sim()
        sim.run(2)

        tmpdir = tempfile.mkdtemp()
        try:
            save_simulation_checkpoint(sim, tmpdir)
            new_rm = RandomMating(offspring_per_pair=4)
            loaded = NSimulation.from_checkpoint(tmpdir, mating_regime=new_rm)
            assert loaded.mating_regime.offspring_per_pair == 4
        finally:
            shutil.rmtree(tmpdir)

    def test_checkpoint_continue_run(self):
        """Loaded checkpoint should be able to continue_run."""
        sim = _make_sim()
        sim.run(2)

        tmpdir = tempfile.mkdtemp()
        try:
            save_simulation_checkpoint(sim, tmpdir)
            loaded = NSimulation.from_checkpoint(tmpdir)
            loaded.continue_run(2)
            assert loaded.generation == 3
            assert np.all(np.isfinite(loaded.phenotypes['Y']))
        finally:
            shutil.rmtree(tmpdir)
