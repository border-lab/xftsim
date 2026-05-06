"""
Unit tests for I/O error handling.

Tests:
1. load_haplotypes_npz with nonexistent file
2. load_phenotypes_npz with nonexistent file
3. load_effects_npz with nonexistent file
4. load_architecture with nonexistent dir
5. load_simulation_checkpoint with nonexistent dir
6. save_simulation_checkpoint creates expected directory structure
7. _deserialize_mating_regime with unknown type
"""
import numpy as np
import pytest
import tempfile
import os
import shutil
import json

from xftsim.io import (
    save_haplotypes_npz, load_haplotypes_npz,
    save_phenotypes_npz, load_phenotypes_npz,
    save_effects_npz, load_effects_npz,
    save_architecture, load_architecture,
    save_simulation_checkpoint, load_simulation_checkpoint,
    _deserialize_mating_regime,
)
from xftsim.effect import AdditiveEffects
from xftsim.arch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.mate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.sim import NSimulation

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestLoadNonexistentFile:
    def test_load_haplotypes_nonexistent(self):
        with pytest.raises((FileNotFoundError, OSError)):
            load_haplotypes_npz('/tmp/nonexistent_haplotypes.npz')

    def test_load_phenotypes_nonexistent(self):
        with pytest.raises((FileNotFoundError, OSError)):
            load_phenotypes_npz('/tmp/nonexistent_phenotypes.npz')

    def test_load_effects_nonexistent(self):
        with pytest.raises((FileNotFoundError, OSError)):
            load_effects_npz('/tmp/nonexistent_effects.npz')

    def test_load_architecture_nonexistent(self):
        with pytest.raises((FileNotFoundError, OSError)):
            load_architecture('/tmp/nonexistent_architecture_dir')

    def test_load_checkpoint_nonexistent(self):
        with pytest.raises((FileNotFoundError, OSError)):
            load_simulation_checkpoint('/tmp/nonexistent_checkpoint_dir')


class TestCheckpointDirectoryStructure:
    def test_checkpoint_creates_directories(self):
        """save_simulation_checkpoint should create expected subdirectories."""
        hap = TestSimulation.founder_haplotypes(n=50, m=10, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])

        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=RandomMating(offspring_per_pair=2),
            recombination_map=RecombinationMap.constant_map(m=10),
            seed=42,
        )
        sim.run(2)

        tmpdir = tempfile.mkdtemp()
        try:
            save_simulation_checkpoint(sim, tmpdir)
            assert os.path.exists(os.path.join(tmpdir, 'meta.json'))
            assert os.path.isdir(os.path.join(tmpdir, 'architecture'))
            assert os.path.isdir(os.path.join(tmpdir, 'haplotypes'))
            assert os.path.isdir(os.path.join(tmpdir, 'phenotypes'))
            assert os.path.isdir(os.path.join(tmpdir, 'pedigrees'))
            assert os.path.exists(os.path.join(tmpdir, 'rng_state.npz'))
            assert os.path.exists(os.path.join(tmpdir, 'history_keys.npz'))
        finally:
            shutil.rmtree(tmpdir)


class TestDeserializeMatingRegime:
    def test_unknown_type_raises(self):
        """Unknown mating regime type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown mating"):
            _deserialize_mating_regime({'type': 'FancyMating', 'params': {}})

    def test_random_mating_deserialize(self):
        """RandomMating should deserialize correctly."""
        result = _deserialize_mating_regime({
            'type': 'RandomMating',
            'offspring_per_pair': 3,
        })
        assert isinstance(result, RandomMating)
        assert result.offspring_per_pair == 3


class TestRoundtripEffects:
    def test_additive_effects_roundtrip(self):
        """Save and load additive effects should preserve values."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=20, seed=42)
        tmpfile = tempfile.mktemp(suffix='.npz')
        try:
            save_effects_npz(eff, tmpfile)
            loaded = load_effects_npz(tmpfile)
            np.testing.assert_array_equal(loaded.effects, eff.effects)
            assert loaded.standardized == eff.standardized
        finally:
            if os.path.exists(tmpfile):
                os.remove(tmpfile)
