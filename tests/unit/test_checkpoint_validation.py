"""
Unit tests for checkpoint validation and IO error paths.

Tests:
1. _deserialize_mating_regime unknown type
2. Simulation.from_checkpoint missing mating regime
3. Simulation.from_checkpoint missing recombination map
4. Simulation._validate effect dimension mismatch
5. IO: save/load architecture roundtrip with CNoiseComponent
6. IO: save/load architecture roundtrip with sibling components
"""
import numpy as np
import pytest
import json
import os

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray, PhenotypeArray, PedigreeArray
from xftsim.arch import (
    Architecture, GeneticComponent, NoiseComponent, AggregationComponent,
    CNoiseComponent, MVGeneticComponent, MotherComponent, FatherComponent,
)
from xftsim.effect import AdditiveEffects, MultivariateEffects
from xftsim.mate import RandomMating, LinearAssortativeMating
from xftsim.sim import Simulation
from xftsim.reproduce import RecombinationMap

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


# ── _deserialize_mating_regime ───────────────────────────────────────────────

class TestDeserializeMatingRegime:
    def test_unknown_type_raises(self):
        """Unknown mating regime type should raise ValueError."""
        from xftsim.io import _deserialize_mating_regime
        with pytest.raises(ValueError, match="Unknown mating regime"):
            _deserialize_mating_regime({'type': 'FancyMating', 'offspring_per_pair': 2})

    def test_random_mating_roundtrip(self):
        """RandomMating serialization roundtrip."""
        from xftsim.io import _serialize_mating_regime, _deserialize_mating_regime
        mate = RandomMating(offspring_per_pair=3)
        config = _serialize_mating_regime(mate)
        restored = _deserialize_mating_regime(config)
        assert isinstance(restored, RandomMating)
        assert restored.offspring_per_pair == 3

    def test_assortative_mating_roundtrip(self):
        """LinearAssortativeMating serialization roundtrip."""
        from xftsim.io import _serialize_mating_regime, _deserialize_mating_regime
        mate = LinearAssortativeMating(
            component_names=['Y'],
            r=0.3,
            offspring_per_pair=2,
        )
        config = _serialize_mating_regime(mate)
        restored = _deserialize_mating_regime(config)
        assert isinstance(restored, LinearAssortativeMating)
        assert restored.r == 0.3
        assert restored.component_names == ['Y']


# ── Simulation.from_checkpoint ──────────────────────────────────────────────

class TestFromCheckpointValidation:
    def _make_and_save_checkpoint(self, tmp_path, include_mating=True, include_rmap=True):
        """Build a simulation and save a checkpoint."""
        from xftsim.io import save_simulation_checkpoint
        n, m = 20, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        mate = RandomMating(offspring_per_pair=2) if include_mating else None
        rmap = RecombinationMap.constant_map(m=m, p=0.5) if include_rmap else None
        if mate is not None and rmap is not None:
            sim = Simulation(hap, arch, mate, rmap, seed=42)
            sim.run(2)
            save_simulation_checkpoint(sim, str(tmp_path / 'ckpt'))
        return str(tmp_path / 'ckpt')

    def test_from_checkpoint_works(self, tmp_path):
        """Basic from_checkpoint should work when checkpoint is complete."""
        ckpt_dir = self._make_and_save_checkpoint(tmp_path)
        sim = Simulation.from_checkpoint(ckpt_dir)
        assert sim.generation >= 1

    def test_from_checkpoint_no_mating_in_checkpoint(self, tmp_path):
        """from_checkpoint with no mating regime should raise."""
        # Create a valid checkpoint first, then tamper with it
        ckpt_dir = self._make_and_save_checkpoint(tmp_path)
        # Remove the mating key entirely so load returns None
        meta_path = os.path.join(ckpt_dir, 'meta.json')
        with open(meta_path) as f:
            meta = json.load(f)
        del meta['mating']
        with open(meta_path, 'w') as f:
            json.dump(meta, f)

        with pytest.raises(ValueError, match="[Nn]o mating regime"):
            Simulation.from_checkpoint(ckpt_dir)

    def test_from_checkpoint_override_mating(self, tmp_path):
        """from_checkpoint with explicit mating regime overrides saved one."""
        ckpt_dir = self._make_and_save_checkpoint(tmp_path)
        new_mate = RandomMating(offspring_per_pair=4)
        sim = Simulation.from_checkpoint(ckpt_dir, mating_regime=new_mate)
        assert sim.mating_regime.offspring_per_pair == 4

    def test_from_checkpoint_no_rmap_in_checkpoint(self, tmp_path):
        """from_checkpoint with no recombination map should raise."""
        ckpt_dir = self._make_and_save_checkpoint(tmp_path)
        # Remove recombination map file so load returns None for it
        rmap_path = os.path.join(ckpt_dir, 'recombination_map.npz')
        if os.path.exists(rmap_path):
            os.remove(rmap_path)

        with pytest.raises((ValueError, FileNotFoundError)):
            Simulation.from_checkpoint(ckpt_dir)


# ── Simulation._validate ────────────────────────────────────────────────────

class TestSimulationValidation:
    def test_validate_dimension_mismatch_raises(self):
        """Architecture with wrong m should raise at run() time."""
        n, m = 20, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        # Create effects with wrong dimension
        eff_wrong = AdditiveEffects.from_h2(h2=0.5, m=m + 5, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff_wrong))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        mate = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        sim = Simulation(hap, arch, mate, rmap, seed=42)
        with pytest.raises(ValueError, match="[Ee]ffect dimension mismatch"):
            sim.run(1)

    def test_validate_mv_dimension_mismatch_raises(self):
        """MVGeneticComponent with wrong m should raise at run() time."""
        n, m = 20, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        # Create multivariate effects with wrong dimension
        eff_wrong = MultivariateEffects.from_h2_rg(
            h2=[0.5, 0.5], rg=0.5, m=m + 3, seed=42,
        )
        arch = Architecture()
        arch.add(('Y1.G', 'Y2.G'), MVGeneticComponent(eff_wrong))
        arch.add('Y1.E', NoiseComponent(variance=0.5))
        arch.add('Y2.E', NoiseComponent(variance=0.5))
        arch.add('Y1', AggregationComponent('Y1.G + Y1.E'))
        arch.add('Y2', AggregationComponent('Y2.G + Y2.E'))
        mate = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        sim = Simulation(hap, arch, mate, rmap, seed=42)
        with pytest.raises(ValueError, match="[Ee]ffect dimension mismatch"):
            sim.run(1)


# ── IO roundtrip for complex architectures ───────────────────────────────────

class TestArchitectureRoundtripComplex:
    def test_cnoise_architecture_roundtrip(self, tmp_path):
        """Architecture with CNoiseComponent survives save/load."""
        from xftsim.io import save_architecture, load_architecture
        m = 10
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        cov = np.array([[0.3, 0.1], [0.1, 0.2]])
        arch = Architecture()
        arch.add(('Y1.G', 'Y2.G'), MVGeneticComponent(
            MultivariateEffects.from_h2_rg(h2=[0.5, 0.5], rg=0.3, m=m, seed=42)
        ))
        arch.add(('Y1.E', 'Y2.E'), CNoiseComponent(cov))
        arch.add('Y1', AggregationComponent('Y1.G + Y1.E'))
        arch.add('Y2', AggregationComponent('Y2.G + Y2.E'))
        save_architecture(arch, str(tmp_path / 'arch'))
        loaded = load_architecture(str(tmp_path / 'arch'))
        assert len(loaded._nodes) == len(arch._nodes)
        # Verify all outputs match
        orig_outputs = sorted(out for n in arch._nodes for out in n.outputs)
        loaded_outputs = sorted(out for n in loaded._nodes for out in n.outputs)
        assert orig_outputs == loaded_outputs

    def test_vt_architecture_roundtrip(self, tmp_path):
        """Architecture with VT (MotherComponent) survives save/load."""
        from xftsim.io import save_architecture, load_architecture
        m = 10
        arch = Architecture()
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.VT', MotherComponent('Y'))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.VT + Y.E'))
        save_architecture(arch, str(tmp_path / 'arch'))
        loaded = load_architecture(str(tmp_path / 'arch'))
        assert len(loaded._nodes) == 4
        # Find the VT node
        vt_nodes = [n for n in loaded._nodes
                    if isinstance(n.component, MotherComponent)]
        assert len(vt_nodes) == 1
        assert vt_nodes[0].component.phenotype_name == 'Y'

    def test_architecture_with_many_components(self, tmp_path):
        """Architecture with genetic + noise + VT + aggregation."""
        from xftsim.io import save_architecture, load_architecture
        m = 10
        arch = Architecture()
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.3))
        arch.add('Y.VT_m', MotherComponent('Y'))
        arch.add('Y.VT_f', FatherComponent('Y'))
        arch.add('Y', AggregationComponent('Y.G + Y.E + Y.VT_m + Y.VT_f'))
        save_architecture(arch, str(tmp_path / 'arch'))
        loaded = load_architecture(str(tmp_path / 'arch'))
        assert len(loaded._nodes) == len(arch._nodes)


# ── Simulation checkpoint full roundtrip ─────────────────────────────────────

class TestSimulationCheckpointFull:
    def test_checkpoint_preserves_generation(self, tmp_path):
        """Checkpoint preserves generation counter."""
        from xftsim.io import save_simulation_checkpoint
        n, m = 40, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        mate = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        sim = Simulation(hap, arch, mate, rmap, seed=42)
        sim.run(3)
        save_simulation_checkpoint(sim, str(tmp_path / 'ckpt'))
        restored = Simulation.from_checkpoint(str(tmp_path / 'ckpt'))
        assert restored.generation == sim.generation

    def test_checkpoint_continue_run(self, tmp_path):
        """Checkpoint → from_checkpoint → continue_run works."""
        from xftsim.io import save_simulation_checkpoint
        n, m = 40, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        mate = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        sim = Simulation(hap, arch, mate, rmap, seed=42)
        sim.run(2)
        save_simulation_checkpoint(sim, str(tmp_path / 'ckpt'))
        restored = Simulation.from_checkpoint(str(tmp_path / 'ckpt'))
        restored.continue_run(3)
        assert restored.generation == sim.generation + 3
        # All phenotypes should be finite
        pheno = restored.phenotype_history[restored.generation]
        assert np.all(np.isfinite(pheno['Y']))

    def test_assortative_mating_checkpoint(self, tmp_path):
        """LinearAssortativeMating survives checkpoint roundtrip."""
        from xftsim.io import save_simulation_checkpoint
        n, m = 40, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        mate = LinearAssortativeMating(component_names=['Y'], r=0.5, offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        sim = Simulation(hap, arch, mate, rmap, seed=42)
        sim.run(2)
        save_simulation_checkpoint(sim, str(tmp_path / 'ckpt'))
        restored = Simulation.from_checkpoint(str(tmp_path / 'ckpt'))
        assert isinstance(restored.mating_regime, LinearAssortativeMating)
        assert restored.mating_regime.r == 0.5
