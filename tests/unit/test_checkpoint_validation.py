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

    def test_serialize_unsupported_regime_raises(self):
        """Unsupported regimes must fail loud at save time, not silently
        drop their parameters and explode on load.
        """
        from xftsim.io import _serialize_mating_regime

        class FancyMating:
            offspring_per_pair = 2

        with pytest.raises(ValueError, match="[Cc]annot serialize"):
            _serialize_mating_regime(FancyMating())

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

    def test_threshold_component_roundtrip(self, tmp_path):
        """Architecture with ThresholdComponent (liability-threshold model)
        survives save/load with source and threshold preserved.
        """
        from xftsim.io import save_architecture, load_architecture
        from xftsim.arch import ThresholdComponent
        m = 10
        arch = Architecture()
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y_liability', AggregationComponent('Y.G + Y.E'))
        arch.add('Y_diagnosis', ThresholdComponent(source='Y_liability',
                                                   threshold=1.5))
        save_architecture(arch, str(tmp_path / 'arch'))
        loaded = load_architecture(str(tmp_path / 'arch'))
        thr = [n.component for n in loaded._nodes
               if isinstance(n.component, ThresholdComponent)]
        assert len(thr) == 1
        assert thr[0].source == 'Y_liability'
        assert thr[0].threshold == 1.5

    def test_save_architecture_unsupported_component_raises(self, tmp_path):
        """An unrecognized ArchComponent subclass must fail loud at save
        time, and must not leave a partial architecture directory behind.
        """
        from xftsim.io import save_architecture
        from xftsim.arch import ArchComponent

        class FakeUnregisteredComponent(ArchComponent):
            name = "fake_unregistered"
            kind = "generative"
            accepts_grouping = False

            def compute(self, *args, **kwargs):
                return np.zeros(1)

        m = 10
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.fake', FakeUnregisteredComponent())
        out = str(tmp_path / 'arch')
        with pytest.raises(ValueError, match="[Cc]annot serialize component"):
            save_architecture(arch, out)
        assert not os.path.exists(out)


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

    def test_checkpoint_preserves_results(self, tmp_path):
        """Per-generation Statistic results round-trip through save/load."""
        from xftsim.io import save_simulation_checkpoint
        from xftsim.stats import SampleStatistics, GenerationResult
        n, m = 40, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        mate = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        sim = Simulation(hap, arch, mate, rmap, seed=42,
                          statistics=[SampleStatistics()])
        sim.run(3)
        # Sanity: original sim accumulated one result per generation
        assert len(sim.results) == 3
        save_simulation_checkpoint(sim, str(tmp_path / 'ckpt'))
        restored = Simulation.from_checkpoint(str(tmp_path / 'ckpt'),
                                               statistics=[SampleStatistics()])
        assert len(restored.results) == len(sim.results)
        for orig, got in zip(sim.results, restored.results):
            assert isinstance(got, GenerationResult)
            assert got.generation == orig.generation
            assert set(got.statistics) == set(orig.statistics)
            for name, val in orig.statistics.items():
                np.testing.assert_array_equal(got.statistics[name]['cov'],
                                              val['cov'])
                assert got.statistics[name]['keys'] == val['keys']

    def test_checkpoint_continue_run_appends_results(self, tmp_path):
        """After resume, new generations append to the loaded results list."""
        from xftsim.io import save_simulation_checkpoint
        from xftsim.stats import SampleStatistics
        n, m = 40, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        mate = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        sim = Simulation(hap, arch, mate, rmap, seed=42,
                          statistics=[SampleStatistics()])
        sim.run(2)
        save_simulation_checkpoint(sim, str(tmp_path / 'ckpt'))
        restored = Simulation.from_checkpoint(str(tmp_path / 'ckpt'),
                                               statistics=[SampleStatistics()])
        restored.continue_run(3)
        # 2 from before + 3 added on resume
        assert len(restored.results) == 5
        gens = [r.generation for r in restored.results]
        assert gens == [0, 1, 2, 3, 4]

    def test_save_unsupported_regime_raises_before_disk_writes(self, tmp_path):
        """Save must fail loud for an unsupported regime, and must not leave
        a partially written checkpoint directory behind. We use a duck-typed
        fake here because all the public mating regimes are now serializable;
        this guards the fail-loud contract for any future regime types.
        """
        from xftsim.io import save_simulation_checkpoint

        class FakeUnregisteredMating:
            offspring_per_pair = 2

            def mate(self, samples, rng=None, phenotypes=None):
                raise NotImplementedError

        n, m = 20, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        sim = Simulation(hap, arch, FakeUnregisteredMating(), rmap, seed=42)
        # No sim.run() — the regime's mate() is never called; we're testing
        # only that save refuses unsupported types loudly and atomically.
        ckpt = str(tmp_path / 'ckpt')
        with pytest.raises(ValueError, match="[Cc]annot serialize"):
            save_simulation_checkpoint(sim, ckpt)
        # No partial checkpoint directory should exist.
        assert not os.path.exists(ckpt)

    def test_batched_mating_checkpoint_roundtrip(self, tmp_path):
        """BatchedMating(RandomMating(...)) survives save/load and resumes.
        Exercises the recursive serialization path for nested regimes.
        """
        from xftsim.io import save_simulation_checkpoint
        from xftsim.mate import BatchedMating
        n, m = 40, 10
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        mate = BatchedMating(regime=RandomMating(offspring_per_pair=2),
                             max_batch_size=20)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        sim = Simulation(hap, arch, mate, rmap, seed=42)
        sim.run(2)
        save_simulation_checkpoint(sim, str(tmp_path / 'ckpt'))
        restored = Simulation.from_checkpoint(str(tmp_path / 'ckpt'))
        assert isinstance(restored.mating_regime, BatchedMating)
        assert restored.mating_regime.max_batch_size == 20
        assert isinstance(restored.mating_regime.regime, RandomMating)
        assert restored.mating_regime.regime.offspring_per_pair == 2
        # Continued run should work end-to-end.
        restored.continue_run(2)
        assert restored.generation == sim.generation + 2

    def test_general_assortative_mating_helper_roundtrip(self):
        """Helper-level (de)serialization for GeneralAssortativeMating.

        ``GeneralAssortativeMating.__init__`` imports ``hexaly``, which is
        a heavy optional dep, so we bypass __init__ via ``__new__`` and set
        attributes manually to get a real instance whose isinstance check
        passes. Round-tripping through the JSON-friendly dict and back
        must preserve every field; an end-to-end test that actually
        constructs the regime via __init__ would be gated on hexaly.
        """
        from xftsim.io import (
            _serialize_mating_regime, _deserialize_mating_regime,
        )
        from xftsim.mate import GeneralAssortativeMating

        regime = GeneralAssortativeMating.__new__(GeneralAssortativeMating)
        regime.component_names = ['Y1', 'Y2']
        regime.cross_corr = np.array([[0.3, 0.1], [0.1, 0.4]])
        regime.offspring_per_pair = 3
        regime.solver_params = {
            'time_limit': 60, 'nb_threads': 2, 'tolerance': 1e-5,
            'verbosity': 0, 'time_between_displays': 15,
            'termination_interval': 15,
        }

        config = _serialize_mating_regime(regime)
        assert config['type'] == 'GeneralAssortativeMating'
        assert config['component_names'] == ['Y1', 'Y2']
        np.testing.assert_array_equal(
            np.asarray(config['cross_corr']), regime.cross_corr,
        )
        assert config['offspring_per_pair'] == 3
        assert config['solver_params']['time_limit'] == 60

        # Round-trip through JSON to catch any non-JSON-serializable values.
        import json
        roundtripped_config = json.loads(json.dumps(config))

        # Deserialize requires hexaly (since it constructs via __init__).
        # If hexaly isn't available, just verify the dict is recoverable.
        try:
            import hexaly.optimizer  # noqa: F401
        except ImportError:
            pytest.skip("hexaly not installed; skipping deserialize half")

        restored = _deserialize_mating_regime(roundtripped_config)
        assert isinstance(restored, GeneralAssortativeMating)
        assert restored.component_names == regime.component_names
        np.testing.assert_array_equal(restored.cross_corr, regime.cross_corr)
        assert restored.offspring_per_pair == regime.offspring_per_pair

    def test_batched_general_assortative_helper_recursive(self):
        """Helper-level: BatchedMating wrapping a (faked) GeneralAssortativeMating
        round-trips recursively through the JSON-friendly dict.
        """
        from xftsim.io import _serialize_mating_regime
        from xftsim.mate import BatchedMating, GeneralAssortativeMating

        inner = GeneralAssortativeMating.__new__(GeneralAssortativeMating)
        inner.component_names = ['A', 'B', 'C']
        inner.cross_corr = np.eye(3) * 0.5
        inner.offspring_per_pair = 2
        inner.solver_params = {'time_limit': 30, 'nb_threads': 1,
                               'tolerance': 1e-4, 'verbosity': 0,
                               'time_between_displays': 5,
                               'termination_interval': 5}
        outer = BatchedMating(regime=inner, max_batch_size=500)

        config = _serialize_mating_regime(outer)
        assert config['type'] == 'BatchedMating'
        assert config['max_batch_size'] == 500
        assert config['regime']['type'] == 'GeneralAssortativeMating'
        assert config['regime']['component_names'] == ['A', 'B', 'C']
        np.testing.assert_array_equal(
            np.asarray(config['regime']['cross_corr']), inner.cross_corr,
        )

    def test_checkpoint_resume_rng_is_deterministic(self, tmp_path):
        """``run(N)`` and ``run(K) → save → from_checkpoint → continue_run(N-K)``
        must consume ``self.rng`` identically — same algorithm, same state
        key, same position, same Gaussian-cache state at the end.

        This is the strongest determinism guarantee the simulator currently
        supports for save/resume. We do *not* assert byte-equal haplotypes /
        phenotypes / pedigrees because meiosis (in xftsim/reproduce.py)
        currently uses ``np.random`` globally inside a numba ``prange``
        kernel, which is racey and non-deterministic even within a single
        process. That is a pre-existing simulator-level issue independent
        of checkpointing — fixing it would require routing ``self.rng``
        through the meiosis kernel and removing the parallel reduction (or
        using per-thread RNG).

        What this test *does* prove: every random draw from ``self.rng``
        (NoiseComponent draws, mate-assignment draws, stats draws if any)
        happens in the same order with the same values whether you stop
        at gen K and resume, or run straight through to gen N. So any
        downstream reproducibility you build on top of ``self.rng`` (e.g.
        re-running a checkpointed sim with a known seed under a future
        deterministic-meiosis fix) survives the round-trip.
        """
        from xftsim.io import save_simulation_checkpoint
        from xftsim.stats import SampleStatistics

        N, K = 5, 2

        def _build(seed):
            n, m = 40, 10
            hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
            eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=seed)
            arch = Architecture()
            arch.add('Y.G', GeneticComponent(eff))
            arch.add('Y.E', NoiseComponent(variance=0.5))
            arch.add('Y', AggregationComponent('Y.G + Y.E'))
            mate = LinearAssortativeMating(component_names=['Y'], r=0.3,
                                           offspring_per_pair=2)
            rmap = RecombinationMap.constant_map(m=m, p=0.5)
            return Simulation(hap, arch, mate, rmap, seed=seed,
                               statistics=[SampleStatistics()])

        # Path A: straight run(N).
        sim_a = _build(seed=42)
        sim_a.run(N)

        # Path B: run(K) → save → from_checkpoint → continue_run(N-K).
        sim_b = _build(seed=42)
        sim_b.run(K)
        save_simulation_checkpoint(sim_b, str(tmp_path / 'ckpt'))
        sim_b = Simulation.from_checkpoint(str(tmp_path / 'ckpt'),
                                            statistics=[SampleStatistics()])
        sim_b.continue_run(N - K)

        assert sim_a.generation == sim_b.generation == N - 1
        # Both paths should have produced one result per generation.
        assert len(sim_a.results) == len(sim_b.results) == N

        # The actual determinism assertion: self.rng round-trips identically.
        state_a = sim_a.rng.get_state()
        state_b = sim_b.rng.get_state()
        assert state_a[0] == state_b[0], "RNG algorithm differs"
        np.testing.assert_array_equal(state_a[1], state_b[1])  # state key
        assert state_a[2] == state_b[2], "RNG position differs"
        assert state_a[3] == state_b[3], "Gaussian-cache flag differs"
        assert state_a[4] == state_b[4], "Cached Gaussian differs"

    def test_checkpoint_missing_results_back_compat(self, tmp_path):
        """Old checkpoints without results.pkl still load (empty results)."""
        from xftsim.io import save_simulation_checkpoint
        n, m = 20, 10
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
        ckpt = str(tmp_path / 'ckpt')
        save_simulation_checkpoint(sim, ckpt)
        # Simulate an older checkpoint by removing the new file.
        os.remove(os.path.join(ckpt, 'results.pkl'))
        restored = Simulation.from_checkpoint(ckpt)
        assert restored.results == []
