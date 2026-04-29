"""
Unit tests for GRG-aware checkpointing.

Verifies that ``save_simulation_checkpoint`` persists GRG-backed founder
haplotypes natively (as a ``.grg`` file plus metadata sidecar) instead of
materializing to dense, and that ``load_simulation_checkpoint`` /
``NSimulation.from_checkpoint`` round-trip them back into a
``GraphHaplotypeOperator``.

Skipped entirely when any of pygrgl / msprime / the ``grg`` CLI binary are
unavailable, since the test fixture is built via msprime + grg-convert.
"""
import os
import shutil

import numpy as np
import pytest

pygrgl = pytest.importorskip("pygrgl")
pytest.importorskip("msprime")
if shutil.which("grg") is None:
    pytest.skip("`grg` CLI not on PATH; activate the xftsim venv to run these tests",
                allow_module_level=True)

from xftsim.founders import founder_haplotypes_from_msprime_grg
from xftsim.io import save_simulation_checkpoint, load_simulation_checkpoint
from xftsim.narch import (
    Architecture, GeneticComponent, NoiseComponent, AggregationComponent,
)
from xftsim.neffect import AdditiveEffects
from xftsim.nmate import RandomMating
from xftsim.nsim import NSimulation
from xftsim.reproduce import RecombinationMap
from xftsim.struct import GraphHaplotypeOperator, DenseHaplotypeArray


def _make_grg_founder():
    """Build a tiny synthetic GRG founder set (n=20, ~hundreds of variants)."""
    return founder_haplotypes_from_msprime_grg(
        n=20, sequence_length=10_000, mutation_rate=1e-6,
    )


def _make_sim_with_grg_founder(seed=42):
    hap = _make_grg_founder()
    m = hap.m
    eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=seed)
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(eff))
    arch.add('Y.E', NoiseComponent(variance=0.5))
    arch.add('Y', AggregationComponent('Y.G + Y.E'))
    mate = RandomMating(offspring_per_pair=2)
    rmap = RecombinationMap.constant_map(m=m, p=0.5)
    # retain_haplotypes=10 keeps the gen-0 GRG in history through several
    # offspring generations so we actually exercise the GRG-save path.
    sim = NSimulation(hap, arch, mate, rmap, seed=seed,
                      retain_haplotypes=10)
    return sim


class TestGRGCheckpoint:
    def test_grg_founder_persisted_natively(self, tmp_path):
        """A GRG founder must be saved as a .grg + metadata sidecar, not as
        a materialized-dense .npz.
        """
        sim = _make_sim_with_grg_founder()
        sim.run(2)
        ckpt = str(tmp_path / 'ckpt')
        save_simulation_checkpoint(sim, ckpt)

        hap_dir = os.path.join(ckpt, 'haplotypes')
        assert os.path.exists(os.path.join(hap_dir, 'gen_0.grg'))
        assert os.path.exists(os.path.join(hap_dir, 'gen_0.grg.meta.npz'))
        # Critically, NO dense materialization was written for gen 0.
        assert not os.path.exists(os.path.join(hap_dir, 'gen_0.npz'))

    def test_grg_founder_roundtrip(self, tmp_path):
        """Round-trip preserves type (still GraphHaplotypeOperator),
        dimensions, sample/variant metadata, and allele frequencies.
        """
        sim = _make_sim_with_grg_founder()
        sim.run(2)
        original = sim.haplotype_history[0]
        original_af = original.recompute_af().copy()

        ckpt = str(tmp_path / 'ckpt')
        save_simulation_checkpoint(sim, ckpt)
        loaded = load_simulation_checkpoint(ckpt)

        restored = loaded['haplotype_history'][0]
        assert isinstance(restored, GraphHaplotypeOperator)
        assert restored.n == original.n
        assert restored.m == original.m
        assert restored.generation == original.generation
        np.testing.assert_array_equal(restored.samples.iid, original.samples.iid)
        np.testing.assert_array_equal(restored.samples.fid, original.samples.fid)
        np.testing.assert_array_equal(restored.samples.sex, original.samples.sex)
        np.testing.assert_array_equal(restored.variants.vid, original.variants.vid)
        np.testing.assert_array_equal(restored.variants.chrom, original.variants.chrom)
        # Allele frequencies must agree to numerical precision — proves the
        # GRG itself was correctly preserved, not just the metadata.
        np.testing.assert_allclose(restored.recompute_af(), original_af)

    def test_grg_founder_continue_run(self, tmp_path):
        """from_checkpoint → continue_run works with a GRG founder."""
        sim = _make_sim_with_grg_founder()
        sim.run(2)
        ckpt = str(tmp_path / 'ckpt')
        save_simulation_checkpoint(sim, ckpt)

        restored = NSimulation.from_checkpoint(ckpt)
        # Founder should still be GRG, not materialized dense.
        assert isinstance(restored.haplotype_history[0], GraphHaplotypeOperator)
        restored.continue_run(2)
        assert restored.generation == sim.generation + 2
        # Offspring generations are dense (GRG.meiosis returns dense).
        assert isinstance(
            restored.haplotype_history[restored.generation], DenseHaplotypeArray,
        )
