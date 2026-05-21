"""
Unit tests for meiosis RNG determinism.

The meiosis kernel used to call ``np.random.binomial`` against numba's
internal RNG with no path back to the simulation's seeded
``self.rng``. Two ``Simulation(seed=42).run(N)`` calls therefore produced
*different* haplotypes, defeating the simulation seed entirely. These
tests pin the new contract: given a seeded master rng (the simulation's
``self.rng``, or any ``np.random.RandomState`` for direct callers), the
offspring genotypes are byte-identical across runs — both for the dense
kernel and the GRG-native path.
"""
import os
import sys
import tempfile

import numpy as np
import pytest

from xftsim.arch import (
    Architecture, GeneticComponent, NoiseComponent, AggregationComponent,
)
from xftsim.effect import AdditiveEffects
from xftsim.mate import RandomMating, MateAssignment
from xftsim.reproduce import (
    RecombinationMap, meiosis, _meiosis_pair_seeded, _spawn_meiosis_seeds,
)
from xftsim.sim import Simulation
from xftsim.struct import (
    SampleMeta, VariantMeta, DenseHaplotypeArray, GraphHaplotypeOperator,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


def _make_parents(n, m, seed):
    """Build a parental DenseHaplotypeArray with fixed, seeded genotypes."""
    rng = np.random.RandomState(seed)
    geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
    sm = SampleMeta(iid=np.arange(n))
    vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
    return DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)


def _make_assignment(n_offspring, mat_idx, pat_idx):
    sm = SampleMeta(iid=np.arange(n_offspring), generation=1)
    return MateAssignment(
        offspring_samples=sm,
        maternal_idx=np.asarray(mat_idx, dtype=np.int64),
        paternal_idx=np.asarray(pat_idx, dtype=np.int64),
    )


# ── Low-level helpers ──────────────────────────────────────────────────────

class TestSpawnSeeds:
    def test_same_master_rng_produces_same_seeds(self):
        """Two RandomStates seeded identically produce identical seed arrays."""
        a = _spawn_meiosis_seeds(np.random.RandomState(42), 100)
        b = _spawn_meiosis_seeds(np.random.RandomState(42), 100)
        np.testing.assert_array_equal(a, b)

    def test_consumes_one_draw_from_rng(self):
        """_spawn_meiosis_seeds should consume exactly one rng.randint draw.

        Pinning this means callers (the dense and GRG meiosis paths) can
        rely on identical rng-state advance per meiosis call.
        """
        rng1 = np.random.RandomState(42)
        _spawn_meiosis_seeds(rng1, 50)
        post_spawn = rng1.randint(0, 2 ** 31 - 1)

        rng2 = np.random.RandomState(42)
        rng2.randint(0, 2 ** 31 - 1)  # one direct draw
        post_direct = rng2.randint(0, 2 ** 31 - 1)

        assert post_spawn == post_direct


class TestMeiosisPairSeeded:
    """`_meiosis_pair_seeded(p, seed)` must be deterministic and return
    *two* phase vectors drawn back-to-back from the seeded stream.
    """

    def test_identical_across_calls(self):
        p = np.full(50, 0.3, dtype=np.float64)
        m1, p1 = _meiosis_pair_seeded(p, np.uint32(123))
        m2, p2 = _meiosis_pair_seeded(p, np.uint32(123))
        np.testing.assert_array_equal(m1, m2)
        np.testing.assert_array_equal(p1, p2)

    def test_mat_and_pat_differ(self):
        """Two consecutive draws from the same stream should differ — they
        consume different positions of the underlying RNG.
        """
        p = np.full(200, 0.5, dtype=np.float64)
        mat, pat = _meiosis_pair_seeded(p, np.uint32(7))
        # Vanishingly unlikely to be equal for m=200, p=0.5.
        assert not np.array_equal(mat, pat)

    def test_different_seeds_differ(self):
        p = np.full(200, 0.5, dtype=np.float64)
        m1, _ = _meiosis_pair_seeded(p, np.uint32(1))
        m2, _ = _meiosis_pair_seeded(p, np.uint32(2))
        assert not np.array_equal(m1, m2)


# ── Dense kernel ───────────────────────────────────────────────────────────

class TestDenseMeiosisDeterminism:
    def test_meiosis_function_deterministic(self):
        """`reproduce.meiosis(..., rng=...)` is deterministic given the same
        parents, assignment, recombination map, and seeded rng.
        """
        parents = _make_parents(n=20, m=50, seed=0)
        rmap = RecombinationMap.constant_map(m=50, p=0.5)
        assignment = _make_assignment(8, [0, 1, 2, 3, 4, 5, 6, 7],
                                       [10, 11, 12, 13, 14, 15, 16, 17])
        out1 = meiosis(parents, rmap,
                       assignment.maternal_idx, assignment.paternal_idx,
                       rng=np.random.RandomState(42))
        out2 = meiosis(parents, rmap,
                       assignment.maternal_idx, assignment.paternal_idx,
                       rng=np.random.RandomState(42))
        np.testing.assert_array_equal(out1, out2)

    def test_different_seeds_produce_different_offspring(self):
        """Sanity: a different master rng should produce different offspring
        (otherwise the seed is being ignored).
        """
        parents = _make_parents(n=20, m=50, seed=0)
        rmap = RecombinationMap.constant_map(m=50, p=0.5)
        assignment = _make_assignment(8, [0, 1, 2, 3, 4, 5, 6, 7],
                                       [10, 11, 12, 13, 14, 15, 16, 17])
        out1 = meiosis(parents, rmap,
                       assignment.maternal_idx, assignment.paternal_idx,
                       rng=np.random.RandomState(42))
        out2 = meiosis(parents, rmap,
                       assignment.maternal_idx, assignment.paternal_idx,
                       rng=np.random.RandomState(43))
        assert not np.array_equal(out1, out2)

    def test_none_rng_still_works(self):
        """Backward compatibility: omitting rng must still run (just
        non-deterministically) so direct callers that don't pass an rng
        keep working.
        """
        parents = _make_parents(n=8, m=20, seed=0)
        rmap = RecombinationMap.constant_map(m=20, p=0.5)
        assignment = _make_assignment(4, [0, 1, 2, 3], [4, 5, 6, 7])
        out = meiosis(parents, rmap,
                      assignment.maternal_idx, assignment.paternal_idx)
        assert out.shape == (4, 20, 2)
        assert set(np.unique(out).tolist()).issubset({0, 1})


# ── End-to-end: Simulation.run ─────────────────────────────────────────────

def _build_dense_sim(seed):
    n, m = 30, 12
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
    eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=seed)
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(eff))
    arch.add('Y.E', NoiseComponent(variance=0.5))
    arch.add('Y', AggregationComponent('Y.G + Y.E'))
    mate = RandomMating(offspring_per_pair=2)
    rmap = RecombinationMap.constant_map(m=m, p=0.5)
    return Simulation(hap, arch, mate, rmap, seed=seed)


class TestSimulationDeterminism:
    def test_two_runs_same_seed_identical_haplotypes(self):
        """`Simulation(seed=42).run(N)` is bit-for-bit reproducible across
        runs. This is the headline determinism property — what users actually
        care about. Previously broken because meiosis bypassed `self.rng`.
        """
        s1 = _build_dense_sim(seed=42); s1.run(3)
        s2 = _build_dense_sim(seed=42); s2.run(3)

        # Compare every retained generation, not just the latest, to catch
        # divergence at any point.
        assert set(s1.haplotype_history.keys()) == set(s2.haplotype_history.keys())
        for gen in s1.haplotype_history:
            h1 = s1.haplotype_history[gen].genotypes
            h2 = s2.haplotype_history[gen].genotypes
            np.testing.assert_array_equal(
                h1, h2,
                err_msg=f"haplotype divergence at generation {gen}",
            )

    def test_different_seeds_produce_different_simulations(self):
        s1 = _build_dense_sim(seed=42); s1.run(3)
        s2 = _build_dense_sim(seed=43); s2.run(3)
        h1 = s1.haplotype_history[s1.generation].genotypes
        h2 = s2.haplotype_history[s2.generation].genotypes
        assert not np.array_equal(h1, h2)


# ── GRG-native path ────────────────────────────────────────────────────────

# Gated on pygrgl + msprime + grg CLI (consistent with other GRG tests).
pygrgl = pytest.importorskip("pygrgl")
pytest.importorskip("msprime")
import shutil
if shutil.which("grg") is None:
    pytest.skip("`grg` CLI not on PATH; activate the xftsim venv to run these tests",
                allow_module_level=True)

from xftsim.founders import founder_haplotypes_from_msprime_grg


def _two_grg_sims(seed=42):
    """Build two `Simulation`s from independent reloads of the same founder
    GRG file. Necessary because each `Simulation.run` mutates the underlying
    GRG in place — we can't share one founder operator across two runs.
    """
    founder = founder_haplotypes_from_msprime_grg(
        n=10, sequence_length=5000, mutation_rate=1e-6,
    )
    m = founder.m
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, 'f.grg')
        pygrgl.save_grg(founder._grg, path)
        g1 = pygrgl.load_mutable_grg(path)
        g2 = pygrgl.load_mutable_grg(path)
        h1 = GraphHaplotypeOperator(
            grg=g1, generation=0,
            samples=founder.samples, variants=founder.variants,
        )
        h2 = GraphHaplotypeOperator(
            grg=g2, generation=0,
            samples=founder.samples, variants=founder.variants,
        )

        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=seed)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        mate = RandomMating(offspring_per_pair=2)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)

        sim1 = Simulation(h1, arch, mate, rmap, seed=seed)
        sim2 = Simulation(h2, arch, mate, rmap, seed=seed)
        sim1.run(2)
        sim2.run(2)
        return sim1, sim2


class TestGRGMeiosisDeterminism:
    def test_two_grg_sims_same_seed_identical_dense(self):
        """Two `Simulation`s using GRG-native meiosis with the same seed
        materialize to byte-identical dense haplotypes.
        """
        sim1, sim2 = _two_grg_sims(seed=42)
        d1 = sim1.haplotype_history[sim1.generation].to_dense().genotypes
        d2 = sim2.haplotype_history[sim2.generation].to_dense().genotypes
        np.testing.assert_array_equal(d1, d2)
