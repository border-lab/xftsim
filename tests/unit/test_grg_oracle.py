"""
Oracle-based mutation-correctness tests for GRG recombination.

For each offspring haplotype produced by the recombination algorithm,
verifies that the mutations it inherits (by walking up the GRG from
the offspring node to roots) match *exactly* the mutations expected
from splicing its source parents' haplotypes at the recorded
breakpoints/segments.

Adapted from ``haplotype_oracle.py`` in the upstream GRG benchmarks.

Both the Python (``grg_recombination.NonDuplicationRecombination``) and
C++ (``grg_recombination_native.NonDuplicationRecombination``) backends
are tested when available.
"""

from collections import deque
from dataclasses import dataclass
from typing import List, Set, Tuple

import numpy as np
import pytest

pygrgl = pytest.importorskip("pygrgl")

from xftsim.grg_recombination import (
    NonDuplicationRecombination as _PyRecomb,
    _phase_to_segments,
)
from xftsim.reproduce import _meiosis_pair_seeded, _spawn_meiosis_seeds


# ---------------------------------------------------------------------------
# Oracle helpers
# ---------------------------------------------------------------------------

def collect_ancestral_mutation_positions(grg, node_id: int) -> Set[int]:
    """All mutation positions inherited by *node_id* (self + all ancestors).

    BFS walk up via ``get_up_edges``.  Returns a set of base-pair positions.
    Positions are stable across ``sort_mutations()`` (unlike MutationIds).
    """
    visited = set()
    queue = deque([node_id])
    visited.add(node_id)
    positions: Set[int] = set()

    while queue:
        n = queue.popleft()
        for mut_id in grg.get_mutations_for_node(n):
            positions.add(grg.get_mutation_by_id(mut_id).position)
        for parent in grg.get_up_edges(n):
            if parent not in visited:
                visited.add(parent)
                queue.append(parent)

    return positions


def expected_positions_from_segments(
    segments: List[Tuple[int, int]],
    parent_pos_cache: dict,
) -> Set[int]:
    """Compute the expected mutation-position set for one offspring haplotype.

    *segments* is ``[(parent_id, end_bp), ...]`` as produced by
    ``_phase_to_segments``.  Each segment covers ``[start, end)`` where
    *start* is the previous segment's *end* (or 0 for the first segment).
    """
    expected: Set[int] = set()
    start = 0
    for source_parent, end in segments:
        source_pos = parent_pos_cache[source_parent]
        for p in source_pos:
            if start <= p < end:
                expected.add(p)
        start = end
    return expected


@dataclass
class _HaplotypeRecord:
    """Bookkeeping for one offspring haplotype node."""
    raw_id: int
    parent_haps: Tuple[int, int]
    segments: List[Tuple[int, int]]


# ---------------------------------------------------------------------------
# Backend discovery
# ---------------------------------------------------------------------------

def _available_backends():
    """Return list of (label, recomb_class) for installed backends."""
    backends = [("python", _PyRecomb)]
    try:
        from xftsim.grg_recombination_native import (
            NonDuplicationRecombination as _CppRecomb,
        )
        backends.append(("cpp", _CppRecomb))
    except ImportError:
        pass
    return backends


BACKENDS = _available_backends()
BACKEND_IDS = [b[0] for b in BACKENDS]


# ---------------------------------------------------------------------------
# GRG fixture
# ---------------------------------------------------------------------------

_GRG_SEARCH_PATHS = [
    "grg_files/50inds_1k_snps.grg",
    "../grg_files/50inds_1k_snps.grg",
    "grgl/benchmark/test_small/50inds_1k_snps.grg",
]

import os as _os

def _find_grg_file():
    repo = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    for rel in _GRG_SEARCH_PATHS:
        path = _os.path.join(repo, rel)
        if _os.path.exists(path):
            return path
    parent = _os.path.dirname(repo)
    for name in ("grg", "GRG-Implementation-", "grgl"):
        for root, dirs, files in _os.walk(_os.path.join(parent, name)):
            for f in files:
                if f.endswith(".grg") and "50inds" in f:
                    return _os.path.join(root, f)
            if len(dirs) > 20:
                break
    return None


def _load_fresh_grg(path):
    grg = pygrgl.load_mutable_grg(path)
    return grg


def _try_msprime_grg():
    """Generate a small GRG via msprime + grgl CLI."""
    try:
        from xftsim.founders import founder_haplotypes_from_msprime_grg
        op = founder_haplotypes_from_msprime_grg(
            n=20,
            sequence_length=100_000,
            recombination_rate=1e-4,
            mutation_rate=1e-4,
        )
        return op._grg, op._grg_path if hasattr(op, "_grg_path") else None
    except Exception:
        return None, None


@pytest.fixture(params=BACKEND_IDS)
def backend(request):
    """Parametrized recombination backend class."""
    for label, cls in BACKENDS:
        if label == request.param:
            return label, cls
    pytest.skip(f"backend {request.param} not available")


# Store the path globally so every test can load a fresh copy
_GRG_PATH = _find_grg_file()


def _require_grg():
    if _GRG_PATH is None:
        pytest.skip("No .grg test file found")
    return _GRG_PATH


# ---------------------------------------------------------------------------
# Core test harness
# ---------------------------------------------------------------------------

def _run_meiosis_with_oracle(
    grg_path: str,
    recomb_cls,
    n_offspring: int,
    recomb_p: float,
    seed: int,
    n_generations: int = 1,
):
    """Run meiosis, record segments, then oracle-verify every offspring.

    Returns ``(num_pass, num_fail, failures)`` where *failures* is a list
    of dicts describing each failed offspring.
    """
    grg = _load_fresh_grg(grg_path)
    m = grg.num_mutations
    genome_length = int(grg.bp_range[1])

    pos_bp = np.fromiter(
        (grg.get_mutation_by_id(i).position for i in range(m)),
        dtype=np.int64,
        count=m,
    )

    total_pass = 0
    total_fail = 0
    all_failures = []

    rng = np.random.RandomState(seed)

    for gen in range(n_generations):
        parent_sample_nodes = list(grg.get_sample_nodes())
        n_parents = len(parent_sample_nodes) // 2
        actual_n_off = min(n_offspring, n_parents)

        parent_pos_cache = {}
        for node_id in parent_sample_nodes:
            parent_pos_cache[node_id] = collect_ancestral_mutation_positions(
                grg, node_id
            )

        recomb_p_arr = np.full(m, recomb_p)
        seeds = _spawn_meiosis_seeds(rng, actual_n_off)

        recomb = recomb_cls(grg)
        recomb.defer_sample_updates = True

        mat_idx = np.arange(actual_n_off) % n_parents
        pat_idx = (np.arange(actual_n_off) + 1) % n_parents

        records: List[_HaplotypeRecord] = []
        new_offspring_grg_ids = []

        for i in range(actual_n_off):
            mat_i = int(mat_idx[i])
            pat_i = int(pat_idx[i])
            mat_haps = (
                parent_sample_nodes[2 * mat_i],
                parent_sample_nodes[2 * mat_i + 1],
            )
            pat_haps = (
                parent_sample_nodes[2 * pat_i],
                parent_sample_nodes[2 * pat_i + 1],
            )

            mat_phase, pat_phase = _meiosis_pair_seeded(recomb_p_arr, seeds[i])

            for parent_haps, phase in (
                (mat_haps, mat_phase),
                (pat_haps, pat_phase),
            ):
                segments = _phase_to_segments(
                    phase, pos_bp, parent_haps, genome_length
                )
                neg_id = recomb.recombine_multi(segments)
                raw_id = recomb.NEGATIVE_NODE_IDS[abs(neg_id) - 1]
                new_offspring_grg_ids.append(raw_id)
                records.append(
                    _HaplotypeRecord(
                        raw_id=raw_id,
                        parent_haps=parent_haps,
                        segments=segments,
                    )
                )

        new_offspring_grg_ids.sort()
        grg.set_samples(new_offspring_grg_ids)
        recomb._pending_sample_removals.clear()
        grg.sort_mutations()

        # Refresh pos_bp after sort_mutations (IDs change, positions don't,
        # but mutation count may have changed due to soft-delete compaction).
        m = grg.num_mutations
        pos_bp = np.fromiter(
            (grg.get_mutation_by_id(i).position for i in range(m)),
            dtype=np.int64,
            count=m,
        )

        # Oracle check for this generation
        for rec in records:
            expected = expected_positions_from_segments(
                rec.segments, parent_pos_cache
            )
            actual = collect_ancestral_mutation_positions(grg, rec.raw_id)

            if expected == actual:
                total_pass += 1
            else:
                total_fail += 1
                missing = expected - actual
                extra = actual - expected
                all_failures.append(
                    {
                        "generation": gen,
                        "offspring_id": rec.raw_id,
                        "parent_haps": rec.parent_haps,
                        "num_segments": len(rec.segments),
                        "missing_count": len(missing),
                        "extra_count": len(extra),
                        "missing_positions": sorted(missing)[:10],
                        "extra_positions": sorted(extra)[:10],
                    }
                )

    return total_pass, total_fail, all_failures


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOracleSingleGeneration:
    """Single-generation oracle checks at various recombination rates."""

    def test_moderate_recombination(self, backend):
        """p=0.25: typical recombination rate."""
        grg_path = _require_grg()
        label, cls = backend
        n_pass, n_fail, failures = _run_meiosis_with_oracle(
            grg_path, cls, n_offspring=20, recomb_p=0.25, seed=42
        )
        assert n_fail == 0, (
            f"[{label}] {n_fail}/{n_pass + n_fail} offspring failed oracle: "
            f"{failures[:3]}"
        )
        assert n_pass > 0

    def test_no_recombination(self, backend):
        """p=0: each offspring haplotype inherits exactly one parent haplotype."""
        grg_path = _require_grg()
        label, cls = backend
        n_pass, n_fail, failures = _run_meiosis_with_oracle(
            grg_path, cls, n_offspring=20, recomb_p=0.0, seed=43
        )
        assert n_fail == 0, (
            f"[{label}] {n_fail} failures at p=0: {failures[:3]}"
        )
        assert n_pass > 0

    def test_high_recombination(self, backend):
        """p=0.49: near-maximum crossover rate, many segment boundaries."""
        grg_path = _require_grg()
        label, cls = backend
        n_pass, n_fail, failures = _run_meiosis_with_oracle(
            grg_path, cls, n_offspring=20, recomb_p=0.49, seed=44
        )
        assert n_fail == 0, (
            f"[{label}] {n_fail} failures at p=0.49: {failures[:3]}"
        )
        assert n_pass > 0

    def test_free_recombination(self, backend):
        """p=0.5: independent segregation at every locus."""
        grg_path = _require_grg()
        label, cls = backend
        n_pass, n_fail, failures = _run_meiosis_with_oracle(
            grg_path, cls, n_offspring=20, recomb_p=0.5, seed=45
        )
        assert n_fail == 0, (
            f"[{label}] {n_fail} failures at p=0.5: {failures[:3]}"
        )
        assert n_pass > 0

    def test_many_offspring(self, backend):
        """Larger offspring cohort to stress-test cache handling."""
        grg_path = _require_grg()
        label, cls = backend
        n_pass, n_fail, failures = _run_meiosis_with_oracle(
            grg_path, cls, n_offspring=50, recomb_p=0.25, seed=46
        )
        assert n_fail == 0, (
            f"[{label}] {n_fail}/{n_pass + n_fail} failures: {failures[:3]}"
        )
        assert n_pass >= 50 * 2  # 2 haplotypes per offspring


class TestOracleMultiGeneration:
    """Multi-generation oracle checks — errors compound across generations."""

    def test_two_generations(self, backend):
        grg_path = _require_grg()
        label, cls = backend
        n_pass, n_fail, failures = _run_meiosis_with_oracle(
            grg_path,
            cls,
            n_offspring=20,
            recomb_p=0.25,
            seed=50,
            n_generations=2,
        )
        assert n_fail == 0, (
            f"[{label}] {n_fail} failures across 2 generations: {failures[:3]}"
        )
        assert n_pass > 0

    def test_three_generations(self, backend):
        grg_path = _require_grg()
        label, cls = backend
        n_pass, n_fail, failures = _run_meiosis_with_oracle(
            grg_path,
            cls,
            n_offspring=20,
            recomb_p=0.25,
            seed=51,
            n_generations=3,
        )
        assert n_fail == 0, (
            f"[{label}] {n_fail} failures across 3 generations: {failures[:3]}"
        )
        assert n_pass > 0


class TestOracleBackendParity:
    """Verify Python and C++ backends produce identical results."""

    def test_identical_offspring_ids(self):
        """Both backends produce the same offspring node IDs given same inputs."""
        grg_path = _require_grg()
        if len(BACKENDS) < 2:
            pytest.skip("need both Python and C++ backends")

        py_cls = BACKENDS[0][1]
        cpp_cls = BACKENDS[1][1]

        for seed in (42, 99, 1337):
            grg_py = _load_fresh_grg(grg_path)
            grg_cpp = _load_fresh_grg(grg_path)

            m = grg_py.num_mutations
            genome_length = int(grg_py.bp_range[1])
            pos_bp = np.fromiter(
                (grg_py.get_mutation_by_id(i).position for i in range(m)),
                dtype=np.int64,
                count=m,
            )
            recomb_p = np.full(m, 0.25)
            parent_samples = list(grg_py.get_sample_nodes())
            n_parents = len(parent_samples) // 2
            n_off = 15

            rng_py = np.random.RandomState(seed)
            rng_cpp = np.random.RandomState(seed)
            seeds_py = _spawn_meiosis_seeds(rng_py, n_off)
            seeds_cpp = _spawn_meiosis_seeds(rng_cpp, n_off)
            np.testing.assert_array_equal(seeds_py, seeds_cpp)

            def _run(grg, cls, seeds):
                recomb = cls(grg)
                recomb.defer_sample_updates = True
                samples = list(grg.get_sample_nodes())
                ids = []
                for i in range(n_off):
                    mat_i = i % n_parents
                    pat_i = (i + 1) % n_parents
                    mat_haps = (samples[2 * mat_i], samples[2 * mat_i + 1])
                    pat_haps = (samples[2 * pat_i], samples[2 * pat_i + 1])
                    mat_phase, pat_phase = _meiosis_pair_seeded(
                        recomb_p, seeds[i]
                    )
                    for parent_haps, phase in (
                        (mat_haps, mat_phase),
                        (pat_haps, pat_phase),
                    ):
                        segs = _phase_to_segments(
                            phase, pos_bp, parent_haps, genome_length
                        )
                        neg_id = recomb.recombine_multi(segs)
                        raw_id = recomb.NEGATIVE_NODE_IDS[abs(neg_id) - 1]
                        ids.append(raw_id)
                new_ids = sorted(ids)
                grg.set_samples(new_ids)
                recomb._pending_sample_removals.clear()
                grg.sort_mutations()
                return ids

            py_ids = _run(grg_py, py_cls, seeds_py)
            cpp_ids = _run(grg_cpp, cpp_cls, seeds_cpp)

            assert py_ids == cpp_ids, (
                f"seed={seed}: offspring IDs differ: "
                f"py={py_ids[:5]}... cpp={cpp_ids[:5]}..."
            )

    def test_identical_topology(self):
        """Both backends produce GRGs with identical node/edge counts."""
        grg_path = _require_grg()
        if len(BACKENDS) < 2:
            pytest.skip("need both Python and C++ backends")

        py_cls = BACKENDS[0][1]
        cpp_cls = BACKENDS[1][1]

        for n_gen in (1, 2):
            grg_py = _load_fresh_grg(grg_path)
            grg_cpp = _load_fresh_grg(grg_path)

            m = grg_py.num_mutations
            genome_length = int(grg_py.bp_range[1])
            pos_bp = np.fromiter(
                (grg_py.get_mutation_by_id(i).position for i in range(m)),
                dtype=np.int64,
                count=m,
            )
            recomb_p = np.full(m, 0.25)

            def _run_gen(grg, cls, rng):
                samples = list(grg.get_sample_nodes())
                n_parents = len(samples) // 2
                n_off = min(20, n_parents)
                seeds = _spawn_meiosis_seeds(rng, n_off)
                recomb = cls(grg)
                recomb.defer_sample_updates = True
                ids = []
                for i in range(n_off):
                    mat_i = i % n_parents
                    pat_i = (i + 1) % n_parents
                    mat_haps = (samples[2 * mat_i], samples[2 * mat_i + 1])
                    pat_haps = (samples[2 * pat_i], samples[2 * pat_i + 1])
                    mat_phase, pat_phase = _meiosis_pair_seeded(
                        recomb_p, seeds[i]
                    )
                    for haps, phase in (
                        (mat_haps, mat_phase),
                        (pat_haps, pat_phase),
                    ):
                        segs = _phase_to_segments(
                            phase, pos_bp, haps, genome_length
                        )
                        neg_id = recomb.recombine_multi(segs)
                        raw_id = recomb.NEGATIVE_NODE_IDS[abs(neg_id) - 1]
                        ids.append(raw_id)
                new_ids = sorted(ids)
                grg.set_samples(new_ids)
                recomb._pending_sample_removals.clear()
                grg.sort_mutations()

            rng_py = np.random.RandomState(42)
            rng_cpp = np.random.RandomState(42)
            for _ in range(n_gen):
                _run_gen(grg_py, py_cls, rng_py)
                _run_gen(grg_cpp, cpp_cls, rng_cpp)

            assert grg_py.num_nodes == grg_cpp.num_nodes, (
                f"n_gen={n_gen}: nodes differ "
                f"py={grg_py.num_nodes} cpp={grg_cpp.num_nodes}"
            )
            assert grg_py.num_edges == grg_cpp.num_edges, (
                f"n_gen={n_gen}: edges differ "
                f"py={grg_py.num_edges} cpp={grg_cpp.num_edges}"
            )

    def test_identical_mutation_sets(self):
        """Both backends produce offspring with identical mutation sets."""
        grg_path = _require_grg()
        if len(BACKENDS) < 2:
            pytest.skip("need both Python and C++ backends")

        py_cls = BACKENDS[0][1]
        cpp_cls = BACKENDS[1][1]

        grg_py = _load_fresh_grg(grg_path)
        grg_cpp = _load_fresh_grg(grg_path)

        m = grg_py.num_mutations
        genome_length = int(grg_py.bp_range[1])
        pos_bp = np.fromiter(
            (grg_py.get_mutation_by_id(i).position for i in range(m)),
            dtype=np.int64,
            count=m,
        )
        recomb_p = np.full(m, 0.25)
        n_off = 20

        rng_py = np.random.RandomState(42)
        rng_cpp = np.random.RandomState(42)

        def _run_and_collect(grg, cls, rng):
            samples = list(grg.get_sample_nodes())
            n_parents = len(samples) // 2
            seeds = _spawn_meiosis_seeds(rng, n_off)
            recomb = cls(grg)
            recomb.defer_sample_updates = True
            ids = []
            for i in range(n_off):
                mat_i = i % n_parents
                pat_i = (i + 1) % n_parents
                mat_haps = (samples[2 * mat_i], samples[2 * mat_i + 1])
                pat_haps = (samples[2 * pat_i], samples[2 * pat_i + 1])
                mat_phase, pat_phase = _meiosis_pair_seeded(
                    recomb_p, seeds[i]
                )
                for haps, phase in (
                    (mat_haps, mat_phase),
                    (pat_haps, pat_phase),
                ):
                    segs = _phase_to_segments(
                        phase, pos_bp, haps, genome_length
                    )
                    neg_id = recomb.recombine_multi(segs)
                    raw_id = recomb.NEGATIVE_NODE_IDS[abs(neg_id) - 1]
                    ids.append(raw_id)
            new_ids = sorted(ids)
            grg.set_samples(new_ids)
            recomb._pending_sample_removals.clear()
            grg.sort_mutations()
            return ids

        py_ids = _run_and_collect(grg_py, py_cls, rng_py)
        cpp_ids = _run_and_collect(grg_cpp, cpp_cls, rng_cpp)

        assert py_ids == cpp_ids, "Offspring IDs differ"

        for py_id, cpp_id in zip(py_ids, cpp_ids):
            py_muts = collect_ancestral_mutation_positions(grg_py, py_id)
            cpp_muts = collect_ancestral_mutation_positions(grg_cpp, cpp_id)
            assert py_muts == cpp_muts, (
                f"Mutation sets differ for offspring {py_id}: "
                f"{len(py_muts - cpp_muts)} in py only, "
                f"{len(cpp_muts - py_muts)} in cpp only"
            )


class TestOracleEdgeCases:
    """Edge cases that have historically been tricky for GRG recombination."""

    def test_self_mating(self, backend):
        """Same individual as both mother and father."""
        grg_path = _require_grg()
        label, cls = backend
        grg = _load_fresh_grg(grg_path)
        m = grg.num_mutations
        genome_length = int(grg.bp_range[1])
        pos_bp = np.fromiter(
            (grg.get_mutation_by_id(i).position for i in range(m)),
            dtype=np.int64,
            count=m,
        )
        parent_samples = list(grg.get_sample_nodes())
        parent_pos_cache = {}
        for nid in parent_samples:
            parent_pos_cache[nid] = collect_ancestral_mutation_positions(
                grg, nid
            )

        recomb_p = np.full(m, 0.25)
        recomb = cls(grg)
        recomb.defer_sample_updates = True

        rng = np.random.RandomState(77)
        seeds = _spawn_meiosis_seeds(rng, 5)
        records = []
        ids = []

        for i in range(5):
            # Same individual as mother and father
            haps = (parent_samples[0], parent_samples[1])
            mat_phase, pat_phase = _meiosis_pair_seeded(recomb_p, seeds[i])
            for phase in (mat_phase, pat_phase):
                segs = _phase_to_segments(
                    phase, pos_bp, haps, genome_length
                )
                neg_id = recomb.recombine_multi(segs)
                raw_id = recomb.NEGATIVE_NODE_IDS[abs(neg_id) - 1]
                ids.append(raw_id)
                records.append(_HaplotypeRecord(raw_id, haps, segs))

        new_ids = sorted(ids)
        grg.set_samples(new_ids)
        recomb._pending_sample_removals.clear()
        grg.sort_mutations()

        for rec in records:
            expected = expected_positions_from_segments(
                rec.segments, parent_pos_cache
            )
            actual = collect_ancestral_mutation_positions(grg, rec.raw_id)
            missing = expected - actual
            extra = actual - expected
            assert expected == actual, (
                f"[{label}] self-mating offspring {rec.raw_id}: "
                f"{len(missing)} missing, {len(extra)} extra"
            )

    def test_single_segment_no_crossover(self, backend):
        """Recombination with a single segment (no crossover at all)."""
        grg_path = _require_grg()
        label, cls = backend
        grg = _load_fresh_grg(grg_path)
        m = grg.num_mutations
        genome_length = int(grg.bp_range[1])
        pos_bp = np.fromiter(
            (grg.get_mutation_by_id(i).position for i in range(m)),
            dtype=np.int64,
            count=m,
        )
        parent_samples = list(grg.get_sample_nodes())
        parent_pos_cache = {}
        for nid in parent_samples:
            parent_pos_cache[nid] = collect_ancestral_mutation_positions(
                grg, nid
            )

        recomb = cls(grg)
        recomb.defer_sample_updates = True
        ids = []
        records = []

        for i in range(10):
            parent = parent_samples[i]
            segs = [(parent, genome_length)]
            neg_id = recomb.recombine_multi(segs)
            raw_id = recomb.NEGATIVE_NODE_IDS[abs(neg_id) - 1]
            ids.append(raw_id)
            records.append(
                _HaplotypeRecord(raw_id, (parent, parent), segs)
            )

        new_ids = sorted(ids)
        grg.set_samples(new_ids)
        recomb._pending_sample_removals.clear()
        grg.sort_mutations()

        for rec in records:
            expected = expected_positions_from_segments(
                rec.segments, parent_pos_cache
            )
            actual = collect_ancestral_mutation_positions(grg, rec.raw_id)
            assert expected == actual, (
                f"[{label}] single-segment offspring {rec.raw_id}: "
                f"expected {len(expected)} mutations, got {len(actual)}"
            )

    def test_many_crossovers(self, backend):
        """Very high recombination (p~0.5) producing many short segments."""
        grg_path = _require_grg()
        label, cls = backend
        n_pass, n_fail, failures = _run_meiosis_with_oracle(
            grg_path, cls, n_offspring=10, recomb_p=0.499, seed=88
        )
        assert n_fail == 0, (
            f"[{label}] {n_fail} failures with many crossovers: {failures[:3]}"
        )

    def test_all_offspring_from_same_parents(self, backend):
        """All offspring share the same two parents — stresses cache reuse."""
        grg_path = _require_grg()
        label, cls = backend
        grg = _load_fresh_grg(grg_path)
        m = grg.num_mutations
        genome_length = int(grg.bp_range[1])
        pos_bp = np.fromiter(
            (grg.get_mutation_by_id(i).position for i in range(m)),
            dtype=np.int64,
            count=m,
        )
        parent_samples = list(grg.get_sample_nodes())
        parent_pos_cache = {}
        for nid in parent_samples:
            parent_pos_cache[nid] = collect_ancestral_mutation_positions(
                grg, nid
            )

        recomb_p = np.full(m, 0.25)
        recomb = cls(grg)
        recomb.defer_sample_updates = True

        rng = np.random.RandomState(99)
        n_off = 30
        seeds = _spawn_meiosis_seeds(rng, n_off)
        ids = []
        records = []

        haps = (parent_samples[0], parent_samples[1])
        for i in range(n_off):
            mat_phase, _ = _meiosis_pair_seeded(recomb_p, seeds[i])
            segs = _phase_to_segments(mat_phase, pos_bp, haps, genome_length)
            neg_id = recomb.recombine_multi(segs)
            raw_id = recomb.NEGATIVE_NODE_IDS[abs(neg_id) - 1]
            ids.append(raw_id)
            records.append(_HaplotypeRecord(raw_id, haps, segs))

        new_ids = sorted(ids)
        grg.set_samples(new_ids)
        recomb._pending_sample_removals.clear()
        grg.sort_mutations()

        n_fail = 0
        for rec in records:
            expected = expected_positions_from_segments(
                rec.segments, parent_pos_cache
            )
            actual = collect_ancestral_mutation_positions(grg, rec.raw_id)
            if expected != actual:
                n_fail += 1
        assert n_fail == 0, (
            f"[{label}] {n_fail}/{n_off} offspring from same parents failed"
        )
