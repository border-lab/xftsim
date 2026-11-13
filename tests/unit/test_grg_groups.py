"""
Option B position hierarchy (``build_groups``) and GRG compaction.

Checks the invariants the position hierarchy must preserve: it leaves every
haplotype's genotype and path counts unchanged; groups are mutation-free with
2..B parents and one child; short parent lists are left alone; parents are
ordered by ancestral span. Also checks Python/native
parity, meiosis-level exactness against dense genotypes with and without
compaction, and the checkpoint round trip.

Compaction tests need pygrgl built from the grgl ``drop-unreachable`` branch
and are skipped otherwise.
"""
import os

import numpy as np
import pytest

pygrgl = pytest.importorskip("pygrgl")
pytest.importorskip("msprime")

import xftsim.struct as xs
from xftsim.arch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.effect import AdditiveEffects
from xftsim.founders import founder_haplotypes_from_msprime_grg
from xftsim.grg_recombination import (
    NonDuplicationRecombination as PyRecomb,
    _switches_to_segments,
)
from xftsim.io import (
    _load_graph_haplotypes_from_checkpoint,
    _save_graph_haplotypes_to_checkpoint,
    save_simulation_checkpoint,
)
from xftsim.mate import RandomMating
from xftsim.reproduce import (
    RecombinationMap, _crossover_model, _meiosis_pair_switches_seeded, _spawn_meiosis_seeds,
)
from xftsim.sim import Simulation

try:
    from xftsim.grg_recombination_native import NonDuplicationRecombination as CppRecomb
except ImportError:
    CppRecomb = None

BACKENDS = [pytest.param(PyRecomb, id="python"),
            pytest.param(CppRecomb, id="cpp", marks=pytest.mark.skipif(
                CppRecomb is None, reason="native recombiner not built"))]

needs_drop_unreachable = pytest.mark.skipif(
    not xs._save_grg_supports_drop_unreachable(),
    reason="pygrgl.save_grg lacks drop_unreachable (needs the grgl drop-unreachable branch)")

CROSSOVERS = 3.0  # expected per meiosis: enough for most offspring to exceed K parents
SEED = 11


# ---------------------------------------------------------------------------
# Fixtures and graph helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def founders_file(tmp_path_factory):
    """Founders saved once (unsimplified); each test loads its own copy."""
    f = founder_haplotypes_from_msprime_grg(
        n=60, sequence_length=1_000_000, recombination_rate=1e-8,
        mutation_rate=1e-8, seed=SEED)
    path = str(tmp_path_factory.mktemp("founders") / "founders.grg")
    pygrgl.save_grg(f._grg, path, allow_simplify=False)
    return path, f.samples, f.variants


def _load(founders_file, options=None):
    path, samples, variants = founders_file
    return xs.GraphHaplotypeOperator(pygrgl.load_mutable_grg(path), samples=samples,
                                     variants=variants, meiosis_options=options)


def _rmap(op):
    return RecombinationMap.constant_map(op.m, p=CROSSOVERS / op.m, pos_bp=op.variants.pos_bp)


def _haplotypes(grg):
    """(m, 2n) haploid dosages from matmul. A doubled path would show up as a 2."""
    m = grg.num_mutations
    blocks = []
    for start in range(0, m, 512):
        stop = min(start + 512, m)
        unit = np.zeros((stop - start, m))
        unit[np.arange(stop - start), np.arange(start, stop)] = 1.0
        blocks.append(pygrgl.matmul(grg, unit, pygrgl.TraversalDirection.DOWN,
                                    by_individual=False))
    return np.rint(np.vstack(blocks)).astype(np.int64)


def _span(grg, node):
    """Ancestral span (min, max mutation position over node and ancestors), from scratch."""
    seen, stack, lo, hi = {node}, [node], None, None
    while stack:
        v = stack.pop()
        for mid in grg.get_mutations_for_node(v):
            p = grg.get_mutation_by_id(mid).position
            lo = p if lo is None else min(lo, p)
            hi = p if hi is None else max(hi, p)
        for u in grg.get_up_edges(v):
            if u not in seen:
                seen.add(u)
                stack.append(u)
    return None if lo is None else (lo, hi)


def _positions(grg, node):
    """Mutation positions reachable upward from node."""
    seen, stack, out = {node}, [node], set()
    while stack:
        v = stack.pop()
        out.update(grg.get_mutation_by_id(mid).position for mid in grg.get_mutations_for_node(v))
        for u in grg.get_up_edges(v):
            if u not in seen:
                seen.add(u)
                stack.append(u)
    return out


def _snapshot(grg):
    return (grg.num_nodes, grg.num_edges,
            [sorted(grg.get_up_edges(v)) for v in range(grg.num_nodes)])


def _one_generation(grg, recomb_cls, seed):
    """Random-mating meiosis straight on the recombiner, up to set_samples.

    Like GraphHaplotypeOperator.meiosis, callers run build_groups next and
    then sort_mutations; pygrgl.matmul is only safe after the sort.
    Returns (recombiner, new sample node ids, segments per offspring).
    """
    m = grg.num_mutations
    pos = np.fromiter((grg.get_mutation_by_id(i).position for i in range(m)), np.int64, m)
    p = np.full(m, CROSSOVERS / m)
    p[0] = 0.5
    cumlam, special_idx, special_p = _crossover_model(p)
    parents = list(grg.get_sample_nodes())
    n_ind = len(parents) // 2
    genome_length = int(grg.bp_range[1])
    rng = np.random.RandomState(seed)
    mothers, fathers = rng.randint(n_ind, size=n_ind), rng.randint(n_ind, size=n_ind)
    seeds = _spawn_meiosis_seeds(rng, n_ind)
    recomb = recomb_cls(grg)
    recomb.defer_sample_updates = True
    neg_ids, all_segments = [], []
    for i in range(n_ind):
        switches = _meiosis_pair_switches_seeded(cumlam, special_idx, special_p, seeds[i])
        for par, sw in zip((mothers[i], fathers[i]), switches):
            segments = _switches_to_segments(sw, pos, (parents[2 * par], parents[2 * par + 1]),
                                             genome_length)
            all_segments.append(segments)
            neg_ids.append(recomb.recombine_multi(segments))
    raw = recomb.NEGATIVE_NODE_IDS
    new = [raw[abs(x) - 1] for x in neg_ids]
    grg.set_samples(new)
    recomb._pending_sample_removals.clear()
    return recomb, new, all_segments


def _max_sample_parents(grg):
    return max(grg.num_up_edges(s) for s in grg.get_sample_nodes())


def _tree_leaves(grg, node, first_group):
    """Leaves under a regrouped sample's group tree, in up-edge order."""
    out = []
    for u in grg.get_up_edges(node):
        if u >= first_group:
            out.extend(_tree_leaves(grg, u, first_group))
        else:
            out.append(u)
    return out


# ---------------------------------------------------------------------------
# build_groups invariants (both backends)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("recomb_cls", BACKENDS)
class TestBuildGroupsInvariants:
    @pytest.mark.parametrize("fanin,min_parents", [(4, 8), (2, 8), (8, 4)])
    def test_invariants(self, founders_file, recomb_cls, fanin, min_parents):
        grg = _load(founders_file)._grg
        recomb, samples, _ = _one_generation(grg, recomb_cls, seed=1)
        threshold = max(fanin, min_parents)
        before = {s: list(grg.get_up_edges(s)) for s in samples}
        reach_before = [_positions(grg, s) for s in samples]
        spans = {p: _span(grg, p) for ps in before.values() for p in ps}
        nodes0, edges0 = grg.num_nodes, grg.num_edges

        stats = recomb.build_groups(samples, fanin, min_parents)
        grg.sort_mutations()

        # Same mutations reachable from every haplotype, along a single path
        # (matmul counts paths, so a second one would show up as a 2).
        assert [_positions(grg, s) for s in samples] == reach_before
        geno = _haplotypes(grg)
        assert geno.max() == 1
        pos = np.array([grg.get_mutation_by_id(i).position for i in range(grg.num_mutations)])
        assert [set(pos[geno[:, j] == 1].tolist()) for j in range(len(samples))] == reach_before

        regrouped = [s for s in samples if len(before[s]) > threshold]
        assert regrouped, "fixture too small: no offspring exceeded the threshold"
        assert stats["samples_regrouped"] == len(regrouped)
        assert stats["groups_made"] == grg.num_nodes - nodes0
        assert stats["edges_added"] == grg.num_edges - edges0 == stats["groups_made"]

        for s in samples:
            if len(before[s]) <= threshold:
                assert list(grg.get_up_edges(s)) == before[s]
                continue
            top = grg.get_up_edges(s)
            assert 2 <= len(top) <= fanin
            # Parents ordered by (lo, hi, id), empty spans last.
            key = {p: (spans[p] or (np.inf, np.inf)) + (p,) for p in before[s]}
            assert _tree_leaves(grg, s, nodes0) == sorted(before[s], key=key.get)

        for g in range(nodes0, grg.num_nodes):
            assert len(grg.get_mutations_for_node(g)) == 0
            assert 2 <= grg.num_up_edges(g) <= fanin
            assert grg.num_down_edges(g) == 1

    def test_fanin_below_two_rejected(self, founders_file, recomb_cls):
        grg = _load(founders_file)._grg
        recomb, samples, _ = _one_generation(grg, recomb_cls, seed=1)
        with pytest.raises(ValueError, match="fanin"):
            recomb.build_groups(samples, 1, 8)

    def test_recombiner_still_valid_after_grouping(self, founders_file, recomb_cls):
        """Caches stay consistent: offspring of regrouped samples are exact."""
        grg = _load(founders_file)._grg
        m = grg.num_mutations
        # Read before meiosis: until sort_mutations, moved mutations leave
        # placeholder entries in the mutation table.
        pos = np.fromiter((grg.get_mutation_by_id(i).position for i in range(m)), np.int64, m)
        recomb, samples, _ = _one_generation(grg, recomb_cls, seed=1)
        recomb.build_groups(samples, 4, 8)
        genome_length = int(grg.bp_range[1])
        rng = np.random.RandomState(5)
        cache, children, expected_sets = {}, [], []
        for _ in range(20):
            a, b = (samples[i] for i in rng.choice(len(samples), 2, replace=False))
            cuts = sorted(int(pos[j]) for j in rng.choice(np.arange(1, m), 3, replace=False))
            segments = [(a, cuts[0]), (b, cuts[1]), (a, cuts[2]), (b, genome_length)]
            for h in (a, b):
                cache.setdefault(h, _positions(grg, h))
            neg = recomb.recombine_multi(segments)
            child = recomb.NEGATIVE_NODE_IDS[abs(neg) - 1]  # native: a copy, so read after
            expected, start = set(), 0
            for h, end in segments:
                expected.update(p for p in cache[h] if start <= p < end)
                start = end
            assert _positions(grg, child) == expected
            children.append(child)
            expected_sets.append(expected)
        # Dosages too: a doubled path would pass the set check but give a 2.
        grg.set_samples(children)
        grg.sort_mutations()
        geno = _haplotypes(grg)
        assert geno.max() == 1
        pos = np.array([grg.get_mutation_by_id(i).position for i in range(m)])
        assert [set(pos[geno[:, j] == 1].tolist()) for j in range(len(children))] == expected_sets

    def test_group_of_empty_span_parents(self, recomb_cls):
        """Parents without mutations anywhere above them sort last; a chunk of
        only such parents still becomes a (span-less) group."""
        grg = pygrgl.MutableGRG(2, 2, True)
        with_muts = []
        for k in range(8):
            v = grg.make_node()
            grg.add_mutation(pygrgl.Mutation(1000 * (8 - k), "A", "G"), v)
            with_muts.append(v)
        empty = [grg.make_node() for _ in range(4)]
        for v in empty + with_muts:
            grg.connect(v, 0)
        grg.get_mutations_for_node(0, True)  # sort the node->mutation index
        recomb = recomb_cls(grg)
        nodes0 = grg.num_nodes
        stats = recomb.build_groups([0, 1], 4, 8)
        assert stats == {"samples_regrouped": 1, "groups_made": 3, "edges_added": 3}
        leaves = _tree_leaves(grg, 0, nodes0)
        assert leaves == sorted(with_muts, key=lambda v: _span(grg, v)) + sorted(empty)
        empty_group = [g for g in range(nodes0, grg.num_nodes)
                       if sorted(grg.get_up_edges(g)) == sorted(empty)]
        assert len(empty_group) == 1 and _span(grg, empty_group[0]) is None
        assert _positions(grg, 0) == {1000 * (k + 1) for k in range(8)}


# ---------------------------------------------------------------------------
# Python reference vs native
# ---------------------------------------------------------------------------

@pytest.mark.skipif(CppRecomb is None, reason="native recombiner not built")
class TestBuildGroupsParity:
    def test_identical_graphs_over_generations(self, founders_file):
        """Same meiosis + grouping on both backends gives the same graph each generation.

        From generation 2 on, recombination walks through generation-1 groups.
        """
        g_py = _load(founders_file)._grg
        g_cpp = _load(founders_file)._grg
        for gen in range(1, 4):
            r_py, s_py, _ = _one_generation(g_py, PyRecomb, seed=gen)
            r_cpp, s_cpp, _ = _one_generation(g_cpp, CppRecomb, seed=gen)
            assert s_py == s_cpp
            assert _snapshot(g_py) == _snapshot(g_cpp)
            assert r_py.build_groups(s_py, 4, 8) == r_cpp.build_groups(s_cpp, 4, 8)
            assert _snapshot(g_py) == _snapshot(g_cpp)
            del r_py, r_cpp
            g_py.sort_mutations()
            g_cpp.sort_mutations()
        np.testing.assert_array_equal(_haplotypes(g_py), _haplotypes(g_cpp))


# ---------------------------------------------------------------------------
# GraphHaplotypeOperator.meiosis with GraphMeiosisOptions
# ---------------------------------------------------------------------------

def _run(op, gens, seed=SEED):
    """Random-mating generations; returns the final operator and per-generation genotypes."""
    rng = np.random.RandomState(seed)
    mating = RandomMating(offspring_per_pair=2)
    rmap = _rmap(op)
    genos = [op.to_dense().genotypes]
    for _ in range(gens):
        op = op.meiosis(mating.mate(op.samples, rng=rng), rmap, rng=rng)
        genos.append(op.to_dense().genotypes)
    return op, genos


def _dense_run(founders_file, gens, seed=SEED):
    grg_op = _load(founders_file)
    dense = grg_op.to_dense()
    rmap = _rmap(grg_op)
    rng = np.random.RandomState(seed)
    mating = RandomMating(offspring_per_pair=2)
    genos = [dense.genotypes]
    for _ in range(gens):
        dense = dense.meiosis(mating.mate(dense.samples, rng=rng), rmap, rng=rng)
        genos.append(dense.genotypes)
    return genos


def _live_counts(grg):
    """Nodes reachable upward from the samples, and edges between them."""
    seen = set(grg.get_sample_nodes())
    stack = list(seen)
    while stack:
        for u in grg.get_up_edges(stack.pop()):
            if u not in seen:
                seen.add(u)
                stack.append(u)
    return len(seen), sum(grg.num_up_edges(v) for v in seen)


class TestGraphMeiosisOptions:
    def test_defaults_are_off(self):
        opts = xs.GraphMeiosisOptions()
        assert (opts.group_fanin, opts.compact_every, opts.compact_dir) == (0, 0, None)

    @pytest.mark.parametrize("kwargs", [dict(group_fanin=1), dict(group_fanin=-2),
                                        dict(group_min_parents=-1), dict(compact_every=-1)])
    def test_invalid(self, kwargs):
        with pytest.raises(ValueError):
            xs.GraphMeiosisOptions(**kwargs)


class TestOptionBMeiosis:
    GENS = 4

    def test_option_b_matches_plain_and_dense(self, founders_file):
        dense = _dense_run(founders_file, self.GENS)
        plain_final, plain = _run(_load(founders_file), self.GENS)
        opts = xs.GraphMeiosisOptions(group_fanin=4, group_min_parents=8)
        final, grouped = _run(_load(founders_file, opts), self.GENS)
        for gen in range(self.GENS + 1):
            np.testing.assert_array_equal(plain[gen], dense[gen])
            np.testing.assert_array_equal(grouped[gen], dense[gen])
        assert final.meiosis_options is opts
        stats = final.last_meiosis_stats
        assert stats["groups_made"] > 0 and stats["group_edges"] == stats["groups_made"]
        assert stats["compacted"] is False
        # Recombinants list tens of pieces without groups; with them no sample
        # keeps more than max(fanin, min_parents) parents.
        assert _max_sample_parents(plain_final._grg) > 8
        assert _max_sample_parents(final._grg) <= 8

    @pytest.mark.parametrize("fanin", [0, 4])
    def test_meiosis_stats_match_the_graph(self, founders_file, fanin):
        """Stats checked against the graph itself, not against their formulas."""
        op = _load(founders_file, xs.GraphMeiosisOptions(group_fanin=fanin))
        grg = op._grg
        nodes0, edges0 = grg.num_nodes, grg.num_edges
        rng = np.random.RandomState(SEED)
        child = op.meiosis(RandomMating(offspring_per_pair=2).mate(op.samples, rng=rng),
                           _rmap(op), rng=rng)
        stats = child.last_meiosis_stats
        assert stats["edges"] is None and stats["compacted"] is False
        assert stats["nodes"] == grg.num_nodes
        assert stats["meiosis_edges"] + stats["group_edges"] == grg.num_edges - edges0
        assert stats["meiosis_nodes"] + stats["groups_made"] == grg.num_nodes - nodes0
        # Every new node is an offspring sample, a bubble or a group. Only
        # bubbles carry mutations, though a later bubble can take all of an
        # earlier one's, so some bubbles end up empty.
        new = set(range(nodes0, grg.num_nodes))
        samples = set(grg.get_sample_nodes())
        assert samples <= new and len(samples) == stats["offspring_haplotypes"] == 2 * child.n
        assert len(new - samples) == stats["bubbles"] + stats["groups_made"]
        with_muts = sum(1 for v in new if len(grg.get_mutations_for_node(v)) > 0)
        assert 0 < with_muts <= stats["bubbles"]
        # CROSSOVERS expected per meiosis: Poisson(360) over 120 meioses.
        assert 280 < stats["crossovers"] < 440
        assert 0 < stats["recombinant_haplotypes"] <= stats["offspring_haplotypes"]
        if fanin:
            assert stats["groups_made"] > 0 and stats["samples_regrouped"] > 0
        else:
            assert stats["groups_made"] == stats["samples_regrouped"] == 0
        if CppRecomb is not None:
            assert stats["visits"] > 0

    def test_population_size_change(self, founders_file):
        """n of a stale parent stays the parent's (Simulation reads prev_hap.n)."""
        op = _load(founders_file, xs.GraphMeiosisOptions(group_fanin=4))
        rng = np.random.RandomState(0)
        child = op.meiosis(RandomMating(offspring_per_pair=1).mate(op.samples, rng=rng),
                           _rmap(op), rng=rng)
        assert op.stale and not child.stale
        assert (op.n, child.n) == (60, 30)
        assert child._grg.num_individuals == 30

    def test_compact_dir_checked_before_meiosis(self, founders_file, tmp_path, monkeypatch):
        monkeypatch.setattr(xs, "_save_grg_supports_drop_unreachable", lambda: True)
        op = _load(founders_file, xs.GraphMeiosisOptions(
            compact_every=1, compact_dir=str(tmp_path / "missing")))
        rng = np.random.RandomState(0)
        assignment = RandomMating(offspring_per_pair=2).mate(op.samples, rng=rng)
        with pytest.raises(RuntimeError, match="not writable"):
            op.meiosis(assignment, _rmap(op), rng=rng)
        assert not op.stale and not op.graph_released

    def test_failed_compaction_save_keeps_graph(self, founders_file, monkeypatch):
        monkeypatch.setattr(xs, "_save_grg_supports_drop_unreachable", lambda: True)

        def failing_save(*args, **kwargs):
            raise OSError("disk full")
        monkeypatch.setattr(pygrgl, "save_grg", failing_save)
        dense = _dense_run(founders_file, 1)
        op = _load(founders_file, xs.GraphMeiosisOptions(group_fanin=4, compact_every=1))
        rng = np.random.RandomState(SEED)
        assignment = RandomMating(offspring_per_pair=2).mate(op.samples, rng=rng)
        with pytest.warns(UserWarning, match="disk full"):
            child = op.meiosis(assignment, _rmap(op), rng=rng)
        assert not op.graph_released and child.last_meiosis_stats["compacted"] is False
        np.testing.assert_array_equal(child.to_dense().genotypes, dense[1])

    def test_failed_compaction_reload_keeps_file(self, founders_file, tmp_path, monkeypatch):
        monkeypatch.setattr(xs, "_save_grg_supports_drop_unreachable", lambda: True)
        real_save = pygrgl.save_grg

        def save(grg, path, allow_simplify=True, drop_unreachable=False):
            return real_save(grg, path, allow_simplify=allow_simplify)

        def failing_load(path):
            raise MemoryError("out of memory")
        op = _load(founders_file, xs.GraphMeiosisOptions(compact_every=1,
                                                         compact_dir=str(tmp_path)))
        monkeypatch.setattr(pygrgl, "save_grg", save)
        monkeypatch.setattr(pygrgl, "load_mutable_grg", failing_load)
        rng = np.random.RandomState(0)
        assignment = RandomMating(offspring_per_pair=2).mate(op.samples, rng=rng)
        with pytest.raises(RuntimeError, match="kept for recovery"):
            op.meiosis(assignment, _rmap(op), rng=rng)
        assert [f for f in os.listdir(str(tmp_path)) if f.startswith("xftsim_compact_")]

    def test_compaction_requires_patched_pygrgl(self, founders_file, monkeypatch):
        monkeypatch.setattr(xs, "_save_grg_supports_drop_unreachable", lambda: False)
        op = _load(founders_file, xs.GraphMeiosisOptions(group_fanin=4, compact_every=1))
        samples_before = list(op._grg.get_sample_nodes())
        rng = np.random.RandomState(0)
        assignment = RandomMating(offspring_per_pair=2).mate(op.samples, rng=rng)
        with pytest.raises(RuntimeError, match="drop-unreachable"):
            op.meiosis(assignment, _rmap(op), rng=rng)
        # Refused before touching the graph.
        assert list(op._grg.get_sample_nodes()) == samples_before
        assert not op.graph_released


@needs_drop_unreachable
class TestCompaction:
    GENS = 4

    @pytest.mark.parametrize("compact_every", [1, 2])
    def test_exact_against_dense(self, founders_file, compact_every):
        dense = _dense_run(founders_file, self.GENS)
        opts = xs.GraphMeiosisOptions(group_fanin=4, compact_every=compact_every)
        _, genos = _run(_load(founders_file, opts), self.GENS)
        for gen in range(self.GENS + 1):
            np.testing.assert_array_equal(genos[gen], dense[gen])

    def test_compaction_keeps_groups_and_drops_dead_nodes(self, founders_file):
        """Compacted graph = live part of the uncompacted one; groups survive."""
        uncompacted, _ = _run(_load(founders_file, xs.GraphMeiosisOptions(group_fanin=4)),
                              self.GENS)
        compacted, _ = _run(_load(founders_file, xs.GraphMeiosisOptions(
            group_fanin=4, compact_every=1)), self.GENS)
        grg = compacted._grg
        assert (grg.num_nodes, grg.num_edges) == _live_counts(uncompacted._grg)
        assert grg.num_edges < uncompacted._grg.num_edges
        assert grg.nodes_are_ordered
        assert _live_counts(grg) == (grg.num_nodes, grg.num_edges)  # nothing dead left
        assert _max_sample_parents(grg) <= 8
        stats = compacted.last_meiosis_stats
        assert stats["compacted"] is True
        assert stats["nodes_dropped"] > 0
        assert (stats["nodes"], stats["edges"]) == (grg.num_nodes, grg.num_edges)

    def test_bp_range_survives(self, founders_file):
        op = _load(founders_file, xs.GraphMeiosisOptions(group_fanin=4, compact_every=1))
        bp_range = tuple(op._grg.bp_range)
        final, _ = _run(op, 2)
        assert tuple(final._grg.bp_range) == bp_range

    def test_parent_graph_released(self, founders_file):
        op = _load(founders_file, xs.GraphMeiosisOptions(compact_every=1))
        rng = np.random.RandomState(0)
        assignment = RandomMating(offspring_per_pair=2).mate(op.samples, rng=rng)
        child = op.meiosis(assignment, _rmap(op), rng=rng)
        assert op.graph_released and not child.graph_released
        assert (op.n, op.m) == (child.n, child.m)  # served from metadata
        with pytest.raises(RuntimeError, match="no longer holds a graph"):
            op.matvec(np.ones(op.m))
        child.matvec(np.ones(child.m))


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def _grouped(self, founders_file, **kwargs):
        opts = xs.GraphMeiosisOptions(group_fanin=4, group_min_parents=8, **kwargs)
        final, genos = _run(_load(founders_file, opts), 2)
        return final, genos[-1]

    def _check_round_trip(self, tmp_path, op, genotypes):
        _save_graph_haplotypes_to_checkpoint(op, str(tmp_path), 2)
        loaded = _load_graph_haplotypes_from_checkpoint(str(tmp_path), 2)
        assert loaded.meiosis_options == xs.GraphMeiosisOptions(
            group_fanin=4, group_min_parents=8, compact_every=op.meiosis_options.compact_every)
        np.testing.assert_array_equal(loaded.to_dense().genotypes, genotypes)
        # The default (simplifying) save would splice out the one-child groups.
        assert _max_sample_parents(loaded._grg) <= 8
        return loaded

    def test_option_b_saved_unsimplified_without_patch(self, founders_file, tmp_path,
                                                        monkeypatch):
        op, genotypes = self._grouped(founders_file)
        monkeypatch.setattr(xs, "_save_grg_supports_drop_unreachable", lambda: False)
        with pytest.warns(UserWarning, match="drop_unreachable"):
            loaded = self._check_round_trip(tmp_path, op, genotypes)
        # Nothing dropped: dead nodes are kept.
        assert (loaded._grg.num_nodes, loaded._grg.num_edges) == (op._grg.num_nodes,
                                                                  op._grg.num_edges)

    @needs_drop_unreachable
    def test_option_b_saved_compacted(self, founders_file, tmp_path):
        op, genotypes = self._grouped(founders_file)
        loaded = self._check_round_trip(tmp_path, op, genotypes)
        assert (loaded._grg.num_nodes, loaded._grg.num_edges) == _live_counts(op._grg)

    def test_old_sidecar_loads_with_default_options(self, founders_file, tmp_path):
        op, _ = self._grouped(founders_file)
        _save_graph_haplotypes_to_checkpoint(op, str(tmp_path), 2)
        meta_path = os.path.join(str(tmp_path), "gen_2.grg.meta.npz")
        data = dict(np.load(meta_path, allow_pickle=True))
        for key in [k for k in data if k.startswith("meiosis_")]:
            del data[key]
        np.savez_compressed(meta_path, **data)
        loaded = _load_graph_haplotypes_from_checkpoint(str(tmp_path), 2)
        assert loaded.meiosis_options == xs.GraphMeiosisOptions()

    @needs_drop_unreachable
    def test_simulation_resume_matches_uninterrupted_run(self, founders_file, tmp_path):
        """Compacting simulation, checkpoint mid-run (a released generation is
        retained), resume: same genetic values as running straight through."""
        def make_sim():
            founders = _load(founders_file, xs.GraphMeiosisOptions(group_fanin=4,
                                                                   compact_every=1))
            arch = Architecture()
            arch.add('Y.G', GeneticComponent(AdditiveEffects.from_h2(h2=0.5, m=founders.m,
                                                                     seed=SEED)))
            arch.add('Y.E', NoiseComponent(variance=0.5))
            arch.add('Y', AggregationComponent('Y.G + Y.E'))
            return Simulation(founder_haplotypes=founders, architecture=arch,
                              mating_regime=RandomMating(offspring_per_pair=2),
                              recombination_map=_rmap(founders), seed=SEED,
                              retain_haplotypes=1, retain_phenotypes=6)

        straight = make_sim()
        straight.run(6)

        first = make_sim()
        first.run(4)
        assert first.haplotype_history[2].graph_released
        save_simulation_checkpoint(first, str(tmp_path))
        assert sorted(os.listdir(os.path.join(str(tmp_path), "haplotypes"))) == [
            "gen_3.grg", "gen_3.grg.meta.npz"]
        resumed = Simulation.from_checkpoint(str(tmp_path))
        assert resumed.haplotypes.meiosis_options.compact_every == 1
        resumed.continue_run(2)
        assert resumed.generation == straight.generation == 5
        np.testing.assert_array_equal(resumed.haplotypes.to_dense().genotypes,
                                      straight.haplotypes.to_dense().genotypes)
        for gen in (4, 5):
            np.testing.assert_allclose(resumed.phenotype_history[gen]['Y.G'],
                                       straight.phenotype_history[gen]['Y.G'], atol=1e-10)
