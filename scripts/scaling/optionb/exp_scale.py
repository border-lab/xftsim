"""Option B (position hierarchy + dead-node GC) at scale, vs the current code.

Leaner than exp_hier.py so it runs at 500k: compact array-backed span cache,
numpy live marks, tract tracking as plain lists, no per-parent classification.

Per generation (one JSON line):
  added_offspring  edges created by meiosis (offspring up-edges + bubbles)
  added_groups     edges created by the hierarchy pass (= group nodes made)
  live_edges       edges after GC (every remaining edge has a live child)
  live_nodes       nodes reachable upward from the current samples
  node_slots       grg.num_nodes, including disconnected dead slots (prototype artifact)
  deg_rec          mean up-degree of recombinant offspring right after meiosis
  visits_per_xo    recombiner visits per crossover (does recursion get deeper?)
  t_meiosis / t_hier / t_gc / t_stats  seconds

--mode current runs the unmodified algorithm (no hierarchy, no GC) as the exactness twin.
G = X beta is saved for generations <= --save-g-gens.
"""
import argparse, bisect, json, os, pickle, resource, sys, time, warnings
from array import array
warnings.filterwarnings("ignore")
import numpy as np
import pygrgl
import xftsim.struct as xs
import xftsim.grg_recombination as xgr
import xftsim.grg_recombination_native as xgrn
from xftsim.reproduce import RecombinationMap
from xftsim.mate import RandomMating, LinearAssortativeMating
from xftsim.effect import AdditiveEffects

ap = argparse.ArgumentParser()
ap.add_argument("--founders", required=True)
ap.add_argument("--gens", type=int, default=20)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--mode", choices=["optionB", "current"], default="optionB")
ap.add_argument("--hier", type=int, default=4)
ap.add_argument("--hier-min", type=int, default=8)
ap.add_argument("--no-tracts", action="store_true")
ap.add_argument("--save-drop-every", type=int, default=0,
                help="compact via save_grg(allow_simplify=False, drop_unreachable=True) + reload every K gens "
                     "(needs the patched pygrgl)")
ap.add_argument("--no-gc", action="store_true", help="skip the Python mark-and-sweep (compaction does the dropping)")
ap.add_argument("--rebuild-every", type=int, default=0,
                help="copy the live graph into a fresh, renumbered MutableGRG every K gens (frees dead node slots)")
ap.add_argument("--save-g-gens", type=int, default=10)
ap.add_argument("--out", required=True)
a = ap.parse_args()

UP = pygrgl.TraversalDirection.UP
GB = 1e9


def rss_gb():
    import psutil
    return psutil.Process().memory_info().rss / GB


meta = pickle.load(open(a.founders + ".meta.pkl", "rb"))
grg0 = pygrgl.load_mutable_grg(a.founders + ".grg")
hap = xs.GraphHaplotypeOperator(grg0, samples=meta["samples"], variants=meta["variants"])
grg = hap._grg
m = hap.m
rmap = RecombinationMap.from_haplotypes_with_cM(hap)
beta = np.asarray(AdditiveEffects.from_h2(h2=0.5, m=m, seed=a.seed).effects, dtype=float).ravel()
mating = RandomMating(offspring_per_pair=2)
rng = np.random.RandomState(a.seed)

# mutation positions; meiosis re-sorts mutations by position, which is the identity
# here (founders are sorted, duplicates dropped) -- spot-checked every generation
pos_by_mut = np.fromiter((grg.get_mutation_by_id(i).position for i in range(m)), dtype=np.int64, count=m)
assert np.all(np.diff(pos_by_mut) >= 0)
check_ids = np.random.RandomState(0).choice(m, size=min(200, m), replace=False)

# --- crossover capture --------------------------------------------------------
recorded = []
_orig = xgr._switches_to_segments


def _rec(switches, pos_bp, hap_ids, genome_length):
    recorded.append(np.asarray(switches).tolist())
    return _orig(switches, pos_bp, hap_ids, genome_length)


xgr._switches_to_segments = _rec
_last = []
_orig_init = xgrn.NonDuplicationRecombination.__init__


def _init(self, *args, **kw):
    _orig_init(self, *args, **kw)
    _last[:] = [self]


xgrn.NonDuplicationRecombination.__init__ = _init

# --- ancestry tracts (lists: starts, founder ids) -----------------------------
tracts = None if a.no_tracts else [([0], [h]) for h in range(2 * hap.n)]


def splice(t0, t1, sw):
    cur = 0
    if sw and sw[0] == 0:
        cur, sw = 1, sw[1:]
    bounds = [0] + sw + [m]
    out_s, out_f = [], []
    for k in range(len(bounds) - 1):
        lo, hi = bounds[k], bounds[k + 1]
        s, f = (t0, t1)[cur]
        i = bisect.bisect_right(s, lo) - 1
        while i < len(s) and s[i] < hi:
            if not out_f or out_f[-1] != f[i]:
                out_s.append(max(s[i], lo)); out_f.append(f[i])
            i += 1
        cur ^= 1
    return out_s, out_f


# --- span cache: lo/hi per node id; lo = UNKNOWN or NONE sentinels ------------
UNKNOWN, NONE = -2, -1
span_lo = array("q")
span_hi = array("q")


def grow(n):
    if len(span_lo) < n:
        pad = n - len(span_lo)
        span_lo.extend([UNKNOWN] * pad)
        span_hi.extend([0] * pad)


def get_span(g, node):
    if span_lo[node] != UNKNOWN:
        return node
    stack = [(node, False)]
    while stack:
        v, done = stack.pop()
        if span_lo[v] != UNKNOWN:
            continue
        if done:
            lo, hi = None, None
            muts = g.get_mutations_for_node(v, False)
            if muts:
                p = pos_by_mut[muts]
                lo, hi = int(p.min()), int(p.max())
            for u in g.get_up_edges(v):
                if span_lo[u] >= 0:
                    lo = span_lo[u] if lo is None or span_lo[u] < lo else lo
                    hi = span_hi[u] if hi is None or span_hi[u] > hi else hi
            if lo is None:
                span_lo[v] = NONE
            else:
                span_lo[v] = lo; span_hi[v] = hi
        else:
            stack.append((v, True))
            for u in g.get_up_edges(v):
                if span_lo[u] == UNKNOWN:
                    stack.append((u, False))
    return node


def build_hierarchy(g, samples, B, kmin):
    grow(g.num_nodes)
    made = 0
    for s in samples:
        P = g.get_up_edges(s)
        if len(P) <= kmin:
            continue
        items = []
        for p in P:
            get_span(g, p)
            lo = span_lo[p]
            items.append((lo, span_hi[p], p) if lo >= 0 else (1 << 62, 1 << 62, p))
        items.sort()
        level = items
        while len(level) > B:
            nxt = []
            for k in range(0, len(level), B):
                chunk = level[k:k + B]
                if len(chunk) == 1:
                    nxt.append(chunk[0]); continue
                gn = g.make_node()
                made += 1
                for c in chunk:
                    g.connect(c[2], gn)
                lo = min(c[0] for c in chunk); hi = max(c[1] for c in chunk)
                grow(gn + 1)
                span_lo[gn] = lo; span_hi[gn] = hi
                nxt.append((lo, hi, gn))
            level = nxt
        for p in P:
            g.disconnect(p, s)
        for c in level:
            g.connect(c[2], s)
    return made


live_prev = None


def gc_dead(g, samples):
    """Mark ancestors of current samples; drop the up-edges of nodes that just died."""
    global live_prev
    live_ids = np.asarray(pygrgl.get_topo_order(g, UP, samples), dtype=np.int64)
    live = np.zeros(g.num_nodes, dtype=bool)
    live[live_ids] = True
    if live_prev is None:
        cand = np.flatnonzero(~live)
    else:
        prev = np.zeros(g.num_nodes, dtype=bool)
        prev[:live_prev.size] = live_prev
        prev[live_prev.size:] = True            # nodes created this generation
        cand = np.flatnonzero(prev & ~live)
    removed = 0
    for v in cand.tolist():
        for p in g.get_up_edges(v):
            g.disconnect(p, v)
            removed += 1
    live_prev = live
    return int(live_ids.size), removed


def rebuild_live(g, samples):
    """Copy only the live graph (ancestors of the current samples) into a new MutableGRG.

    Samples keep their order (node i = sample i); other live nodes are numbered in
    upward topological order, so every parent gets a larger id than its children.
    Mutation ids are preserved; mutations sitting on dead nodes get no node (count 0).
    Stands in for a native "drop unreachable nodes" compaction; groups are kept."""
    global live_prev, span_lo, span_hi
    grow(g.num_nodes)
    order = pygrgl.get_topo_order(g, UP, samples)
    n_s = len(samples)
    old2new = np.full(g.num_nodes, -1, dtype=np.int64)
    old2new[np.asarray(samples, dtype=np.int64)] = np.arange(n_s)
    h = pygrgl.MutableGRG(n_s, g.ploidy, g.is_phased, initial_node_capacity=len(order))
    sample_set = old2new >= 0
    for v in order:
        if not sample_set[v]:
            old2new[v] = h.make_node()
    o2n = old2new.tolist()
    for v in order:
        nv = o2n[v]
        for p in g.get_up_edges(v):
            h.connect(o2n[p], nv)
    for mid, node in g.get_mutation_node_pairs():
        nn = o2n[node] if node != pygrgl.INVALID_NODE and node < len(o2n) else -1
        new_id = h.add_mutation(g.get_mutation_by_id(mid), nn if nn >= 0 else pygrgl.INVALID_NODE)
        assert new_id == mid
    h.get_mutations_for_node(0, True)   # sort the node->mutation index (native recombiner forbids lazy sorting)
    # carry the span cache over to the new ids
    lo_old = np.frombuffer(span_lo, dtype=np.int64)[:g.num_nodes]
    hi_old = np.frombuffer(span_hi, dtype=np.int64)[:g.num_nodes]
    keep = np.flatnonzero(old2new >= 0)
    lo_new = np.full(h.num_nodes, UNKNOWN, dtype=np.int64)
    hi_new = np.zeros(h.num_nodes, dtype=np.int64)
    lo_new[old2new[keep]] = lo_old[keep]
    hi_new[old2new[keep]] = hi_old[keep]
    span_lo = array("q", lo_new.tobytes()); span_hi = array("q", hi_new.tobytes())
    live_prev = np.ones(h.num_nodes, dtype=bool)
    return h


fh = open(a.out + ".jsonl", "w")


def emit(row):
    fh.write(json.dumps(row) + "\n"); fh.flush()
    print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)


Gs = {"0": hap.matvec(beta)}
emit(dict(gen=0, mode=a.mode, n_hap=2 * hap.n, m=m, edges=grg.num_edges, nodes=grg.num_nodes,
          rss_gb=round(rss_gb(), 2)))
for gen in range(1, a.gens + 1):
    assignment = mating.mate(hap.samples, rng=rng)
    e0, n0 = grg.num_edges, grg.num_nodes
    recorded.clear()
    t = time.perf_counter()
    hap = hap.meiosis(assignment, rmap, rng=rng)
    t_meiosis = time.perf_counter() - t
    grg = hap._grg
    audit = _last[0]._impl.audit
    _last.clear()          # the recombiner holds a reference to the graph
    added_offspring = grg.num_edges - e0
    bubbles = grg.num_nodes - n0 - 2 * hap.n
    assert all(grg.get_mutation_by_id(int(i)).position == pos_by_mut[i] for i in check_ids)

    t = time.perf_counter()
    samples = grg.get_sample_nodes()
    deg = np.fromiter((grg.num_up_edges(s) for s in samples), dtype=np.int64, count=len(samples))
    nsw = np.fromiter((len(sw) - (1 if sw and sw[0] == 0 else 0) for sw in recorded), dtype=np.int64,
                      count=len(recorded))
    rec = nsw > 0
    row = dict(gen=gen, t_meiosis=round(t_meiosis, 2), added_offspring=added_offspring, bubbles=bubbles,
               frac_recomb=round(float(rec.mean()), 4), crossovers=int(nsw.sum()),
               deg_rec=round(float(deg[rec].mean()), 2), deg_nonrec=round(float(deg[~rec].mean()), 3),
               visits=audit["visits"], visits_per_xo=round(audit["visits"] / max(1, int(nsw.sum())), 1))
    if tracts is not None:
        mat, pat = assignment.maternal_idx, assignment.paternal_idx
        new = []
        for i in range(hap.n):
            for k, par in ((0, int(mat[i])), (1, int(pat[i]))):
                sw = recorded[2 * i + k]
                new.append(splice(tracts[2 * par], tracts[2 * par + 1], sw))
        tracts = new
        ntr = np.fromiter((len(tr[1]) for tr in tracts), dtype=np.int64, count=len(tracts))
        row.update(tracts_rec=round(float(ntr[rec].mean()), 2),
                   deg_per_tract_rec=round(float(deg[rec].sum() / ntr[rec].sum()), 2))
    row["t_stats"] = round(time.perf_counter() - t, 2)

    if a.mode == "optionB":
        t = time.perf_counter()
        groups = build_hierarchy(grg, samples, a.hier, a.hier_min)
        row.update(added_groups=groups, t_hier=round(time.perf_counter() - t, 2))
        if not a.no_gc:
            t = time.perf_counter()
            live_nodes, removed = gc_dead(grg, samples)
            row.update(live_nodes=live_nodes, gc_removed=removed, t_gc=round(time.perf_counter() - t, 2))
            row["live_edges"] = grg.num_edges        # after GC every remaining edge has a live child
        if a.save_drop_every and gen % a.save_drop_every == 0:
            import gc as _gc
            t = time.perf_counter()
            path = a.out + ".compact.grg"
            pygrgl.save_grg(grg, path, allow_simplify=False, drop_unreachable=True)
            t_save = time.perf_counter() - t
            meta_s, meta_v, gnum = hap.samples, hap.variants, hap.generation
            del hap, grg, samples                    # free the old graph before loading the new one
            _gc.collect()
            rss_between = rss_gb()
            hap = xs.GraphHaplotypeOperator(pygrgl.load_mutable_grg(path), generation=gnum,
                                            samples=meta_s, variants=meta_v)
            os.remove(path)
            grg = hap._grg
            samples = grg.get_sample_nodes()
            span_lo = array("q"); span_hi = array("q")   # node ids were renumbered
            live_prev = np.ones(grg.num_nodes, dtype=bool)
            row.update(compact_nodes=grg.num_nodes, compact_edges=grg.num_edges, t_compact_save=round(t_save, 2),
                       t_compact=round(time.perf_counter() - t, 2), rss_between_gb=round(rss_between, 2),
                       nodes_ordered=grg.nodes_are_ordered)
            row.setdefault("live_nodes", grg.num_nodes); row.setdefault("live_edges", grg.num_edges)
        if a.rebuild_every and gen % a.rebuild_every == 0:
            t = time.perf_counter()
            new = rebuild_live(grg, samples)
            assert new.num_edges == row["live_edges"], (new.num_edges, row["live_edges"])
            hap = xs.GraphHaplotypeOperator(new, generation=hap.generation, samples=hap.samples,
                                            variants=hap.variants)
            del new
            grg = hap._grg
            row.update(t_rebuild=round(time.perf_counter() - t, 2))
    row.update(edges=grg.num_edges, node_slots=grg.num_nodes, rss_gb=round(rss_gb(), 2))
    if gen <= a.save_g_gens:
        Gs[str(gen)] = hap.matvec(beta)
    emit(row)

np.savez_compressed(a.out + ".G.npz", **Gs)
print("done", flush=True)
