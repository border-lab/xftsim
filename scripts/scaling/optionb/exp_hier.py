"""Instrumented GRG meiosis: where do the edges come from, and what do compaction passes recover?

Per generation it records
  * nodes / edges added, bubbles (= new nodes - new haplotypes)
  * each offspring haplotype's up-degree (edges added for that haplotype), split by
    the number of crossovers in its meiosis, and its number of ancestry tracts
    (maximal runs inherited from one founder haplotype), tracked exactly here
  * what each offspring is attached to: founder-graph nodes, bubbles, or
    earlier-generation haplotype nodes
  * the "live" graph (ancestors of current samples) vs the stored graph
  * optional: what save_grg(simplify) and MutableGRG.reduce() would leave

--compact-every K actually replaces the graph by its simplified reload every K
generations (and --reduce also runs reduce_until on it), so later generations are
built on the compacted graph. G = X beta is written per generation for exactness
checks against the baseline run.
"""
import argparse, bisect, csv, json, os, pickle, sys, tempfile, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pygrgl
import xftsim.struct as xs
import xftsim.grg_recombination as xgr
from xftsim.reproduce import RecombinationMap
from xftsim.mate import RandomMating
from xftsim.effect import AdditiveEffects

ap = argparse.ArgumentParser()
ap.add_argument("--founders", required=True)
ap.add_argument("--gens", type=int, default=20)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--measure-every", type=int, default=0, help="non-destructive simplify/reduce estimate")
ap.add_argument("--compact-every", type=int, default=0)
ap.add_argument("--reduce", action="store_true")
ap.add_argument("--no-live", action="store_true")
ap.add_argument("--cm-scale", type=float, default=1.0, help="stretch the genetic map (more crossovers per meiosis)")
ap.add_argument("--hier", type=int, default=2, help="fan-in of the per-offspring position hierarchy")
ap.add_argument("--gc", action="store_true", help="disconnect nodes with no current descendants each generation")
ap.add_argument("--save-mode", choices=["none", "simplify", "plain"], default="none",
                help="save+reload: simplify (before grouping) or plain allow_simplify=False (after GC)")
ap.add_argument("--save-every", type=int, default=1)
ap.add_argument("--hier-min", type=int, default=4, help="only regroup offspring with more parents than this")
ap.add_argument("--out", required=True)
a = ap.parse_args()

tmpdir = tempfile.mkdtemp(dir=os.path.dirname(a.out) or ".")


def load_op(path, samples, variants, generation):
    g = pygrgl.load_mutable_grg(path)
    return xs.GraphHaplotypeOperator(g, generation=generation, samples=samples, variants=variants)


meta = pickle.load(open(a.founders + ".meta.pkl", "rb"))
hap = load_op(a.founders + ".grg", meta["samples"], meta["variants"], 0)
variants = meta["variants"]
grg = hap._grg
N0_nodes = grg.num_nodes
m = hap.m
from types import SimpleNamespace as _NS
_v = hap.variants
rmap = RecombinationMap.from_haplotypes_with_cM(_NS(variants=_NS(
    pos_cM=np.asarray(_v.pos_cM) * a.cm_scale, chrom=_v.chrom, vid=_v.vid, pos_bp=_v.pos_bp)))
beta = AdditiveEffects.from_h2(h2=0.5, m=m, seed=a.seed)
beta_vec = np.asarray(beta.effects, dtype=float).ravel()
mating = RandomMating(offspring_per_pair=2)
rng = np.random.RandomState(a.seed)

# Ancestry tracts per haplotype: (starts[], founder_hap[]) in variant-index space.
tracts = [(np.array([0]), np.array([h])) for h in range(2 * hap.n)]

recorded = []
_orig = xgr._switches_to_segments


def _rec(switches, pos_bp, hap_ids, genome_length):
    recorded.append(np.asarray(switches).copy())
    return _orig(switches, pos_bp, hap_ids, genome_length)


xgr._switches_to_segments = _rec

import xftsim.grg_recombination_native as xgrn
_last_recomb = []
_orig_init = xgrn.NonDuplicationRecombination.__init__


def _init(self, *args, **kw):
    _orig_init(self, *args, **kw)
    _last_recomb[:] = [self]


xgrn.NonDuplicationRecombination.__init__ = _init


def splice(t0, t1, switches):
    """Offspring tracts from a parent's two tract lists and its switch loci."""
    sw = list(np.asarray(switches))
    cur = 0
    if sw and sw[0] == 0:
        cur = 1
        sw = sw[1:]
    bounds = [0] + sw + [m]
    srcs = (t0, t1)
    out_s, out_f = [], []
    for k in range(len(bounds) - 1):
        lo, hi = bounds[k], bounds[k + 1]
        s, f = srcs[cur]
        i = bisect.bisect_right(s, lo) - 1
        while i < len(s) and s[i] < hi:
            start = max(s[i], lo)
            if out_f and out_f[-1] == f[i]:
                pass  # same founder continues across the switch
            else:
                out_s.append(start); out_f.append(f[i])
            i += 1
        cur ^= 1
    return (np.array(out_s), np.array(out_f))


def live_stats(g):
    samples = list(g.get_sample_nodes())
    live = pygrgl.get_topo_order(g, pygrgl.TraversalDirection.UP, samples)
    le = 0
    for v in live:
        le += g.num_up_edges(v)
    return len(live), le


def save_reload(g, gen, samples, simplify):
    global live_prev
    path = os.path.join(tmpdir, f"s{gen}.grg")
    t = time.perf_counter()
    pygrgl.save_grg(g, path, allow_simplify=simplify)
    op = load_op(path, samples, variants, gen)
    os.remove(path)
    span.clear()          # node ids are renumbered by the reload
    live_prev = None
    return op, time.perf_counter() - t


def simplify_reload(g, gen, samples):
    path = os.path.join(tmpdir, f"g{gen}.grg")
    t = time.perf_counter()
    pygrgl.save_grg(g, path, allow_simplify=True)
    op = load_op(path, samples, variants, gen)
    t_save = time.perf_counter() - t
    os.remove(path)
    return op, t_save


fh = open(a.out + ".jsonl", "w")
INF = float("inf")
span = {}  # node -> (lo, hi) ancestral bp span; immutable once computed (bubbling keeps the union)


def get_span(g, node, pos_by_mut):
    if node in span:
        return span[node]
    stack = [(node, False)]
    while stack:
        v, done = stack.pop()
        if v in span:
            continue
        if done:
            lo, hi = INF, -INF
            muts = g.get_mutations_for_node(v, False)
            if len(muts):
                p = pos_by_mut[np.asarray(muts)]
                lo, hi = float(p.min()), float(p.max())
            for u in g.get_up_edges(v):
                su = span[u]
                if su:
                    lo = min(lo, su[0]); hi = max(hi, su[1])
            span[v] = None if lo == INF else (lo, hi)
        else:
            stack.append((v, True))
            for u in g.get_up_edges(v):
                if u not in span:
                    stack.append((u, False))
    return span[node]


live_prev = None


def gc_dead(g):
    """Drop every edge whose child can no longer reach a current sample (hierarchy-safe;
    unlike save_grg(simplify) it never splices out live pass-through nodes)."""
    global live_prev
    samples = list(g.get_sample_nodes())
    live_now = set(pygrgl.get_topo_order(g, pygrgl.TraversalDirection.UP, samples))
    cand = live_prev if live_prev is not None else range(g.num_nodes)
    removed = 0
    for v in cand:
        if v not in live_now:
            for p in list(g.get_up_edges(v)):
                g.disconnect(p, v)
                removed += 1
    live_prev = live_now
    return removed


def build_hierarchy(g, samples, B, kmin):
    """Replace each sample's long parent list by a B-ary tree of mutation-free group
    nodes over its parents sorted by ancestral span, so a descendant that inherits a
    sub-interval can attach to a few groups instead of every piece."""
    pos_by_mut = np.array([g.get_mutation_by_id(i).position for i in range(g.num_mutations)])
    made = 0
    for s in samples:
        P = list(g.get_up_edges(s))
        if len(P) <= kmin:
            continue
        items = []
        for p in P:
            sp = get_span(g, p, pos_by_mut)
            items.append((sp if sp else (INF, INF), p))
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
                for _, p in chunk:
                    g.connect(p, gn)
                sp = (min(c[0][0] for c in chunk), max(c[0][1] for c in chunk))
                span[gn] = sp
                nxt.append((sp, gn))
            level = nxt
        for p in P:
            g.disconnect(p, s)
        for _, p in level:
            g.connect(p, s)
    return made
W = None
offspring_ids_all = set()
Gs = {}


def emit(row):
    fh.write(json.dumps(row) + "\n"); fh.flush()
    print(" ".join(f"{k}={v}" for k, v in row.items()), flush=True)


Gs[0] = hap.matvec(beta_vec)
emit(dict(gen=0, nodes=grg.num_nodes, edges=grg.num_edges))
for gen in range(1, a.gens + 1):
    assignment = mating.mate(hap.samples, rng=rng)
    n0, e0 = grg.num_nodes, grg.num_edges
    recorded.clear()
    t = time.perf_counter()
    hap = hap.meiosis(assignment, rmap, rng=rng)
    t_meiosis = time.perf_counter() - t
    audit = _last_recomb[0]._impl.audit if _last_recomb else {}
    grg = hap._grg
    n_hap = 2 * hap.n
    new_nodes, new_edges = grg.num_nodes - n0, grg.num_edges - e0
    samples = list(grg.get_sample_nodes())

    # per-offspring stats
    deg = np.array([grg.num_up_edges(s) for s in samples])
    nsw = np.array([len(sw) - (1 if len(sw) and sw[0] == 0 else 0) for sw in recorded])
    mat, pat = assignment.maternal_idx, assignment.paternal_idx
    new_tracts = []
    for i in range(hap.n):
        for k, par in ((0, int(mat[i])), (1, int(pat[i]))):
            new_tracts.append(splice(tracts[2 * par], tracts[2 * par + 1], recorded[2 * i + k]))
    tracts = new_tracts
    ntr = np.array([len(t[1]) for t in tracts])

    # what offspring attach to
    kinds = {"founder": 0, "prev_hap": 0, "bubble": 0}
    for s in samples:
        for p in grg.get_up_edges(s):
            if p < N0_nodes:
                kinds["founder"] += 1
            elif p in offspring_ids_all:
                kinds["prev_hap"] += 1
            else:
                kinds["bubble"] += 1
    offspring_ids_all.update(samples)

    rec = nsw > 0
    save_info = {}
    if a.save_mode == "simplify" and gen % a.save_every == 0:
        nb, eb = grg.num_nodes, grg.num_edges
        hap, ts = save_reload(grg, gen, hap.samples, True)
        grg = hap._grg
        samples = list(grg.get_sample_nodes())
        save_info = dict(save_nodes_before=nb, save_edges_before=eb, save_nodes_after=grg.num_nodes,
                         save_edges_after=grg.num_edges, t_save=round(ts, 2))
    t = time.perf_counter()
    e_before = grg.num_edges
    groups = build_hierarchy(grg, samples, a.hier, a.hier_min) if a.hier else 0
    t_hier = time.perf_counter() - t
    hier_extra_edges = grg.num_edges - e_before
    t = time.perf_counter()
    gc_removed = gc_dead(grg) if a.gc else 0
    t_gc = time.perf_counter() - t
    if a.save_mode == "plain" and gen % a.save_every == 0:
        nb, eb = grg.num_nodes, grg.num_edges
        hap, ts = save_reload(grg, gen, hap.samples, False)
        grg = hap._grg
        save_info = dict(save_nodes_before=nb, save_edges_before=eb, save_nodes_after=grg.num_nodes,
                         save_edges_after=grg.num_edges, t_save=round(ts, 2))
    row = dict(**save_info, visits=audit.get("visits"), attaches=audit.get("connect_calls_in_attach"),
               gc_removed=gc_removed, t_gc=round(t_gc, 2), groups=groups, hier_extra_edges=hier_extra_edges, t_hier=round(t_hier, 2), edges_after_hier=grg.num_edges,
               gen=gen, t_meiosis=round(t_meiosis, 2), nodes=grg.num_nodes, edges=grg.num_edges,
               new_nodes=new_nodes, new_edges=new_edges, bubbles=new_nodes - n_hap,
               sum_offspring_deg=int(deg.sum()),
               frac_recomb=round(rec.mean(), 3),
               deg_nonrec=round(deg[~rec].mean(), 2), deg_rec=round(deg[rec].mean(), 2),
               tracts_mean=round(ntr.mean(), 2), tracts_rec=round(ntr[rec].mean(), 2),
               deg_per_tract_rec=round(deg[rec].sum() / ntr[rec].sum(), 2),
               att_founder=kinds["founder"], att_bubble=kinds["bubble"], att_prev_hap=kinds["prev_hap"])
    for k in (1, 2, 3):
        sel = nsw == k
        row[f"deg_x{k}"] = round(deg[sel].mean(), 1) if sel.any() else ""
    if not a.no_live:
        ln, le = live_stats(grg)
        row.update(live_nodes=ln, live_edges=le)
    if a.measure_every and gen % a.measure_every == 0:
        op2, ts = simplify_reload(grg, gen, hap.samples)
        row.update(simpl_nodes=op2._grg.num_nodes, simpl_edges=op2._grg.num_edges, t_simpl=round(ts, 2))
        t = time.perf_counter()
        op2._grg.reduce_until(10, 1000, 0.8)
        row.update(red_nodes=op2._grg.num_nodes, red_edges=op2._grg.num_edges,
                   t_reduce=round(time.perf_counter() - t, 2))
        del op2
    if a.compact_every and gen % a.compact_every == 0:
        t = time.perf_counter()
        hap, _ = simplify_reload(grg, gen, hap.samples)
        if a.reduce:
            hap._grg.reduce_until(10, 1000, 0.8)
        grg = hap._grg
        # node ids are renumbered by the reload: forget id-based bookkeeping
        N0_nodes = -1
        offspring_ids_all = set(grg.get_sample_nodes())
        row.update(compact_nodes=grg.num_nodes, compact_edges=grg.num_edges,
                   t_compact=round(time.perf_counter() - t, 2))
    Gs[gen] = hap.matvec(beta_vec)
    emit(row)

np.savez_compressed(a.out + ".G.npz", **{str(k): v for k, v in Gs.items()})
print("done", flush=True)
