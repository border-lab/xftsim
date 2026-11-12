"""Tables for the option B scaling runs. Memory is estimated from live graph size:
134 B per node + 15-34 B per edge (mem_bench.py; Lane baselines give ~15-19 B/edge).

    python summarize_scale.py <runs dir> [current-code CSV from scaling_run.py, e.g. n500k_L20_g30.csv]
"""
import csv, json, os, sys

D = sys.argv[1]
LANE = sys.argv[2] if len(sys.argv) > 2 else ""
NODE_B, EDGE_LO, EDGE_HI = 134, 15, 34


def load(name):
    p = os.path.join(D, name + ".jsonl")
    return {r["gen"]: r for r in map(json.loads, open(p))} if os.path.exists(p) else {}


def mem(nodes, edges):
    return f"{(NODE_B * nodes + EDGE_LO * edges) / 1e9:.1f}-{(NODE_B * nodes + EDGE_HI * edges) / 1e9:.1f}"


for tag, n_hap in (("100k", 2e5), ("250k", 5e5), ("500k", 1e6), ("100k_g100", 2e5)):
    B = load(f"B_{tag}")
    C = load(f"C_{tag}") if not tag.endswith("g100") else {}
    if len(B) < 2:
        continue
    print(f"\n== {tag}  (option B vs current)")
    print("gen | B added/hap  B deg_rec  B visits/xo | B live edges  B live nodes  B mem GB | "
          "C added/hap  C deg_rec  C stored edges  C nodes  C mem GB")
    gens = sorted(g for g in B if g > 0)
    pick = [g for g in gens if g in (1, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100)] or gens
    if gens[-1] not in pick:
        pick.append(gens[-1])
    for g in pick:
        b = B[g]
        line = (f"{g:3d} | {(b['added_offspring'] + b['added_groups']) / n_hap:10.1f}  {b['deg_rec']:9.1f}  "
                f"{b['visits_per_xo']:10.1f} | {b['live_edges']:12,}  {b['live_nodes']:12,}  "
                f"{mem(b['live_nodes'], b['live_edges']):>9}")
        c = C.get(g)
        if c:
            line += (f" | {c['added_offspring'] / n_hap:10.1f}  {c['deg_rec']:8.1f}  {c['edges']:14,}  "
                     f"{c['node_slots']:9,}  {mem(c['node_slots'], c['edges']):>9}")
        print(line)

if LANE and os.path.exists(LANE):
    rows = [r for r in csv.DictReader(open(LANE)) if r["stage"] == "generation"]
    print("\n== Lane current code, 500k x 20 Mb (measured RSS)")
    for r in rows:
        if int(r["generation"]) in (1, 10, 20, 30):
            print(f"gen {r['generation']:>2}: edges {int(r['edges']):,}  nodes {int(r['nodes']):,}  "
                  f"rss {r['rss_gb']} GB  peak {r['peak_gb']} GB")
