"""Bytes per node and per edge in a pygrgl MutableGRG (RSS deltas)."""
import gc, sys, numpy as np, psutil, pygrgl
proc = psutil.Process()
rss = lambda: proc.memory_info().rss
g = pygrgl.load_mutable_grg(sys.argv[1])
base_nodes = g.num_nodes
gc.collect(); r0 = rss()
N = 3_000_000
first = g.make_node()
for _ in range(N - 1):
    g.make_node()
gc.collect(); r1 = rss()
print(f"bytes/node (no edges): {(r1 - r0) / N:.1f}")
# give each new node 4 parents among earlier new nodes + 1 child: like a group node
rng = np.random.default_rng(0)
E = 0
ids = np.arange(first, first + N)
for k in range(4):
    src = ids[:-1]; dst = ids[1:]
    perm = rng.permutation(N - 1)
    for s, d in zip(src[perm].tolist(), dst.tolist()):
        if s < d:
            g.connect(s, d); E += 1
gc.collect(); r2 = rss()
print(f"edges added: {E:,}; bytes/edge: {(r2 - r1) / E:.1f} (each edge is stored as a down- and an up-edge)")
