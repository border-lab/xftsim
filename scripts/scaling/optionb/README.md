# GRG edge growth and option B (position hierarchy)

Prototype scripts behind the edge-growth investigation (2026-11-12). They
drive `GraphHaplotypeOperator.meiosis` directly, using the same founders, mating
and seeds as `../scaling_run.py`, and log one JSON line per generation.

## Findings these scripts produced

- **Why edges grow quadratically.** GRG edges carry no genomic interval. An offspring
  that inherits part of a parent haplotype therefore attaches to the parent's
  pieces (founder-graph nodes and bubbles), not to the parent itself. That costs
  about 13-18 edges per ancestry tract, and tracts accumulate by about one per
  crossover per generation. On top of that, nothing is ever deleted, and most
  stored edges are dead lineages.
- **Option B.** After meiosis, each offspring with many parents has them grouped,
  sorted by ancestral span, into a 4-ary tree of mutation-free group nodes. Later
  descendants then attach to a few groups instead of re-listing every piece.
  - Edges added per haplotype per generation stay flat at about 13.8 (100k, 250k,
    500k; 100 generations at 100k).
  - The live graph levels off at about 90-125 edges and 25-34 nodes per haplotype.
  - Genotypes match the current code exactly.

## Scripts

| Script | Purpose |
| --- | --- |
| `build_founders.py` | Build stdpopsim founders once and cache them (`.grg` + `.meta.pkl`). |
| `exp_edges.py` | Instrumented current code: per-offspring degree vs ancestry tracts, live vs stored graph, simplify/`reduce()` estimates, `--compact-every`. |
| `exp_hier.py` | First option B prototype, with save/reload variants (`--save-mode`). |
| `exp_scale.py` | Lean option B runner for 100k-500k. `--mode current` is the exactness twin. `--rebuild-every K` copies the live graph into a fresh MutableGRG (no grgl changes needed). `--save-drop-every K` compacts with `save_grg(allow_simplify=False, drop_unreachable=True)`. |
| `check_exact.py` | Compare per-generation genetic values between runs. |
| `mem_bench.py` | Bytes per node and per edge in a MutableGRG. |
| `summarize_scale.py`, `plot_edges.py`, `plot_scale.py` | Tables and figures. |

`--save-drop-every` needs pygrgl built from the grgl branch `drop-unreachable`
(commit `4e07c3c` on top of `8e716cb`, in the local `grgl/` clone). That
branch adds the `drop_unreachable` save option, and also fixes the inverted
seed-list check in `get_bfs_order` and `get_dfs_order`.

## Example

```bash
python build_founders.py --people 20000 --out cache/f20k_L20
python exp_scale.py --founders cache/f20k_L20 --gens 10 --mode current --out runs/cur
python exp_scale.py --founders cache/f20k_L20 --gens 30 --rebuild-every 2 --out runs/optB
python check_exact.py runs/cur.G.npz runs/optB.G.npz
```

## Limits

- The group builder and GC are Python, so they are slow and memory-heavy at 500k.
- pygrgl cannot delete nodes, so dead slots accumulate between compactions.
- For accurate time and memory, port the group builder to the native recombiner
  (`xftsim/native`) and compact with the `drop_unreachable` save.
