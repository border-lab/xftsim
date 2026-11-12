"""Figures for the GRG edge-growth investigation (20k people x chr22 20 Mb, 30 generations)."""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = sys.argv[1]
OUT = sys.argv[2]
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def load(name):
    return [json.loads(l) for l in open(os.path.join(RUNS, name + ".jsonl"))]


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel("generation", color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)


plt.rcParams.update({"font.size": 10, "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
base = load("base30")
g = [r["gen"] for r in base[1:]]

# ---- Figure 1: why ----------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
rec_deg = [r["deg_rec"] for r in base[1:]]
tracts = [r["tracts_rec"] for r in base[1:]]
a1.plot(tracts, rec_deg, "-o", color=BLUE, lw=2, ms=4)
for r in base[1:]:
    if r["gen"] in (1, 10, 20, 30):
        a1.annotate(f"gen {r['gen']}", (r["tracts_rec"], r["deg_rec"]), textcoords="offset points",
                    xytext=(6, -12), color=INK2, fontsize=9)
style(a1, "Edges per recombinant offspring track its ancestry tracts", "edges added for one offspring haplotype")
a1.set_xlabel("ancestry tracts carried by the offspring haplotype", color=INK2)
a1.set_xlim(left=0); a1.set_ylim(bottom=0)

new = [r["new_edges"] / 1e6 for r in base[1:]]
n_hap = 40000
pred = [n_hap * ((1 - r["frac_recomb"]) * 1 + r["frac_recomb"] * r["deg_rec"]) / 1e6 for r in base[1:]]
a2.bar(g, new, color=BLUE, width=0.7, label="measured")
a2.plot(g, pred, color=ORANGE, lw=2, label="N_hap x [P(no xo) x 1 + P(xo) x edges per recombinant]")
style(a2, "Edges added each generation grow linearly", "edges added (millions)")
a2.legend(frameon=False, fontsize=8.5, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_edge_mechanism.png"), dpi=160)

# ---- Figure 2: fixes ----------------------------------------------------------
fig, (b1, b2) = plt.subplots(1, 2, figsize=(11, 4.2))
compact = load("compact1")
h4gc = load("hier4gc")
b1.plot(g, [r["edges"] / 1e6 for r in base[1:]], color=BLUE, lw=2, label="current (nothing removed)")
b1.plot([r["gen"] for r in compact[1:]], [r["compact_edges"] / 1e6 for r in compact[1:]],
        color=ORANGE, lw=2, label="+ drop dead nodes every generation")
b1.plot([r["gen"] for r in h4gc[1:]], [r["edges"] / 1e6 for r in h4gc[1:]],
        color=AQUA, lw=2, label="+ position hierarchy (prototype)")
style(b1, "Edges stored in the graph", "edges (millions)")
b1.set_ylim(bottom=0)
b1.legend(frameon=False, fontsize=8.5, loc="upper left")

h4 = load("hier4")
b2.plot(g, new, color=BLUE, lw=2, label="current")
b2.plot([r["gen"] for r in h4[1:]], [(r["new_edges"] + r["hier_extra_edges"]) / 1e6 for r in h4[1:]],
        color=AQUA, lw=2, label="position hierarchy (incl. its own edges)")
style(b2, "Edges added each generation", "edges added (millions)")
b2.set_ylim(bottom=0)
b2.legend(frameon=False, fontsize=8.5, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig_edge_fixes.png"), dpi=160)
print("wrote", OUT)
