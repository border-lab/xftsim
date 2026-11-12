"""Option B scaling figure: per-haplotype edges added and held, vs the current code."""
import csv, json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
plt.rcParams.update({"font.size": 10, "figure.facecolor": BG, "axes.facecolor": BG})
L = lambda p: [json.loads(l) for l in open(p)][1:]
lane = [r for r in csv.DictReader(open(sys.argv[1])) if r["stage"] == "generation"]
runs = [("option B, 100k", "scale/B_100k_g100.jsonl", 2e5, ORANGE),
        ("option B, 250k", "scale/B_250k.jsonl", 5e5, AQUA),
        ("option B, 500k", "scale/B_500k.jsonl", 1e6, YELLOW)]


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel("generation", color=INK2); ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2)


fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
lg = [int(r["generation"]) for r in lane]
le = [int(r["edges"]) for r in lane]
a1.plot(lg[1:], [(le[i] - le[i - 1]) / 1e6 for i in range(1, len(le))], color=BLUE, lw=2, label="current code, 500k (Lane)")
a2.plot(lg[1:], [e / 1e6 for e in le[1:]], color=BLUE, lw=2, label="current code, 500k: all stored edges")
for name, path, nh, col in runs:
    rows = L(path)
    g = [r["gen"] for r in rows]
    a1.plot(g, [(r["added_offspring"] + r["added_groups"]) / nh for r in rows], color=col, lw=2, label=name)
    a2.plot(g, [r["live_edges"] / nh for r in rows], color=col, lw=2, label=f"{name}: live edges")
style(a1, "Edges added each generation, per haplotype", "edges per haplotype")
a1.set_ylim(bottom=0); a1.legend(frameon=False, fontsize=8.5, loc="upper left")
style(a2, "Edges held in the graph, per haplotype", "edges per haplotype (log scale)")
a2.set_yscale("log"); a2.legend(frameon=False, fontsize=8.5, loc="upper left")
fig.tight_layout()
fig.savefig("figs/fig_optionB_scaling.png", dpi=160)
print("wrote figs/fig_optionB_scaling.png")
