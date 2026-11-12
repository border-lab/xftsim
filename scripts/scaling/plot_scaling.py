#!/usr/bin/env python
"""Figures and table for an xftsim + GRG scaling sweep (needs only matplotlib).

    python scripts/scaling/plot_scaling.py scaling_results/lane_20261106 \
        --out scaling_results/figures

Reads every finished run (<name>.csv + <name>.json from scaling_run.py) in the
results directory and writes:
  fig1_run_costs.png          time (founders vs. forward generations) and peak memory per run
  fig2_time_horizon.png       graph growth and time per generation over a long run
  fig3_cost_vs_graph_size.png time per generation and peak memory against graph size
  scaling_table.csv           one row per run
Times are summed from each stage's measured seconds. Peak memory is the larger
of the Python process and the `grg convert` child process.
"""
import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, MUTED, BASELINE, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7", "#e9e8e3"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]      # categorical slots 1-3, validated all-pairs


def load(folder):
    runs = {}
    for path in glob.glob(os.path.join(folder, "*.json")):
        info = json.load(open(path))
        if info["status"] != "done":
            continue
        args, rows = info["args"], list(csv.DictReader(open(path[:-5] + ".csv")))
        f = next(r for r in rows if r["stage"] == "founders")
        gens = [r for r in rows if r["stage"] == "generation" and r["generation"] != "0"]
        cpu = info["machine"].get("cpu", "").replace("Intel(R) Xeon(R) ", "Xeon ")
        span = float(f["span_mb"])
        region = (f"{args['chrom']} whole ({span:.0f} Mb)" if args["length_mb"] <= 0
                  else f"{args['chrom']} {args['length_mb']:g} Mb")
        runs[args["name"]] = dict(
            name=args["name"], people=args["people"], region=region, span_mb=span,
            variants=int(f["variants"]), founders_s=float(f["seconds"]),
            convert_s=float(f["seconds_ts_to_grg"]), cpu=cpu.replace("CPU ", "").split(" @")[0],
            gen=[int(r["generation"]) for r in gens], gen_s=[float(r["seconds"]) for r in gens],
            edges=[float(r["edges"]) for r in gens], peak=[float(r["peak_gb"]) for r in gens],
            founder_edges=float(f["edges"]),
            peak_gb=max(float(r[k]) for r in rows for k in ("peak_gb", "peak_child_gb")))
    return runs


def people_label(n):
    return f"{n // 1_000_000}M" if n >= 1_000_000 else f"{n // 1000}k"


def count_label(n):
    return f"{n / 1e6:.2f}M" if n >= 1_000_000 else f"{n / 1e3:.0f}k"


def size_label(r):
    """'500k × 6.35M' : individuals × variants."""
    return f"{people_label(r['people'])} × {count_label(r['variants'])}"


def run_label(r):
    return f"{size_label(r)} · {r['region']} · {len(r['gen'])} gens"


def style(ax, title, xlabel=None, ylabel=None):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=10)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=10)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9)


def line(ax, x, y, color, label=None, markers=None, offset=(8, 0), leader=False):
    if markers is None:
        markers = len(x) <= 12               # markers only on short series
    ax.plot(x, y, color=color, linewidth=2, marker="o" if markers else None, markersize=6,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
    ax.plot(x[-1:], y[-1:], color=color, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4)
    if label:
        ax.annotate(label, (x[-1], y[-1]), xytext=offset, textcoords="offset points",
                    va="center", fontsize=9, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkB=5) if leader else None)


def save(fig, out, name, footnote=None):
    if footnote:
        fig.text(0.01, 0.015, footnote, fontsize=8.5, color=MUTED)
        fig.tight_layout(rect=(0, 0.05, 0.98, 0.93))
    else:
        fig.tight_layout(rect=(0, 0, 0.98, 0.93))
    fig.savefig(os.path.join(out, name), dpi=200, facecolor=SURFACE)
    plt.close(fig)


def cpu_note(runs):
    cpus = sorted({r["cpu"] for r in runs})
    if len(cpus) < 2:
        return None, {}
    old = [c for c in cpus if "E5" in c] or cpus[-1:]
    marks = {r["name"]: "†" for r in runs if r["cpu"] in old}
    return (f"† ran on an older node ({old[0]}); the rest on {', '.join(c for c in cpus if c not in old)}. "
            "Times for † runs are inflated by the slower CPU.", marks)


def fig_run_costs(runs, out):
    order = sorted(runs.values(), key=lambda r: r["founders_s"] + sum(r["gen_s"]))
    note, marks = cpu_note(order)
    labels = [run_label(r) + marks.get(r["name"], "") for r in order]
    y = range(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(12, 0.6 * len(order) + 2.2), facecolor=SURFACE,
                             gridspec_kw=dict(width_ratios=(1.35, 1)))
    ax = axes[0]
    style(ax, "Run time (minutes)")
    founders = [r["founders_s"] / 60 for r in order]
    forward = [sum(r["gen_s"]) / 60 for r in order]
    ax.barh(y, founders, height=0.6, color=SERIES[0], edgecolor=SURFACE, linewidth=2,
            label="building founders (msprime + GRG)")
    ax.barh(y, forward, left=founders, height=0.6, color=SERIES[1], edgecolor=SURFACE, linewidth=2,
            label="forward generations")
    for i, (a, b) in enumerate(zip(founders, forward)):
        ax.annotate(f"{a + b:.0f} min", (a + b, i), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK2)
    ax.set_yticks(list(y), labels)
    ax.set_xlim(0, max(a + b for a, b in zip(founders, forward)) * 1.15)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK2)
    ax = axes[1]
    style(ax, "Peak memory (GB)")
    peaks = [r["peak_gb"] for r in order]
    ax.barh(y, peaks, height=0.6, color=SERIES[2], edgecolor=SURFACE, linewidth=2)
    for i, p in enumerate(peaks):
        ax.annotate(f"{p:.0f} GB", (p, i), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK2)
    ax.set_yticks(list(y), [""] * len(order))
    ax.set_xlim(0, max(peaks) * 1.15)
    ax.grid(axis="y", visible=False)
    fig.suptitle("What each run cost: individuals × variants (Lane cluster, one CPU core per run)",
                 x=0.01, ha="left", fontsize=13, color=INK)
    save(fig, out, "fig1_run_costs.png", note)


def fig_horizon(runs, out):
    r = max(runs.values(), key=lambda r: len(r["gen"]))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=SURFACE)
    ax = axes[0]
    style(ax, "Graph size", "generation", "edges (billions)")
    gens = [0] + r["gen"]
    edges = [r["founder_edges"] / 1e9] + [e / 1e9 for e in r["edges"]]
    line(ax, gens, edges, SERIES[0], f"{edges[-1]:.2f}B")
    added = [(b - a) / 1e6 for a, b in zip([r["founder_edges"]] + r["edges"], r["edges"])]
    ax.text(0.03, 0.92, f"edges added per generation:\n{added[0]:.0f}M (gen 1) → {added[-1]:.0f}M "
            f"(gen {r['gen'][-1]})", transform=ax.transAxes, va="top", fontsize=9, color=INK2)
    ax.set_ylim(bottom=0)
    ax = axes[1]
    style(ax, "Time per generation", "generation", "seconds")
    line(ax, r["gen"], r["gen_s"], SERIES[0], f"{r['gen_s'][-1]:.0f} s")
    total = (r["founders_s"] + sum(r["gen_s"])) / 60
    ax.text(0.03, 0.92, f"total run time {total:.0f} min\n(founders {r['founders_s'] / 60:.1f} min)",
            transform=ax.transAxes, va="top", fontsize=9, color=INK2)
    ax.set_ylim(bottom=0)
    fig.suptitle(f"Time horizon: {people_label(r['people'])} individuals × "
                 f"{count_label(r['variants'])} variants ({r['region']}), "
                 f"{len(r['gen'])} generations (Lane, {r['cpu']})",
                 x=0.01, ha="left", fontsize=13, color=INK)
    save(fig, out, "fig2_time_horizon.png")


def fig_cost_vs_graph(runs, out):
    # Compare runs on the most common CPU only, so time is like for like.
    cpus = [r["cpu"] for r in runs.values()]
    cpu = max(set(cpus), key=cpus.count)
    chosen = sorted((r for r in runs.values() if r["cpu"] == cpu),
                    key=lambda r: -r["edges"][-1])[:3]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=SURFACE)
    for ax, key, ylabel, title in ((axes[0], "gen_s", "seconds", "Time per generation"),
                                   (axes[1], "peak", "GB", "Peak memory")):
        style(ax, title, "graph edges (billions)", ylabel)
        for color, r in zip(SERIES, chosen):
            x = [e / 1e9 for e in r["edges"]]
            y = r[key]
            label = f"{size_label(r)} ({r['region'].split(' (')[0]})"
            if key == "peak":
                mx, my = sum(x) / len(x), sum(y) / len(y)
                slope = sum((a - mx) * (b - my) for a, b in zip(x, y)) / sum((a - mx) ** 2 for a in x) / 10
                label += f": {slope:.1f} GB per 100M edges"
            # The shortest run ends inside the others' lines; label it below them.
            # On the memory panel the middle run's label would cross the longest
            # run's line, so drop it below that line with a leader.
            shortest = r is chosen[-1] and len(chosen) > 1
            middle = key == "peak" and len(chosen) > 2 and r is chosen[1]
            offset = (8, -14) if shortest else (14, -45) if middle else (8, 0)
            line(ax, x, y, color, label, markers=False, offset=offset, leader=middle)
        ax.set_xlim(0, max(r["edges"][-1] for r in chosen) / 1e9 * 1.55)
        ax.set_ylim(bottom=0)
    fig.suptitle(f"Cost follows graph size (Lane, {cpu}; one line per run, one generation per step)",
                 x=0.01, ha="left", fontsize=13, color=INK)
    save(fig, out, "fig3_cost_vs_graph_size.png")


def table(runs, out):
    rows = []
    for r in sorted(runs.values(), key=lambda r: (r["people"], r["span_mb"], len(r["gen"]))):
        rows.append(dict(run=r["name"], people=r["people"], region=r["region"], variants=r["variants"],
                         generations=len(r["gen"]), founders_min=round(r["founders_s"] / 60, 1),
                         ts_to_grg_min=round(r["convert_s"] / 60, 1),
                         generations_min=round(sum(r["gen_s"]) / 60, 1),
                         total_min=round((r["founders_s"] + sum(r["gen_s"])) / 60, 1),
                         first_gen_s=round(r["gen_s"][0]), last_gen_s=round(r["gen_s"][-1]),
                         founder_edges_M=round(r["founder_edges"] / 1e6),
                         final_edges_M=round(r["edges"][-1] / 1e6),
                         peak_gb=round(r["peak_gb"], 1), cpu=r["cpu"]))
    with open(os.path.join(out, "scaling_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("results", help="results directory from scaling_run.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = load(args.results)
    fig_run_costs(runs, args.out)
    fig_horizon(runs, args.out)
    fig_cost_vs_graph(runs, args.out)
    for row in table(runs, args.out):
        print(row)
    print(f"figures and table in {args.out}")


if __name__ == "__main__":
    main()
