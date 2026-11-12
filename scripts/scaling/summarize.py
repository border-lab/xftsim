#!/usr/bin/env python
"""Collect scaling_run.py results into one table.

    python scripts/scaling/summarize.py RESULTS_DIR [RESULTS_DIR ...]

Prints one row per run and writes summary.csv into the first directory.
A run whose status is still "running" was killed or is still going; on Slurm,
`sacct -j <job> --format=JobName,State,Elapsed,MaxRSS,ReqMem` says which
(OUT_OF_MEMORY, TIMEOUT, ...). Peak memory is the larger of the Python
process and the `grg convert` child process.
"""
import csv
import glob
import json
import os
import sys


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def summarize(json_path):
    with open(json_path) as f:
        info = json.load(f)
    args = info["args"]
    rows = []
    csv_path = json_path[:-len(".json")] + ".csv"
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
    founders = next((r for r in rows if r["stage"] == "founders"), {})
    gens = [r for r in rows if r["stage"] == "generation" and r["generation"] not in ("", "0")]
    last = rows[-1] if rows else {}
    peaks = [num(r.get(k)) for r in rows for k in ("peak_gb", "peak_child_gb")]
    peaks = [p for p in peaks if p is not None]
    per_gen = [num(r["seconds"]) for r in gens]
    stage = (f"gen {last.get('generation')}" if last.get("stage") == "generation"
             else last.get("stage", "-"))
    return {
        "run": args["name"],
        "people": args["people"],
        "region": (f"{args['chrom']} whole" if args["length_mb"] <= 0
                   else f"{args['chrom']} {args['length_mb']:g} Mb"),
        "span_mb": num(founders.get("span_mb")),
        "variants": num(founders.get("variants")),
        "founders_s": num(founders.get("seconds")),
        "ts_to_grg_s": num(founders.get("seconds_ts_to_grg")),
        "gens_done": f"{len(gens)}/{args['gens']}",
        "first_gen_s": per_gen[0] if per_gen else None,
        "last_gen_s": per_gen[-1] if per_gen else None,
        "founder_edges_M": num(founders.get("edges")) / 1e6 if founders.get("edges") else None,
        "final_edges_M": num(gens[-1]["edges"]) / 1e6 if gens else None,
        "peak_gb": max(peaks) if peaks else None,
        "status": info["status"] if info["status"] == "done" else f"stopped at {stage}",
        "host": info["machine"].get("host", ""),
    }


def fmt(v):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:,.0f}" if abs(v) >= 100 else f"{v:.2f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def main():
    dirs = sys.argv[1:] or ["."]
    paths = sorted(p for d in dirs for p in glob.glob(os.path.join(d, "*.json")))
    table = [summarize(p) for p in paths]
    if not table:
        sys.exit(f"no results in {dirs}")
    cols = list(table[0])
    widths = {c: max(len(c), *(len(fmt(r[c])) for r in table)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in table:
        print("  ".join(fmt(r[c]).ljust(widths[c]) for c in cols))
    out = os.path.join(dirs[0], "summary.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(table)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
