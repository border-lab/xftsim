#!/usr/bin/env python
"""Time and memory of one xftsim + GRG simulation on realistic founders.

Founders come from a published stdpopsim demographic model with a real
genetic map (default: HomSap OutOfAfrica_3G09, CEU, HapMapII_GRCh38), are
converted to a GRG, and are then simulated forward with GRG-native meiosis
under random mating (or assortative mating with --mating am).

Every stage and every generation appends a row to <out>/<name>.csv as soon
as it finishes, so a run killed for memory or time still records how far it
got. <out>/<name>.json holds the settings and machine, and its status changes
from "running" to "done" only if the run finishes.

    python scripts/scaling/scaling_run.py --name n500k_L20 --people 500000 \
        --left-mb 17 --length-mb 20 --gens 10 --out results/

--length-mb 0 simulates the whole chromosome. Memory columns are in GB
(1e9 bytes): rss_gb is the current resident memory, peak_gb the peak of this
process, peak_child_gb the peak of the largest finished child process (the
`grg convert` step runs as one).
"""
import argparse
import csv
import json
import os
import platform
import resource
import socket
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np

import xftsim
import xftsim.founders as xfounders
import xftsim.struct as xstruct
from xftsim.arch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.effect import AdditiveEffects
from xftsim.mate import LinearAssortativeMating, RandomMating
from xftsim.reproduce import RecombinationMap, _crossover_model, _meiosis_pair_switches_seeded
from xftsim.sim import Simulation

GB = 1e9
FIELDS = ["name", "stage", "generation", "seconds", "seconds_ts_to_grg", "seconds_meiosis",
          "people", "haplotypes", "variants", "span_mb", "total_cM", "nodes", "edges",
          "var_G", "var_Y", "rss_gb", "peak_gb", "peak_child_gb", "unix_time"]


def peak_gb(who):
    """Peak resident memory of this process or of its largest finished child."""
    peak = resource.getrusage(who).ru_maxrss
    return peak / GB if sys.platform == "darwin" else peak * 1024 / GB  # bytes vs KiB


def current_gb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / GB
    except ImportError:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / GB


def machine():
    info = {"host": socket.gethostname(), "platform": platform.platform(),
            "python": platform.python_version(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
            "slurm_mem_per_node_mb": os.environ.get("SLURM_MEM_PER_NODE", "")}
    try:
        with open("/proc/cpuinfo") as f:
            info["cpu"] = next(l.split(":", 1)[1].strip() for l in f if l.startswith("model name"))
    except (OSError, StopIteration):
        info["cpu"] = platform.processor()
    try:
        import psutil
        info["node_memory_gb"] = psutil.virtual_memory().total / GB
    except ImportError:
        pass
    from importlib.metadata import version
    for pkg in ("xftsim", "pygrgl", "msprime", "tskit", "stdpopsim", "numpy", "numba"):
        try:
            info[f"version_{pkg}"] = version(pkg)
        except Exception:
            info[f"version_{pkg}"] = ""
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", required=True, help="run name; output file stem")
    ap.add_argument("--people", type=int, required=True, help="diploid founders (constant N)")
    ap.add_argument("--pop", default="CEU")
    ap.add_argument("--model", default="OutOfAfrica_3G09")
    ap.add_argument("--chrom", default="chr22")
    ap.add_argument("--left-mb", type=float, default=17.0, help="window start (Mb)")
    ap.add_argument("--length-mb", type=float, default=20.0,
                    help="window length (Mb); 0 = whole chromosome")
    ap.add_argument("--genetic-map", default="HapMapII_GRCh38")
    ap.add_argument("--gens", type=int, default=10, help="generations of meiosis after the founders")
    ap.add_argument("--mating", choices=["random", "am"], default="random")
    ap.add_argument("--r", type=float, default=0.5, help="spousal correlation for --mating am")
    ap.add_argument("--h2", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    json_path = os.path.join(args.out, f"{args.name}.json")
    summary = {"status": "running", "args": vars(args), "machine": machine()}

    def save_summary():
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

    save_summary()
    fh = open(os.path.join(args.out, f"{args.name}.csv"), "w", newline="")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    writer.writeheader()

    def emit(**row):
        row = {"name": args.name, "people": args.people, **row,
               "rss_gb": round(current_gb(), 3),
               "peak_gb": round(peak_gb(resource.RUSAGE_SELF), 3),
               "peak_child_gb": round(peak_gb(resource.RUSAGE_CHILDREN), 3),
               "unix_time": round(time.time(), 1)}
        writer.writerow(row)
        fh.flush()
        shown = {k: row[k] for k in ("stage", "generation", "seconds", "nodes", "edges",
                                     "rss_gb", "peak_gb", "peak_child_gb") if row.get(k) not in (None, "")}
        print(" ".join(f"{k}={v}" for k, v in shown.items()), flush=True)

    emit(stage="start")

    # Compile the meiosis kernels now so generation 1 is not charged for JIT.
    _meiosis_pair_switches_seeded(*_crossover_model(np.array([0.5, 0.1])), np.uint32(0))

    # ---- founders: coalescent simulation, then tree sequence -> GRG ---------
    convert_seconds = [0.0]
    original_convert = xfounders._tree_sequence_to_mutable_grg

    def timed_convert(*a, **kw):
        start = time.perf_counter()
        out = original_convert(*a, **kw)
        convert_seconds[0] = time.perf_counter() - start
        return out

    xfounders._tree_sequence_to_mutable_grg = timed_convert
    whole = args.length_mb <= 0
    left = None if whole else int(args.left_mb * 1e6)
    right = None if whole else left + int(args.length_mb * 1e6)
    start = time.perf_counter()
    founders = xfounders.founder_haplotypes_from_stdpopsim_grg(
        samples={args.pop: args.people}, model_id=args.model, chromosome=args.chrom,
        genetic_map=args.genetic_map, left=left, right=right, seed=args.seed)
    founder_seconds = time.perf_counter() - start
    grg = founders._grg
    pos_bp, pos_cM = founders.variants.pos_bp, founders.variants.pos_cM
    span_mb = float(pos_bp[-1] - pos_bp[0]) / 1e6
    total_cM = float(pos_cM[-1] - pos_cM[0])
    # Meiosis modifies the founder graph in place, so record its size now.
    founder_nodes, founder_edges = grg.num_nodes, grg.num_edges
    emit(stage="founders", seconds=round(founder_seconds, 2),
         seconds_ts_to_grg=round(convert_seconds[0], 2), haplotypes=2 * founders.n,
         variants=founders.m, span_mb=round(span_mb, 2), total_cM=round(total_cM, 2),
         nodes=founder_nodes, edges=founder_edges)

    # ---- forward simulation --------------------------------------------------
    rmap = RecombinationMap.from_haplotypes_with_cM(founders)
    effects = AdditiveEffects.from_h2(h2=args.h2, m=founders.m, seed=args.seed)
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(effects))
    arch.add('Y.E', NoiseComponent(variance=1.0 - args.h2))
    arch.add('Y', AggregationComponent('Y.G + Y.E'))
    if args.mating == "random":
        mating = RandomMating(offspring_per_pair=2)
    else:
        mating = LinearAssortativeMating(['Y'], r=args.r, offspring_per_pair=2)

    meiosis_seconds = [0.0]
    original_meiosis = xstruct.GraphHaplotypeOperator.meiosis

    def timed_meiosis(self, *a, **kw):
        start = time.perf_counter()
        out = original_meiosis(self, *a, **kw)
        meiosis_seconds[0] = time.perf_counter() - start
        return out

    xstruct.GraphHaplotypeOperator.meiosis = timed_meiosis
    clock = [time.perf_counter()]
    gen_seconds = []

    def report(sim):
        g = sim.generation
        live = sim.haplotypes._grg
        pheno = sim.phenotype_history[g]
        seconds = time.perf_counter() - clock[0]
        if g > 0:
            gen_seconds.append((seconds, meiosis_seconds[0]))
        emit(stage="generation", generation=g, seconds=round(seconds, 2),
             seconds_meiosis=round(meiosis_seconds[0], 2) if g > 0 else "",
             haplotypes=2 * sim.haplotypes.n, variants=sim.haplotypes.m,
             nodes=live.num_nodes, edges=live.num_edges,
             var_G=round(float(np.var(pheno['Y.G'])), 4),
             var_Y=round(float(np.var(pheno['Y'])), 4))
        clock[0] = time.perf_counter()

    sim = Simulation(
        founder_haplotypes=founders, architecture=arch, mating_regime=mating,
        recombination_map=rmap, seed=args.seed,
        retain_haplotypes=0, retain_phenotypes=1,
        standardize_with='founders', callbacks=[report],
    )
    clock[0] = time.perf_counter()
    sim.run(args.gens + 1)

    final = sim.haplotypes._grg
    summary.update({
        "status": "done",
        "people": args.people, "variants": founders.m,
        "span_mb": span_mb, "total_cM": total_cM,
        "seconds_founders": founder_seconds, "seconds_ts_to_grg": convert_seconds[0],
        "founder_nodes": founder_nodes, "founder_edges": founder_edges,
        "final_nodes": final.num_nodes, "final_edges": final.num_edges,
        "generations": len(gen_seconds),
        "mean_seconds_per_generation": float(np.mean([s for s, _ in gen_seconds])),
        "mean_seconds_meiosis": float(np.mean([m for _, m in gen_seconds])),
        "peak_gb": peak_gb(resource.RUSAGE_SELF),
        "peak_child_gb": peak_gb(resource.RUSAGE_CHILDREN),
    })
    save_summary()
    fh.close()
    print(f"done: {args.name}", flush=True)


if __name__ == "__main__":
    main()
