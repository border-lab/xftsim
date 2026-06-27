"""
Parity tests: C++ NonDuplicationRecombiner vs Python NonDuplicationRecombination.

For each .grg file passed in via --grg, constructs both backends on a freshly
loaded copy and runs --num-generations generations under simulate_grg_recombination
with a fixed PRNG seed on each backend. Repeats the whole pass --warmup +
--runs times (mirroring benchmark_recombination.py): warmup passes don't
contribute to timings, timed passes do; every pass independently parity-checks
the two backends.

Two modes, selected by --timing-mode:

PARITY mode (default) -- per-pass checks:
  1. Returned offspring NodeID lists per generation are identical.
  2. Final audit dicts match key-for-key, value-for-value.
  3. Post-recombination GRG state matches (num_nodes, num_edges).
  4. Both backends' audit_check identities pass.
  5. Stats dict key sets match.
  Both backends constructed with instrument=True; C++ pre-prune disabled so
  audit counts stay byte-identical to the Python reference. Wallclock numbers
  include the instrumentation overhead on both sides and are NOT representative
  of production timing. Output: `parity_results_<basename>.json`.

TIMING mode (--timing-mode) -- per-pass checks:
  1. Returned offspring NodeID lists per generation are identical.
  3. Post-recombination GRG state matches (num_nodes, num_edges).
  Both backends constructed with instrument=False; C++ pre-prune left at its
  production default (True). Wallclock numbers are the production-relevant
  measurement -- no chrono probes, no stats increments, pre-prune active.
  Audit/stats comparisons are skipped (pre-prune diverges audit counters by
  design; instrument=False zeros the C++ stats dict). Output:
  `timing_results_<basename>.json`.

Usage:
    cd benchmark/native
    ../../.venv/bin/python test_parity.py
    ../../.venv/bin/python test_parity.py --warmup 1 --runs 3
    ../../.venv/bin/python test_parity.py --grg ../../grg_files/16000inds_1m_snps.grg --runs 1
    ../../.venv/bin/python test_parity.py --timing-mode --warmup 1 --runs 3
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import pygrgl


# Resolve __file__ to a canonical absolute path BEFORE any cwd-relative
# manipulation. Without realpath(), `__file__` can be a bare basename (when
# invoked as `python test_parity.py` from inside native/) or a path through a
# symlink (common on cluster scratch/home mounts). Both forms make the
# subsequent abspath(join(..., "..")) cwd-dependent, which is the exact failure
# mode the parity script must avoid -- it has to import the benchmark/ source
# tree, not whatever happens to be one level up from cwd.
_THIS_FILE = os.path.realpath(__file__)
NATIVE_DIR = os.path.dirname(_THIS_FILE)
BENCHMARK_DIR = os.path.dirname(NATIVE_DIR)
REPO_ROOT = os.path.dirname(BENCHMARK_DIR)
sys.path.insert(0, BENCHMARK_DIR)

# Default parity-test output directory. Mirrors benchmark_recombination.py's
# convention of writing under benchmark/output/. Override on the CLI with
# --output-dir to redirect into a custom location (e.g. a slurm-managed dir).
DEFAULT_PARITY_OUTPUT_DIR = os.path.join(BENCHMARK_DIR, "output", "parity")

from grg_recombination import (
    NonDuplicationRecombination as PyImpl,
    simulate_grg_recombination,
)
from grg_recombination_native import NonDuplicationRecombination as CppImpl

# Fail loud if the C++ extension didn't actually back CppImpl. The wrapper in
# grg_recombination_native.py would already ImportError if the .so were
# missing, but this guards against a future Python-only fallback being added
# and silently making the "parity test" compare Python-to-Python.
from grg_recomb_native import _grg_recomb_native as _native_so
import grg_recombination_native as _native_wrapper
import grg_recombination as _py_module
assert CppImpl is not PyImpl, "CppImpl resolved to the Python class; native import path is broken"
assert _native_wrapper._NativeRecombiner.__module__.endswith("_grg_recomb_native"), (
    "grg_recombination_native is not bound to the C++ extension; "
    f"got {_native_wrapper._NativeRecombiner.__module__}"
)


class _FakeBenchmark:
    """Minimal benchmark stub for simulate_grg_recombination."""

    def __init__(self, num_offspring_per_couple=2, expected_crossovers=1.5):
        self.num_offspring_per_couple = num_offspring_per_couple
        self._expected_crossovers = expected_crossovers

    def get_breakpoints(self, bp_range, expected_crossovers=None):
        if expected_crossovers is None:
            expected_crossovers = self._expected_crossovers
        N = bp_range[1]
        num_bp = np.random.poisson(expected_crossovers)
        if num_bp == 0:
            return [], 0
        bps = sorted(np.random.uniform(0, N, num_bp).astype(int).tolist())
        return bps, num_bp


def run_single_pass(grg_path, base_seed, num_generations, num_offspring_per_couple,
                    timing_mode=False):
    """One full pass: load fresh GRGs, construct both backends, run all generations.

    Same `base_seed` is used on both backends so breakpoint sampling and parent
    shuffling are identical across implementations. Caller varies `base_seed`
    across passes so workloads differ between warmup/timed runs (mirrors the
    natural PRNG drift in benchmark_recombination.py).

    `timing_mode=False` (parity mode): both backends with instrument=True; C++
    pre-prune OFF for byte-identical audit dicts. Wallclock includes
    instrumentation overhead.

    `timing_mode=True` (timing mode): both backends with instrument=False; C++
    pre-prune at its production default (True). Wallclock is production-
    relevant. Audit/stats dicts are populated where possible (Python audit is
    always-on, C++ audit is unconditional too) but are NOT compared -- pre-prune
    diverges them by design. The returned dict sets `audit_check` and `stats`
    to None to signal "do not compare" downstream.
    """
    bench = _FakeBenchmark(num_offspring_per_couple=num_offspring_per_couple)
    instrument = not timing_mode

    # --- Python reference ---
    g_py = pygrgl.load_mutable_grg(grg_path)
    py_recomb = PyImpl(g_py, instrument=instrument)
    py_offspring_per_gen = []
    py_total_t = 0.0
    for gen in range(num_generations):
        np.random.seed(base_seed + gen)
        t0 = time.perf_counter()
        offspring, _ = simulate_grg_recombination(bench, py_recomb, g_py.bp_range, int(g_py.bp_range[1]))
        py_total_t += time.perf_counter() - t0
        py_offspring_per_gen.append(list(offspring))

    # --- C++ implementation ---
    g_cpp = pygrgl.load_mutable_grg(grg_path)
    cpp_recomb = CppImpl(g_cpp, instrument=instrument)
    if not timing_mode:
        # Parity mode: disable the push-site pre-prune so the C++ visits the same
        # nodes as the Python reference; otherwise the audit `visits`, `pruning`,
        # and `skip_already_visited` counters diverge (correctly -- pre-prune
        # skips visits Python would still pay for). The perf benefit of pre-prune
        # is measured by timing mode and by `benchmark_recombination.py
        # --diagnostics`.
        cpp_recomb.pre_prune_enabled = False
    # Timing mode leaves pre_prune_enabled at its production default (True).
    cpp_offspring_per_gen = []
    cpp_total_t = 0.0
    for gen in range(num_generations):
        np.random.seed(base_seed + gen)
        t0 = time.perf_counter()
        offspring, _ = simulate_grg_recombination(bench, cpp_recomb, g_cpp.bp_range, int(g_cpp.bp_range[1]))
        cpp_total_t += time.perf_counter() - t0
        cpp_offspring_per_gen.append(list(offspring))

    return {
        "py": {
            "offspring_per_gen": py_offspring_per_gen,
            "audit": None if timing_mode else py_recomb.audit,
            "stats": None if timing_mode else py_recomb.stats,
            "num_nodes": g_py.num_nodes,
            "num_edges": g_py.num_edges,
            "wallclock": py_total_t,
            "audit_check": None if timing_mode else py_recomb.audit_check(raise_on_fail=False),
        },
        "cpp": {
            "offspring_per_gen": cpp_offspring_per_gen,
            "audit": None if timing_mode else cpp_recomb.audit,
            "stats": None if timing_mode else cpp_recomb.stats,
            "num_nodes": g_cpp.num_nodes,
            "num_edges": g_cpp.num_edges,
            "wallclock": cpp_total_t,
            "audit_check": None if timing_mode else cpp_recomb.audit_check(raise_on_fail=False),
        },
    }


def check_pass_parity(result, timing_mode=False):
    """Returns (passed: bool, failures: list[str]) for one pass's result dict.

    Compared to compare_and_report below, this is the silent check used by each
    warmup/timed pass; reporting happens at the per-file summary level.

    Timing mode skips audit-dict, audit-identity, and stats-key checks
    (they don't apply when pre-prune is ON and instrument is OFF). Offspring
    NodeID and GRG state checks still run -- pre-prune skips pruning-only
    visits, so topology is unchanged.
    """
    py = result["py"]
    cpp = result["cpp"]
    failures = []

    # (1) Offspring lists identical per generation. Holds in both modes -- pre-
    # prune skips only visits that would have been pure pruning, leaving the
    # output topology untouched.
    for g, (po, co) in enumerate(zip(py["offspring_per_gen"], cpp["offspring_per_gen"])):
        if po != co:
            n_mismatch = sum(1 for a, b in zip(po, co) if a != b)
            failures.append(
                f"gen {g}: offspring lists differ ({n_mismatch}/{len(po)} mismatched, "
                f"py len={len(po)} cpp len={len(co)})"
            )

    # (3) GRG state matches. Holds in both modes for the same reason as (1).
    if py["num_nodes"] != cpp["num_nodes"]:
        failures.append(f"num_nodes mismatch py={py['num_nodes']} cpp={cpp['num_nodes']}")
    if py["num_edges"] != cpp["num_edges"]:
        failures.append(f"num_edges mismatch py={py['num_edges']} cpp={cpp['num_edges']}")

    if timing_mode:
        # Skip audit-dict, audit-identity, and stats-key checks. They don't apply:
        # pre-prune is ON (audit `visits`/`pruning`/`skip_already_visited` diverge
        # by design) and instrument=False zeros the C++ stats dict.
        return (not failures, failures)

    # (2) Final audit dict matches.
    audit_keys = set(py["audit"].keys()) | set(cpp["audit"].keys())
    audit_diffs = [(k, py["audit"].get(k, "MISSING"), cpp["audit"].get(k, "MISSING"))
                   for k in audit_keys if py["audit"].get(k) != cpp["audit"].get(k)]
    if audit_diffs:
        failures.append(f"audit diffs ({len(audit_diffs)}):")
        for k, pv, cv in audit_diffs[:10]:
            failures.append(f"  {k}: py={pv}, cpp={cv}")

    # (4) Audit identities pass on both.
    for backend_name, ck in [("py", py["audit_check"]), ("cpp", cpp["audit_check"])]:
        for ident, r in ck.items():
            if not r["pass"]:
                failures.append(f"{backend_name} audit identity FAIL [{ident}]: "
                                f"lhs={r['lhs']} rhs={r['rhs']}")

    # (5) Stats key sets match (values differ — timing varies — but shape must).
    py_stats_keys = set(py["stats"].keys())
    cpp_stats_keys = set(cpp["stats"].keys())
    if py_stats_keys != cpp_stats_keys:
        missing = py_stats_keys - cpp_stats_keys
        extra = cpp_stats_keys - py_stats_keys
        failures.append(f"stats key mismatch: missing in cpp={missing}, extra in cpp={extra}")

    return (not failures, failures)


def _mean_std(values):
    """Mean and population stdev for a list (mirrors benchmark_recombination.py)."""
    if not values:
        return 0.0, 0.0
    m = sum(values) / len(values)
    if len(values) == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / len(values)
    return m, math.sqrt(var)


def run_file(grg_path, args):
    """Run num_warmup + num_runs passes against one .grg, collect aggregate stats.

    Each pass uses a distinct base_seed so warmup vs timed passes see different
    workloads (matching the natural PRNG drift in benchmark_recombination.py).
    Both backends within a single pass share that pass's seed, so they see
    identical workloads -- the parity check is strict bit-for-bit (or
    topology-only in timing mode; see check_pass_parity).
    """
    name = os.path.basename(grg_path)
    total_passes = args.warmup + args.runs
    mode_label = "TIMING" if args.timing_mode else "PARITY"
    print(f"\n=== {name}  [{mode_label} mode] ===")
    print(f"  warmup={args.warmup}  runs={args.runs}  num_generations={args.num_generations}")

    pass_records = []
    py_timed_times = []
    cpp_timed_times = []
    overall_passed = True
    last_result = None

    for pass_idx in range(total_passes):
        is_warmup = pass_idx < args.warmup
        label = "warmup" if is_warmup else "timed"
        # Distinct seed per pass so warmup vs timed workloads differ.
        pass_seed = args.seed + pass_idx * args.num_generations

        result = run_single_pass(grg_path,
                                 pass_seed,
                                 args.num_generations,
                                 args.offspring_per_couple,
                                 timing_mode=args.timing_mode)
        passed, failures = check_pass_parity(result, timing_mode=args.timing_mode)
        overall_passed = overall_passed and passed

        py_t_ms = result["py"]["wallclock"] * 1000
        cpp_t_ms = result["cpp"]["wallclock"] * 1000
        speedup = result["py"]["wallclock"] / max(result["cpp"]["wallclock"], 1e-9)

        marker = "OK" if passed else "FAIL"
        print(f"  [{label} {pass_idx+1}/{total_passes}] "
              f"py={py_t_ms:9.2f}ms cpp={cpp_t_ms:9.2f}ms speedup={speedup:6.2f}x  [{marker}]")
        if not passed:
            for f in failures:
                print(f"    {f}")

        if not is_warmup:
            py_timed_times.append(result["py"]["wallclock"])
            cpp_timed_times.append(result["cpp"]["wallclock"])

        pass_records.append({
            "pass_index": pass_idx,
            "is_warmup": is_warmup,
            "seed": pass_seed,
            "parity_passed": passed,
            "py_wallclock_s": result["py"]["wallclock"],
            "cpp_wallclock_s": result["cpp"]["wallclock"],
            "num_nodes_after": result["py"]["num_nodes"],
            "num_edges_after": result["py"]["num_edges"],
            "failures": failures if not passed else [],
        })
        last_result = result

    py_mean, py_std = _mean_std(py_timed_times)
    cpp_mean, cpp_std = _mean_std(cpp_timed_times)
    speedup = py_mean / max(cpp_mean, 1e-12)

    summary_marker = ("TIMING OK" if args.timing_mode else "PARITY OK") if overall_passed else (
        "TIMING FAILURES" if args.timing_mode else "PARITY FAILURES")
    print(f"  --- aggregate over {args.runs} timed run(s) ---")
    print(f"  py  mean_time_ms = {py_mean*1000:9.2f} ± {py_std*1000:.2f}")
    print(f"  cpp mean_time_ms = {cpp_mean*1000:9.2f} ± {cpp_std*1000:.2f}")
    print(f"  speedup (mean)   = {speedup:.2f}x")
    print(f"  result: {summary_marker}")

    return {
        "passed": overall_passed,
        "py_mean_s": py_mean,
        "py_std_s": py_std,
        "cpp_mean_s": cpp_mean,
        "cpp_std_s": cpp_std,
        "speedup": speedup,
        "passes": pass_records,
        # Last pass's audit dicts -- useful for spot-checking from the JSON
        # without bloating output with one audit dict per pass. None in timing
        # mode (audit comparison doesn't apply with pre-prune ON).
        "last_py_audit": last_result["py"]["audit"],
        "last_cpp_audit": last_result["cpp"]["audit"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grg",
        action="append",
        default=None,
        help="Specific .grg file(s) to test. Defaults to grg_files/{50inds_1k,500inds_10k} from the repo root.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-generations", type=int, default=1)
    parser.add_argument("--offspring-per-couple", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=0,
                        help="Number of warmup passes (parity-checked but not timed). Default 0.")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of timed passes. Default 1.")
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_PARITY_OUTPUT_DIR,
        help=(f"Directory for the result JSON. Auto-created if missing. "
              f"Default: {DEFAULT_PARITY_OUTPUT_DIR}"),
    )
    parser.add_argument(
        "--timing-mode",
        action="store_true",
        help=(
            "Switch to production-equivalent timing measurement: both backends "
            "with instrument=False, C++ pre-prune ON (production default). "
            "Audit/stats parity checks are skipped (they don't apply); "
            "offspring NodeID and num_nodes/num_edges checks still run. Output "
            "JSON is named `timing_results_<basename>.json` instead of "
            "`parity_results_<basename>.json`."
        ),
    )
    args = parser.parse_args()

    if args.runs < 1:
        print("--runs must be >= 1")
        sys.exit(2)

    # Diagnostic banner: prints where each module was actually loaded from.
    # Run from inside vs outside benchmark/native/ should produce identical
    # paths here -- if they differ, the cluster is picking up a stale module.
    mode_str = "TIMING (instrument=False, C++ pre-prune ON)" if args.timing_mode \
        else "PARITY (instrument=True, C++ pre-prune OFF)"
    print("=" * 70)
    print(f"test_parity.py     : {_THIS_FILE}")
    print(f"  mode             : {mode_str}")
    print(f"  BENCHMARK_DIR    : {BENCHMARK_DIR}")
    print(f"  REPO_ROOT        : {REPO_ROOT}")
    print(f"  cwd              : {os.getcwd()}")
    print(f"  python           : {sys.executable}")
    print(f"  pygrgl           : {pygrgl.__file__}")
    print(f"  grg_recombination: {_py_module.__file__}")
    print(f"  ...native wrapper: {_native_wrapper.__file__}")
    print(f"  ...C++ extension : {_native_so.__file__}")
    print("=" * 70)

    if args.grg is None:
        defaults = [
            os.path.join(REPO_ROOT, "grg_files", "50inds_1k_snps.grg"),
            os.path.join(REPO_ROOT, "grg_files", "500inds_10k_snps.grg"),
        ]
        args.grg = [p for p in defaults if os.path.exists(p)]
        if not args.grg:
            print("No default .grg files found; pass --grg PATH.")
            sys.exit(1)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    overall = True
    written_paths = []
    for grg_path in args.grg:
        agg = run_file(grg_path, args)
        overall = overall and agg["passed"]

        # Per-GRG output file. Parity mode -> `parity_results_<basename>.json`;
        # timing mode -> `timing_results_<basename>.json`. The different prefixes
        # let downstream tooling (e.g. the comparison report) tell them apart
        # without parsing the JSON.
        basename = os.path.splitext(os.path.basename(grg_path))[0]
        prefix = "timing_results" if args.timing_mode else "parity_results"
        output_path = os.path.join(output_dir, f"{prefix}_{basename}.json")
        record = {
            "timestamp": run_timestamp,
            "mode": "timing" if args.timing_mode else "parity",
            "grg_file": os.path.basename(grg_path),
            "args": {
                "seed": args.seed,
                "num_generations": args.num_generations,
                "offspring_per_couple": args.offspring_per_couple,
                "warmup": args.warmup,
                "runs": args.runs,
                "timing_mode": args.timing_mode,
            },
            "parity_passed": agg["passed"],
            "py_mean_time_s": agg["py_mean_s"],
            "py_std_time_s": agg["py_std_s"],
            "cpp_mean_time_s": agg["cpp_mean_s"],
            "cpp_std_time_s": agg["cpp_std_s"],
            "speedup": agg["speedup"],
            "passes": agg["passes"],
            "last_py_audit": agg["last_py_audit"],
            "last_cpp_audit": agg["last_cpp_audit"],
        }
        with open(output_path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"  saved: {output_path}")
        written_paths.append(output_path)

    mode_word = "Timing" if args.timing_mode else "Parity"
    print(f"\n{mode_word} results written to {len(written_paths)} file(s) in {output_dir}")

    print()
    if overall:
        print(f"ALL {mode_word.upper()} CHECKS PASSED")
    else:
        print(f"{mode_word.upper()} FAILURES — see above")
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
