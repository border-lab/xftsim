#!/usr/bin/env python
"""Performance benchmarks for xftsim core operations.

Profiles founder generation, meiosis, matvec, architecture compute,
full simulation loop, and I/O checkpoint roundtrip at various scales.

Usage:
    python benchmarks/bench_core.py
"""
import sys
import os
import time
import tempfile
import shutil
import warnings

import numpy as np

# Ensure xftsim is importable from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from xftsim.struct import (
    SampleMeta, VariantMeta, DenseHaplotypeArray, NPhenotypeArray,
)
from xftsim.effect import AdditiveEffects
from xftsim.arch import (
    Architecture, GeneticComponent, NoiseComponent, AggregationComponent,
)
from xftsim.mate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.sim import NSimulation
from xftsim.io import save_simulation_checkpoint, load_simulation_checkpoint

# Suppress repetitive warnings during benchmarks
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SCALES = [
    ("small",  100,   50),
    ("medium", 1000,  500),
    ("large",  5000,  2000),
    ("xl",     10000, 5000),
]


def _make_haplotypes(n, m, seed=42):
    """Create DenseHaplotypeArray with balanced sex."""
    rng = np.random.RandomState(seed)
    af = rng.uniform(0.1, 0.9, size=m)
    af_row = af[np.newaxis, :]  # shape (1, m)
    geno = np.empty((n, m, 2), dtype=np.int8)
    geno[:, :, 0] = (rng.random_sample((n, m)) < af_row).astype(np.int8)
    geno[:, :, 1] = (rng.random_sample((n, m)) < af_row).astype(np.int8)
    sex = np.tile([0, 1], (n + 1) // 2)[:n]
    samples = SampleMeta(iid=np.arange(n), sex=sex)
    variants = VariantMeta(vid=np.arange(m), af=af)
    return DenseHaplotypeArray(genotypes=geno, samples=samples, variants=variants)


def _make_architecture(m, h2=0.5, seed=123):
    """Y = G + E architecture."""
    effects = AdditiveEffects.from_h2(h2=h2, m=m, seed=seed)
    arch = Architecture()
    arch.add("Y.G", GeneticComponent(effects))
    arch.add("Y.E", NoiseComponent(variance=1.0 - h2))
    arch.add("Y", AggregationComponent("Y.G + Y.E"))
    return arch


def _timer(fn, repeats=3):
    """Run fn `repeats` times, return (mean_seconds, std_seconds)."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    arr = np.array(times)
    return float(arr.mean()), float(arr.std())


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def bench_founders(n, m, repeats=3):
    """Time founder_haplotypes_uniform_AFs(n, m)."""
    from xftsim.founders import founder_haplotypes_uniform_AFs

    def fn():
        founder_haplotypes_uniform_AFs(n, m)

    return _timer(fn, repeats)


def bench_meiosis(n, m, repeats=3):
    """Time DenseHaplotypeArray.meiosis() with random mate assignment."""
    hap = _make_haplotypes(n, m)
    rmap = RecombinationMap.constant_map(m=m, p=0.5)
    mate = RandomMating(offspring_per_pair=2)
    rng = np.random.RandomState(0)
    assignment = mate.mate(hap.samples, rng=rng)

    def fn():
        hap.meiosis(assignment, rmap)

    return _timer(fn, repeats)


def bench_matvec(n, m, k=1, repeats=3):
    """Time hap.matvec(weights) for (m,) or (m,k) weights."""
    hap = _make_haplotypes(n, m)
    if k == 1:
        weights = np.random.RandomState(0).normal(size=m)
    else:
        weights = np.random.RandomState(0).normal(size=(m, k))

    def fn():
        hap.matvec(weights)

    return _timer(fn, repeats)


def bench_standardized_matvec(n, m, repeats=3):
    """Time hap.standardized_matvec(weights)."""
    hap = _make_haplotypes(n, m)
    weights = np.random.RandomState(0).normal(size=m)

    def fn():
        hap.standardized_matvec(weights)

    return _timer(fn, repeats)


def bench_architecture_compute(n, m, repeats=3):
    """Time one architecture.compute() call (gen 0 phenotype computation)."""
    hap = _make_haplotypes(n, m)
    arch = _make_architecture(m)
    rng = np.random.RandomState(0)

    def fn():
        arch.compute(hap, rng=rng, phenotype_history={}, pedigree_history={}, generation=0)

    return _timer(fn, repeats)


def bench_simulation(n, m, generations, repeats=1):
    """Time NSimulation.run(generations) end to end."""
    def fn():
        hap = _make_haplotypes(n, m, seed=42)
        arch = _make_architecture(m)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        mate = RandomMating(offspring_per_pair=2)
        sim = NSimulation(
            founder_haplotypes=hap,
            architecture=arch,
            mating_regime=mate,
            recombination_map=rmap,
            seed=0,
        )
        sim.run(generations)

    return _timer(fn, repeats)


def bench_io_checkpoint(n, m, repeats=1):
    """Time save + load checkpoint roundtrip."""
    hap = _make_haplotypes(n, m, seed=42)
    arch = _make_architecture(m)
    rmap = RecombinationMap.constant_map(m=m, p=0.5)
    mate = RandomMating(offspring_per_pair=2)
    sim = NSimulation(
        founder_haplotypes=hap,
        architecture=arch,
        mating_regime=mate,
        recombination_map=rmap,
        seed=0,
    )
    sim.run(3)

    def fn():
        tmpdir = tempfile.mkdtemp(prefix="xft_bench_")
        try:
            save_simulation_checkpoint(sim, tmpdir)
            load_simulation_checkpoint(tmpdir)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    return _timer(fn, repeats)


# ---------------------------------------------------------------------------
# Main: run all benchmarks, print formatted table
# ---------------------------------------------------------------------------

def main():
    results = []  # list of (operation, n, m, g, mean_s, std_s)

    print("xftsim performance benchmarks")
    print(f"NumPy {np.__version__}, Python {sys.version.split()[0]}")
    print()
    print("Running benchmarks (this may take a minute)...")
    print()

    # --- Founder generation ---
    for label, n, m in SCALES:
        reps = 3 if n <= 5000 else 1
        mean_s, std_s = bench_founders(n, m, repeats=reps)
        results.append(("founder_generation", n, m, "-", mean_s, std_s))
        print(f"  founder_generation  n={n:>5}  m={m:>5}  {mean_s:.4f}s")

    # --- Meiosis ---
    for label, n, m in SCALES:
        reps = 3 if n <= 5000 else 1
        mean_s, std_s = bench_meiosis(n, m, repeats=reps)
        results.append(("meiosis", n, m, "-", mean_s, std_s))
        print(f"  meiosis             n={n:>5}  m={m:>5}  {mean_s:.4f}s")

    # --- Matvec (diploid) ---
    for label, n, m in SCALES:
        reps = 5 if n <= 5000 else 2
        mean_s, std_s = bench_matvec(n, m, k=1, repeats=reps)
        results.append(("matvec", n, m, "-", mean_s, std_s))
        print(f"  matvec              n={n:>5}  m={m:>5}  {mean_s:.4f}s")

    # --- Standardized matvec ---
    for label, n, m in SCALES:
        reps = 5 if n <= 5000 else 2
        mean_s, std_s = bench_standardized_matvec(n, m, repeats=reps)
        results.append(("std_matvec", n, m, "-", mean_s, std_s))
        print(f"  std_matvec          n={n:>5}  m={m:>5}  {mean_s:.4f}s")

    # --- Architecture compute ---
    for label, n, m in SCALES:
        reps = 3 if n <= 5000 else 1
        mean_s, std_s = bench_architecture_compute(n, m, repeats=reps)
        results.append(("arch_compute", n, m, "-", mean_s, std_s))
        print(f"  arch_compute        n={n:>5}  m={m:>5}  {mean_s:.4f}s")

    # --- Full simulation loop ---
    sim_configs = [
        (100,  50,   10),
        (1000, 500,  5),
        (1000, 500,  10),
        (5000, 2000, 5),
    ]
    for n, m, g in sim_configs:
        reps = 2 if n <= 1000 else 1
        mean_s, std_s = bench_simulation(n, m, g, repeats=reps)
        results.append(("full_simulation", n, m, str(g), mean_s, std_s))
        print(f"  full_simulation     n={n:>5}  m={m:>5}  g={g:<3}  {mean_s:.4f}s")

    # --- I/O checkpoint roundtrip ---
    io_configs = [
        (100,  50),
        (1000, 500),
        (5000, 2000),
    ]
    for n, m in io_configs:
        reps = 2 if n <= 1000 else 1
        mean_s, std_s = bench_io_checkpoint(n, m, repeats=reps)
        results.append(("io_checkpoint", n, m, "-", mean_s, std_s))
        print(f"  io_checkpoint       n={n:>5}  m={m:>5}  {mean_s:.4f}s")

    # --- Summary table ---
    print()
    print("=" * 72)
    header = f"{'Operation':<22} {'n':>6} {'m':>6} {'g':>4} {'mean(s)':>9} {'std(s)':>9}"
    print(header)
    print("-" * 72)
    for op, n, m, g, mean_s, std_s in results:
        print(f"{op:<22} {n:>6} {m:>6} {str(g):>4} {mean_s:>9.4f} {std_s:>9.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
