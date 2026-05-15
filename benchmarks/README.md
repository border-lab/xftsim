# xftsim Performance Benchmarks

## Overview

The `bench_core.py` script profiles the core operations in xftsim at various
data scales (number of individuals `n` and number of variants `m`). It measures
wall-clock time for each operation, running multiple repeats and reporting
mean and standard deviation.

## Benchmarked Operations

| Operation | Description |
|-----------|-------------|
| `founder_generation` | Create founder haplotypes with uniform allele frequencies |
| `meiosis` | Perform meiosis (recombination + offspring generation) from a random mating assignment |
| `matvec` | Diploid haplotype-by-weight matrix-vector product |
| `std_matvec` | Standardized (mean-centered, variance-scaled) matvec |
| `arch_compute` | Single call to `Architecture.compute()` for a Y = G + E model |
| `full_simulation` | End-to-end `Simulation.run()` over multiple generations |
| `io_checkpoint` | Save and load a simulation checkpoint (directory with npz + JSON files) |

## Data Scales

The benchmarks use four scale levels:

| Label  | n (individuals) | m (variants) |
|--------|-----------------|--------------|
| small  | 100             | 50           |
| medium | 1,000           | 500          |
| large  | 5,000           | 2,000        |
| xl     | 10,000          | 5,000        |

Full simulation benchmarks additionally vary the number of generations (`g`).

## How to Run

From the repository root:

```bash
python benchmarks/bench_core.py
```

The script takes approximately 1-2 minutes to complete depending on hardware.
All output is printed to stdout in a formatted table.

To save results to a file:

```bash
python benchmarks/bench_core.py | tee benchmarks/results_$(date +%Y-%m-%d).txt
```

## Baseline Results

See `baseline_results.txt` for the initial baseline captured on 2026-02-08
(AMD Ryzen 9 9950X, 123.5 GB RAM, Python 3.12.7, NumPy 1.26.4).

## Notes

- Timing uses `time.perf_counter()` for high-resolution wall-clock measurements.
- Each benchmark runs multiple repeats (typically 3-5 for smaller scales, 1-2 for
  larger scales) to account for variance.
- The `io_checkpoint` benchmark creates and cleans up temporary directories for each
  iteration, so its timings include filesystem overhead.
- Results may vary across runs due to system load, thermal throttling, and other factors.
  Compare relative timings across operations rather than relying on absolute numbers.
