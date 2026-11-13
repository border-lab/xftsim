# Development Workflow Changelog

Changes to development tooling, CI/CD, testing infrastructure, and documentation systems.

## [Unreleased]

### Option B Native Port: Tests and Demo Flags (2026-11-12)

- Added `tests/unit/test_grg_groups.py` (35 tests), covering both the Python
  and native backends.
  - **`build_groups` invariants:** same reachable mutations per haplotype;
    single paths (matmul dosages never reach 2); groups are mutation-free
    with 2..B parents and one child; samples at or below the threshold are
    untouched; leaves are ordered by independently computed spans; stats
    match graph deltas.
  - **Empty spans:** a hand-built graph covers a group whose members all have
    empty spans.
  - **Python/native parity:** identical graphs over three generations.
  - **Cache consistency:** recombination after grouping stays exact,
    including dosages.
  - **Stats:** checked against the graph (edge and node deltas, how new nodes
    split into samples, bubbles and groups), not against their formulas.
  - **Failure paths:** a bad `compact_dir` fails before meiosis; a failed
    save keeps the uncompacted graph; a failed reload keeps the file.
  - **Stale parents:** a shrinking population keeps the parent's `n`.
  - **Exactness:** GRG genotypes equal dense every generation with option B
    on, with and without compaction.
  - **Compaction:** the compacted graph equals the live part of the
    uncompacted one; groups survive; `bp_range` is kept; the parent graph is
    released.
  - **Checkpoints:** round trip; a compacting simulation resumed from a
    checkpoint matches an uninterrupted run.
  - The 7 compaction tests skip unless pygrgl has `drop_unreachable`.
- Ran the adversarial review (`devtools/adversarial_review.md`) with a fresh
  agent. It found no genotype or dosage violation: dense comparisons were exact
  in 13 (B, K, `compact_every`) configurations. Its findings are fixed:
  - compaction failure handling;
  - the `group_min_parents` docstring;
  - the demo holding the founders' graph;
  - a comment on the node-ordering flag;
  - weak or tautological tests.
- Not fixed, because it predates option B and existing tests rely on it
  (`test_checkpoint_grg.py::test_grg_founder_persisted_natively`): a retained
  GRG generation that meiosis ran on still wraps the offspring's graph, so the
  checkpoint saves it as `gen_{t-1}.grg` with generation t's genotypes.
  `GraphHaplotypeOperator.stale` now flags such operators.
- `scripts/demo_stdpopsim_grg.py`: added `--option-b`, `--fanin`,
  `--min-parents` and `--compact-every`, so `--check-dense` can confirm
  option B. It prints `EXACT MATCH` with and without compaction.
- `xftsim/native/test_parity.py` was not extended: it imports
  `simulate_grg_recombination` from the old `benchmark/` tree, which xftsim
  does not have, so it cannot run here. The parity checks are in the pytest
  file above instead.

### GRG Edge-Growth Instrumentation and Option B Prototype (2026-11-12)

- Added `scripts/scaling/optionb/`: scripts that explain why GRG edges grow
  quadratically over generations, and a prototype of the fix (option B).
  - **Explanation:** recombinant offspring re-list their parents' pieces, at about
    13-18 edges per ancestry tract, and dead lineages are never removed.
  - **Option B:** a per-offspring position hierarchy of mutation-free group
    nodes, plus dead-node removal.
  - `exp_scale.py` runs option B at 100k-500k, with an exactness twin of the
    current code. `README.md` summarizes the scripts, findings and limits.
- Compaction options:
  - `--rebuild-every` works with stock pygrgl.
  - `--save-drop-every` needs the `drop_unreachable` save option from the local
    grgl branch `drop-unreachable`.

### GRG Scaling Sweep Harness (2026-11-01)

- Added `scripts/scaling/` — time and memory of xftsim + GRG simulations on
  stdpopsim founders (OutOfAfrica_3G09, CEU, HapMapII_GRCh38 map).
  `scaling_run.py` runs one setting and appends a CSV row per stage and per
  generation (time, meiosis time, GRG nodes and edges, var(G)/var(Y),
  current and peak memory), so a run killed for memory or time still records
  how far it got. `grid.tsv` lists the sweep (population size, region length,
  time horizon); `setup_lane.sh` builds the environment on the Lane cluster
  from the bundled grgl source (branch `grg_modify_improve`, commit
  `8e716cb`) and checks GRG against dense; `submit_sweep.sh` submits the grid
  to Slurm; `run_local.sh` runs it on a workstation; `summarize.py` and
  `plot_scaling.py` build the summary table and figures.

### Demo script: `--restandardize` replaces `--fixed-effects` (2026-10-16)

- Fixed allele effects are now the simulator's default
  (`standardize_with='founders'`), so `scripts/demo_stdpopsim_grg.py`
  drops `--fixed-effects` (which built per-allele effects by hand) and adds
  `--restandardize` to show the previous per-generation re-standardization.
- `tests/numerical/test_founder_standardization.py`: genetic values in both
  modes checked against the spec formula computed directly from genotypes,
  the var(G) ratchet under `'current'` versus heterozygosity decay under
  `'founders'`, and checkpoint resume (including pre-change checkpoints).
- `tests/pipeline/test_grg_stdpopsim_sim.py`: GRG/dense parity now runs in
  both modes, each with freshly simulated founders (GRG meiosis mutates the
  founders' graph in place, so a shared fixture cannot be reused across
  runs).

### stdpopsim + GRG Demo Script (2026-10-06)

- Added `scripts/demo_stdpopsim_grg.py` — command-line walkthrough of
  stdpopsim founders -> GRG -> forward simulation with GRG-native meiosis,
  printing per generation the timing, GRG node and edge counts with their
  percentage growth since the previous generation (plus total growth since
  the founders), and var(G)/var(Y). Options: `--pops`, `--model`,
  `--chrom`, `--left-mb`, `--length-mb`, `--genetic-map`, `--gens`, `--h2`,
  `--causal-min-maf`, `--seed`; `--fixed-effects` freezes per-allele
  effects at founder allele frequencies instead of re-standardizing every
  generation (v0.9 default), to contrast the two var(G) trajectories;
  `--check-dense` reruns the simulation on dense genotypes and reports exact
  agreement (genotypes are extracted in variant blocks rather than via
  `to_dense()`'s m x m identity); `--show-guard` shows meiosis refusing
  founders that keep recurrent mutations.

### GRG + stdpopsim Regression Tests (2026-10-06)

- Added `tests/pipeline/test_grg_stdpopsim_sim.py` — seeded stdpopsim
  founders run four generations on GRG-backed and dense haplotypes with the
  same seed and a `from_haplotypes_with_cM` map; genotypes and genetic
  values must agree exactly. The existing GRG parity tests use synthetic
  GRGs without recurrent mutations, and the oracle tests compare mutation
  *positions*, so neither could see the duplicate-mutation column swap.
- Added `tests/unit/test_grg_duplicate_mutations.py` — builds a
  duplicate-free tree sequence plus one site with two identical A->T
  mutations, checks that GRG meiosis rejects exactly that duplicate (the
  test fails if the guard is disabled) and that
  `_drop_multi_mutation_sites` fixes it; msprime founder seeding and
  filtering.
- Added `tests/unit/test_crossover_sampler.py` — per-locus switch
  frequencies and independence for both sampler routes (Poisson events and
  per-locus draws), route selection, input validation, and equivalence of
  the vectorized `_phase_to_segments` / `_switches_to_segments` with the
  original per-locus loop.
- Extended `tests/unit/test_recombination_map.py` with
  `from_haplotypes_with_cM` tests, including chromosome starts that share a
  `pos_bp` with the previous chromosome.
- The changes went through the `devtools/adversarial_review.md` protocol
  before commit; the review's findings (chromosome-start linkage, a guard
  test that passed without the guard, shared msprime seeds, slower
  sampling for constant intermediate-`p` maps) are fixed in the same
  commit.

### C++ Native Extension Build System (2026-06-27)

- Added `xftsim/native/` — CMake + pybind11 build system for the C++
  `NonDuplicationRecombiner`. Links against the grgl source tree as a CMake
  subproject (auto-detects relative paths, overrideable via `-DGRGL_ROOT`).
  Uses grgl's bundled pybind11 to ensure ABI compatibility with `pygrgl`.
- Added `tests/unit/test_grg_oracle.py` — 25 mutation-level correctness
  tests using a GRG-walk oracle to verify both the Python and C++ backends
  produce exactly the right inherited mutations per offspring. Cross-backend
  parity tests confirm the two implementations agree cell-by-cell.
- Added `xftsim/native/test_parity.py` — standalone parity harness for
  development-time testing of the C++ backend against the Python reference
  outside of pytest.

### Test Suite Split: Unit vs Notebook Integration (2026-05-26)

- Renamed `tests/integration/` → `tests/pipeline/`. The 35 hand-coded
  simulation-pipeline tests were not actually executing the example
  notebooks; this rename surfaces what they actually do (Python-coded
  end-to-end Simulation runs).
- Added a new `tests/integration/` populated exclusively with
  notebook-execution tests: one module per `docs/examples/0N_*.ipynb`,
  each loading the notebook, executing its code cells in a fresh
  namespace, and asserting that the resulting scientific quantities
  (variances, correlations, regression coefficients, etc.) are within
  tolerance of their expected values.
- The notebook runner lives in `tests/integration/conftest.py` as
  `run_notebook(path) -> dict`. It does not spin up a Jupyter kernel —
  the example notebooks are plain numpy / pandas without magics, so a
  direct `exec` is faster and gives us the post-execution globals for
  assertions.
- Per-notebook test modules:
  - 01-05: fast, run by default
  - 06: `xfail` (documents the stale
    `HasemanElstonEstimator(filter_name='sibpair')` call in the notebook
    that the repo owner has chosen not to patch)
  - 07: `skipif` on missing `pygrgl` + glink fixture files
  - 08: `skipif` behind `XFTSIM_RUN_SLOW_EXAMPLES=1` (notebook §5 runs
    n=10k, m=2k OLS regression)
- New layout reflects the project testing philosophy: **unit** = function
  I/O verification; **integration** = run example notebooks and verify
  results within tolerance. `tests/numerical/` and `tests/manuscript/`
  remain as specialised middle layers.

### Adversarial Review Workflow (2026-03-16)

- Added `devtools/math_spec.md` — mathematical specification extracted from manuscript
  - Defines 10 checkable invariants (standardization, h2 round-trip, HE formula, etc.)
  - Source of truth hierarchy: manuscript → math_spec → code → tests
- Added `devtools/adversarial_review.md` — per-commit review protocol
  - Structured checklist covering standardization, effect sizes, estimators, test quality
  - Red flags section for common bug patterns (tautological tests, wide tolerances)
  - Output format template for review reports
- Updated `devtools/claude.md` with adversarial review section and devtools manifest

### Dev Environment Overhaul (2026-02-20)

- Replaced micromamba `xftsim-test` env with standard Python venv (`.venv`)
- Added `scripts/setup-dev.sh` — auto-detects Python >=3.10, creates venv, installs deps
- Added `requirements-lock.txt` via pip-freeze for reproducible environments
- Added dependency tiers in `setup.py`: `[legacy]`, `[docs]`, `[dev]`, `[all]`
- Bumped numba constraint from `>=0.56` to `>=0.58` (Python 3.10+ only)
- Added Python <3.12 runtime warning in `xftsim/__init__.py`
- Updated `devtools/build_docs.sh` to prefer `.venv` over micromamba
- Simplified CI to use `pip install -e ".[legacy,dev]"` instead of manual dep list
- Fixed `pyproject.toml` build-backend (`setuptools.build_meta` replacing invalid `_legacy`)
- Replaced deprecated `pkg_resources` with `importlib.resources` in legacy data module

### Added
- Test suite with pytest (`tests/` directory)
- Demo testing framework (`tests/test_demos.py`) covering UGRM and BGRM demos
- pytest configuration (`pytest.ini`)
- `claude.md` with project documentation, roadmap, and AI assistant instructions
- Version management system (`devtools/bump_version.py`)
  - Pre-commit hook auto-increments dev version on each commit
  - Manual script for release version bumps (patch/minor/major)
- `devtools/install_hooks.sh` for setting up git hooks
- `devtools/build_docs.sh` for building documentation locally
- Split changelogs: `CHANGELOG.md` (software) and `devtools/CHANGELOG.dev.md` (workflow)

### Changed
- Updated `.gitignore` to exclude test virtual environments
- Reorganized development tools into `devtools/` directory
- Symlinked `claude.md` from `devtools/` to project root

### Fixed
- API documentation: removed duplicate automodule directives (warnings reduced from 1551 to 212)
- Removed references to non-existent `lsmate` module
- Removed references to non-existent `extensions` page
- Fixed `source_suffix` in Sphinx config to properly handle .md files
- **Fixed API autodoc**: ReadTheDocs now installs xftsim package; local builds use xftsim-test environment
- Updated `build_docs.sh` to automatically use xftsim-test environment for proper API generation
- Updated `.readthedocs.yaml` to install xftsim package during docs build
