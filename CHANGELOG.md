# Changelog

All notable changes to xftsim will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For development workflow changes (testing, CI/CD, tooling), see [devtools/CHANGELOG.dev.md](devtools/CHANGELOG.dev.md).

## [Unreleased]

### Fixed

- **save_architecture silently dropping unsupported component parameters**:
  Same silent-data-loss shape as the mating regime fix. Previously,
  components not in the hard-coded handler list (`ThresholdComponent`, plus
  any user-defined `ArchComponent` subclass) had their parameters silently
  dropped on save and only failed at load time with `Unknown component type`.
  `save_architecture` now raises `ValueError` immediately for unsupported
  components and validates the full architecture before any disk writes,
  so a partial directory is never left behind. Also added native
  serialization for `ThresholdComponent` (the liability-threshold model).
- **save_simulation_checkpoint silently dropping mating regime parameters**:
  Previously, calling `save_simulation_checkpoint` with a mating regime other
  than `RandomMating` or `LinearAssortativeMating` (e.g. `GeneralAssortativeMating`,
  `BatchedMating`) appeared to succeed but wrote only the regime's class name to
  `meta.json` — all parameters (including `cross_corr`, the matrix that defines
  the experiment) were dropped. The failure surfaced only later, at
  `from_checkpoint` time, with a `ValueError`. `save_simulation_checkpoint` now
  raises `ValueError` immediately for unsupported regimes, and validates the
  regime before any disk writes so a failure doesn't leave a partial checkpoint
  directory. The deserializer's behavior is unchanged.
- **LinearAssortativeMating**: Fixed mating score computation to match legacy
  `LinearAssortativeMatingRegime`. The new code was using the mean of
  standardized traits with r as the direct mixing parameter, producing
  negligible cross-mate correlations for multi-trait scenarios. Now uses the
  sum of standardized traits and computes the latent correlation R = K*r
  (adapting to within-person covariance structure at each generation). This
  reproduces the manuscript's constant-entry xAM results: e.g., 5 traits with
  r=0.2 now produces rg ≈ 0.30 at generation 5 (was ~0.02 before fix).
- **test_simulation.py**: Updated `HasemanElstonEstimator` call to use new
  GRM-based API (`phenotype_keys=`) instead of removed `filter_name='sibpair'`
  parameter.

### Added

- **Checkpoint support for `GeneralAssortativeMating` and `BatchedMating`**:
  Previously these regimes raised at save time (after the loud-fail fix in
  this release). Now `_serialize_mating_regime` / `_deserialize_mating_regime`
  handle both natively, including the recursion required for `BatchedMating`
  to wrap any other supported regime. `GeneralAssortativeMating`'s
  `cross_corr` matrix and `solver_params` round-trip via inline JSON
  (small K, typically <50). The deserialize step constructs via
  `__init__`, which still requires `hexaly` to be installed when restoring
  a `GeneralAssortativeMating` checkpoint — this is the main deployment
  caveat for resuming high-dimensional xAM simulations on machines
  without the optimizer.
- **Checkpoint resume is RNG-state deterministic**: `run(N)` and
  `run(K) → save → from_checkpoint → continue_run(N-K)` now provably end
  with byte-identical `self.rng` state (algorithm name, state key,
  position, Gaussian-cache state). New test
  `test_checkpoint_resume_rng_is_deterministic` exercises this. Note: the
  test deliberately does **not** assert byte-equal haplotypes / phenotypes,
  because meiosis (`xftsim/reproduce.py`) currently uses `np.random`
  globally inside a `numba.prange` parallel kernel, which is racey and
  non-deterministic even within a single process. That nondeterminism is
  pre-existing and orthogonal to checkpointing.
- **Checkpoints persist GRG-backed founders natively**: When
  `sim.haplotype_history` contains a `GraphHaplotypeOperator` (typically the
  founder generation when using `founder_haplotypes_from_msprime_grg` or
  `founder_haplotypes_from_stdpopsim_grg`), `save_simulation_checkpoint`
  now writes a `.grg` file plus a metadata sidecar instead of materializing
  the GRG to a dense int8 haplotype array. This avoids dense-blow-up on
  whole-genome GRGs (e.g. ~64 GB raw at n=8000, m=4M variants), and the
  load path detects the `.grg` per-generation so the round-tripped
  `haplotype_history` keeps the GRG type. Backwards-compatible: older
  checkpoints that wrote materialized-dense `.npz` for GRG founders still
  load (as `DenseHaplotypeArray`). `save_simulation_checkpoint` now also
  raises `TypeError` for haplotype types it doesn't know how to persist,
  rather than silently skipping them.
- **Checkpoint preserves per-generation Statistic results**: `save_simulation_checkpoint`
  now writes `sim.results` (the `list[GenerationResult]` produced by registered
  `Statistic` objects) to `results.pkl` in the checkpoint directory, and
  `NSimulation.from_checkpoint` restores it. Long-running simulations that save
  partway through no longer lose accumulated statistics on resume; new generations
  from `continue_run` append to the loaded list. Backwards-compatible: checkpoints
  produced before this change load with `sim.results = []`.
- **tests/manuscript/**: Manuscript reproduction test suite that validates the
  refactored simulator against published quantitative results (constant-entry
  xAM scenarios from Supplementary Figures S5-S6).
- **founders.founder_haplotypes_from_stdpopsim_grg**: New helper that simulates
  founder genotypes via a stdpopsim demographic model (default
  `HomSap` / `OutOfAfrica_3G09`), converts the resulting TreeSequence to a GRG
  through the grgl CLI, and returns a `GraphHaplotypeOperator`. Mirrors the
  existing msprime-based helper but draws samples per stdpopsim population
  (e.g. `{"YRI": 100, "CEU": 100, "CHB": 100}`). Sub-region selection uses
  `left`/`right` base-pair coordinates (stdpopsim's `length_multiplier` is
  deprecated upstream). When `mutation_rate` is not specified, the function
  falls back to the demographic model's calibrated rate
  (`model.mutation_rate`) when available, avoiding stdpopsim's
  contig-vs-model rate-mismatch warning.
- **tests/integration/test_grg_founders_stdpopsim.py**: Integration test for
  the new stdpopsim-based founder helper, mirroring `test_grg_founders.py`.
- **setup.py**: Added `msprime`, `tskit`, and `stdpopsim` to the `grg`
  extras_require alongside `pygrgl`. These are imported directly at the top
  of `xftsim/founders.py` for the GRG-based founder helpers and were
  previously only available transitively. Install with
  `pip install xftsim[grg]`.

---

## [Unreleased] - ajay branch refactor

Ground-up rewrite of xftsim: new numpy-backed data structures, abstract linear
operators, a lavaan-style formula DSL for architecture definition, GRG
(graph-based genotype) support, GWAS/PGS, a CLI, full I/O serialization, and
comprehensive documentation and testing.

### Added

#### Core modules

- **narch.py** -- Architecture system built on a DAG of `ArchNode` objects with
  topological-sort execution. `Architecture` class supports both programmatic
  construction (`arch.add()`) and formula parsing (`Architecture.from_formula()`).
- **nsim.py** -- `NSimulation` generation loop: meiosis, phenotype computation,
  mating, retention policy, callbacks, early stopping. Stores results as
  `GenerationResult` objects. Supports `run()`, `continue_run()`, and
  `from_checkpoint()`.
- **neffect.py** -- `EffectSpec` ABC with three concrete implementations:
  `AdditiveEffects` (dense per-variant weights), `MultivariateEffects`
  (multi-trait effect matrices), and `SparseEffects` (k-causal-variant model
  via `from_h2()`).
- **nmate.py** -- `NMateAssignment` dataclass and mating regimes:
  `RandomMating` (sex-aware shuffle-pair-expand) and
  `LinearAssortativeMating` (rank-order pairing on standardized phenotypic
  composite with configurable `r`).
- **nfilter.py** -- `Filter` ABC with implementations: `TrioFilter` (parent-
  offspring trios with cross-generation lookup), `SibPairFilter` (vectorized
  sibling-pair extraction), `UnrelatedFilter`, `AscertainmentFilter`, and
  `SubsampleFilter`. `TrioView` and `SibPairView` dataclasses for downstream
  analysis.
- **nstats.py** -- `Statistic` ABC with implementations:
  `SampleStatistics` (within-sample covariance matrices),
  `HasemanElstonEstimator`, `ParentOffspringRegression`, and
  `MatingStatistics`. `GenerationResult` dataclass for per-generation storage.
- **parser.py** -- Formula DSL parser producing `list[ArchNode]`. Supports
  lavaan-style `Y ~ genetic(eff) + noise(0.5)` syntax with one component per
  line. Features include:
  - Tuple LHS for multivariate outputs: `(Y1, Y2) ~ mvGenetic(eff)`
  - Pipe operator `|` for sample-level grouping (FID, sex, mother, father,
    extra fields); implicit `| IID` default
  - `founder=` keyword for parental component fallback at generation 0
  - All built-in functions: `genetic`, `mvGenetic`, `haplotypeGenetic`,
    `noise`, `cnoise`, `mother`, `father`, `parent`, `sibling_mean`,
    `sibling_sum`, `sibling_any`, `sibling_count`, `sibling_eldest`,
    `sibling_youngest`
  - Expression evaluator with shunting-yard algorithm for arithmetic in
    parameter positions
- **io.py** -- Full serialization stack:
  - `save/load_haplotypes_npz` -- DenseHaplotypeArray round-trip
  - `save/load_phenotypes_npz` -- NPhenotypeArray round-trip
  - `save/load_effects_npz` -- EffectSpec round-trip (all three subclasses)
  - `save/load_architecture` -- Architecture via JSON metadata + effect .npz
    files, handling all component types
  - `save/load_simulation_checkpoint` -- Full simulation state to directory
    (architecture, haplotype/phenotype/pedigree histories, RNG state, mating
    regime, recombination map, metadata)
  - `load_grg` -- GRG file to `GraphHaplotypeOperator` with optional BIM
  - `genotypes_to_pseudo_haplotypes` -- conversion utility
- **ngwas.py** -- Vectorized GWAS (beta/SE/t/p per variant) and PGS scoring
  (raw and standardized). Works with any `HaplotypeOperator` backend.
- **cli.py** -- Command-line interface built on `typer` + `rich`:
  - `xftsim run` -- run simulation from YAML/JSON config
  - `xftsim resume` -- resume from checkpoint
  - `xftsim info` -- inspect checkpoint metadata
  - `xftsim demo` -- run built-in demo simulation
  - Auto TTY detection with `--plain`/`--rich`/`--quiet`/`--verbose` flags
  - `build_simulation_from_config()` for programmatic config-to-simulation
- **founders.py** -- Vectorized founder haplotype generation from allele
  frequencies (replaced per-variant loop with broadcasting).

#### Architecture components

- `GeneticComponent` -- diploid additive genetic values via `matvec`
- `MVGeneticComponent` -- multivariate genetic values (inherits from Genetic)
- `HaplotypeGeneticComponent` -- haplotype-specific effects via
  `matvec_maternal`/`matvec_paternal`, with `haplotype=` kwarg
- `NoiseComponent` -- univariate random noise with optional grouping for
  shared environment
- `CNoiseComponent` -- multivariate correlated noise across features
- `AggregationComponent` -- sum of upstream component outputs
- `MotherComponent`, `FatherComponent`, `ParentComponent` -- vertical
  transmission from parental phenotypes with `founder=` noise fallback
- `SiblingMeanComponent`, `SiblingSumComponent`, `SiblingAnyComponent`,
  `SiblingCountComponent`, `SiblingEldestComponent`,
  `SiblingYoungestComponent` -- indirect genetic effects via sibling
  phenotype aggregation within family groups

#### Data structures (struct.py)

- `SampleMeta` -- frozen dataclass for sample metadata (IID, FID, sex,
  generation, extras dict)
- `VariantMeta` -- frozen dataclass for variant metadata (VID, chrom, pos,
  AF, alleles, extras dict) with bracket-access for extra fields
- `HaplotypeOperator` -- abstract base class defining the genotype operator
  interface: `matvec`, `rmatvec`, `matvec_maternal`, `matvec_paternal`,
  `standardized_matvec`, `recompute_af`, `to_dense`, `meiosis`, `__getitem__`
- `DenseHaplotypeArray(HaplotypeOperator)` -- numpy-backed (n, m, 2) array
  implementation with full operator support
- `GraphHaplotypeOperator(HaplotypeOperator)` -- pygrgl-backed GRG
  implementation using graph traversals for matvec/AF without materialization;
  meiosis and getitem delegate to dense
- `NPhenotypeArray` -- dict-like phenotype container with set/get/contains/
  keys/subset operations
- `PedigreeArray` -- offspring-indexed pedigree with maternal/paternal index
  arrays
- `NHaplotypeArrayAccessor` -- compatibility accessor for legacy code

#### GRG integration

- `GraphHaplotypeOperator` wraps pygrgl for graph-based genotype
  representation; lazy import so pygrgl is not a required dependency
- `standardized_matvec` uses centering trick (`G@v - 2*af@v`) to avoid
  materialization
- AF computation via UP traversal with result caching
- `to_dense()` via identity matrix through DOWN haploid traversal
- `load_grg()` I/O function with optional BIM metadata parsing
- Tests use `pytest.importorskip("pygrgl")` for clean skipping

#### Testing (3400+ tests)

- Organized into `tests/unit/`, `tests/integration/`, `tests/numerical/`
- Unit tests for every module: struct, neffect, narch, parser, nsim, nmate,
  nfilter, nstats, io, ngwas, cli, reproduce, haplotype operations, GRG
- Integration tests: formula-based simulations, multi-generation pipelines,
  VT + assortative mating, filters + statistics + callbacks, checkpoint
  resume, CLI end-to-end (subprocess-based), architecture I/O round-trip
- Numerical tests: heritability estimation, genetic/phenotypic variance
  decomposition, allele frequency drift, covariance structure, Mendelian
  segregation, HWE, recombination rates, assortative mating spouse
  correlation, parent-offspring correlation, sibling correlation, VT
  equilibrium convergence, GRG-vs-dense equivalence
- Test factories in `tests/testdata.py`: `TestGenomes`, `TestEffects`,
  `TestMeta`, `TestSimulation`, `TestGRG`
- `stochastic_seed` fixture for reproducible numerical tests

#### CI/CD

- GitHub Actions workflow (`.github/workflows/ci.yml`): Python 3.10/3.11/3.12
  matrix on Ubuntu, pytest with coverage, concurrency groups for stale run
  cancellation

#### Documentation

- Sphinx API reference (`docs/api/`): 12 RST files covering nsim, narch,
  neffect, nmate, nfilter, nstats, ngwas, io, struct, parser, cli
- Sphinx guides (`docs/guides/`): quickstart, formula DSL reference
- Sphinx configuration with autodoc, intersphinx, viewcode, nbsphinx
- Numpy-style docstrings across all new modules (nsim, narch, neffect, nmate,
  nfilter, nstats, io, struct, parser, ngwas, founders)
- Design document: `docs/plans/2026-02-05-refactor-design.md`
- Testing specification: `docs/plans/testing-spec.md`
- Daily development notes in `docs/plans/devnotes/`

#### Example notebooks

- `docs/examples/01_simple_simulation.ipynb` -- univariate getting started
- `docs/examples/02_bivariate_assortative.ipynb` -- multivariate traits with
  assortative mating
- `docs/examples/03_vertical_transmission.ipynb` -- vertical transmission and
  trio analysis
- `docs/examples/04_gwas_pgs.ipynb` -- vectorized GWAS + PGS workflow
- `docs/examples/05_checkpoint_resume.ipynb` -- save/load/resume simulation
- `docs/examples/06_sibling_effects.ipynb` -- sibling components, SibPairFilter,
  Haseman-Elston estimation

#### Performance

- `benchmarks/bench_core.py` -- timing suite for founder generation, meiosis,
  matvec, architecture compute, full simulation, and I/O operations

#### Package metadata

- Updated `setup.py` and `pyproject.toml` with new classifiers, dependencies,
  and `xftsim` CLI entry point
- pytest configuration migrated from `pytest.ini` to `pyproject.toml`

### Changed

- **Data backend**: Replaced xarray-based data structures (~220 usages) with
  numpy-backed arrays and frozen dataclasses (`SampleMeta`, `VariantMeta`,
  `NPhenotypeArray`, `PedigreeArray`). The xarray accessor pattern is removed
  in favor of direct attribute access.
- **Effect system**: Replaced old `AdditiveEffects` (xarray-backed, tied to
  `ComponentIndex`) with `EffectSpec` ABC and three implementations
  (`AdditiveEffects`, `MultivariateEffects`, `SparseEffects`). Effects are
  plain numpy arrays fixed at architecture creation time.
- **Architecture**: Replaced manual component ordering and explicit index
  construction with a DAG-based `Architecture` class. Components are added
  via `arch.add()` or parsed from formula strings. Execution order is
  determined automatically by topological sort.
- **Simulation loop**: Replaced `XftSimulation` with `NSimulation`. The new
  loop is: meiosis -> phenotype computation (via architecture DAG) -> mating
  -> repeat, with configurable retention policy for memory management.
- **Mating**: Replaced `MatingRegime`/`RandomMatingRegime` with `RandomMating`
  and `LinearAssortativeMating`. The new mating system uses sex-aware
  shuffle-pair-expand and produces `NMateAssignment` dataclasses.
- **Filters and statistics**: Replaced old filter/stats system with new ABC-
  based `Filter` and `Statistic` hierarchies. Filters are now cross-generation
  aware (`filter.apply(generation, phenotype_history, pedigree_history)`).
- **I/O**: Replaced scattered save/load functions with a unified serialization
  stack supporting all data types, architectures, and full simulation
  checkpoints.
- **DemoSimulation**: Rewrote to use new system (`NSimulation` + `narch` +
  `nmate`) while preserving the same public API.
- **README.md**: Complete rewrite reflecting the new API, formula DSL, quick
  start example, feature list, and module reference table.
- **Type hints**: Added `from __future__ import annotations` and PEP 604
  union syntax across 10 modules (nsim, narch, neffect, nmate, nfilter,
  nstats, parser, io, founders, ngwas).
- **Founder generation**: Replaced per-variant Python loop with vectorized
  numpy broadcasting in `founders.py`.

### Deprecated

- **Legacy modules moved to `xftsim/legacy/`**: The following 9 modules have
  been relocated. The original module paths (`xftsim/arch.py`,
  `xftsim/sim.py`, etc.) now contain 2-line shim files that re-export from
  `xftsim/legacy/` for backwards compatibility, but these shims will be
  removed in a future release:
  - `arch.py` -- old architecture system
  - `sim.py` -- old simulation loop (`XftSimulation`)
  - `mate.py` -- old mating regimes (`MatingRegime`, `RandomMatingRegime`)
  - `effect.py` -- old xarray-backed effects
  - `filters.py` -- old filter system
  - `stats.py` -- old statistics
  - `index.py` -- old `XftIndex`, `ComponentIndex`, `DiploidVariantIndex`
  - `proc.py` -- old processing utilities
  - `data.py` -- old data loading

### Removed

- **`||` operator**: The double-pipe operator for founder fallback has been
  removed from the formula DSL. Founder fallback is now specified via the
  `founder=` keyword argument on `parent()`, `mother()`, and `father()`
  functions, eliminating `|` vs `||` parser precedence ambiguity.
- **xarray dependency for core data flow**: xarray is no longer used in the
  new simulation pipeline. All core data structures are numpy-backed.
- **Manual component ordering**: Architecture components no longer need to be
  manually ordered. Topological sort determines execution order automatically.
- **Explicit index construction**: `ComponentIndex` and `DiploidVariantIndex`
  are no longer required for the new system. Variant and sample metadata are
  handled by `VariantMeta` and `SampleMeta` dataclasses.

## [0.3.0] - 2026-01-13

### Fixed
- API documentation now builds correctly on ReadTheDocs and locally

### Changed
- Documentation build script now uses xftsim-test environment for proper API autodoc

## [0.2.0] - 2024

### Added
- `__version__` attribute in `xftsim/__init__.py`
- Run/install timing features
- Improved README for peer review

## [0.1.0] - Initial Release

### Added
- Core simulation framework
- Forward-time genetic simulation
- Multiple mating regime implementations (Random, Linear Assortative, K-Assortative, Batched)
- Phenogenetic architecture system
- Statistical estimators (GWAS, Haseman-Elston, heritability)
- I/O support for PLINK, VCF, Zarr formats
- CEU hg19 recombination map
- Comprehensive documentation with Jupyter notebook tutorials
