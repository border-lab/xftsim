# Changelog

All notable changes to xftsim will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

For development workflow changes (testing, CI/CD, tooling), see [devtools/CHANGELOG.dev.md](devtools/CHANGELOG.dev.md).

## [Unreleased]

### Added

- **Native cross-mate correlation solver (`xftsim/matchsolver.py`),
  replacing Hexaly as the default for `GeneralAssortativeMating`**: the
  proprietary Hexaly Optimizer is no longer required to use general
  assortative mating. `GeneralAssortativeMating` gains a `solver`
  parameter, `'native'` (new default) or `'hexaly'`, and the
  `import hexaly.optimizer` in `__init__` now happens only when
  `solver='hexaly'` is requested — previously the class could not even be
  constructed without a Hexaly license.

  The native solver exploits the fact that the objective depends on the
  permutation only through the K x K statistic `M(P) = Z' P Y`, so the
  search happens in K^2 dimensions rather than over n x n structures. A
  greedy residual-tracking construction is followed by swap local search
  in which each candidate swap changes `M` by a rank-1 matrix, giving an
  exact O(K^2) objective delta. Both stages are numba-jitted (numba is
  already a hard dependency) and the final residual is recomputed from the
  permutation, so reported diagnostics carry no accumulated float drift.

  Practical consequences: memory drops from the O(n^2) of the
  Koopmans-Beckmann encoding Hexaly was given to O(nK), so mate groups of
  10^5 are routine rather than infeasible. Measured on jointly-normal test
  instances with K = 10, reaching 0.005 max absolute correlation error:
  n = 20,000 in 6-7 s and n = 100,000 in 27-29 s, on a 14-core machine with
  the JIT already warm. Neighbor-list construction is queried in parallel;
  the rest of the solve is single-threaded. Smaller K is much cheaper
  (K = 5 at n = 20,000 takes under a second) and larger K much dearer
  (K = 15 does not reach 0.005 at n = 20,000 within the default budget).
  Wrapping in `BatchedMating` speeds large solves up but is not accuracy-free.
  The merged cross-correlation is the size-weighted average of the per-batch
  values, and because every batch chases the same target on similarly
  distributed phenotypes the per-batch residuals share a direction and do not
  cancel: the merged error stays at roughly the per-batch floor rather than
  averaging below it. A batch must therefore be large enough to hit the target
  on its own, so `BatchedMating` now defaults to `max_batch_size='auto'`,
  sizing batches to the smallest that attains the inner regime's tolerance
  (see `xftsim.matchsolver.min_pairs_for_tol`) and warning when the whole
  sample is too small to reach it. The previous fixed default of 1000
  individuals was below the floor for K >= 5 at `tol = 0.005` and silently
  produced batches that each missed the target.

  The attainable error grows with the number of components K and shrinks with
  the pairs per batch: measured minimum pairs to reach `tol = 0.005` were
  about 250 at K = 3, 500 at K = 5, 2,000 at K = 8, and 8,000 at K = 10. For
  K up to about 6 the target is reachable essentially exactly; above that the
  binding limit is how far the local search drives the residual within the
  evaluation budget, and above K = 10 the batch-sizing estimate is
  extrapolated and warns. A target below the reachable floor for the given
  sample and K can never be met: the solver detects this stall and returns
  early instead of exhausting `max_evals`, and the
  warning it raises names the floor and suggests loosening `tol` or
  simulating more individuals rather than simply spending more compute.

  Convergence is now visible rather than silent: `solver_params['tol']`
  (default 0.005, max absolute entrywise correlation error) sets the
  target, a `UserWarning` is emitted if the solve finishes above it, and
  `regime.last_result` exposes the achieved residual matrix, evaluation
  count, and convergence flag (it is `None` before the first `mate()` call
  and under `solver='hexaly'`, which reports no diagnostics). Infeasible
  targets — those whose implied joint correlation matrix is not positive
  definite — warn explicitly instead of quietly returning a best effort.
  `solver_params['stall_evals']` bounds how long the solver keeps trying
  without improving before it gives up.

  Backward compatibility: checkpoints written before this change carry no
  `solver` key and are deserialized with `solver='hexaly'`, so existing
  runs resume on exactly the solver they were created with.
  `solver_params` keys are validated against the selected solver, so
  Hexaly-only keys (`nb_threads`, `time_limit`, ...) raise rather than
  being silently ignored by the native path.

- **Meiosis crossover sampling is now tied to `sim.rng`**: Previously the
  meiosis kernels in `xftsim/reproduce.py` drew per-locus crossover
  indicators via `np.random.binomial` against numba's internal RNG, with
  no connection to the `Simulation`'s seeded `self.rng`. Two
  `Simulation(seed=42).run(N)` calls therefore produced **different**
  haplotypes — the simulation seed only controlled noise / mate
  assignment, not transmission. This is the bug discussed at the end of
  the prior checkpointing session: `save → from_checkpoint → continue_run`
  was rng-state deterministic but not haplotype deterministic.

  Fix uses `np.random.SeedSequence.spawn(n_offspring)` to derive one
  independent uint32 seed per offspring from `self.rng`. The dense kernel
  (`_meiosis_3d`) re-seeds numba's thread-local RNG via
  `np.random.seed(seeds[i])` at the top of each `nb.prange` iteration, so
  the result is invariant to thread scheduling — the same per-offspring
  draws happen regardless of which thread picks up which `i`. `nb.prange`
  is preserved (no parallelism lost). Memory cost is `4 * n_offspring`
  bytes for the seed array (negligible).

  For the GRG-native meiosis path the same seed derivation feeds a new
  JIT'd helper `_meiosis_pair_seeded(p, seed)` that seeds and draws both
  maternal and paternal phase vectors inside a single JIT call. This was
  needed because Python-level `np.random.seed()` does **not** propagate
  to numba's internal RNG (despite what older numba docs suggest);
  seeding and drawing have to live in the same JIT function. The pair
  helper matches the dense kernel's per-offspring behavior exactly
  (one seed, two consecutive draws against the same numba stream), so
  given the same `rng` both meiosis paths sample the same phases.

  API surface: `HaplotypeOperator.meiosis(assignment, recombination_map,
  rng=None)` — `rng` is a new optional keyword. `Simulation.run` and
  `Simulation.continue_run` pass `self.rng`; direct callers that omit
  `rng` get the historical non-deterministic behavior (preserves
  back-compat).

  Test suite in `tests/unit/test_meiosis_determinism.py` (11 tests):
  per-offspring seed determinism, `_meiosis_pair_seeded` round-trip,
  `meiosis()` function determinism, end-to-end `Simulation` two-run
  haplotype equality, GRG path two-run dense-equality.

- **GRG-native meiosis via the bubble-insertion (node-insertion) algorithm**:
  `GraphHaplotypeOperator.meiosis()` now performs recombination directly on
  the GRG instead of materializing to dense and delegating to the numba
  `_meiosis_3d` kernel. Previously every generation rebuilt the full
  `(n, m, 2)` int8 matrix from the GRG, defeating the memory savings of
  holding a GRG-backed founder set. The new path adds offspring sample
  nodes via `pygrgl.MutableGRG.make_node`, adds bubble nodes when a
  query interval requires only a subset of an ancestor's mutations, then
  calls `set_samples` + `sort_mutations` to promote offspring and demote
  parents. Returns a new `GraphHaplotypeOperator` wrapping the same
  (mutated) GRG. The parent operator's view is stale after meiosis by
  design — matches forward-time semantics.

  Implementation in new module `xftsim/grg_recombination.py`:
  `NonDuplicationRecombination` class plus a `_phase_to_segments` helper
  that bridges xftsim's per-locus Bernoulli `RecombinationMap` to the
  algorithm's bp-space segment input. Per-locus phase sampling delegates
  to `xftsim.reproduce._meiosis_i`, preserving the existing dense-meiosis
  distribution (chromosome boundaries forced to p=0.5, per-locus
  probabilities respected). Algorithm follows the recombination spec's
  Node Insertion approach: at each ancestor of a parent haplotype the
  query interval's mutations are either direct-attached, bubbled into a
  new node, or pruned per the standard decision matrix.

  GRG positions for segment construction come from `pygrgl` directly
  (`grg.get_mutation_by_id(i).position`) rather than `variants.pos_bp`,
  because some founder helpers (e.g.
  `founder_haplotypes_from_msprime_grg`) overwrite `pos_bp` with
  `np.arange(m)` as a sequential-index placeholder. Using GRG-internal
  positions keeps the segment intervals aligned with how the algorithm
  filters mutations.

  Includes a `debug_mode` class attribute that, when set to `True`, dumps
  a per-visit decision trace to stdout for each `recombine_multi` call
  (which decision branch fired at each node, what bubble was created,
  what was pruned). Useful for diagnosing the algorithm's behavior on
  specific topologies; off by default.

- **`tests/unit/test_grg.py::TestMeiosis::test_grg_dense_phase_equivalence`**:
  Deterministic equivalence test for the new GRG meiosis path. The dense
  kernel (`_meiosis_3d`) and the GRG path can't be seeded against each
  other directly — the dense kernel calls `_meiosis_i` inside a
  `numba.prange`, whose per-thread RNG state isn't reachable from Python.
  This test sidesteps that by pre-sampling phase vectors outside both
  paths via `monkeypatch.setattr(xftsim.reproduce, "_meiosis_i", ...)`,
  running GRG meiosis with the patched queue, and checking the offspring
  genotypes cell-by-cell against a reference computed by directly
  indexing parent genotypes with the same phases.

### Changed

- **All four GRG loaders switched from `pygrgl.load_immutable_grg` to
  `pygrgl.load_mutable_grg`** ([io.py:336](xftsim/io.py),
  [io.py:673](xftsim/io.py),
  [founders.py:240](xftsim/founders.py),
  [founders.py:410](xftsim/founders.py)).
  `GraphHaplotypeOperator.meiosis()` requires a mutable GRG (calls
  `make_node` / `connect` / `set_samples`); immutable GRGs cannot host
  the recombination algorithm. Mutable GRGs expose the full immutable
  read API.

- **`GraphHaplotypeOperator.meiosis()` return type**: now returns
  `GraphHaplotypeOperator` (wrapping the in-place-mutated GRG) instead of
  `DenseHaplotypeArray`. Consumers that previously relied on the
  dense-after-meiosis behavior should call `offspring.to_dense()`
  explicitly. Three tests updated to reflect the new type:
  - `test_grg.py::test_meiosis_returns_dense` → `test_meiosis_returns_graph`
  - `test_grg_numerical.py::test_gen1_is_dense` → `test_gen1_is_graph`
  - `test_grg_sim.py::test_multi_gen_uses_dense_after_meiosis` →
    `test_multi_gen_stays_graph_after_meiosis`
  Two other tests (`test_meiosis_offspring_count`,
  `test_offspring_alleles_from_parents`) gained a `.to_dense()` call
  before accessing `.genotypes` (an attribute that only exists on
  `DenseHaplotypeArray`).

### Known issues

- **pygrgl `matmul` DOWN and `save_grg` produce wrong results when
  `nodes_are_ordered=False`**, which the recombination algorithm
  produces because it adds bubble nodes via `make_node()` (which
  defaults to `force_ordered=False`). pygrgl's matmul docstring states
  that nodes-are-ordered=True allows iterating NodeIDs as a substitute
  for graph traversal; with the property violated, matmul DOWN returns
  0 for samples that should carry mutations (sometimes also inflated
  counts like 13). `save_grg` separately drops mutation-carrying root
  nodes whose IDs sit above the topological-order range, silently
  losing those mutations on reload. Confirmed via direct comparison of
  `pygrgl.matmul` DOWN output against `get_down_edges` reachability
  walks: edges, mutations, and sample status are all internally
  consistent in the post-meiosis GRG; only `matmul` and `save_grg`
  disagree with the live edge list.

  `GraphHaplotypeOperator.meiosis()` includes a partial workaround: the
  returned operator carries a `_grg_dirty` flag, and the first DOWN-matmul
  call (`matvec`, `matvec_maternal`, `matvec_paternal`, `to_dense`,
  or transitively `standardized_matvec`) triggers `_ensure_fresh_grg()`,
  which saves the GRG to a temp file and reloads it, restoring
  `nodes_are_ordered=True`. This fixes the case where the algorithm only
  direct-attaches offspring to parent sample nodes (no bubbles created —
  trivially correct allele-frequency, AF-stable across generations,
  binary dense output). It does **not** fix the case where the algorithm
  creates bubble nodes (any non-trivial crossover), because save+reload
  drops the bubble nodes during serialization, losing their mutations
  on the offspring side.

  Until the custom pygrgl exposes a `sort_nodes()` (analogous to
  `sort_mutations()`) or fixes matmul/save to traverse the live edge
  list when `nodes_are_ordered=False`, GRG-native meiosis is correct
  in terms of the GRG topology it produces (verified via
  `verify_offspring_mutations`-style up-walks and via a `compute_post_recomb_anc_counts`
  cardinality multitree check — both pass) but downstream phenotype
  computation that flows through `matmul` DOWN can drop mutations
  carried by bubble nodes. Affects multi-segment offspring (any
  realistic recombination rate); affects roughly 0.5–2% of cells on a
  10-individual / 300-variant test depending on phase density.

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

  The returned `GraphHaplotypeOperator` carries real metadata on both axes.
  Per-variant `pos_bp` and ref/alt alleles are read directly from the GRG
  mutations (mirroring the `_extract_variant_meta_from_grg` pattern); per-variant
  `pos_cM` is the cumulative recombination distance integrated from the contig's
  recombination map (`contig.recombination_map.get_cumulative_mass(pos_bp) * 100`),
  so it is correct for non-uniform maps such as `genetic_map="HapMapII_GRCh38"`
  instead of the linear `pos_bp * mean_rate * 100` approximation. `vid` is
  formatted as `"{chrom}:{pos_bp}:{ref}:{alt}"` (PLINK-style). Per-individual
  population labels are read from the stdpopsim TreeSequence
  (`ts.population(ind.population).metadata["name"]`) and surfaced two ways:
  sample IIDs are prefixed with the population name (`"YRI_0"`, `"CEU_3"`,
  `"CHB_4"`), and the full per-individual label is stored on
  `samples.extra["population"]`, where it is automatically consumed by
  `GroupingComponent.get_grouping_variable` for per-ancestry phenotype grouping
  without further plumbing. The `extra` dict is preserved through
  `SampleMeta.subset`, `with_generation`, and meiosis, so labels survive
  multi-generation simulations.
- **tests/integration/test_grg_founders.py and test_grg_founders_stdpopsim.py**:
  Integration tests for the two GRG founder helpers (msprime and stdpopsim
  paths). Both files were brought into line with the repo's integration-test
  conventions: module-level `pytest.importorskip` guards for `pygrgl` /
  `msprime` / `stdpopsim` so the file skips cleanly when those deps are
  unavailable, a module-scoped operator fixture so the expensive `grg convert`
  step runs once per file (~3-4s instead of per-test), and a `TestX` class
  grouping focused single-behavior tests with one-line docstrings (no `print`
  statements, no `__main__` block). The stdpopsim file's assertions cover the
  full metadata surface: type, sample/variant counts, GRG-internal sample count
  (`2 * n`), GRG-vs-`VariantMeta` mutation-count agreement, population labels
  in iid prefix and `extra` dict (with per-population counts matching the input
  `SAMPLES` dict), `pos_bp` monotonic and confined to `[left, right)`, `pos_cM`
  monotonic, ref/alt alleles populated with real nucleotides (not `"0"`/`"1"`),
  and structured `vid` format. 21 tests total (7 msprime + 14 stdpopsim).
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

- **arch.py** -- Architecture system built on a DAG of `ArchNode` objects with
  topological-sort execution. `Architecture` class supports both programmatic
  construction (`arch.add()`) and formula parsing (`Architecture.from_formula()`).
- **sim.py** -- `Simulation` generation loop: meiosis, phenotype computation,
  mating, retention policy, callbacks, early stopping. Stores results as
  `GenerationResult` objects. Supports `run()`, `continue_run()`, and
  `from_checkpoint()`.
- **effect.py** -- `EffectSpec` ABC with three concrete implementations:
  `AdditiveEffects` (dense per-variant weights), `MultivariateEffects`
  (multi-trait effect matrices), and `SparseEffects` (k-causal-variant model
  via `from_h2()`).
- **mate.py** -- `MateAssignment` dataclass and mating regimes:
  `RandomMating` (sex-aware shuffle-pair-expand) and
  `LinearAssortativeMating` (rank-order pairing on standardized phenotypic
  composite with configurable `r`).
- **filter.py** -- `Filter` ABC with implementations: `TrioFilter` (parent-
  offspring trios with cross-generation lookup), `SibPairFilter` (vectorized
  sibling-pair extraction), `UnrelatedFilter`, `AscertainmentFilter`, and
  `SubsampleFilter`. `TrioView` and `SibPairView` dataclasses for downstream
  analysis.
- **stats.py** -- `Statistic` ABC with implementations:
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
  - `save/load_phenotypes_npz` -- PhenotypeArray round-trip
  - `save/load_effects_npz` -- EffectSpec round-trip (all three subclasses)
  - `save/load_architecture` -- Architecture via JSON metadata + effect .npz
    files, handling all component types
  - `save/load_simulation_checkpoint` -- Full simulation state to directory
    (architecture, haplotype/phenotype/pedigree histories, RNG state, mating
    regime, recombination map, metadata)
  - `load_grg` -- GRG file to `GraphHaplotypeOperator` with optional BIM
  - `genotypes_to_pseudo_haplotypes` -- conversion utility
- **gwas.py** -- Vectorized GWAS (beta/SE/t/p per variant) and PGS scoring
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
- `PhenotypeArray` -- dict-like phenotype container with set/get/contains/
  keys/subset operations
- `PedigreeArray` -- offspring-indexed pedigree with maternal/paternal index
  arrays
- `HaplotypeArrayAccessor` -- compatibility accessor for legacy code

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
- Unit tests for every module: struct, effect, arch, parser, sim, mate,
  filter, stats, io, gwas, cli, reproduce, haplotype operations, GRG
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

- Sphinx API reference (`docs/api/`): 12 RST files covering sim, arch,
  effect, mate, filter, stats, gwas, io, struct, parser, cli
- Sphinx guides (`docs/guides/`): quickstart, formula DSL reference
- Sphinx configuration with autodoc, intersphinx, viewcode, nbsphinx
- Numpy-style docstrings across all new modules (sim, arch, effect, mate,
  filter, stats, io, struct, parser, gwas, founders)
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
  `PhenotypeArray`, `PedigreeArray`). The xarray accessor pattern is removed
  in favor of direct attribute access.
- **Effect system**: Replaced old `AdditiveEffects` (xarray-backed, tied to
  `ComponentIndex`) with `EffectSpec` ABC and three implementations
  (`AdditiveEffects`, `MultivariateEffects`, `SparseEffects`). Effects are
  plain numpy arrays fixed at architecture creation time.
- **Architecture**: Replaced manual component ordering and explicit index
  construction with a DAG-based `Architecture` class. Components are added
  via `arch.add()` or parsed from formula strings. Execution order is
  determined automatically by topological sort.
- **Simulation loop**: Replaced `XftSimulation` with `Simulation`. The new
  loop is: meiosis -> phenotype computation (via architecture DAG) -> mating
  -> repeat, with configurable retention policy for memory management.
- **Mating**: Replaced `MatingRegime`/`RandomMatingRegime` with `RandomMating`
  and `LinearAssortativeMating`. The new mating system uses sex-aware
  shuffle-pair-expand and produces `MateAssignment` dataclasses.
- **Filters and statistics**: Replaced old filter/stats system with new ABC-
  based `Filter` and `Statistic` hierarchies. Filters are now cross-generation
  aware (`filter.apply(generation, phenotype_history, pedigree_history)`).
- **I/O**: Replaced scattered save/load functions with a unified serialization
  stack supporting all data types, architectures, and full simulation
  checkpoints.
- **DemoSimulation**: Rewrote to use new system (`Simulation` + `arch` +
  `mate`) while preserving the same public API.
- **README.md**: Complete rewrite reflecting the new API, formula DSL, quick
  start example, feature list, and module reference table.
- **Type hints**: Added `from __future__ import annotations` and PEP 604
  union syntax across 10 modules (sim, arch, effect, mate, filter,
  stats, parser, io, founders, gwas).
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
