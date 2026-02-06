# xftsim Testing Specification

**Date:** 2026-02-06
**Status:** Draft — from brainstorming session

---

## Principles

1. **Layered tests:** unit (fast, deterministic) / integration (sim loop) / numerical (stochastic, slow)
2. **Seed control for unit/integration:** deterministic, reproducible
3. **Stochastic defense for numerical tests:** random seeds logged on failure, not fixed in source — prevents agents from gaming test targets
4. **Theory-derived tolerances:** where possible, SE from formula (e.g., `O(1/sqrt(N))`), generous fixed tolerances otherwise. Never empirically tuned.
5. **Human review list:** when a tolerance is uncertain, a test is flaky, or an expectation might be wrong — flag it, don't silently adjust. Agents should ask for help immediately when theoretical expectations aren't obvious.

---

## Test Organization

```
tests/
├── unit/
│   ├── test_struct.py        # SampleMeta, VariantMeta, PhenotypeArray, PedigreeArray
│   ├── test_haplotype.py     # DenseHaplotypeArray (matvec, subsetting, etc.)
│   ├── test_effect.py        # EffectSpec classes
│   ├── test_parser.py        # Formula parser (Phase 1: minimal, Phase 3: full)
│   ├── test_arch.py          # Architecture, ArchNode, toposort, execution
│   ├── test_grouping.py      # | operator: all resolution paths × usage modes
│   ├── test_multioutput.py   # Multi-output DAG nodes
│   ├── test_mate.py          # Mating regimes
│   ├── test_callback.py      # Callback firing, early stopping
│   ├── test_filter.py        # Filter correctness (trios, sib pairs)
│   └── test_io.py            # I/O round-trip tests
├── integration/
│   ├── test_simple_sim.py    # Single-gen, multi-gen, minimal architecture
│   ├── test_vt_sim.py        # Vertical transmission + founder fallback
│   ├── test_ige_sim.py       # Indirect genetic effects
│   ├── test_multivariate_sim.py  # Bivariate+ architectures
│   └── test_pedigree.py      # Pedigree integrity across generations
├── numerical/
│   ├── test_covariance.py    # Phenotypic covariance structure matches construction
│   ├── test_mating.py        # Spouse correlations under assortative mating
│   └── test_drift.py         # AF drift properties (if needed)
├── conftest.py               # Thin pytest fixtures wrapping testdata
├── testdata.py               # Deterministic generator functions (seeds explicit)
└── HUMAN_REVIEW.md           # Running list of tests needing human judgment
```

---

## Test Data

### Factory module: `tests/testdata.py`

Deterministic generators with explicit seeds. Usable inside and outside pytest.

```python
class TestGenomes:
    @staticmethod
    def simple(n=500, m=100, seed=42) -> DenseHaplotypeArray: ...

    @staticmethod
    def biallelic_known_af(n, af, seed=42) -> DenseHaplotypeArray: ...

    @staticmethod
    def two_chrom(n=500, m_per_chrom=50, seed=42) -> DenseHaplotypeArray: ...

class TestEffects:
    @staticmethod
    def additive(m=100, h2=0.5, seed=42) -> AdditiveEffects: ...

    @staticmethod
    def multivariate(m=100, h2=[0.5, 0.3], rg=0.2, seed=42) -> MultivariateEffects: ...

class TestMeta:
    @staticmethod
    def samples(n=500, n_fam=100, seed=42) -> SampleMeta: ...

    @staticmethod
    def variants(m=100, n_chrom=2) -> VariantMeta: ...
```

### Fixtures: `tests/conftest.py`

Thin wrappers calling testdata functions. Pytest dependency injection.

---

## Unit Tests

### Data structures (`test_struct.py`)

**SampleMeta:**
- Construction with all fields
- Default FID (same as IID)
- Boolean subsetting preserves alignment
- Extra field access
- Shape consistency after subsetting

**VariantMeta:**
- Construction, bracket access for extras
- Subsetting preserves field alignment

**PhenotypeArray:**
- Get/set by string key
- Warn on duplicate name
- Boolean subsetting preserves SampleMeta alignment
- SampleMeta travels with data

**PedigreeArray:**
- Construction with valid indices
- Index bounds checking

### HaplotypeOperator (`test_haplotype.py`)

**DenseHaplotypeArray:**
- Construction from (n, m, 2) array
- `matvec(effects)` matches manual `G @ effects`
- `matvec_maternal(effects)` matches `hap[:,:,0] @ effects`
- `matvec_paternal(effects)` matches `hap[:,:,1] @ effects`
- `standardized_matvec` produces correct scaling given known AFs
- `rmatvec(phenotypes)` matches `G.T @ phenotypes`
- Boolean subsetting on samples and variants
- `recompute_af()` matches `np.mean(hap, axis=(0,2))`
- Shape consistency: n_samples, n_variants properties
- `to_dense()` returns self

### Effects (`test_effect.py`)

- `AdditiveEffects.from_h2`: output shape matches n_variants
- `AdditiveEffects.from_matrix`: round-trips correctly
- `MultivariateEffects.from_h2_rg`: shapes correct, correlations in valid range
- `MultivariateEffects.from_covg`: matches input covariance
- `SparseEffects.from_h2`: correct number of nonzero entries
- `standardized` flag is stored and accessible

### Parser (`test_parser.py`)

Parser output is a `list[ArchNode]`. Tests compare parsed output against programmatically constructed ArchNodes.

**Phase 1 (minimal grammar) — test categories:**
- Valid univariate component: `height.G ~ genetic(eff)` → correct ArchNode
- Valid noise: `height.E ~ noise(0.2)` → correct ArchNode
- Valid aggregation: `height ~ height.G + height.E` → correct inputs/outputs
- Scalar multiplication: `height ~ 0.3 * height.G + height.E`
- Multi-statement formula: correct node count and wiring
- Effect name resolution: bare names resolved from effects dict
- Toposort: execution order respects dependencies

**Phase 1 error categories:**
- Unknown function name
- Missing effect in effects dict
- Missing LHS
- Missing RHS
- Undefined reference (input name not produced by any node)
- Duplicate output name

**Phase 3 (extended grammar) — additional categories:**
- Tuple LHS: `(A.G, B.G) ~ mvGenetic(eff)` → multi-output node
- Grouping: `noise(0.1) | FID` → grouping field set
- Founder fallback: `parent(X, founder=noise(0.3))` → correct kwarg
- Sibling references: `sibling_mean(X)` → default `| FID` grouping
- Multivariate noise: `(A.E, B.E) ~ cnoise(cov=...)` → correct typing

**Phase 3 error categories:**
- Circular dependency detection
- `|` on non-groupable function (e.g., `genetic(eff) | FID`)
- `cnoise` without tuple LHS
- Invalid grouping variable (not in SampleMeta)

### Architecture and execution (`test_arch.py`)

- Programmatic API: `arch.add(...)` produces correct ArchNodes
- Parsed API produces same ArchNodes as programmatic for equivalent formulas
- Toposort places dependencies before dependents
- Execution: `arch.compute(...)` writes correct values to PhenotypeArray
- Aggregation arithmetic: +, -, *, / produce correct results

### Grouping operator (`test_grouping.py`)

Dedicated section — `|` has three resolution paths × two usage modes:

**Resolution paths:**
- Core SampleMeta field: `| FID`, `| sex`
- SampleMeta extra: `| school`
- Relational via pedigree: `| mother`, `| father`

**Usage modes (generative):**
- `noise(v) | FID`: within-group values identical, between-group independent
- `noise(v) | mother`: maternal half-sibs get same value
- Bare `noise(v)`: equivalent to `noise(v) | IID` (all values independent)

**Usage modes (aggregating):**
- `mean(X) | FID`: within-family mean, broadcast to all members
- `sibling_mean(X) | mother`: mean among maternal half-sibs
- `sibling_mean(X)` (bare): defaults to `| FID`

### Multi-output nodes (`test_multioutput.py`)

- `(A.G, B.G) ~ mvGenetic(eff)`: both outputs written to PhenotypeArray
- Node fires exactly once
- Downstream of A.G and B.G both depend on single node
- Toposort places multi-output node before any dependents of either output

### Mating (`test_mate.py`)

- Random mating: uniform pairing, no self-mating
- Assortative mating: spouse correlation matches target
- Offspring count per pair matches `offspring_per_pair`
- Population size maintained
- Sex-consistent mating (if applicable)
- MateAssignment contains valid SampleMeta + parent index arrays

### Callbacks (`test_callback.py`)

- Callbacks fire each generation in correct order
- `sim.stop = True` halts simulation at correct generation
- Callback receives correct state (generation, histories populated)
- Multiple callbacks execute in list order

### Filters (`test_filter.py`)

- TrioFilter: every trio has (offspring, mother, father) from correct generations
- TrioFilter: indices valid into respective PhenotypeArrays
- TrioFilter: every offspring with both parents retained appears in output
- SibPairFilter: pairs share FID, no self-pairing
- Filters computed once per generation, not per-statistic

### I/O (`test_io.py`)

- Dense format: write → read → compare (exact round-trip)
- DenseHaplotypeArray round-trip preserves haplotypes, SampleMeta, VariantMeta
- PhenotypeArray round-trip preserves values and SampleMeta
- GRG round-trip (Phase 4)

---

## Integration Tests

### Simple simulation (`test_simple_sim.py`)

- Founder generation: haplotypes loaded, phenotypes computed, mates assigned
- Single generation: founder → phenotypes → mates → offspring → phenotypes
- Multi-generation (5-10 gens): runs without error
- History retention: old entries become None at correct generation
- Population size stable across generations

### Vertical transmission (`test_vt_sim.py`)

- Gen 0: `founder=` fallback produces values (not parent lookup)
- Gen 1+: `parent(X)` returns actual parent phenotype values (exact lookup)
- VT coefficient applied correctly

### Indirect genetic effects (`test_ige_sim.py`)

- `mother(height.G, founder=0)` returns mother's genetic component
- `height.G_mat` (maternal haplotype) differs from `mother(height.G)` (mother's diploid)
- Indirect = parent diploid - transmitted haplotype (by construction)

### Multivariate simulation (`test_multivariate_sim.py`)

- Two-trait architecture runs without error
- Cross-trait genetic components come from same mvGenetic node

### Pedigree integrity (`test_pedigree.py`)

- Every offspring has exactly one mother and one father
- Maternal haplotypes trace to correct parent (offspring hap[:,0] came from mother)
- Paternal haplotypes trace to correct parent
- FID assignment consistent with parent pairing
- No self-mating in pedigree
- Pedigree indices valid across generations

---

## Numerical Tests

**Stochastic protocol:**
- Random seed drawn at runtime, logged on failure for reproduction
- Theory-derived tolerances where available (SE ≈ `O(1/sqrt(N))` with generous k)
- Generous fixed tolerances where theory is messy
- When tolerance is uncertain, flag in `HUMAN_REVIEW.md` — don't silently adjust
- Default: N=10,000, k=4 (≈ 1/16,000 false positive rate)

### Phenotypic covariance structure (`test_covariance.py`)

Primary numerical target — direct consequence of construction.

- Independent components have near-zero cross-covariance
- Correlated components (cnoise, mvGenetic) reflect specified covariance
- `Var(A + B) ≈ Var(A) + Var(B) + 2*Cov(A,B)` for aggregation
- Noise variance matches specified value
- Grouped noise: within-group variance ≈ 0, between-group variance ≈ specified

### Mating (`test_mating.py`)

- Random mating: spouse phenotype correlation ≈ 0
- Assortative mating: spouse correlation ≈ specified level
- Assortative mating inflates phenotypic variance over generations (direction check)

---

## CI Strategy

| Layer | Trigger | Speed target |
|-------|---------|-------------|
| Unit | Every commit | < 10s |
| Integration | Every commit | < 60s |
| Numerical | PR / nightly | < 5 min |

### Stochastic failure protocol
1. Log seed + full state as artifact
2. Retry once with new seed — if passes, flag as flaky, don't block
3. If both fail, real failure
4. Flaky tests accumulate on `HUMAN_REVIEW.md` — address periodically

### Performance baselines (Phase 3+)
- Coarse wall-clock upper bounds, not micro-benchmarks
- Catch accidental quadratic blowups
- Run with numerical tests (PR/nightly)

---

## Agent Debugging Protocol

When a test fails, agents should follow this order:

1. **Check test code** for obvious bugs (wrong index, typo, etc.)
2. **If the theoretical expectation isn't obviously correct, ask a human immediately.** Do not attempt to derive quantitative genetics theory. Do not go down rabbit holes.
3. **Check implementation** for bugs
4. **Never silently adjust expected values or tolerances.** Document why in `HUMAN_REVIEW.md` and flag for review.

---

## Phased Test Implementation

### Phase 1: Core abstractions
- `test_struct.py` — all data structure tests
- `test_haplotype.py` — DenseHaplotypeArray tests
- `test_effect.py` — EffectSpec tests
- `test_parser.py` — minimal grammar (univariate, noise, aggregation, errors)
- `test_arch.py` — programmatic API, toposort, basic execution

### Phase 2: Minimal simulation
- `test_mate.py` — random mating
- `test_simple_sim.py` — founder + multi-gen
- `test_pedigree.py` — pedigree integrity
- `test_io.py` — dense format round-trip
- `test_covariance.py` — simple additive model covariance

### Phase 3: Full architecture
- `test_parser.py` — extended grammar (|, founder=, tuple LHS, sibling refs, cnoise)
- `test_grouping.py` — all resolution paths × usage modes
- `test_multioutput.py` — multi-output DAG nodes
- `test_vt_sim.py` — vertical transmission
- `test_ige_sim.py` — indirect genetic effects
- `test_multivariate_sim.py` — bivariate+ architectures
- `test_mate.py` — assortative mating
- `test_callback.py` — callback tests
- `test_filter.py` — filter tests
- `test_mating.py` — numerical mating tests
- `test_covariance.py` — extended covariance tests (multivariate, grouped)
- Performance baselines

### Phase 4: GRG integration
- GraphHaplotypeOperator passes same matvec tests as DenseHaplotypeArray
- GRG meiosis correctness
- GRG duplication for history
- GRG I/O round-trip
- GRG ↔ dense comparison

---

## Statistics (Phase 3)

Initial implementation: `SampleStatistics` class computing within-sample covariance matrix of all components and phenotypes.

Tests:
- Correct keys in GenerationResult
- Output shape matches number of components
- Covariance values match direct numpy computation on PhenotypeArray

Other statistics (HasemanElston, parent-offspring regression, etc.) deferred — tests written alongside implementation per the principle: every Statistic subclass must have tests for (a) correct output keys, (b) correct shape, (c) at least one numerical sanity check.

---

## HUMAN_REVIEW.md

Running list maintained at `tests/HUMAN_REVIEW.md`. Contains:
- Tests with uncertain tolerances
- Flaky test history
- Tests where theoretical expectation needs verification
- Tolerance adjustments with rationale

Reviewed periodically. Agents add to this list; humans resolve items.
