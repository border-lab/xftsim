# xftsim Refactor Design Document

**Date:** 2026-02-05
**Branch:** ajay
**Status:** In progress — brainstorming phase

---

## Motivation

1. **Usability** — Architecture definition is verbose and error-prone (manual component ordering, explicit index construction)
2. **Flexibility** — Need abstract linear operators (dense numpy, graph-based GRG, future sparse/lazy)
3. **Maintainability** — xarray permeates the codebase (~220 usages), accessor pattern is fragile

## Approach

Ground-up rewrite with phased implementation. Old and new can run in parallel during transition.

---

## Core Data Structures

### SampleMeta

Frozen dataclass holding sample metadata.

```python
@dataclass(frozen=True)
class SampleMeta:
    iid: np.ndarray          # individual IDs (required)
    fid: np.ndarray          # family IDs (default: same as iid)
    sex: np.ndarray          # 0=female, 1=male
    generation: int           # generation number
    extra: dict[str, np.ndarray] = field(default_factory=dict)
```

- Supports subsetting: `samples[bool_mask]` returns new SampleMeta
- `extra` dict for arbitrary metadata (ancestry PCs, batch IDs, etc.)

### VariantMeta

Frozen dataclass holding variant metadata.

```python
@dataclass(frozen=True)
class VariantMeta:
    vid: np.ndarray           # variant IDs (required)
    chrom: np.ndarray = None
    pos_bp: np.ndarray = None
    pos_cM: np.ndarray = None
    af: np.ndarray = None
    alleles: np.ndarray = None
    extra: dict[str, np.ndarray] = field(default_factory=dict)

    def __getitem__(self, key: str) -> np.ndarray:
        """Core fields via attribute, extras via bracket: variants['coding']"""
        if hasattr(self, key) and getattr(self, key) is not None:
            return getattr(self, key)
        return self.extra[key]
```

### HaplotypeOperator

Abstract base class for all genotype representations.

```python
class HaplotypeOperator(ABC):
    samples: SampleMeta
    variants: VariantMeta

    # Core matvec operations
    def matvec(self, effects) -> np.ndarray: ...              # diploid G @ effects
    def rmatvec(self, phenotypes) -> np.ndarray: ...          # G.T @ phenotypes
    def matvec_maternal(self, effects) -> np.ndarray: ...     # hap[:,0] @ effects
    def matvec_paternal(self, effects) -> np.ndarray: ...     # hap[:,1] @ effects

    # Standardized versions (standardizes on the fly using stored AFs)
    def standardized_matvec(self, effects, af=None) -> np.ndarray: ...
    # ... standardized maternal/paternal variants too

    # AF management
    def recompute_af(self): ...  # update empirical AFs from current data

    # Subsetting (returns same type)
    def __getitem__(self, idx) -> HaplotypeOperator: ...
    # Supports: H[sample_bool, variant_bool], H[sample_bool, :], H[:, variant_bool]

    # Meiosis (delegated to implementation)
    @abstractmethod
    def meiosis(self, assignment: MateAssignment, recombination_map) -> HaplotypeOperator: ...

    # Materialization escape hatch
    def to_dense(self) -> DenseHaplotypeArray: ...
```

**Haplotype convention:**
- `genotypes[:, :, 0]` = maternal haplotypes (inherited from mother)
- `genotypes[:, :, 1]` = paternal haplotypes (inherited from father)

**Concrete implementations:**
- `DenseHaplotypeArray` — NumPy-backed `(n, m, 2)` array
- `GraphHaplotypeOperator` — Wraps grapp/glink GRG (Phase 4)

**Standardization:** The operator handles standardization via different matvec methods. Effects are stored as-is with a `standardized: bool` flag. The architecture node picks the right matvec based on the flag. AFs can be optionally recomputed per generation (for drift tracking) or kept fixed from founders.

### PhenotypeArray

Thin wrapper around a flat dict of arrays.

```python
class PhenotypeArray:
    values: dict[str, np.ndarray]   # component/phenotype name → (n,) values
    samples: SampleMeta

    def __getitem__(self, key: str) -> np.ndarray:
        return self.values[key]

    def __setitem__(self, key: str, val: np.ndarray):
        self.values[key] = val
```

- Names like `height.G`, `height.E`, `height` are just string keys — the dot is a human convention, not parsed
- Warn on duplicate names
- SampleMeta travels with the data to prevent misalignment

### PedigreeArray

Integer index arrays computed once at reproduction time.

```python
class PedigreeArray:
    offspring_samples: SampleMeta
    maternal_idx: np.ndarray   # (n,) index into parent generation's SampleMeta
    paternal_idx: np.ndarray   # (n,) index into parent generation's SampleMeta
```

---

## Formula DSL

### Grammar

```
# Univariate components
height.G ~ genetic(eff)
height.E ~ noise(0.2)

# Multivariate components (correlated across traits)
(height.G, bmi.G) ~ mvGenetic(mv_effects)
(height.E, bmi.E) ~ mvNoise(cov=[[0.2, 0.05], [0.05, 0.3]])

# Aggregation (always univariate, supports +, *, -, /)
height ~ height.G + height.E
height ~ height.G + height.E + height.G * height.E   # GxE

# Vertical transmission
height.VT ~ 0.3 * parent(height)
height.VT ~ 0.3 * mother(height) + 0.3 * father(height)

# Founder fallback (|| operator)
height.VT ~ 0.3 * parent(height) || noise(0.3)
# "if no parent (gen 0), sample from N(0, 0.3) instead"
# No fallback specified → warning + default to 0.0

# Haplotype-specific genetic components
height.G_mat ~ haplotypeGenetic(eff, haplotype='maternal')
height.G_pat ~ haplotypeGenetic(eff, haplotype='paternal')

# Grouped/correlated noise (shared environment)
height.famEnv ~ cnoise(0.1) | FID
# "correlated noise grouped by family — siblings get same draw"
# | grouping_var syntax — resolves from SampleMeta

# Sibling reference functions
edu.sib ~ sibling_mean(edu)
risk.sib ~ sibling_count(smoker)
outcome.sib ~ sibling_eldest(outcome)
```

### Parent/Sibling Reference Primitives

**Parent references** (cross-generational):
- `parent(X)` — average of mother and father
- `mother(X)` — mother's value
- `father(X)` — father's value

Always reference `phenotype_history[gen - 1]` + pedigree lookup. Never recompute.

**Sibling references** (within-generation):
- `sibling_mean(X)` — average of siblings' X
- `sibling_sum(X)` — sum
- `sibling_any(X)` — 1 if any sibling has X > 0
- `sibling_count(X)` — count with X > 0
- `sibling_eldest(X)` — eldest sibling's X
- `sibling_youngest(X)` — youngest sibling's X

Siblings defined as same-FID-same-generation.

Cycles avoided by Option A: only reference sibling *components*, not final phenotypes.

### Indirect Genetic Effects

Computed via arithmetic on existing components — no special primitive needed:

```
# Haplotype-specific genetic values
height.G_mat ~ haplotypeGenetic(eff, haplotype='maternal')
height.G_pat ~ haplotypeGenetic(eff, haplotype='paternal')
height.G ~ height.G_mat + height.G_pat

# Indirect = parent's diploid G minus what was transmitted
height.indirect ~ (mother(height.G) - height.G_mat) + (father(height.G) - height.G_pat)
```

### Execution Model

- Parse formula at `Architecture.__init__()` → build DAG → topological sort
- Store sorted execution plan (flat list of operations)
- Each generation: replay the sorted list
- Multivariate nodes `(a.G, b.G) ~ mvEffects(...)` fire once, produce multiple outputs
- Downstream nodes depend on output names, not node identity

---

## Effect Specification

Effects defined separately from the DSL, referenced by name.

```python
class EffectSpec:
    effects: np.ndarray          # (m,) or (m, k) — computed once at creation
    standardized: bool           # True = effects for standardized genotypes
    variant_mask: np.ndarray     # boolean, which variants are causal

class AdditiveEffects(EffectSpec):
    @classmethod
    def from_h2(cls, h2, variants, standardized=True): ...

    @classmethod
    def from_matrix(cls, effects, standardized=True): ...   # manual

class MultivariateEffects(EffectSpec):
    @classmethod
    def from_h2_rg(cls, h2: list, rg, variants): ...       # h2 + genetic correlation

    @classmethod
    def from_covg(cls, covg: np.ndarray, variants): ...     # full covariance matrix

class SparseEffects(EffectSpec):
    @classmethod
    def from_h2(cls, h2, k_causal, variants): ...
```

**Key decisions:**
- Effects sampled once at architecture creation, fixed for all generations
- Support: `h2`, `rg`, full `covg` parameterization
- Manual matrix specification allowed (`from_matrix`)
- Sparse effect overlap handled via composition (separate shared/specific components, not a complex overlap parameter)

### Sparse Effect Overlap via Composition

Instead of parameterizing overlap, decompose into explicit components:

```python
# Shared genetic effects (causal for both a and b)
effects_ab = MultivariateEffects.from_h2_rg(h2=[0.3, 0.2], rg=0.8, variants=shared_mask)

# Trait-specific effects
effects_a = AdditiveEffects.from_h2(h2=0.2, variants=a_specific_mask)
effects_b = AdditiveEffects.from_h2(h2=0.1, variants=b_specific_mask)
```

```
(a.Gab, b.Gab) ~ mvGenetic(effects_ab)
a.Ga ~ genetic(effects_a)
b.Gb ~ genetic(effects_b)
a.G ~ a.Gab + a.Ga
b.G ~ b.Gab + b.Gb
```

---

## Mating

Completely separate from Architecture. Orthogonal concerns.

```python
arch = Architecture(formula, effects={...})
mating = LinearAssortativeMating(
    phenotypes=['height', 'bmi'],   # string keys into PhenotypeArray
    weights=[1.0, 0.5],
    offspring_per_pair=2,
)
sim = Simulation(arch, mating, founders)
```

Mating produces offspring metadata:

```python
class MateAssignment:
    offspring_samples: SampleMeta    # new iids, fids, sex
    maternal_idx: np.ndarray         # index into parent SampleMeta
    paternal_idx: np.ndarray         # index into parent SampleMeta
```

This feeds directly into `HaplotypeOperator.meiosis(assignment, recombination_map)` and into `PedigreeArray`.

---

## Simulation

### Top-Level API

```python
sim = Simulation(
    founder_haplotypes=haplotypes,
    architecture=arch,
    mating_regime=mating,
    recombination_map=recomb_map,
    statistics=[SampleStatistics(), HasemanElston(filter='sibs')],
    filters={'trios': TrioFilter(), 'sibs': SibPairFilter()},
    callbacks=[...],              # optional per-generation hooks
    retain_haplotypes=1,          # how many past generations to keep
    retain_phenotypes=2,          # minimum enforced by architecture's parent() depth
)
sim.run(n_generations=10)
```

### Generation Loop

```
1. Reproduce: haplotypes.meiosis(assignment, recomb_map) → offspring haplotypes
2. Compute phenotypes: architecture.compute(gen, haplotype_history, phenotype_history, pedigree_history)
3. Assign mates: mating.mate(phenotypes) → MateAssignment
4. Compute filters: {name: filter.apply(phenotypes, pedigree_history) for each filter}
5. Estimate statistics: stat.estimate(sim_state, filtered_views)
6. Run callbacks
7. Enforce retention (set old history entries to None)
```

### History (not "stores")

```python
class Simulation:
    haplotype_history: dict[int, HaplotypeOperator | None]
    phenotype_history: dict[int, PhenotypeArray | None]
    pedigree_history: dict[int, PedigreeArray | None]
    generation: int

    @property
    def haplotypes(self):
        return self.haplotype_history[self.generation]

    @property
    def phenotypes(self):
        return self.phenotype_history[self.generation]
```

**Retention:** configurable via `retain_haplotypes` and `retain_phenotypes`. Minimum `retain_phenotypes` enforced by architecture's `parent()` lookback depth. Non-retained entries set to `None`.

**GRG history:** For graph-based operators, history stores duplicated GRG copies (cheap since GRGs are compressed). Roadmap: single GRG with endpoint views when grgl adds support.

### Statistics

Defined upfront, computed eagerly each generation, stored as results.

```python
class GenerationResult:
    generation: int
    statistics: dict[str, Any]
```

### Filters

Named filters computed once per generation, shared across stats:

```python
sim = Simulation(
    ...,
    filters={'trios': TrioFilter(), 'sibs': SibPairFilter()},
    statistics=[
        SampleStatistics(),                            # no filter
        HasemanElston(filter='sibs'),                  # uses sib pairs
        ParentOffspringRegression(filter='trios'),     # uses trios
    ],
)
```

Filters produce structured relational subsets (trios, sib pairs, within-family groups) — more than simple boolean masks.

### Callbacks

Simple `list[Callable]` replacing the formal `PostProcessor` system. For custom per-generation hooks (saving snapshots, logging, etc.).

---

## Module Organization

Flat layout:

```
xftsim/
├── struct.py        # SampleMeta, VariantMeta, HaplotypeOperator, PhenotypeArray, PedigreeArray
├── arch.py          # Architecture, formula parser, execution plan
├── effect.py        # EffectSpec classes
├── mate.py          # Mating regimes, MateAssignment
├── reproduce.py     # RecombinationMap (meiosis delegated to HaplotypeOperator)
├── sim.py           # Simulation loop, history management
├── stats.py         # Statistics classes
├── filters.py       # Filter classes (TrioFilter, SibPairFilter, etc.)
├── io.py            # I/O (GRG + one dense format for now)
├── ped.py           # Pedigree utilities
└── utils.py         # Utilities
```

---

## Phased Implementation Plan

### Phase 1: Core abstractions
- HaplotypeOperator protocol + DenseHaplotypeArray
- SampleMeta, VariantMeta, PhenotypeArray, PedigreeArray
- EffectSpec classes
- Basic formula parser (subset of full spec)
- Unit tests for data structures

### Phase 2: Minimal simulation
- Port meiosis to work with HaplotypeOperator
- Random mating only
- Single additive genetic component
- Prove end-to-end works
- **Checkpoint:** minimal simulation runs correctly

### Phase 3: Full architecture
- Complete formula syntax (multivariate, VT, indirect, sibling refs)
- Dependency graph + topological sort
- Assortative mating
- Statistics and filters
- Port remaining architecture components

### Phase 4: Integration
- GraphHaplotypeOperator (grapp/glink wrapper)
- Additional I/O formats
- Migration guide
- Performance benchmarks

---

## DSL Examples

### Simple univariate with noise
```
height.G ~ genetic(eff)
height.E ~ noise(0.4)
height ~ height.G + height.E
```

### Bivariate with assortative mating
```python
mv_eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.2, variants=variant_meta)

arch = Architecture("""
    (height.G, bmi.G) ~ mvGenetic(mv_eff)
    height.E ~ noise(0.5)
    bmi.E ~ noise(0.7)
    height ~ height.G + height.E
    bmi ~ bmi.G + bmi.E
""", effects={'mv_eff': mv_eff})

mating = LinearAssortativeMating(phenotypes=['height', 'bmi'], weights=[1.0, 0.5])
```

### Vertical transmission
```python
arch = Architecture("""
    height.G ~ genetic(eff)
    height.VT ~ 0.3 * parent(height) || noise(0.3)
    height.E ~ noise(0.2)
    height ~ height.G + height.VT + height.E
""", effects={'eff': eff})
```

### Indirect genetic effects
```python
arch = Architecture("""
    height.G_mat ~ haplotypeGenetic(eff, haplotype='maternal')
    height.G_pat ~ haplotypeGenetic(eff, haplotype='paternal')
    height.G ~ height.G_mat + height.G_pat
    height.indirect ~ (mother(height.G) - height.G_mat) + (father(height.G) - height.G_pat)
    height.E ~ noise(0.2)
    height ~ height.G + height.indirect + height.E
""", effects={'eff': eff})
```

### Overlapping pleiotropic effects via composition
```python
eff_ab = MultivariateEffects.from_h2_rg(h2=[0.3, 0.2], rg=0.8, variants=shared_mask)
eff_a = AdditiveEffects.from_h2(h2=0.2, variants=a_only_mask)
eff_b = AdditiveEffects.from_h2(h2=0.1, variants=b_only_mask)

arch = Architecture("""
    (a.Gab, b.Gab) ~ mvGenetic(eff_ab)
    a.Ga ~ genetic(eff_a)
    b.Gb ~ genetic(eff_b)
    a.G ~ a.Gab + a.Ga
    b.G ~ b.Gab + b.Gb
    a.E ~ noise(0.5)
    b.E ~ noise(0.7)
    a ~ a.G + a.E
    b ~ b.G + b.E
""", effects={'eff_ab': eff_ab, 'eff_a': eff_a, 'eff_b': eff_b})
```

### Shared family environment
```
height.G ~ genetic(eff)
height.famEnv ~ cnoise(0.1) | FID
height.E ~ noise(0.3)
height ~ height.G + height.famEnv + height.E
```

### GxE interaction
```
income.G ~ genetic(eff)
income.E ~ noise(0.3)
income.GxE ~ income.G * income.E
income ~ income.G + income.E + income.GxE + noise(0.1)
```

---

## Open Design Questions (from critique)

### Resolved
1. **Effect standardization** — Operator handles via different matvec methods. AFs can optionally be recomputed per generation. Effects are fixed at architecture creation.
2. **PhenotypeArray namespace** — Flat dict, no structure. Dot in names is a convention, not parsed. Warn on duplicates.
3. **Mating phenotype access** — Mating takes `phenotypes: list[str]` + `weights`. Works for both phenotypes and components since they're all string keys.
4. **Offspring SampleMeta** — Mating regime creates offspring metadata in MateAssignment.
5. **Filtering** — Named filters computed once per gen, shared across stats. Support structured relational subsets (trios, sib pairs).

### In Progress (continuing critique)
6. `| FID` grouping — execution engine needs SampleMeta access
7. Siblings — same-FID-same-generation definition
8. Multivariate node execution — multi-output nodes in DAG
9. Post-processors → callbacks
10. No backwards compatibility shim (clean break)
