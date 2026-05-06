# xftsim Next Steps: Brainstorm & Task List for Ajay

**Date:** 2026-02-09
**Status:** Refactor core is complete (3448 tests passing on `ajay` branch). This document describes the remaining development work and new feature integrations.

---

## 1. External Library Integrations

The big new capability: generate realistic founder haplotypes and recombination maps from coalescent simulations (stdpopsim/msprime) and optionally compress them into GRG format for memory-efficient downstream use.

### 1A. stdpopsim → msprime → xftsim Pipeline (`xftsim/nfounders.py`)

**Goal:** One-liner to generate realistic founder haplotypes from any stdpopsim species/demographic model.

**Pipeline overview:**
```
stdpopsim species catalog
    → msprime.sim_ancestry() + sim_mutations()
        → tskit.TreeSequence
            → xftsim DenseHaplotypeArray (or → GRG → GraphHaplotypeOperator)
```

**Tasks:**

1. **`founders_from_stdpopsim()`** — High-level convenience function
   ```python
   def founders_from_stdpopsim(
       species_id: str = "HomSap",
       demographic_model_id: str | None = None,  # e.g., "OutOfAfrica_3G09"
       chromosome: str | None = None,             # e.g., "chr22"
       genetic_map_id: str | None = None,         # e.g., "HapMapII_GRCh37"
       n_samples: int | dict[str, int] = 100,     # per-population sample sizes
       length: float | None = None,               # for generic contigs
       seed: int | None = None,
   ) -> tuple[DenseHaplotypeArray, RecombinationMap]:
   ```
   - Wraps stdpopsim's `get_species()`, `get_contig()`, `get_demographic_model()`
   - Calls msprime engine internally
   - Returns *both* founder haplotypes and a position-aware recombination map
   - Should handle: single-pop, multi-pop (with population labels in SampleMeta), generic contigs, chromosome subsets

2. **`founders_from_msprime()`** — Direct msprime interface (lower level)
   ```python
   def founders_from_msprime(
       n_samples: int,
       sequence_length: float,
       recombination_rate: float | msprime.RateMap,
       demography: msprime.Demography | None = None,
       mutation_rate: float = 1e-8,
       model: str = "hudson",
       seed: int | None = None,
   ) -> tuple[DenseHaplotypeArray, RecombinationMap]:
   ```
   - For users who want to specify their own demography/parameters directly
   - Fewer dependencies (no stdpopsim needed)

3. **`tree_sequence_to_haplotypes()`** — Converter from tskit.TreeSequence
   ```python
   def tree_sequence_to_haplotypes(
       ts: tskit.TreeSequence,
       generation: int = 0,
   ) -> DenseHaplotypeArray:
   ```
   - Core workhorse: extract genotype matrix from tree sequence
   - `ts.genotype_matrix()` returns `(n_sites, n_haploid_samples)` — need to reshape to `(n_individuals, n_variants, 2)` (our diploid convention)
   - Extract variant metadata: positions, alleles, allele frequencies
   - Extract sample metadata: individual IDs, population labels → `SampleMeta`
   - Handle ploidy correctly (tskit stores haploid samples; we need diploid individuals)

4. **`recombination_map_from_msprime()`** — Convert msprime.RateMap → xftsim RecombinationMap
   ```python
   def recombination_map_from_msprime(
       rate_map: msprime.RateMap,
       variant_positions: np.ndarray,
   ) -> RecombinationMap:
   ```
   - Given variant physical positions (bp) and a continuous rate map, compute per-interval recombination probabilities
   - Formula: `p_i = 1 - exp(-rate * distance_bp)` for Haldane map function (or use the genetic distances directly)
   - Must handle chromosome boundaries, centromere gaps (rate=0 regions)

5. **`recombination_map_from_stdpopsim()`** — From stdpopsim genetic map
   ```python
   def recombination_map_from_stdpopsim(
       species_id: str,
       genetic_map_id: str,
       chromosome: str,
       variant_positions: np.ndarray,
   ) -> RecombinationMap:
   ```
   - Downloads genetic map, interpolates to variant positions
   - Alternative to getting it bundled with `founders_from_stdpopsim()`

**Key design decisions needed:**
- How to handle multi-population samples (store pop labels where? FID field? New SampleMeta field?)
- Whether to thin sites (stdpopsim can generate millions of SNPs per chromosome; xftsim simulations typically use hundreds to thousands)
- Whether to support `length_multiplier` for quick scaled-down simulations
- Sex chromosome handling (stdpopsim supports X/Y; our diploid convention assumes autosomal)

### 1B. GRG Integration Enhancements

**Current state:** `load_grg()` works, `GraphHaplotypeOperator` supports matvec/rmatvec/to_dense. Meiosis materializes to dense.

**Tasks:**

6. **Tree sequence → GRG programmatic conversion**
   - Currently `grg convert input.trees output.grg` is CLI-only
   - Write a Python wrapper: `tree_sequence_to_grg(ts, output_path, **opts)` that calls the CLI or links to C++ directly
   - This enables: `stdpopsim → msprime → TreeSequence → GRG → GraphHaplotypeOperator` in one script

7. **GRG-aware `founders_from_stdpopsim()` variant**
   - Option to return `GraphHaplotypeOperator` instead of `DenseHaplotypeArray`
   - Useful for very large founder cohorts (>10k individuals × >100k variants)
   - Pipeline: simulate → write temp `.trees` → convert to `.grg` → load

8. **Improve GRG meiosis performance** (stretch goal)
   - Currently materializes entire genotype matrix for meiosis
   - Could investigate partial materialization (only needed haplotypes)
   - Or: after gen 0, keep everything dense (GRG mainly useful for founder storage)

### 1C. RecombinationMap Enhancements

**Current state:** `RecombinationMap` stores per-variant recombination probabilities. Supports constant maps and chromosome-boundary enforcement. No physical/genetic position awareness.

**Tasks:**

9. **Position-aware RecombinationMap**
   - Store physical positions (bp) and genetic positions (cM) alongside per-variant probabilities
   - Enable construction from genetic map files (PLINK-format `.map`, HapMap-format, stdpopsim genetic maps)
   - `RecombinationMap.from_genetic_map(positions_bp, positions_cM, variant_positions)`
   - `RecombinationMap.from_rate_map(msprime_rate_map, variant_positions)`

10. **Sex-specific recombination** (stretch goal)
    - Store separate male/female recombination probabilities
    - `RecombinationMap(p_female=..., p_male=...)` with dispatch in meiosis

---

## 2. I/O Format Expansion

### 2A. PLINK BED/BIM/FAM Import

**Current state:** `founder_haplotypes_from_plink_bfile()` exists in `founders.py` but uses legacy `pandas_plink` (broken in some envs). Not integrated with new structures.

**Tasks:**

11. **Robust PLINK reader** — rewrite using `bed_reader` or direct binary parsing
    - Read .bed (genotype matrix), .bim (variant info), .fam (sample info)
    - Map to `DenseHaplotypeArray` with proper `SampleMeta`/`VariantMeta`
    - Handle missing genotypes (PLINK missing → NaN or imputed)
    - Pseudo-phasing for unphased data (random assignment of heterozygous alleles to maternal/paternal)

12. **PLINK writer** — export simulation results
    - `save_haplotypes_plink(haplotypes, prefix)` → .bed/.bim/.fam
    - Useful for downstream analysis with PLINK, GCTA, LDSC, etc.

### 2B. VCF Import/Export

13. **VCF reader** — load phased VCF as founder haplotypes
    - Use `cyvcf2` or `pysam` for reading
    - Extract GT field (phased haplotypes)
    - Map to `DenseHaplotypeArray`

14. **VCF writer** — export to VCF format
    - Write phased VCF with sample/variant metadata
    - Useful for interop with bcftools, BEAGLE, etc.

### 2C. Tree Sequence I/O

15. **TreeSequence export** — write xftsim haplotypes as .trees file
    - Enables roundtrip: `msprime → xftsim → .trees → tsinfer/tsdate`
    - Complex: need to construct proper tskit tables (nodes, edges, sites, mutations)
    - May be lower priority; start with just the import direction

---

## 3. Multi-Chromosome / Multi-Locus Support

**Current state:** Simulations operate on a single contiguous genotype matrix. `VariantMeta` stores `chrom` and `pos_bp` but the simulation loop doesn't treat chromosomes specially. `RecombinationMap` enforces `p=0.5` at chromosome boundaries.

**Tasks:**

16. **Multi-chromosome founder generation**
    - `founders_from_stdpopsim()` should support simulating multiple chromosomes
    - Option A: concatenate into one genotype matrix with chromosome boundaries in VariantMeta (current approach works)
    - Option B: separate HaplotypeOperator per chromosome (architectural change)
    - Richard's preference needed — concatenation is simpler and the RecombinationMap already handles boundaries

17. **Genome-wide simulation helpers**
    - `founders_from_stdpopsim_genome()` — simulate all autosomes for a species
    - Concatenates chromosomes, builds combined RecombinationMap with proper boundaries
    - Very large (hundreds of thousands of variants for realistic sims) — may want GRG backend

---

## 4. Enhanced Phenogenetic Architecture Features

### 4A. Effect Size Distributions from DFE Catalog

**stdpopsim includes a DFE (distribution of fitness effects) catalog.**

18. **DFE-aware effect generation**
    - `AdditiveEffects.from_dfe(dfe, m, ...)` — sample effect sizes from a DFE
    - Or integrate with stdpopsim DFE objects for realistic effect size distributions
    - Maps to the `selection.py` module in stdpopsim

### 4B. LD-Aware Effect Assignment

19. **Causal variant selection respecting LD**
    - Currently `AdditiveEffects.from_h2()` assigns effects to random variants
    - With real genotype data, should be able to select causal variants with specific MAF/LD properties
    - `AdditiveEffects.from_h2(h2, haplotypes, maf_range=(0.01, 0.5), n_causal=1000)`
    - Needs founder haplotypes as input to compute LD

### 4C. Gene-Environment Interaction (GxE)

20. **GxE component** (stretch goal)
    - New architecture component: `GxEComponent` that multiplies genetic value by an environmental moderator
    - Formula DSL: `gxe(eff, E_moderator)`
    - Relatively easy to add given the component framework

---

## 5. Simulation Analysis & Post-Processing

### 5A. Integration with grapp for Post-Simulation GWAS

21. **grapp GWAS on simulation output**
    - After simulation, export haplotypes to GRG format
    - Run `grapp assoc` (association testing) on simulated data
    - Compare with xftsim's built-in `ngwas.py` GWAS
    - Useful for validating GWAS methods under known architectures

22. **grapp PCA for population structure verification**
    - Run PCA on founder haplotypes via `grapp.linalg.proPCA`
    - Verify population structure matches stdpopsim demographic model
    - Useful for quality control of multi-population simulations

### 5B. Summary Statistics Enhancements

23. **Linkage disequilibrium statistics**
    - LD decay curves, r² between causal and tag variants
    - Important for evaluating PGS performance under realistic LD

24. **Population differentiation (F_ST)**
    - When using multi-population founders, track F_ST across generations
    - Weir-Cockerham or Hudson estimator

---

## 6. Testing & Validation

25. **Integration tests for stdpopsim pipeline**
    - Test `founders_from_stdpopsim("HomSap", "OutOfAfrica_3G09", chromosome="chr22", n_samples=50)`
    - Verify: correct dimensions, valid genotypes (0/1), sensible allele frequencies, variant positions monotonically increasing
    - Test with and without genetic map
    - Test multi-population sampling

26. **Integration tests for msprime pipeline**
    - Test `founders_from_msprime(n=100, sequence_length=1e6, recombination_rate=1e-8)`
    - Verify genotype matrix dimensions, recombination map has correct number of intervals

27. **Numerical validation: coalescent properties**
    - Expected heterozygosity from msprime should match 4*Ne*mu for single-pop models
    - LD decay should match expectations for given recombination rate
    - Multi-pop F_ST should match demographic model predictions

28. **Roundtrip tests: tree sequence → xftsim → simulate → check**
    - Generate founders from msprime
    - Run xftsim simulation for a few generations
    - Verify that allele frequencies drift as expected, phenotype-genotype correlations match architecture

29. **GRG conversion roundtrip**
    - tree_sequence → GRG → GraphHaplotypeOperator → to_dense()
    - Verify genotypes match `ts.genotype_matrix()` exactly

---

## 7. Documentation

30. **Tutorial notebook: "Realistic Human Simulation"**
    - End-to-end example: stdpopsim HomSap → msprime → xftsim → GWAS → PGS evaluation
    - Show how to use HapMap genetic map for recombination
    - Show multi-population setup (e.g., 3-pop Out of Africa model)

31. **Tutorial notebook: "Working with Real Genotype Data"**
    - Load PLINK/VCF data as founders
    - Set up recombination map from genetic map file
    - Run simulation, compute statistics

32. **API docs for new modules**
    - Sphinx RST for nfounders.py (or wherever the new functions live)
    - Update quickstart guide with stdpopsim example

---

## 8. Infrastructure & Quality

33. **Dependency management**
    - stdpopsim, msprime, tskit are optional dependencies (like pygrgl)
    - Lazy imports with helpful error messages: "pip install xftsim[stdpopsim]"
    - Add extras_require to setup.py: `'stdpopsim': ['stdpopsim', 'msprime', 'tskit']`
    - Add extras_require: `'plink': ['bed-reader']` for PLINK I/O

34. **CI matrix expansion**
    - Add optional-dependency test jobs (stdpopsim, GRG)
    - These can be slower / run less frequently

35. **Benchmark expansion**
    - Benchmark tree_sequence_to_haplotypes at various scales
    - Benchmark GRG conversion pipeline
    - Compare GRG vs dense founder storage for large N

---

## Priority Ordering (Suggested)

### Phase A: Foundation (do first)
- **Task 3**: `tree_sequence_to_haplotypes()` — core converter
- **Task 4**: `recombination_map_from_msprime()` — core converter
- **Task 2**: `founders_from_msprime()` — wraps Task 3 + 4
- **Task 9**: Position-aware RecombinationMap enhancements
- **Task 25-26**: Integration tests

### Phase B: stdpopsim Integration
- **Task 1**: `founders_from_stdpopsim()` — wraps Task 2 + stdpopsim catalog
- **Task 5**: `recombination_map_from_stdpopsim()`
- **Task 27-28**: Numerical validation tests
- **Task 30**: Tutorial notebook

### Phase C: GRG Pipeline
- **Task 6**: `tree_sequence_to_grg()` wrapper
- **Task 7**: GRG-aware founders function
- **Task 29**: GRG roundtrip tests

### Phase D: I/O Expansion
- **Task 11-12**: PLINK BED read/write
- **Task 13-14**: VCF read/write
- **Task 31**: Real data tutorial notebook

### Phase E: Stretch Goals
- **Tasks 16-17**: Multi-chromosome support
- **Tasks 18-20**: DFE, LD-aware effects, GxE
- **Tasks 21-24**: grapp integration, LD/F_ST statistics
- **Task 10**: Sex-specific recombination

---

## Architecture Notes

### Where to put new code

```
xftsim/
├── nfounders.py      # NEW: founders_from_stdpopsim, founders_from_msprime,
│                     #      tree_sequence_to_haplotypes
├── founders.py       # EXISTING: keep founder_haplotypes_from_AFs, etc. (simple/synthetic)
├── reproduce.py      # EXISTING: RecombinationMap enhancements go here
├── io.py             # EXISTING: add PLINK/VCF readers, tree_sequence_to_grg wrapper
├── struct.py         # EXISTING: no changes needed (DenseHaplotypeArray, GraphHaplotypeOperator)
├── narch.py          # EXISTING: GxE component if needed
└── ...
```

### Dependency layering

```
                  xftsim core (numpy only)
                 /          |            \
           msprime     stdpopsim      pygrgl
           (optional)  (optional)    (optional)
                \          |
                 tskit (shared)
```

All external deps are optional. Core xftsim works with just numpy. Lazy imports with clear error messages.

### Key tskit → xftsim data flow

```python
# tskit.TreeSequence internals:
ts.num_samples        # haploid sample count (2 * n_individuals for diploid)
ts.num_sites          # number of segregating sites (= n_variants)
ts.genotype_matrix()  # shape (n_sites, n_haploid_samples), dtype int8
ts.individuals()      # individual table (diploid individuals)
ts.sites()            # site table (positions, ancestral alleles)
ts.mutations()        # mutation table (derived alleles, parent mutations)

# Conversion to xftsim:
# 1. geno = ts.genotype_matrix().T  → (n_haploid, n_sites)
# 2. reshape to (n_individuals, n_sites, 2) using ts.individuals
# 3. build VariantMeta from ts.sites (positions, alleles, compute AF)
# 4. build SampleMeta from ts.individuals (IDs, populations)
```

---

## Quick Reference: Library APIs

### stdpopsim
```python
import stdpopsim
species = stdpopsim.get_species("HomSap")
model = species.get_demographic_model("OutOfAfrica_3G09")
contig = species.get_contig("chr22", genetic_map="HapMapII_GRCh37")
engine = stdpopsim.get_default_engine()
ts = engine.simulate(model, contig, {"YRI": 50, "CEU": 50}, seed=42)
```

### msprime
```python
import msprime
ts = msprime.sim_ancestry(100, sequence_length=1e6, recombination_rate=1e-8, random_seed=42)
ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=42)
```

### grgl
```python
import pygrgl
grg = pygrgl.load_immutable_grg("file.grg")
result = pygrgl.matmul(grg, weights, pygrgl.TraversalDirection.DOWN, by_individual=True)
```

### grg convert (CLI)
```bash
grg convert input.trees output.grg          # tree sequence → GRG
grg construct input.vcf.gz -j 4 -o out.grg  # VCF → GRG
```

### Current xftsim (for reference)
```python
from xftsim.founders import founder_haplotypes_uniform_AFs
from xftsim.effect import AdditiveEffects
from xftsim.arch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.mate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.sim import NSimulation
from xftsim.io import load_grg

hap = founder_haplotypes_uniform_AFs(n=1000, m=500)
eff = AdditiveEffects.from_h2(h2=0.5, m=500, seed=42)
arch = Architecture()
arch.add("Y.G", GeneticComponent(eff))
arch.add("Y.E", NoiseComponent(0.5))
arch.add("Y", AggregationComponent("Y.G + Y.E"))
rmap = RecombinationMap.constant_map(m=500, p=0.01)
sim = NSimulation(hap, arch, RandomMating(), rmap, seed=1)
sim.run(10)
```
