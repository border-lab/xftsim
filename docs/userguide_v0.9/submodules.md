## Submodule organization

`xftsim` is organized into the following submodules. The DSL parser
(`parser`), the filter system (`filters`) and the GWAS / PGS module
(`gwas`) are new in v0.9; the post-processor module (`proc`) has been
absorbed into `sim.Simulation` via the `retain_haplotypes` /
`retain_phenotypes` constructor kwargs.

| submodule | contents | tutorials |
|-----------|----------|-----------|
| `arch`      | Formula-driven phenogenetic architectures (`Architecture`, `ArchComponent`, `ArchNode`, the `genetic`/`noise`/`cnoise`/`threshold`/`mother`/`father`/`parent`/`sibling_*` builtins, and the `AggregationComponent` expression evaluator) | [arch](./arch.ipynb), [advgen](./advgen.ipynb) |
| `effect`    | Genetic effect specifications (`EffectSpec`, `AdditiveEffects`, `MultivariateEffects`, `SparseEffects`) | [arch](./arch.ipynb), [advgen](./advgen.ipynb) |
| `filters`   | Sample filters and views (`TrioFilter`/`TrioView`, `SibPairFilter`/`SibPairView`, `UnrelatedFilter`, `AscertainmentFilter`, `SubsampleFilter`) | [stats](./stats.ipynb) |
| `founders`  | Creating and importing founder haplotypes (`founder_haplotypes_uniform_AFs`, `founder_haplotypes_from_AFs`, `founder_haplotypes_from_plink_bfile`, `founder_haplotypes_from_sgkit_dataset`, `founder_haplotypes_from_msprime_grg`) | [founder](./founder.ipynb) |
| `gwas`      | GWAS sumstats, polygenic indices, and related estimators | [stats](./stats.ipynb) |
| `index`     | Indices for working with individual-, variant-, and component-level data (`SampleIndex`, `DiploidVariantIndex`, `HaploidVariantIndex`, `ComponentIndex`) | [indexing](./indexing.ipynb) |
| `io`        | Reading and writing data: PLINK, sgkit / VCF, GRG, npz, and full simulation checkpoints | [founder](./founder.ipynb) |
| `mate`      | Mating regimes (`RandomMating`, `LinearAssortativeMating`, `GeneralAssortativeMating`, `BatchedMating`) and the `MateAssignment` dataclass | [mate](./mate.ipynb) |
| `parser`    | Formula-DSL parser that turns a multi-line string into `ArchNode`s for `arch.Architecture` |  |
| `ped`       | Pedigree data structures |  |
| `reproduce` | Recombination maps and meiosis (`RecombinationMap`, `meiosis`) | [rmaps](./rmaps.ipynb) |
| `sim`       | Simulation class for setting up and running experiments | [simulation](./simulation.ipynb) |
| `stats`     | Per-generation statistics (`SampleStatistics`, `MatingStatistics`, `HasemanElstonEstimator`, `ParentOffspringRegression`) and the `GenerationResult` container | [stats](./stats.ipynb) |
| `struct`    | Data structures: `SampleMeta`, `VariantMeta`, `DenseHaplotypeArray`, `GraphHaplotypeOperator`, `StandardizedHaplotypeOperator`, `PhenotypeArray`, `PedigreeArray`, `GeneticMap` | [struct](./struct.ipynb) |
| `utils`     | Utility functions and profiling helpers |  |
