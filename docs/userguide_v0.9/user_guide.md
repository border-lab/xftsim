% module xftsim

# User guide (v0.9)

This is the user guide for the v0.9 (`ajay`/`v0.9alpha`) refactor of
`xftsim`. The big-picture changes from the legacy interface are:

- **Architectures are now built from a lavaan-style formula string** parsed
  by `xftsim.parser`. The legacy `ArchitectureComponent` class hierarchy
  (`AdditiveGeneticComponent`, `LinearTransformationComponent`,
  `LinearVerticalComponent`, `ProductComponent`,
  `SumAllTransformation`, …) has been replaced by a small set of DSL
  primitives (`genetic`, `noise`, `cnoise`, `threshold`, `mother`,
  `father`, `parent`, `sibling_*`) plus free-form arithmetic
  aggregation expressions.
- **Data structures are now plain numpy under the hood.**
  `HaplotypeArray` has been split into `DenseHaplotypeArray`
  (numpy-backed) and `GraphHaplotypeOperator` (GRG-backed), both
  implementing a common `HaplotypeOperator` ABC.
  `PhenotypeArray` is now a thin dict-of-named-arrays wrapper instead
  of an xarray accessor.
- **Mating regimes have new short names**: `RandomMating`,
  `LinearAssortativeMating`, `GeneralAssortativeMating`,
  `BatchedMating`. Component selection uses `component_names=[...]`
  instead of a `ComponentIndex`.
- **`xftsim.proc.LimitMemory` is gone.** Memory retention is now
  controlled directly on `Simulation` via the `retain_haplotypes` and
  `retain_phenotypes` kwargs.
- **Results live in `sim.results`** (a list of `GenerationResult`
  dataclasses), not `sim.results_store`.

```{toctree}
:maxdepth: 4

Nuts and bolts <nutsandbolts>
Anatomy of a simulation <simulation>
Founder data <founder>
Mating regimes <mate>
Recombination maps <rmaps>
Phenogenetic architectures <arch>
Statistics <stats>
Advanced genetic architectures <advgen>
```
