% module xftsim

# eXtensible Forward Time SIMulator

`xftsim` simulates complex phenotype/genotype data with an emphasis on short timescale phenomena. `xftsim` is designed with two primary goals:

> - make it easy for statistical geneticists to perform reproducible and systematic sensitivity analyses to better understand limitations and assumptions
> - enable evaulation of methods for analyzing complex traits under realistically complex generative models

<!-- ```{toctree}
:maxdepth: 2

Getting started <gettingstarted/getting_started>
User guide <userguide/user_guide>
Example gallery <examples>
API reference <api>
```
 -->


```{toctree}
:maxdepth: 4
:caption: Getting started

Installation <gettingstarted/install>
Quickstart <gettingstarted/quickstart>
```

```{toctree}
:maxdepth: 4
:caption: User guide

Nuts and bolts <userguide_v0.9/nutsandbolts>
Anatomy of a simulation <userguide_v0.9/simulation>
Founder data <userguide_v0.9/founder>
Recombination maps <userguide_v0.9/rmaps>
Phenogenetic architectures <userguide_v0.9/arch>
Mating regimes <userguide_v0.9/mate>
Statistics <userguide_v0.9/stats>
Data structures <userguide_v0.9/struct>
Indexing <userguide_v0.9/indexing>
Advanced genetic architectures <userguide_v0.9/advgen>
```

```{toctree}
:maxdepth: 3
:caption: Guides (New)

guides/index
```

```{toctree}
:maxdepth: 3
:caption: API Reference (New)

api/index
```

```{toctree}
:maxdepth: 2
:caption: Example Notebooks

Simple Simulation <examples/01_simple_simulation>
Bivariate Assortative Mating <examples/02_bivariate_assortative>
Vertical Transmission <examples/03_vertical_transmission>
GWAS and PGS <examples/04_gwas_pgs>
Checkpoint and Resume <examples/05_checkpoint_resume>
Sibling Effects <examples/06_sibling_effects>
GRG Genotypes <examples/07_grg_genotypes>
```

```{toctree}
:maxdepth: 4
:caption: API reference (Legacy)

Submodule organization <api_ref/submodules>
arch module <api_ref/arch>
effect module <api_ref/effect>
founders module <api_ref/founders>
index module <api_ref/index>
io module <api_ref/io>
mate module <api_ref/mate>
ped module <api_ref/ped>
reproduce module <api_ref/reproduce>
sim module <api_ref/sim>
stats module <api_ref/stats>
struct module <api_ref/struct>
utils module <api_ref/utils>
```
