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

Nuts and bolts <userguide/nutsandbolts>
Anatomy of a simulation <userguide/simulation>
Founder data <userguide/founder>
Recombination maps <userguide/rmaps>
Phenogenetic architectures <userguide/arch>
Mating regimes <userguide/mate>
Statistics <userguide/stats>
Post-processing <userguide/proc>
Advanced genetic architectures <userguide/advgen>
```

```{toctree}
:maxdepth: 2
:caption: Guides (New)

Guides <guides/index>
Quick Start <guides/quickstart>
Formula DSL <guides/formula>
```

```{toctree}
:maxdepth: 2
:caption: API Reference (New)

Overview <api/index>
NSimulation <api/nsim>
Architecture & Components <api/narch>
Effects <api/neffect>
Mating <api/nmate>
Filters <api/nfilter>
Statistics <api/nstats>
GWAS & PGS <api/ngwas>
I/O <api/io>
Data Structures <api/struct>
Formula Parser <api/parser>
CLI <api/cli>
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
proc module <api_ref/proc>
reproduce module <api_ref/reproduce>
sim module <api_ref/sim>
stats module <api_ref/stats>
struct module <api_ref/struct>
utils module <api_ref/utils>
```
