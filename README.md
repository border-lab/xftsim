<img src="./xftsimlogo.svg" width="20%">

# xftsim -- Forward-Time Genetic Simulation Framework

`xftsim` simulates complex phenotype/genotype data with an emphasis on
short-timescale phenomena relevant to statistical genetics. It provides a
formula DSL for defining phenogenetic architectures, efficient
numpy-backed data structures, and a modular simulation loop.

## Key Features

- **Formula DSL** -- define genetic architectures in a concise, lavaan-style syntax
- **Modular architecture** -- plug-and-play components (genetic, noise, aggregation, vertical transmission, sibling effects)
- **Assortative mating** -- rank-order pairing on phenotypic composites with configurable spousal correlation
- **Vertical transmission** -- parental phenotype effects via `parent()`, `mother()`, `father()` with founder fallbacks
- **Sibling effects** -- `sibling_mean()`, `sibling_sum()`, and other within-family aggregations
- **GRG support** -- load graph-based genotype representations via `load_grg()`
- **GWAS and PGS** -- per-variant association testing and polygenic score computation from simulation output
- **CLI** -- command-line interface for running simulations, resuming from checkpoints, and demos
- **Checkpointing** -- save/restore full simulation state to disk
- **3200+ tests** -- comprehensive unit, integration, and numerical test suite

## Installation

```bash
pip install xftsim
```

## Development Setup

Requires Python >= 3.10 (3.12 recommended).

```bash
git clone https://github.com/rborder/xftsim
cd xftsim

# Option A: convenience script (auto-detects Python, creates .venv, installs everything)
./scripts/setup-dev.sh

# Option B: manual
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt    # pinned deps for reproducibility
pip install --no-deps -e .              # editable install
```

Activate the environment:

```bash
source .venv/bin/activate
```

### Dependency Tiers

| Extra | Contents | Install with |
|-------|----------|-------------|
| *(core)* | numpy, scipy, pandas, numba, xarray, typer, rich, pyyaml | `pip install -e .` |
| `[legacy]` | sgkit, nptyping, funcy, networkx, pandas_plink | `pip install -e ".[legacy]"` |
| `[docs]` | sphinx, myst-parser, nbsphinx, ipython | `pip install -e ".[docs]"` |
| `[dev]` | pytest, flake8, pip-tools | `pip install -e ".[dev]"` |
| `[all]` | everything above | `pip install -e ".[all]"` |

### Updating Dependencies

```bash
# After changing setup.py constraints:
pip install -e ".[all]"
pip freeze --exclude-editable > requirements-lock.txt
git add requirements-lock.txt && git commit -m "Update dependency lock"
```

### Building Docs

```bash
./devtools/build_docs.sh          # build
./devtools/build_docs.sh clean    # clean + rebuild
./devtools/build_docs.sh serve    # build + serve at http://localhost:8000
```

## Quick Start

A minimal simulation: 1000 individuals, 200 variants, h2 = 0.5, random mating,
5 generations.

```python
import numpy as np
from xftsim.neffect import AdditiveEffects
from xftsim.narch import Architecture
from xftsim.nmate import RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.nsim import NSimulation
from xftsim.nstats import SampleStatistics
from xftsim.founders import founder_haplotypes_uniform_AFs

# 1. Founder haplotypes (n=1000 individuals, m=200 variants)
hap = founder_haplotypes_uniform_AFs(n=1000, m=200)

# 2. Additive effects targeting h2=0.5
eff = AdditiveEffects.from_h2(h2=0.5, m=200, seed=42)

# 3. Architecture via formula DSL (one component per line)
arch = Architecture(
    formula="""
    Y.G ~ genetic(eff)
    Y.E ~ noise(0.5)
    Y ~ Y.G + Y.E
    """,
    effects={'eff': eff},
)

# 4. Mating and recombination
mating = RandomMating(offspring_per_pair=2)
recomb = RecombinationMap.constant_map(m=200, p=0.5)

# 5. Run simulation
sim = NSimulation(
    founder_haplotypes=hap,
    architecture=arch,
    mating_regime=mating,
    recombination_map=recomb,
    statistics=[SampleStatistics()],
    seed=42,
)
sim.run(n_generations=5)

# 6. Access results
pheno = sim.phenotypes                          # current generation's phenotypes
print(list(pheno.keys))                         # ['Y.G', 'Y.E', 'Y']
print(f"Var(Y) = {np.var(pheno['Y']):.3f}")

for r in sim.results:
    stats = r.statistics['SampleStatistics']
    idx = stats['keys'].index('Y')
    print(f"Gen {r.generation}: Var(Y) = {stats['var'][idx]:.3f}")
```

## Formula DSL

The architecture is defined with a multi-line formula string where each line
specifies one component. The parser expects **exactly one component per line**.

### Syntax

```
LHS ~ function(args)
LHS ~ function(args) | GROUPING
LHS ~ arithmetic_expression
```

### Components

| Function | Description | Example |
|----------|-------------|---------|
| `genetic(eff)` | Additive genetic value (genotypes x effects) | `Y.G ~ genetic(eff)` |
| `mvGenetic(eff)` | Multivariate genetic value (k traits) | `(h.G, b.G) ~ mvGenetic(eff)` |
| `haplotypeGenetic(eff)` | Single-haplotype genetic value | `Y.mat ~ haplotypeGenetic(eff, haplotype='maternal')` |
| `noise(var)` | Independent N(0, var) noise | `Y.E ~ noise(0.5)` |
| `cnoise(cov=...)` | Correlated multivariate noise | `(h.E, b.E) ~ cnoise(cov=[[0.3,0.1],[0.1,0.4]])` |
| `parent(pheno)` | Midparent vertical transmission | `Y.VT ~ parent(Y, founder=noise(0.2))` |
| `mother(pheno)` | Maternal vertical transmission | `Y.mat ~ mother(Y, founder=noise(0.1))` |
| `father(pheno)` | Paternal vertical transmission | `Y.pat ~ father(Y, founder=noise(0.1))` |
| `sibling_mean(pheno)` | Mean of sibling phenotypes | `Y.sib ~ sibling_mean(Y)` |
| arithmetic | Sum, difference, product of components | `Y ~ Y.G + Y.E + Y.VT` |

### Grouping

The `|` operator specifies grouping for noise components. Grouped noise draws
one shared value per group (e.g., per family) and broadcasts to all members:

```
Y.shared ~ noise(0.3) | FID
```

Available grouping variables: `FID`, `sex`, `mother`, `father`, or custom
fields on `SampleMeta.extra`.

### Multivariate Example

```python
from xftsim.neffect import MultivariateEffects

eff = MultivariateEffects.from_h2_rg(h2=[0.6, 0.4], rg=0.3, m=200, seed=1)

arch = Architecture(
    formula="""
    (height.G, bmi.G) ~ mvGenetic(eff)
    height.E ~ noise(0.4)
    bmi.E ~ noise(0.6)
    height ~ height.G + height.E
    bmi ~ bmi.G + bmi.E
    """,
    effects={'eff': eff},
)
```

### Vertical Transmission Example

The `founder=` keyword provides a fallback component for generation 0
(when no parental phenotypes exist):

```python
arch = Architecture(
    formula="""
    Y.G ~ genetic(eff)
    Y.VT ~ parent(Y, founder=noise(0.2))
    Y.E ~ noise(0.3)
    Y ~ Y.G + Y.VT + Y.E
    """,
    effects={'eff': eff},
)
```

## Effect Specifications

| Class | Description | Factory Methods |
|-------|-------------|-----------------|
| `AdditiveEffects` | Univariate, all variants causal | `.from_h2(h2, m)`, `.from_array(arr)` |
| `MultivariateEffects` | k-trait correlated effects | `.from_h2_rg(h2, rg, m)`, `.from_covg(covg, m)`, `.from_array(arr)` |
| `SparseEffects` | Subset of variants causal | `.from_h2(h2, m, k_causal)` |

## Mating Regimes

| Class | Description |
|-------|-------------|
| `RandomMating(offspring_per_pair=2)` | Random pairing by sex |
| `LinearAssortativeMating(component_names, r, offspring_per_pair=2)` | Rank-order assortative mating on phenotypic composite |

## CLI

```bash
xftsim run config.yaml          # run a simulation from YAML config
xftsim resume checkpoint_dir/   # resume from checkpoint
xftsim info checkpoint_dir/     # inspect checkpoint metadata
xftsim demo UGRM                # run a built-in demo simulation
```

## Module Overview

| Module | Description |
|--------|-------------|
| `nsim` | `NSimulation` -- forward-time simulation loop |
| `narch` | `Architecture`, `ArchNode`, component classes |
| `neffect` | `EffectSpec` hierarchy (additive, multivariate, sparse) |
| `parser` | Formula DSL parser |
| `nmate` | Mate assignment (`RandomMating`, `LinearAssortativeMating`) |
| `nfilter` | Filters (`TrioFilter`, `SibPairFilter`) for structured views |
| `nstats` | Per-generation statistics (`SampleStatistics`) |
| `ngwas` | GWAS and PGS computation |
| `struct` | Core data structures (`DenseHaplotypeArray`, `NPhenotypeArray`, `SampleMeta`, `PedigreeArray`) |
| `founders` | Founder haplotype generation |
| `reproduce` | Meiosis and `RecombinationMap` |
| `io` | Save/load haplotypes, phenotypes, effects, architectures, checkpoints, GRG |
| `cli` | Command-line interface |

## Gotchas

- **Formula parser**: one component per line. Do NOT write `genetic(eff) + noise(0.5)` on a single line.
- **`filters` parameter**: `NSimulation(filters=...)` takes a `dict[str, Filter]`, not a list. Example: `filters={'trio': TrioFilter()}`.
- **`TrioFilter()` and `SampleStatistics()`**: take no constructor arguments.
- **`pheno.keys`**: this is a property, not a method -- use `pheno.keys` not `pheno.keys()`.
- **Integer indexing**: `hap[0]` produces a 2D array that breaks `subset()` -- use `hap[[0]]` for a single individual.
- **Retention policy**: `retain_haplotypes=1` means only the most recent generation is kept; older ones are pruned after each generation.
- **Validation timing**: `_validate()` runs at `sim.run()` time, not in `__init__` -- dimension mismatches raise at run time.

## Examples

See the example notebooks in [`docs/examples/`](docs/examples/):

- `01_simple_simulation.ipynb` -- univariate trait with random mating
- `02_bivariate_assortative.ipynb` -- bivariate traits with assortative mating
- `03_vertical_transmission.ipynb` -- vertical transmission and sibling effects

## Testing

```bash
source .venv/bin/activate
pytest
```

The test suite includes 3400+ unit, integration, and numerical tests.

## License

GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
