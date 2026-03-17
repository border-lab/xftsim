# xftsim - eXtensible Forward Time SIMulator

## Overview

`xftsim` is a Python package for forward-time genetic simulation designed for statistical geneticists. It simulates complex phenotype/genotype data with emphasis on short timescale phenomena.

**Primary Goals:**
- Enable reproducible and systematic sensitivity analyses to understand limitations and assumptions in genetic analysis
- Evaluate statistical methods for analyzing complex traits under realistically complex generative models
- Facilitate development of robust methods that account for complex mating and transmission dynamics

**Project Metadata:**
- **Author:** Richard Border (rborder@cs.cmu.edu) and collaborators
- **Version:** 0.3.0
- **License:** GPL-3.0
- **Python:** 3.9.6+ (tested on MacOS 13.4, Ubuntu 22.04/24.04, PopOS 22.04, RHEL 7)
- **PyPI:** https://pypi.org/project/xftsim/
- **Documentation:** https://xftsim.readthedocs.io
- **GitHub:** https://github.com/rborder/xftsim

---

## Scientific Context

The tool was developed to address a critical gap in statistical genetics: widely-used estimators (LD-score regression, GWAS, heritability estimators) rely on assumptions that are often violated in practice. Key violations include:

### Multivariate Cross-trait Assortative Mating (xAM)
Mate selection is correlated across multiple phenotypes. Empirical analysis shows cross-mate correlation structures are **high-dimensional** - in UK Biobank, 8 canonical variates are needed to explain 90% of cross-mate variance across 34 phenotypes. This is far more complex than bivariate models typically assumed.

### Vertical Transmission (VT)
Parents transmit more than genetic material to offspring - environments, resources, and behaviors are also passed down. This creates gene-environment correlations that bias conventional estimators.

### Key Findings from Manuscript
- **xAM inflates estimates**: Even mild multivariate xAM (r=0.2 across 5 traits) combined with VT (5% of variance) can inflate genetic correlation estimates from 0 to >0.5
- **Sample size exacerbates bias**: Increasing GWAS sample size beyond ~1M can yield more spurious than on-target associations under xAM
- **Within-family designs help but have caveats**: Sibling-difference GWAS mitigates many biases but can still be affected by G×E interactions and sample ascertainment

---

## Library Capabilities

| Feature | Approach |
|---------|----------|
| **Flexible mating regimes** | Arbitrary target cross-mate cross-trait correlation structures across multiple phenotypes or components; extensible via simulation state → mate pairing abstraction |
| **Vertical transmission / direct and indirect effects** | Arbitrary causal relationships between phenotype components within and across generations; extensible via architectural component template class |
| **Additive genetic architectures** | Parametric infinitesimal and non-infinitesimal multivariate additive architectures or manual per-locus effects; extensible via additive-genetic component template class |
| **Non-additive genetic architectures** | Gene-by-phenotype component interactions including G×G or G×E effects; extensible via architectural component template class |
| **Variable population size** | Arbitrary distributions of mates/offspring per individual; extensible via simulation state → realized matings abstraction |
| **Realistic genotype data** | Real haplotype data or synthetic data; simple and map-based recombination; PLINK binary and VCF I/O utilities |
| **Population statistics / estimators** | True heritabilities, correlations, cross-mate correlations; efficient GWAS, genetic (co)variance estimation, cross-validated PGI estimation; extensible via estimator template class |
| **Sample ascertainment** | Arbitrary ascertainment regimes at individual or family level for each estimator |

---

## Key Terminology (Glossary)

| Term | Definition |
|------|------------|
| **Bivariate xAM** | Cross-mate correlations between two traits (e.g., tall people pair with educated partners) |
| **Multivariate xAM** | Cross-mate correlations involving multiple traits simultaneously |
| **High-dimensional xAM** | When multiple independent linear combinations are needed to explain cross-mate similarity |
| **Vertical transmission (VT)** | Environmentally-mediated effects of parent phenotypes on offspring (e.g., wealth inheritance through bequests) |
| **Indirect genetic effects** | When genetic variants "tag" causal factors external to an individual (e.g., parental genotype affecting offspring environment) |
| **Off-target associations** | GWAS hits at variants causal for other traits under xAM, not the focal trait |
| **Panmictic** | Obtained from a randomly mating population |
| **Polygenic index (PGI)** | Sum of allele counts weighted by estimated effect sizes (also: polygenic risk score) |
| **Genetic correlation** | Correlation of genetic effects across traits (can be defined via effects r_β or PGI r_score) |
| **Heritability (h²)** | Ratio of additive genetic variance to total phenotypic variance |
| **Haseman-Elston estimator** | Method-of-moments estimator for heritability/genetic correlation (approximately equivalent to LD-score regression) |

---

## Repository Structure

```
xftsim/
├── xftsim/                 # Main package source code
│   ├── __init__.py         # Package init, Config class, __version__
│   ├── maps/               # Genetic recombination maps (CEU hg19)
│   └── *.py                # Core modules (16 files)
├── tests/                  # Test suite
│   ├── conftest.py         # pytest fixtures
│   └── test_demos.py       # Demo simulation tests
├── devtools/               # Development tools and documentation
│   ├── claude.md           # AI assistant instructions (symlinked to root)
│   ├── bump_version.py     # Version management script
│   ├── install_hooks.sh    # Git hooks installer
│   └── CHANGELOG.dev.md    # Dev workflow changelog
├── docs/                   # Sphinx documentation
│   ├── gettingstarted/     # Installation and quickstart
│   ├── userguide/          # Comprehensive tutorials (11 Jupyter notebooks)
│   ├── api_ref/            # Auto-generated API reference
│   └── _static/            # Static assets (logos, CSS)
├── claude.md -> devtools/claude.md  # Symlink to AI instructions
├── CHANGELOG.md            # Software changelog
├── pytest.ini              # pytest configuration
├── setup.py                # Package installation
├── environment.yml         # Conda environment
├── README.md               # Project README
└── LICENSE                 # GPL-3.0 license
```

---

## Source Code Manifest

### Package Modules (`xftsim/`)

| Module | Description |
|--------|-------------|
| `__init__.py` | Package initialization, `Config` class for global settings (threads, verbosity), `__version__` |
| `sim.py` | **Core simulation orchestrator** - `Simulation` class that runs forward-time simulations |
| `arch.py` | **Phenogenetic architectures** - Defines how phenotypes are generated from genetic/environmental components (`Architecture`, `AdditiveGeneticComponent`, `DominanceComponent`, `GxEComponent`, etc.) |
| `struct.py` | **Data structures** - `HaplotypeArray`, `PhenotypeArray`, `GeneticMap` for managing genetic/phenotypic data with xarray |
| `index.py` | **Indexing system** - `XftIndex`, `SampleIndex`, `VariantIndex`, `ComponentIndex` for managing data dimensions |
| `mate.py` | **Mating regimes** - `MateAssignment`, `RandomMatingRegime`, `LinearAssortativeMatingRegime`, `KAssortativeMatingRegime`, `BatchedMatingRegime` |
| `reproduce.py` | **Reproduction mechanics** - `RecombinationMap`, `Meiosis`, `VerticalTransmission` for sexual reproduction and phenotypic transmission |
| `effect.py` | **Genetic effects** - `AdditiveEffects` class with standardization/scaling options for effect sizes |
| `stats.py` | **Statistical estimators** - `Statistic` base class, `SampleStatistics`, `MatingStatistics`, `HasemanElstonEstimator`, GWAS, heritability estimation |
| `filters.py` | **Sample filtering** - `SampleFilter`, `PassFilter`, `AscertainmentFilter` for statistical estimators |
| `founders.py` | **Founder generation** - Functions to generate founder haplotypes from allele frequencies or real data |
| `ped.py` | **Pedigree representation** - `Pedigree` class using NetworkX directed graphs |
| `proc.py` | **Post-processors** - `PostProcessor`, `LimitMemory`, `WriteToDisk` for managing simulation output |
| `io.py` | **Input/output** - PLINK, VCF, Zarr format support, genotype-to-haplotype conversion |
| `utils.py` | **Utilities** - Profiling decorator, indexing helpers, data manipulation functions |
| `data.py` | **Data loading** - `get_ceu_map()` for loading CEU haplotype maps |

### Data Files (`xftsim/maps/`)

| File | Description |
|------|-------------|
| `ceu.hg19.map` | CEU population genetic recombination map (hg19 reference) |

---

## Documentation Manifest

### Getting Started (`docs/gettingstarted/`)

| File | Description |
|------|-------------|
| `getting_started.md` | Index for getting started section |
| `install.md` | Installation instructions |
| `quickstart.ipynb` | Quickstart tutorial notebook |

### User Guide (`docs/userguide/`)

| File | Description |
|------|-------------|
| `user_guide.md` | Index for user guide section |
| `nutsandbolts.md` | Core concepts and fundamentals |
| `simulation.ipynb` | Anatomy of simulations |
| `founder.ipynb` | Founder data handling |
| `rmaps.ipynb` | Recombination maps |
| `arch.ipynb` | Phenogenetic architectures |
| `mate.ipynb` | Mating regimes |
| `stats.ipynb` | Statistics and estimators |
| `proc.ipynb` | Post-processing |
| `advgen.ipynb` | Advanced genetic architectures |
| `indexing.ipynb` | Indexing system |
| `struct.ipynb` | Data structures |
| `submodules.md` | Submodules reference |

### API Reference (`docs/api_ref/`)

| File | Description |
|------|-------------|
| `index.md` | API reference index |
| `arch.md` | Architecture module API |
| `effect.md` | Effect module API |
| `founders.md` | Founders module API |
| `io.md` | I/O module API |
| `mate.md` | Mating module API |
| `ped.md` | Pedigree module API |
| `proc.md` | Post-processor module API |
| `reproduce.md` | Reproduction module API |
| `sim.md` | Simulation module API |
| `stats.md` | Statistics module API |
| `struct.md` | Data structures module API |
| `utils.md` | Utilities module API |
| `submodules.md` | Submodules overview |

### Other Documentation

| File | Description |
|------|-------------|
| `docs/index.md` | Main documentation landing page |
| `docs/api.md` | API documentation index |
| `docs/modules.md` | Module overview |
| `docs/examples.md` | Example gallery |
| `docs/xftsim.md` | Package overview |
| `docs/conf.py` | Sphinx configuration |
| `docs/requirements.txt` | Documentation build dependencies |

---

## Configuration Files

| File | Description |
|------|-------------|
| `setup.py` | Package installation configuration (setuptools) |
| `environment.yml` | Conda environment specification |
| `.readthedocs.yaml` | ReadTheDocs build configuration |
| `.gitignore` | Git ignore patterns |
| `xftsim/meta.yaml` | Conda package build recipe |
| `docs/Makefile` | Sphinx documentation build automation |
| `docs/make.bat` | Windows Sphinx build script |

---

## Core Dependencies

- `numpy`, `pandas`, `scipy` - Numerical computing
- `networkx` - Pedigree graphs
- `numba` (0.56.4) - JIT compilation for performance
- `xarray` - Multi-dimensional labeled arrays
- `pandas_plink` - PLINK format I/O
- `sgkit` - Genomic toolkit
- `funcy` - Functional utilities
- `nptyping` - Type hints

**Optional:** `pygraphviz` for automatic causal diagram generation

---

## Simulation Methodology

### Unidimensional xAM (Linear Sorting)
For exchangeable cross-mate correlations, males and females are independently ordered on a linear combination of phenotypes plus Gaussian noise and paired. Used for simulations with exchangeable correlation structures.

### High-dimensional xAM (Quadratic Assignment Problem)
For arbitrary cross-mate correlation structures, the problem is formulated as finding a permutation P* minimizing ||Ỹ'PY - Ω̂||²_F. This is equivalent to the Quadratic Assignment Problem (NP-hard), solved approximately using Hexaly Optimizer. Achieves target correlations to third decimal place.

### Phenotypic Generative Model
```
X := (X[1],...,X[K]) ← meiosis(X*, X**)     # Offspring haplotypes from parents
G_k ← X[k]β_k for k = 1,...,K                # Additive genetic component
ε_k ~ N(0, σ²_ε)                             # Individual-specific noise
T_k ← √(θ/(2K)) Σ(Y*_k + Y**_k)              # Vertical transmission
E_k ← T_k + ε_k                              # Environmental component
Y_k ← G_k + E_k + √(φ/(σ²_g(σ²_e-θ)))(G_k∘E_k)  # With optional G×E
```

---

## Key Classes and Usage Patterns

### Running a Simulation

```python
import xftsim as xft

# Create founder haplotypes
founders = xft.founders.founder_haplotypes_uniform_AFs(n=8000, m=4000)

# Define genetic architecture
arch = xft.arch.Architecture([
    xft.arch.AdditiveGeneticComponent(beta=effects),
    xft.arch.AdditiveNoiseComponent(variances=[0.4]),
    xft.arch.SumComponent(phenotypes, sum_components=['additiveGenetic', 'additiveNoise'])
])

# Define mating regime
mating = xft.mate.LinearAssortativeMatingRegime(r=0.3, ...)

# Define recombination
rmap = xft.reproduce.RecombinationMap(p=0.25, vid=founders.vid, chrom=founders.chrom)

# Create and run simulation
sim = xft.sim.Simulation(
    founder_haplotypes=founders,
    mating_regime=mating,
    recombination_map=rmap,
    architecture=arch,
    statistics=[xft.stats.SampleStatistics(), xft.stats.HasemanElstonEstimator()],
    post_processors=[xft.proc.LimitMemory(n_haplotype_generations=2)]
)
sim.run(n_generations=10)
```

### Quick Demo

```python
import xftsim as xft

demo = xft.sim.DemoSimulation('BGRM')
demo.run(3)
xft.utils.print_tree(demo.results)
```

---

## Architecture Patterns

The codebase follows a modular, extensible architecture:

1. **Indexing Layer** (`index.py`): Manages sample, variant, and component dimensions
2. **Data Layer** (`struct.py`): xarray-backed data structures for haplotypes and phenotypes
3. **Genetic Effects** (`effect.py`): Effect size management with standardization options
4. **Architecture Layer** (`arch.py`): Composable phenotype generation components
5. **Mating Layer** (`mate.py`): Pluggable mating regime implementations
6. **Reproduction Layer** (`reproduce.py`): Recombination and transmission mechanics
7. **Statistics Layer** (`stats.py`): Extensible estimator framework
8. **Post-processing** (`proc.py`): Memory management and output handling
9. **Simulation Orchestration** (`sim.py`): Ties everything together

---

## Development Notes

- Test suite in `tests/` using pytest (run with `pytest tests/ -v`)
- Some module docstrings are incomplete (noted in `todo.org`)
- Performance-critical code uses Numba JIT compilation
- Type hints are used throughout via `nptyping`
- High-dimensional xAM simulations require Hexaly Optimizer (free academic license)
- Dev environment: `./scripts/setup-dev.sh` creates `.venv` with Python 3.12 (>=3.10 supported)
- Activate: `source .venv/bin/activate`
- Lock file: `requirements-lock.txt` ensures consistent deps across machines

---

## Development Tools (`devtools/`)

| File | Purpose |
|------|---------|
| `claude.md` | This file - AI assistant instructions and project documentation (symlinked to repo root) |
| `math_spec.md` | Mathematical specification — invariants that code must preserve |
| `adversarial_review.md` | Per-commit adversarial review protocol and checklist |
| `bump_version.py` | Version management script. Usage: `python devtools/bump_version.py [dev|patch|minor|major|show]` |
| `install_hooks.sh` | Installs git pre-commit hook for automatic dev version bumping |
| `build_docs.sh` | Build documentation (prefers `.venv`, falls back to xftsim-test) |
| `CHANGELOG.dev.md` | Changelog for development workflow changes (testing, CI/CD, tooling) |
| `../scripts/setup-dev.sh` | Create fresh `.venv` dev environment with all dependencies |

### Building Documentation

```bash
./devtools/build_docs.sh          # Build docs
./devtools/build_docs.sh clean    # Clean and rebuild
./devtools/build_docs.sh serve    # Build and serve locally at http://localhost:8000
```

**Note:** The script automatically uses the `.venv` environment (created by `./scripts/setup-dev.sh`). If `.venv` is not available, it falls back to the legacy `xftsim-test` micromamba environment or the current PATH.

### Version Management

The pre-commit hook automatically increments the dev version on each commit:
- `0.2.0` → `0.2.0.dev1` → `0.2.0.dev2` → ...

For releases, use the bump script:
```bash
python devtools/bump_version.py patch   # 0.2.0.dev5 → 0.2.1
python devtools/bump_version.py minor   # 0.2.1 → 0.3.0
python devtools/bump_version.py major   # 0.3.0 → 1.0.0
```

To skip auto-versioning for a commit: `git commit --no-verify`

### Changelogs

- **`CHANGELOG.md`** - Software changes (features, bug fixes, API changes)
- **`devtools/CHANGELOG.dev.md`** - Development workflow changes (testing, CI/CD, tooling)

---

## Instructions for AI Assistants

- **NEVER add "Co-Authored-By: Claude" or any similar AI attribution to commits, code, or documentation**
- Do not add any co-author, contributor, or attribution lines referencing AI/Claude/LLM assistance
- **Update changelogs when making changes:**
  - Software changes → `CHANGELOG.md`
  - Dev workflow changes → `devtools/CHANGELOG.dev.md`
- Use semantic versioning: `X.Y.Z` for releases, `X.Y.Z.devN` for development versions
- The pre-commit hook auto-bumps dev version; no manual version changes needed for regular commits

---

## Adversarial Review Workflow

**Mathematical specification:** `devtools/math_spec.md`
**Review protocol:** `devtools/adversarial_review.md`

The manuscript (`current_manu_draft.md`) is the source of mathematical truth.
`devtools/math_spec.md` extracts the key invariants into a checkable format.

### When to review

Run adversarial review on every commit that touches:
- `struct.py` (standardization, matvec, haplotype operations)
- `nstats.py` / `stats.py` (HE estimator, GWAS, statistics)
- `neffect.py` / `effect.py` (effect sizes, h2 targeting)
- `narch.py` / `arch.py` (phenotype generation, VT)
- `tests/numerical/` (numerical validation tests)

### How to run

Ask Claude Code to review against the spec:
```
Review the last commit against devtools/math_spec.md adversarially.
```

### Key invariants (summary)

1. `standardized_matvec` = `((G - 2p) / sqrt(2pq)) @ v` (NOT just centering)
2. `from_h2(h2)` draws beta ~ N(0, h2/m) matched to standardized genotypes
3. h2 round-trip: design h2 ≈ realized h2 at generation 0
4. HE estimator uses GRM formula `cov_g = Y'(KY-Y) / (tr(K²)-n)`, NOT sibling-ICC
5. Tests must compute expected values from the SPEC, not from the code under test

---

## Roadmap / TODO

### High Priority - Testing & CI/CD

- [x] **Demo testing system**: Create automated tests that run all demos and verify expected outputs
- [ ] **Package test suite**: Implement comprehensive pytest-based testing for all modules
- [x] **Version management**: Implement semantic versioning with `.dev` tags for code in flux
  - Format: `X.Y.Z` for stable releases, `X.Y.Z.devN` for development
  - Pre-commit hook auto-increments dev version
  - `devtools/bump_version.py` for release bumps (patch/minor/major)
- [ ] **Automated PyPI publishing**: Set up GitHub Actions for automatic PyPI releases on tags
- [ ] **Automated conda-forge publishing**: Configure feedstock for conda-forge distribution

### High Priority - Installation & Dependencies

- [ ] **Simplify installation**: Make it easier to install in different environments (conda, pip, from source)
- [ ] **Simplify dependencies**: Audit and reduce listed dependencies to essential packages only
- [ ] **Pin versions appropriately**: Balance reproducibility with flexibility

### High Priority - Documentation

- [x] **Fix API documentation**: Fixed - ReadTheDocs now installs xftsim, local builds use xftsim-test env
- [ ] **Consistent docstrings**: Standardize docstring format across all modules (NumPy style)
- [ ] **Remove unneeded docstrings**: Clean up placeholder or redundant documentation
- [ ] **Verify docs compile**: Ensure `make html` succeeds without errors/warnings

### Medium Priority - Project Management

- [ ] **Automated changelog**: Set up automatic changelog generation from commits/PRs
- [x] **CHANGELOG.md**: Create and maintain a changelog file

### Long Term - Architecture (BLOCKED)

- [ ] **Move away from xarray**: Replace xarray with alternative data structures
  - **Owner:** Ajay (dev)
  - **Status:** Do not start until Ajay requests assistance
  - **Rationale:** TBD by Ajay

---

## Data Format Support

- **Input:** PLINK binary (bfiles), VCF, Zarr, sgkit datasets
- **Output:** Zarr, HDF5-compatible formats
- **Genetic Maps:** CEU (hg19) included; pyrho map support

---

## Related Resources

- **Code Supplement:** https://github.com/border-lab/xftmanu_code_supplement (reproduction code for manuscript)
- **Hexaly Optimizer:** https://www.hexaly.com/ (for high-dimensional xAM; free academic licenses)
