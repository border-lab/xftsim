# Dev Environment Redesign

**Date:** 2026-02-20
**Status:** Approved

## Problem

The existing dev environment (`xftsim-test` via micromamba, Python 3.9) is broken:
- Missing core deps (`typer`, `rich`) — `import xftsim` fails
- Python 3.9 violates `setup.py`'s `python_requires='>=3.10'`
- Editable install points to stale path
- CI works around `setup.py` issues with manual dep installation
- No reproducible lock file — environment drifts across machines

## Design

### Dependency tiers in `setup.py`

| Tier | Extras key | Packages |
|------|-----------|----------|
| core | *(base `install_requires`)* | numpy, scipy, pandas, numba>=0.58, xarray, typer>=0.9, rich, pyyaml |
| legacy | `[legacy]` | sgkit, nptyping, funcy, networkx, pandas_plink |
| grg | `[grg]` | pygrgl |
| docs | `[docs]` | sphinx, sphinx-rtd-theme, myst-parser, nbsphinx, ipython, nbconvert |
| dev | `[dev]` | pytest, pytest-timeout, flake8, pip-tools |
| all | `[all]` | union of all above |

### What gets tracked in git

| File | Purpose |
|------|---------|
| `setup.py` | Loose version constraints, single source of truth for tiers |
| `requirements-lock.txt` | Pinned versions via `pip-compile` for cross-machine reproducibility |
| `scripts/setup-dev.sh` | Finds Python >=3.10, creates `.venv`, installs everything |

### Developer workflow

```bash
# First time or after clone:
./scripts/setup-dev.sh

# Or manually:
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
pip install --no-deps -e .

# Updating deps:
source .venv/bin/activate
pip-compile setup.py --extra all -o requirements-lock.txt
git add requirements-lock.txt && git commit -m "Update dependency lock"
```

### `scripts/setup-dev.sh`

- Finds best available Python (prefers 3.12 > 3.11 > 3.10)
- Creates `.venv` in project root
- Installs from lock file, then editable install with `--no-deps`
- Prints activation instructions

### CI changes

Simplify `.github/workflows/ci.yml` to use `pip install -e ".[dev]"` instead of manual dep list hack.

### Documentation changes

- Update `devtools/claude.md` to reference venv workflow instead of micromamba
- Build dev docs locally to verify Sphinx pipeline works with new env

### Version constraint changes

- Bump numba from `>=0.56` to `>=0.58` (aligns with `python_requires>=3.10`)

### What stays gitignored

- `.venv/` — contains platform-specific binaries, not portable
