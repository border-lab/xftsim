# Development Workflow Changelog

Changes to development tooling, CI/CD, testing infrastructure, and documentation systems.

## [Unreleased]

### Adversarial Review Workflow (2026-03-16)

- Added `devtools/math_spec.md` — mathematical specification extracted from manuscript
  - Defines 10 checkable invariants (standardization, h2 round-trip, HE formula, etc.)
  - Source of truth hierarchy: manuscript → math_spec → code → tests
- Added `devtools/adversarial_review.md` — per-commit review protocol
  - Structured checklist covering standardization, effect sizes, estimators, test quality
  - Red flags section for common bug patterns (tautological tests, wide tolerances)
  - Output format template for review reports
- Updated `devtools/claude.md` with adversarial review section and devtools manifest

### Dev Environment Overhaul (2026-02-20)

- Replaced micromamba `xftsim-test` env with standard Python venv (`.venv`)
- Added `scripts/setup-dev.sh` — auto-detects Python >=3.10, creates venv, installs deps
- Added `requirements-lock.txt` via pip-freeze for reproducible environments
- Added dependency tiers in `setup.py`: `[legacy]`, `[docs]`, `[dev]`, `[all]`
- Bumped numba constraint from `>=0.56` to `>=0.58` (Python 3.10+ only)
- Added Python <3.12 runtime warning in `xftsim/__init__.py`
- Updated `devtools/build_docs.sh` to prefer `.venv` over micromamba
- Simplified CI to use `pip install -e ".[legacy,dev]"` instead of manual dep list
- Fixed `pyproject.toml` build-backend (`setuptools.build_meta` replacing invalid `_legacy`)
- Replaced deprecated `pkg_resources` with `importlib.resources` in legacy data module

### Added
- Test suite with pytest (`tests/` directory)
- Demo testing framework (`tests/test_demos.py`) covering UGRM and BGRM demos
- pytest configuration (`pytest.ini`)
- `claude.md` with project documentation, roadmap, and AI assistant instructions
- Version management system (`devtools/bump_version.py`)
  - Pre-commit hook auto-increments dev version on each commit
  - Manual script for release version bumps (patch/minor/major)
- `devtools/install_hooks.sh` for setting up git hooks
- `devtools/build_docs.sh` for building documentation locally
- Split changelogs: `CHANGELOG.md` (software) and `devtools/CHANGELOG.dev.md` (workflow)

### Changed
- Updated `.gitignore` to exclude test virtual environments
- Reorganized development tools into `devtools/` directory
- Symlinked `claude.md` from `devtools/` to project root

### Fixed
- API documentation: removed duplicate automodule directives (warnings reduced from 1551 to 212)
- Removed references to non-existent `lsmate` module
- Removed references to non-existent `extensions` page
- Fixed `source_suffix` in Sphinx config to properly handle .md files
- **Fixed API autodoc**: ReadTheDocs now installs xftsim package; local builds use xftsim-test environment
- Updated `build_docs.sh` to automatically use xftsim-test environment for proper API generation
- Updated `.readthedocs.yaml` to install xftsim package during docs build
