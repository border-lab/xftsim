# Dev Environment Setup — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create a fresh venv-based dev environment with pip-compile lock file, dependency tiers, Python version warning, and locally-built docs.

**Architecture:** Single `.venv` created by `scripts/setup-dev.sh`, deps tiered via `setup.py` extras (`[legacy]`, `[docs]`, `[dev]`, `[all]`), versions pinned by `requirements-lock.txt` from pip-compile. Python <3.12 emits a runtime warning on import.

**Tech Stack:** Python 3.12, pip-tools (pip-compile), setuptools, venv, Sphinx

---

### Task 1: Update `setup.py` dependency tiers

**Files:**
- Modify: `setup.py` (lines 23-49)

**Step 1: Edit setup.py**

Replace the `install_requires` and `extras_require` blocks in `setup.py` with:

```python
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "numba>=0.58",
        "xarray",
        "typer>=0.9",
        "rich",
        "pyyaml",
    ],

    extras_require={
        'legacy': [
            "sgkit",
            "nptyping",
            "funcy",
            "networkx",
            "pandas_plink",
        ],
        'grg': [
            "pygrgl",
        ],
        'docs': [
            "sphinx>=7",
            "sphinx-rtd-theme",
            "sphinx-autodoc-typehints",
            "myst-parser",
            "nbsphinx",
            "nbconvert",
            "ipython",
        ],
        'dev': [
            "pytest",
            "pytest-timeout",
            "flake8",
            "pip-tools",
        ],
        'all': [
            "xftsim[legacy,docs,dev]",
        ],
    },
```

**Step 2: Verify syntax**

Run: `python3 -c "exec(open('setup.py').read())"`
Expected: No output (no syntax errors)

**Step 3: Commit**

```bash
git add setup.py
git commit -m "Add docs/all dependency tiers, bump numba>=0.58"
```

---

### Task 2: Add Python version warning to `xftsim/__init__.py`

**Files:**
- Modify: `xftsim/__init__.py` (top of file, after line 1)

**Step 1: Add version check**

Add this block after `import numpy as np` (line 1) and before `__version__`:

```python
import sys
import warnings

if sys.version_info < (3, 12):
    warnings.warn(
        f"xftsim recommends Python >= 3.12 (you have {sys.version_info.major}.{sys.version_info.minor}). "
        "Some features may not work correctly on older versions.",
        stacklevel=2,
    )
```

**Step 2: Verify it parses**

Run: `python3 -c "import ast; ast.parse(open('xftsim/__init__.py').read()); print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add xftsim/__init__.py
git commit -m "Warn when Python < 3.12"
```

---

### Task 3: Create `scripts/setup-dev.sh`

**Files:**
- Create: `scripts/setup-dev.sh`

**Step 1: Create the scripts directory and script**

The script should:
- Accept an optional argument for a specific Python binary (e.g., `python3.12`)
- Auto-detect best Python: try `python3.12`, `python3.11`, `python3.10`, `python3` in order
- Reject anything below 3.10
- Remove existing `.venv` if present
- Create `.venv` via `python -m venv`
- If `requirements-lock.txt` exists, install from it then `pip install --no-deps -e .`
- If no lock file, install via `pip install -e ".[all]"` then generate lock file with `pip freeze --exclude-editable`
- Print activation instructions at the end

**Step 2: Make executable**

Run: `chmod +x scripts/setup-dev.sh`

**Step 3: Verify script parses**

Run: `bash -n scripts/setup-dev.sh`
Expected: No output (no syntax errors)

**Step 4: Commit**

```bash
git add scripts/setup-dev.sh
git commit -m "Add setup-dev.sh convenience script"
```

---

### Task 4: Create the venv and generate lock file

**Step 1: Run the setup script**

Run: `./scripts/setup-dev.sh python3.12`
Expected: Creates `.venv`, installs all deps, generates `requirements-lock.txt`

**Step 2: Verify xftsim imports**

Run: `source .venv/bin/activate && python -c "import xftsim; print(xftsim.__version__)"`
Expected: Prints `0.3.0.devXX` with no errors

**Step 3: Run the test suite**

Run: `source .venv/bin/activate && python -m pytest tests/ --tb=short -q`
Expected: Tests run (some may fail due to pre-existing issues, but pytest itself should work and xftsim should import)

**Step 4: Commit the lock file**

```bash
git add requirements-lock.txt
git commit -m "Add requirements-lock.txt (Python 3.12)"
```

---

### Task 5: Build docs locally

**Files:**
- Modify: `devtools/build_docs.sh` (lines 18-26, update to use .venv)

**Step 1: Update build_docs.sh to prefer .venv**

Replace the sphinx-build detection block (lines 18-26) with logic that:
- First checks for `$REPO_ROOT/.venv/bin/sphinx-build`
- Falls back to the legacy xftsim-test micromamba path
- Falls back to `sphinx-build` from PATH

**Step 2: Build the docs**

Run: `source .venv/bin/activate && ./devtools/build_docs.sh clean`
Expected: Docs build completes, output at `docs/_build/html/index.html`

**Step 3: Commit**

```bash
git add devtools/build_docs.sh
git commit -m "Update build_docs.sh to prefer .venv for Sphinx"
```

---

### Task 6: Update CI to use setup.py extras

**Files:**
- Modify: `.github/workflows/ci.yml` (lines 27-44)

**Step 1: Simplify the install step**

Replace the `Install dependencies` step (lines 27-44) with:

```yaml
      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -e ".[legacy,dev]"
```

**Step 2: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "Simplify CI: use setup.py extras instead of manual dep list"
```

---

### Task 7: Update devtools/claude.md

**Files:**
- Modify: `devtools/claude.md` (lines 309-338)

**Step 1: Update Development Notes section (line 316)**

Replace:
```
- Test environment: `micromamba activate xftsim-test` (Python 3.9, required for numba)
```
with:
```
- Dev environment: `./scripts/setup-dev.sh` creates `.venv` with Python 3.12 (>=3.10 supported)
- Activate: `source .venv/bin/activate`
- Lock file: `requirements-lock.txt` ensures consistent deps across machines
```

**Step 2: Update Development Tools table (around line 320)**

Add row:
```
| `../scripts/setup-dev.sh` | Create fresh `.venv` dev environment with all dependencies |
```

**Step 3: Update Building Documentation section (lines 330-338)**

Replace the note about xftsim-test with:
```
**Note:** The script automatically uses the `.venv` environment (created by `./scripts/setup-dev.sh`). If `.venv` is not available, it falls back to the legacy `xftsim-test` micromamba environment or the current PATH.
```

**Step 4: Commit**

```bash
git add devtools/claude.md
git commit -m "Update dev docs: venv workflow replaces micromamba"
```

---

### Task 8: Update CHANGELOG

**Files:**
- Modify: `devtools/CHANGELOG.dev.md`

**Step 1: Add entry for dev environment changes**

Add under the latest version heading (or create one):

```markdown
## Dev Environment Overhaul (2026-02-20)

- Replaced micromamba `xftsim-test` env with standard Python venv (`.venv`)
- Added `scripts/setup-dev.sh` — auto-detects Python >=3.10, creates venv, installs deps
- Added `requirements-lock.txt` via pip-compile for reproducible environments
- Added dependency tiers in `setup.py`: `[legacy]`, `[docs]`, `[dev]`, `[all]`
- Bumped numba constraint from `>=0.56` to `>=0.58` (Python 3.10+ only)
- Added Python <3.12 runtime warning in `xftsim/__init__.py`
- Updated `devtools/build_docs.sh` to prefer `.venv` over micromamba
- Simplified CI to use `pip install -e ".[legacy,dev]"` instead of manual dep list
```

**Step 2: Commit**

```bash
git add devtools/CHANGELOG.dev.md
git commit -m "Update dev changelog for venv overhaul"
```

---

### Task 9: Verify everything end-to-end

**Step 1: Clean slate test — remove .venv and recreate**

Run `./scripts/setup-dev.sh` from scratch.

**Step 2: Verify import + version warning**

Run: `python -c "import xftsim; print(xftsim.__version__)"`
Expected: Version printed, no warning (since we're on 3.12)

**Step 3: Run tests**

Run: `python -m pytest tests/ --tb=short -q`
Expected: pytest runs, xftsim loads

**Step 4: Build docs**

Run: `./devtools/build_docs.sh clean`
Expected: Docs build to `docs/_build/html/`

**Step 5: Verify lock file reproduces**

Create fresh venv manually from lock file:
```bash
python3.12 -m venv /tmp/xft-test-venv
source /tmp/xft-test-venv/bin/activate
pip install -r requirements-lock.txt
pip install --no-deps -e .
python -c "import xftsim; print(xftsim.__version__)"
```
Expected: Same version, no import errors. Clean up with `rm -rf /tmp/xft-test-venv`.
