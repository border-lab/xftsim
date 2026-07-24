"""Integration test for docs/examples/07_grg_genotypes.ipynb.

The notebook needs both the ``pygrgl`` package and the glink fixture
files under ``~/Dropbox/grg/glink/tests/fixtures/datasets/``. Skipped
if either is missing. When both are present, the notebook is executed
end-to-end and a few key results are checked.
"""
from __future__ import annotations

import os
import pathlib

import pytest

from .conftest import EXAMPLES_DIR, run_notebook

NOTEBOOK = EXAMPLES_DIR / "07_grg_genotypes.ipynb"

_PYGRGL_AVAILABLE = True
try:
    import pygrgl  # noqa: F401
except ImportError:
    _PYGRGL_AVAILABLE = False

_FIXTURE_BASE = pathlib.Path(
    os.path.expanduser("~/Dropbox/grg/glink/tests/fixtures/datasets")
)
_FIXTURES_AVAILABLE = (
    (_FIXTURE_BASE / "tiny_clean" / "genotypes.grg").exists()
    and (_FIXTURE_BASE / "small_clean" / "genotypes.grg").exists()
)


@pytest.mark.skipif(not _PYGRGL_AVAILABLE, reason="pygrgl not installed")
@pytest.mark.skipif(
    not _FIXTURES_AVAILABLE,
    reason=f"glink fixture files not present under {_FIXTURE_BASE}",
)
def test_grg_simulation_matches_dense_at_generation_0():
    ns = run_notebook(NOTEBOOK)

    # The notebook compares GRG-vs-dense variance at gen 0; the values
    # for each component should match exactly (same haplotypes, same
    # effects, same seed).
    gen0_grg = ns["gen0_grg"]
    gen0_dense = ns["gen0_dense"]
    for key in gen0_grg["keys"]:
        idx = gen0_grg["keys"].index(key)
        v_grg = float(gen0_grg["var"][idx])
        v_dense = float(gen0_dense["var"][idx])
        assert abs(v_grg - v_dense) < 1e-8, (
            f"{key}: GRG var = {v_grg:.6f}, dense = {v_dense:.6f}"
        )
