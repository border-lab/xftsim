"""Integration test for docs/examples/06_sibling_effects.ipynb.

The notebook currently fails because it calls
``HasemanElstonEstimator(filter_name='sibpair')`` — the ``filter_name``
parameter was removed when the HE estimator was rewritten to use the
GRM directly. We mark this expected-failure so the test exists and
documents the breakage without breaking CI; once the notebook is
updated, drop the ``xfail`` marker.
"""
from __future__ import annotations

import pytest

from .conftest import EXAMPLES_DIR, run_notebook

NOTEBOOK = EXAMPLES_DIR / "06_sibling_effects.ipynb"


@pytest.mark.xfail(
    reason=(
        "06_sibling_effects.ipynb uses HasemanElstonEstimator(filter_name='sibpair'), "
        "an API that was removed when HE moved to a GRM-based formulation. "
        "Notebook is intentionally not patched (per repo owner)."
    ),
    strict=True,
)
def test_runs_end_to_end():
    run_notebook(NOTEBOOK)
