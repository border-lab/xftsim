"""Integration test for docs/examples/08_direct_indirect_effects.ipynb.

The notebook builds a vertical-transmission architecture with
transmitted/non-transmitted (T/NT) parental allele scores and recovers
direct + indirect effects via OLS regression on the four T/NT columns.

This is the slowest example notebook by far (§5 runs n_fam = 10,000,
m = 2,000), so it only runs when ``XFTSIM_RUN_SLOW_EXAMPLES=1`` is set.
Also requires ``statsmodels`` and ``matplotlib`` (the notebook imports
both at the top).
"""
from __future__ import annotations

import os

import pytest

from .conftest import EXAMPLES_DIR, run_notebook

NOTEBOOK = EXAMPLES_DIR / "08_direct_indirect_effects.ipynb"

_RUN_SLOW = os.environ.get("XFTSIM_RUN_SLOW_EXAMPLES", "") == "1"

_STATSMODELS_AVAILABLE = True
try:
    import statsmodels  # noqa: F401
except ImportError:
    _STATSMODELS_AVAILABLE = False

_MATPLOTLIB_AVAILABLE = True
try:
    import matplotlib  # noqa: F401
except ImportError:
    _MATPLOTLIB_AVAILABLE = False


pytestmark = [
    pytest.mark.skipif(
        not _RUN_SLOW,
        reason=(
            "08_direct_indirect_effects runs a large simulation (n=10k, m=2k). "
            "Set XFTSIM_RUN_SLOW_EXAMPLES=1 to enable."
        ),
    ),
    pytest.mark.skipif(
        not _STATSMODELS_AVAILABLE, reason="statsmodels not installed"
    ),
    pytest.mark.skipif(
        not _MATPLOTLIB_AVAILABLE, reason="matplotlib not installed"
    ),
]


def test_t_nt_decomposition_is_exact():
    ns = run_notebook(NOTEBOOK)
    ph = ns["ph"]  # last-assigned phenotype dict from the big run
    import numpy as np
    # NT = PDGE - T must hold to machine precision (by construction).
    nt_recon = ph["TraitB.PDGE_m"] - ph["TraitB.T_mat"]
    err = float(np.abs(nt_recon - ph["TraitB.NT_m"]).max())
    assert err < 1e-9, f"max |PDGE_m - T_mat - NT_m| = {err:.2e}"
    # Diploid DGE = T_mat + T_pat must also hold exactly.
    dge_recon = ph["TraitB.T_mat"] + ph["TraitB.T_pat"]
    err_dge = float(np.abs(dge_recon - ph["TraitB.DGE"]).max())
    assert err_dge < 1e-9


def test_ntc_regression_recovers_indirect_coefficient():
    ns = run_notebook(NOTEBOOK)
    ols = ns["ols"]
    alpha_m = ns["alpha_m"]
    alpha_f = ns["alpha_f"]

    # ols.params order: [intercept, T_mat, T_pat, NT_m, NT_f]
    # NT_m coef should ≈ alpha_m, NT_f coef should ≈ alpha_f.
    # T_mat coef should ≈ 1 + alpha_m, T_pat coef should ≈ 1 + alpha_f.
    coefs = list(ols.params)
    nt_m_hat, nt_f_hat = coefs[3], coefs[4]
    t_mat_hat, t_pat_hat = coefs[1], coefs[2]

    # n=10k should give tight estimates; allow ±0.05 for SE.
    assert abs(nt_m_hat - alpha_m) < 0.05, (
        f"NT_m coef = {nt_m_hat:.3f}, expected ≈ {alpha_m:.3f}"
    )
    assert abs(nt_f_hat - alpha_f) < 0.05
    # Direct effect (T - NT) should be 1 by construction.
    direct_m = t_mat_hat - nt_m_hat
    direct_f = t_pat_hat - nt_f_hat
    assert abs(direct_m - 1.0) < 0.05, f"direct (mat) = {direct_m:.3f}"
    assert abs(direct_f - 1.0) < 0.05, f"direct (pat) = {direct_f:.3f}"
