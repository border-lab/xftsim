"""Integration test for docs/examples/02_bivariate_assortative.ipynb.

Two genetically correlated traits (rg = 0.3) under linear assortative
mating on height (r = 0.3). Verifies the realised genetic correlation
at generation 0 tracks the target, and that AM inflates Var(height.G)
relative to random mating by generation 4.
"""
from __future__ import annotations

import numpy as np

from .conftest import EXAMPLES_DIR, run_notebook

NOTEBOOK = EXAMPLES_DIR / "02_bivariate_assortative.ipynb"


def test_genetic_correlation_at_gen0_tracks_target():
    ns = run_notebook(NOTEBOOK)
    sim_am = ns["sim_am"]

    # gen 0 is in phenotype_history (retain_phenotypes=5)
    pheno0 = sim_am.phenotype_history[0]
    rg_realised = float(
        np.corrcoef(pheno0["height.G"], pheno0["bmi.G"])[0, 1]
    )
    # Target rg=0.3, with n=1000 finite-sample noise
    assert 0.15 < rg_realised < 0.45, f"rg = {rg_realised:.3f}"


def test_AM_inflates_height_genetic_variance_vs_RM():
    ns = run_notebook(NOTEBOOK)
    var_hg_am = ns["var_hg_am"]
    var_hg_rm = ns["var_hg_rm"]
    # Both lists are length 5 (gen 0..4)
    # By gen 4 the AM regime should have inflated Var(height.G) above RM.
    assert var_hg_am[-1] > var_hg_rm[-1], (
        f"AM final Var(height.G) = {var_hg_am[-1]:.3f}, "
        f"RM final = {var_hg_rm[-1]:.3f} — AM should inflate"
    )
    # And gen 0 should be approximately equal (same founders, same seed)
    assert abs(var_hg_am[0] - var_hg_rm[0]) < 0.05


def test_final_covariance_matrix_recovers_phenotypic_correlation():
    ns = run_notebook(NOTEBOOK)
    rp = ns["rp"]
    # rp is the realised cor(height, bmi) under AM at the final generation.
    # Genetic rg=0.3 and AM induces some additional cross-trait correlation,
    # but rp should remain in a wide positive band.
    assert 0.0 < rp < 0.6, f"rp = {rp:.3f}, expected positive and < 0.6"
