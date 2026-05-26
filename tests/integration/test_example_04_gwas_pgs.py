"""Integration test for docs/examples/04_gwas_pgs.ipynb.

GWAS + polygenic score example. Verifies that:

- GWAS recovers true effect sizes (true vs estimated beta correlation
  exceeds a generous threshold)
- PGS computed with true (oracle) weights recovers Y.G almost exactly
  and approximates h^2 against the total phenotype Y
- Out-of-sample PGS R² is positive
"""
from __future__ import annotations

from .conftest import EXAMPLES_DIR, run_notebook

NOTEBOOK = EXAMPLES_DIR / "04_gwas_pgs.ipynb"


def test_gwas_recovers_true_effect_sizes():
    ns = run_notebook(NOTEBOOK)
    beta_corr = ns["beta_corr"]
    # n=2000 in-sample GWAS should give a clear (>0.3) correlation between
    # true and estimated effects when h2=0.5 and all variants causal.
    assert beta_corr > 0.3, f"corr(true beta, est beta) = {beta_corr:.3f}"


def test_pgs_with_true_weights_recovers_genetic_value():
    ns = run_notebook(NOTEBOOK)
    r2_pgs_yg = ns["r2_pgs_yg"]
    r2_pgs_y = ns["r2_pgs_y"]
    # PGS computed with the true beta against standardized genotypes should
    # reproduce Y.G almost exactly.
    assert r2_pgs_yg > 0.95, f"R²(PGS_true, Y.G) = {r2_pgs_yg:.3f}"
    # And R²(PGS_true, Y) should approximate the design h^2 = 0.5.
    assert 0.35 < r2_pgs_y < 0.65, f"R²(PGS_true, Y) = {r2_pgs_y:.3f}"


def test_unrelated_filter_reduces_sample_size():
    ns = run_notebook(NOTEBOOK)
    hap = ns["hap"]
    unrelated_view = ns["unrelated_view"]
    n_total = hap.n
    n_unrel = len(unrelated_view.indices)
    # After 3 generations with offspring_per_pair=2, ~half should be one
    # per family — but always at least one per FID.
    assert 0 < n_unrel < n_total


def test_out_of_sample_pgs_r2_is_positive():
    ns = run_notebook(NOTEBOOK)
    r2_oos = ns["r2_oos"]
    # OOS R² is noisier than in-sample but should still be positive.
    assert r2_oos > 0.0, f"OOS R² = {r2_oos:.3f}"
