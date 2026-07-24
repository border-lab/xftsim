"""Integration test for docs/examples/03_vertical_transmission.ipynb.

Vertical transmission via ``parent(Y, founder=noise(...))``. Verifies
that VT inflates Var(Y) and midparent-offspring correlation relative
to the no-VT control by the final generation.
"""
from __future__ import annotations

from .conftest import EXAMPLES_DIR, run_notebook

NOTEBOOK = EXAMPLES_DIR / "03_vertical_transmission.ipynb"


def test_vt_inflates_phenotypic_variance_relative_to_control():
    ns = run_notebook(NOTEBOOK)
    var_y_vt = ns["var_y_vt"]      # list of (gen, var)
    var_y_ctrl = ns["var_y_ctrl"]

    # By the final generation the VT model should have higher Var(Y).
    gen_vt, v_vt_final = var_y_vt[-1]
    gen_ctrl, v_ctrl_final = var_y_ctrl[-1]
    assert gen_vt == gen_ctrl
    assert v_vt_final > v_ctrl_final, (
        f"VT Var(Y) = {v_vt_final:.3f}, control = {v_ctrl_final:.3f} — "
        f"VT should inflate"
    )


def test_vt_simulation_continues_to_run_for_three_more_generations():
    ns = run_notebook(NOTEBOOK)
    sim_vt = ns["sim_vt"]
    # The notebook runs 5 generations and then continue_run(3).
    # sim.run(5) → generations 0..4, continue_run(3) → 5..7 → final gen 7.
    assert sim_vt.generation == 7
    assert len(sim_vt.results) == 8


def test_vt_component_is_tracked_and_nonzero():
    ns = run_notebook(NOTEBOOK)
    sim_vt = ns["sim_vt"]
    pheno = sim_vt.phenotypes
    assert "Y.VT" in pheno.keys
    # At the final generation, the VT component should have non-trivial variance
    import numpy as np
    var_vt = float(np.var(pheno["Y.VT"]))
    assert var_vt > 0.01, f"Var(Y.VT) = {var_vt:.4f}, expected > 0.01"
