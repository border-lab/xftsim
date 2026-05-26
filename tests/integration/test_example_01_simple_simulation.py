"""Integration test for docs/examples/01_simple_simulation.ipynb.

Verifies the univariate Y = G + E simulation (h2 = 0.5, n = 1000, m = 200)
reaches the expected generation and produces variance components within
tolerance.
"""
from __future__ import annotations

import numpy as np

from .conftest import EXAMPLES_DIR, run_notebook

NOTEBOOK = EXAMPLES_DIR / "01_simple_simulation.ipynb"


def test_runs_to_generation_4_and_recovers_h2():
    ns = run_notebook(NOTEBOOK)

    sim = ns["sim"]
    pheno = ns["pheno"]

    # sim.run(5) means generations 0..4
    assert sim.generation == 4

    # Phenotype components present
    keys = list(pheno.keys)
    assert {"Y.G", "Y.E", "Y"}.issubset(set(keys))

    # Variance additivity: Var(Y) ≈ Var(G) + Var(E), Cov(G, E) ≈ 0
    var_g = float(np.var(pheno["Y.G"]))
    var_e = float(np.var(pheno["Y.E"]))
    var_y = float(np.var(pheno["Y"]))
    cov_ge = float(np.cov(pheno["Y.G"], pheno["Y.E"])[0, 1])

    assert abs(cov_ge) < 0.05, f"Cov(G, E) = {cov_ge:.3f}, expected ~0"
    assert abs(var_y - (var_g + var_e)) < 0.05

    # Realised h2 within a wide tolerance (n=1000 is small)
    h2_realised = var_g / var_y
    assert 0.35 < h2_realised < 0.65, f"h2 = {h2_realised:.3f}, expected ~0.5"


def test_results_present_per_generation():
    ns = run_notebook(NOTEBOOK)
    sim = ns["sim"]

    # SampleStatistics ran each generation 0..4
    assert len(sim.results) == 5
    for r in sim.results:
        assert "SampleStatistics" in r.statistics
        stats = r.statistics["SampleStatistics"]
        assert "Y.G" in stats["keys"]
        assert "Y" in stats["keys"]
