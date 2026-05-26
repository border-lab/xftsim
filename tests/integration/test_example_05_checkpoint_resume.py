"""Integration test for docs/examples/05_checkpoint_resume.ipynb.

Verifies the checkpoint round-trip: save → load → resume produces the
exact same phenotypes at the checkpointed generation, and continue_run
advances further.
"""
from __future__ import annotations

import numpy as np

from .conftest import EXAMPLES_DIR, run_notebook

NOTEBOOK = EXAMPLES_DIR / "05_checkpoint_resume.ipynb"


def test_loaded_simulation_matches_original_phenotypes_exactly():
    ns = run_notebook(NOTEBOOK)
    sim = ns["sim"]
    sim_loaded = ns["sim_loaded"]

    # The notebook stores the original sim at generation 4 (after run(5)),
    # then runs continue_run(5) on the loaded copy, advancing it to gen 9.
    # We compare the gen-4 phenotypes that both share.
    pheno_orig = sim.phenotype_history[sim.generation]  # gen 4
    pheno_loaded = sim_loaded.phenotype_history[4]
    assert np.allclose(pheno_orig["Y"], pheno_loaded["Y"])
    assert np.allclose(pheno_orig["Y.G"], pheno_loaded["Y.G"])


def test_continue_run_advances_loaded_simulation():
    ns = run_notebook(NOTEBOOK)
    sim_loaded = ns["sim_loaded"]
    # Original run(5) → gen 4. continue_run(5) → gen 9.
    assert sim_loaded.generation == 9


def test_npz_roundtrip_recovers_exact_arrays():
    ns = run_notebook(NOTEBOOK)
    sim_loaded = ns["sim_loaded"]
    pheno_loaded = ns["pheno_loaded"]
    hap_loaded = ns["hap_loaded"]
    eff_loaded = ns["eff_loaded"]
    eff = ns["eff"]

    # Phenotype and effect arrays should round-trip exactly through npz.
    assert np.allclose(pheno_loaded["Y"], sim_loaded.phenotypes["Y"])
    assert np.array_equal(hap_loaded.genotypes, sim_loaded.haplotypes.genotypes)
    assert np.allclose(eff_loaded.effects, eff.effects)
    assert eff_loaded.standardized == eff.standardized
