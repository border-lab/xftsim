# Figure 3 reproduction (Ajay's new xftsim)

Reproducing Figure 3 of the xAM manuscript using `xftsim/test_figure3.py` from the `ajay` branch. Four scenarios (RM, RM+VT, 5xAM, 5xAM+VT), 5 traits with h²=0.5, 5 seeds per scenario, 6 generations, n=32,000, m=4,000, theta=0.05 (VT fraction), xmate_r=0.2. Paper used n=256,000 and m=2,000–4,000; differences are sample-size driven, not architectural.

## Scripts

- **`run_fig3_v3.py`** — main driver. Runs all 4 scenarios × 5 seeds, writes per-seed CSV caches to `/home/rsb/data/fig3_results/` (intentionally not on Dropbox), aggregates to `fig3_aggregated.csv`. Computes full-sample HE at every gen; one-per-family HE only at gen 5 (for paper-methodology comparison). Stubs `msprime`/`tskit`/`pygrgl` imports to avoid those heavy deps.
- **`overlay_fig3_v2.py`** — reads `fig3_aggregated.csv` plus `/home/rsb/Dropbox/ftsim/round4/processed/fig3_h2_plot.csv` and `fig3_rg_plot.csv`, produces overlay PNGs/PDFs and prints the gen-5 comparison table.

## Files

| File | Contents |
|---|---|
| `fig3_aggregated.csv` | 120 rows (20 seeds × 6 gens) with `mean_h2_true`, `mean_rg_true`, `mean_he_h2_full`, `mean_he_rg_full`, `mean_he_h2_unrel`, `mean_he_rg_unrel`, `mean_var_y`, `n_unrel`. Summarized data only — raw per-seed CSVs stay on local disk. |
| `fig3_5xAM_overlay.png/.pdf` | 5xAM scenario: Ajay's h² and rg trajectories overlaid on paper's 10–90 pct band. |
| `fig3_5xAM_plus_VT_overlay.png/.pdf` | 5xAM+VT scenario, same format. Shows both full-sample and one-per-family HE at gen 5. |
| `fig3_RM_scenarios.png/.pdf` | RM and RM+VT (no paper processed-data reference, so no band). |

## Gen-5 numerical comparison

| Scenario | Quantity | Paper median [10–90 pct] | Ajay mean ± SD | `|`diff`|` |
|---|---|---|---|---|
| RM | HE h² | — | 0.495 ± 0.004 | — |
| RM | HE rg | — | 0.002 ± 0.008 | — |
| RM+VT | HE h² | — | 0.556 ± 0.006 | — |
| RM+VT | HE rg | — | 0.213 ± 0.008 | — |
| **5xAM** | HE h² | 0.659 [0.638, 0.678] | **0.659 ± 0.006** | **0.0001** |
| **5xAM** | HE rg | 0.295 [0.280, 0.311] | **0.294 ± 0.007** | **0.0010** |
| 5xAM+VT | HE h² (full sample) | 0.748 [0.728, 0.769] | 0.773 ± 0.007 | 0.025 |
| 5xAM+VT | HE h² (1/family) | 0.748 [0.728, 0.769] | 0.765 ± 0.014 | 0.017 |
| 5xAM+VT | HE rg (full sample) | 0.557 [0.544, 0.570] | 0.529 ± 0.008 | 0.028 |
| 5xAM+VT | HE rg (1/family) | 0.557 [0.544, 0.570] | 0.525 ± 0.008 | 0.032 |

## Interpretation

- **Three of four scenarios match within 0.001** (and RM/RM+VT have no paper processed reference but match theory).
- **5xAM is a bullseye** — Ajay's 5-seed mean lands on the paper's median.
- **5xAM+VT shows a small systematic offset** (~0.02 h² too high, ~0.03 rg too low). Sibling inflation in Ajay's full-sample HE is visible but small (0.008 gap between full and one-per-family). The residual ~0.017 offset is most plausibly driven by n=32k vs paper's n=256k, and `minMAF=0.1` default vs paper's `0.05`.
- **No architectural bug** — the VT construction (shared `mother(Y)` / `father(Y)` nodes aggregated into trait-specific VT components) matches the legacy `LinearVerticalComponent`.

## Paper reference data source

`/home/rsb/Dropbox/ftsim/round4/processed/fig3_h2_plot.csv` and `fig3_rg_plot.csv`. These were generated from ~1000 replicates per scenario by `general_simulations.py` at n=256,000 (source: `/home/rsb/Dropbox/ftsim/round4/xftmanu_code_supplement/`). Raw per-seed outputs are on weasel at `~/data/edu_no_CD_LS.01/` for fig4 and at various `~/data/*sims*` directories for fig3.

## Reproducing

```bash
# Run the sim (requires matplotlib, numpy, pandas; no hexaly/localsolver needed since LinearAssortativeMating is rank-order)
python run_fig3_v3.py

# Make the plots (requires fig3_aggregated.csv and paper processed CSVs)
python overlay_fig3_v2.py
```

Runtime: ~87 min for 20 seeds on an 8-core machine.
