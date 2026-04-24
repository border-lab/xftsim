# Paper-scale 5xAM+VT investigation

Investigation of the ~0.04 rg offset we saw in Ajay's figure 3 reproduction vs the paper's archived values, at paper-exact parameters.

## Punchline

**Both NEW and OLD xftsim produce rg ≈ 0.514 at paper-exact parameters (n=512k, m=1045, halved, native HE).** The paper's archived value (0.557, mean of 750 seeds, SD=0.010) cannot be reproduced by the code at `/home/rsb/Dropbox/ftsim/xftsim/`. The offset is between the current OLD-xftsim codebase and the paper's archived data — not between Ajay's new code and OLD xftsim.

## Parameters (from `revision_sims/submit_vdecomp_full.sh` for 5xAM+VT)

- n=512,000; m∈{1045, 2730, 4855}; minMAF=0.05
- kphen=kmate=5; rmate=0.2; theta=0.05; h2=0.5
- Pipeline: `LinearAssortativeMatingRegime` + `LinearVerticalComponent(normalize=True)` + `[::2]` halving each generation + `HasemanElstonEstimator` on halved pop

## Results table

| Setup | h²_HE | rg_HE | seeds | source |
|---|---|---|---|---|
| **Paper archive (merged_011024.csv)** | **0.748** | **0.557** | 750 | `/home/rsb/Dropbox/ftsim/round4/data/sim_results/merged_tabla_redux_results_011024.csv` |
| NEW xftsim — n=32k, m=4k | 0.773 | 0.529 | 5 | `fig3_reproduction/` (our earlier run) |
| NEW xftsim — n=256k, m=4k | 0.747 | 0.513 | 5 | `5xAM_plus_VT_seed{1-5}.csv` |
| OLD xftsim — n=256k, m=4k, no halving, manual HE | 0.754 | 0.514 | 1 | `old_fig3_seed1.csv` |
| OLD xftsim — n=256k, m=4k, halved, manual HE | 0.754 | 0.517 | 1 | `old_halved_seed1.csv` |
| OLD xftsim — n=512k, m=1045, halved, manual HE | 0.722 | 0.514 | 1 | `old_halved_n512k_m1045_seed1.csv` |
| **OLD xftsim — n=512k, m=1045, halved, native HE** | **0.746** | **0.514** | 1 | `old_native_he_seed1.csv` |

Paper's 750-seed distribution: mean 0.557, SD 0.010, range [0.526, 0.590]. Our 0.514 is outside the min. See `paper_he_rg_hist.png`.

## Scripts

- **`run_fig3_paperscale.py`** — NEW xftsim 5xAM+VT at n=256k, m=4k (uses `xftsim/test_figure3.py` with monkey-patched founders + HE).
- **`run_old_fig3.py`** — OLD xftsim 5xAM+VT, no halving, with manual one-per-family HE.
- **`run_old_fig3_halved.py`** — adds `[::2]` halving each generation.
- **`run_old_fig3_native_he.py`** — uses OLD xftsim's own `HasemanElstonEstimator` (same code path as paper's `general_simulations_vdecomp.py`).

## Interpretation

1. **Ajay's new xftsim faithfully reproduces OLD xftsim** (rg agrees to 0.003 at matched params).
2. **The paper's archived 0.557 is not reproducible from the current OLD-xftsim codebase**, even with identical parameters, halving pipeline, and native HE. Most likely the paper's original 2024-01 runs used an xftsim installed in the `xft-rev` conda env on PSC Bridges-2 (see submit script) that differs in some subtle way from the Dropbox working copy. Candidate commits around that date in `/home/rsb/Dropbox/ftsim/xftsim` git log: `ff4f231` (2024-01-09, "standardize built into general mating"), `095c494` (2024-01-09, "fixed qap"). Could also be numpy/scipy/BLAS differences, but unlikely to produce a systematic 0.04 offset.
3. **The scientific finding is unchanged**: under xAM+VT, HE rg inflates from ~0 at gen 0 to ~0.51–0.56 at gen 5. The exact endpoint is ~0.04 lower in our reproduction, but the qualitative pattern and order of magnitude match.

## Next-step ideas (if you want to chase down the 0.04)

- Check the `xft-rev` conda env on PSC (if accessible) — try to install that exact version locally.
- Run several seeds of `general_simulations_vdecomp.py` as-is (preserving its exact flow incl. `HasemanElstonEstimatorSibship` in `estimators` list) at n=512k m=1045 and see if the `HE_regression` key differs from what we get in isolation.
- Diff `xftsim/stats.py` against commits from before 2024-01-10 to see if HE changed in a way that affects corr_HE.

Not worth chasing for the primary reproduction question — Ajay's code is validated against OLD xftsim, which is what the paper "should have" produced.
