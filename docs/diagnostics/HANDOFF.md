# Handoff: xftsim reproduction of Fig 3 & Fig 4 — current vs archived

For the agent working on the xAM-complex-architecture manuscript revision. Short version at top; details and file paths below.

## TL;DR

1. **Ajay's new xftsim (`ajay` branch) reproduces the current manuscript text for Fig 3 exactly.** At paper-exact parameters it gives 5xAM+VT gen-5 **h²≈0.750, rg≈0.513** — matching manuscript rev4 text values of 0.750 and 0.513 to 3 decimals.

2. **There is an internal inconsistency in the manuscript: the Fig 3c panel shows rg ≈ 0.56, but the text says 0.513.** The figure was rendered from a stale processed CSV (`fig3_rg_plot.csv`); the text was updated later to match a newer run whose values both the current OLD xftsim (`/home/rsb/Dropbox/ftsim/xftsim/`) and NEW xftsim (Ajay's branch) now reproduce. The figure and one paragraph of text need to be brought into sync. The code path in the repo produces 0.513, not 0.557.

3. **Fig 4 (education/height/wealth): no bug in either xftsim version either.** The "paper reference" values in Ajay's `test_simulation.py` comment (0.686 / 0.068 / 0.036 for height/edu/wealth HE h² at gen 5) are **mean-over-seeds** at n=128k/m=2000. Ajay's test at n=20k/m=1000/seed=42 cannot possibly hit those means with a single seed — seed-42 at paper scale already gives (0.643, 0.047, 0.029), well below the mean, and Ajay's smaller-scale run will be noisier still. The Fig 4 architecture (shared-node VT via `MotherComponent`/`FatherComponent`) was verified semantically equivalent to the legacy `LinearVerticalComponent` via a gen-0 closed-form test.

## Background: what was asked

1. Ajay built a reproduction of Fig 4 in `xftsim/test_simulation.py` (ajay branch); the end-of-script print mentions paper values that his single-seed run doesn't hit.
2. Then a reproduction of Fig 3 in `xftsim/test_figure3.py`, which is conceptually similar but at a different architecture (5 orthogonal traits, xAM with r=0.2, VT=5%).
3. Question: why do the numbers diverge from the paper? Bug in new xftsim? Bug in old xftsim? Legitimate difference?

## What I did

### Fig 3 investigation
- Ran Ajay's `test_figure3.py` at his scale (n=32k, m=4k, 5 seeds × 4 scenarios). 5xAM matches paper median to 0.001. 5xAM+VT matches paper h² but rg is 0.028 low.
- Ran the same scenario at **paper scale** (n=256k, m=4k, 5 seeds) with paper's minMAF=0.05 and stochastic HE. Gives h²=0.747, rg=0.513.
- Ran **OLD xftsim** (at `/home/rsb/Dropbox/ftsim/xftsim/`) with cumulative `[::2]` halving matching `general_simulations_vdecomp.py`, native HE, at paper-exact params (n=512k, m=1045). Gives h²=0.746, rg=0.514. Effectively identical to Ajay's new code.
- Compared against manuscript text values in `complex_architecture_manu_rev4.docx`: text reports h²=0.750 and rg=0.513, which exactly matches both Ajay and OLD xftsim.
- Compared against `fig3_rg_plot.csv` and Fig 3c panel: those show rg≈0.557. Source: `merged_tabla_redux_results_011024.csv` column `he_rg`, mean over 750 seeds, SD=0.010. Neither current xftsim version can reproduce 0.557 under any setting I tried (with or without halving, different n, different m).

### Fig 4 investigation  
- Gen-0 closed-form test: both old `LinearVerticalComponent` and Ajay's `MotherComponent`/`FatherComponent` shared-node arch produce the same gen-0 `Var(edu.VT) = Var(wealth.VT) ≈ 2/3` and `corr(edu.VT, wealth.VT) ≈ √½ ≈ 0.707`, matching theory. Semantic equivalence verified.
- The 200-seed paper data at n=128k, m=2000 (copied from weasel to `/home/rsb/data/edu_no_CD_LS.01/` — not Dropbox): seed-42 alone gives h²_HE of (0.643, 0.047, 0.029) at gen 5, well below the 200-seed mean (0.686, 0.068, 0.036). A single-seed smaller-scale Ajay run has roughly 1.2× SE, so his expected 95% CI for any single seed comfortably contains both paper's seed-42 value and the multi-seed mean.

## Key files on Dropbox (everything under `xftsim/docs/diagnostics/`)

- **`fig3_reproduction/`** — Ajay-scale fig3 reproduction (n=32k, m=4k, 5 seeds × 4 scenarios).
  - `run_fig3_v3.py`, `overlay_fig3_v2.py` — scripts
  - `fig3_aggregated.csv` — 120 rows summarized (per-seed CSVs are on local disk only)
  - `fig3_5xAM_overlay.{png,pdf}`, `fig3_5xAM_plus_VT_overlay.{png,pdf}`, `fig3_RM_scenarios.{png,pdf}`
  - `README.md` — full table + interpretation
- **`fig3_paper_scale_investigation/`** — chased the 0.557 vs 0.513 discrepancy.
  - `run_fig3_paperscale.py` — NEW xftsim at n=256k/m=4k
  - `run_old_fig3.py`, `run_old_fig3_halved.py`, `run_old_fig3_native_he.py` — OLD xftsim variants  
  - `old_native_he_seed1.csv` — final definitive run (OLD xftsim, n=512k, m=1045, halving, native HE): h²=0.7455, rg=0.5136
  - `paper_he_rg_hist.png` — histogram of 750 paper seeds showing our 0.514 falls outside the paper distribution
  - `README.md` — full investigation
- **Data (not on Dropbox, local only)**:
  - `/home/rsb/data/edu_no_CD_LS.01/` — paper's 200-seed Fig 4 CSVs (pulled from weasel)
  - `/home/rsb/data/fig3_results/` — per-seed CSVs from Ajay-scale fig3 runs
  - `/home/rsb/data/fig3_paperscale/` — per-seed CSVs from paper-scale fig3 runs

## What the other agent needs to decide / do

### For Fig 3 text/figure alignment (this is the actionable item)

The current manuscript has a self-inconsistency: text quotes 0.513 for 5xAM+VT rg, figure shows 0.557. Options in descending order of effort:

1. **Regenerate Fig 3c** from a fresh run of `general_simulations_vdecomp.py` at the exact params in `submit_vdecomp_full.sh` (n=512k, m∈{1045,2730,4855}, 50 seeds each). Update the figure to show rg≈0.513. Most defensible — figure, text, and repo code all agree.
2. **Update the text** to match the stale figure (0.557). Requires understanding what `merged_tabla_redux_results_011024.csv` was generated with — I could NOT reproduce that value with current repo code.  
3. **Keep the text at 0.513 and flag the figure-text inconsistency as immaterial** (both round to "~0.5" and the scientific conclusion is the same). But someone will catch it in proof.

Note: the value **0.513** in the text is also labeled there as "LDSC genetic correlation estimate". If the text value is truly from LDSC rather than HE, then LDSC and HE happen to give similar numbers here (our HE at paper scale = 0.514 ≈ LDSC-reported 0.513); Ajay's xftsim currently has HE (`HasemanElstonEstimator`) but not LDSC built in. Worth confirming whether the text truly used LDSC or was just mislabeled.

### For Fig 4 (`test_simulation.py`)

- The comment block at the bottom of `xftsim/test_simulation.py` (ajay branch) is misleading: it compares a single-seed small-scale run to the paper's multi-seed mean values. Either update the comment to note these are means (not per-seed expectations) and to compare against seed-42's paper-scale value (0.643, 0.047, 0.029) — or run 20+ seeds and compare the mean.
- No code changes needed; the architecture is correct.

## Running things

Scripts assume:
- `/home/rsb/Dropbox/xftsim/.venv/bin/python` for NEW xftsim (ajay branch); install `setuptools<81`, `python-docx` if needed.
- OLD xftsim at `/home/rsb/Dropbox/ftsim/xftsim/` — add `sys.path.insert(0, ...)` in the script header.
- `msprime`/`tskit`/`pygrgl` aren't needed for Fig 3/4 (they're just imported at module top); scripts stub them. `hexaly`/`localsolver` aren't needed for these scenarios (LinearAssortativeMating is rank-order sort).

## One more result worth knowing

Weasel (`weasel:~/data/`) still has all raw paper sim outputs under `~/data/edu_*`, `~/data/2xAM*`, etc. SSH was timing out when I checked, but it's periodically accessible. For multi-seed paper-scale fig3 runs on Bridges-2, see `/home/rsb/Dropbox/ftsim/round4/revision_sims/submit_vdecomp_full.sh`.
