"""Run fig3 scenarios: trajectory from full-sample HE + gen-5 one-per-family HE.

Optimized: one-per-family HE only at gen 5 (for paper-methodology comparison).
All other gens: only full-sample HE (fast, from sim.results statistics).
"""
import sys, os, time
class _Stub:
    def __getattr__(self, n): return _Stub()
    def __call__(self, *a, **k): return _Stub()
sys.modules.setdefault('msprime', _Stub())
sys.modules.setdefault('tskit', _Stub())
sys.modules.setdefault('pygrgl', _Stub())

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd

OUTDIR = '/home/rsb/data/fig3_results'
os.makedirs(OUTDIR, exist_ok=True)

sys.path.insert(0, '/home/rsb/Dropbox/xftsim/xftsim')
import importlib
fig3 = importlib.import_module('xftsim.test_figure3')

def he_one_per_family(hap, pheno, keys):
    """Compute HE on one individual per FID. Returns (mean_h2, mean_rg, n_sub)."""
    K = len(keys); triu = np.triu_indices(K, k=1)
    fids = hap.samples.fid
    _, first_idx = np.unique(fids, return_index=True)
    keep = np.sort(first_idx)
    hap_sub = hap.subset(sample_idx=keep)
    G = hap_sub.to_diploid_standardized(scale=True).astype(np.float64)
    n, m = G.shape
    Y = np.column_stack([pheno[k][keep] for k in keys]).astype(np.float64)
    Y = (Y - Y.mean(0)) / np.where(Y.std(0) < 1e-15, 1.0, Y.std(0))
    GtY = G.T @ Y; KY = G @ GtY / m
    GtG = G.T @ G; trK2 = float(np.sum(GtG**2) / (m*m))
    denom = trK2 - n
    if abs(denom) < 1e-15:
        return np.nan, np.nan, n
    cov_g = Y.T @ (KY - Y) / denom
    d = np.sqrt(np.abs(np.diag(cov_g))); d[d < 1e-15] = 1.0
    rg_mat = cov_g / np.outer(d, d)
    return float(np.mean(np.diag(cov_g))), float(np.mean(rg_mat[triu])), n


def extract_rows(sim, trait_names):
    """Per-gen rows with true h², true rg, full-sample HE h², HE rg. Unrel HE only at final gen."""
    rows = []
    K = len(trait_names); triu = np.triu_indices(K, k=1)
    final_gen = max(r.generation for r in sim.results)

    for result in sim.results:
        gen = result.generation
        stats = result.statistics
        pheno = sim.phenotype_history.get(gen)
        hap = sim.haplotype_history.get(gen)

        # True from phenotype arrays
        mean_h2_true = mean_rg_true = np.nan
        mean_var_y = np.nan
        if pheno is not None:
            G = np.column_stack([pheno[f'{t}.G'] for t in trait_names])
            Y = np.column_stack([pheno[t] for t in trait_names])
            var_y = Y.var(axis=0); var_g = G.var(axis=0)
            mean_h2_true = float(np.mean(var_g / var_y))
            cov_g = np.cov(G.T)
            d = np.sqrt(np.abs(np.diag(cov_g))); d[d < 1e-15] = 1.0
            rg_true = cov_g / np.outer(d, d)
            mean_rg_true = float(np.mean(rg_true[triu]))
            mean_var_y = float(np.mean(var_y))

        # Full-sample HE (from registered statistics)
        he = stats.get('HasemanElstonEstimator', {}) or {}
        mean_he_h2_full = mean_he_rg_full = np.nan
        if he:
            vals = [he[t]['h2'] for t in trait_names if t in he]
            if vals:
                mean_he_h2_full = float(np.mean(vals))
            if '_cov_g' in he:
                cov_g = he['_cov_g']
                d = np.sqrt(np.abs(np.diag(cov_g))); d[d < 1e-15] = 1.0
                rg = cov_g / np.outer(d, d)
                mean_he_rg_full = float(np.mean(rg[triu]))

        # One-per-family HE only at final gen (expensive)
        mean_he_h2_unrel = mean_he_rg_unrel = np.nan
        n_unrel = 0
        if gen == final_gen and hap is not None and pheno is not None:
            mean_he_h2_unrel, mean_he_rg_unrel, n_unrel = he_one_per_family(hap, pheno, trait_names)

        rows.append({
            'gen': int(gen), 'mean_var_y': mean_var_y,
            'mean_h2_true': mean_h2_true, 'mean_rg_true': mean_rg_true,
            'mean_he_h2_full': mean_he_h2_full, 'mean_he_rg_full': mean_he_rg_full,
            'mean_he_h2_unrel': mean_he_h2_unrel, 'mean_he_rg_unrel': mean_he_rg_unrel,
            'n_unrel': int(n_unrel),
        })
    return rows


scenarios = ['RM', 'RM+VT', '5xAM', '5xAM+VT']
n_runs = 5   # reduced from 10 for speed
trait_names = fig3.trait_names

t_start = time.time()
rows_all = []

# Resume from cached per-seed CSVs
for si, scenario in enumerate(scenarios):
    for seed in range(1, n_runs + 1):
        safename = scenario.replace('+','_plus_').replace(' ','')
        per_seed_csv = f'{OUTDIR}/{safename}_seed{seed}.csv'
        if os.path.exists(per_seed_csv):
            df_prev = pd.read_csv(per_seed_csv)
            if 'mean_he_h2_full' in df_prev.columns:
                rows_all.extend(df_prev.to_dict('records'))
                print(f'[{si*n_runs + seed}/{len(scenarios)*n_runs}] SKIP {scenario} seed={seed}', flush=True)
                continue
            else:
                os.remove(per_seed_csv)  # v1 schema, redo
        t0 = time.time()
        print(f'[{si*n_runs + seed}/{len(scenarios)*n_runs}] {scenario} seed={seed} ... ', end='', flush=True)
        sim = fig3.build_sim(scenario, seed=seed)
        # Only final gen needs hap+pheno for one-per-family HE; others only need pheno for true h²/rg
        sim.retain_haplotypes = 1   # just latest gen
        sim.retain_phenotypes = 100  # need all for true h² trajectory
        sim.run(n_generations=6)
        rows = extract_rows(sim, trait_names)
        for r in rows:
            r['scenario'] = scenario
            r['seed'] = seed
        pd.DataFrame(rows).to_csv(per_seed_csv, index=False)
        rows_all.extend(rows)
        pd.DataFrame(rows_all).to_csv(f'{OUTDIR}/fig3_aggregated.csv', index=False)
        el = time.time() - t0
        total_el = time.time() - t_start
        done = si*n_runs + seed
        remaining = (len(scenarios)*n_runs - done)
        eta_min = (total_el / done) * remaining / 60 if done > 0 else 0
        g5 = [r for r in rows if r['gen']==5][0]
        print(f'done ({el:.1f}s) gen5 h2_full={g5["mean_he_h2_full"]:.4f} h2_unrel={g5["mean_he_h2_unrel"]:.4f} rg_full={g5["mean_he_rg_full"]:.4f} rg_unrel={g5["mean_he_rg_unrel"]:.4f} | ETA {eta_min:.1f}m', flush=True)
        del sim

print(f'\nTotal time: {(time.time()-t_start)/60:.1f} min')
print('DONE')
