"""Paper-scale 5xAM+VT reproduction at n=256k, m=4k, minMAF=0.05.

Tests whether the 0.02-0.03 offset seen at Ajay's n=32k is scale-driven.
Uses stochastic HE trace (n_probe=100) for speed.
Saves to /home/rsb/data/fig3_paperscale/ (NOT Dropbox).
"""
import sys, os, time
class _Stub:
    def __getattr__(self, n): return _Stub()
    def __call__(self, *a, **k): return _Stub()
sys.modules.setdefault('msprime', _Stub())
sys.modules.setdefault('tskit', _Stub())
sys.modules.setdefault('pygrgl', _Stub())

import matplotlib; matplotlib.use('Agg')
import numpy as np
import pandas as pd

OUTDIR = '/home/rsb/data/fig3_paperscale'
os.makedirs(OUTDIR, exist_ok=True)

sys.path.insert(0, '/home/rsb/Dropbox/xftsim/xftsim')
import importlib
fig3 = importlib.import_module('xftsim.test_figure3')

# ── Override to paper scale + minMAF + stochastic HE ────────────────────────
fig3.n_individuals = 256_000
fig3.n_loci = 4000

# Monkey-patch founders to use minMAF=0.05 matching paper
_orig_founders = fig3.founder_haplotypes_uniform_AFs
def _founders_minmaf05(n, m):
    return _orig_founders(n, m, minMAF=0.05)
fig3.founder_haplotypes_uniform_AFs = _founders_minmaf05

# Monkey-patch HasemanElstonEstimator inside the build_sim to use n_probe=100
from xftsim.nstats import HasemanElstonEstimator, SampleStatistics
_orig_HE = HasemanElstonEstimator
class _HE100(HasemanElstonEstimator):
    def __init__(self, phenotype_keys=None, n_probe=100):
        super().__init__(phenotype_keys=phenotype_keys, n_probe=n_probe)
# patch the symbol in fig3's namespace
fig3.HasemanElstonEstimator = _HE100


def he_one_per_family(hap, pheno, keys, n_probe=100):
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
    # Stochastic trace for large n
    rng = np.random.RandomState()
    probes = rng.randn(n, n_probe)
    KP = G @ (G.T @ probes) / m
    trK2 = float(np.sum(KP**2) / n_probe)
    denom = trK2 - n
    if abs(denom) < 1e-15:
        return np.nan, np.nan, n
    cov_g = Y.T @ (KY - Y) / denom
    d = np.sqrt(np.abs(np.diag(cov_g))); d[d < 1e-15] = 1.0
    rg_mat = cov_g / np.outer(d, d)
    return float(np.mean(np.diag(cov_g))), float(np.mean(rg_mat[triu])), n


def extract_rows(sim, trait_names):
    rows = []
    K = len(trait_names); triu = np.triu_indices(K, k=1)
    final_gen = max(r.generation for r in sim.results)
    for result in sim.results:
        gen = result.generation
        stats = result.statistics
        pheno = sim.phenotype_history.get(gen)
        hap = sim.haplotype_history.get(gen)
        # True
        mean_h2_true = mean_rg_true = mean_var_y = np.nan
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
        # Full-sample HE
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
        # One-per-family HE only at final gen
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


scenarios = ['5xAM+VT']  # only the problematic one
n_runs = 5
trait_names = fig3.trait_names

t_start = time.time()
rows_all = []

for si, scenario in enumerate(scenarios):
    for seed in range(1, n_runs + 1):
        safename = scenario.replace('+','_plus_').replace(' ','')
        per_seed_csv = f'{OUTDIR}/{safename}_seed{seed}.csv'
        if os.path.exists(per_seed_csv):
            df_prev = pd.read_csv(per_seed_csv)
            rows_all.extend(df_prev.to_dict('records'))
            print(f'[{si*n_runs + seed}/{len(scenarios)*n_runs}] SKIP {scenario} seed={seed}', flush=True)
            continue
        t0 = time.time()
        print(f'[{si*n_runs + seed}/{len(scenarios)*n_runs}] {scenario} seed={seed} (n=256k, m=4k) ... ', end='', flush=True)
        sim = fig3.build_sim(scenario, seed=seed)
        sim.retain_haplotypes = 1
        sim.retain_phenotypes = 100
        sim.run(n_generations=6)
        rows = extract_rows(sim, trait_names)
        for r in rows:
            r['scenario'] = scenario; r['seed'] = seed
        pd.DataFrame(rows).to_csv(per_seed_csv, index=False)
        rows_all.extend(rows)
        pd.DataFrame(rows_all).to_csv(f'{OUTDIR}/fig3_paperscale_aggregated.csv', index=False)
        el = time.time() - t0
        total_el = time.time() - t_start
        done = si*n_runs + seed
        eta_min = (total_el / done) * (len(scenarios)*n_runs - done) / 60 if done > 0 else 0
        g5 = [r for r in rows if r['gen']==5][0]
        print(f'done ({el:.1f}s) gen5 h2_full={g5["mean_he_h2_full"]:.4f} h2_unrel={g5["mean_he_h2_unrel"]:.4f} rg_full={g5["mean_he_rg_full"]:.4f} rg_unrel={g5["mean_he_rg_unrel"]:.4f} | ETA {eta_min:.1f}m', flush=True)
        del sim

print(f'\nTotal time: {(time.time()-t_start)/60:.1f} min')
print('DONE')
