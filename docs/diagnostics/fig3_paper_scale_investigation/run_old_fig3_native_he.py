"""Run OLD xftsim 5xAM+VT at paper scale using OLD xftsim's OWN HasemanElstonEstimator.

Eliminates the possibility that my manual HE reimplementation differs.
Still runs the full paper pipeline (halving + estimators).
"""
import os, sys, warnings, time
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["OMP_NUM_THREADS"] = '8'
os.environ["MKL_NUM_THREADS"] = '8'
os.environ["OPENBLAS_NUM_THREADS"] = '8'

sys.path.insert(0, '/home/rsb/Dropbox/ftsim/xftsim')

import argparse
p = argparse.ArgumentParser()
p.add_argument('--n', type=int, default=512_000)
p.add_argument('--m', type=int, default=1045)
p.add_argument('--seed', type=int, default=1)
p.add_argument('--theta', type=float, default=0.05)
p.add_argument('--r', type=float, default=0.2)
p.add_argument('--h2', type=float, default=0.5)
p.add_argument('--K', type=int, default=5)
p.add_argument('--minMAF', type=float, default=0.05)
p.add_argument('--out', type=str, default='/home/rsb/data/fig3_paperscale/old_native_he.csv')
args = p.parse_args()

import xftsim as xft
import numpy as np
import pandas as pd

np.random.seed(args.seed)
n, m, theta, h2, K, r_mate = args.n, args.m, args.theta, args.h2, args.K, args.r

afs = np.random.uniform(args.minMAF, 1 - args.minMAF, m)
founders = xft.founders.founder_haplotypes_from_AFs(n=n, afs=afs, diploid=True)
rmap = xft.reproduce.RecombinationMap.constant_map_from_haplotypes(founders, p=50/m)

phenos = [f'y{i}' for i in range(K)]
pheno_ind = xft.index.ComponentIndex.from_product(phenos, 'phenotype')
parent_pheno_ind = xft.index.ComponentIndex.from_product(phenos, 'phenotype', [0, 1])
parent_pheno_ind.comp_type = 'outcome'
genetic_ind = xft.index.ComponentIndex.from_product(phenos, 'addGen')
inherited_ind = xft.index.ComponentIndex.from_product(phenos, 'vert')
noise_ind = xft.index.ComponentIndex.from_product(phenos, 'noise')
env_ind = xft.index.ComponentIndex.from_product(phenos, 'env')

ve = 1 - h2
noise_var = ve - theta
vg = np.ones(K) * h2

eff = xft.effect.NonOverlappingEffects(vg=vg, variant_indexer=founders.xft.get_variant_indexer(),
                                       component_indexer=genetic_ind)
genetic_comp = xft.arch.AdditiveGeneticComponent(eff)
noise_comp = xft.arch.AdditiveNoiseComponent(variances=np.ones(K)*noise_var, component_index=noise_ind)
vert_comp = xft.arch.LinearVerticalComponent(
    input_cindex=parent_pheno_ind, output_cindex=inherited_ind,
    coefficient_matrix=np.ones((K, 2*K)) * np.sqrt(theta / (2*K)),
    founder_variances=np.ones(2*K), normalize=True)
env_comp = xft.arch.SumAllTransformation(input_cindex=inherited_ind.merge(noise_ind),
                                          output_component_name='env', output_comp_type='intermediate')
inter_comp = xft.arch.SumAllTransformation(input_cindex=env_ind.merge(genetic_ind),
                                            output_component_name='phenotype', output_comp_type='outcome')
arch = xft.arch.Architecture([genetic_comp, noise_comp, vert_comp, env_comp, inter_comp])
mate_reg = xft.mate.LinearAssortativeMatingRegime(
    offspring_per_pair=2, mates_per_female=1, r=r_mate, component_index=pheno_ind)

# Use OLD xftsim's OWN HasemanElstonEstimator as in the paper's sim
sample_stats = [xft.stats.SampleStatistics(),
                xft.stats.HasemanElstonEstimator(component_index=pheno_ind),
                xft.stats.MatingStatistics()]

sim = xft.sim.Simulation(architecture=arch, founder_haplotypes=founders,
                         mating_regime=mate_reg, statistics=[],
                         recombination_map=rmap, filter_sample=False)

print(f"OLD xftsim 5xAM+VT (native HE) | n={n} m={m} seed={args.seed}", flush=True)
t0 = time.time()
for GEN in range(6):
    t_gen = time.time()
    sim.increment_generation()
    sim.reproduce()
    sim.compute_phenotypes()
    # halve as in paper
    if GEN > 0:
        sim.haplotypes = sim.haplotypes[::2, :]
        sim.phenotypes = sim.phenotypes[::2, :]
    sim.mate()
    sim.update_pedigree()
    # run native HE on halved pop
    sim.statistics = sample_stats
    sim.estimate_statistics()
    sim.process()
    n_now = sim.haplotypes.shape[0]
    print(f"  gen {GEN} done ({time.time()-t_gen:.1f}s) pop={n_now} [total {time.time()-t0:.1f}s]", flush=True)

# Extract HE_regression from final gen
res = sim.results_store[5]
print(f"\nresults_store[5] keys: {list(res.keys())}", flush=True)
he_reg = res['HE_regression']
print(f"HE_regression keys: {list(he_reg.keys())}", flush=True)

# cov_HE and corr_HE are pandas DataFrames, indexed by the phenotype tuples
cov_HE = he_reg['cov_HE']
corr_HE = he_reg['corr_HE']
print(f"\ncov_HE shape: {cov_HE.shape}")
print(f"cov_HE (truncated):\n{cov_HE.iloc[:3, :3]}")

# Mean diagonal (h2_HE) and mean upper triangle (rg_HE)
triu = np.triu_indices(K, k=1)
h2_he_mean = float(np.mean(np.diag(cov_HE.values)))
rg_he_mean = float(np.mean(corr_HE.values[triu]))
print(f"\nGen 5 HE (NATIVE old xftsim):")
print(f"  h2_HE mean = {h2_he_mean:.4f}")
print(f"  rg_HE mean = {rg_he_mean:.4f}")

# True values
tmp = sim.phenotype_store[5]
addGen = tmp.xft[{'component_name': 'addGen', 'vorigin_relative': -1}].to_numpy()
tot = tmp.xft[{'component_name': 'phenotype', 'vorigin_relative': -1}].to_numpy()
h2_true_mean = float(np.mean(addGen.var(axis=0) / tot.var(axis=0)))
rg_true_mat = np.corrcoef(addGen.T)
rg_true_mean = float(np.mean(rg_true_mat[triu]))
print(f"\nGen 5 TRUE: h2={h2_true_mean:.4f} rg={rg_true_mean:.4f}")

# Sample size info
n_sample = sim.haplotypes.shape[0]
print(f"\nHE sample size: {n_sample}")

out = dict(
    seed=args.seed, n=args.n, m=args.m, halved=True,
    h2_true_mean=h2_true_mean, rg_true_mean=rg_true_mean,
    h2_HE_native=h2_he_mean, rg_HE_native=rg_he_mean,
    n_HE_sample=n_sample, engine='OLD_xftsim_native_HE',
)
pd.DataFrame([out]).to_csv(args.out, index=False)
print(f"\nSaved to {args.out}")
print(f"Total time: {(time.time()-t0)/60:.1f} min")
