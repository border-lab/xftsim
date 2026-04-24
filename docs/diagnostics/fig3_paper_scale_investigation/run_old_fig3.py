"""Run OLD xftsim's fig3 replica (5xAM+VT) at paper scale.

Uses the actual paper code from general_simulations.py (LinearAssortativeMatingRegime,
LinearVerticalComponent) but without the [::2] halving or RandomSiblingSubsampleFilter,
so HE runs on one-per-family only at final gen (matching our new xftsim measurement).

This tells us: does OLD xftsim at paper scale also give rg~0.513, or does it give
rg~0.557? Definitively identifies whether the gap is new-vs-old or scale-independent.
"""
import os, sys, warnings, time
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ["OMP_NUM_THREADS"] = '8'
os.environ["MKL_NUM_THREADS"] = '8'
os.environ["OPENBLAS_NUM_THREADS"] = '8'
os.environ["NUMEXPR_NUM_THREADS"] = '1'

sys.path.insert(0, '/home/rsb/Dropbox/ftsim/xftsim')

import argparse
p = argparse.ArgumentParser()
p.add_argument('--n', type=int, default=256_000)
p.add_argument('--m', type=int, default=4000)
p.add_argument('--seed', type=int, default=1)
p.add_argument('--gens', type=int, default=6)
p.add_argument('--theta', type=float, default=0.05)
p.add_argument('--r', type=float, default=0.2)
p.add_argument('--h2', type=float, default=0.5)
p.add_argument('--K', type=int, default=5)
p.add_argument('--minMAF', type=float, default=0.05)
p.add_argument('--out', type=str, default='/home/rsb/scratch/xft_diag/old_fig3_results.csv')
args = p.parse_args()

import xftsim as xft
import numpy as np
import pandas as pd

np.random.seed(args.seed)
n, m, theta, h2, K, r_mate = args.n, args.m, args.theta, args.h2, args.K, args.r

# Founders
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
noise_var = ve - theta  # = 0.45 when h2=0.5, theta=0.05
vg = np.ones(K) * h2

# Orthogonal genetic effects
eff = xft.effect.NonOverlappingEffects(
    vg=vg, variant_indexer=founders.xft.get_variant_indexer(),
    component_indexer=genetic_ind)
genetic_comp = xft.arch.AdditiveGeneticComponent(eff)

noise_comp = xft.arch.AdditiveNoiseComponent(
    variances=np.ones(K) * noise_var,
    component_index=noise_ind)

vert_comp = xft.arch.LinearVerticalComponent(
    input_cindex=parent_pheno_ind,
    output_cindex=inherited_ind,
    coefficient_matrix=np.ones((K, 2*K)) * np.sqrt(theta / (2*K)),
    founder_variances=np.ones(2*K),
    normalize=True)

env_comp = xft.arch.SumAllTransformation(
    input_cindex=inherited_ind.merge(noise_ind),
    output_component_name='env', output_comp_type='intermediate')

inter_comp = xft.arch.SumAllTransformation(
    input_cindex=env_ind.merge(genetic_ind),
    output_component_name='phenotype', output_comp_type='outcome')

arch = xft.arch.Architecture([genetic_comp, noise_comp, vert_comp, env_comp, inter_comp])

mate_reg = xft.mate.LinearAssortativeMatingRegime(
    offspring_per_pair=2, mates_per_female=1, r=r_mate,
    component_index=pheno_ind)

sim = xft.sim.Simulation(
    architecture=arch, founder_haplotypes=founders,
    mating_regime=mate_reg, statistics=[],
    recombination_map=rmap, filter_sample=False)

print(f"Running OLD xftsim 5xAM+VT at n={n}, m={m}, seed={args.seed}, minMAF={args.minMAF}", flush=True)
t0 = time.time()
for GEN in range(args.gens):
    t_gen = time.time()
    sim.increment_generation()
    sim.reproduce()
    sim.compute_phenotypes()
    sim.mate()
    sim.update_pedigree()
    sim.process()
    print(f"  gen {GEN} done ({time.time()-t_gen:.1f}s) [total {time.time()-t0:.1f}s]", flush=True)

# Extract gen-5 data for HE
final_gen = args.gens - 1
tmp = sim.phenotype_store[final_gen]

# Compute true h² and rg from genetic components
addGen = tmp.xft[{'component_name': 'addGen', 'vorigin_relative': -1}].to_numpy()
tot = tmp.xft[{'component_name': 'phenotype', 'vorigin_relative': -1}].to_numpy()
var_g = addGen.var(axis=0)
var_y = tot.var(axis=0)
h2_true_per = var_g / var_y
rg_true_mat = np.corrcoef(addGen.T)
triu = np.triu_indices(K, k=1)

print(f"\nGen {final_gen} TRUE values:")
print(f"  h²_true mean = {np.mean(h2_true_per):.4f}")
print(f"  rg_true mean = {np.mean(rg_true_mat[triu]):.4f}")

# Compute HE on one-per-family subsample (to match our new-xftsim measurement)
hap = sim.haplotype_store[final_gen]
ped = sim.pedigree
sample_idx = hap.xft.get_sample_indexer()
fids = np.array(sample_idx.fid)
_, first_idx = np.unique(fids, return_index=True)
keep = np.sort(first_idx)
print(f"  one-per-family n: {len(keep)}")

G = hap[keep, :].xft.to_diploid_standardized().astype(np.float64)
Y = tot[keep, :]
Y = (Y - Y.mean(axis=0)) / np.where(Y.std(axis=0) < 1e-15, 1.0, Y.std(axis=0))

n_sub, m_sub = G.shape
GtY = G.T @ Y
KY = G @ GtY / m_sub
# Stochastic trace
rng = np.random.RandomState(args.seed + 1000)
n_probe = 100
probes = rng.randn(n_sub, n_probe)
KP = G @ (G.T @ probes) / m_sub
trK2 = float(np.sum(KP**2) / n_probe)
denom = trK2 - n_sub
cov_g = Y.T @ (KY - Y) / denom
d = np.sqrt(np.abs(np.diag(cov_g)))
d[d < 1e-15] = 1.0
rg_he = cov_g / np.outer(d, d)

print(f"\nGen {final_gen} HE one-per-family (OLD xftsim):")
print(f"  h²_HE mean = {np.mean(np.diag(cov_g)):.4f}")
print(f"  rg_HE mean = {np.mean(rg_he[triu]):.4f}")

# Save
out = dict(
    seed=args.seed, n=args.n, m=args.m, theta=args.theta, r=args.r, minMAF=args.minMAF,
    h2_true_mean=float(np.mean(h2_true_per)),
    rg_true_mean=float(np.mean(rg_true_mat[triu])),
    h2_HE_unrel=float(np.mean(np.diag(cov_g))),
    rg_HE_unrel=float(np.mean(rg_he[triu])),
    n_unrel=int(n_sub),
    engine='OLD_xftsim',
)
pd.DataFrame([out]).to_csv(args.out, index=False)
print(f"\nSaved to {args.out}")
print(f"Total time: {(time.time()-t0)/60:.1f} min")
