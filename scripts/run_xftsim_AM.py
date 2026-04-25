"""Family simulation with clean DGE / IGE / E partition.

Re-implementation of an older xftsim family simulation against the new
numpy-backed API on the `ajay`/`ap_indirect` branch. The architecture is
deliberately structured so that direct genetic, indirect genetic, and
non-genetic components are separable in the output, and so that the
transmitted vs non-transmitted parental allele effects can be recovered
exactly via subtraction of genetic values.

Per-trait components written to `phenotype_history[g]`:

    <T>.T_mat, <T>.T_pat   transmitted DGE, by parent of origin
    <T>.DGE                child's own direct genetic score (= T_mat + T_pat)
    <T>.PDGE_m, <T>.PDGE_f full parental DGE (lookup; normalize=False)
    <T>.NT_m, <T>.NT_f     non-transmitted parental DGE (= PDGE - T)
    <T>.E                  environmental / non-genetic
    <T>.IGE                indirect genetic effect (TraitB only under the
                           collaborator's model). Sourced from FULL parental
                           DGE (= T + NT), reflecting the biological
                           pathway parent_genome → parent_phenotype →
                           child_environment → child_phenotype. Variance
                           pinned to B_vIGE every generation via a
                           normalize=True parental lookup.
    <T>                    total phenotype (sum of DGE + E [+ IGE])

Note on total phenotype variance under IGE: because IGE sources from
parent's full DGE, transmitted alleles influence the child *both*
directly (via child's own DGE) and indirectly (via parental phenotype).
This induces a positive cov(child DGE, child IGE) of order
sqrt(0.5·B_DGE·B_vIGE), inflating var(<T>) above the marginal-component
sum. That confound is a real biological feature; the T_mat / T_pat /
NT_m / NT_f outputs are emitted precisely so downstream analyses (e.g.
Kong-style non-transmitted-coefficient estimators) can identify the
pure indirect effect.

First-cut scope: uniform-AF founders, uniform recombination, random /
single-trait AM / cross-trait AM, TraitB-only IGE. Empirical genomes,
pyrho recombination maps, VCF output, and popstrat are deferred.

The phenotype_component_covariances file is emitted as a DataFrame via
pandas.to_pickle matching the original collaborator-side spec; the file
is a simulation artifact consumed only by the same analysis pipeline.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from xftsim.founders import founder_haplotypes_uniform_AFs
from xftsim.narch import Architecture
from xftsim.neffect import AdditiveEffects
from xftsim.nmate import LinearAssortativeMating, RandomMating
from xftsim.nsim import NSimulation
from xftsim.reproduce import RecombinationMap


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

def draw_bivariate_dge_betas(A_DGE, B_DGE, DGE_ABcov, afs, rng):
    """Draw an (m, 2) bivariate-normal per-allele beta matrix.

    DGE_ABcov is a correlation in [-1, 1]. Betas are drawn in standardized-
    genotype convention and then rescaled by 1/sqrt(2·p·(1-p)) per SNP.

    Why: HaplotypeGeneticComponent uses raw 0/1 haplotype dosages (it does
    not call standardized_matvec), so for E[var(DGE)] ≈ h² the per-allele
    effect β on raw dosages must satisfy Σ 2p(1-p)·β² = h². Drawing
    β_std ~ N(0, h²/m) and setting β = β_std / sqrt(2p(1-p)) achieves this.
    """
    m = len(afs)
    cov_snp = np.array([
        [A_DGE / m,                                   DGE_ABcov * np.sqrt(A_DGE * B_DGE) / m],
        [DGE_ABcov * np.sqrt(A_DGE * B_DGE) / m,      B_DGE / m],
    ])
    betas_std = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov_snp, size=m)
    scale = 1.0 / np.sqrt(2.0 * afs * (1.0 - afs))
    return betas_std * scale[:, None]


def build_architecture(trait_names, trait_variances, ige_config, betas, *,
                        founder_dge_var):
    """Build the formula DSL string and effects dict.

    Parameters
    ----------
    trait_names : list[str]
        e.g. ['TraitA', 'TraitB']. Length 1 or 2.
    trait_variances : dict[str, dict]
        Per trait: {'DGE': float, 'E': float, 'IGE': float}. 'IGE' is
        total IGE variance on the trait (0 for traits without IGE).
    ige_config : dict or None
        {'trait': 'TraitB', 'p_mom': 0.5, 'p_dad': 0.5} or None.
    betas : dict[str, np.ndarray]
        Per trait: 1-D effect array of length m.
    founder_dge_var : dict[str, float]
        Stand-in var for PDGE lookup at gen 0 (≈ DGE variance per trait).

    Returns
    -------
    (formula_str, effects_dict)
    """
    effects = {}
    lines = []

    for T in trait_names:
        eff_name = f"{T}_eff"
        # standardized=False: betas are already on the raw-dosage scale
        # (rescaled by 1/sqrt(2p(1-p)) in draw_bivariate_dge_betas).
        effects[eff_name] = AdditiveEffects.from_array(betas[T], standardized=False)
        # Per-haplotype DGE (transmitted allele scores, by parent of origin)
        lines.append(f"{T}.T_mat ~ haplotypeGenetic({eff_name}, haplotype='maternal')")
        lines.append(f"{T}.T_pat ~ haplotypeGenetic({eff_name}, haplotype='paternal')")
        lines.append(f"{T}.DGE ~ {T}.T_mat + {T}.T_pat")
        # Environment
        lines.append(f"{T}.E ~ noise({trait_variances[T]['E']})")

    if ige_config is not None:
        T = ige_config['trait']
        v = trait_variances[T]['IGE']
        p_mom = ige_config['p_mom']
        p_dad = ige_config['p_dad']

        # Raw parental full DGE — used for T/NT subtraction in genetic-score
        # units. normalize=False keeps the PDGE value on the same scale as
        # the child's haplotypeGenetic outputs, so subtraction yields the
        # *actual* effect of the non-transmitted parental haplotype.
        lines.append(
            f"{T}.PDGE_m ~ mother({T}.DGE, normalize=False, "
            f"founder=noise({founder_dge_var[T]}))"
        )
        lines.append(
            f"{T}.PDGE_f ~ father({T}.DGE, normalize=False, "
            f"founder=noise({founder_dge_var[T]}))"
        )
        lines.append(f"{T}.NT_m ~ {T}.PDGE_m - {T}.T_mat")
        lines.append(f"{T}.NT_f ~ {T}.PDGE_f - {T}.T_pat")

        # Separate normalize=True lookup feeds the IGE channel. Standardizing
        # each generation pins var(IGE) to v (= B_vIGE) regardless of drift
        # or AM-induced variance inflation in parental DGE.
        lines.append(
            f"{T}.IGE_src_m ~ mother({T}.DGE, normalize=True, founder=noise(1.0))"
        )
        lines.append(
            f"{T}.IGE_src_f ~ father({T}.DGE, normalize=True, founder=noise(1.0))"
        )
        c_m = np.sqrt(v * p_mom)
        c_d = np.sqrt(v * p_dad)
        lines.append(f"{T}.IGE ~ {c_m} * {T}.IGE_src_m + {c_d} * {T}.IGE_src_f")

    # Totals
    for T in trait_names:
        parts = [f"{T}.DGE", f"{T}.E"]
        if ige_config is not None and ige_config['trait'] == T:
            parts.append(f"{T}.IGE")
        lines.append(f"{T} ~ " + " + ".join(parts))

    formula = "\n".join(lines)
    return formula, effects


# ---------------------------------------------------------------------------
# Mating regime
# ---------------------------------------------------------------------------

def build_mating_regime(AMmode, AMtrait, AMcorr, trait_names):
    """Construct the mating regime per spec.

    Note: the original collaborator's script multiplied AMcorr by 3 in
    `cross` mode. That factor was specific to the *old* API's internal
    `r` parameterization. In the new API, `r` is documented as the
    target per-trait spousal correlation, so AMcorr is passed through
    unchanged for both single- and cross-trait AM here.
    """
    if AMmode == "random" or AMcorr == 0.0:
        return RandomMating(offspring_per_pair=2)
    if AMmode == "single":
        if AMtrait not in trait_names:
            raise ValueError(
                f"--AMtrait {AMtrait} not in trait list {trait_names}"
            )
        return LinearAssortativeMating(
            component_names=[AMtrait], r=AMcorr, offspring_per_pair=2,
        )
    if AMmode == "cross":
        return LinearAssortativeMating(
            component_names=list(trait_names), r=AMcorr, offspring_per_pair=2,
        )
    raise ValueError(f"unknown AMmode={AMmode}")


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_trios(sim, child_gen, out_dir):
    """Write a (child_iid, parent1_iid, parent2_iid) table keyed by family.

    Under `offspring_per_pair=2`, each family has two siblings in the
    child generation; we keep the first-seen sibling per FID.
    """
    child = sim.phenotype_history[child_gen].samples
    parent = sim.phenotype_history[child_gen - 1].samples
    ped = sim.pedigree_history[child_gen]

    # First offspring per family
    _, first_idx = np.unique(child.fid, return_index=True)
    first_idx = np.sort(first_idx)

    mom_iid = parent.iid[ped.maternal_idx[first_idx]]
    dad_iid = parent.iid[ped.paternal_idx[first_idx]]
    kid_iid = child.iid[first_idx]

    df = pd.DataFrame({
        'child':   _as_str(kid_iid),
        'parent1': _as_str(mom_iid),
        'parent2': _as_str(dad_iid),
    })
    df.to_csv(out_dir / "trios.tab", sep="\t", header=False, index=False)
    return df, first_idx


def write_child_phenotypes(sim, child_gen, first_idx, trait_names, out_dir):
    """Per-trait TSVs: one row per trio-child, columns [ID, VAL]."""
    ph = sim.phenotype_history[child_gen]
    iid = _as_str(ph.samples.iid[first_idx])
    for T in trait_names:
        vals = ph[T][first_idx]
        pd.DataFrame({'ID': iid, 'VAL': vals}).to_csv(
            out_dir / f"children{T}.tab", sep="\t", header=False, index=False,
        )


def write_dge_betas(betas, trait_names, m, out_dir):
    """Write per-variant DGE betas in a single CSV indexed by vid."""
    vid = np.arange(m, dtype=int)
    df = pd.DataFrame({T: betas[T] for T in trait_names}, index=vid)
    df.index.name = 'vid'
    df.to_csv(out_dir / "xftsim_DGE_betas.csv")


def write_parent_child_phenotypes(sim, child_gen, trios_df, trait_names,
                                   has_ige, ige_trait, out_dir):
    """Wide parent-child table with phenotype and every component column.

    Keeps the DGE / IGE / E partition (and T/NT for the IGE trait) visible
    in the output so downstream analyses can compare transmitted vs
    non-transmitted allele effects directly.
    """
    child = sim.phenotype_history[child_gen]
    parent = sim.phenotype_history[child_gen - 1]
    child_iid = _as_str(child.samples.iid)
    parent_iid = _as_str(parent.samples.iid)

    child_df = pd.DataFrame({k: child[k] for k in child.keys}, index=child_iid)
    parent_df = pd.DataFrame({k: parent[k] for k in parent.keys}, index=parent_iid)

    per_trait_cols = []
    for T in trait_names:
        per_trait_cols.extend([T, f"{T}.DGE", f"{T}.E"])
    if has_ige:
        T = ige_trait
        per_trait_cols.extend([
            f"{T}.IGE",
            f"{T}.T_mat", f"{T}.T_pat",
            f"{T}.NT_m",  f"{T}.NT_f",
            f"{T}.PDGE_m", f"{T}.PDGE_f",
        ])

    out = trios_df.copy()
    for col in per_trait_cols:
        out[f"child_{col}"]   = out.child.map(child_df[col])
        out[f"parent1_{col}"] = out.parent1.map(parent_df[col])
        out[f"parent2_{col}"] = out.parent2.map(parent_df[col])

    out = out.dropna()
    out.to_csv(
        out_dir / "parent_child_phenotypes.txt",
        sep=" ", header=True, index=False,
    )
    return out


def write_component_covariance(sim, child_gen, first_idx, out_dir):
    """Pickle the covariance of trio-child phenotype components."""
    ph = sim.phenotype_history[child_gen]
    df = pd.DataFrame({k: ph[k][first_idx] for k in ph.keys})
    df.cov().to_pickle(out_dir / "phenotype_component_covariances.pkl")


def _as_str(arr):
    return np.asarray(arr).astype(str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--founder_zarr', default=None)
    p.add_argument('--chr_count', type=int, default=1)
    p.add_argument('--nfam', type=int, required=True)
    p.add_argument('--empirical_genomes', choices=["True", "False"], default="False")
    p.add_argument('--empirical_recombination', choices=["True", "False"], default="False")
    p.add_argument('--recombination_path_template', default=None)
    p.add_argument('--p_IGEmom', type=float, default=0.5)
    p.add_argument('--p_IGEdad', type=float, default=0.5)
    p.add_argument('--A_DGE', type=float, required=True)
    p.add_argument('--B_DGE', type=float, required=True)
    p.add_argument('--DGE_ABcov', type=float, required=True,
                   help="Correlation (not covariance) between TraitA and TraitB DGE effects, in [-1, 1].")
    p.add_argument('--B_vIGE', type=float, required=True)
    p.add_argument('--generations', type=int, required=True)
    p.add_argument('--out_directory', required=True)
    p.add_argument('--replicate', required=True)
    p.add_argument('--output_format', choices=["vcf", "plink"], default="vcf")
    p.add_argument('--AMmode', choices=["random", "single", "cross"], required=True)
    p.add_argument('--AMtrait', choices=["TraitA", "TraitB"], default="TraitB")
    p.add_argument('--AMcorr', type=float, required=True)
    p.add_argument('--selected_chrs', type=int, nargs='+', default=None)
    p.add_argument('--trait_count', type=int, choices=[1, 2], default=2)
    p.add_argument('--popstrat', type=int, choices=[0, 1], default=0)
    p.add_argument('--snps_per_chr', type=int, default=10_000,
                   help="Synthetic-founders only: SNPs per chromosome when --empirical_genomes=False.")
    p.add_argument('--seed', type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()

    if args.generations < 2:
        raise ValueError("--generations must be >= 2")
    if args.empirical_genomes == "True":
        raise NotImplementedError(
            "empirical_genomes=True (sgkit/zarr loader) is deferred; "
            "run with --empirical_genomes False for now."
        )
    if args.empirical_recombination == "True":
        raise NotImplementedError(
            "empirical_recombination=True (pyrho map) is deferred."
        )
    if args.output_format == "vcf":
        print("[warn] VCF output deferred; skipping chr*.vcf.gz generation.",
              file=sys.stderr)
    if args.popstrat == 1:
        raise NotImplementedError("--popstrat 1 (two-cohort drift) deferred.")

    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out_directory) / f"replicate_{args.replicate}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Founders ----------------------------------------------------------
    m = args.snps_per_chr * args.chr_count
    n_samples = 2 * args.nfam
    founder_hap = founder_haplotypes_uniform_AFs(n=n_samples, m=m)

    # --- Effect sizes ------------------------------------------------------
    all_trait_names = ["TraitA", "TraitB"]
    trait_names = all_trait_names[: args.trait_count]

    afs = np.asarray(founder_hap.variants.af, dtype=np.float64)
    beta_mat = draw_bivariate_dge_betas(
        args.A_DGE, args.B_DGE, args.DGE_ABcov, afs, rng,
    )
    betas = {T: beta_mat[:, i] for i, T in enumerate(all_trait_names)}

    # --- Variances & IGE config -------------------------------------------
    A_E = 1.0 - args.A_DGE
    B_E = 1.0 - (args.B_DGE + args.B_vIGE)
    if A_E < 0 or B_E < 0:
        raise ValueError(
            f"Residual E variance negative: A_E={A_E}, B_E={B_E}. "
            "Check A_DGE, B_DGE, B_vIGE."
        )

    trait_variances = {
        "TraitA": {"DGE": args.A_DGE, "E": A_E, "IGE": 0.0},
        "TraitB": {"DGE": args.B_DGE, "E": B_E, "IGE": args.B_vIGE},
    }
    founder_dge_var = {"TraitA": args.A_DGE, "TraitB": args.B_DGE}

    has_ige = args.B_vIGE > 0 and "TraitB" in trait_names
    ige_config = None
    if has_ige:
        ige_config = {
            "trait": "TraitB",
            "p_mom": args.p_IGEmom,
            "p_dad": args.p_IGEdad,
        }

    # --- Architecture ------------------------------------------------------
    formula, effects = build_architecture(
        trait_names, trait_variances, ige_config,
        {T: betas[T] for T in trait_names},
        founder_dge_var=founder_dge_var,
    )
    arch = Architecture(formula=formula, effects=effects)

    # --- Recombination & mating -------------------------------------------
    rmap = RecombinationMap.constant_map(m=m, p=0.5)
    mating = build_mating_regime(
        args.AMmode, args.AMtrait, args.AMcorr, trait_names,
    )

    # --- Run --------------------------------------------------------------
    sim = NSimulation(
        founder_haplotypes=founder_hap,
        architecture=arch,
        mating_regime=mating,
        recombination_map=rmap,
        seed=args.seed,
    )
    sim.run(args.generations)

    # --- Outputs ----------------------------------------------------------
    child_gen = args.generations - 1

    trios_df, first_idx = write_trios(sim, child_gen, out_dir)
    write_child_phenotypes(sim, child_gen, first_idx, trait_names, out_dir)
    if args.A_DGE != 0 or args.B_DGE != 0:
        write_dge_betas(betas, trait_names, m, out_dir)
    write_parent_child_phenotypes(
        sim, child_gen, trios_df, trait_names,
        has_ige, "TraitB", out_dir,
    )
    write_component_covariance(sim, child_gen, first_idx, out_dir)

    print(f"[ok] wrote replicate outputs to {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        os._exit(1)
    os._exit(0)
