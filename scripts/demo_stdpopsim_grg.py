#!/usr/bin/env python
"""Demo: stdpopsim founders -> GRG -> forward-time xftsim with GRG-native recombination.

Run from the repository root with the project environment, e.g.

    .venv-xftsim/bin/python scripts/demo_stdpopsim_grg.py
    .venv-xftsim/bin/python scripts/demo_stdpopsim_grg.py --pops YRI=200,CEU=200,CHB=200 --check-dense
    .venv-xftsim/bin/python scripts/demo_stdpopsim_grg.py --pops CEU=10000 --length-mb 10
    .venv-xftsim/bin/python scripts/demo_stdpopsim_grg.py --show-guard
    .venv-xftsim/bin/python scripts/demo_stdpopsim_grg.py --causal-min-maf 0.05
    .venv-xftsim/bin/python scripts/demo_stdpopsim_grg.py --restandardize
    .venv-xftsim/bin/python scripts/demo_stdpopsim_grg.py --pops YRI=200,CEU=200 --track-ancestry

Steps: (1) simulate founders with a published stdpopsim demographic model and
convert the tree sequence to a GRG; (2) build the forward recombination map
from the same genetic map stdpopsim used; (3) define a phenotype and mating
model; (4) run generations, each one doing GRG-native meiosis. Optional:
--check-dense reruns the identical simulation on dense genotypes and checks
the two agree exactly; --show-guard shows meiosis refusing a GRG that still
contains recurrent (duplicate) mutations; --restandardize standardizes
genotypes with each generation's own allele frequencies instead of the
founders' (standardize_with='current'), which rescales allele effects as they
drift and makes var(G) ratchet down as rare variants are lost; compare it with
the default to see the difference; --track-ancestry carries each
person's stdpopsim population forward as ancestry proportions (mean of the
parents') and reports how the admixed population homogenizes.
"""
import argparse
import contextlib
import os
import sys
import tempfile
import textwrap
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from xftsim.arch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.effect import AdditiveEffects
from xftsim.founders import founder_haplotypes_from_stdpopsim_grg
from xftsim.inherit import AncestryProportions, Maternal
from xftsim.mate import RandomMating
from xftsim.reproduce import RecombinationMap, _crossover_model, _meiosis_pair_switches_seeded
from xftsim.sim import Simulation
from xftsim.stats import MetadataSummary
from xftsim.struct import DenseHaplotypeArray


@contextlib.contextmanager
def quiet_subprocess_output():
    """Hide stdout/stderr written by child processes (grg convert); replay on error."""
    sys.stdout.flush()
    sys.stderr.flush()
    with tempfile.TemporaryFile(mode="w+") as log:
        saved = os.dup(1), os.dup(2)
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        try:
            yield
        except BaseException:
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            log.seek(0)
            sys.stderr.write(log.read())
            raise
        finally:
            os.dup2(saved[0], 1)
            os.dup2(saved[1], 2)
            os.close(saved[0])
            os.close(saved[1])


def parse_pops(text):
    pops = {}
    for item in text.split(","):
        name, count = item.split("=")
        pops[name.strip()] = int(count)
    return pops


def make_founders(args, drop_multi_mutation_sites=True):
    left = int(args.left_mb * 1e6)
    right = left + int(args.length_mb * 1e6)
    with quiet_subprocess_output():
        return founder_haplotypes_from_stdpopsim_grg(
            samples=parse_pops(args.pops), model_id=args.model,
            chromosome=args.chrom, species_id=args.species,
            genetic_map=args.genetic_map, left=left, right=right,
            seed=args.seed, drop_multi_mutation_sites=drop_multi_mutation_sites,
        )


def grg_genotypes(op, block=1024):
    """(n, m, 2) int8 haplotypes from a GRG, in variant blocks (no m x m identity)."""
    import pygrgl
    n, m = op.n, op.m
    out = np.empty((n, m, 2), dtype=np.int8)
    for start in range(0, m, block):
        stop = min(start + block, m)
        unit = np.zeros((stop - start, m))
        unit[np.arange(stop - start), np.arange(start, stop)] = 1.0
        hap = pygrgl.matmul(op._grg, unit, pygrgl.TraversalDirection.DOWN,
                            by_individual=False)          # (block, 2n)
        out[:, start:stop, 0] = hap[:, 0::2].T            # maternal
        out[:, start:stop, 1] = hap[:, 1::2].T            # paternal
    return out


def make_effects(founders, args):
    """beta ~ N(0, h2/m) on standardized genotypes; optionally only variants
    with founder MAF >= --causal-min-maf are causal (rescaled so sum beta^2 = h2).

    The simulation standardizes genotypes with the founders' allele
    frequencies p0, so each allele's effect beta / sqrt(2 p0 (1 - p0)) is
    fixed across generations (--restandardize uses each generation's own
    frequencies instead).
    """
    af = founders.recompute_af()
    beta = AdditiveEffects.from_h2(h2=args.h2, m=founders.m, seed=args.seed).effects.copy()
    if args.causal_min_maf > 0:
        beta[np.minimum(af, 1 - af) < args.causal_min_maf] = 0.0
        beta *= np.sqrt(args.h2 / np.sum(beta ** 2))
    return AdditiveEffects.from_array(beta)


def build_simulation(founders, effects, rmap, args, callbacks=()):
    arch = Architecture()
    arch.add('Y.G', GeneticComponent(effects))
    arch.add('Y.E', NoiseComponent(variance=1.0 - args.h2))
    arch.add('Y', AggregationComponent('Y.G + Y.E'))
    tracking = {}
    if args.track_ancestry:
        tracking = dict(
            metadata_rules=[AncestryProportions('population'), Maternal('population')],
            statistics=[MetadataSummary()],
        )
    return Simulation(
        founder_haplotypes=founders, architecture=arch,
        mating_regime=RandomMating(offspring_per_pair=2),
        recombination_map=rmap, seed=args.seed,
        retain_haplotypes=0, retain_phenotypes=args.gens,
        standardize_with='current' if args.restandardize else 'founders',
        callbacks=list(callbacks), **tracking,
    )


def report_ancestry(sim):
    """Per-generation mean and variance of each ancestry proportion."""
    first = sim.results[0].statistics['MetadataSummary']
    cols = sorted(k for k in first if k.startswith('ancestry_'))
    names = [c[len('ancestry_'):] for c in cols]
    print("[ancestry] expected ancestry proportions (mean of parents'), "
          "carried forward each generation")
    print("      gen " + " ".join(f"{'mean ' + n:>11}" for n in names)
          + " " + " ".join(f"{'var ' + n:>10}" for n in names))
    for result in sim.results:
        summary = result.statistics['MetadataSummary']
        print(f"      {result.generation:>3} "
              + " ".join(f"{summary[c]['mean']:>11.3f}" for c in cols) + " "
              + " ".join(f"{summary[c]['var']:>10.4f}" for c in cols))
    print("      (means stay at the founder mix; variances shrink about 2x per "
          "generation under random mating)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--pops", default="CEU=500",
                    help="population=diploid count, comma-separated (default CEU=500)")
    ap.add_argument("--species", default="HomSap")
    ap.add_argument("--model", default="OutOfAfrica_3G09")
    ap.add_argument("--chrom", default="chr22")
    ap.add_argument("--left-mb", type=float, default=20.0, help="window start (Mb)")
    ap.add_argument("--length-mb", type=float, default=2.0, help="window length (Mb)")
    ap.add_argument("--genetic-map", default=None,
                    help="stdpopsim map id, e.g. HapMapII_GRCh38 (downloaded on first use)")
    ap.add_argument("--gens", type=int, default=5, help="generations including founders")
    ap.add_argument("--h2", type=float, default=0.5)
    ap.add_argument("--causal-min-maf", type=float, default=0.0,
                    help="only variants with founder MAF >= this are causal (default: all)")
    ap.add_argument("--restandardize", action="store_true",
                    help="standardize genotypes with each generation's allele "
                         "frequencies (default: the founders', so allele effects "
                         "stay fixed)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--check-dense", action="store_true",
                    help="rerun on dense genotypes and require exact agreement")
    ap.add_argument("--show-guard", action="store_true",
                    help="show meiosis refusing founders that keep recurrent mutations")
    ap.add_argument("--track-ancestry", action="store_true",
                    help="carry population labels forward as ancestry proportions "
                         "and report them per generation")
    args = ap.parse_args()

    # Compile the meiosis kernels up front so generation timings are honest.
    _meiosis_pair_switches_seeded(*_crossover_model(np.array([0.5, 0.1])), np.uint32(0))

    # ---- 1. founders -------------------------------------------------------
    pops = parse_pops(args.pops)
    lo = args.left_mb
    print(f"[1/4] stdpopsim  {args.species} / {args.model} / "
          f"{args.chrom}:{lo:g}-{lo + args.length_mb:g} Mb, samples {pops}"
          + (f", map {args.genetic_map}" if args.genetic_map else ", uniform map"))
    t0 = time.perf_counter()
    founders = make_founders(args)
    grg = founders._grg
    af = founders.recompute_af()
    maf = np.minimum(af, 1 - af)
    print(f"      simulated + converted to GRG in {time.perf_counter() - t0:.1f} s")
    print(f"      GRG: {grg.num_samples:,} haplotype samples ({founders.n:,} people), "
          f"{founders.m:,} variants, {grg.num_nodes:,} nodes, {grg.num_edges:,} edges")
    print(f"      variants: {np.mean(maf < 0.01):.0%} with MAF < 1%, "
          f"{int(np.sum(np.round(af * 2 * founders.n) == 1)):,} singletons")

    # ---- 2. recombination map ---------------------------------------------
    rmap = RecombinationMap.from_haplotypes_with_cM(founders)
    pos_cM = founders.variants.pos_cM
    print(f"[2/4] recombination map from stdpopsim's genetic map: "
          f"{pos_cM[-1] - pos_cM[0]:.2f} cM over {args.length_mb:g} Mb "
          f"-> {rmap._probabilities[1:].sum():.3f} expected crossovers per meiosis")

    # ---- 3. model ----------------------------------------------------------
    effects = make_effects(founders, args)
    n_causal = int(np.count_nonzero(effects.effects))
    causal = (f"all {n_causal:,} variants causal" if args.causal_min_maf <= 0 else
              f"{n_causal:,} variants with MAF >= {args.causal_min_maf:g} causal")
    scaling = ("re-standardized with each generation's frequencies" if args.restandardize
               else "allele effects fixed at founder frequencies")
    print(f"[3/4] model: Y = G + E, h2 = {args.h2} ({causal}; {scaling}); "
          f"random mating, 2 offspring per pair (constant N = {founders.n:,})")

    dense_founders = None
    if args.check_dense:
        dense_founders = DenseHaplotypeArray(
            genotypes=grg_genotypes(founders), samples=founders.samples,
            variants=founders.variants)

    # ---- 4. forward simulation --------------------------------------------
    print(f"[4/4] forward simulation, {args.gens} generations (GRG-native meiosis)")
    print(f"      {'gen':>3} {'seconds':>8} {'nodes':>11} {'nodes +%':>9} "
          f"{'edges':>12} {'edges +%':>9} {'var(G)':>8} {'var(Y)':>8}")
    print("      (+% = growth since the previous generation)")
    clock = [time.perf_counter()]
    start_size = (grg.num_nodes, grg.num_edges)
    prev_size = [start_size]

    def growth(new, old):
        return f"{100.0 * (new - old) / old:+.2f}%"

    def report(sim):
        now = time.perf_counter()
        g = sim.generation
        pheno = sim.phenotype_history[g]
        live = sim.haplotypes._grg
        nodes, edges = live.num_nodes, live.num_edges
        if g == 0:
            node_pct = edge_pct = "-"
        else:
            node_pct = growth(nodes, prev_size[0][0])
            edge_pct = growth(edges, prev_size[0][1])
        print(f"      {g:>3} {now - clock[0]:>8.2f} {nodes:>11,} {node_pct:>9} "
              f"{edges:>12,} {edge_pct:>9} "
              f"{np.var(pheno['Y.G']):>8.3f} {np.var(pheno['Y']):>8.3f}")
        prev_size[0] = (nodes, edges)
        clock[0] = now

    sim = build_simulation(founders, effects, rmap, args, callbacks=[report])
    t0 = time.perf_counter()
    sim.run(args.gens)
    final = sim.haplotypes._grg
    print(f"      done in {time.perf_counter() - t0:.1f} s; final population is "
          f"{type(sim.haplotypes).__name__} with n={sim.haplotypes.n:,}")
    print(f"      total GRG growth since founders: nodes "
          f"{growth(final.num_nodes, start_size[0])}, edges "
          f"{growth(final.num_edges, start_size[1])}")
    if args.track_ancestry:
        report_ancestry(sim)

    # ---- optional: exact agreement with the dense path ---------------------
    if args.check_dense:
        print("[check] same simulation on dense genotypes (same seeds) ...")
        dense_sim = build_simulation(dense_founders, effects, rmap, args)
        dense_sim.run(args.gens)
        g_grg = grg_genotypes(sim.haplotypes)
        g_dense = dense_sim.haplotypes.genotypes
        n_diff = int(np.sum(g_grg != g_dense))
        dG = max(np.max(np.abs(sim.phenotype_history[g]['Y.G']
                               - dense_sim.phenotype_history[g]['Y.G']))
                 for g in range(args.gens))
        meta_same = all(
            sim.phenotype_history[g].samples.extra.keys()
            == dense_sim.phenotype_history[g].samples.extra.keys()
            and all(np.array_equal(v, dense_sim.phenotype_history[g].samples.extra[k])
                    for k, v in sim.phenotype_history[g].samples.extra.items())
            for g in range(args.gens))
        verdict = ("EXACT MATCH" if n_diff == 0 and dG < 1e-9 and meta_same
                   else "MISMATCH")
        print(f"        genotype cells differing at generation {args.gens - 1}: "
              f"{n_diff:,} of {g_dense.size:,}")
        if args.track_ancestry:
            print(f"        per-person metadata identical in every generation: {meta_same}")
        print(f"        max |G_grg - G_dense| over all generations: {dG:.1e}  -> {verdict}")

    # ---- optional: duplicate-mutation guard --------------------------------
    if args.show_guard:
        print("[guard] same founders, but keeping sites with recurrent mutations ...")
        raw = make_founders(args, drop_multi_mutation_sites=False)
        keys = list(zip(raw.variants.pos_bp, raw.variants.zero_allele, raw.variants.one_allele))
        print(f"        {raw.m - founders.m} extra mutations; "
              f"{len(keys) - len(set(keys))} duplicate (position, ref, alt) keys")
        assignment = RandomMating(offspring_per_pair=2).mate(
            raw.samples, rng=np.random.RandomState(0))
        try:
            raw.meiosis(assignment, RecombinationMap.from_haplotypes_with_cM(raw),
                        rng=np.random.RandomState(0))
            print("        meiosis ran: no duplicates in this sample (try a larger window)")
        except ValueError as err:
            print("        meiosis refused:")
            print(textwrap.fill(str(err), width=78, initial_indent="          ",
                                subsequent_indent="          "))


if __name__ == "__main__":
    main()
