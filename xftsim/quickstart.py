"""
Quickstart simulation for xftsim, ported to the refactored API.

Reproduces the spirit of the legacy ReadTheDocs quickstart
(https://xftsim.readthedocs.io/en/latest/gettingstarted/quickstart.html)
on top of the current ``NSimulation`` / ``Architecture`` / ``nmate`` stack.

API translation table (legacy -> current)
-----------------------------------------
    xft.sim.Simulation                                -> xftsim.nsim.NSimulation
    xft.arch.GCTA_Architecture(h2=..., ...)           -> xftsim.narch.Architecture
                                                         + xftsim.neffect.AdditiveEffects.from_h2
    xft.reproduce.RecombinationMap.constant_map_from_haplotypes
                                                      -> RecombinationMap.from_haplotypes
    xft.mate.LinearAssortativeMatingRegime            -> xftsim.nmate.LinearAssortativeMating
    xft.stats.MatingStatistics / SampleStatistics /   -> xftsim.nstats.MatingStatistics /
        HasemanElstonEstimator                              SampleStatistics /
                                                            HasemanElstonEstimator
    xft.proc.LimitMemory(n_haplotype_generations=1)   -> NSimulation(retain_haplotypes=1, ...)
    sim.results_store[gen][...]                       -> sim.results[i].statistics[...]
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from xftsim.founders import founder_haplotypes_uniform_AFs
from xftsim.neffect import AdditiveEffects
from xftsim.narch import Architecture
from xftsim.nmate import LinearAssortativeMating
from xftsim.nfilter import TrioFilter
from xftsim.nsim import NSimulation
from xftsim.nstats import (
    SampleStatistics,
    HasemanElstonEstimator,
    MatingStatistics,
)
from xftsim.reproduce import RecombinationMap


SEED = 123
N_INDIVIDUALS = 8000
N_LOCI = 1000
N_GENERATIONS = 5
TRAIT_NAMES = ['pheno_1', 'pheno_2']
H2 = 0.5
RECOMB_P = 0.1
SPOUSAL_R = 0.5


def build_sim() -> NSimulation:
    """Construct the quickstart NSimulation."""
    np.random.seed(SEED)

    # 1. Founder haplotypes: 8000 individuals at 1000 diploid sites.
    founder_haplotypes = founder_haplotypes_uniform_AFs(
        n=N_INDIVIDUALS, m=N_LOCI,
    )

    # 2. Genetic architecture: two independent traits with h2 = 0.5 each
    #    (matches GCTA_Architecture(h2=[.5, .5], ...) with no induced rg).
    effects = {
        f'{name}_eff': AdditiveEffects.from_h2(h2=H2, m=N_LOCI, seed=SEED + i)
        for i, name in enumerate(TRAIT_NAMES)
    }
    formula_lines = []
    for name in TRAIT_NAMES:
        formula_lines.append(f'{name}.G ~ genetic({name}_eff)')
        formula_lines.append(f'{name}.E ~ noise({1.0 - H2})')
        formula_lines.append(f'{name} ~ {name}.G + {name}.E')
        formula_lines.append('')
    architecture = Architecture(formula='\n'.join(formula_lines), effects=effects)

    # 3. Recombination map: constant 10% probability between adjacent loci.
    recombination_map = RecombinationMap.from_haplotypes(
        founder_haplotypes, p=RECOMB_P,
    )

    # 4. Mating regime: cross-trait assortative mating, target spousal r = 0.5.
    mating_regime = LinearAssortativeMating(
        component_names=TRAIT_NAMES,
        r=SPOUSAL_R,
        offspring_per_pair=2,
    )

    # 5. Statistics + trio filter (MatingStatistics needs TrioView for spouse r).
    statistics = [
        MatingStatistics(),
        SampleStatistics(),
        HasemanElstonEstimator(phenotype_keys=TRAIT_NAMES),
    ]
    filters = {'trio': TrioFilter()}

    # 6. Tie everything together. retain_haplotypes=1 plays the role of the
    #    legacy LimitMemory(n_haplotype_generations=1) post-processor.
    return NSimulation(
        founder_haplotypes=founder_haplotypes,
        architecture=architecture,
        mating_regime=mating_regime,
        recombination_map=recombination_map,
        statistics=statistics,
        filters=filters,
        retain_haplotypes=1,
        retain_phenotypes=2,
        seed=SEED,
    )


def summarize(sim: NSimulation) -> pd.DataFrame:
    """Collect per-generation HE rg, phenotypic rg, and spouse correlations."""
    rows = []
    for result in sim.results:
        gen = result.generation
        stats = result.statistics

        he = stats.get('HasemanElstonEstimator') or {}
        cov_g = he.get('_cov_g')
        keys = he.get('_keys', [])
        if cov_g is not None and len(keys) == 2:
            d = np.sqrt(np.abs(np.diag(cov_g)))
            d[d < 1e-15] = 1.0
            rg_he = float(cov_g[0, 1] / (d[0] * d[1]))
        else:
            rg_he = np.nan

        ss = stats.get('SampleStatistics') or {}
        cov = ss.get('cov')
        ss_keys = ss.get('keys', [])
        if cov is not None and all(t in ss_keys for t in TRAIT_NAMES):
            i, j = ss_keys.index(TRAIT_NAMES[0]), ss_keys.index(TRAIT_NAMES[1])
            denom = np.sqrt(cov[i, i] * cov[j, j])
            rho_pheno = float(cov[i, j] / denom) if denom > 0 else np.nan
        else:
            rho_pheno = np.nan

        ms = stats.get('MatingStatistics') or {}
        spouse = ms.get('spouse_correlations', {})

        rows.append({
            'generation': gen,
            'rg_HE': rg_he,
            'rho_pheno': rho_pheno,
            'spouse_r_pheno_1': spouse.get('pheno_1', np.nan),
            'spouse_r_pheno_2': spouse.get('pheno_2', np.nan),
            'n_mating_pairs': ms.get('n_mating_pairs', np.nan),
        })
    return pd.DataFrame.from_records(rows)


def phenotypes_to_dataframe(sim: NSimulation) -> pd.DataFrame:
    """Convert the current generation's NPhenotypeArray to a DataFrame."""
    pheno = sim.phenotypes
    data = {key: pheno[key] for key in pheno.keys}
    df = pd.DataFrame(data)
    df.insert(0, 'iid', pheno.samples.iid)
    df.insert(1, 'fid', pheno.samples.fid)
    df.insert(2, 'sex', pheno.samples.sex)
    df.insert(3, 'generation', pheno.samples.generation)
    return df


if __name__ == '__main__':
    sim = build_sim()
    sim.run(N_GENERATIONS)

    summary = summarize(sim)
    print('Per-generation summary:')
    print(summary.to_string(index=False))

    print('\nFinal-generation phenotypes (head):')
    pheno_df = phenotypes_to_dataframe(sim)
    print(pheno_df.head().to_string(index=False))
