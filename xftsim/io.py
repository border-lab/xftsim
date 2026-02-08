"""I/O functions for xftsim data structures.

Provides serialization (save/load) for haplotypes, phenotypes, effects,
architectures, and full simulation checkpoints. Also provides import
functions for PLINK and sgkit datasets, and GRG loading.

Public API
----------
save_haplotypes_npz / load_haplotypes_npz
    Round-trip DenseHaplotypeArray to/from compressed .npz.
save_phenotypes_npz / load_phenotypes_npz
    Round-trip NPhenotypeArray to/from compressed .npz.
save_effects_npz / load_effects_npz
    Round-trip any EffectSpec subclass to/from compressed .npz.
save_architecture / load_architecture
    Round-trip Architecture to/from a directory (JSON + .npz).
save_simulation_checkpoint / load_simulation_checkpoint
    Round-trip full simulation state to/from a directory.
load_grg
    Load a GRG file as a GraphHaplotypeOperator.
read_plink1_as_pseudohaplotypes
    Import PLINK 1 binary files as DenseHaplotypeArray.
haplotypes_from_sgkit_dataset
    Import sgkit Dataset as DenseHaplotypeArray.
"""
import warnings
import json
import os
import numpy as np
import numba as nb
import pandas as pd
import dask.array as da
import nptyping as npt
import xarray as xr
from nptyping import NDArray, Int8, Int64, Float64, Bool, Shape, Float, Int
from typing import Any, Hashable, List, Iterable, Callable, Union, Dict
from functools import cached_property
import pandas_plink as pp
from dask.diagnostics import ProgressBar

import xftsim as xft


@nb.njit
def _genotypes_to_pseudo_haplotypes_3d(genotypes):
    """
    Converts genotype data to pseudo-haplotype 3D array.

    Parameters
    ----------
    genotypes : ndarray
        2D array of genotype data (n, m) with values 0, 1, 2.

    Returns
    -------
    ndarray
        3D array of pseudo-haplotype data (n, m, 2).
    """
    n, m = genotypes.shape
    haplotypes = np.zeros((n, m, 2), dtype=np.int8)
    for i in range(n):
        for j in range(m):
            g = genotypes[i, j]
            if g == 0:
                haplotypes[i, j, 0] = 0
                haplotypes[i, j, 1] = 0
            elif g == 1:
                # Randomly assign which haplotype gets the 1
                if np.random.random() < 0.5:
                    haplotypes[i, j, 0] = 1
                    haplotypes[i, j, 1] = 0
                else:
                    haplotypes[i, j, 0] = 0
                    haplotypes[i, j, 1] = 1
            elif g == 2:
                haplotypes[i, j, 0] = 1
                haplotypes[i, j, 1] = 1
    return haplotypes


def genotypes_to_pseudo_haplotypes(genotypes: np.ndarray) -> np.ndarray:
    """
    Converts genotype data to pseudo-haplotype 3D array.

    Parameters
    ----------
    genotypes : np.ndarray
        2D array of genotype data (n, m) with values 0, 1, 2.

    Returns
    -------
    np.ndarray
        3D array of pseudo-haplotype data (n, m, 2).
    """
    return _genotypes_to_pseudo_haplotypes_3d(genotypes.astype(np.int8))


def read_plink1_as_pseudohaplotypes(path: str,
                                    generation: int = 0) -> xft.struct.NHaplotypeArray:
    """
    Reads in PLINK 1 binary genotype data and returns a NHaplotypeArray object containing pseudo-haplotypes by
    randomly assigning haplotypes at heterozygous sites.

    Parameters
    ----------
    path : str
        The file path to the PLINK 1 binary genotype data.
    generation : int, optional
        Generation number. Default is 0.

    Returns
    -------
    xft.struct.NHaplotypeArray
        Pseudo-haplotype array. The "pseudo-" prefix refers to the fact that the
        plink bfile format doesn't track phase.

    Raises
    ------
    ValueError
        If the specified file path does not exist or is not in the expected format.
    """
    # Read genotype data
    bim, fam, bed = pp.read_plink(path)

    # bed is (m, n) in dask, transpose to (n, m)
    with ProgressBar():
        genotypes = bed.T.compute().astype(np.int8)

    n, m = genotypes.shape

    # Convert to 3D pseudo-haplotypes
    haplotypes_3d = genotypes_to_pseudo_haplotypes(genotypes)

    # Create variant IDs
    if np.all(bim.snp.values == '.'):
        vid = np.char.add(
            np.char.add(bim.chrom.values.astype(str), ':'),
            bim.pos.values.astype(str)
        )
    else:
        vid = bim.snp.values

    # Create variant metadata
    chrom = bim.chrom.values
    pos_bp = bim.pos.values
    pos_cM = bim.cm.values if not np.all(bim.cm.values == 0) else None
    zero_allele = bim.a0.values
    one_allele = bim.a1.values

    variants = xft.struct.VariantMeta(
        vid=vid,
        chrom=chrom,
        pos_bp=pos_bp,
        pos_cM=pos_cM,
        zero_allele=zero_allele,
        one_allele=one_allele,
    )

    # Create sample metadata
    iid = fam.iid.values.astype(str)
    fid = fam.fid.values.astype(str)
    # pandas_plink: gender 1=male, 2=female, 0=unknown
    # xftsim: sex 0=female, 1=male
    sex = (2 - fam.gender.values).astype(int)
    sex[fam.gender.values == 0] = 0  # unknown -> default to female

    samples = xft.struct.SampleMeta(iid=iid, fid=fid, sex=sex)

    return xft.struct.NHaplotypeArray(
        genotypes=haplotypes_3d,
        generation=generation,
        samples=samples,
        variants=variants,
    )


def haplotypes_from_sgkit_dataset(gdat: xr.Dataset,
                                  generation: int = 0) -> xft.struct.NHaplotypeArray:
    """Construct haplotype array from sgkit DataArray.
    Useful in conjuction with sgkit.io.vcf.vcf_to_zarr() and sgkit.load_dataset()

    Parameters
    ----------
    gdat : xr.Dataset
        Dataset generated by sgkit.load_dataset()
    generation : int, optional
        Generation number. Default is 0.

    Returns
    -------
    xft.struct.NHaplotypeArray
        Haplotype array.
    """
    # Genotypes are already in 3D format (samples, variants, ploidy)
    # sgkit uses (variants, samples, ploidy), so we need to transpose
    genotypes = gdat.call_genotype.values.transpose(1, 0, 2).astype(np.int8)

    # Create sample metadata
    iid = gdat.sample_id.values.astype(str)
    samples = xft.struct.SampleMeta(iid=iid)

    # Create variant metadata
    chrom = np.array(gdat.contigs)[gdat.variant_contig.values]
    pos_bp = gdat.variant_position.values
    vid = np.char.add(
        np.char.add(chrom.astype(str), ':'),
        pos_bp.astype(str)
    )

    # Get alleles if available
    alleles = gdat.variant_allele.values if 'variant_allele' in gdat else None
    zero_allele = alleles[:, 0].astype(str) if alleles is not None else None
    one_allele = alleles[:, 1].astype(str) if alleles is not None else None

    variants = xft.struct.VariantMeta(
        vid=vid,
        chrom=chrom,
        pos_bp=pos_bp,
        zero_allele=zero_allele,
        one_allele=one_allele,
    )

    return xft.struct.NHaplotypeArray(
        genotypes=genotypes,
        generation=generation,
        samples=samples,
        variants=variants,
    )


def save_haplotypes_npz(haplotypes: xft.struct.NHaplotypeArray,
                        path: str) -> None:
    """
    Save NHaplotypeArray to compressed numpy format.

    Parameters
    ----------
    haplotypes : xft.struct.NHaplotypeArray
        The haplotype data to save.
    path : str
        The path to save to (will add .npz extension if not present).
    """
    save_dict = {
        'genotypes': haplotypes.genotypes,
        'generation': np.array([haplotypes.generation]),
        # Sample metadata
        'sample_iid': haplotypes.samples.iid,
        'sample_fid': haplotypes.samples.fid,
        'sample_sex': haplotypes.samples.sex,
        # Variant metadata
        'variant_vid': haplotypes.variants.vid,
    }

    # Add optional variant metadata if present
    if haplotypes.variants.chrom is not None:
        save_dict['variant_chrom'] = haplotypes.variants.chrom
    if haplotypes.variants.pos_bp is not None:
        save_dict['variant_pos_bp'] = haplotypes.variants.pos_bp
    if haplotypes.variants.pos_cM is not None:
        save_dict['variant_pos_cM'] = haplotypes.variants.pos_cM
    if haplotypes.variants.af is not None:
        save_dict['variant_af'] = haplotypes.variants.af
    if haplotypes.variants.zero_allele is not None:
        save_dict['variant_zero_allele'] = haplotypes.variants.zero_allele
    if haplotypes.variants.one_allele is not None:
        save_dict['variant_one_allele'] = haplotypes.variants.one_allele

    np.savez_compressed(path, **save_dict)


def load_haplotypes_npz(path: str) -> xft.struct.NHaplotypeArray:
    """
    Load NHaplotypeArray from compressed numpy format.

    Parameters
    ----------
    path : str
        The path to load from.

    Returns
    -------
    xft.struct.NHaplotypeArray
        The loaded haplotype array.
    """
    data = np.load(path, allow_pickle=True)

    # Load sample metadata
    samples = xft.struct.SampleMeta(
        iid=data['sample_iid'],
        fid=data['sample_fid'],
        sex=data['sample_sex'],
    )

    # Load variant metadata
    variants = xft.struct.VariantMeta(
        vid=data['variant_vid'],
        chrom=data['variant_chrom'] if 'variant_chrom' in data else None,
        pos_bp=data['variant_pos_bp'] if 'variant_pos_bp' in data else None,
        pos_cM=data['variant_pos_cM'] if 'variant_pos_cM' in data else None,
        af=data['variant_af'] if 'variant_af' in data else None,
        zero_allele=data['variant_zero_allele'] if 'variant_zero_allele' in data else None,
        one_allele=data['variant_one_allele'] if 'variant_one_allele' in data else None,
    )

    return xft.struct.DenseHaplotypeArray(
        genotypes=data['genotypes'],
        generation=int(data['generation'][0]),
        samples=samples,
        variants=variants,
    )


def load_grg(path: str, generation: int = 0,
             bim_path: str = None) -> "xft.struct.GraphHaplotypeOperator":
    """
    Load a GRG file and return a GraphHaplotypeOperator.

    Parameters
    ----------
    path : str
        Path to the .grg file.
    generation : int, optional
        Generation number. Default 0.
    bim_path : str, optional
        Path to a PLINK .bim file for chromosome/allele metadata.
        If None, variant metadata is extracted from the GRG itself.

    Returns
    -------
    xft.struct.GraphHaplotypeOperator
    """
    import pygrgl
    grg = pygrgl.load_immutable_grg(path)

    # Extract sample metadata from GRG
    n = grg.num_individuals
    if grg.has_individual_ids:
        iid = np.array([grg.get_individual_id(i) for i in range(n)])
    else:
        iid = np.arange(n, dtype=np.int64)
    samples = xft.struct.SampleMeta(iid=iid, generation=generation)

    # Variant metadata: from BIM if provided, else from GRG
    variants = None
    if bim_path is not None:
        bim = pd.read_csv(bim_path, sep='\t', header=None,
                          names=['chrom', 'vid', 'cm', 'pos_bp', 'a1', 'a0'])
        m_bim = len(bim)
        m_grg = grg.num_mutations
        if m_bim != m_grg:
            raise ValueError(
                f"BIM has {m_bim} variants but GRG has {m_grg} mutations"
            )
        pos_cM = bim['cm'].values
        if np.all(pos_cM == 0):
            pos_cM = None
        variants = xft.struct.VariantMeta(
            vid=bim['vid'].values.astype(str),
            chrom=bim['chrom'].values,
            pos_bp=bim['pos_bp'].values,
            pos_cM=pos_cM,
            zero_allele=bim['a0'].values.astype(str),
            one_allele=bim['a1'].values.astype(str),
        )

    return xft.struct.GraphHaplotypeOperator(
        grg, generation=generation, samples=samples, variants=variants,
    )


def save_phenotypes_npz(phenotypes: xft.struct.NPhenotypeArray,
                         path: str) -> None:
    """
    Save NPhenotypeArray to compressed numpy format.

    Parameters
    ----------
    phenotypes : xft.struct.NPhenotypeArray
        The phenotype data to save.
    path : str
        The path to save to (will add .npz extension if not present).
    """
    save_dict = {
        'sample_iid': phenotypes.samples.iid,
        'sample_fid': phenotypes.samples.fid,
        'sample_sex': phenotypes.samples.sex,
        'pheno_keys': np.array(list(phenotypes.keys)),
    }
    for key in phenotypes.keys:
        save_dict[f'pheno_{key}'] = phenotypes[key]
    np.savez_compressed(path, **save_dict)


def load_phenotypes_npz(path: str) -> xft.struct.NPhenotypeArray:
    """
    Load NPhenotypeArray from compressed numpy format.

    Parameters
    ----------
    path : str
        The path to load from.

    Returns
    -------
    xft.struct.NPhenotypeArray
    """
    data = np.load(path, allow_pickle=True)
    samples = xft.struct.SampleMeta(
        iid=data['sample_iid'],
        fid=data['sample_fid'],
        sex=data['sample_sex'],
    )
    values = {}
    for key in data['pheno_keys']:
        values[str(key)] = data[f'pheno_{key}']
    return xft.struct.NPhenotypeArray(samples=samples, values=values)


def save_effects_npz(effects: "xft.neffect.EffectSpec", path: str) -> None:
    """
    Save an EffectSpec (any subclass) to compressed numpy format.

    Parameters
    ----------
    effects : xft.neffect.EffectSpec
        The effect specification to save.
    path : str
        The path to save to.
    """
    save_dict = {
        'effects': effects.effects,
        'standardized': np.array([effects.standardized]),
        'variant_mask': effects.variant_mask,
        'class_name': np.array([type(effects).__name__]),
    }
    np.savez_compressed(path, **save_dict)


def load_effects_npz(path: str) -> "xft.neffect.EffectSpec":
    """
    Load an EffectSpec from compressed numpy format.

    Parameters
    ----------
    path : str
        The path to load from.

    Returns
    -------
    xft.neffect.EffectSpec
        The loaded effect specification (concrete subclass).
    """
    from xftsim.neffect import AdditiveEffects, MultivariateEffects, SparseEffects

    data = np.load(path, allow_pickle=True)
    effects = data['effects']
    standardized = bool(data['standardized'][0])
    variant_mask = data['variant_mask']
    class_name = str(data['class_name'][0])

    cls_map = {
        'AdditiveEffects': AdditiveEffects,
        'MultivariateEffects': MultivariateEffects,
        'SparseEffects': SparseEffects,
    }
    if class_name not in cls_map:
        raise ValueError(f"Unknown EffectSpec class: {class_name}")

    return cls_map[class_name](
        effects=effects, standardized=standardized, variant_mask=variant_mask,
    )


def save_architecture(arch: "xft.narch.Architecture", dir_path: str) -> None:
    """
    Save an Architecture to a directory (JSON metadata + effect .npz files).

    Parameters
    ----------
    arch : xft.narch.Architecture
        The architecture to save.
    dir_path : str
        Directory path (created if it doesn't exist).
    """
    from xftsim.narch import (
        GeneticComponent, MVGeneticComponent, HaplotypeGeneticComponent,
        NoiseComponent, CNoiseComponent, AggregationComponent,
        MotherComponent, FatherComponent, ParentComponent,
        _SiblingComponent, _ParentalComponent,
    )

    os.makedirs(dir_path, exist_ok=True)
    node_specs = []
    effect_idx = 0

    for node in arch._nodes:
        comp = node.component
        spec = {
            'outputs': node.outputs,
            'inputs': node.inputs,
            'grouping': node.grouping,
            'component_type': type(comp).__name__,
        }

        if isinstance(comp, HaplotypeGeneticComponent):
            eff_name = f'effect_{effect_idx}'
            save_effects_npz(comp.effects, os.path.join(dir_path, f'{eff_name}.npz'))
            spec['effect_file'] = f'{eff_name}.npz'
            spec['haplotype'] = comp.haplotype
            effect_idx += 1
        elif isinstance(comp, (GeneticComponent, MVGeneticComponent)):
            eff_name = f'effect_{effect_idx}'
            save_effects_npz(comp.effects, os.path.join(dir_path, f'{eff_name}.npz'))
            spec['effect_file'] = f'{eff_name}.npz'
            effect_idx += 1
        elif isinstance(comp, NoiseComponent):
            spec['variance'] = comp.variance
        elif isinstance(comp, CNoiseComponent):
            spec['cov'] = comp.cov.tolist()
        elif isinstance(comp, AggregationComponent):
            spec['expression'] = comp.expression
        elif isinstance(comp, _ParentalComponent):
            spec['phenotype_name'] = comp.phenotype_name
        elif isinstance(comp, _SiblingComponent):
            spec['source_name'] = comp.source_name

        node_specs.append(spec)

    with open(os.path.join(dir_path, 'architecture.json'), 'w') as f:
        json.dump(node_specs, f, indent=2)


def load_architecture(dir_path: str) -> "xft.narch.Architecture":
    """
    Load an Architecture from a directory.

    Parameters
    ----------
    dir_path : str
        Directory containing architecture.json and effect .npz files.

    Returns
    -------
    xft.narch.Architecture
    """
    from xftsim.narch import (
        Architecture, GeneticComponent, MVGeneticComponent,
        HaplotypeGeneticComponent, NoiseComponent, CNoiseComponent,
        AggregationComponent, MotherComponent, FatherComponent,
        ParentComponent, BUILTINS, _SIBLING_COMPONENTS,
    )

    with open(os.path.join(dir_path, 'architecture.json'), 'r') as f:
        node_specs = json.load(f)

    arch = Architecture()
    comp_map = {
        'GeneticComponent': lambda s: GeneticComponent(
            load_effects_npz(os.path.join(dir_path, s['effect_file']))
        ),
        'MVGeneticComponent': lambda s: MVGeneticComponent(
            load_effects_npz(os.path.join(dir_path, s['effect_file']))
        ),
        'HaplotypeGeneticComponent': lambda s: HaplotypeGeneticComponent(
            load_effects_npz(os.path.join(dir_path, s['effect_file'])),
            haplotype=s['haplotype'],
        ),
        'NoiseComponent': lambda s: NoiseComponent(variance=s['variance']),
        'CNoiseComponent': lambda s: CNoiseComponent(cov=np.array(s['cov'])),
        'AggregationComponent': lambda s: AggregationComponent(expression=s['expression']),
        'MotherComponent': lambda s: MotherComponent(phenotype_name=s['phenotype_name']),
        'FatherComponent': lambda s: FatherComponent(phenotype_name=s['phenotype_name']),
        'ParentComponent': lambda s: ParentComponent(phenotype_name=s['phenotype_name']),
    }
    # Add sibling component constructors
    for sib_name, sib_cls in _SIBLING_COMPONENTS.items():
        cls_name = sib_cls.__name__
        comp_map[cls_name] = lambda s, c=sib_cls: c(source_name=s['source_name'])

    for spec in node_specs:
        comp_type = spec['component_type']
        if comp_type not in comp_map:
            raise ValueError(f"Unknown component type: {comp_type}")
        comp = comp_map[comp_type](spec)
        arch.add(
            outputs=spec['outputs'],
            component=comp,
            inputs=spec.get('inputs', []),
            grouping=spec.get('grouping'),
        )

    return arch


def _serialize_mating_regime(regime) -> dict:
    """Serialize a mating regime to a JSON-compatible dict."""
    from xftsim.nmate import RandomMating, LinearAssortativeMating
    if isinstance(regime, LinearAssortativeMating):
        return {
            'type': 'LinearAssortativeMating',
            'component_names': regime.component_names,
            'r': regime.r,
            'offspring_per_pair': regime.offspring_per_pair,
        }
    elif isinstance(regime, RandomMating):
        return {
            'type': 'RandomMating',
            'offspring_per_pair': regime.offspring_per_pair,
        }
    else:
        return {'type': type(regime).__name__}


def _deserialize_mating_regime(config: dict):
    """Deserialize a mating regime from a dict."""
    from xftsim.nmate import RandomMating, LinearAssortativeMating
    mtype = config['type']
    if mtype == 'RandomMating':
        return RandomMating(offspring_per_pair=config['offspring_per_pair'])
    elif mtype == 'LinearAssortativeMating':
        return LinearAssortativeMating(
            component_names=config['component_names'],
            r=config['r'],
            offspring_per_pair=config['offspring_per_pair'],
        )
    else:
        raise ValueError(f"Unknown mating regime type: {mtype}")


def save_simulation_checkpoint(sim: "xft.nsim.NSimulation",
                               dir_path: str) -> None:
    """
    Save a simulation checkpoint to a directory.

    Saves haplotype history (dense only), phenotype history, pedigree history,
    architecture, generation counter, and RNG state.

    Parameters
    ----------
    sim : xft.nsim.NSimulation
        The simulation to checkpoint.
    dir_path : str
        Directory path (created if it doesn't exist).
    """
    os.makedirs(dir_path, exist_ok=True)

    # Save architecture
    save_architecture(sim.architecture, os.path.join(dir_path, 'architecture'))

    # Save metadata (including mating regime config)
    mating_config = _serialize_mating_regime(sim.mating_regime)
    meta = {
        'generation': sim.generation,
        'retain_haplotypes': sim.retain_haplotypes,
        'retain_phenotypes': sim.retain_phenotypes,
        'mating': mating_config,
    }
    with open(os.path.join(dir_path, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # Save recombination map
    rmap = sim.recombination_map
    np.savez_compressed(
        os.path.join(dir_path, 'recombination_map.npz'),
        probabilities=rmap._probabilities,
        vid=rmap.vid,
        chrom=rmap.chrom,
    )

    # Save RNG state
    rng_state = sim.rng.get_state()
    np.savez(os.path.join(dir_path, 'rng_state.npz'),
             state_key=rng_state[1], pos=np.array([rng_state[2]]),
             has_gauss=np.array([rng_state[3]]),
             cached_gaussian=np.array([rng_state[4]]))

    # Save haplotype history (dense only — GRG founders are not checkpointed)
    hap_dir = os.path.join(dir_path, 'haplotypes')
    os.makedirs(hap_dir, exist_ok=True)
    for gen, hap in sim.haplotype_history.items():
        if isinstance(hap, xft.struct.DenseHaplotypeArray):
            save_haplotypes_npz(hap, os.path.join(hap_dir, f'gen_{gen}.npz'))
        elif isinstance(hap, xft.struct.GraphHaplotypeOperator):
            # Materialize GRG to dense for checkpointing
            save_haplotypes_npz(hap.to_dense(), os.path.join(hap_dir, f'gen_{gen}.npz'))

    # Save phenotype history
    pheno_dir = os.path.join(dir_path, 'phenotypes')
    os.makedirs(pheno_dir, exist_ok=True)
    for gen, pheno in sim.phenotype_history.items():
        save_phenotypes_npz(pheno, os.path.join(pheno_dir, f'gen_{gen}.npz'))

    # Save pedigree history
    ped_dir = os.path.join(dir_path, 'pedigrees')
    os.makedirs(ped_dir, exist_ok=True)
    for gen, ped in sim.pedigree_history.items():
        np.savez_compressed(
            os.path.join(ped_dir, f'gen_{gen}.npz'),
            maternal_idx=ped.maternal_idx,
            paternal_idx=ped.paternal_idx,
            parent_n=np.array([ped.parent_n]),
            offspring_iid=ped.offspring_samples.iid,
            offspring_fid=ped.offspring_samples.fid,
            offspring_sex=ped.offspring_samples.sex,
        )

    # Save generation keys for each history
    np.savez(os.path.join(dir_path, 'history_keys.npz'),
             haplotype_gens=np.array(list(sim.haplotype_history.keys())),
             phenotype_gens=np.array(list(sim.phenotype_history.keys())),
             pedigree_gens=np.array(list(sim.pedigree_history.keys())))


def load_simulation_checkpoint(dir_path: str) -> dict:
    """
    Load a simulation checkpoint from a directory.

    Returns a dict with all saved state — use this to inspect results
    or to reconstruct a simulation for continued execution.

    Parameters
    ----------
    dir_path : str
        Directory containing checkpoint files.

    Returns
    -------
    dict
        Keys: architecture, generation, retain_haplotypes, retain_phenotypes,
        rng_state, haplotype_history, phenotype_history, pedigree_history.
    """
    # Load metadata
    with open(os.path.join(dir_path, 'meta.json'), 'r') as f:
        meta = json.load(f)

    # Load architecture
    architecture = load_architecture(os.path.join(dir_path, 'architecture'))

    # Load RNG state
    rng_data = np.load(os.path.join(dir_path, 'rng_state.npz'))
    rng = np.random.RandomState()
    rng.set_state((
        'MT19937',
        rng_data['state_key'],
        int(rng_data['pos'][0]),
        int(rng_data['has_gauss'][0]),
        float(rng_data['cached_gaussian'][0]),
    ))

    # Load history keys
    keys_data = np.load(os.path.join(dir_path, 'history_keys.npz'))

    # Load haplotype history
    haplotype_history = {}
    hap_dir = os.path.join(dir_path, 'haplotypes')
    for gen in keys_data['haplotype_gens']:
        gen = int(gen)
        haplotype_history[gen] = load_haplotypes_npz(
            os.path.join(hap_dir, f'gen_{gen}.npz')
        )

    # Load phenotype history
    phenotype_history = {}
    pheno_dir = os.path.join(dir_path, 'phenotypes')
    for gen in keys_data['phenotype_gens']:
        gen = int(gen)
        phenotype_history[gen] = load_phenotypes_npz(
            os.path.join(pheno_dir, f'gen_{gen}.npz')
        )

    # Load pedigree history
    pedigree_history = {}
    ped_dir = os.path.join(dir_path, 'pedigrees')
    for gen in keys_data['pedigree_gens']:
        gen = int(gen)
        ped_data = np.load(os.path.join(ped_dir, f'gen_{gen}.npz'))
        samples = xft.struct.SampleMeta(
            iid=ped_data['offspring_iid'],
            fid=ped_data['offspring_fid'],
            sex=ped_data['offspring_sex'],
        )
        pedigree_history[gen] = xft.struct.PedigreeArray(
            offspring_samples=samples,
            maternal_idx=ped_data['maternal_idx'],
            paternal_idx=ped_data['paternal_idx'],
            parent_n=int(ped_data['parent_n'][0]),
        )

    # Load recombination map
    recombination_map = None
    rmap_path = os.path.join(dir_path, 'recombination_map.npz')
    if os.path.exists(rmap_path):
        from xftsim.reproduce import RecombinationMap
        rmap_data = np.load(rmap_path, allow_pickle=True)
        recombination_map = RecombinationMap(
            p=rmap_data['probabilities'],
            m=len(rmap_data['probabilities']),
            vid=rmap_data['vid'],
            chrom=rmap_data['chrom'],
        )

    # Load mating regime
    mating_regime = None
    if 'mating' in meta:
        mating_regime = _deserialize_mating_regime(meta['mating'])

    return {
        'architecture': architecture,
        'generation': meta['generation'],
        'retain_haplotypes': meta['retain_haplotypes'],
        'retain_phenotypes': meta['retain_phenotypes'],
        'rng': rng,
        'haplotype_history': haplotype_history,
        'phenotype_history': phenotype_history,
        'pedigree_history': pedigree_history,
        'recombination_map': recombination_map,
        'mating_regime': mating_regime,
    }


# Legacy functions for XarrayHaplotypeArray compatibility

def plink1_variant_index(ppxr: xr.DataArray) -> xft.index.DiploidVariantIndex:
    """
    Create a DiploidVariantIndex object from a plink file DataArray generated by pandas_plink.

    Parameters
    ----------
    ppxr : xr.DataArray
        An xarray DataArray representing a plink file.

    Returns
    -------
    xft.index.DiploidVariantIndex
        A DiploidVariantIndex object.
    """
    if np.all(ppxr.snp.values == '.'):
        vid = np.char.add(np.char.add(ppxr.chrom.values.astype(str), ':'), ppxr.pos.values.astype(str))
    else:
        vid = ppxr.snp.values
    if np.all(ppxr.cm.values == 0):
        cm = np.full(ppxr.cm.shape, fill_value=np.NaN)
    else:
        cm = ppxr.cm
    return xft.index.DiploidVariantIndex(
        vid=vid,
        chrom=ppxr.chrom.values,
        zero_allele=ppxr.a0.values,
        one_allele=ppxr.a1.values,
        pos_bp=ppxr.pos,
        pos_cM=cm,
    )


def plink1_sample_index(ppxr: xr.DataArray,
                        generation: int = 0) -> xft.index.SampleIndex:
    """
    Create a SampleIndex object from a plink file DataArray generated by pandas_plink.

    Parameters
    ----------
    ppxr : xr.DataArray
        An xarray DataArray representing a plink file.
    generation : int, optional
        The generation of the individuals, by default 0.

    Returns
    -------
    xft.index.SampleIndex
        A SampleIndex object.
    """
    return xft.index.SampleIndex(
        iid=ppxr.iid.values.astype(str),
        fid=ppxr.fid.values.astype(str),
        sex=2 - ppxr.gender.values.astype(int),
        generation=generation,
    )
