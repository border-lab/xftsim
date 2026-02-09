Migration Guide: Old API to New API
====================================

This guide helps existing xftsim users migrate from the legacy xarray-based API
(now located under ``xftsim/legacy/``) to the new numpy-backed API introduced on
the ``ajay`` branch. The new API is a ground-up rewrite with significant
simplifications, better performance, and a formula DSL for architecture
definition.

.. contents:: Table of Contents
   :local:
   :depth: 2


Why the Rewrite
---------------

The legacy API was built on xarray and pandas with complex multi-level indexing
(``ComponentIndex``, ``SampleIndex``, ``HaploidVariantIndex``). While flexible,
this design had several drawbacks:

- **Performance**: xarray label-based indexing added overhead to the inner
  simulation loop. Every phenotype read/write went through multi-level
  coordinate lookups.
- **Complexity**: ``ComponentIndex`` with its ``(phenotype_name, component_name,
  vorigin_relative)`` tuples was difficult to construct and debug.
- **Verbosity**: Defining even a simple architecture required manually
  constructing index objects, effect objects with variant indexers, and
  component objects with input/output indices.
- **Dependency weight**: xarray, dask, and nptyping were required even for
  simple simulations.

The new API replaces xarray with plain numpy arrays and a thin ``dict``-based
phenotype container, uses a formula DSL for architecture definition, and
implements the HaplotypeOperator abstract base class to support both dense
arrays and graph-based representations (GRG).


Class and Module Name Mapping
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 40 20

   * - Old API (``xftsim.legacy.*``)
     - New API (``xftsim.*``)
     - Module

   * - ``xr.DataArray`` (haplotypes)
     - ``DenseHaplotypeArray`` / ``GraphHaplotypeOperator``
     - ``struct``

   * - ``xr.DataArray`` (phenotypes)
     - ``NPhenotypeArray``
     - ``struct``

   * - ``SampleIndex``
     - ``SampleMeta``
     - ``struct``

   * - ``DiploidVariantIndex`` / ``HaploidVariantIndex``
     - ``VariantMeta``
     - ``struct``

   * - ``ComponentIndex``
     - Flat ``dict[str, np.ndarray]`` inside ``NPhenotypeArray``
     - ``struct``

   * - ``effect.AdditiveEffects``
     - ``neffect.AdditiveEffects``
     - ``neffect``

   * - ``effect.EffectSizeDistribution``
     - ``neffect.AdditiveEffects.from_h2()`` / ``SparseEffects.from_h2()``
     - ``neffect``

   * - (no multivariate effects)
     - ``neffect.MultivariateEffects``
     - ``neffect``

   * - ``arch.ArchitectureComponent``
     - ``narch.ArchComponent`` (ABC)
     - ``narch``

   * - ``arch.AdditiveGeneticComponent``
     - ``narch.GeneticComponent``
     - ``narch``

   * - ``arch.AdditiveNoiseComponent``
     - ``narch.NoiseComponent``
     - ``narch``

   * - ``arch.CorrelatedNoiseComponent``
     - ``narch.CNoiseComponent``
     - ``narch``

   * - ``arch.SumAllTransformation``
     - ``narch.AggregationComponent``
     - ``narch``

   * - ``arch.LinearTransformationComponent``
     - ``narch.AggregationComponent`` (arithmetic expressions)
     - ``narch``

   * - ``arch.Architecture``
     - ``narch.Architecture`` (with formula DSL)
     - ``narch``

   * - ``arch.GCTA_Architecture``
     - No direct equivalent; use ``Architecture`` + formula
     - ``narch``

   * - ``mate.MatingRegime``
     - (removed -- use concrete classes directly)
     - --

   * - ``mate.RandomMatingRegime``
     - ``nmate.RandomMating``
     - ``nmate``

   * - ``mate.LinearAssortativeMatingRegime``
     - ``nmate.LinearAssortativeMating``
     - ``nmate``

   * - ``mate.MateAssignment``
     - ``nmate.NMateAssignment``
     - ``nmate``

   * - ``sim.Simulation``
     - ``nsim.NSimulation``
     - ``nsim``

   * - ``stats.*`` (various)
     - ``nstats.SampleStatistics``, ``nstats.HasemanElstonEstimator``, etc.
     - ``nstats``

   * - ``proc.*`` (post-processors)
     - ``callbacks`` parameter on ``NSimulation``
     - ``nsim``

   * - (no filters)
     - ``nfilter.TrioFilter``, ``SibPairFilter``, ``AscertainmentFilter``, etc.
     - ``nfilter``

   * - (no formula parser)
     - ``parser.parse_formula``
     - ``parser``


Key Conceptual Changes
----------------------

xarray to numpy
~~~~~~~~~~~~~~~~

The most fundamental change is that **xarray DataArrays are gone**. Phenotypes
are stored in ``NPhenotypeArray``, which is a thin wrapper around
``dict[str, np.ndarray]`` with attached ``SampleMeta``. Haplotypes are stored in
``DenseHaplotypeArray`` (a 3D numpy array of shape ``(n, m, 2)``).

**Old**::

    # Phenotypes were xr.DataArray with multi-level coordinates
    phenotypes.loc[:, component_index.unique_identifier] = values
    Y = phenotypes.xft[sample_index, component_index]

**New**::

    # Phenotypes are a flat dict of named arrays
    phenotypes['Y.G'] = values  # set
    Y = phenotypes['Y']         # get -- returns np.ndarray of shape (n,)

ComponentIndex to flat names
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The old ``ComponentIndex`` with its three-level
``(phenotype_name, component_name, vorigin_relative)`` addressing is replaced
by simple string keys. Names like ``'height.G'`` or ``'Y.E'`` are just
conventions -- the dot is not parsed or meaningful to the system.

**Old**::

    cindex = xft.index.ComponentIndex.from_product(
        phenotype_name=['height'],
        component_name=['addGenetic'],
        vorigin_relative=[-1],
    )

**New**::

    # Just use a string name. No index construction needed.
    arch.add('height.G', GeneticComponent(eff))

Effect objects
~~~~~~~~~~~~~~

In the old API, ``AdditiveEffects`` required a ``variant_indexer`` (to know
about allele frequencies for standardization) and a ``component_indexer``
(to know which phenotype columns to write to). In the new API, effect objects
are simple containers of numpy arrays. Standardization is handled by the
``HaplotypeOperator`` at matvec time.

**Old**::

    beta = xft.effect.AdditiveEffects(
        beta=effect_array,
        variant_indexer=haploid_variant_index,
        component_indexer=component_index,
        standardized=True,
        scaled=True,
    )

**New**::

    eff = AdditiveEffects.from_h2(h2=0.5, m=200, seed=42)
    # or
    eff = AdditiveEffects.from_array(effect_array, standardized=True)

Architecture: component list to DAG with formula DSL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The old ``Architecture`` took a list of ``ArchitectureComponent`` objects and
executed them in order, with each component reading/writing to specific
``ComponentIndex`` slices of the xarray phenotype DataArray.

The new ``Architecture`` is a DAG of ``ArchNode`` objects that are
topologically sorted. You can build it either programmatically or via the
formula DSL.

**Old**::

    arch = xft.arch.Architecture(components=[
        AdditiveGeneticComponent(beta=effects),
        AdditiveNoiseComponent(variances=[0.5], phenotype_name=['height']),
        SumAllTransformation(input_cindex=sum_input_cindex),
    ])

**New (formula DSL)**::

    arch = Architecture(
        formula="""
        Y.G ~ genetic(eff)
        Y.E ~ noise(0.5)
        Y ~ Y.G + Y.E
        """,
        effects={'eff': eff},
    )

**New (programmatic)**::

    arch = Architecture()
    arch.add('Y.G', GeneticComponent(eff))
    arch.add('Y.E', NoiseComponent(0.5))
    arch.add('Y', AggregationComponent('Y.G + Y.E'))

Simulation lifecycle
~~~~~~~~~~~~~~~~~~~~

The old ``Simulation`` started at ``generation=-1`` and called
``run_generation()`` which internally called ``increment_generation()``,
``reproduce()``, ``compute_phenotypes()``, ``mate()``, etc. Generation 0
was special-cased with founder initialization.

The new ``NSimulation`` starts at ``generation=0`` and ``run(n)`` simulates
generations 0 through n-1. Generation 0 computes phenotypes from founder
haplotypes directly. Post-processors are replaced by callbacks.

**Old**::

    sim = xft.sim.Simulation(
        founder_haplotypes=haplotypes,
        mating_regime=RandomMatingRegime(...),
        recombination_map=recomb_map,
        architecture=arch,
        statistics=[...],
        post_processors=[...],
    )
    sim.run(n_generations=5)

**New**::

    sim = NSimulation(
        founder_haplotypes=hap,
        architecture=arch,
        mating_regime=RandomMating(offspring_per_pair=2),
        recombination_map=recomb_map,
        statistics=[SampleStatistics()],
        callbacks=[my_callback_fn],  # replaces post_processors
        filters={'trio': TrioFilter()},
        seed=42,
    )
    sim.run(n_generations=5)


Code Migration Examples
-----------------------

Creating founder haplotypes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Old (xarray-based DiploidArray)**::

    import xftsim as xft

    # Generate from allele frequencies
    variant_index = xft.index.DiploidVariantIndex(
        vid=vid_array,
        chrom=chrom_array,
        zero_allele=a0,
        one_allele=a1,
        pos_bp=pos,
        pos_cM=cm,
    )
    # Complex xarray DataArray construction
    haplotypes = xft.founders.founder_haplotypes(
        n=1000,
        variant_index=variant_index,
    )

**New (DenseHaplotypeArray)**::

    from xftsim.founders import founder_haplotypes_uniform_AFs, founder_haplotypes_from_AFs
    import numpy as np

    # Quick generation from uniform allele frequencies
    hap = founder_haplotypes_uniform_AFs(n=1000, m=200)

    # From specified allele frequencies
    afs = np.random.uniform(0.1, 0.9, 200)
    hap = founder_haplotypes_from_AFs(n=1000, afs=afs)

    # From PLINK files
    from xftsim.io import read_plink1_as_pseudohaplotypes
    hap = read_plink1_as_pseudohaplotypes('/path/to/plink_prefix')

    # From sgkit/VCF
    from xftsim.io import haplotypes_from_sgkit_dataset
    import sgkit
    ds = sgkit.load_dataset('/path/to/zarr')
    hap = haplotypes_from_sgkit_dataset(ds)

    # From GRG (graph-based)
    from xftsim.io import load_grg
    hap = load_grg('/path/to/file.grg')

Defining effects
~~~~~~~~~~~~~~~~

**Old (EffectSizeDistribution + AdditiveEffects)**::

    import xftsim as xft

    # Required: construct variant and component indexers first
    variant_indexer = xft.index.HaploidVariantIndex(
        vid=vid, chrom=chrom, zero_allele=a0, one_allele=a1,
        pos_bp=pos, pos_cM=cm, af=afs,
    )
    component_indexer = xft.index.ComponentIndex.from_product(
        phenotype_name=['height'],
        component_name=['addGenetic'],
        vorigin_relative=[-1],
    )
    # Then construct effects with those indexers
    effects = xft.effect.AdditiveEffects(
        beta=beta_array,
        variant_indexer=variant_indexer,
        component_indexer=component_indexer,
        standardized=True,
        scaled=True,
    )

**New (AdditiveEffects, MultivariateEffects, SparseEffects)**::

    from xftsim.neffect import AdditiveEffects, MultivariateEffects, SparseEffects

    # Generate effects targeting h2=0.5
    eff = AdditiveEffects.from_h2(h2=0.5, m=200, seed=42)

    # From a known effect vector
    eff = AdditiveEffects.from_array(beta_array, standardized=True)

    # Multivariate effects for 2 correlated traits
    mv_eff = MultivariateEffects.from_h2_rg(
        h2=[0.5, 0.3], rg=0.5, m=200, seed=42
    )

    # Sparse effects (only 50 of 200 variants causal)
    sp_eff = SparseEffects.from_h2(h2=0.5, m=200, k_causal=50, seed=42)

Building architectures
~~~~~~~~~~~~~~~~~~~~~~

**Old (manual component construction)**::

    import xftsim as xft
    import numpy as np

    # 1. Create effect sizes
    variant_indexer = xft.index.HaploidVariantIndex(...)
    component_indexer_g = xft.index.ComponentIndex.from_product(
        phenotype_name=['height'], component_name=['addGenetic'], vorigin_relative=[-1]
    )
    effects = xft.effect.AdditiveEffects(
        beta=beta, variant_indexer=variant_indexer,
        component_indexer=component_indexer_g,
    )

    # 2. Create noise component (requires its own ComponentIndex)
    component_indexer_e = xft.index.ComponentIndex.from_product(
        phenotype_name=['height'], component_name=['noise'], vorigin_relative=[-1]
    )
    noise = xft.arch.AdditiveNoiseComponent(
        variances=[0.5], component_index=component_indexer_e
    )

    # 3. Create sum transformation (requires merging input indices)
    sum_input = xft.index.XftIndex.reduce_merge([component_indexer_g, component_indexer_e])
    sum_component = xft.arch.SumAllTransformation(input_cindex=sum_input)

    # 4. Assemble architecture
    arch = xft.arch.Architecture(
        components=[
            xft.arch.AdditiveGeneticComponent(beta=effects),
            noise,
            sum_component,
        ]
    )

**New (formula DSL -- recommended)**::

    from xftsim.neffect import AdditiveEffects
    from xftsim.narch import Architecture

    eff = AdditiveEffects.from_h2(h2=0.5, m=200, seed=42)

    arch = Architecture(
        formula="""
        Y.G ~ genetic(eff)
        Y.E ~ noise(0.5)
        Y ~ Y.G + Y.E
        """,
        effects={'eff': eff},
    )

**New (programmatic)**::

    from xftsim.narch import (
        Architecture, GeneticComponent, NoiseComponent, AggregationComponent
    )
    from xftsim.neffect import AdditiveEffects

    eff = AdditiveEffects.from_h2(h2=0.5, m=200, seed=42)

    arch = Architecture()
    arch.add('Y.G', GeneticComponent(eff))
    arch.add('Y.E', NoiseComponent(0.5))
    arch.add('Y', AggregationComponent('Y.G + Y.E'))

Vertical transmission
~~~~~~~~~~~~~~~~~~~~~

**Old**::

    # Required: MaternalVerticalComponent or LinearTransformationComponent
    # with carefully constructed input/output ComponentIndex objects
    # that reference parental phenotype components via vorigin_relative
    vt_component = xft.arch.LinearTransformationComponent(
        input_cindex=parent_cindex,
        output_cindex=vt_output_cindex,
        coefficient_matrix=np.eye(1),
    )

**New**::

    arch = Architecture(
        formula="""
        Y.G ~ genetic(eff)
        Y.E ~ noise(0.3)
        Y.VT ~ parent(Y, founder=noise(0.2))
        Y ~ Y.G + Y.E + Y.VT
        """,
        effects={'eff': eff},
    )

The ``parent(Y)`` component looks up the midparent phenotype ``Y`` from the
previous generation. The ``founder=noise(0.2)`` provides a fallback for
generation 0 when no parents exist. You can also use ``mother(Y)`` or
``father(Y)`` for sex-specific vertical transmission.

Running simulations
~~~~~~~~~~~~~~~~~~~

**Old**::

    sim = xft.sim.Simulation(
        founder_haplotypes=haplotypes,          # xr.DataArray
        mating_regime=xft.mate.RandomMatingRegime(
            offspring_per_pair=xft.utils.ConstantCount(2),
        ),
        recombination_map=recomb_map,
        architecture=arch,
        statistics=[some_stat],
        post_processors=[some_proc],
    )
    sim.run(n_generations=5)

    # Access results
    phenotypes = sim.phenotype_store[generation]   # xr.DataArray
    haplotypes = sim.haplotype_store[generation]   # xr.DataArray

**New**::

    from xftsim.nsim import NSimulation
    from xftsim.nmate import RandomMating
    from xftsim.reproduce import RecombinationMap
    from xftsim.nstats import SampleStatistics
    from xftsim.nfilter import TrioFilter

    sim = NSimulation(
        founder_haplotypes=hap,                 # DenseHaplotypeArray
        architecture=arch,
        mating_regime=RandomMating(offspring_per_pair=2),
        recombination_map=RecombinationMap.constant_map(m=200, p=0.5),
        retain_haplotypes=2,
        retain_phenotypes=5,
        statistics=[SampleStatistics()],
        filters={'trio': TrioFilter()},
        callbacks=[],                           # replaces post_processors
        seed=42,
    )
    sim.run(n_generations=5)

    # Access results
    phenotypes = sim.phenotype_history[generation]  # NPhenotypeArray
    haplotypes = sim.haplotype_history[generation]  # DenseHaplotypeArray
    Y = sim.phenotypes['Y']                         # np.ndarray (current gen)

    # Continue from current state
    sim.continue_run(n_additional=3)

Mating
~~~~~~

**Old**::

    # Random mating
    regime = xft.mate.RandomMatingRegime(
        offspring_per_pair=xft.utils.ConstantCount(2),
        mates_per_female=xft.utils.ConstantCount(1),
        female_offspring_per_pair='balanced',
        sex_aware=False,
    )

    # Assortative mating
    component_index = xft.index.ComponentIndex.from_product(
        phenotype_name=['height'],
        component_name=['phenotype'],
        vorigin_relative=[-1],
    )
    regime = xft.mate.LinearAssortativeMatingRegime(
        component_index=component_index,
        r=0.5,
        offspring_per_pair=xft.utils.ConstantCount(2),
    )

**New**::

    from xftsim.nmate import RandomMating, LinearAssortativeMating

    # Random mating
    mating = RandomMating(offspring_per_pair=2)

    # Assortative mating -- just pass phenotype names as strings
    mating = LinearAssortativeMating(
        component_names=['Y'],
        r=0.5,
        offspring_per_pair=2,
    )

Filters and statistics
~~~~~~~~~~~~~~~~~~~~~~

The new API introduces a ``Filter`` system that extracts structured views
(trios, sib-pairs, etc.) from simulation history. These views are then
consumed by ``Statistic`` objects. This replaces the old monolithic
statistics approach.

**New**::

    from xftsim.nfilter import TrioFilter, SibPairFilter, AscertainmentFilter
    from xftsim.nstats import (
        SampleStatistics,
        HasemanElstonEstimator,
        ParentOffspringRegression,
    )

    sim = NSimulation(
        ...,
        filters={
            'trio': TrioFilter(),
            'sibpair': SibPairFilter(),
        },
        statistics=[
            SampleStatistics(),
            HasemanElstonEstimator(filter_name='sibpair'),
            ParentOffspringRegression(filter_name='trio'),
        ],
    )
    sim.run(5)

    # Results are stored as a list of GenerationResult objects
    for result in sim.results:
        gen = result.generation
        stats = result.statistics['SampleStatistics']
        var_y = stats['var'][stats['keys'].index('Y')]
        print(f"Gen {gen}: Var(Y) = {var_y:.3f}")

I/O and checkpointing
~~~~~~~~~~~~~~~~~~~~~

The old API used pickle serialization. The new API provides structured
numpy-based serialization for all components.

**New I/O functions**::

    from xftsim.io import (
        save_haplotypes_npz, load_haplotypes_npz,
        save_phenotypes_npz, load_phenotypes_npz,
        save_effects_npz, load_effects_npz,
        save_architecture, load_architecture,
        save_simulation_checkpoint, load_simulation_checkpoint,
    )

    # Save/load individual components
    save_haplotypes_npz(hap, 'founders.npz')
    hap = load_haplotypes_npz('founders.npz')

    save_effects_npz(eff, 'effects.npz')
    eff = load_effects_npz('effects.npz')

    save_architecture(arch, 'arch_dir/')
    arch = load_architecture('arch_dir/')

    # Full simulation checkpoint (saves everything to a directory)
    save_simulation_checkpoint(sim, 'checkpoint_dir/')

    # Resume from checkpoint
    sim = NSimulation.from_checkpoint(
        'checkpoint_dir/',
        callbacks=[...],
        filters={...},
        statistics=[...],
    )
    sim.continue_run(n_additional=5)


Things Removed or Simplified
-----------------------------

The following old API features have been removed or do not have direct
equivalents in the new API:

**Removed classes and concepts**:

- ``ComponentIndex`` -- replaced by flat string keys.
- ``SampleIndex`` -- replaced by ``SampleMeta`` (plain numpy arrays, no
  xarray coordinate wrapping).
- ``HaploidVariantIndex`` / ``DiploidVariantIndex`` -- replaced by
  ``VariantMeta`` (simple container, no ploidy distinction in the index).
- ``FounderInitialization`` subclasses (``GaussianFounderInitialization``,
  ``ZeroFounderInitialization``, etc.) -- founder phenotype initialization is
  handled automatically by the architecture DAG.
- ``GCTA_Architecture`` convenience class -- use the formula DSL instead.
- ``InfinitessimalArchitecture``, ``SpikeSlabArchitecture`` -- never
  implemented; use ``AdditiveEffects`` or ``SparseEffects`` with the formula.
- ``BinarizingTransformation`` -- not yet ported. Apply thresholding in a
  callback if needed.
- ``ProductComponent`` -- use arithmetic expressions in
  ``AggregationComponent`` (e.g., ``Y ~ X1 * X2``).
- ``BatchedMatingRegime``, ``FilteredMatingRegime``,
  ``GeneralAssortativeMatingRegime`` -- not yet ported. Only ``RandomMating``
  and ``LinearAssortativeMating`` are available.
- ``xft.ped.Pedigree`` -- replaced by ``PedigreeArray`` (integer index arrays).
- ``MateAssignment`` with ``SampleIndex`` -- replaced by ``NMateAssignment``
  with integer arrays.
- ``control`` dict on ``Simulation`` -- standardization is now handled by
  ``HaplotypeOperator.standardized_matvec()`` based on the effect's
  ``standardized`` flag.
- Post-processors (``proc.*``) -- use callbacks instead (any
  ``Callable[[NSimulation], None]``).
- ``filter_sample`` / ``SampleFilter`` on ``Simulation`` -- use
  ``AscertainmentFilter``, ``SubsampleFilter``, or ``UnrelatedFilter`` via the
  ``filters`` dict.
- xarray accessor (``phenotypes.xft[...]``) -- access phenotype arrays
  directly via ``phenotypes['key']``.
- Dependency graph visualization (``draw_dependency_graph()``) -- not yet
  ported. The DAG is topologically sorted internally.

**Simplified behaviors**:

- ``offspring_per_pair`` is now a plain ``int``, not a ``VariableCount``
  object.
- ``mates_per_female`` is fixed at 1 (monogamous mating).
- Mating is always sex-aware (samples are split into females and males by
  the ``sex`` field of ``SampleMeta``).
- Effect objects no longer store allele frequencies. Allele frequencies live
  on ``VariantMeta`` and standardization is done at matvec time by the
  ``HaplotypeOperator``.
- History retention is configurable via ``retain_haplotypes`` and
  ``retain_phenotypes`` parameters, replacing the old approach of keeping
  everything.


Common Gotchas
--------------

1. **Formula parser: one component per line.** The formula parser expects
   exactly one component per line. Do NOT write
   ``Y ~ genetic(eff) + noise(0.5)`` on one line. Instead::

       Y.G ~ genetic(eff)
       Y.E ~ noise(0.5)
       Y ~ Y.G + Y.E

2. **``DenseHaplotypeArray`` generation default.** The ``generation=0``
   default on ``DenseHaplotypeArray`` can override what you set on
   ``SampleMeta.generation``. Pass ``generation=`` explicitly if you need a
   different value.

3. **Retention policy prunes history.** With ``retain_haplotypes=1``,
   generation 0 haplotypes are dropped after generation 2 finishes. If your
   callbacks or statistics need older generations, increase the retention.

4. **Filters parameter is a dict, not a list.** Pass filters as
   ``filters={'trio': TrioFilter()}``, not ``filters=[TrioFilter()]``.

5. **``TrioFilter()`` and ``SampleStatistics()`` take no constructor
   arguments.** Do not pass any parameters to their constructors.

6. **Sibling components need explicit inputs.** When adding a sibling
   component programmatically, you must pass ``inputs=['Y']`` explicitly::

       arch.add('Y.sib', SiblingMeanComponent('Y'), inputs=['Y'])

7. **Integer indexing on haplotypes.** ``hap[0]`` returns a 2D array that
   breaks ``subset()``. Use ``hap[[0]]`` to select a single individual while
   preserving the 3D shape.

8. **Validation happens at ``run()`` time.** Dimension mismatches between
   effects and haplotypes raise errors when ``sim.run()`` is called, not
   at ``NSimulation.__init__()`` time.

9. **Toposort cycle detection is lazy.** Cycles in the architecture DAG are
   detected during topological sort (when ``arch.nodes`` is accessed), not
   when ``arch.add()`` is called.

10. **Self-loops are allowed.** An architecture node that references its own
    output in its inputs is permitted (Kahn's algorithm skips self-edges).
    This is intentional for certain recursive architectures.

11. **Checkpoint file structure.** The checkpoint directory uses
    ``meta.json`` (not ``metadata.json``), and the mating regime key is
    ``'mating'`` (not ``'mating_regime'``).

12. **``PedigreeArray`` constructor.** Takes ``offspring_samples``,
    ``maternal_idx``, ``paternal_idx``, and ``parent_n`` -- not ``iid`` or
    ``n_parents``.

13. **``sim.results`` is a list, not a dict.** Each entry is a
    ``GenerationResult(generation=g, statistics={...})``. The statistics
    are nested: ``result.statistics['SampleStatistics']`` returns a dict
    with ``'cov'``, ``'var'``, and ``'keys'``.


Complete Example: Before and After
----------------------------------

Below is a side-by-side comparison of a complete simulation setup.

Old API::

    import xftsim as xft
    import numpy as np

    # 1. Founder haplotypes (xarray-based)
    haplotypes = xft.founders.founder_haplotypes(n=500, m=100)

    # 2. Effect sizes (requires variant + component indexers)
    variant_indexer = haplotypes.xft.get_variant_indexer()
    comp_g = xft.index.ComponentIndex.from_product(
        ['height'], ['addGenetic'], [-1]
    )
    effects = xft.effect.AdditiveEffects(
        beta=np.random.randn(100, 1) * np.sqrt(0.5 / 100),
        variant_indexer=variant_indexer,
        component_indexer=comp_g,
    )

    # 3. Noise
    comp_e = xft.index.ComponentIndex.from_product(
        ['height'], ['noise'], [-1]
    )
    noise = xft.arch.AdditiveNoiseComponent(
        variances=[0.5], component_index=comp_e
    )

    # 4. Sum transformation
    sum_input = xft.index.XftIndex.reduce_merge([comp_g, comp_e])
    sum_comp = xft.arch.SumAllTransformation(input_cindex=sum_input)

    # 5. Architecture
    arch = xft.arch.Architecture(components=[
        xft.arch.AdditiveGeneticComponent(beta=effects),
        noise,
        sum_comp,
    ])

    # 6. Mating + recombination
    regime = xft.mate.RandomMatingRegime(
        offspring_per_pair=xft.utils.ConstantCount(2),
    )
    recomb = xft.reproduce.RecombinationMap(m=100)

    # 7. Simulation
    sim = xft.sim.Simulation(
        founder_haplotypes=haplotypes,
        mating_regime=regime,
        recombination_map=recomb,
        architecture=arch,
    )
    sim.run(n_generations=5)

New API::

    import numpy as np
    from xftsim.founders import founder_haplotypes_uniform_AFs
    from xftsim.neffect import AdditiveEffects
    from xftsim.narch import Architecture
    from xftsim.nmate import RandomMating
    from xftsim.reproduce import RecombinationMap
    from xftsim.nsim import NSimulation
    from xftsim.nstats import SampleStatistics

    # 1. Founder haplotypes
    hap = founder_haplotypes_uniform_AFs(n=500, m=100)

    # 2. Effect sizes
    eff = AdditiveEffects.from_h2(h2=0.5, m=100, seed=42)

    # 3-5. Architecture (3 lines instead of ~20)
    arch = Architecture(
        formula="""
        Y.G ~ genetic(eff)
        Y.E ~ noise(0.5)
        Y ~ Y.G + Y.E
        """,
        effects={'eff': eff},
    )

    # 6. Mating + recombination
    mating = RandomMating(offspring_per_pair=2)
    recomb = RecombinationMap.constant_map(m=100, p=0.5)

    # 7. Simulation
    sim = NSimulation(
        founder_haplotypes=hap,
        architecture=arch,
        mating_regime=mating,
        recombination_map=recomb,
        statistics=[SampleStatistics()],
        seed=42,
    )
    sim.run(n_generations=5)

    # Access results
    for r in sim.results:
        stats = r.statistics['SampleStatistics']
        idx = stats['keys'].index('Y')
        print(f"Gen {r.generation}: Var(Y) = {stats['var'][idx]:.3f}")
