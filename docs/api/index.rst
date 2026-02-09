API Reference
=============

This section documents the public API of the refactored ``xftsim`` modules.
These are the ``n*`` modules that replace the legacy architecture.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Module
     - Description
   * - :doc:`nsim`
     - Forward-time simulation loop (``NSimulation``)
   * - :doc:`narch`
     - Architecture DAG, components, and nodes
   * - :doc:`neffect`
     - Genetic effect specifications (``EffectSpec``, ``AdditiveEffects``, etc.)
   * - :doc:`nmate`
     - Mate assignment (``RandomMating``, ``LinearAssortativeMating``)
   * - :doc:`nfilter`
     - Filters and filtered views (trios, sib-pairs)
   * - :doc:`nstats`
     - Per-generation statistics
   * - :doc:`ngwas`
     - GWAS and polygenic scores
   * - :doc:`io`
     - Serialization and I/O
   * - :doc:`struct`
     - Core data structures (``HaplotypeOperator``, ``NPhenotypeArray``, etc.)
   * - :doc:`parser`
     - Formula DSL parser
   * - :doc:`cli`
     - Command-line interface

.. toctree::
   :hidden:

   nsim
   narch
   neffect
   nmate
   nfilter
   nstats
   ngwas
   io
   struct
   parser
   cli
