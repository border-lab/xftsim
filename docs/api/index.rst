API Reference
=============

This section documents the public API of the refactored ``xftsim`` modules.
These are the ``n*`` modules that replace the legacy architecture.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Module
     - Description
   * - :doc:`sim`
     - Forward-time simulation loop (``Simulation``)
   * - :doc:`arch`
     - Architecture DAG, components, and nodes
   * - :doc:`effect`
     - Genetic effect specifications (``EffectSpec``, ``AdditiveEffects``, etc.)
   * - :doc:`mate`
     - Mate assignment (``RandomMating``, ``LinearAssortativeMating``)
   * - :doc:`filters`
     - Filters and filtered views (trios, sib-pairs)
   * - :doc:`stats`
     - Per-generation statistics
   * - :doc:`gwas`
     - GWAS and polygenic scores
   * - :doc:`io`
     - Serialization and I/O
   * - :doc:`struct`
     - Core data structures (``HaplotypeOperator``, ``PhenotypeArray``, etc.)
   * - :doc:`parser`
     - Formula DSL parser
   * - :doc:`cli`
     - Command-line interface

.. toctree::
   :hidden:

   sim
   arch
   effect
   mate
   filters
   stats
   gwas
   io
   struct
   parser
   cli
