Getting Started
===============

Installation
------------

Install ``xftsim`` from source (pip-installable package coming soon):

.. code-block:: bash

   git clone https://github.com/rborder/xftsim.git
   cd xftsim
   pip install -e .

Dependencies are installed automatically. For GRG support, also install
``pygrgl``:

.. code-block:: bash

   pip install pygrgl

Minimal Example
---------------

Run a simple forward-time simulation with additive genetic effects and
environmental noise:

.. code-block:: python

   import numpy as np
   from xftsim.struct import DenseHaplotypeArray, SampleMeta, VariantMeta
   from xftsim.neffect import AdditiveEffects
   from xftsim.narch import Architecture
   from xftsim.nmate import RandomMating
   from xftsim.reproduce import RecombinationMap
   from xftsim.nsim import NSimulation

   # --- Founder haplotypes ---
   n_samples = 200
   n_variants = 50
   rng = np.random.default_rng(42)

   genotypes = rng.integers(0, 2, size=(n_samples, 2, n_variants)).astype(np.int8)

   sample_meta = SampleMeta(
       iid=[f"ind_{i}" for i in range(n_samples)],
       fid=[f"fam_{i}" for i in range(n_samples)],
       generation=0,
   )
   variant_meta = VariantMeta(
       vid=[f"SNP_{i}" for i in range(n_variants)],
       chrom=np.ones(n_variants, dtype=int),
       pos_bp=np.arange(n_variants) * 1000,
   )
   haplotypes = DenseHaplotypeArray(genotypes, sample_meta, variant_meta)

   # --- Effects ---
   effects = {
       'eff': AdditiveEffects(
           betas=rng.normal(0, 0.1, size=n_variants),
           variant_meta=variant_meta,
       ),
   }

   # --- Architecture (formula DSL) ---
   formula = """
   # Genetic component
   height.G ~ genetic(eff)
   # Environmental noise
   height.E ~ noise(0.5)
   # Total phenotype
   height ~ height.G + height.E
   """

   arch = Architecture.from_formula(formula, effects=effects)

   # --- Run simulation ---
   sim = NSimulation(
       founder_haplotypes=haplotypes,
       architecture=arch,
       mating_regime=RandomMating(n_offspring=200),
       recombination_map=RecombinationMap.uniform(n_variants, rate=1e-8),
       retain_haplotypes=1,
       retain_phenotypes=2,
   )

   sim.run(n_generations=5)

   # Access results
   latest_pheno = sim.phenotype_history[sim.generation]
   print(f"Generation {sim.generation}: mean height = "
         f"{latest_pheno.get_column('height').mean():.3f}")

What Next?
----------

* :doc:`formula` -- Full reference for the formula DSL.
* :doc:`../api/index` -- Complete API documentation.
