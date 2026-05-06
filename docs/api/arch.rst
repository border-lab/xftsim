Architecture & Components
=========================

The architecture system defines the phenogenetic architecture as a
directed acyclic graph (DAG) of ``ArchNode`` objects. Each node wraps an
``ArchComponent`` that computes one piece of the phenotype model.

Architecture
------------

.. autoclass:: xftsim.arch.Architecture
   :members:
   :undoc-members:
   :show-inheritance:

ArchNode
--------

.. autoclass:: xftsim.arch.ArchNode
   :members:
   :undoc-members:
   :show-inheritance:

Component ABC
-------------

.. autoclass:: xftsim.arch.ArchComponent
   :members:
   :undoc-members:
   :show-inheritance:

Genetic Components
------------------

.. autoclass:: xftsim.arch.GeneticComponent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: xftsim.arch.MVGeneticComponent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: xftsim.arch.HaplotypeGeneticComponent
   :members:
   :undoc-members:
   :show-inheritance:

Noise Components
----------------

.. autoclass:: xftsim.arch.NoiseComponent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: xftsim.arch.CNoiseComponent
   :members:
   :undoc-members:
   :show-inheritance:

Aggregation
-----------

.. autoclass:: xftsim.arch.AggregationComponent
   :members:
   :undoc-members:
   :show-inheritance:

Parental Components
-------------------

.. autoclass:: xftsim.arch.ParentComponent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: xftsim.arch.MotherComponent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: xftsim.arch.FatherComponent
   :members:
   :undoc-members:
   :show-inheritance:

Sibling Components
------------------

.. autoclass:: xftsim.arch.SiblingMeanComponent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: xftsim.arch.SiblingSumComponent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: xftsim.arch.SiblingAnyComponent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: xftsim.arch.SiblingCountComponent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: xftsim.arch.SiblingEldestComponent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: xftsim.arch.SiblingYoungestComponent
   :members:
   :undoc-members:
   :show-inheritance:

BUILTINS Registry
-----------------

.. autodata:: xftsim.arch.BUILTINS
   :no-value:
