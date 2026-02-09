Architecture & Components
========================

The architecture system defines the phenogenetic architecture as a
directed acyclic graph (DAG) of ``ArchNode`` objects. Each node wraps an
``ArchComponent`` that computes one piece of the phenotype model.

Architecture
------------

.. autoclass:: xftsim.narch.Architecture
   :members:
   :undoc-members:
   :show-inheritance:

ArchNode
--------

.. autoclass:: xftsim.narch.ArchNode
   :members:
   :undoc-members:
   :show-inheritance:

Component ABC
-------------

.. autoclass:: xftsim.narch.ArchComponent
   :members:
   :undoc-members:
   :show-inheritance:

Genetic Components
------------------

.. autoclass:: xftsim.narch.GeneticComponent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: xftsim.narch.MVGeneticComponent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: xftsim.narch.HaplotypeGeneticComponent
   :members:
   :undoc-members:
   :show-inheritance:

Noise Components
----------------

.. autoclass:: xftsim.narch.NoiseComponent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: xftsim.narch.CNoiseComponent
   :members:
   :undoc-members:
   :show-inheritance:

Aggregation
-----------

.. autoclass:: xftsim.narch.AggregationComponent
   :members:
   :undoc-members:
   :show-inheritance:

Parental Components
-------------------

.. autoclass:: xftsim.narch.ParentComponent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: xftsim.narch.MotherComponent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: xftsim.narch.FatherComponent
   :members:
   :undoc-members:
   :show-inheritance:

Sibling Components
------------------

.. autoclass:: xftsim.narch.SiblingMeanComponent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: xftsim.narch.SiblingSumComponent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: xftsim.narch.SiblingAnyComponent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: xftsim.narch.SiblingCountComponent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: xftsim.narch.SiblingEldestComponent
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: xftsim.narch.SiblingYoungestComponent
   :members:
   :undoc-members:
   :show-inheritance:

BUILTINS Registry
-----------------

.. autodata:: xftsim.narch.BUILTINS
   :no-value:
