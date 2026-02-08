"""
Legacy xftsim modules.

These modules contain the original (pre-refactor) implementations of:
  - arch (phenogenetic architectures)
  - sim (simulation loop)
  - mate (mate assignment)
  - effect (genetic effects)
  - filters (sample filtering)
  - stats (estimation)
  - index (indexing)
  - proc (post-processing)
  - data (data download utilities)

They are preserved for backward compatibility and for the DemoSimulation class.
New code should use the primary modules (narch, nsim, nmate, neffect, nfilter, nstats).

Submodules are loaded on demand; import them explicitly, e.g.::

    from xftsim.legacy import arch
    from xftsim.legacy.index import SampleIndex
"""
