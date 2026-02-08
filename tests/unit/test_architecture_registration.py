"""
Unit tests for Architecture node registration and duplicate output detection.

Tests:
1. Duplicate output name raises ValueError
2. Multiple nodes with different outputs work
3. _output_map tracks all outputs
4. Toposort invalidation on add
5. Architecture repr
"""
import numpy as np
import pytest

from xftsim.narch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.neffect import AdditiveEffects


class TestArchitectureRegistration:
    def test_duplicate_output_raises(self):
        """Adding a node with an already-used output name should raise."""
        arch = Architecture()
        arch.add('Y.E', NoiseComponent(variance=1.0))
        with pytest.raises(ValueError, match="Duplicate output name"):
            arch.add('Y.E', NoiseComponent(variance=0.5))  # duplicate!

    def test_duplicate_in_multioutput_raises(self):
        """Duplicate within multi-output should raise."""
        arch = Architecture()
        arch.add('Y.E', NoiseComponent(variance=1.0))
        # Try to add a second node that includes 'Y.E' in its outputs
        from xftsim.neffect import MultivariateEffects
        from xftsim.narch import MVGeneticComponent
        mv = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.2, m=10, seed=42)
        with pytest.raises(ValueError, match="Duplicate output name"):
            arch.add(['Y.E', 'Y2.G'], MVGeneticComponent(mv))

    def test_output_map_tracks_outputs(self):
        """_output_map should contain all registered outputs."""
        arch = Architecture()
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))

        assert 'Y.G' in arch._output_map
        assert 'Y.E' in arch._output_map
        assert 'Y' in arch._output_map

    def test_toposort_invalidated_on_add(self):
        """Adding a node should invalidate the cached toposort."""
        arch = Architecture()
        arch.add('Y.E', NoiseComponent(variance=1.0))
        # Access nodes to trigger toposort
        _ = arch.nodes
        assert arch._sorted is not None

        # Add another node should invalidate
        arch.add('Y', AggregationComponent('Y.E'))
        assert arch._sorted is None

    def test_repr(self):
        """Architecture repr should show node count and output names."""
        arch = Architecture()
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))

        r = repr(arch)
        assert 'Architecture' in r
        assert 'nodes=3' in r
        assert 'Y.G' in r
        assert 'Y.E' in r
        assert 'Y' in r
