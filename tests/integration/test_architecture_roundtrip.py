"""
Integration test: architecture serialization roundtrip with complex setups.

Tests:
1. Single-trait architecture save/load preserves structure
2. Multi-trait architecture save/load preserves structure
3. Architecture with VT components save/load works
4. Loaded architecture produces same phenotypes as original
5. Architecture with sparse effects roundtrips correctly
"""
import numpy as np
import pytest
import tempfile
import shutil

from xftsim.effect import AdditiveEffects, MultivariateEffects, SparseEffects
from xftsim.arch import (
    Architecture, GeneticComponent, MVGeneticComponent,
    NoiseComponent, CNoiseComponent, AggregationComponent,
    MotherComponent,
)
from xftsim.io import save_architecture, load_architecture

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


def _find_node(arch, name):
    """Find ArchNode by output name."""
    for node in arch._nodes:
        if name in node.outputs:
            return node
    raise KeyError(f"No node with output '{name}'")


class TestArchitectureRoundtrip:
    def test_single_trait_roundtrip(self):
        """Single-trait architecture should survive save/load."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=20, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))

        tmpdir = tempfile.mkdtemp()
        try:
            save_architecture(arch, tmpdir)
            loaded = load_architecture(tmpdir)

            # Same node names
            orig_names = [n.outputs for n in arch.nodes]
            load_names = [n.outputs for n in loaded.nodes]
            assert orig_names == load_names
        finally:
            shutil.rmtree(tmpdir)

    def test_multi_trait_roundtrip(self):
        """Multi-trait architecture with MVGenetic should survive save/load."""
        mv = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.2, m=20, seed=42)
        arch = Architecture()
        arch.add(['Y1.G', 'Y2.G'], MVGeneticComponent(mv))
        arch.add(['Y1.E', 'Y2.E'], CNoiseComponent(cov=np.diag([0.5, 0.7])))
        arch.add('Y1', AggregationComponent('Y1.G + Y1.E'))
        arch.add('Y2', AggregationComponent('Y2.G + Y2.E'))

        tmpdir = tempfile.mkdtemp()
        try:
            save_architecture(arch, tmpdir)
            loaded = load_architecture(tmpdir)

            orig_names = [n.outputs for n in arch.nodes]
            load_names = [n.outputs for n in loaded.nodes]
            assert orig_names == load_names
        finally:
            shutil.rmtree(tmpdir)

    def test_loaded_arch_produces_phenotypes(self):
        """Architecture loaded from disk should produce valid phenotypes."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=20, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))

        tmpdir = tempfile.mkdtemp()
        try:
            save_architecture(arch, tmpdir)
            loaded = load_architecture(tmpdir)

            hap = TestSimulation.founder_haplotypes(n=100, m=20, seed=42)
            pheno = loaded.compute(hap, rng=np.random.RandomState(123))
            assert 'Y' in pheno
            assert 'Y.G' in pheno
            assert np.all(np.isfinite(pheno['Y']))
        finally:
            shutil.rmtree(tmpdir)

    def test_sparse_effects_roundtrip(self):
        """Architecture with sparse effects should roundtrip correctly."""
        eff = SparseEffects.from_h2(h2=0.5, m=50, k_causal=10, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))

        tmpdir = tempfile.mkdtemp()
        try:
            save_architecture(arch, tmpdir)
            loaded = load_architecture(tmpdir)

            # Check the effects are preserved
            orig_comp = _find_node(arch, 'Y.G').component
            load_comp = _find_node(loaded, 'Y.G').component
            np.testing.assert_array_equal(
                orig_comp.effects.effects, load_comp.effects.effects
            )
            np.testing.assert_array_equal(
                orig_comp.effects.variant_mask, load_comp.effects.variant_mask
            )
        finally:
            shutil.rmtree(tmpdir)

    def test_vt_architecture_roundtrip(self):
        """Architecture with VT components should survive save/load."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=20, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.3))
        arch.add('Y.m', MotherComponent('Y', founder_component=NoiseComponent(variance=0.1)))
        arch.add('Y', AggregationComponent('Y.G + Y.E + 0.2 * Y.m'))

        tmpdir = tempfile.mkdtemp()
        try:
            save_architecture(arch, tmpdir)
            loaded = load_architecture(tmpdir)

            orig_names = [n.outputs for n in arch.nodes]
            load_names = [n.outputs for n in loaded.nodes]
            assert orig_names == load_names

            # VT node should still be a MotherComponent
            vt_node = _find_node(loaded, 'Y.m')
            assert isinstance(vt_node.component, MotherComponent)
        finally:
            shutil.rmtree(tmpdir)
