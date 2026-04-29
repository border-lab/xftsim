"""
Unit tests for IO error paths and edge cases.

Tests:
1. load_effects_npz: unknown class name
2. load_architecture: unknown component type
3. _deserialize_mating_regime: unknown type
4. save/load_phenotypes_npz: empty phenotype
5. save_effects_npz: SparseEffects roundtrip
6. Architecture save with parental components
7. Custom mating class serialization
"""
import numpy as np
import pytest
import tempfile
import os
import json

from xftsim.struct import SampleMeta, NPhenotypeArray
from xftsim.neffect import AdditiveEffects, SparseEffects
from xftsim.narch import (
    Architecture, GeneticComponent, NoiseComponent, AggregationComponent,
    MotherComponent,
)
from xftsim.io import (
    save_effects_npz, load_effects_npz,
    save_phenotypes_npz, load_phenotypes_npz,
    save_architecture, load_architecture,
    _serialize_mating_regime, _deserialize_mating_regime,
)


class TestLoadEffectsErrors:
    def test_unknown_class_name_raises(self):
        """Unknown EffectSpec class in saved file should raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'bad_effects.npz')
            np.savez_compressed(path,
                                effects=np.zeros(10),
                                standardized=np.array([True]),
                                variant_mask=np.ones(10, dtype=bool),
                                class_name=np.array(['UnknownEffects']))
            with pytest.raises(ValueError, match="Unknown EffectSpec class"):
                load_effects_npz(path)

    def test_sparse_effects_roundtrip(self):
        """SparseEffects should survive save/load."""
        eff = SparseEffects.from_h2(h2=0.5, m=20, k_causal=5, seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'sparse.npz')
            save_effects_npz(eff, path)
            loaded = load_effects_npz(path)
            assert isinstance(loaded, SparseEffects)
            np.testing.assert_allclose(loaded.effects, eff.effects)
            np.testing.assert_array_equal(loaded.variant_mask, eff.variant_mask)
            assert loaded.standardized == eff.standardized


class TestLoadArchitectureErrors:
    def test_unknown_component_type_raises(self):
        """Unknown component type in architecture JSON should raise."""
        with tempfile.TemporaryDirectory() as tmpdir:
            arch_dir = os.path.join(tmpdir, 'arch')
            os.makedirs(arch_dir)
            node_specs = [{
                'outputs': ['Y'],
                'inputs': [],
                'grouping': None,
                'component_type': 'FakeComponent',
            }]
            with open(os.path.join(arch_dir, 'architecture.json'), 'w') as f:
                json.dump(node_specs, f)
            with pytest.raises(ValueError, match="Unknown component type"):
                load_architecture(arch_dir)

    def test_parental_component_roundtrip(self):
        """Architecture with MotherComponent should survive save/load."""
        arch = Architecture()
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y.VTm', MotherComponent('Y'))
        arch.add('Y', AggregationComponent('Y.E + Y.VTm'), inputs=['Y.E', 'Y.VTm'])
        with tempfile.TemporaryDirectory() as tmpdir:
            arch_dir = os.path.join(tmpdir, 'arch')
            save_architecture(arch, arch_dir)
            loaded = load_architecture(arch_dir)
            assert len(loaded.nodes) == 3
            mother_nodes = [n for n in loaded.nodes
                            if isinstance(n.component, MotherComponent)]
            assert len(mother_nodes) == 1
            assert mother_nodes[0].component.phenotype_name == 'Y'


class TestDeserializeMatingRegimeErrors:
    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown mating regime type"):
            _deserialize_mating_regime({'type': 'FancyMating'})

    def test_custom_class_raises_on_serialize(self):
        """Custom mating classes must raise at save time. Previously they
        silently produced ``{'type': name}`` with all parameters dropped.
        """
        class CustomMating:
            pass
        with pytest.raises(ValueError, match="[Cc]annot serialize"):
            _serialize_mating_regime(CustomMating())


class TestEmptyPhenotypeIO:
    def test_empty_phenotype_roundtrip(self):
        """Phenotype with no values should roundtrip."""
        sm = SampleMeta(iid=np.arange(5))
        pheno = NPhenotypeArray(samples=sm)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'empty_pheno.npz')
            save_phenotypes_npz(pheno, path)
            loaded = load_phenotypes_npz(path)
            assert len(loaded.keys) == 0
