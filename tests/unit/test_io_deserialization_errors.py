"""
Unit tests for IO deserialization error paths.

Tests:
1. Unknown EffectSpec class raises ValueError
2. Unknown component type in architecture raises ValueError
3. Unknown mating regime type raises ValueError
4. load_haplotypes_npz with missing optional fields
5. load_effects_npz roundtrip preserves class name
"""
import numpy as np
import pytest
import tempfile
import os

from xftsim.neffect import AdditiveEffects, MultivariateEffects, SparseEffects
from xftsim.io import (
    save_effects_npz, load_effects_npz,
    save_architecture, load_architecture,
    _serialize_mating_regime, _deserialize_mating_regime,
    save_haplotypes_npz, load_haplotypes_npz,
)
from xftsim.narch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.nmate import RandomMating, LinearAssortativeMating
from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray


class TestEffectsDeserialization:
    def test_unknown_class_name_raises(self):
        """Unknown EffectSpec class should raise ValueError."""
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, 'bad_effects.npz')
        # Save with a fake class name
        np.savez(path,
                 effects=np.array([1.0, 2.0]),
                 standardized=np.array([True]),
                 variant_mask=np.array([True, True]),
                 class_name=np.array(['BogusEffects']))
        with pytest.raises(ValueError, match="Unknown EffectSpec class"):
            load_effects_npz(path)

    def test_additive_roundtrip_class_name(self):
        """AdditiveEffects roundtrip should preserve class name."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, 'eff.npz')
        save_effects_npz(eff, path)
        loaded = load_effects_npz(path)
        assert isinstance(loaded, AdditiveEffects)

    def test_multivariate_roundtrip_class_name(self):
        """MultivariateEffects roundtrip should preserve class name."""
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.2, m=10, seed=42)
        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, 'eff.npz')
        save_effects_npz(eff, path)
        loaded = load_effects_npz(path)
        assert isinstance(loaded, MultivariateEffects)


class TestArchitectureDeserialization:
    def test_unknown_component_type_raises(self):
        """Unknown component type in saved architecture should raise."""
        tmpdir = tempfile.mkdtemp()
        import json
        # Create a minimal architecture JSON with a bogus component type
        # Format is a list of node specs (not a dict with 'nodes' key)
        meta = [{
            'outputs': ['Y'],
            'inputs': [],
            'grouping': None,
            'component_type': 'NonexistentComponent',
        }]
        with open(os.path.join(tmpdir, 'architecture.json'), 'w') as f:
            json.dump(meta, f)
        with pytest.raises(ValueError, match="Unknown component type"):
            load_architecture(tmpdir)


class TestMatingRegimeDeserialization:
    def test_unknown_mating_type_raises(self):
        """Unknown mating regime type should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown mating regime type"):
            _deserialize_mating_regime({'type': 'BogusRegime'})

    def test_random_mating_roundtrip(self):
        """RandomMating serialization roundtrip."""
        rm = RandomMating(offspring_per_pair=3)
        config = _serialize_mating_regime(rm)
        loaded = _deserialize_mating_regime(config)
        assert isinstance(loaded, RandomMating)
        assert loaded.offspring_per_pair == 3

    def test_assortative_mating_roundtrip(self):
        """LinearAssortativeMating serialization roundtrip."""
        am = LinearAssortativeMating(
            component_names=['Y', 'Z'], r=0.6, offspring_per_pair=2,
        )
        config = _serialize_mating_regime(am)
        loaded = _deserialize_mating_regime(config)
        assert isinstance(loaded, LinearAssortativeMating)
        assert loaded.r == 0.6
        assert loaded.component_names == ['Y', 'Z']


class TestHaplotypesNpzOptional:
    def test_load_with_minimal_variantmeta(self):
        """Haplotypes with minimal VariantMeta (no chrom, etc.) should roundtrip."""
        n, m = 10, 5
        sm = SampleMeta(iid=np.arange(n))
        vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
        geno = np.random.RandomState(42).randint(0, 2, (n, m, 2)).astype(np.int8)
        hap = DenseHaplotypeArray(geno, samples=sm, variants=vm)

        tmpdir = tempfile.mkdtemp()
        path = os.path.join(tmpdir, 'hap.npz')
        save_haplotypes_npz(hap, path)
        loaded = load_haplotypes_npz(path)
        assert loaded.n == n
        assert loaded.m == m
        np.testing.assert_array_equal(loaded.genotypes, geno)
