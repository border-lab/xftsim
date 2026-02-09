"""
Tests covering io.py gaps.

Targets uncovered lines:
- genotypes_to_pseudo_haplotypes (lines 59-78, 81-96)
- Architecture save/load with HaplotypeGeneticComponent, CNoiseComponent,
  MVGeneticComponent, ParentalComponent, SiblingComponent (lines 505-525, 564-578)
- GraphHaplotypeOperator checkpointing (lines 683-685)
- _serialize/_deserialize mating regime edge cases
- load_effects_npz unknown class
- save_simulation_checkpoint and load_simulation_checkpoint coverage
"""
import json
import os
import tempfile

import numpy as np
import pytest

from xftsim.struct import (
    SampleMeta, VariantMeta, DenseHaplotypeArray, NPhenotypeArray, PedigreeArray,
)
from xftsim.neffect import AdditiveEffects, MultivariateEffects, SparseEffects
from xftsim.narch import (
    Architecture, GeneticComponent, MVGeneticComponent,
    HaplotypeGeneticComponent, NoiseComponent, CNoiseComponent,
    AggregationComponent, MotherComponent, FatherComponent,
    ParentComponent, SiblingMeanComponent,
)
from xftsim.io import (
    genotypes_to_pseudo_haplotypes,
    save_haplotypes_npz, load_haplotypes_npz,
    save_phenotypes_npz, load_phenotypes_npz,
    save_effects_npz, load_effects_npz,
    save_architecture, load_architecture,
    save_simulation_checkpoint, load_simulation_checkpoint,
    _serialize_mating_regime, _deserialize_mating_regime,
)


# ---------------------------------------------------------------------------
# genotypes_to_pseudo_haplotypes
# ---------------------------------------------------------------------------

class TestGenotypesToPseudoHaplotypes:
    def test_basic_conversion(self):
        geno = np.array([[0, 1, 2], [2, 0, 1]], dtype=np.int8)
        hap = genotypes_to_pseudo_haplotypes(geno)
        assert hap.shape == (2, 3, 2)
        # Homozygous ref: both 0
        assert hap[0, 0, 0] == 0 and hap[0, 0, 1] == 0
        # Homozygous alt: both 1
        assert hap[0, 2, 0] == 1 and hap[0, 2, 1] == 1
        # Het: sum is 1
        assert hap[0, 1, 0] + hap[0, 1, 1] == 1

    def test_all_zeros(self):
        geno = np.zeros((5, 3), dtype=np.int8)
        hap = genotypes_to_pseudo_haplotypes(geno)
        assert np.all(hap == 0)

    def test_all_twos(self):
        geno = np.full((5, 3), 2, dtype=np.int8)
        hap = genotypes_to_pseudo_haplotypes(geno)
        assert np.all(hap == 1)


# ---------------------------------------------------------------------------
# Architecture save/load with extended components
# ---------------------------------------------------------------------------

class TestArchitectureSaveLoadExtended:
    def test_haplotype_genetic_component(self, tmp_path):
        """HaplotypeGeneticComponent roundtrip."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        arch = Architecture()
        arch.add('Y.G', HaplotypeGeneticComponent(eff, haplotype='maternal'))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))

        save_dir = str(tmp_path / "arch")
        save_architecture(arch, save_dir)
        loaded = load_architecture(save_dir)
        assert len(loaded._nodes) == 3
        # Check haplotype component preserved
        comp = loaded._nodes[0].component
        assert isinstance(comp, HaplotypeGeneticComponent)
        assert comp.haplotype == 'maternal'

    def test_cnoise_component(self, tmp_path):
        """CNoiseComponent roundtrip."""
        cov = np.array([[0.5, 0.1], [0.1, 0.3]])
        arch = Architecture()
        arch.add(['Y1.E', 'Y2.E'], CNoiseComponent(cov=cov))

        save_dir = str(tmp_path / "arch")
        save_architecture(arch, save_dir)
        loaded = load_architecture(save_dir)
        comp = loaded._nodes[0].component
        assert isinstance(comp, CNoiseComponent)
        assert np.allclose(comp.cov, cov)

    def test_mv_genetic_component(self, tmp_path):
        """MVGeneticComponent roundtrip."""
        eff = MultivariateEffects.from_h2_rg(h2=[0.4, 0.6], rg=0.3, m=10, seed=42)
        arch = Architecture()
        arch.add(['Y1.G', 'Y2.G'], MVGeneticComponent(eff))

        save_dir = str(tmp_path / "arch")
        save_architecture(arch, save_dir)
        loaded = load_architecture(save_dir)
        comp = loaded._nodes[0].component
        assert isinstance(comp, MVGeneticComponent)

    def test_parental_component(self, tmp_path):
        """Mother/Father/Parent component roundtrip."""
        arch = Architecture()
        arch.add('Y.M', MotherComponent('Y'), inputs=['Y'])
        arch.add('Y.F', FatherComponent('Y'), inputs=['Y'])

        save_dir = str(tmp_path / "arch")
        save_architecture(arch, save_dir)
        loaded = load_architecture(save_dir)
        assert isinstance(loaded._nodes[0].component, MotherComponent)
        assert isinstance(loaded._nodes[1].component, FatherComponent)

    def test_sibling_component(self, tmp_path):
        """SiblingMeanComponent roundtrip."""
        arch = Architecture()
        arch.add('Y.sib', SiblingMeanComponent('Y'), inputs=['Y'])

        save_dir = str(tmp_path / "arch")
        save_architecture(arch, save_dir)
        loaded = load_architecture(save_dir)
        assert isinstance(loaded._nodes[0].component, SiblingMeanComponent)
        assert loaded._nodes[0].component.source_name == 'Y'

    def test_unknown_component_type_raises(self, tmp_path):
        """Loading with unknown component_type raises ValueError."""
        save_dir = str(tmp_path / "arch")
        os.makedirs(save_dir)
        node_specs = [{
            'outputs': ['X'],
            'inputs': [],
            'grouping': None,
            'component_type': 'BogusComponent',
        }]
        with open(os.path.join(save_dir, 'architecture.json'), 'w') as f:
            json.dump(node_specs, f)
        with pytest.raises(ValueError, match="Unknown component type"):
            load_architecture(save_dir)


# ---------------------------------------------------------------------------
# Effects save/load edge cases
# ---------------------------------------------------------------------------

class TestEffectsSaveLoadCoverage:
    def test_sparse_effects_roundtrip(self, tmp_path):
        eff = SparseEffects.from_h2(h2=0.5, m=20, k_causal=5, seed=42)
        path = str(tmp_path / "sparse.npz")
        save_effects_npz(eff, path)
        loaded = load_effects_npz(path)
        assert isinstance(loaded, SparseEffects)
        assert np.allclose(loaded.effects, eff.effects)

    def test_multivariate_effects_roundtrip(self, tmp_path):
        eff = MultivariateEffects.from_h2_rg(h2=[0.4, 0.6], rg=0.3, m=20, seed=42)
        path = str(tmp_path / "mv.npz")
        save_effects_npz(eff, path)
        loaded = load_effects_npz(path)
        assert isinstance(loaded, MultivariateEffects)
        assert np.allclose(loaded.effects, eff.effects)

    def test_unknown_class_raises(self, tmp_path):
        """Manually create npz with unknown class name."""
        path = str(tmp_path / "fake.npz")
        np.savez_compressed(
            path,
            effects=np.zeros(5),
            standardized=np.array([True]),
            variant_mask=np.ones(5, dtype=bool),
            class_name=np.array(["FakeEffects"]),
        )
        with pytest.raises(ValueError, match="Unknown EffectSpec class"):
            load_effects_npz(path)


# ---------------------------------------------------------------------------
# Mating regime serialization
# ---------------------------------------------------------------------------

class TestMatingRegimeSerialization:
    def test_random_mating_roundtrip(self):
        from xftsim.nmate import RandomMating
        rm = RandomMating(offspring_per_pair=3)
        config = _serialize_mating_regime(rm)
        assert config['type'] == 'RandomMating'
        loaded = _deserialize_mating_regime(config)
        assert isinstance(loaded, RandomMating)
        assert loaded.offspring_per_pair == 3

    def test_assortative_mating_roundtrip(self):
        from xftsim.nmate import LinearAssortativeMating
        am = LinearAssortativeMating(component_names=['Y'], r=0.5, offspring_per_pair=2)
        config = _serialize_mating_regime(am)
        assert config['type'] == 'LinearAssortativeMating'
        loaded = _deserialize_mating_regime(config)
        assert isinstance(loaded, LinearAssortativeMating)
        assert loaded.r == 0.5

    def test_unknown_mating_type_serialize(self):
        """Serializing unknown type gives minimal dict."""
        class FakeMating:
            pass
        config = _serialize_mating_regime(FakeMating())
        assert config['type'] == 'FakeMating'

    def test_unknown_mating_type_deserialize_raises(self):
        with pytest.raises(ValueError, match="Unknown mating regime type"):
            _deserialize_mating_regime({'type': 'FakeMating'})


# ---------------------------------------------------------------------------
# Full checkpoint with different architectures
# ---------------------------------------------------------------------------

class TestCheckpointCoverage:
    def _make_sim(self, arch=None, n=20, m=10):
        from xftsim.founders import founder_haplotypes_uniform_AFs
        from xftsim.nmate import RandomMating
        from xftsim.reproduce import RecombinationMap
        from xftsim.nsim import NSimulation

        np.random.seed(42)
        hap = founder_haplotypes_uniform_AFs(n=n, m=m)

        if arch is None:
            eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
            arch = Architecture()
            arch.add('Y.G', GeneticComponent(eff))
            arch.add('Y.E', NoiseComponent(variance=0.5))
            arch.add('Y', AggregationComponent('Y.G + Y.E'))

        rm = RecombinationMap.constant_map(m=m, p=0.5)
        mating = RandomMating(offspring_per_pair=2)
        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=mating, recombination_map=rm, seed=42,
        )
        return sim

    def test_checkpoint_with_assortative_mating(self, tmp_path):
        from xftsim.nmate import LinearAssortativeMating
        from xftsim.reproduce import RecombinationMap
        from xftsim.nsim import NSimulation
        from xftsim.founders import founder_haplotypes_uniform_AFs

        np.random.seed(42)
        hap = founder_haplotypes_uniform_AFs(n=20, m=10)
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        rm = RecombinationMap.constant_map(m=10, p=0.5)
        mating = LinearAssortativeMating(
            component_names=['Y'], r=0.3, offspring_per_pair=2,
        )
        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=mating, recombination_map=rm, seed=42,
        )
        sim.run(2)

        ckpt_dir = str(tmp_path / "ckpt")
        save_simulation_checkpoint(sim, ckpt_dir)
        loaded = load_simulation_checkpoint(ckpt_dir)
        # run(2) = gen 0 and gen 1 → generation is 1
        assert loaded['generation'] == 1
        assert isinstance(loaded['mating_regime'], LinearAssortativeMating)

    def test_haplotype_save_load_full_metadata(self, tmp_path):
        """Roundtrip with all optional variant metadata."""
        geno = np.random.RandomState(42).randint(0, 2, (5, 3, 2)).astype(np.int8)
        samples = SampleMeta(iid=np.array(["a", "b", "c", "d", "e"]))
        variants = VariantMeta(
            vid=np.array(["v1", "v2", "v3"]),
            chrom=np.array([1, 1, 2]),
            pos_bp=np.array([100, 200, 300]),
            pos_cM=np.array([1.0, 2.0, 3.0]),
            af=np.array([0.3, 0.4, 0.5]),
            zero_allele=np.array(["A", "C", "G"]),
            one_allele=np.array(["T", "G", "C"]),
        )
        hap = DenseHaplotypeArray(genotypes=geno, samples=samples, variants=variants)
        path = str(tmp_path / "hap.npz")
        save_haplotypes_npz(hap, path)
        loaded = load_haplotypes_npz(path)
        assert np.array_equal(loaded.variants.chrom, [1, 1, 2])
        assert np.array_equal(loaded.variants.zero_allele, ["A", "C", "G"])
        assert np.array_equal(loaded.variants.one_allele, ["T", "G", "C"])
        assert np.allclose(loaded.variants.af, [0.3, 0.4, 0.5])
        assert np.allclose(loaded.variants.pos_cM, [1.0, 2.0, 3.0])

    def test_haplotype_save_load_minimal_metadata(self, tmp_path):
        """Roundtrip with no optional variant metadata."""
        geno = np.zeros((3, 2, 2), dtype=np.int8)
        hap = DenseHaplotypeArray(genotypes=geno)
        path = str(tmp_path / "hap_min.npz")
        save_haplotypes_npz(hap, path)
        loaded = load_haplotypes_npz(path)
        assert loaded.variants.chrom is None
        assert loaded.variants.pos_cM is None
