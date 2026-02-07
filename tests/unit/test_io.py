"""
I/O round-trip tests for haplotype, phenotype, and effect save/load.
"""
import numpy as np
import pytest
import tempfile
import os

from xftsim.struct import DenseHaplotypeArray, SampleMeta, VariantMeta, NPhenotypeArray
from xftsim.io import (
    save_haplotypes_npz, load_haplotypes_npz,
    save_phenotypes_npz, load_phenotypes_npz,
    save_effects_npz, load_effects_npz,
)
from xftsim.neffect import AdditiveEffects, MultivariateEffects, SparseEffects

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestHaplotypeIO:
    def test_roundtrip_genotypes(self, tmp_path):
        """Save and load should produce identical genotypes."""
        hap = TestSimulation.founder_haplotypes(n=50, m=20)
        path = str(tmp_path / "test_hap.npz")
        save_haplotypes_npz(hap, path)
        loaded = load_haplotypes_npz(path)
        np.testing.assert_array_equal(loaded.genotypes, hap.genotypes)

    def test_roundtrip_metadata(self, tmp_path):
        """Save and load should preserve sample and variant metadata."""
        hap = TestSimulation.founder_haplotypes(n=50, m=20)
        path = str(tmp_path / "test_hap.npz")
        save_haplotypes_npz(hap, path)
        loaded = load_haplotypes_npz(path)
        np.testing.assert_array_equal(loaded.iid, hap.iid)
        np.testing.assert_array_equal(loaded.fid, hap.fid)
        np.testing.assert_array_equal(loaded.sex, hap.sex)
        np.testing.assert_array_equal(loaded.vid, hap.vid)
        assert loaded.generation == hap.generation

    def test_roundtrip_with_af(self, tmp_path):
        """Allele frequencies should survive round-trip."""
        hap = TestSimulation.founder_haplotypes(n=50, m=20)
        path = str(tmp_path / "test_hap.npz")
        save_haplotypes_npz(hap, path)
        loaded = load_haplotypes_npz(path)
        np.testing.assert_array_almost_equal(loaded.af, hap.af)


class TestPhenotypeIO:
    def _make_pheno(self, n=50):
        samples = SampleMeta(iid=np.arange(n))
        pheno = NPhenotypeArray(samples=samples)
        rng = np.random.RandomState(42)
        pheno['Y.G'] = rng.randn(n)
        pheno['Y.E'] = rng.randn(n)
        pheno['Y'] = pheno['Y.G'] + pheno['Y.E']
        return pheno

    def test_roundtrip_values(self, tmp_path):
        """Save and load should produce identical phenotype values."""
        pheno = self._make_pheno()
        path = str(tmp_path / "test_pheno.npz")
        save_phenotypes_npz(pheno, path)
        loaded = load_phenotypes_npz(path)
        for key in pheno.keys:
            np.testing.assert_array_equal(loaded[key], pheno[key])

    def test_roundtrip_keys(self, tmp_path):
        """All keys should be preserved."""
        pheno = self._make_pheno()
        path = str(tmp_path / "test_pheno.npz")
        save_phenotypes_npz(pheno, path)
        loaded = load_phenotypes_npz(path)
        assert set(loaded.keys) == set(pheno.keys)

    def test_roundtrip_samples(self, tmp_path):
        """Sample metadata should be preserved."""
        pheno = self._make_pheno()
        path = str(tmp_path / "test_pheno.npz")
        save_phenotypes_npz(pheno, path)
        loaded = load_phenotypes_npz(path)
        np.testing.assert_array_equal(loaded.samples.iid, pheno.samples.iid)

    def test_roundtrip_from_simulation(self, tmp_path):
        """Phenotypes from a real simulation should round-trip."""
        from xftsim.nsim import NSimulation
        from xftsim.nmate import RandomMating
        from xftsim.reproduce import RecombinationMap

        hap = TestSimulation.founder_haplotypes(n=100, m=20, seed=42)
        arch = TestSimulation.simple_architecture(m=20, h2=0.5, seed=123)
        rmap = RecombinationMap.constant_map(m=20, p=0.5)
        mate = RandomMating(offspring_per_pair=2)
        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=mate, recombination_map=rmap, seed=42,
        )
        sim.run(1)
        pheno = sim.phenotype_history[0]

        path = str(tmp_path / "sim_pheno.npz")
        save_phenotypes_npz(pheno, path)
        loaded = load_phenotypes_npz(path)

        for key in pheno.keys:
            np.testing.assert_array_almost_equal(loaded[key], pheno[key])


class TestEffectsIO:
    def test_roundtrip_additive(self, tmp_path):
        """AdditiveEffects should round-trip exactly."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=100, seed=42)
        path = str(tmp_path / "eff.npz")
        save_effects_npz(eff, path)
        loaded = load_effects_npz(path)
        assert isinstance(loaded, AdditiveEffects)
        np.testing.assert_array_equal(loaded.effects, eff.effects)
        assert loaded.standardized == eff.standardized
        np.testing.assert_array_equal(loaded.variant_mask, eff.variant_mask)

    def test_roundtrip_multivariate(self, tmp_path):
        """MultivariateEffects should round-trip exactly."""
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.2, m=50, seed=42)
        path = str(tmp_path / "mv_eff.npz")
        save_effects_npz(eff, path)
        loaded = load_effects_npz(path)
        assert isinstance(loaded, MultivariateEffects)
        np.testing.assert_array_equal(loaded.effects, eff.effects)
        assert loaded.k == 2
        assert loaded.standardized == eff.standardized

    def test_roundtrip_sparse(self, tmp_path):
        """SparseEffects should round-trip exactly."""
        eff = SparseEffects.from_h2(h2=0.5, m=100, k_causal=10, seed=42)
        path = str(tmp_path / "sparse_eff.npz")
        save_effects_npz(eff, path)
        loaded = load_effects_npz(path)
        assert isinstance(loaded, SparseEffects)
        np.testing.assert_array_equal(loaded.effects, eff.effects)
        np.testing.assert_array_equal(loaded.variant_mask, eff.variant_mask)
        assert loaded.m_causal == 10

    def test_standardized_flag_preserved(self, tmp_path):
        """The standardized flag should survive round-trip."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=50, standardized=False, seed=42)
        path = str(tmp_path / "eff_unstd.npz")
        save_effects_npz(eff, path)
        loaded = load_effects_npz(path)
        assert loaded.standardized is False

    def test_properties_preserved(self, tmp_path):
        """m, m_causal, k properties should match after round-trip."""
        eff = SparseEffects.from_h2(h2=0.5, m=80, k_causal=15, seed=42)
        path = str(tmp_path / "eff_props.npz")
        save_effects_npz(eff, path)
        loaded = load_effects_npz(path)
        assert loaded.m == eff.m
        assert loaded.m_causal == eff.m_causal
        assert loaded.k == eff.k


class TestArchitectureIO:
    def test_roundtrip_simple(self, tmp_path):
        """Simple genetic + noise + aggregation should round-trip."""
        from xftsim.io import save_architecture, load_architecture
        from xftsim.narch import (
            Architecture, GeneticComponent, NoiseComponent, AggregationComponent,
        )
        eff = AdditiveEffects.from_h2(h2=0.5, m=20, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))

        dir_path = str(tmp_path / "arch_simple")
        save_architecture(arch, dir_path)
        loaded = load_architecture(dir_path)

        assert len(loaded.nodes) == 3
        assert loaded.nodes[0].outputs == ['Y.G']
        assert loaded.nodes[1].outputs == ['Y.E']
        assert loaded.nodes[2].outputs == ['Y']

    def test_roundtrip_effects_match(self, tmp_path):
        """Loaded architecture should have identical effects."""
        from xftsim.io import save_architecture, load_architecture
        from xftsim.narch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent

        eff = AdditiveEffects.from_h2(h2=0.5, m=20, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))

        dir_path = str(tmp_path / "arch_eff")
        save_architecture(arch, dir_path)
        loaded = load_architecture(dir_path)

        # GeneticComponent is the first node
        orig_eff = arch._nodes[0].component.effects
        loaded_eff = loaded._nodes[0].component.effects
        np.testing.assert_array_equal(loaded_eff.effects, orig_eff.effects)

    def test_roundtrip_multivariate(self, tmp_path):
        """MVGeneticComponent + CNoiseComponent should round-trip."""
        from xftsim.io import save_architecture, load_architecture
        from xftsim.narch import (
            Architecture, MVGeneticComponent, CNoiseComponent, AggregationComponent,
        )
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.2, m=20, seed=42)
        cov = np.array([[0.5, 0.1], [0.1, 0.7]])
        arch = Architecture()
        arch.add(['Y1.G', 'Y2.G'], MVGeneticComponent(eff))
        arch.add(['Y1.E', 'Y2.E'], CNoiseComponent(cov=cov))
        arch.add('Y1', AggregationComponent('Y1.G + Y1.E'))
        arch.add('Y2', AggregationComponent('Y2.G + Y2.E'))

        dir_path = str(tmp_path / "arch_mv")
        save_architecture(arch, dir_path)
        loaded = load_architecture(dir_path)

        assert len(loaded.nodes) == 4
        loaded_cov = loaded._nodes[1].component.cov
        np.testing.assert_array_almost_equal(loaded_cov, cov)

    def test_roundtrip_vertical_transmission(self, tmp_path):
        """ParentComponent should round-trip."""
        from xftsim.io import save_architecture, load_architecture
        from xftsim.narch import (
            Architecture, GeneticComponent, NoiseComponent,
            ParentComponent, AggregationComponent,
        )
        eff = AdditiveEffects.from_h2(h2=0.5, m=20, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.VT', ParentComponent('Y'))
        arch.add('Y.E', NoiseComponent(variance=0.3))
        arch.add('Y', AggregationComponent('Y.G + Y.VT + Y.E'))

        dir_path = str(tmp_path / "arch_vt")
        save_architecture(arch, dir_path)
        loaded = load_architecture(dir_path)

        assert len(loaded.nodes) == 4
        assert loaded._nodes[1].component.phenotype_name == 'Y'

    def test_roundtrip_produces_identical_phenotypes(self, tmp_path):
        """Loaded architecture should produce identical phenotypes."""
        from xftsim.io import save_architecture, load_architecture
        from xftsim.narch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent

        eff = AdditiveEffects.from_h2(h2=0.5, m=20, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))

        dir_path = str(tmp_path / "arch_compute")
        save_architecture(arch, dir_path)
        loaded = load_architecture(dir_path)

        hap = TestSimulation.founder_haplotypes(n=50, m=20, seed=42)
        rng1 = np.random.RandomState(99)
        rng2 = np.random.RandomState(99)
        p1 = arch.compute(hap, rng=rng1)
        p2 = loaded.compute(hap, rng=rng2)

        np.testing.assert_array_equal(p1['Y.G'], p2['Y.G'])
        np.testing.assert_array_equal(p1['Y.E'], p2['Y.E'])
        np.testing.assert_array_equal(p1['Y'], p2['Y'])


class TestSimulationCheckpoint:
    def _make_sim(self, m=20, n=100, seed=42):
        from xftsim.narch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
        from xftsim.nsim import NSimulation
        from xftsim.nmate import RandomMating
        from xftsim.reproduce import RecombinationMap

        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=seed)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))

        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        mate = RandomMating(offspring_per_pair=2)
        return NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=mate, recombination_map=rmap,
            retain_haplotypes=5, retain_phenotypes=5, seed=seed,
        )

    def test_checkpoint_roundtrip(self, tmp_path):
        """Checkpoint save/load should preserve generation and history keys."""
        from xftsim.io import save_simulation_checkpoint, load_simulation_checkpoint

        sim = self._make_sim()
        sim.run(3)
        dir_path = str(tmp_path / "checkpoint")
        save_simulation_checkpoint(sim, dir_path)
        loaded = load_simulation_checkpoint(dir_path)

        assert loaded['generation'] == sim.generation
        assert set(loaded['haplotype_history'].keys()) == set(sim.haplotype_history.keys())
        assert set(loaded['phenotype_history'].keys()) == set(sim.phenotype_history.keys())
        assert set(loaded['pedigree_history'].keys()) == set(sim.pedigree_history.keys())

    def test_checkpoint_phenotypes_match(self, tmp_path):
        """Loaded phenotypes should exactly match original."""
        from xftsim.io import save_simulation_checkpoint, load_simulation_checkpoint

        sim = self._make_sim()
        sim.run(2)
        dir_path = str(tmp_path / "checkpoint_pheno")
        save_simulation_checkpoint(sim, dir_path)
        loaded = load_simulation_checkpoint(dir_path)

        for gen in sim.phenotype_history:
            for key in sim.phenotype_history[gen].keys:
                np.testing.assert_array_almost_equal(
                    loaded['phenotype_history'][gen][key],
                    sim.phenotype_history[gen][key],
                )

    def test_checkpoint_haplotypes_match(self, tmp_path):
        """Loaded haplotypes should exactly match original."""
        from xftsim.io import save_simulation_checkpoint, load_simulation_checkpoint

        sim = self._make_sim()
        sim.run(2)
        dir_path = str(tmp_path / "checkpoint_hap")
        save_simulation_checkpoint(sim, dir_path)
        loaded = load_simulation_checkpoint(dir_path)

        for gen in sim.haplotype_history:
            np.testing.assert_array_equal(
                loaded['haplotype_history'][gen].genotypes,
                sim.haplotype_history[gen].genotypes,
            )

    def test_checkpoint_pedigree_match(self, tmp_path):
        """Loaded pedigrees should match original."""
        from xftsim.io import save_simulation_checkpoint, load_simulation_checkpoint

        sim = self._make_sim()
        sim.run(3)
        dir_path = str(tmp_path / "checkpoint_ped")
        save_simulation_checkpoint(sim, dir_path)
        loaded = load_simulation_checkpoint(dir_path)

        for gen in sim.pedigree_history:
            orig = sim.pedigree_history[gen]
            load = loaded['pedigree_history'][gen]
            np.testing.assert_array_equal(load.maternal_idx, orig.maternal_idx)
            np.testing.assert_array_equal(load.paternal_idx, orig.paternal_idx)
            assert load.parent_n == orig.parent_n

    def test_checkpoint_architecture_functional(self, tmp_path):
        """Loaded architecture should produce same genetic values."""
        from xftsim.io import save_simulation_checkpoint, load_simulation_checkpoint

        sim = self._make_sim()
        sim.run(1)
        dir_path = str(tmp_path / "checkpoint_arch")
        save_simulation_checkpoint(sim, dir_path)
        loaded = load_simulation_checkpoint(dir_path)

        hap = loaded['haplotype_history'][0]
        rng = np.random.RandomState(99)
        pheno = loaded['architecture'].compute(hap, rng=rng)
        # Y.G should match (deterministic given same genotypes and effects)
        np.testing.assert_array_almost_equal(
            pheno['Y.G'], sim.phenotype_history[0]['Y.G']
        )

    def test_checkpoint_recombination_map(self, tmp_path):
        """RecombinationMap should be saved and loaded."""
        from xftsim.io import save_simulation_checkpoint, load_simulation_checkpoint

        sim = self._make_sim()
        sim.run(1)
        dir_path = str(tmp_path / "checkpoint_rmap")
        save_simulation_checkpoint(sim, dir_path)
        loaded = load_simulation_checkpoint(dir_path)

        assert loaded['recombination_map'] is not None
        np.testing.assert_array_almost_equal(
            loaded['recombination_map']._probabilities,
            sim.recombination_map._probabilities,
        )

    def test_checkpoint_mating_regime(self, tmp_path):
        """Mating regime config should be saved and loaded."""
        from xftsim.io import save_simulation_checkpoint, load_simulation_checkpoint

        sim = self._make_sim()
        sim.run(1)
        dir_path = str(tmp_path / "checkpoint_mate")
        save_simulation_checkpoint(sim, dir_path)
        loaded = load_simulation_checkpoint(dir_path)

        assert loaded['mating_regime'] is not None
        from xftsim.nmate import RandomMating
        assert isinstance(loaded['mating_regime'], RandomMating)
        assert loaded['mating_regime'].offspring_per_pair == 2

    def test_checkpoint_resume(self, tmp_path):
        """Simulation should be resumable from checkpoint via from_checkpoint."""
        from xftsim.io import save_simulation_checkpoint
        from xftsim.nsim import NSimulation

        sim = self._make_sim()
        sim.run(3)
        dir_path = str(tmp_path / "checkpoint_resume")
        save_simulation_checkpoint(sim, dir_path)

        # Resume and run 2 more generations
        resumed = NSimulation.from_checkpoint(dir_path)
        assert resumed.generation == 2
        resumed.continue_run(2)
        assert resumed.generation == 4
        assert np.all(np.isfinite(resumed.phenotype_history[4]['Y']))

    def test_checkpoint_resume_preserves_gen0_yg(self, tmp_path):
        """Resumed simulation should have the same gen-0 Y.G (if still in history)."""
        from xftsim.io import save_simulation_checkpoint
        from xftsim.nsim import NSimulation

        sim = self._make_sim()
        sim.run(2)
        gen0_yg = sim.phenotype_history[0]['Y.G'].copy()

        dir_path = str(tmp_path / "checkpoint_yg")
        save_simulation_checkpoint(sim, dir_path)

        resumed = NSimulation.from_checkpoint(dir_path)
        np.testing.assert_array_almost_equal(
            resumed.phenotype_history[0]['Y.G'], gen0_yg
        )


# ── I/O error handling and edge cases ─────────────────────────────────────

class TestIOErrorHandling:
    def test_load_haplotypes_nonexistent_file(self):
        with pytest.raises((FileNotFoundError, OSError)):
            load_haplotypes_npz("/nonexistent/path.npz")

    def test_load_phenotypes_nonexistent_file(self):
        with pytest.raises((FileNotFoundError, OSError)):
            load_phenotypes_npz("/nonexistent/path.npz")

    def test_load_effects_nonexistent_file(self):
        with pytest.raises((FileNotFoundError, OSError)):
            load_effects_npz("/nonexistent/path.npz")

    def test_load_architecture_nonexistent_dir(self):
        from xftsim.io import load_architecture
        with pytest.raises((FileNotFoundError, OSError)):
            load_architecture("/nonexistent/dir")


class TestIOEdgeCases:
    def test_haplotypes_single_sample(self, tmp_path):
        """Single-sample haplotype should roundtrip."""
        geno = np.array([[[1, 0], [0, 1], [1, 1]]], dtype=np.int8)
        sm = SampleMeta(iid=np.array([0]))
        hap = DenseHaplotypeArray(genotypes=geno, samples=sm)
        path = str(tmp_path / "single.npz")
        save_haplotypes_npz(hap, path)
        loaded = load_haplotypes_npz(path)
        assert loaded.n == 1
        np.testing.assert_array_equal(loaded.genotypes, geno)

    def test_haplotypes_single_variant(self, tmp_path):
        """Single-variant haplotype should roundtrip."""
        geno = np.zeros((5, 1, 2), dtype=np.int8)
        hap = DenseHaplotypeArray(genotypes=geno)
        path = str(tmp_path / "single_var.npz")
        save_haplotypes_npz(hap, path)
        loaded = load_haplotypes_npz(path)
        assert loaded.m == 1
        np.testing.assert_array_equal(loaded.genotypes, geno)

    def test_phenotype_empty_keys(self, tmp_path):
        """Phenotype with no keys should roundtrip."""
        sm = SampleMeta(iid=np.arange(5))
        pheno = NPhenotypeArray(samples=sm)
        path = str(tmp_path / "empty_pheno.npz")
        save_phenotypes_npz(pheno, path)
        loaded = load_phenotypes_npz(path)
        assert len(list(loaded.keys)) == 0
        assert loaded.samples.n == 5

    def test_phenotype_many_keys(self, tmp_path):
        """Phenotype with many keys should roundtrip."""
        sm = SampleMeta(iid=np.arange(10))
        pheno = NPhenotypeArray(samples=sm)
        rng = np.random.RandomState(42)
        for i in range(20):
            pheno._values[f'trait_{i}'] = rng.randn(10)
        path = str(tmp_path / "many_keys.npz")
        save_phenotypes_npz(pheno, path)
        loaded = load_phenotypes_npz(path)
        assert set(loaded.keys) == set(pheno.keys)
        for key in pheno.keys:
            np.testing.assert_array_almost_equal(loaded[key], pheno[key])

    def test_effects_from_array_roundtrip(self, tmp_path):
        """Effects created from raw array should roundtrip."""
        arr = np.array([0.1, -0.2, 0.3, 0.0, 0.5])
        eff = AdditiveEffects.from_array(arr, standardized=False)
        path = str(tmp_path / "raw_eff.npz")
        save_effects_npz(eff, path)
        loaded = load_effects_npz(path)
        np.testing.assert_array_equal(loaded.effects, arr)
        assert loaded.standardized is False

    def test_checkpoint_retention_values_preserved(self, tmp_path):
        """Retention settings should be preserved in checkpoint."""
        from xftsim.io import save_simulation_checkpoint, load_simulation_checkpoint
        from xftsim.nsim import NSimulation
        from xftsim.nmate import RandomMating
        from xftsim.reproduce import RecombinationMap

        hap = TestSimulation.founder_haplotypes(n=100, m=20)
        arch = TestSimulation.simple_architecture(m=20, h2=0.5)
        rmap = RecombinationMap.constant_map(m=20, p=0.5)
        mate = RandomMating(offspring_per_pair=2)
        sim = NSimulation(
            hap, arch, mate, rmap, seed=42,
            retain_haplotypes=3, retain_phenotypes=7,
        )
        sim.run(2)
        dir_path = str(tmp_path / "retention")
        save_simulation_checkpoint(sim, dir_path)
        loaded = load_simulation_checkpoint(dir_path)
        assert loaded['retain_haplotypes'] == 3
        assert loaded['retain_phenotypes'] == 7

    def test_load_effects_unknown_class(self, tmp_path):
        """Loading effects with an unknown class name should raise ValueError."""
        # Create a valid effects file, then tamper with the class_name
        eff = AdditiveEffects.from_h2(h2=0.5, m=20, seed=42)
        path = str(tmp_path / "tampered.npz")
        np.savez_compressed(path,
            effects=eff.effects,
            standardized=np.array([eff.standardized]),
            variant_mask=eff.variant_mask,
            class_name=np.array(['FakeEffectsClass']),
        )
        with pytest.raises(ValueError, match="Unknown EffectSpec class"):
            load_effects_npz(path)

    def test_architecture_with_sibling_roundtrip(self, tmp_path):
        """Architecture with sibling component should roundtrip."""
        from xftsim.io import save_architecture, load_architecture
        from xftsim.narch import Architecture, NoiseComponent, SiblingMeanComponent, AggregationComponent

        arch = Architecture()
        arch.add('X', NoiseComponent(variance=1.0))
        arch.add('X.mean', SiblingMeanComponent('X'), inputs=['X'])

        dir_path = str(tmp_path / "arch_sibling")
        save_architecture(arch, dir_path)
        loaded = load_architecture(dir_path)

        assert len(loaded.nodes) == 2
        assert loaded.nodes[1].outputs == ['X.mean']


class TestGenotypesToPseudoHaplotypes:
    """Tests for the genotypes_to_pseudo_haplotypes function."""

    def test_homozygous_zero(self):
        """Genotype 0 should produce (0, 0) haplotypes."""
        from xftsim.io import genotypes_to_pseudo_haplotypes
        geno = np.array([[0, 0, 0]], dtype=np.int8)
        hap = genotypes_to_pseudo_haplotypes(geno)
        assert hap.shape == (1, 3, 2)
        np.testing.assert_array_equal(hap[0, :, 0] + hap[0, :, 1], [0, 0, 0])

    def test_homozygous_two(self):
        """Genotype 2 should produce (1, 1) haplotypes."""
        from xftsim.io import genotypes_to_pseudo_haplotypes
        geno = np.array([[2, 2, 2]], dtype=np.int8)
        hap = genotypes_to_pseudo_haplotypes(geno)
        np.testing.assert_array_equal(hap[0, :, 0], [1, 1, 1])
        np.testing.assert_array_equal(hap[0, :, 1], [1, 1, 1])

    def test_heterozygous_sum(self):
        """Genotype 1 should produce haplotypes summing to 1."""
        from xftsim.io import genotypes_to_pseudo_haplotypes
        geno = np.array([[1, 1, 1, 1, 1]], dtype=np.int8)
        hap = genotypes_to_pseudo_haplotypes(geno)
        sums = hap[0, :, 0] + hap[0, :, 1]
        np.testing.assert_array_equal(sums, [1, 1, 1, 1, 1])

    def test_output_shape(self):
        """Output should be (n, m, 2)."""
        from xftsim.io import genotypes_to_pseudo_haplotypes
        geno = np.zeros((10, 5), dtype=np.int8)
        hap = genotypes_to_pseudo_haplotypes(geno)
        assert hap.shape == (10, 5, 2)

    def test_output_dtype(self):
        """Output should be int8."""
        from xftsim.io import genotypes_to_pseudo_haplotypes
        geno = np.zeros((3, 3), dtype=np.int8)
        hap = genotypes_to_pseudo_haplotypes(geno)
        assert hap.dtype == np.int8

    def test_diploid_sum_preserved(self):
        """Sum across haplotypes should equal original genotype."""
        from xftsim.io import genotypes_to_pseudo_haplotypes
        rng = np.random.RandomState(42)
        geno = rng.choice([0, 1, 2], size=(20, 10)).astype(np.int8)
        hap = genotypes_to_pseudo_haplotypes(geno)
        diploid_sum = hap[:, :, 0] + hap[:, :, 1]
        np.testing.assert_array_equal(diploid_sum, geno)


class TestCheckpointCorruptionHandling:
    """Tests for corrupted/missing checkpoint files."""

    def test_checkpoint_missing_metadata(self, tmp_path):
        """Checkpoint with missing metadata.json should raise."""
        from xftsim.io import load_simulation_checkpoint
        dir_path = str(tmp_path / "bad_checkpoint")
        os.makedirs(dir_path)
        with pytest.raises((FileNotFoundError, OSError, KeyError)):
            load_simulation_checkpoint(dir_path)

    def test_checkpoint_corrupt_metadata(self, tmp_path):
        """Checkpoint with invalid JSON metadata should raise."""
        from xftsim.io import load_simulation_checkpoint
        dir_path = str(tmp_path / "corrupt_checkpoint")
        os.makedirs(dir_path)
        with open(os.path.join(dir_path, "meta.json"), "w") as f:
            f.write("not valid json {{{")
        with pytest.raises((json.JSONDecodeError, ValueError, KeyError)):
            load_simulation_checkpoint(dir_path)


import json
