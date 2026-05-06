"""
Unit tests for I/O module edge cases.

Tests:
1. save/load haplotypes roundtrip: genotypes, samples, variants preserved
2. save/load haplotypes: generation preserved
3. save/load phenotypes with many keys (10+): all preserved
4. save/load effects: AdditiveEffects, MultivariateEffects, SparseEffects each roundtrip
5. load from nonexistent directory: appropriate error
6. Architecture with sibling components: serializes/deserializes correctly
7. Architecture with cnoise components: cov matrix preserved
8. Checkpoint with assortative mating: mating regime serialized
9. Checkpoint with retention: only retained generations saved
"""
import numpy as np
import pytest
import tempfile
import os
import sys
import shutil

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray, NPhenotypeArray
from xftsim.effect import AdditiveEffects, MultivariateEffects, SparseEffects
from xftsim.arch import (
    Architecture, GeneticComponent, NoiseComponent, CNoiseComponent,
    AggregationComponent, SiblingMeanComponent, SiblingSumComponent,
    MotherComponent,
)
from xftsim.mate import RandomMating, LinearAssortativeMating
from xftsim.reproduce import RecombinationMap
from xftsim.sim import NSimulation
from xftsim.io import (
    save_haplotypes_npz, load_haplotypes_npz,
    save_phenotypes_npz, load_phenotypes_npz,
    save_effects_npz, load_effects_npz,
    save_architecture, load_architecture,
    save_simulation_checkpoint, load_simulation_checkpoint,
)


class TestHaplotypesRoundtrip:
    def test_genotypes_samples_variants_preserved(self):
        """Haplotype roundtrip preserves genotypes, samples, and variants."""
        n, m = 100, 50
        rng = np.random.RandomState(42)
        geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)

        iid = np.arange(n)
        fid = np.repeat(np.arange(n // 10), 10)
        sex = np.tile([0, 1], n // 2)
        samples = SampleMeta(iid=iid, fid=fid, sex=sex)

        vid = np.array([f'rs{i}' for i in range(m)])
        chrom = np.repeat([1, 2], m // 2)
        pos_bp = np.arange(1000, 1000 + m * 1000, 1000)
        af = rng.uniform(0.1, 0.9, m)
        variants = VariantMeta(vid=vid, chrom=chrom, pos_bp=pos_bp, af=af)

        hap = DenseHaplotypeArray(genotypes=geno, samples=samples, variants=variants)

        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, 'haplotypes.npz')
            save_haplotypes_npz(hap, path)
            loaded = load_haplotypes_npz(path)

            # Check genotypes
            assert loaded.n == n
            assert loaded.m == m
            np.testing.assert_array_equal(loaded.genotypes, geno)

            # Check samples
            np.testing.assert_array_equal(loaded.samples.iid, iid)
            np.testing.assert_array_equal(loaded.samples.fid, fid)
            np.testing.assert_array_equal(loaded.samples.sex, sex)

            # Check variants
            np.testing.assert_array_equal(loaded.variants.vid, vid)
            np.testing.assert_array_equal(loaded.variants.chrom, chrom)
            np.testing.assert_array_equal(loaded.variants.pos_bp, pos_bp)
            np.testing.assert_allclose(loaded.variants.af, af)
        finally:
            shutil.rmtree(tmpdir)

    def test_generation_preserved(self):
        """Haplotype roundtrip preserves generation number."""
        n, m = 20, 10
        geno = np.zeros((n, m, 2), dtype=np.int8)
        hap = DenseHaplotypeArray(genotypes=geno, generation=5)

        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, 'hap_gen5.npz')
            save_haplotypes_npz(hap, path)
            loaded = load_haplotypes_npz(path)
            assert loaded.generation == 5
        finally:
            shutil.rmtree(tmpdir)


class TestPhenotypeManyKeys:
    def test_many_keys_preserved(self):
        """Phenotype with 10+ keys all preserved in roundtrip."""
        n = 50
        samples = SampleMeta(iid=np.arange(n))
        pheno = NPhenotypeArray(samples)

        rng = np.random.RandomState(42)
        keys = [f'trait_{i}' for i in range(15)]
        for key in keys:
            pheno[key] = rng.randn(n)

        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, 'pheno_many.npz')
            save_phenotypes_npz(pheno, path)
            loaded = load_phenotypes_npz(path)

            assert len(loaded.keys) == 15
            assert set(loaded.keys) == set(keys)
            for key in keys:
                np.testing.assert_allclose(loaded[key], pheno[key])
        finally:
            shutil.rmtree(tmpdir)


class TestEffectsRoundtrip:
    def test_additive_effects_roundtrip(self):
        """AdditiveEffects roundtrip preserves all data."""
        m = 100
        eff = AdditiveEffects.from_h2(h2=0.6, m=m, seed=42)

        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, 'additive.npz')
            save_effects_npz(eff, path)
            loaded = load_effects_npz(path)

            assert isinstance(loaded, AdditiveEffects)
            np.testing.assert_allclose(loaded.effects, eff.effects)
            assert loaded.standardized == eff.standardized
            np.testing.assert_array_equal(loaded.variant_mask, eff.variant_mask)
        finally:
            shutil.rmtree(tmpdir)

    def test_multivariate_effects_roundtrip(self):
        """MultivariateEffects roundtrip preserves all data."""
        m = 100
        h2 = [0.5, 0.3, 0.4]
        rg = 0.25
        eff = MultivariateEffects.from_h2_rg(h2=h2, rg=rg, m=m, seed=42)

        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, 'multivariate.npz')
            save_effects_npz(eff, path)
            loaded = load_effects_npz(path)

            assert isinstance(loaded, MultivariateEffects)
            assert loaded.k == 3
            np.testing.assert_allclose(loaded.effects, eff.effects)
            assert loaded.standardized == eff.standardized
        finally:
            shutil.rmtree(tmpdir)

    def test_sparse_effects_roundtrip(self):
        """SparseEffects roundtrip preserves all data including mask."""
        m = 200
        k_causal = 20
        eff = SparseEffects.from_h2(h2=0.7, m=m, k_causal=k_causal, seed=42)

        tmpdir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmpdir, 'sparse.npz')
            save_effects_npz(eff, path)
            loaded = load_effects_npz(path)

            assert isinstance(loaded, SparseEffects)
            np.testing.assert_allclose(loaded.effects, eff.effects)
            np.testing.assert_array_equal(loaded.variant_mask, eff.variant_mask)
            assert loaded.variant_mask.sum() == k_causal
        finally:
            shutil.rmtree(tmpdir)


class TestLoadNonexistentDirectory:
    def test_load_architecture_nonexistent(self):
        """Loading architecture from nonexistent directory raises error."""
        tmpdir = tempfile.mkdtemp()
        nonexistent = os.path.join(tmpdir, 'does_not_exist')
        try:
            with pytest.raises((FileNotFoundError, OSError)):
                load_architecture(nonexistent)
        finally:
            shutil.rmtree(tmpdir)

    def test_load_checkpoint_nonexistent(self):
        """Loading checkpoint from nonexistent directory raises error."""
        tmpdir = tempfile.mkdtemp()
        nonexistent = os.path.join(tmpdir, 'does_not_exist')
        try:
            with pytest.raises((FileNotFoundError, OSError)):
                load_simulation_checkpoint(nonexistent)
        finally:
            shutil.rmtree(tmpdir)


class TestArchitectureSiblingComponents:
    def test_sibling_mean_component(self):
        """Architecture with SiblingMeanComponent serializes correctly."""
        m = 20
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])
        arch.add('Y.sib', SiblingMeanComponent('Y'), inputs=['Y'], grouping='FID')

        tmpdir = tempfile.mkdtemp()
        try:
            save_architecture(arch, tmpdir)
            loaded = load_architecture(tmpdir)

            assert len(loaded._nodes) == 4
            sib_node = loaded._nodes[3]
            assert isinstance(sib_node.component, SiblingMeanComponent)
            assert sib_node.component.source_name == 'Y'
            assert sib_node.grouping == 'FID'
            assert sib_node.outputs == ['Y.sib']
        finally:
            shutil.rmtree(tmpdir)

    def test_sibling_sum_component(self):
        """Architecture with SiblingSumComponent serializes correctly."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('A.sib_sum', SiblingSumComponent('A'), inputs=['A'], grouping='FID')

        tmpdir = tempfile.mkdtemp()
        try:
            save_architecture(arch, tmpdir)
            loaded = load_architecture(tmpdir)

            assert len(loaded._nodes) == 2
            sib_node = loaded._nodes[1]
            assert isinstance(sib_node.component, SiblingSumComponent)
            assert sib_node.component.source_name == 'A'
        finally:
            shutil.rmtree(tmpdir)


class TestArchitectureCNoiseComponent:
    def test_cnoise_cov_matrix_preserved(self):
        """CNoiseComponent covariance matrix preserved in roundtrip."""
        cov = np.array([
            [1.0, 0.3, 0.1],
            [0.3, 1.0, 0.2],
            [0.1, 0.2, 1.0],
        ])
        arch = Architecture()
        arch.add(['E1', 'E2', 'E3'], CNoiseComponent(cov=cov))

        tmpdir = tempfile.mkdtemp()
        try:
            save_architecture(arch, tmpdir)
            loaded = load_architecture(tmpdir)

            assert len(loaded._nodes) == 1
            comp = loaded._nodes[0].component
            assert isinstance(comp, CNoiseComponent)
            np.testing.assert_allclose(comp.cov, cov)
            assert comp.cov.shape == (3, 3)
        finally:
            shutil.rmtree(tmpdir)


class TestCheckpointAssortativeMating:
    def test_assortative_mating_serialized(self):
        """Checkpoint with LinearAssortativeMating preserves regime."""
        n, m = 50, 20
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)

        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])

        mating = LinearAssortativeMating(
            component_names=['Y'], r=0.6, offspring_per_pair=2,
        )

        sim = NSimulation(
            founder_haplotypes=hap,
            architecture=arch,
            mating_regime=mating,
            recombination_map=RecombinationMap.constant_map(m=m),
            seed=42,
        )
        sim.run(2)

        tmpdir = tempfile.mkdtemp()
        try:
            save_simulation_checkpoint(sim, tmpdir)
            loaded = load_simulation_checkpoint(tmpdir)

            mating_loaded = loaded['mating_regime']
            assert isinstance(mating_loaded, LinearAssortativeMating)
            assert mating_loaded.r == 0.6
            assert mating_loaded.component_names == ['Y']
            assert mating_loaded.offspring_per_pair == 2
        finally:
            shutil.rmtree(tmpdir)


class TestCheckpointRetention:
    def test_only_retained_generations_saved(self):
        """Checkpoint with retention only saves retained generations."""
        n, m = 50, 20
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)

        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])

        sim = NSimulation(
            founder_haplotypes=hap,
            architecture=arch,
            mating_regime=RandomMating(offspring_per_pair=2),
            recombination_map=RecombinationMap.constant_map(m=m),
            seed=42,
            retain_haplotypes=1,  # Keep last 1 generation
            retain_phenotypes=2,  # Keep last 2 generations
        )
        sim.run(5)  # Run 5 generations: 0, 1, 2, 3, 4 (current gen becomes 4)

        tmpdir = tempfile.mkdtemp()
        try:
            save_simulation_checkpoint(sim, tmpdir)
            loaded = load_simulation_checkpoint(tmpdir)

            # Check that only retained generations are saved
            hap_gens = sorted(loaded['haplotype_history'].keys())
            pheno_gens = sorted(loaded['phenotype_history'].keys())

            # With retain_haplotypes=1 at gen 4: keep g >= 4-1=3, so gens 3,4
            assert len(hap_gens) == 2
            assert hap_gens == [3, 4]

            # With retain_phenotypes=2 at gen 4: keep g >= 4-2=2, so gens 2,3,4
            assert len(pheno_gens) == 3
            assert pheno_gens == [2, 3, 4]

            # Verify loaded state
            assert loaded['generation'] == 4
            assert loaded['retain_haplotypes'] == 1
            assert loaded['retain_phenotypes'] == 2
        finally:
            shutil.rmtree(tmpdir)

    def test_retention_large_value_keeps_more(self):
        """Checkpoint with large retention value keeps more generations."""
        n, m = 30, 15
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)

        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'), inputs=['Y.G', 'Y.E'])

        sim = NSimulation(
            founder_haplotypes=hap,
            architecture=arch,
            mating_regime=RandomMating(offspring_per_pair=2),
            recombination_map=RecombinationMap.constant_map(m=m),
            seed=42,
            retain_haplotypes=10,  # Keep last 10 (more than we'll run)
            retain_phenotypes=10,  # Keep last 10
        )
        sim.run(3)  # Run 3 generations: 0, 1, 2 (current gen becomes 2)

        tmpdir = tempfile.mkdtemp()
        try:
            save_simulation_checkpoint(sim, tmpdir)
            loaded = load_simulation_checkpoint(tmpdir)

            hap_gens = sorted(loaded['haplotype_history'].keys())
            pheno_gens = sorted(loaded['phenotype_history'].keys())

            # With retain=10 at gen 2: keep g >= 2-10=-8, so all generations
            # run(3) creates gens 0,1,2 so we should have all 3
            assert len(hap_gens) == 3
            assert hap_gens == [0, 1, 2]
            assert len(pheno_gens) == 3
            assert pheno_gens == [0, 1, 2]
        finally:
            shutil.rmtree(tmpdir)
