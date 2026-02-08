"""
Unit tests for haplotype and phenotype NPZ save/load roundtrips.

Tests:
1. Haplotype roundtrip preserves genotypes
2. Haplotype roundtrip preserves sample metadata
3. Haplotype roundtrip preserves variant metadata
4. Haplotype roundtrip with optional fields (chrom, pos_bp, alleles)
5. Phenotype roundtrip preserves values
6. Phenotype roundtrip preserves multiple keys
7. Effects roundtrip preserves effect arrays
8. Single sample/variant roundtrip
"""
import numpy as np
import pytest
import tempfile
import os

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray, NPhenotypeArray
from xftsim.neffect import AdditiveEffects, MultivariateEffects
from xftsim.io import (
    save_haplotypes_npz, load_haplotypes_npz,
    save_phenotypes_npz, load_phenotypes_npz,
    save_effects_npz, load_effects_npz,
)


def _make_hap(n=10, m=5, seed=42, chrom=None, pos_bp=None, generation=0):
    sm = SampleMeta(iid=np.arange(n), fid=np.arange(n) // 2, generation=generation)
    kwargs = {'vid': np.array([f'v{i}' for i in range(m)])}
    if chrom is not None:
        kwargs['chrom'] = chrom
    if pos_bp is not None:
        kwargs['pos_bp'] = pos_bp
    vm = VariantMeta(**kwargs)
    rng = np.random.RandomState(seed)
    geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
    return DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm, generation=generation)


class TestHaplotypeRoundtrip:
    def test_genotypes_preserved(self):
        """Genotype data matches after save/load."""
        hap = _make_hap()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'hap.npz')
            save_haplotypes_npz(hap, path)
            loaded = load_haplotypes_npz(path)
            np.testing.assert_array_equal(loaded.genotypes, hap.genotypes)

    def test_sample_metadata_preserved(self):
        """Sample IID and FID preserved."""
        hap = _make_hap()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'hap.npz')
            save_haplotypes_npz(hap, path)
            loaded = load_haplotypes_npz(path)
            np.testing.assert_array_equal(loaded.samples.iid, hap.samples.iid)
            np.testing.assert_array_equal(loaded.samples.fid, hap.samples.fid)

    def test_variant_metadata_preserved(self):
        """Variant IDs preserved."""
        hap = _make_hap()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'hap.npz')
            save_haplotypes_npz(hap, path)
            loaded = load_haplotypes_npz(path)
            np.testing.assert_array_equal(loaded.variants.vid, hap.variants.vid)

    def test_generation_preserved(self):
        """Generation counter preserved."""
        hap = _make_hap(generation=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'hap.npz')
            save_haplotypes_npz(hap, path)
            loaded = load_haplotypes_npz(path)
            assert loaded.generation == 5

    def test_optional_variant_fields(self):
        """Optional chrom and pos_bp preserved."""
        chrom = np.array(['1', '1', '2', '2', '3'])
        pos_bp = np.array([100, 200, 100, 200, 100])
        hap = _make_hap(chrom=chrom, pos_bp=pos_bp)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'hap.npz')
            save_haplotypes_npz(hap, path)
            loaded = load_haplotypes_npz(path)
            np.testing.assert_array_equal(loaded.variants.chrom, chrom)
            np.testing.assert_array_equal(loaded.variants.pos_bp, pos_bp)

    def test_single_sample(self):
        """n=1 roundtrip."""
        hap = _make_hap(n=1, m=5)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'hap.npz')
            save_haplotypes_npz(hap, path)
            loaded = load_haplotypes_npz(path)
            assert loaded.n == 1
            np.testing.assert_array_equal(loaded.genotypes, hap.genotypes)

    def test_single_variant(self):
        """m=1 roundtrip."""
        hap = _make_hap(n=10, m=1)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'hap.npz')
            save_haplotypes_npz(hap, path)
            loaded = load_haplotypes_npz(path)
            assert loaded.m == 1


class TestPhenotypeRoundtrip:
    def test_values_preserved(self):
        """Phenotype values match after save/load."""
        sm = SampleMeta(iid=np.arange(5))
        pheno = NPhenotypeArray(samples=sm)
        pheno._values['Y'] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'pheno.npz')
            save_phenotypes_npz(pheno, path)
            loaded = load_phenotypes_npz(path)
            np.testing.assert_array_equal(loaded['Y'], pheno['Y'])

    def test_multiple_keys_preserved(self):
        """Multiple phenotype keys preserved."""
        sm = SampleMeta(iid=np.arange(5))
        pheno = NPhenotypeArray(samples=sm)
        pheno._values['A'] = np.ones(5)
        pheno._values['B'] = np.zeros(5)
        pheno._values['C'] = np.arange(5, dtype=float)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'pheno.npz')
            save_phenotypes_npz(pheno, path)
            loaded = load_phenotypes_npz(path)
            assert set(loaded.keys) == {'A', 'B', 'C'}
            np.testing.assert_array_equal(loaded['A'], np.ones(5))
            np.testing.assert_array_equal(loaded['B'], np.zeros(5))


class TestEffectsRoundtrip:
    def test_additive_effects_roundtrip(self):
        """AdditiveEffects preserved after save/load."""
        effects = AdditiveEffects.from_h2(m=20, h2=0.5, seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'effects.npz')
            save_effects_npz(effects, path)
            loaded = load_effects_npz(path)
            np.testing.assert_allclose(loaded.effects, effects.effects)

    def test_multivariate_effects_roundtrip(self):
        """MultivariateEffects preserved after save/load."""
        effects = MultivariateEffects.from_h2_rg(m=20, h2=[0.5, 0.3], rg=0.5, seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'effects.npz')
            save_effects_npz(effects, path)
            loaded = load_effects_npz(path)
            np.testing.assert_allclose(loaded.effects, effects.effects)
            assert loaded.effects.shape == effects.effects.shape
