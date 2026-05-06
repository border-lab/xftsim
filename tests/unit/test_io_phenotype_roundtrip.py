"""
Unit tests for phenotype and haplotype I/O roundtrip.

Tests:
1. Phenotype save/load preserves keys and values
2. Phenotype with multiple keys
3. Phenotype preserves sample metadata
4. Haplotype save/load preserves genotypes
5. Haplotype preserves metadata
6. Effect save/load with SparseEffects
7. Effect save/load with MultivariateEffects
"""
import numpy as np
import pytest
import tempfile

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray, NPhenotypeArray
from xftsim.effect import AdditiveEffects, MultivariateEffects, SparseEffects
from xftsim.io import (
    save_phenotypes_npz, load_phenotypes_npz,
    save_haplotypes_npz, load_haplotypes_npz,
    save_effects_npz, load_effects_npz,
)


class TestPhenotypeIO:
    def test_single_key_roundtrip(self):
        sm = SampleMeta(iid=np.arange(10))
        pheno = NPhenotypeArray(sm)
        pheno['Y'] = np.random.RandomState(42).randn(10)

        with tempfile.NamedTemporaryFile(suffix='.npz') as f:
            save_phenotypes_npz(pheno, f.name)
            loaded = load_phenotypes_npz(f.name)
            assert 'Y' in loaded
            np.testing.assert_allclose(loaded['Y'], pheno['Y'])

    def test_multiple_keys_roundtrip(self):
        sm = SampleMeta(iid=np.arange(5))
        pheno = NPhenotypeArray(sm)
        pheno['A'] = np.ones(5)
        pheno['B'] = np.ones(5) * 2
        pheno['C'] = np.ones(5) * 3

        with tempfile.NamedTemporaryFile(suffix='.npz') as f:
            save_phenotypes_npz(pheno, f.name)
            loaded = load_phenotypes_npz(f.name)
            assert set(loaded.keys) == {'A', 'B', 'C'}
            np.testing.assert_allclose(loaded['A'], 1.0)
            np.testing.assert_allclose(loaded['C'], 3.0)

    def test_preserves_sample_meta(self):
        sm = SampleMeta(
            iid=np.array([10, 20, 30]),
            fid=np.array([1, 1, 2]),
            sex=np.array([0, 1, 0]),
        )
        pheno = NPhenotypeArray(sm)
        pheno['Y'] = np.array([1.0, 2.0, 3.0])

        with tempfile.NamedTemporaryFile(suffix='.npz') as f:
            save_phenotypes_npz(pheno, f.name)
            loaded = load_phenotypes_npz(f.name)
            np.testing.assert_array_equal(loaded.samples.iid, sm.iid)
            np.testing.assert_array_equal(loaded.samples.fid, sm.fid)
            np.testing.assert_array_equal(loaded.samples.sex, sm.sex)


class TestHaplotypeIO:
    def test_roundtrip_preserves_genotypes(self):
        rng = np.random.RandomState(42)
        geno = rng.randint(0, 2, size=(20, 10, 2)).astype(np.int8)
        sm = SampleMeta(iid=np.arange(20))
        vm = VariantMeta(vid=np.array([f'v{i}' for i in range(10)]))
        hap = DenseHaplotypeArray(geno, samples=sm, variants=vm)

        with tempfile.NamedTemporaryFile(suffix='.npz') as f:
            save_haplotypes_npz(hap, f.name)
            loaded = load_haplotypes_npz(f.name)
            assert loaded.n == 20
            assert loaded.m == 10
            np.testing.assert_array_equal(loaded.genotypes, geno)

    def test_roundtrip_preserves_metadata(self):
        geno = np.zeros((5, 3, 2), dtype=np.int8)
        sm = SampleMeta(iid=np.array([100, 200, 300, 400, 500]))
        vm = VariantMeta(vid=np.array(['rs1', 'rs2', 'rs3']))
        hap = DenseHaplotypeArray(geno, samples=sm, variants=vm)

        with tempfile.NamedTemporaryFile(suffix='.npz') as f:
            save_haplotypes_npz(hap, f.name)
            loaded = load_haplotypes_npz(f.name)
            np.testing.assert_array_equal(loaded.samples.iid, sm.iid)
            np.testing.assert_array_equal(loaded.variants.vid, vm.vid)


class TestEffectIO:
    def test_additive_roundtrip(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        with tempfile.NamedTemporaryFile(suffix='.npz') as f:
            save_effects_npz(eff, f.name)
            loaded = load_effects_npz(f.name)
            assert isinstance(loaded, AdditiveEffects)
            np.testing.assert_allclose(loaded.effects, eff.effects)
            assert loaded.standardized == eff.standardized

    def test_multivariate_roundtrip(self):
        mv = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.2, m=10, seed=42)
        with tempfile.NamedTemporaryFile(suffix='.npz') as f:
            save_effects_npz(mv, f.name)
            loaded = load_effects_npz(f.name)
            assert isinstance(loaded, MultivariateEffects)
            assert loaded.k == 2
            np.testing.assert_allclose(loaded.effects, mv.effects)

    def test_sparse_roundtrip(self):
        sp = SparseEffects.from_h2(h2=0.5, m=20, k_causal=5, seed=42)
        with tempfile.NamedTemporaryFile(suffix='.npz') as f:
            save_effects_npz(sp, f.name)
            loaded = load_effects_npz(f.name)
            assert isinstance(loaded, SparseEffects)
            np.testing.assert_allclose(loaded.effects, sp.effects)
            np.testing.assert_array_equal(loaded.variant_mask, sp.variant_mask)
