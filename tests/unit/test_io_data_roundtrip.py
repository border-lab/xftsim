"""
Unit tests for I/O data roundtrip: haplotypes, phenotypes, effects.

Tests:
1. save/load_haplotypes_npz: full roundtrip with all metadata
2. save/load_haplotypes_npz: minimal (no optional variant fields)
3. save/load_phenotypes_npz: single key, multiple keys
4. save/load_effects_npz: AdditiveEffects, MultivariateEffects, SparseEffects
5. Phenotype roundtrip preserves key ordering
"""
import numpy as np
import pytest
import os

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray, PhenotypeArray
from xftsim.io import (
    save_haplotypes_npz, load_haplotypes_npz,
    save_phenotypes_npz, load_phenotypes_npz,
    save_effects_npz, load_effects_npz,
)
from xftsim.effect import AdditiveEffects, MultivariateEffects, SparseEffects


class TestHaplotypesRoundtrip:
    def test_full_metadata(self, tmp_path):
        """Haplotypes with all variant metadata should roundtrip."""
        n, m = 10, 5
        rng = np.random.RandomState(42)
        genotypes = rng.binomial(1, 0.3, size=(n, m, 2)).astype(np.int8)
        sm = SampleMeta(
            iid=np.arange(n),
            fid=np.arange(n) // 2,
            sex=np.tile([0, 1], n // 2),
        )
        vm = VariantMeta(
            vid=np.array([f'v{i}' for i in range(m)]),
            chrom=np.array([1, 1, 2, 2, 2]),
            pos_bp=np.array([100, 200, 300, 400, 500]),
            pos_cM=np.array([0.01, 0.02, 0.03, 0.04, 0.05]),
            af=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
            zero_allele=np.array(['A', 'C', 'G', 'T', 'A']),
            one_allele=np.array(['T', 'G', 'A', 'C', 'G']),
        )
        hap = DenseHaplotypeArray(genotypes=genotypes, generation=3, samples=sm, variants=vm)
        path = str(tmp_path / 'hap.npz')
        save_haplotypes_npz(hap, path)
        loaded = load_haplotypes_npz(path)

        assert loaded.generation == 3
        np.testing.assert_array_equal(loaded.genotypes, genotypes)
        np.testing.assert_array_equal(loaded.samples.iid, sm.iid)
        np.testing.assert_array_equal(loaded.samples.fid, sm.fid)
        np.testing.assert_array_equal(loaded.variants.vid, vm.vid)
        np.testing.assert_array_equal(loaded.variants.chrom, vm.chrom)
        np.testing.assert_array_equal(loaded.variants.pos_bp, vm.pos_bp)
        np.testing.assert_allclose(loaded.variants.pos_cM, vm.pos_cM)
        np.testing.assert_allclose(loaded.variants.af, vm.af)
        np.testing.assert_array_equal(loaded.variants.zero_allele, vm.zero_allele)
        np.testing.assert_array_equal(loaded.variants.one_allele, vm.one_allele)

    def test_minimal_metadata(self, tmp_path):
        """Haplotypes with only vid should roundtrip (optional fields stay None)."""
        n, m = 5, 3
        rng = np.random.RandomState(42)
        genotypes = rng.binomial(1, 0.5, size=(n, m, 2)).astype(np.int8)
        sm = SampleMeta(iid=np.arange(n))
        vm = VariantMeta(vid=np.array(['a', 'b', 'c']))
        hap = DenseHaplotypeArray(genotypes=genotypes, generation=0, samples=sm, variants=vm)
        path = str(tmp_path / 'hap_min.npz')
        save_haplotypes_npz(hap, path)
        loaded = load_haplotypes_npz(path)

        assert loaded.n == n
        assert loaded.m == m
        assert loaded.variants.chrom is None
        assert loaded.variants.pos_bp is None
        assert loaded.variants.af is None

    def test_genotype_dtype_preserved(self, tmp_path):
        """Genotypes should be int8 after roundtrip."""
        n, m = 3, 2
        genotypes = np.zeros((n, m, 2), dtype=np.int8)
        sm = SampleMeta(iid=np.arange(n))
        vm = VariantMeta(vid=np.array(['v0', 'v1']))
        hap = DenseHaplotypeArray(genotypes=genotypes, generation=0, samples=sm, variants=vm)
        path = str(tmp_path / 'hap_dtype.npz')
        save_haplotypes_npz(hap, path)
        loaded = load_haplotypes_npz(path)
        assert loaded.genotypes.dtype == np.int8


class TestPhenotypesRoundtrip:
    def test_single_key(self, tmp_path):
        """Single phenotype key roundtrip."""
        sm = SampleMeta(iid=np.arange(5))
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pheno = PhenotypeArray(samples=sm, values={'Y': vals})
        path = str(tmp_path / 'pheno.npz')
        save_phenotypes_npz(pheno, path)
        loaded = load_phenotypes_npz(path)

        assert 'Y' in loaded
        np.testing.assert_array_equal(loaded['Y'], vals)
        assert loaded.samples.n == 5

    def test_multiple_keys(self, tmp_path):
        """Multiple phenotype keys roundtrip."""
        sm = SampleMeta(iid=np.arange(10))
        vals = {
            'Y.G': np.random.randn(10),
            'Y.E': np.random.randn(10),
            'Y': np.random.randn(10),
        }
        pheno = PhenotypeArray(samples=sm, values=vals)
        path = str(tmp_path / 'pheno_multi.npz')
        save_phenotypes_npz(pheno, path)
        loaded = load_phenotypes_npz(path)

        for key in vals:
            assert key in loaded
            np.testing.assert_array_equal(loaded[key], vals[key])

    def test_empty_phenotype(self, tmp_path):
        """Empty phenotype (no keys) roundtrip."""
        sm = SampleMeta(iid=np.arange(5))
        pheno = PhenotypeArray(samples=sm)
        path = str(tmp_path / 'pheno_empty.npz')
        save_phenotypes_npz(pheno, path)
        loaded = load_phenotypes_npz(path)
        assert len(loaded.keys) == 0


class TestEffectsRoundtrip:
    def test_additive_roundtrip(self, tmp_path):
        """AdditiveEffects roundtrip."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        path = str(tmp_path / 'eff.npz')
        save_effects_npz(eff, path)
        loaded = load_effects_npz(path)

        assert isinstance(loaded, AdditiveEffects)
        np.testing.assert_array_equal(loaded.effects, eff.effects)
        assert loaded.standardized == eff.standardized
        np.testing.assert_array_equal(loaded.variant_mask, eff.variant_mask)

    def test_multivariate_roundtrip(self, tmp_path):
        """MultivariateEffects roundtrip."""
        eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.4, m=15, seed=42)
        path = str(tmp_path / 'mv_eff.npz')
        save_effects_npz(eff, path)
        loaded = load_effects_npz(path)

        assert isinstance(loaded, MultivariateEffects)
        np.testing.assert_array_equal(loaded.effects, eff.effects)
        assert loaded.k == 2
        assert loaded.m == 15

    def test_sparse_roundtrip(self, tmp_path):
        """SparseEffects roundtrip."""
        eff = SparseEffects.from_h2(h2=0.5, m=20, k_causal=5, seed=42)
        path = str(tmp_path / 'sparse_eff.npz')
        save_effects_npz(eff, path)
        loaded = load_effects_npz(path)

        assert isinstance(loaded, SparseEffects)
        assert loaded.m == 20
        assert loaded.m_causal == 5
        np.testing.assert_array_equal(loaded.effects, eff.effects)
        np.testing.assert_array_equal(loaded.variant_mask, eff.variant_mask)

    def test_not_standardized_roundtrip(self, tmp_path):
        """Effects with standardized=False should preserve that flag."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, standardized=False, seed=42)
        path = str(tmp_path / 'eff_raw.npz')
        save_effects_npz(eff, path)
        loaded = load_effects_npz(path)
        assert loaded.standardized is False
