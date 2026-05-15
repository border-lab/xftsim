"""
Unit tests for HaplotypeGeneticComponent.

Tests:
1. Invalid haplotype value raises
2. Maternal vs paternal computation
3. Maternal + paternal = diploid genetic value
4. repr
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray
from xftsim.arch import HaplotypeGeneticComponent, ArchNode
from xftsim.effect import AdditiveEffects


def _make_hap(n=10, m=5, seed=42):
    rng = np.random.RandomState(seed)
    sm = SampleMeta(iid=np.arange(n), fid=np.arange(n) // 2)
    vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
    geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
    return DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)


class TestHaplotypeGeneticValidation:
    def test_invalid_haplotype_raises(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=5, seed=42)
        with pytest.raises(ValueError, match="'maternal' or 'paternal'"):
            HaplotypeGeneticComponent(effects=eff, haplotype='both')

    def test_maternal_default(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=5, seed=42)
        comp = HaplotypeGeneticComponent(effects=eff)
        assert comp.haplotype == 'maternal'


class TestHaplotypeGeneticCompute:
    def test_maternal_computation(self):
        hap = _make_hap()
        eff = AdditiveEffects.from_h2(h2=0.5, m=5, seed=42)
        comp = HaplotypeGeneticComponent(effects=eff, haplotype='maternal')
        node = ArchNode(outputs=['Y.Hm'], component=comp, inputs=[])
        from xftsim.struct import PhenotypeArray
        pheno = PhenotypeArray(samples=hap.samples)
        result = comp.compute(node, hap, pheno)
        # Manual: hap[:,:,0] @ effects
        expected = hap.genotypes[:, :, 0].astype(float) @ eff.effects
        np.testing.assert_allclose(result, expected)

    def test_paternal_computation(self):
        hap = _make_hap()
        eff = AdditiveEffects.from_h2(h2=0.5, m=5, seed=42)
        comp = HaplotypeGeneticComponent(effects=eff, haplotype='paternal')
        node = ArchNode(outputs=['Y.Hp'], component=comp, inputs=[])
        from xftsim.struct import PhenotypeArray
        pheno = PhenotypeArray(samples=hap.samples)
        result = comp.compute(node, hap, pheno)
        expected = hap.genotypes[:, :, 1].astype(float) @ eff.effects
        np.testing.assert_allclose(result, expected)

    def test_maternal_plus_paternal_equals_diploid(self):
        """Haploid maternal + paternal = diploid genetic value."""
        hap = _make_hap()
        eff = AdditiveEffects.from_h2(h2=0.5, m=5, seed=42)
        from xftsim.struct import PhenotypeArray
        pheno = PhenotypeArray(samples=hap.samples)

        comp_m = HaplotypeGeneticComponent(effects=eff, haplotype='maternal')
        comp_p = HaplotypeGeneticComponent(effects=eff, haplotype='paternal')
        node_m = ArchNode(outputs=['Y.Hm'], component=comp_m, inputs=[])
        node_p = ArchNode(outputs=['Y.Hp'], component=comp_p, inputs=[])

        val_m = comp_m.compute(node_m, hap, pheno)
        val_p = comp_p.compute(node_p, hap, pheno)

        # Diploid: full genotype @ effects
        diploid = hap.diploid_genotypes.astype(float) @ eff.effects
        np.testing.assert_allclose(val_m + val_p, diploid)


class TestHaplotypeGeneticRepr:
    def test_repr(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=5, seed=42)
        comp = HaplotypeGeneticComponent(effects=eff, haplotype='maternal')
        r = repr(comp)
        assert 'HaplotypeGeneticComponent' in r
        assert 'maternal' in r
