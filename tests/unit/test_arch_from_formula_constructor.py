"""
Unit tests for Architecture constructor with formula= parameter.

Tests:
1. Constructor with formula= directly (not from_formula classmethod)
2. Constructor with formula and effects
3. Nodes are pre-sorted in constructor
4. Architecture with no formula (empty)
5. Non-standardized effects in simulation
"""
import numpy as np
import pytest

from xftsim.arch import Architecture, GeneticComponent, NoiseComponent
from xftsim.effect import AdditiveEffects


class TestArchitectureConstructorFormula:
    def test_constructor_with_formula(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        arch = Architecture(
            formula="Y.G ~ genetic(eff)\nY.E ~ noise(0.5)\nY ~ Y.G + Y.E",
            effects={'eff': eff},
        )
        assert len(arch.nodes) == 3
        outputs = [n.outputs[0] for n in arch.nodes]
        assert 'Y' in outputs
        assert 'Y.G' in outputs
        assert 'Y.E' in outputs

    def test_constructor_nodes_pre_sorted(self):
        """Nodes should be topologically sorted when created via constructor."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        arch = Architecture(
            formula="Y ~ Y.G + Y.E\nY.G ~ genetic(eff)\nY.E ~ noise(0.5)",
            effects={'eff': eff},
        )
        outputs = [n.outputs[0] for n in arch.nodes]
        # Y depends on Y.G and Y.E, so must come last
        assert outputs.index('Y') > outputs.index('Y.G')
        assert outputs.index('Y') > outputs.index('Y.E')

    def test_constructor_no_formula(self):
        arch = Architecture()
        assert len(arch.nodes) == 0
        assert arch._sorted is not None or len(arch._nodes) == 0

    def test_from_formula_equivalent_to_constructor(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        formula = "Y.G ~ genetic(eff)\nY.E ~ noise(0.5)\nY ~ Y.G + Y.E"
        arch1 = Architecture(formula=formula, effects={'eff': eff})
        arch2 = Architecture.from_formula(formula, effects={'eff': eff})
        # Same number of nodes
        assert len(arch1.nodes) == len(arch2.nodes)
        # Same output names
        out1 = [n.outputs[0] for n in arch1.nodes]
        out2 = [n.outputs[0] for n in arch2.nodes]
        assert out1 == out2


class TestNonStandardizedEffects:
    def test_non_standardized_genetic(self):
        """Non-standardized effects should use matvec instead of standardized_matvec."""
        from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray
        eff = AdditiveEffects.from_h2(h2=0.5, m=5, standardized=False, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        n = 10
        rng = np.random.RandomState(42)
        sm = SampleMeta(iid=np.arange(n), fid=np.arange(n) // 2)
        vm = VariantMeta(vid=np.array([f'v{i}' for i in range(5)]))
        geno = rng.randint(0, 2, size=(n, 5, 2)).astype(np.int8)
        hap = DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)
        pheno = arch.compute(hap)
        # Non-standardized: matvec = G @ effects (no centering)
        expected = hap.matvec(eff.effects)
        np.testing.assert_allclose(pheno['Y.G'], expected)

    def test_standardized_genetic(self):
        """Standardized effects should use standardized_matvec."""
        from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray
        eff = AdditiveEffects.from_h2(h2=0.5, m=5, standardized=True, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        n = 10
        rng = np.random.RandomState(42)
        sm = SampleMeta(iid=np.arange(n), fid=np.arange(n) // 2)
        vm = VariantMeta(vid=np.array([f'v{i}' for i in range(5)]))
        geno = rng.randint(0, 2, size=(n, 5, 2)).astype(np.int8)
        hap = DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)
        pheno = arch.compute(hap)
        expected = hap.standardized_matvec(eff.effects)
        np.testing.assert_allclose(pheno['Y.G'], expected)
