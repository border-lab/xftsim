"""
Unit tests for parental component edge cases.

Tests:
1. MotherComponent gen 0 no founder → zeros with warning
2. MotherComponent gen 0 with founder_component → noise fallback
3. MotherComponent missing prev gen phenotype_history → zeros with warning
4. MotherComponent phenotype name not in prev gen → ValueError
5. FatherComponent founder fallback returns noise
6. ParentComponent gen 0 no pedigree → zeros
"""
import numpy as np
import pytest
import warnings

from xftsim.narch import (
    MotherComponent, FatherComponent, ParentComponent,
    NoiseComponent, ArchNode,
)
from xftsim.struct import SampleMeta, NPhenotypeArray, PedigreeArray, DenseHaplotypeArray, VariantMeta


def _dummy_hap(n, m=5):
    sm = SampleMeta(iid=np.arange(n))
    vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
    return DenseHaplotypeArray(np.zeros((n, m, 2), dtype=np.int8), samples=sm, variants=vm)


class TestParentalFounderFallback:
    def test_gen0_no_founder_returns_zeros(self):
        """Generation 0 with no founder_component → zeros + warning."""
        hap = _dummy_hap(10)
        pheno = NPhenotypeArray(hap.samples)
        comp = MotherComponent('Y')
        node = ArchNode(outputs=['Y.m'], component=comp, inputs=[], grouping=None)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = comp.compute(node, hap, pheno, generation=0)
            assert len(w) == 1
            assert 'returning zeros' in str(w[0].message)
        np.testing.assert_array_equal(result, np.zeros(10))

    def test_gen0_with_founder_uses_fallback(self):
        """Generation 0 with founder_component → noise fallback."""
        hap = _dummy_hap(10)
        pheno = NPhenotypeArray(hap.samples)
        founder = NoiseComponent(variance=1.0)
        comp = MotherComponent('Y', founder_component=founder)
        node = ArchNode(outputs=['Y.m'], component=comp, inputs=[], grouping=None)

        rng = np.random.RandomState(42)
        result = comp.compute(node, hap, pheno, generation=0, rng=rng)
        assert result.shape == (10,)
        assert np.all(np.isfinite(result))
        # Should not be all zeros (noise drawn)
        assert not np.allclose(result, 0.0)

    def test_missing_prev_gen_phenotype_returns_zeros(self):
        """When prev gen was pruned from history → zeros + warning."""
        hap = _dummy_hap(4)
        pheno = NPhenotypeArray(hap.samples)
        parent_sm = SampleMeta(iid=np.arange(10), generation=0)
        ped = PedigreeArray(
            offspring_samples=hap.samples,
            maternal_idx=np.array([0, 0, 1, 1]),
            paternal_idx=np.array([5, 5, 6, 6]),
            parent_n=10,
        )
        comp = MotherComponent('Y')
        node = ArchNode(outputs=['Y.m'], component=comp, inputs=[], grouping=None)

        # Provide pedigree but NO phenotype_history for gen 0
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = comp.compute(
                node, hap, pheno,
                generation=1,
                pedigree_history={1: ped},
                phenotype_history={},  # gen 0 pruned
            )
            assert len(w) == 1
            assert 'not in phenotype_history' in str(w[0].message)
        np.testing.assert_array_equal(result, np.zeros(4))

    def test_phenotype_name_not_found_raises(self):
        """Missing phenotype name in prev gen → ValueError."""
        parent_sm = SampleMeta(iid=np.arange(10), generation=0)
        parent_pheno = NPhenotypeArray(parent_sm)
        parent_pheno['X'] = np.ones(10)  # Has 'X' but not 'Y'

        offspring_sm = SampleMeta(iid=np.arange(4), generation=1)
        hap = _dummy_hap(4)
        pheno = NPhenotypeArray(offspring_sm)
        ped = PedigreeArray(
            offspring_samples=offspring_sm,
            maternal_idx=np.array([0, 0, 1, 1]),
            paternal_idx=np.array([5, 5, 6, 6]),
            parent_n=10,
        )

        comp = MotherComponent('Y')
        node = ArchNode(outputs=['Y.m'], component=comp, inputs=[], grouping=None)

        with pytest.raises(ValueError, match="not found"):
            comp.compute(
                node, hap, pheno,
                generation=1,
                pedigree_history={1: ped},
                phenotype_history={0: parent_pheno},
            )

    def test_father_founder_noise(self):
        """FatherComponent with noise founder at gen 0."""
        hap = _dummy_hap(8)
        pheno = NPhenotypeArray(hap.samples)
        founder = NoiseComponent(variance=2.0)
        comp = FatherComponent('Y', founder_component=founder)
        node = ArchNode(outputs=['Y.f'], component=comp, inputs=[], grouping=None)

        rng = np.random.RandomState(99)
        result = comp.compute(node, hap, pheno, generation=0, rng=rng)
        assert result.shape == (8,)
        assert np.all(np.isfinite(result))

    def test_no_pedigree_at_gen_returns_zeros(self):
        """ParentComponent at gen 2 but no pedigree for gen 2 → zeros."""
        hap = _dummy_hap(5)
        pheno = NPhenotypeArray(hap.samples)
        comp = ParentComponent('Y')
        node = ArchNode(outputs=['Y.p'], component=comp, inputs=[], grouping=None)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = comp.compute(
                node, hap, pheno,
                generation=2,
                pedigree_history={1: None},  # gen 2 not in history
            )
            assert len(w) == 1
        np.testing.assert_array_equal(result, np.zeros(5))
