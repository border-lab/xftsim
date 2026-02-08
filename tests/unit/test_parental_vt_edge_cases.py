"""
Unit tests for _ParentalComponent (VT) edge cases.

Tests:
1. MotherComponent at gen 0 with founder_component
2. MotherComponent at gen 0 without founder_component (warns, returns zeros)
3. FatherComponent with pruned parent history (warns, returns zeros)
4. ParentComponent (midparent) correctness
5. Missing phenotype in previous generation raises ValueError
6. Founder component delegates to noise
7. Normal operation with pedigree
8. Repr for all three subclasses
"""
import numpy as np
import pytest
import warnings

from xftsim.narch import (
    MotherComponent, FatherComponent, ParentComponent,
    NoiseComponent, ArchNode,
)
from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray, NPhenotypeArray, PedigreeArray


def _make_hap(n=4, m=1):
    sm = SampleMeta(iid=np.arange(n))
    vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
    geno = np.ones((n, m, 2), dtype=np.int8)
    return DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)


class TestMotherComponentEdgeCases:
    def test_gen0_no_founder_warns(self):
        """At generation 0 with no founder component, warns and returns zeros."""
        comp = MotherComponent('Y')
        hap = _make_hap(n=4)
        pheno = NPhenotypeArray(samples=hap.samples)
        node = ArchNode(outputs=['Y.VT'], component=comp, inputs=[])

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = comp.compute(node, hap, pheno, generation=0)
            assert len(w) == 1
            assert "no pedigree" in str(w[0].message).lower()

        np.testing.assert_array_equal(result, np.zeros(4))

    def test_gen0_with_founder_delegates(self):
        """At generation 0 with founder component, delegates to founder."""
        noise_comp = NoiseComponent(variance=1.0)
        comp = MotherComponent('Y', founder_component=noise_comp)
        hap = _make_hap(n=4)
        pheno = NPhenotypeArray(samples=hap.samples)
        node = ArchNode(outputs=['Y.VT'], component=comp, inputs=[])

        rng = np.random.RandomState(42)
        result = comp.compute(node, hap, pheno, generation=0, rng=rng)
        assert result.shape == (4,)
        # Noise should be non-zero (with overwhelming probability)
        assert not np.all(result == 0)

    def test_normal_operation(self):
        """At gen > 0 with pedigree, returns mother's phenotype values."""
        comp = MotherComponent('Y')
        hap = _make_hap(n=4)
        pheno = NPhenotypeArray(samples=hap.samples)
        node = ArchNode(outputs=['Y.VT'], component=comp, inputs=[])

        # Create parent phenotypes
        parent_sm = SampleMeta(iid=np.arange(6))
        parent_pheno = NPhenotypeArray(samples=parent_sm)
        parent_pheno._values['Y'] = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

        # Create pedigree
        offspring_sm = SampleMeta(iid=np.arange(4), generation=1)
        ped = PedigreeArray(
            offspring_samples=offspring_sm,
            maternal_idx=np.array([0, 0, 2, 4]),
            paternal_idx=np.array([1, 1, 3, 5]),
            parent_n=6,
        )

        result = comp.compute(
            node, hap, pheno,
            generation=1,
            phenotype_history={0: parent_pheno},
            pedigree_history={1: ped},
        )
        np.testing.assert_array_equal(result, [10.0, 10.0, 30.0, 50.0])


class TestFatherComponentEdgeCases:
    def test_pruned_history_warns(self):
        """Previous generation phenotypes pruned → warns and returns zeros."""
        comp = FatherComponent('Y')
        hap = _make_hap(n=4)
        pheno = NPhenotypeArray(samples=hap.samples)
        node = ArchNode(outputs=['Y.VT'], component=comp, inputs=[])

        offspring_sm = SampleMeta(iid=np.arange(4), generation=2)
        ped = PedigreeArray(
            offspring_samples=offspring_sm,
            maternal_idx=np.array([0, 0, 2, 2]),
            paternal_idx=np.array([1, 1, 3, 3]),
            parent_n=10,
        )

        # Gen 2, pedigree exists, but gen 1 phenotypes NOT in history
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = comp.compute(
                node, hap, pheno,
                generation=2,
                phenotype_history={},  # pruned
                pedigree_history={2: ped},
            )
            assert len(w) == 1
            assert "pruned" in str(w[0].message).lower() or "not in phenotype_history" in str(w[0].message).lower()

        np.testing.assert_array_equal(result, np.zeros(4))

    def test_normal_operation(self):
        """Returns father's phenotype values."""
        comp = FatherComponent('Y')
        hap = _make_hap(n=4)
        pheno = NPhenotypeArray(samples=hap.samples)
        node = ArchNode(outputs=['Y.VT'], component=comp, inputs=[])

        parent_sm = SampleMeta(iid=np.arange(6))
        parent_pheno = NPhenotypeArray(samples=parent_sm)
        parent_pheno._values['Y'] = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

        offspring_sm = SampleMeta(iid=np.arange(4), generation=1)
        ped = PedigreeArray(
            offspring_samples=offspring_sm,
            maternal_idx=np.array([0, 0, 2, 4]),
            paternal_idx=np.array([1, 1, 3, 5]),
            parent_n=6,
        )

        result = comp.compute(
            node, hap, pheno,
            generation=1,
            phenotype_history={0: parent_pheno},
            pedigree_history={1: ped},
        )
        np.testing.assert_array_equal(result, [20.0, 20.0, 40.0, 60.0])


class TestParentComponentEdgeCases:
    def test_midparent_average(self):
        """ParentComponent returns 0.5*(mother + father)."""
        comp = ParentComponent('Y')
        hap = _make_hap(n=4)
        pheno = NPhenotypeArray(samples=hap.samples)
        node = ArchNode(outputs=['Y.VT'], component=comp, inputs=[])

        parent_sm = SampleMeta(iid=np.arange(6))
        parent_pheno = NPhenotypeArray(samples=parent_sm)
        parent_pheno._values['Y'] = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

        offspring_sm = SampleMeta(iid=np.arange(4), generation=1)
        ped = PedigreeArray(
            offspring_samples=offspring_sm,
            maternal_idx=np.array([0, 0, 2, 4]),
            paternal_idx=np.array([1, 1, 3, 5]),
            parent_n=6,
        )

        result = comp.compute(
            node, hap, pheno,
            generation=1,
            phenotype_history={0: parent_pheno},
            pedigree_history={1: ped},
        )
        # 0.5*(10+20), 0.5*(10+20), 0.5*(30+40), 0.5*(50+60)
        np.testing.assert_array_equal(result, [15.0, 15.0, 35.0, 55.0])

    def test_missing_phenotype_raises(self):
        """Phenotype not found in previous gen → ValueError."""
        comp = ParentComponent('NONEXISTENT')
        hap = _make_hap(n=2)
        pheno = NPhenotypeArray(samples=hap.samples)
        node = ArchNode(outputs=['X'], component=comp, inputs=[])

        parent_sm = SampleMeta(iid=np.arange(4))
        parent_pheno = NPhenotypeArray(samples=parent_sm)
        parent_pheno._values['Y'] = np.array([1.0, 2.0, 3.0, 4.0])

        offspring_sm = SampleMeta(iid=np.arange(2), generation=1)
        ped = PedigreeArray(
            offspring_samples=offspring_sm,
            maternal_idx=np.array([0, 2]),
            paternal_idx=np.array([1, 3]),
            parent_n=4,
        )

        with pytest.raises(ValueError, match="not found"):
            comp.compute(
                node, hap, pheno,
                generation=1,
                phenotype_history={0: parent_pheno},
                pedigree_history={1: ped},
            )


class TestParentalRepr:
    def test_mother_repr(self):
        comp = MotherComponent('Y')
        assert "MotherComponent" in repr(comp)
        assert "'Y'" in repr(comp)

    def test_father_repr(self):
        comp = FatherComponent('Y')
        assert "FatherComponent" in repr(comp)

    def test_parent_repr(self):
        comp = ParentComponent('Y')
        assert "ParentComponent" in repr(comp)
