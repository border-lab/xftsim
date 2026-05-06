"""
Unit tests for _ParentalComponent, MotherComponent, FatherComponent, ParentComponent.

Tests:
1. Gen-0 returns zeros (no pedigree)
2. Gen-0 with founder_component uses fallback
3. Missing phenotype in prev gen raises ValueError
4. Missing prev_gen phenotype (retention pruned) returns zeros with warning
5. MotherComponent uses maternal_idx
6. FatherComponent uses paternal_idx
7. ParentComponent averages mother and father
8. repr
"""
import numpy as np
import pytest
import warnings

from xftsim.struct import SampleMeta, NPhenotypeArray, PedigreeArray
from xftsim.arch import (
    ArchNode, MotherComponent, FatherComponent, ParentComponent, NoiseComponent,
)

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


def _make_context(n_offspring=4, n_parent=10, seed=42):
    """Create haplotypes, phenotype_history, and pedigree_history for testing."""
    hap = TestSimulation.founder_haplotypes(n=n_offspring, m=5, seed=seed)
    parent_sm = SampleMeta(iid=np.arange(n_parent), generation=0)
    parent_pheno = NPhenotypeArray(
        samples=parent_sm,
        values={'Y': np.arange(n_parent, dtype=float)},
    )
    offspring_sm = SampleMeta(iid=np.arange(n_offspring), generation=1)
    ped = PedigreeArray(
        offspring_samples=offspring_sm,
        maternal_idx=np.array([0, 0, 2, 2]),
        paternal_idx=np.array([1, 1, 3, 3]),
        parent_n=n_parent,
    )
    phenotype_history = {0: parent_pheno}
    pedigree_history = {1: ped}
    return hap, phenotype_history, pedigree_history


class TestGen0Fallback:
    def test_gen0_returns_zeros(self):
        """At generation 0, parental components return zeros with warning."""
        hap = TestSimulation.founder_haplotypes(n=5, m=3, seed=42)
        comp = MotherComponent('Y')
        node = ArchNode(outputs=['Y.m'], component=comp, inputs=['Y'])
        pheno = NPhenotypeArray(samples=hap.samples)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = comp.compute(node, hap, pheno, generation=0)
            assert len(w) == 1
            assert "no pedigree" in str(w[0].message)
        np.testing.assert_array_equal(result, np.zeros(5))

    def test_gen0_with_founder_component(self):
        """At generation 0, founder_component should be used as fallback."""
        hap = TestSimulation.founder_haplotypes(n=5, m=3, seed=42)
        fallback = NoiseComponent(variance=1.0)
        comp = MotherComponent('Y', founder_component=fallback)
        node = ArchNode(outputs=['Y.m'], component=comp, inputs=['Y'])
        pheno = NPhenotypeArray(samples=hap.samples)
        rng = np.random.RandomState(42)
        result = comp.compute(node, hap, pheno, generation=0, rng=rng)
        assert result.shape == (5,)
        assert not np.all(result == 0)  # noise, not zeros


class TestRetentionPrunedFallback:
    def test_pruned_gen_returns_zeros(self):
        """Missing phenotype_history entry returns zeros with warning."""
        hap = TestSimulation.founder_haplotypes(n=4, m=3, seed=42)
        offspring_sm = SampleMeta(iid=np.arange(4), generation=2)
        ped = PedigreeArray(
            offspring_samples=offspring_sm,
            maternal_idx=np.array([0, 0, 2, 2]),
            paternal_idx=np.array([1, 1, 3, 3]),
            parent_n=10,
        )
        comp = MotherComponent('Y')
        node = ArchNode(outputs=['Y.m'], component=comp, inputs=['Y'])
        pheno = NPhenotypeArray(samples=hap.samples)
        # Generation 2, pedigree exists, but prev gen phenotypes pruned
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = comp.compute(
                node, hap, pheno,
                generation=2,
                phenotype_history={},  # empty — pruned
                pedigree_history={2: ped},
            )
            assert len(w) == 1
            assert "not in phenotype_history" in str(w[0].message)
        np.testing.assert_array_equal(result, np.zeros(4))


class TestMissingPhenotype:
    def test_missing_phenotype_raises(self):
        """Phenotype not found in prev gen should raise ValueError."""
        hap, pheno_hist, ped_hist = _make_context()
        comp = MotherComponent('NONEXISTENT')
        node = ArchNode(outputs=['Y.m'], component=comp, inputs=['NONEXISTENT'])
        pheno = NPhenotypeArray(samples=hap.samples)
        with pytest.raises(ValueError, match="not found"):
            comp.compute(
                node, hap, pheno,
                generation=1,
                phenotype_history=pheno_hist,
                pedigree_history=ped_hist,
            )


class TestMotherComponent:
    def test_uses_maternal_idx(self):
        """MotherComponent should index with maternal_idx."""
        hap, pheno_hist, ped_hist = _make_context()
        comp = MotherComponent('Y')
        node = ArchNode(outputs=['Y.m'], component=comp, inputs=['Y'])
        pheno = NPhenotypeArray(samples=hap.samples)
        result = comp.compute(
            node, hap, pheno,
            generation=1,
            phenotype_history=pheno_hist,
            pedigree_history=ped_hist,
        )
        # maternal_idx = [0, 0, 2, 2], parent values = [0, 1, 2, ...]
        np.testing.assert_array_equal(result, [0.0, 0.0, 2.0, 2.0])

    def test_shape(self):
        hap, pheno_hist, ped_hist = _make_context()
        comp = MotherComponent('Y')
        node = ArchNode(outputs=['Y.m'], component=comp, inputs=['Y'])
        pheno = NPhenotypeArray(samples=hap.samples)
        result = comp.compute(
            node, hap, pheno,
            generation=1,
            phenotype_history=pheno_hist,
            pedigree_history=ped_hist,
        )
        assert result.shape == (4,)


class TestFatherComponent:
    def test_uses_paternal_idx(self):
        """FatherComponent should index with paternal_idx."""
        hap, pheno_hist, ped_hist = _make_context()
        comp = FatherComponent('Y')
        node = ArchNode(outputs=['Y.f'], component=comp, inputs=['Y'])
        pheno = NPhenotypeArray(samples=hap.samples)
        result = comp.compute(
            node, hap, pheno,
            generation=1,
            phenotype_history=pheno_hist,
            pedigree_history=ped_hist,
        )
        # paternal_idx = [1, 1, 3, 3]
        np.testing.assert_array_equal(result, [1.0, 1.0, 3.0, 3.0])


class TestParentComponent:
    def test_midparent(self):
        """ParentComponent should average mother and father."""
        hap, pheno_hist, ped_hist = _make_context()
        comp = ParentComponent('Y')
        node = ArchNode(outputs=['Y.p'], component=comp, inputs=['Y'])
        pheno = NPhenotypeArray(samples=hap.samples)
        result = comp.compute(
            node, hap, pheno,
            generation=1,
            phenotype_history=pheno_hist,
            pedigree_history=ped_hist,
        )
        # midparent = 0.5 * (mother + father)
        # maternal_idx=[0,0,2,2], paternal_idx=[1,1,3,3]
        # values = [0,1,2,...] → (0+1)/2=0.5, (0+1)/2=0.5, (2+3)/2=2.5, (2+3)/2=2.5
        np.testing.assert_allclose(result, [0.5, 0.5, 2.5, 2.5])


class TestParentalRepr:
    def test_mother_repr(self):
        comp = MotherComponent('Y')
        assert "MotherComponent" in repr(comp)
        assert "'Y'" in repr(comp)

    def test_father_repr(self):
        comp = FatherComponent('height')
        assert "FatherComponent" in repr(comp)
        assert "'height'" in repr(comp)

    def test_parent_repr(self):
        comp = ParentComponent('BMI')
        assert "ParentComponent" in repr(comp)
