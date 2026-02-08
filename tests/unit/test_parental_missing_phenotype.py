"""
Unit tests for parental component with missing phenotype name in prev gen.

Tests:
1. MotherComponent raises when phenotype_name not in prev_gen phenotypes
2. FatherComponent raises when phenotype_name not in prev_gen phenotypes
3. ParentComponent (midparent) raises when phenotype_name not in prev_gen
"""
import numpy as np
import pytest
import warnings

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray, NPhenotypeArray, PedigreeArray
from xftsim.narch import ArchNode, MotherComponent, FatherComponent, ParentComponent


def _make_context():
    """Create test context with parents and offspring."""
    n_parent = 10
    n_offspring = 6
    # Parent phenotype (only has 'A', not 'B')
    parent_sm = SampleMeta(iid=np.arange(n_parent))
    parent_pheno = NPhenotypeArray(
        samples=parent_sm,
        values={'A': np.ones(n_parent)},
    )
    # Offspring
    offspring_sm = SampleMeta(iid=np.arange(n_offspring), generation=1)
    offspring_vm = VariantMeta(vid=np.array(['v0', 'v1', 'v2']))
    offspring_geno = np.zeros((n_offspring, 3, 2), dtype=np.int8)
    offspring_hap = DenseHaplotypeArray(
        genotypes=offspring_geno, samples=offspring_sm, variants=offspring_vm,
    )
    offspring_pheno = NPhenotypeArray(samples=offspring_sm)
    # Pedigree
    maternal_idx = np.array([0, 1, 2, 3, 4, 0])
    paternal_idx = np.array([5, 6, 7, 8, 9, 5])
    ped = PedigreeArray(
        offspring_samples=offspring_sm,
        maternal_idx=maternal_idx,
        paternal_idx=paternal_idx,
        parent_n=n_parent,
    )
    return offspring_hap, offspring_pheno, parent_pheno, ped


class TestParentalMissingPhenotype:
    def test_mother_missing_phenotype_raises(self):
        """MotherComponent should raise if phenotype not in prev gen."""
        hap, pheno, parent_pheno, ped = _make_context()
        comp = MotherComponent('MISSING')
        node = ArchNode(outputs=['VT'], component=comp, inputs=[])
        with pytest.raises(ValueError, match="phenotype 'MISSING' not found"):
            comp.compute(
                node, hap, pheno,
                generation=1,
                phenotype_history={0: parent_pheno},
                pedigree_history={1: ped},
            )

    def test_father_missing_phenotype_raises(self):
        """FatherComponent should raise if phenotype not in prev gen."""
        hap, pheno, parent_pheno, ped = _make_context()
        comp = FatherComponent('MISSING')
        node = ArchNode(outputs=['VT'], component=comp, inputs=[])
        with pytest.raises(ValueError, match="phenotype 'MISSING' not found"):
            comp.compute(
                node, hap, pheno,
                generation=1,
                phenotype_history={0: parent_pheno},
                pedigree_history={1: ped},
            )

    def test_parent_missing_phenotype_raises(self):
        """ParentComponent should raise if phenotype not in prev gen."""
        hap, pheno, parent_pheno, ped = _make_context()
        comp = ParentComponent('MISSING')
        node = ArchNode(outputs=['VT'], component=comp, inputs=[])
        with pytest.raises(ValueError, match="phenotype 'MISSING' not found"):
            comp.compute(
                node, hap, pheno,
                generation=1,
                phenotype_history={0: parent_pheno},
                pedigree_history={1: ped},
            )

    def test_mother_existing_phenotype_works(self):
        """MotherComponent should work when phenotype exists."""
        hap, pheno, parent_pheno, ped = _make_context()
        comp = MotherComponent('A')
        node = ArchNode(outputs=['VT'], component=comp, inputs=[])
        result = comp.compute(
            node, hap, pheno,
            generation=1,
            phenotype_history={0: parent_pheno},
            pedigree_history={1: ped},
        )
        assert len(result) == 6
        assert np.all(result == 1.0)  # parent values are all 1.0
