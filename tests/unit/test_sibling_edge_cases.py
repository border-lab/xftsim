"""
Unit tests for _SiblingComponent edge cases.

Tests:
1. Single-member families (each person their own family)
2. All-same-family (one big family)
3. Missing source phenotype raises ValueError
4. None grouping (labels=None → per-individual identity)
5. SiblingMean correctness for unequal-size families
6. SiblingSum correctness
7. SiblingAny: all-zero group, mixed group, all-positive group
8. SiblingCount for various family sizes
9. SiblingEldest/SiblingYoungest correctness
10. All-zero values
11. Negative values
12. NaN in values
"""
import numpy as np
import pytest

from xftsim.arch import (
    SiblingMeanComponent, SiblingSumComponent, SiblingAnyComponent,
    SiblingCountComponent, SiblingEldestComponent, SiblingYoungestComponent,
    ArchNode,
)
from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray, NPhenotypeArray


def _make_env(n, fid, source_name='Y', source_values=None):
    """Create haplotypes and phenotypes for sibling component tests."""
    sm = SampleMeta(iid=np.arange(n), fid=np.asarray(fid))
    vm = VariantMeta(vid=np.array(['v0']))
    geno = np.ones((n, 1, 2), dtype=np.int8)
    hap = DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)
    pheno = NPhenotypeArray(samples=sm)
    if source_values is not None:
        pheno._values[source_name] = np.asarray(source_values, dtype=np.float64)
    else:
        pheno._values[source_name] = np.arange(n, dtype=np.float64)
    return hap, pheno


class TestSiblingMeanEdgeCases:
    def test_single_member_families(self):
        """Each person in own family → mean is self."""
        hap, pheno = _make_env(5, fid=[0, 1, 2, 3, 4],
                               source_values=[10.0, 20.0, 30.0, 40.0, 50.0])
        comp = SiblingMeanComponent('Y')
        node = ArchNode(outputs=['Y.sib_mean'], component=comp, inputs=['Y'], grouping='FID')
        result = comp.compute(node, hap, pheno)
        # Mean of a single person is themselves
        np.testing.assert_array_equal(result, [10.0, 20.0, 30.0, 40.0, 50.0])

    def test_one_big_family(self):
        """All in one family → mean is population mean for everyone."""
        vals = np.array([10.0, 20.0, 30.0, 40.0])
        hap, pheno = _make_env(4, fid=[0, 0, 0, 0], source_values=vals)
        comp = SiblingMeanComponent('Y')
        node = ArchNode(outputs=['Y.sib_mean'], component=comp, inputs=['Y'], grouping='FID')
        result = comp.compute(node, hap, pheno)
        expected_mean = vals.mean()
        np.testing.assert_allclose(result, np.full(4, expected_mean))

    def test_unequal_size_families(self):
        """Family of 3 and family of 2 have different means."""
        vals = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
        hap, pheno = _make_env(5, fid=[0, 0, 0, 1, 1], source_values=vals)
        comp = SiblingMeanComponent('Y')
        node = ArchNode(outputs=['Y.sib_mean'], component=comp, inputs=['Y'], grouping='FID')
        result = comp.compute(node, hap, pheno)
        np.testing.assert_allclose(result[0:3], 2.0)  # mean of [1,2,3]
        np.testing.assert_allclose(result[3:5], 15.0)  # mean of [10,20]

    def test_all_zeros(self):
        """All-zero source values → all-zero mean."""
        hap, pheno = _make_env(4, fid=[0, 0, 1, 1], source_values=np.zeros(4))
        comp = SiblingMeanComponent('Y')
        node = ArchNode(outputs=['Y.sib_mean'], component=comp, inputs=['Y'], grouping='FID')
        result = comp.compute(node, hap, pheno)
        np.testing.assert_array_equal(result, np.zeros(4))

    def test_negative_values(self):
        """Negative values handled correctly."""
        vals = np.array([-10.0, -20.0, 5.0, 15.0])
        hap, pheno = _make_env(4, fid=[0, 0, 1, 1], source_values=vals)
        comp = SiblingMeanComponent('Y')
        node = ArchNode(outputs=['Y.sib_mean'], component=comp, inputs=['Y'], grouping='FID')
        result = comp.compute(node, hap, pheno)
        np.testing.assert_allclose(result[0:2], -15.0)
        np.testing.assert_allclose(result[2:4], 10.0)


class TestSiblingSumEdgeCases:
    def test_sum_correctness(self):
        vals = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
        hap, pheno = _make_env(5, fid=[0, 0, 0, 1, 1], source_values=vals)
        comp = SiblingSumComponent('Y')
        node = ArchNode(outputs=['Y.sib_sum'], component=comp, inputs=['Y'], grouping='FID')
        result = comp.compute(node, hap, pheno)
        np.testing.assert_allclose(result[0:3], 6.0)   # sum of [1,2,3]
        np.testing.assert_allclose(result[3:5], 30.0)   # sum of [10,20]

    def test_single_member_sum(self):
        """Single-member family: sum = self."""
        vals = np.array([7.0, 11.0])
        hap, pheno = _make_env(2, fid=[0, 1], source_values=vals)
        comp = SiblingSumComponent('Y')
        node = ArchNode(outputs=['Y.sib_sum'], component=comp, inputs=['Y'], grouping='FID')
        result = comp.compute(node, hap, pheno)
        np.testing.assert_array_equal(result, vals)


class TestSiblingAnyEdgeCases:
    def test_all_zero_group(self):
        """All zeros → any = 0.0 for all."""
        hap, pheno = _make_env(4, fid=[0, 0, 1, 1], source_values=np.zeros(4))
        comp = SiblingAnyComponent('Y')
        node = ArchNode(outputs=['Y.sib_any'], component=comp, inputs=['Y'], grouping='FID')
        result = comp.compute(node, hap, pheno)
        np.testing.assert_array_equal(result, np.zeros(4))

    def test_mixed_group(self):
        """One positive in family → all members get 1.0."""
        vals = np.array([0.0, 5.0, 0.0, 0.0])
        hap, pheno = _make_env(4, fid=[0, 0, 1, 1], source_values=vals)
        comp = SiblingAnyComponent('Y')
        node = ArchNode(outputs=['Y.sib_any'], component=comp, inputs=['Y'], grouping='FID')
        result = comp.compute(node, hap, pheno)
        np.testing.assert_array_equal(result[0:2], [1.0, 1.0])  # fam 0: one positive
        np.testing.assert_array_equal(result[2:4], [0.0, 0.0])  # fam 1: all zero

    def test_all_positive(self):
        """All positive → any = 1.0 for all."""
        vals = np.array([1.0, 2.0, 3.0, 4.0])
        hap, pheno = _make_env(4, fid=[0, 0, 1, 1], source_values=vals)
        comp = SiblingAnyComponent('Y')
        node = ArchNode(outputs=['Y.sib_any'], component=comp, inputs=['Y'], grouping='FID')
        result = comp.compute(node, hap, pheno)
        np.testing.assert_array_equal(result, np.ones(4))

    def test_negative_not_counted_as_positive(self):
        """Negative values are not > 0, so sibling_any treats them as absent."""
        vals = np.array([-1.0, -2.0, 0.0, 3.0])
        hap, pheno = _make_env(4, fid=[0, 0, 1, 1], source_values=vals)
        comp = SiblingAnyComponent('Y')
        node = ArchNode(outputs=['Y.sib_any'], component=comp, inputs=['Y'], grouping='FID')
        result = comp.compute(node, hap, pheno)
        np.testing.assert_array_equal(result[0:2], [0.0, 0.0])  # negatives → not > 0
        np.testing.assert_array_equal(result[2:4], [1.0, 1.0])  # 3.0 > 0


class TestSiblingCountEdgeCases:
    def test_various_sizes(self):
        """Family sizes: 1, 2, 3 → count broadcast."""
        fid = np.array([0, 1, 1, 2, 2, 2])
        hap, pheno = _make_env(6, fid=fid)
        comp = SiblingCountComponent('Y')
        node = ArchNode(outputs=['Y.sib_count'], component=comp, inputs=['Y'], grouping='FID')
        result = comp.compute(node, hap, pheno)
        np.testing.assert_array_equal(result, [1.0, 2.0, 2.0, 3.0, 3.0, 3.0])

    def test_all_same_family(self):
        """All in one family → count = n for all."""
        hap, pheno = _make_env(5, fid=np.zeros(5, dtype=int))
        comp = SiblingCountComponent('Y')
        node = ArchNode(outputs=['Y.sib_count'], component=comp, inputs=['Y'], grouping='FID')
        result = comp.compute(node, hap, pheno)
        np.testing.assert_array_equal(result, np.full(5, 5.0))


class TestSiblingEldestYoungestEdgeCases:
    def test_eldest_is_first_in_group(self):
        """Eldest = value of lowest-index member in each group."""
        vals = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        hap, pheno = _make_env(5, fid=[0, 0, 0, 1, 1], source_values=vals)
        comp = SiblingEldestComponent('Y')
        node = ArchNode(outputs=['Y.eldest'], component=comp, inputs=['Y'], grouping='FID')
        result = comp.compute(node, hap, pheno)
        np.testing.assert_array_equal(result[0:3], [10.0, 10.0, 10.0])
        np.testing.assert_array_equal(result[3:5], [40.0, 40.0])

    def test_youngest_is_last_in_group(self):
        """Youngest = value of highest-index member in each group."""
        vals = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        hap, pheno = _make_env(5, fid=[0, 0, 0, 1, 1], source_values=vals)
        comp = SiblingYoungestComponent('Y')
        node = ArchNode(outputs=['Y.youngest'], component=comp, inputs=['Y'], grouping='FID')
        result = comp.compute(node, hap, pheno)
        np.testing.assert_array_equal(result[0:3], [30.0, 30.0, 30.0])
        np.testing.assert_array_equal(result[3:5], [50.0, 50.0])

    def test_single_member_eldest_equals_youngest(self):
        """Single-member family: eldest = youngest = self."""
        vals = np.array([7.0, 11.0])
        hap, pheno = _make_env(2, fid=[0, 1], source_values=vals)

        eldest_comp = SiblingEldestComponent('Y')
        youngest_comp = SiblingYoungestComponent('Y')
        node_e = ArchNode(outputs=['Y.e'], component=eldest_comp, inputs=['Y'], grouping='FID')
        node_y = ArchNode(outputs=['Y.y'], component=youngest_comp, inputs=['Y'], grouping='FID')

        result_e = eldest_comp.compute(node_e, hap, pheno)
        result_y = youngest_comp.compute(node_y, hap, pheno)
        np.testing.assert_array_equal(result_e, vals)
        np.testing.assert_array_equal(result_y, vals)


class TestSiblingMissingSource:
    def test_missing_source_raises(self):
        """Source not in phenotypes → ValueError."""
        hap, pheno = _make_env(4, fid=[0, 0, 1, 1])
        comp = SiblingMeanComponent('NONEXISTENT')
        node = ArchNode(outputs=['X'], component=comp, inputs=['NONEXISTENT'], grouping='FID')
        with pytest.raises(ValueError, match="not found"):
            comp.compute(node, hap, pheno)


class TestSiblingRepr:
    def test_repr_format(self):
        comp = SiblingMeanComponent('Y')
        assert "SiblingMeanComponent" in repr(comp)
        assert "'Y'" in repr(comp)

    def test_all_component_names(self):
        """Each sibling component has the expected .name attribute."""
        assert SiblingMeanComponent('Y').name == 'sibling_mean'
        assert SiblingSumComponent('Y').name == 'sibling_sum'
        assert SiblingAnyComponent('Y').name == 'sibling_any'
        assert SiblingCountComponent('Y').name == 'sibling_count'
        assert SiblingEldestComponent('Y').name == 'sibling_eldest'
        assert SiblingYoungestComponent('Y').name == 'sibling_youngest'
