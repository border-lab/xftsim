"""
Unit tests for sex-based grouping in _resolve_grouping.

Tests:
1. Sex grouping returns sex array from SampleMeta
2. Sex grouping used in NoiseComponent produces group-shared values
3. Interaction between sex and FID grouping
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray, NPhenotypeArray
from xftsim.narch import _resolve_grouping, NoiseComponent, ArchNode


def _make_hap_with_sex(n=10, m=3, generation=0):
    sex = np.array([0, 1] * (n // 2))
    sm = SampleMeta(iid=np.arange(n), fid=np.arange(n) // 2, sex=sex, generation=generation)
    vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
    geno = np.zeros((n, m, 2), dtype=np.int8)
    return DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm, generation=generation)


class TestSexGrouping:
    def test_sex_returns_sex_array(self):
        hap = _make_hap_with_sex()
        result = _resolve_grouping('sex', hap, generation=0, pedigree_history={})
        expected = np.array([0, 1] * 5)
        np.testing.assert_array_equal(result, expected)

    def test_sex_grouped_noise_shared(self):
        """All individuals of same sex should get the same noise value."""
        comp = NoiseComponent(variance=1.0)
        node = ArchNode(outputs=['Y.E'], component=comp, inputs=[], grouping='sex')
        hap = _make_hap_with_sex(n=10)
        pheno = NPhenotypeArray(samples=hap.samples)
        rng = np.random.RandomState(42)
        result = comp.compute(node, hap, pheno, rng=rng, generation=0, pedigree_history={})
        sex = hap.samples.sex
        vals_sex0 = result[sex == 0]
        vals_sex1 = result[sex == 1]
        assert np.all(vals_sex0 == vals_sex0[0])
        assert np.all(vals_sex1 == vals_sex1[0])
        assert vals_sex0[0] != vals_sex1[0]

    def test_fid_vs_sex_grouping_different(self):
        hap = _make_hap_with_sex(n=10)
        fid_groups = _resolve_grouping('FID', hap, generation=0, pedigree_history={})
        sex_groups = _resolve_grouping('sex', hap, generation=0, pedigree_history={})
        assert len(np.unique(fid_groups)) == 5
        assert len(np.unique(sex_groups)) == 2
