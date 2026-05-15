"""
Extended unit tests for _resolve_grouping and grouped noise.

Tests:
1. None grouping returns None
2. FID grouping returns fid array
3. sex grouping returns sex array
4. mother grouping with pedigree
5. father grouping with pedigree
6. mother/father at gen 0 warns and returns None
7. Extra field grouping
8. Unknown grouping raises ValueError
9. NoiseComponent grouped by FID shares values within families
10. CNoiseComponent grouped produces (n, k) with shared values
"""
import numpy as np
import pytest
import warnings

from xftsim.arch import (
    _resolve_grouping, NoiseComponent, CNoiseComponent, ArchNode,
)
from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray, PedigreeArray

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


def _make_hap(n=10, m=5, seed=42, extra=None):
    sm = SampleMeta(iid=np.arange(n), extra=extra or {})
    vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
    rng = np.random.RandomState(seed)
    geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
    return DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)


class TestResolveGroupingBasic:
    def test_none_returns_none(self):
        hap = _make_hap()
        result = _resolve_grouping(None, hap)
        assert result is None

    def test_fid_returns_fid(self):
        hap = _make_hap()
        result = _resolve_grouping('FID', hap)
        np.testing.assert_array_equal(result, hap.samples.fid)

    def test_sex_returns_sex(self):
        hap = _make_hap()
        result = _resolve_grouping('sex', hap)
        np.testing.assert_array_equal(result, hap.samples.sex)


class TestResolveGroupingPedigree:
    def test_mother_with_pedigree(self):
        hap = _make_hap(n=4)
        offspring_sm = SampleMeta(iid=np.arange(4), generation=1)
        ped = PedigreeArray(
            offspring_samples=offspring_sm,
            maternal_idx=np.array([0, 0, 2, 2]),
            paternal_idx=np.array([1, 1, 3, 3]),
            parent_n=10,
        )
        result = _resolve_grouping(
            'mother', hap,
            generation=1, pedigree_history={1: ped},
        )
        np.testing.assert_array_equal(result, [0, 0, 2, 2])

    def test_father_with_pedigree(self):
        hap = _make_hap(n=4)
        offspring_sm = SampleMeta(iid=np.arange(4), generation=1)
        ped = PedigreeArray(
            offspring_samples=offspring_sm,
            maternal_idx=np.array([0, 0, 2, 2]),
            paternal_idx=np.array([1, 1, 3, 3]),
            parent_n=10,
        )
        result = _resolve_grouping(
            'father', hap,
            generation=1, pedigree_history={1: ped},
        )
        np.testing.assert_array_equal(result, [1, 1, 3, 3])

    def test_mother_gen0_warns(self):
        hap = _make_hap()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _resolve_grouping('mother', hap, generation=0)
            assert len(w) == 1
            assert "no pedigree" in str(w[0].message).lower()
        assert result is None

    def test_father_no_pedigree_warns(self):
        hap = _make_hap()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _resolve_grouping(
                'father', hap,
                generation=2, pedigree_history={},
            )
            assert len(w) == 1
        assert result is None


class TestResolveGroupingExtra:
    def test_extra_field(self):
        extra = {'cohort': np.array([0, 0, 1, 1, 2])}
        hap = _make_hap(n=5, extra=extra)
        result = _resolve_grouping('cohort', hap)
        np.testing.assert_array_equal(result, [0, 0, 1, 1, 2])

    def test_unknown_grouping_raises(self):
        hap = _make_hap()
        with pytest.raises(ValueError, match="Unknown grouping"):
            _resolve_grouping('NONEXISTENT', hap)


class TestGroupedNoise:
    def test_noise_grouped_by_fid(self):
        """NoiseComponent with FID grouping: same family → same noise."""
        sm = SampleMeta(
            iid=np.arange(6),
            fid=np.array([0, 0, 1, 1, 2, 2]),
        )
        vm = VariantMeta(vid=np.array(['v0']))
        geno = np.ones((6, 1, 2), dtype=np.int8)
        hap = DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)

        comp = NoiseComponent(variance=1.0)
        node = ArchNode(outputs=['Y.E'], component=comp, inputs=[], grouping='FID')
        rng = np.random.RandomState(42)
        from xftsim.struct import PhenotypeArray
        pheno = PhenotypeArray(samples=sm)
        result = comp.compute(node, hap, pheno, rng=rng)

        # Same family should have same value
        assert result[0] == result[1]  # family 0
        assert result[2] == result[3]  # family 1
        assert result[4] == result[5]  # family 2
        # Different families should (almost certainly) differ
        assert result[0] != result[2]

    def test_cnoise_grouped_by_fid(self):
        """CNoiseComponent grouped: same family → same multivariate noise."""
        sm = SampleMeta(
            iid=np.arange(4),
            fid=np.array([0, 0, 1, 1]),
        )
        vm = VariantMeta(vid=np.array(['v0']))
        geno = np.ones((4, 1, 2), dtype=np.int8)
        hap = DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)

        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        comp = CNoiseComponent(cov=cov)
        node = ArchNode(outputs=['A', 'B'], component=comp, inputs=[], grouping='FID')
        rng = np.random.RandomState(42)
        from xftsim.struct import PhenotypeArray
        pheno = PhenotypeArray(samples=sm)
        result = comp.compute(node, hap, pheno, rng=rng)

        # Result should be (n, k)
        assert result.shape == (4, 2)
        # Same family: identical vectors
        np.testing.assert_array_equal(result[0], result[1])
        np.testing.assert_array_equal(result[2], result[3])
        # Different families: different vectors
        assert not np.array_equal(result[0], result[2])

    def test_noise_ungrouped(self):
        """NoiseComponent without grouping: all independent."""
        hap = _make_hap(n=10)
        comp = NoiseComponent(variance=1.0)
        node = ArchNode(outputs=['Y.E'], component=comp, inputs=[])
        rng = np.random.RandomState(42)
        from xftsim.struct import PhenotypeArray
        pheno = PhenotypeArray(samples=hap.samples)
        result = comp.compute(node, hap, pheno, rng=rng)
        assert result.shape == (10,)
        # Very unlikely all 10 values are the same
        assert len(np.unique(result)) > 1


class TestCNoiseValidation:
    def test_non_square_cov_raises(self):
        with pytest.raises(ValueError, match="square matrix"):
            CNoiseComponent(cov=np.ones((2, 3)))

    def test_1d_cov_raises(self):
        with pytest.raises(ValueError, match="square matrix"):
            CNoiseComponent(cov=np.ones(3))

    def test_k_property(self):
        comp = CNoiseComponent(cov=np.eye(3))
        assert comp.k == 3

    def test_repr(self):
        comp = CNoiseComponent(cov=np.eye(2))
        assert 'CNoiseComponent' in repr(comp)
        assert '(2, 2)' in repr(comp)
