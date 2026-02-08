"""
Unit tests for noise and cnoise components with grouping.

Tests:
1. Grouped noise: same value within group
2. Grouped noise: different values across groups
3. Grouped cnoise: same vector within group
4. Ungrouped noise: all different
"""
import numpy as np
import pytest

from xftsim.narch import NoiseComponent, CNoiseComponent, ArchNode
from xftsim.struct import SampleMeta, NPhenotypeArray, DenseHaplotypeArray, VariantMeta


def _make_grouped_hap(fids, m=3):
    n = len(fids)
    sm = SampleMeta(iid=np.arange(n), fid=np.array(fids))
    vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
    return DenseHaplotypeArray(np.zeros((n, m, 2), dtype=np.int8), samples=sm, variants=vm)


class TestGroupedNoise:
    def test_same_value_within_group(self):
        """Members of same FID get the same noise."""
        hap = _make_grouped_hap([0, 0, 0, 1, 1])
        pheno = NPhenotypeArray(hap.samples)
        comp = NoiseComponent(variance=1.0)
        node = ArchNode(outputs=['E'], component=comp, inputs=[], grouping='FID')
        rng = np.random.RandomState(42)
        result = comp.compute(node, hap, pheno, rng=rng)

        # Within group 0: all same
        assert result[0] == result[1] == result[2]
        # Within group 1: all same
        assert result[3] == result[4]

    def test_different_across_groups(self):
        """Different groups get different noise (with high probability)."""
        hap = _make_grouped_hap([0, 0, 1, 1])
        pheno = NPhenotypeArray(hap.samples)
        comp = NoiseComponent(variance=1.0)
        node = ArchNode(outputs=['E'], component=comp, inputs=[], grouping='FID')
        rng = np.random.RandomState(42)
        result = comp.compute(node, hap, pheno, rng=rng)

        assert result[0] != result[2]  # group 0 != group 1

    def test_grouped_cnoise_same_within_group(self):
        """CNoiseComponent grouped: same vector within family."""
        hap = _make_grouped_hap([0, 0, 1, 1])
        pheno = NPhenotypeArray(hap.samples)
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        comp = CNoiseComponent(cov=cov)
        node = ArchNode(outputs=['E1', 'E2'], component=comp, inputs=[], grouping='FID')
        rng = np.random.RandomState(42)
        result = comp.compute(node, hap, pheno, rng=rng)

        assert result.shape == (4, 2)
        np.testing.assert_array_equal(result[0], result[1])  # same family
        np.testing.assert_array_equal(result[2], result[3])  # same family
        assert not np.allclose(result[0], result[2])  # different families

    def test_ungrouped_noise_all_different(self):
        """Without grouping, each individual gets different noise."""
        hap = _make_grouped_hap([0, 0, 1, 1])
        pheno = NPhenotypeArray(hap.samples)
        comp = NoiseComponent(variance=1.0)
        node = ArchNode(outputs=['E'], component=comp, inputs=[], grouping=None)
        rng = np.random.RandomState(42)
        result = comp.compute(node, hap, pheno, rng=rng)

        # All values should be different
        assert len(np.unique(result)) == 4
