"""
Unit tests for DenseHaplotypeArray and HaplotypeOperator.
"""
import numpy as np
import pytest
from xftsim.struct import DenseHaplotypeArray, HaplotypeOperator, SampleMeta, VariantMeta


@pytest.fixture
def rng():
    return np.random.RandomState(42)


@pytest.fixture
def simple_haplo(rng):
    """10 samples, 5 variants, random 0/1 haplotypes."""
    geno = rng.randint(0, 2, size=(10, 5, 2)).astype(np.int8)
    return DenseHaplotypeArray(genotypes=geno)


class TestConstruction:
    def test_basic(self, simple_haplo):
        h = simple_haplo
        assert h.n == 10
        assert h.m == 5
        assert h.genotypes.shape == (10, 5, 2)

    def test_is_haplotype_operator(self, simple_haplo):
        assert isinstance(simple_haplo, HaplotypeOperator)

    def test_3d_shape_validation(self):
        with pytest.raises(ValueError, match="3-D"):
            DenseHaplotypeArray(genotypes=np.zeros((10, 5), dtype=np.int8))

    def test_last_dim_2(self):
        with pytest.raises(ValueError, match="last dim = 2"):
            DenseHaplotypeArray(genotypes=np.zeros((10, 5, 3), dtype=np.int8))

    def test_with_metadata(self):
        geno = np.zeros((3, 2, 2), dtype=np.int8)
        sm = SampleMeta(iid=np.array([100, 200, 300]))
        vm = VariantMeta(vid=np.array([10, 20]))
        h = DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)
        np.testing.assert_array_equal(h.samples.iid, [100, 200, 300])
        np.testing.assert_array_equal(h.variants.vid, [10, 20])


class TestMatvec:
    def test_matvec_matches_manual(self, simple_haplo):
        h = simple_haplo
        v = np.ones(5)
        result = h.matvec(v)
        expected = h.diploid_genotypes @ v
        np.testing.assert_allclose(result, expected)

    def test_matvec_maternal(self, simple_haplo):
        h = simple_haplo
        v = np.arange(5, dtype=np.float64)
        result = h.matvec_maternal(v)
        expected = h.genotypes[:, :, 0] @ v
        np.testing.assert_allclose(result, expected)

    def test_matvec_paternal(self, simple_haplo):
        h = simple_haplo
        v = np.arange(5, dtype=np.float64)
        result = h.matvec_paternal(v)
        expected = h.genotypes[:, :, 1] @ v
        np.testing.assert_allclose(result, expected)

    def test_matvec_equals_maternal_plus_paternal(self, simple_haplo):
        h = simple_haplo
        v = np.random.RandomState(1).randn(5)
        total = h.matvec(v)
        mat = h.matvec_maternal(v)
        pat = h.matvec_paternal(v)
        np.testing.assert_allclose(total, mat + pat)

    def test_standardized_matvec(self, simple_haplo):
        h = simple_haplo
        v = np.ones(5)
        af = h.af_empirical
        G = h.diploid_genotypes.astype(np.float64) - 2 * af
        expected = G @ v
        result = h.standardized_matvec(v)
        np.testing.assert_allclose(result, expected)

    def test_rmatvec(self, simple_haplo):
        h = simple_haplo
        v = np.ones(10)
        result = h.rmatvec(v)
        expected = h.diploid_genotypes.T @ v
        np.testing.assert_allclose(result, expected)

    def test_matvec_2d(self, simple_haplo):
        """matvec with (m, k) matrix should return (n, k)."""
        h = simple_haplo
        V = np.eye(5, 3)
        result = h.matvec(V)
        expected = h.diploid_genotypes @ V
        np.testing.assert_allclose(result, expected)
        assert result.shape == (10, 3)


class TestSubsetting:
    def test_getitem_samples(self, simple_haplo):
        h = simple_haplo
        sub = h[np.array([0, 2, 4])]
        assert sub.n == 3
        assert sub.m == 5

    def test_getitem_samples_and_variants(self, simple_haplo):
        h = simple_haplo
        sub = h[np.array([0, 1]), np.array([2, 3])]
        assert sub.n == 2
        assert sub.m == 2

    def test_getitem_bool(self, simple_haplo):
        h = simple_haplo
        mask = np.array([True, False, True, False, True,
                        False, True, False, True, False])
        sub = h[mask]
        assert sub.n == 5
        assert sub.m == 5


class TestProperties:
    def test_af_empirical_shape(self, simple_haplo):
        af = simple_haplo.af_empirical
        assert af.shape == (5,)

    def test_af_empirical_range(self, simple_haplo):
        af = simple_haplo.af_empirical
        assert np.all(af >= 0)
        assert np.all(af <= 1)

    def test_recompute_af(self, simple_haplo):
        af = simple_haplo.recompute_af()
        np.testing.assert_allclose(af, simple_haplo.af_empirical)

    def test_to_dense_is_self(self, simple_haplo):
        assert simple_haplo.to_dense() is simple_haplo

    def test_diploid_genotypes(self, simple_haplo):
        h = simple_haplo
        G = h.diploid_genotypes
        assert G.shape == (10, 5)
        assert G.min() >= 0
        assert G.max() <= 2
