"""
Unit tests for DenseHaplotypeArray operator methods.

Tests:
1. matvec: G @ v
2. rmatvec: G.T @ v
3. standardized_matvec with default af
4. standardized_matvec with explicit af
5. matvec_maternal: hap[:,:,0] @ v
6. matvec_paternal: hap[:,:,1] @ v
7. maternal + paternal = diploid matvec
8. diploid_genotypes property
9. recompute_af
10. to_dense returns self
11. Accessor properties (iid, fid, sex, n_fam, n_female, n_male, vid)
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray


def _make_hap(n=10, m=5, seed=42):
    rng = np.random.RandomState(seed)
    sm = SampleMeta(iid=np.arange(n), fid=np.arange(n) // 2,
                    sex=np.tile([0, 1], n // 2))
    vm = VariantMeta(vid=np.array([f'v{i}' for i in range(m)]))
    geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
    return DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)


class TestMatvec:
    def test_matvec_1d(self):
        hap = _make_hap()
        v = np.ones(5)
        result = hap.matvec(v)
        expected = hap.diploid_genotypes.astype(float) @ v
        np.testing.assert_allclose(result, expected)

    def test_matvec_random_vector(self):
        hap = _make_hap()
        v = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        result = hap.matvec(v)
        expected = hap.diploid_genotypes.astype(float) @ v
        np.testing.assert_allclose(result, expected)


class TestRmatvec:
    def test_rmatvec_1d(self):
        hap = _make_hap()
        v = np.ones(10)
        result = hap.rmatvec(v)
        expected = hap.diploid_genotypes.astype(float).T @ v
        np.testing.assert_allclose(result, expected)

    def test_rmatvec_shape(self):
        hap = _make_hap(n=10, m=5)
        v = np.ones(10)
        result = hap.rmatvec(v)
        assert result.shape == (5,)


class TestStandardizedMatvec:
    def test_standardized_default_af(self):
        hap = _make_hap(n=100, m=5)
        v = np.ones(5)
        result = hap.standardized_matvec(v)
        # Should be centered: mean approximately 0
        assert abs(np.mean(result)) < 0.5

    def test_standardized_explicit_af(self):
        hap = _make_hap()
        v = np.ones(5)
        af = np.full(5, 0.5)
        result = hap.standardized_matvec(v, af=af)
        # G@v - 2*af@v
        expected = hap.diploid_genotypes.astype(float) @ v - 2 * af @ v
        np.testing.assert_allclose(result, expected)


class TestMaternalPaternal:
    def test_maternal_matvec(self):
        hap = _make_hap()
        v = np.ones(5)
        result = hap.matvec_maternal(v)
        expected = hap.genotypes[:, :, 0].astype(float) @ v
        np.testing.assert_allclose(result, expected)

    def test_paternal_matvec(self):
        hap = _make_hap()
        v = np.ones(5)
        result = hap.matvec_paternal(v)
        expected = hap.genotypes[:, :, 1].astype(float) @ v
        np.testing.assert_allclose(result, expected)

    def test_maternal_plus_paternal_equals_diploid(self):
        hap = _make_hap()
        v = np.array([0.1, -0.2, 0.3, -0.1, 0.5])
        maternal = hap.matvec_maternal(v)
        paternal = hap.matvec_paternal(v)
        diploid = hap.matvec(v)
        np.testing.assert_allclose(maternal + paternal, diploid)


class TestDiploidGenotypes:
    def test_shape(self):
        hap = _make_hap()
        G = hap.diploid_genotypes
        assert G.shape == (10, 5)

    def test_values_0_1_2(self):
        hap = _make_hap()
        G = hap.diploid_genotypes
        assert np.all((G >= 0) & (G <= 2))

    def test_sum_of_haplotypes(self):
        hap = _make_hap()
        G = hap.diploid_genotypes
        expected = hap.genotypes[:, :, 0] + hap.genotypes[:, :, 1]
        np.testing.assert_array_equal(G, expected)


class TestRecomputeAF:
    def test_returns_array(self):
        hap = _make_hap()
        af = hap.recompute_af()
        assert isinstance(af, np.ndarray)
        assert af.shape == (5,)

    def test_range_0_1(self):
        hap = _make_hap(n=100, m=5)
        af = hap.recompute_af()
        assert np.all(af >= 0.0)
        assert np.all(af <= 1.0)


class TestToDense:
    def test_returns_self(self):
        hap = _make_hap()
        assert hap.to_dense() is hap


class TestAccessorProperties:
    def test_iid(self):
        hap = _make_hap()
        np.testing.assert_array_equal(hap.iid, np.arange(10))

    def test_fid(self):
        hap = _make_hap()
        expected = np.arange(10) // 2
        np.testing.assert_array_equal(hap.fid, expected)

    def test_sex(self):
        hap = _make_hap()
        assert len(hap.sex) == 10

    def test_n_fam(self):
        hap = _make_hap()
        assert hap.n_fam == 5

    def test_n_female(self):
        hap = _make_hap()
        assert hap.n_female == 5

    def test_n_male(self):
        hap = _make_hap()
        assert hap.n_male == 5

    def test_vid(self):
        hap = _make_hap()
        assert len(hap.vid) == 5


class TestRepr:
    def test_repr(self):
        hap = _make_hap()
        r = repr(hap)
        assert 'DenseHaplotypeArray' in r
        assert 'n=10' in r
        assert 'm=5' in r
