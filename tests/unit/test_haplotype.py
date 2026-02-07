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


# ── Additional DenseHaplotypeArray tests ──────────────────────────────────

class TestSingleVariant:
    """Tests with a single variant (m=1)."""

    def test_single_variant_creation(self):
        geno = np.array([[[0, 1], [1, 0]]]).reshape(2, 1, 2).astype(np.int8)
        h = DenseHaplotypeArray(genotypes=geno)
        assert h.n == 2
        assert h.m == 1

    def test_single_variant_matvec(self):
        geno = np.array([[[1, 1]], [[0, 1]], [[1, 0]]]).astype(np.int8)
        h = DenseHaplotypeArray(genotypes=geno)
        result = h.matvec(np.array([1.0]))
        expected = np.array([2.0, 1.0, 1.0])
        np.testing.assert_allclose(result, expected)


class TestSingleIndividual:
    """Tests with a single individual (n=1)."""

    def test_single_individual_creation(self):
        geno = np.zeros((1, 5, 2), dtype=np.int8)
        h = DenseHaplotypeArray(genotypes=geno)
        assert h.n == 1
        assert h.m == 5

    def test_single_individual_matvec(self):
        geno = np.ones((1, 3, 2), dtype=np.int8)
        h = DenseHaplotypeArray(genotypes=geno)
        v = np.array([1.0, 2.0, 3.0])
        result = h.matvec(v)
        # Diploid = all 2s, so result = 2*1 + 2*2 + 2*3 = 12
        np.testing.assert_allclose(result, np.array([12.0]))


class TestOperatorIdentities:
    """Test mathematical identities of the operator."""

    def test_rmatvec_2d(self):
        """rmatvec with 2D input should work."""
        rng = np.random.RandomState(42)
        geno = rng.randint(0, 2, size=(20, 10, 2)).astype(np.int8)
        h = DenseHaplotypeArray(genotypes=geno)
        V = rng.randn(20, 3)
        result = h.rmatvec(V)
        expected = h.diploid_genotypes.T @ V
        np.testing.assert_allclose(result, expected)
        assert result.shape == (10, 3)

    def test_standardized_matvec_zero_mean(self):
        """Standardized matvec with ones should produce near-zero mean."""
        rng = np.random.RandomState(42)
        geno = rng.randint(0, 2, size=(1000, 20, 2)).astype(np.int8)
        h = DenseHaplotypeArray(genotypes=geno)
        v = np.ones(20)
        result = h.standardized_matvec(v)
        # Mean should be close to 0 (centered genotypes)
        assert abs(result.mean()) < 0.5

    def test_matvec_zero_vector(self):
        """matvec with zero vector should produce zero result."""
        rng = np.random.RandomState(42)
        geno = rng.randint(0, 2, size=(10, 5, 2)).astype(np.int8)
        h = DenseHaplotypeArray(genotypes=geno)
        result = h.matvec(np.zeros(5))
        np.testing.assert_array_equal(result, np.zeros(10))

    def test_af_empirical_matches_genotypes(self):
        """af_empirical should equal column means of maternal+paternal / 2."""
        rng = np.random.RandomState(42)
        geno = rng.randint(0, 2, size=(100, 10, 2)).astype(np.int8)
        h = DenseHaplotypeArray(genotypes=geno)
        af = h.af_empirical
        manual_af = (geno[:, :, 0].mean(axis=0) + geno[:, :, 1].mean(axis=0)) / 2
        np.testing.assert_allclose(af, manual_af, atol=1e-10)

    def test_metadata_preserved_through_getitem(self):
        """Custom samples/variants should survive __getitem__."""
        geno = np.zeros((5, 3, 2), dtype=np.int8)
        sm = SampleMeta(iid=np.array([10, 20, 30, 40, 50]))
        vm = VariantMeta(vid=np.array([100, 200, 300]))
        h = DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)
        sub = h[np.array([0, 2])]
        np.testing.assert_array_equal(sub.samples.iid, [10, 30])
        np.testing.assert_array_equal(sub.variants.vid, [100, 200, 300])

    def test_getitem_variant_subset(self):
        """Two-argument getitem should subset both samples and variants."""
        rng = np.random.RandomState(42)
        geno = rng.randint(0, 2, size=(10, 8, 2)).astype(np.int8)
        vm = VariantMeta(vid=np.arange(8), chrom=np.array([1]*4 + [2]*4))
        h = DenseHaplotypeArray(genotypes=geno, variants=vm)
        sub = h[np.array([0, 1]), np.array([2, 5, 7])]
        assert sub.n == 2
        assert sub.m == 3
        np.testing.assert_array_equal(sub.variants.vid, [2, 5, 7])

    def test_generation_propagation(self):
        """DenseHaplotypeArray should use the generation parameter."""
        sm = SampleMeta(iid=np.arange(3))
        geno = np.zeros((3, 2, 2), dtype=np.int8)
        h = DenseHaplotypeArray(genotypes=geno, generation=5, samples=sm)
        assert h.generation == 5
        assert h.samples.generation == 5


class TestDenseHaplotypeArrayMeiosis:
    """Tests for the meiosis() method on DenseHaplotypeArray."""

    def _make_parents(self, n=100, m=50, seed=42):
        rng = np.random.RandomState(seed)
        geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
        sm = SampleMeta(iid=np.arange(n))
        return DenseHaplotypeArray(genotypes=geno, samples=sm)

    def test_returns_dense_haplotype_array(self):
        from xftsim.nmate import RandomMating
        from xftsim.reproduce import RecombinationMap
        hap = self._make_parents()
        rmap = RecombinationMap.constant_map(m=50, p=0.5)
        mate = RandomMating(offspring_per_pair=2)
        assignment = mate.mate(hap.samples, rng=np.random.RandomState(42))
        offspring = hap.meiosis(assignment, rmap)
        assert isinstance(offspring, DenseHaplotypeArray)

    def test_offspring_shape(self):
        from xftsim.nmate import RandomMating
        from xftsim.reproduce import RecombinationMap
        hap = self._make_parents()
        rmap = RecombinationMap.constant_map(m=50, p=0.5)
        mate = RandomMating(offspring_per_pair=2)
        assignment = mate.mate(hap.samples, rng=np.random.RandomState(42))
        offspring = hap.meiosis(assignment, rmap)
        assert offspring.n == assignment.n_offspring
        assert offspring.m == 50
        assert offspring.genotypes.shape[2] == 2

    def test_generation_incremented(self):
        from xftsim.nmate import RandomMating
        from xftsim.reproduce import RecombinationMap
        hap = self._make_parents()
        rmap = RecombinationMap.constant_map(m=50, p=0.5)
        mate = RandomMating(offspring_per_pair=2)
        assignment = mate.mate(hap.samples, rng=np.random.RandomState(42))
        offspring = hap.meiosis(assignment, rmap)
        assert offspring.generation == 1

    def test_variants_inherited(self):
        from xftsim.nmate import RandomMating
        from xftsim.reproduce import RecombinationMap
        vm = VariantMeta(vid=np.arange(50), chrom=np.array([1]*25 + [2]*25))
        hap = self._make_parents()
        hap = DenseHaplotypeArray(genotypes=hap.genotypes, samples=hap.samples, variants=vm)
        rmap = RecombinationMap.constant_map(m=50, p=0.5)
        mate = RandomMating(offspring_per_pair=2)
        assignment = mate.mate(hap.samples, rng=np.random.RandomState(42))
        offspring = hap.meiosis(assignment, rmap)
        np.testing.assert_array_equal(offspring.vid, hap.vid)

    def test_alleles_binary(self):
        from xftsim.nmate import RandomMating
        from xftsim.reproduce import RecombinationMap
        hap = self._make_parents()
        rmap = RecombinationMap.constant_map(m=50, p=0.5)
        mate = RandomMating(offspring_per_pair=2)
        assignment = mate.mate(hap.samples, rng=np.random.RandomState(42))
        offspring = hap.meiosis(assignment, rmap)
        assert set(np.unique(offspring.genotypes)).issubset({0, 1})


class TestToDiploidStandardized:
    """Tests for DenseHaplotypeArray.to_diploid_standardized()."""

    def test_default_centering(self, rng):
        """Centered genotypes should have near-zero column means."""
        geno = rng.randint(0, 2, size=(200, 10, 2)).astype(np.int8)
        h = DenseHaplotypeArray(genotypes=geno)
        G_std = h.to_diploid_standardized()
        # Column means should be approximately 0 (centered)
        col_means = G_std.mean(axis=0)
        np.testing.assert_allclose(col_means, 0.0, atol=1e-10)

    def test_shape_matches_diploid(self, rng):
        """Output shape should match (n, m)."""
        geno = rng.randint(0, 2, size=(50, 8, 2)).astype(np.int8)
        h = DenseHaplotypeArray(genotypes=geno)
        G_std = h.to_diploid_standardized()
        assert G_std.shape == (50, 8)

    def test_custom_af(self, rng):
        """Custom AF should center using G - 2*af."""
        geno = rng.randint(0, 2, size=(50, 5, 2)).astype(np.int8)
        h = DenseHaplotypeArray(genotypes=geno)
        custom_af = np.full(5, 0.5)
        G_std = h.to_diploid_standardized(af=custom_af)
        G_diploid = h.diploid_genotypes.astype(np.float64)
        expected = G_diploid - 2 * custom_af
        np.testing.assert_allclose(G_std, expected)

    def test_scale_true(self, rng):
        """scale=True should divide by sqrt(2*p*(1-p))."""
        geno = rng.randint(0, 2, size=(100, 5, 2)).astype(np.int8)
        h = DenseHaplotypeArray(genotypes=geno)
        af = h.af_empirical
        G_centered = h.diploid_genotypes.astype(np.float64) - 2 * af
        denom = np.sqrt(2 * af * (1 - af))
        denom[denom == 0] = 1.0
        expected = G_centered / denom
        result = h.to_diploid_standardized(scale=True)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_scale_false_vs_true_differ(self, rng):
        """scale=True and scale=False should give different results when AF != 0.5."""
        geno = rng.randint(0, 2, size=(100, 5, 2)).astype(np.int8)
        h = DenseHaplotypeArray(genotypes=geno)
        unscaled = h.to_diploid_standardized(scale=False)
        scaled = h.to_diploid_standardized(scale=True)
        assert not np.allclose(unscaled, scaled)

    def test_monomorphic_variant_safe(self):
        """Monomorphic variants (AF=0 or 1) should not produce NaN/Inf."""
        geno = np.zeros((20, 3, 2), dtype=np.int8)
        geno[:, 1, :] = 1  # variant 1 is all-1 (AF=1.0)
        h = DenseHaplotypeArray(genotypes=geno)
        result = h.to_diploid_standardized(scale=True)
        assert np.all(np.isfinite(result))
