"""Unit tests for GraphHaplotypeOperator."""
import numpy as np
import pytest

pygrgl = pytest.importorskip("pygrgl")

from xftsim.struct import (
    GraphHaplotypeOperator, DenseHaplotypeArray,
    HaplotypeOperator, SampleMeta, VariantMeta,
    _extract_variant_meta_from_grg,
)
from tests.testdata import TestGRG


# ── helpers ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tiny():
    return TestGRG.tiny_grg()


@pytest.fixture(scope="module")
def tiny_no_bim():
    return TestGRG.tiny_grg_no_bim()


@pytest.fixture(scope="module")
def tiny_dense(tiny):
    """Dense materialization of the tiny GRG (reference for comparison)."""
    return tiny.to_dense()


# ── TestConstruction ─────────────────────────────────────────────────────

class TestConstruction:
    def test_isinstance(self, tiny):
        assert isinstance(tiny, GraphHaplotypeOperator)
        assert isinstance(tiny, HaplotypeOperator)

    def test_n_m(self, tiny):
        assert tiny.n == 20
        assert tiny.m == 100

    def test_samples_from_grg(self, tiny):
        assert tiny.samples.n == 20
        assert len(tiny.samples.iid) == 20

    def test_variants_with_bim(self, tiny):
        assert tiny.variants.m == 100
        assert tiny.variants.chrom is not None
        assert tiny.variants.pos_bp is not None

    def test_variants_without_bim(self, tiny_no_bim):
        assert tiny_no_bim.variants.m == 100
        # VIDs should be "pos:ref:alt" format
        vid0 = str(tiny_no_bim.variants.vid[0])
        assert ":" in vid0
        # No chrom from GRG
        assert tiny_no_bim.variants.chrom is None

    def test_custom_samples(self):
        grg = pygrgl.load_immutable_grg(TestGRG.TINY_GRG_PATH)
        n = grg.num_individuals
        custom = SampleMeta(iid=np.arange(n), sex=np.zeros(n, dtype=int))
        op = GraphHaplotypeOperator(grg, samples=custom)
        assert op.samples is custom or np.array_equal(op.samples.iid, custom.iid)

    def test_sample_mismatch_error(self):
        grg = pygrgl.load_immutable_grg(TestGRG.TINY_GRG_PATH)
        bad = SampleMeta(iid=np.arange(5))
        with pytest.raises(ValueError, match="samples.n"):
            GraphHaplotypeOperator(grg, samples=bad)

    def test_variant_mismatch_error(self):
        grg = pygrgl.load_immutable_grg(TestGRG.TINY_GRG_PATH)
        bad = VariantMeta(vid=np.arange(5))
        with pytest.raises(ValueError, match="variants.m"):
            GraphHaplotypeOperator(grg, variants=bad)


# ── TestMatvec ───────────────────────────────────────────────────────────

class TestMatvec:
    def test_matvec_1d(self, tiny, tiny_dense):
        v = np.ones(tiny.m, dtype=np.float64)
        result = tiny.matvec(v)
        expected = tiny_dense.matvec(v)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_matvec_2d(self, tiny, tiny_dense):
        rng = np.random.RandomState(99)
        v = rng.randn(tiny.m, 3)
        result = tiny.matvec(v)
        expected = tiny_dense.matvec(v)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_rmatvec_1d(self, tiny, tiny_dense):
        v = np.ones(tiny.n, dtype=np.float64)
        result = tiny.rmatvec(v)
        expected = tiny_dense.rmatvec(v)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_rmatvec_2d(self, tiny, tiny_dense):
        rng = np.random.RandomState(99)
        v = rng.randn(tiny.n, 2)
        result = tiny.rmatvec(v)
        expected = tiny_dense.rmatvec(v)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_matvec_maternal(self, tiny, tiny_dense):
        v = np.ones(tiny.m, dtype=np.float64)
        result = tiny.matvec_maternal(v)
        expected = tiny_dense.matvec_maternal(v)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_matvec_paternal(self, tiny, tiny_dense):
        v = np.ones(tiny.m, dtype=np.float64)
        result = tiny.matvec_paternal(v)
        expected = tiny_dense.matvec_paternal(v)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_maternal_plus_paternal_equals_diploid(self, tiny):
        v = np.ones(tiny.m, dtype=np.float64)
        mat = tiny.matvec_maternal(v)
        pat = tiny.matvec_paternal(v)
        dip = tiny.matvec(v)
        np.testing.assert_allclose(mat + pat, dip, atol=1e-10)

    def test_standardized_matvec(self, tiny, tiny_dense):
        af = tiny.recompute_af()
        v = np.ones(tiny.m, dtype=np.float64)
        result = tiny.standardized_matvec(v, af=af)
        expected = tiny_dense.standardized_matvec(v, af=af)
        np.testing.assert_allclose(result, expected, atol=1e-10)


# ── TestAF ───────────────────────────────────────────────────────────────

class TestAF:
    def test_af_matches_dense(self, tiny, tiny_dense):
        af_grg = tiny.recompute_af()
        af_dense = tiny_dense.recompute_af()
        np.testing.assert_allclose(af_grg, af_dense, atol=1e-10)

    def test_af_shape(self, tiny):
        af = tiny.recompute_af()
        assert af.shape == (tiny.m,)

    def test_af_range(self, tiny):
        af = tiny.recompute_af()
        assert np.all(af >= 0.0)
        assert np.all(af <= 1.0)

    def test_af_caching(self, tiny):
        af1 = tiny.recompute_af()
        af2 = tiny.recompute_af()
        assert af1 is af2


# ── TestMaterialization ──────────────────────────────────────────────────

class TestMaterialization:
    def test_to_dense_type(self, tiny):
        dense = tiny.to_dense()
        assert isinstance(dense, DenseHaplotypeArray)

    def test_to_dense_shape(self, tiny):
        dense = tiny.to_dense()
        assert dense.genotypes.shape == (20, 100, 2)

    def test_to_dense_metadata(self, tiny):
        dense = tiny.to_dense()
        assert dense.n == tiny.n
        assert dense.m == tiny.m
        assert dense.generation == tiny.generation

    def test_to_dense_binary_values(self, tiny):
        dense = tiny.to_dense()
        unique = np.unique(dense.genotypes)
        assert set(unique).issubset({0, 1})


# ── TestSubsetting ───────────────────────────────────────────────────────

class TestSubsetting:
    def test_getitem_samples(self, tiny):
        sub = tiny[:10]
        assert isinstance(sub, DenseHaplotypeArray)
        assert sub.n == 10
        assert sub.m == tiny.m

    def test_getitem_samples_and_variants(self, tiny):
        sub = tiny[:10, :50]
        assert isinstance(sub, DenseHaplotypeArray)
        assert sub.n == 10
        assert sub.m == 50


# ── TestMeiosis ──────────────────────────────────────────────────────────

class TestMeiosis:
    def _make_balanced_grg(self, tiny):
        """Create GRG with balanced sex for mating."""
        sex = np.tile([0, 1], (tiny.n + 1) // 2)[:tiny.n]
        samples = SampleMeta(iid=tiny.samples.iid, sex=sex)
        return GraphHaplotypeOperator(tiny._grg, samples=samples)

    def test_meiosis_returns_dense(self, tiny):
        from xftsim.mate import RandomMating
        from xftsim.reproduce import RecombinationMap

        tiny_copy = self._make_balanced_grg(tiny)
        rmap = RecombinationMap.constant_map(m=tiny.m, p=0.5)
        mate = RandomMating(offspring_per_pair=2)
        assignment = mate.mate(tiny_copy.samples, rng=np.random.RandomState(42))
        offspring = tiny_copy.meiosis(assignment, rmap)
        assert isinstance(offspring, DenseHaplotypeArray)
        assert offspring.m == tiny.m

    def test_meiosis_offspring_count(self, tiny):
        """Meiosis should produce expected number of offspring."""
        from xftsim.mate import RandomMating
        from xftsim.reproduce import RecombinationMap

        tiny_copy = self._make_balanced_grg(tiny)
        rmap = RecombinationMap.constant_map(m=tiny.m, p=0.5)
        mate = RandomMating(offspring_per_pair=2)
        assignment = mate.mate(tiny_copy.samples, rng=np.random.RandomState(42))
        offspring = tiny_copy.meiosis(assignment, rmap)
        assert offspring.n == len(assignment.maternal_idx)
        assert offspring.genotypes.shape == (offspring.n, tiny.m, 2)

    def test_offspring_alleles_from_parents(self, tiny):
        """Each offspring allele should be present in the corresponding parent."""
        from xftsim.mate import RandomMating
        from xftsim.reproduce import RecombinationMap

        tiny_copy = self._make_balanced_grg(tiny)
        dense = tiny_copy.to_dense()

        rmap = RecombinationMap.constant_map(m=tiny.m, p=0.5)
        mate = RandomMating(offspring_per_pair=2)
        assignment = mate.mate(tiny_copy.samples, rng=np.random.RandomState(42))
        offspring = tiny_copy.meiosis(assignment, rmap)

        # For each offspring, maternal haplotype should come from mother's two haplotypes
        for i in range(offspring.n):
            mom_idx = assignment.maternal_idx[i]
            dad_idx = assignment.paternal_idx[i]
            for j in range(offspring.m):
                # Offspring maternal allele ([:,j,0]) came from mother
                off_mat = offspring.genotypes[i, j, 0]
                mom_hap0 = dense.genotypes[mom_idx, j, 0]
                mom_hap1 = dense.genotypes[mom_idx, j, 1]
                assert off_mat in (mom_hap0, mom_hap1), (
                    f"Offspring {i} variant {j}: maternal allele {off_mat} "
                    f"not in mother {mom_idx}'s haplotypes ({mom_hap0}, {mom_hap1})"
                )

                # Offspring paternal allele ([:,j,1]) came from father
                off_pat = offspring.genotypes[i, j, 1]
                dad_hap0 = dense.genotypes[dad_idx, j, 0]
                dad_hap1 = dense.genotypes[dad_idx, j, 1]
                assert off_pat in (dad_hap0, dad_hap1), (
                    f"Offspring {i} variant {j}: paternal allele {off_pat} "
                    f"not in father {dad_idx}'s haplotypes ({dad_hap0}, {dad_hap1})"
                )

    def test_meiosis_variant_meta_preserved(self, tiny):
        """Offspring should inherit variant metadata from parents."""
        from xftsim.mate import RandomMating
        from xftsim.reproduce import RecombinationMap

        tiny_copy = self._make_balanced_grg(tiny)
        rmap = RecombinationMap.constant_map(m=tiny.m, p=0.5)
        mate = RandomMating(offspring_per_pair=2)
        assignment = mate.mate(tiny_copy.samples, rng=np.random.RandomState(42))
        offspring = tiny_copy.meiosis(assignment, rmap)
        np.testing.assert_array_equal(offspring.variants.vid, tiny_copy.variants.vid)

    def test_repr(self, tiny):
        r = repr(tiny)
        assert "GraphHaplotypeOperator" in r
        assert "n=20" in r
        assert "m=100" in r
