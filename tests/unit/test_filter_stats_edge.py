"""
Unit tests for filter and stats edge cases.

Tests:
1. TrioFilter: gen 0 returns None, pruned parent gen, key mismatch
2. SibPairFilter: all singletons, mixed family sizes, large families
3. SampleStatistics: empty phenotype, single key (k=1), missing generation
4. SibPairFilter pair counting correctness
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, PhenotypeArray, PedigreeArray
from xftsim.filters import TrioFilter, SibPairFilter, TrioView, SibPairView
from xftsim.stats import SampleStatistics, GenerationResult


def _make_phenotype(n=20, keys=None, fid=None, seed=42):
    """Create a simple phenotype array."""
    rng = np.random.RandomState(seed)
    if keys is None:
        keys = ['Y']
    if fid is None:
        fid = np.arange(n) // 2  # pairs
    sm = SampleMeta(iid=np.arange(n), fid=fid)
    values = {k: rng.randn(n) for k in keys}
    return PhenotypeArray(samples=sm, values=values)


def _make_pedigree(n=20, parent_n=20):
    """Create a simple pedigree."""
    sm = SampleMeta(iid=np.arange(n))
    return PedigreeArray(
        offspring_samples=sm,
        maternal_idx=np.zeros(n, dtype=np.int64),
        paternal_idx=np.ones(n, dtype=np.int64),
        parent_n=parent_n,
    )


# ── TrioFilter ───────────────────────────────────────────────────────────────

class TestTrioFilterEdgeCases:
    def test_gen_zero_returns_none(self):
        """TrioFilter at gen 0 always returns None."""
        f = TrioFilter()
        result = f.apply(0, {0: _make_phenotype()}, {})
        assert result is None

    def test_missing_pedigree_returns_none(self):
        """TrioFilter with no pedigree at generation returns None."""
        f = TrioFilter()
        pheno = _make_phenotype()
        result = f.apply(1, {1: pheno}, {})
        assert result is None

    def test_pruned_parent_gen_returns_none(self):
        """TrioFilter when parent gen (gen-1) is pruned returns None."""
        f = TrioFilter()
        pheno = _make_phenotype()
        ped = _make_pedigree()
        # gen 1 exists, pedigree exists, but gen 0 phenotype is pruned
        result = f.apply(1, {1: pheno}, {1: ped})
        assert result is None

    def test_valid_trio(self):
        """TrioFilter with valid parent + offspring data produces TrioView."""
        f = TrioFilter()
        parent_pheno = _make_phenotype(n=20, keys=['Y'], seed=0)
        child_pheno = _make_phenotype(n=20, keys=['Y'], seed=1)
        ped = _make_pedigree(n=20, parent_n=20)
        result = f.apply(1, {0: parent_pheno, 1: child_pheno}, {1: ped})
        assert isinstance(result, TrioView)
        assert result.n_trios == 20
        assert 'Y' in result.offspring_phenotypes
        assert 'Y' in result.mother_phenotypes
        assert 'Y' in result.father_phenotypes

    def test_key_mismatch_partial(self):
        """TrioFilter when offspring has keys not in parent → only shared keys indexed."""
        f = TrioFilter()
        parent_pheno = _make_phenotype(n=20, keys=['Y'], seed=0)
        child_pheno = _make_phenotype(n=20, keys=['Y', 'X'], seed=1)
        ped = _make_pedigree(n=20, parent_n=20)
        result = f.apply(1, {0: parent_pheno, 1: child_pheno}, {1: ped})
        assert isinstance(result, TrioView)
        # Y should be in both
        assert 'Y' in result.mother_phenotypes
        # X was not in parent, so should not be in mother/father
        assert 'X' not in result.mother_phenotypes
        # But X should be in offspring
        assert 'X' in result.offspring_phenotypes


# ── SibPairFilter ────────────────────────────────────────────────────────────

class TestSibPairFilterEdgeCases:
    def test_all_singletons_returns_zero_pairs(self):
        """When all families are singletons, should return 0 pairs."""
        f = SibPairFilter()
        fid = np.arange(20)  # all unique FID
        pheno = _make_phenotype(n=20, fid=fid)
        result = f.apply(0, {0: pheno}, {})
        assert isinstance(result, SibPairView)
        assert result.n_pairs == 0

    def test_one_pair_per_family(self):
        """Two-member families should produce exactly 1 pair each."""
        f = SibPairFilter()
        fid = np.array([0, 0, 1, 1, 2, 2])
        pheno = _make_phenotype(n=6, fid=fid)
        result = f.apply(0, {0: pheno}, {})
        assert result.n_pairs == 3  # 3 families × 1 pair each

    def test_three_member_family(self):
        """Three-member family produces 3 pairs (3 choose 2)."""
        f = SibPairFilter()
        fid = np.array([0, 0, 0, 1, 1])
        pheno = _make_phenotype(n=5, fid=fid)
        result = f.apply(0, {0: pheno}, {})
        # Family 0: 3 choose 2 = 3 pairs
        # Family 1: 2 choose 2 = 1 pair
        assert result.n_pairs == 4

    def test_four_member_family(self):
        """Four-member family produces 6 pairs (4 choose 2)."""
        f = SibPairFilter()
        fid = np.array([0, 0, 0, 0])
        pheno = _make_phenotype(n=4, fid=fid)
        result = f.apply(0, {0: pheno}, {})
        assert result.n_pairs == 6  # 4 choose 2

    def test_mixed_family_sizes(self):
        """Mix of family sizes produces correct total pairs."""
        f = SibPairFilter()
        # 1 singleton + 1 pair + 1 trio = 0 + 1 + 3 = 4 pairs
        fid = np.array([0, 1, 1, 2, 2, 2])
        pheno = _make_phenotype(n=6, fid=fid)
        result = f.apply(0, {0: pheno}, {})
        assert result.n_pairs == 4

    def test_sib_pair_indices_valid(self):
        """Sib pair indices should be within range."""
        f = SibPairFilter()
        fid = np.array([0, 0, 1, 1])
        pheno = _make_phenotype(n=4, fid=fid)
        result = f.apply(0, {0: pheno}, {})
        assert np.all(result.sib1_idx >= 0)
        assert np.all(result.sib1_idx < 4)
        assert np.all(result.sib2_idx >= 0)
        assert np.all(result.sib2_idx < 4)

    def test_sib_pair_same_family(self):
        """Sib pairs should share same FID."""
        f = SibPairFilter()
        fid = np.array([0, 0, 1, 1])
        pheno = _make_phenotype(n=4, fid=fid)
        result = f.apply(0, {0: pheno}, {})
        for i in range(result.n_pairs):
            assert fid[result.sib1_idx[i]] == fid[result.sib2_idx[i]]

    def test_missing_generation_returns_none(self):
        """SibPairFilter for missing generation returns None."""
        f = SibPairFilter()
        result = f.apply(5, {0: _make_phenotype()}, {})
        assert result is None

    def test_multiple_keys_indexed(self):
        """SibPairFilter should index all phenotype keys."""
        f = SibPairFilter()
        fid = np.array([0, 0, 1, 1])
        pheno = _make_phenotype(n=4, keys=['Y', 'X'], fid=fid)
        result = f.apply(0, {0: pheno}, {})
        assert 'Y' in result.sib1_phenotypes
        assert 'X' in result.sib1_phenotypes
        assert 'Y' in result.sib2_phenotypes
        assert 'X' in result.sib2_phenotypes


# ── SampleStatistics ─────────────────────────────────────────────────────────

class TestSampleStatisticsEdgeCases:
    def test_missing_generation_returns_none(self):
        """SampleStatistics for missing generation returns None."""
        stat = SampleStatistics()
        result = stat.estimate({}, {}, generation=5)
        assert result is None

    def test_single_key_produces_1x1_cov(self):
        """With k=1, covariance should be reshaped to (1,1)."""
        stat = SampleStatistics()
        pheno = _make_phenotype(n=20, keys=['Y'])
        result = stat.estimate({0: pheno}, {}, generation=0)
        assert result['cov'].shape == (1, 1)
        assert result['cov'][0, 0] > 0
        assert len(result['var']) == 1

    def test_two_keys(self):
        """With k=2, covariance should be (2,2)."""
        stat = SampleStatistics()
        pheno = _make_phenotype(n=20, keys=['Y', 'X'])
        result = stat.estimate({0: pheno}, {}, generation=0)
        assert result['cov'].shape == (2, 2)
        assert len(result['var']) == 2
        assert set(result['keys']) == {'Y', 'X'}

    def test_empty_phenotype(self):
        """Phenotype with no keys should return empty arrays."""
        stat = SampleStatistics()
        sm = SampleMeta(iid=np.arange(5))
        pheno = PhenotypeArray(samples=sm)
        result = stat.estimate({0: pheno}, {}, generation=0)
        assert len(result['keys']) == 0
        assert len(result['var']) == 0

    def test_diagonal_equals_var(self):
        """Diagonal of cov should equal var."""
        stat = SampleStatistics()
        pheno = _make_phenotype(n=50, keys=['Y', 'X', 'Z'], seed=42)
        result = stat.estimate({0: pheno}, {}, generation=0)
        np.testing.assert_allclose(result['var'], np.diag(result['cov']))

    def test_cov_symmetric(self):
        """Covariance matrix should be symmetric."""
        stat = SampleStatistics()
        pheno = _make_phenotype(n=50, keys=['Y', 'X', 'Z'], seed=42)
        result = stat.estimate({0: pheno}, {}, generation=0)
        np.testing.assert_allclose(result['cov'], result['cov'].T, atol=1e-12)


# ── GenerationResult ─────────────────────────────────────────────────────────

class TestGenerationResult:
    def test_construction(self):
        """GenerationResult should store generation and statistics."""
        r = GenerationResult(generation=3, statistics={'cov': np.eye(2)})
        assert r.generation == 3
        assert 'cov' in r.statistics

    def test_repr(self):
        """GenerationResult should have a repr."""
        r = GenerationResult(generation=0, statistics={})
        assert '0' in repr(r)
