"""Tests for mating edge cases."""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray
from xftsim.mate import RandomMating, MateAssignment


def _make_haplotypes(n, seed=42):
    rng = np.random.RandomState(seed)
    m = 10
    geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
    sex = np.tile([0, 1], (n + 1) // 2)[:n]
    samples = SampleMeta(iid=np.arange(n), sex=sex)
    variants = VariantMeta(vid=np.arange(m), af=np.full(m, 0.5))
    return DenseHaplotypeArray(genotypes=geno, samples=samples, variants=variants)


class TestMateEdgeCases:
    def test_unbalanced_sex_3_offspring(self):
        """Unbalanced sex ratio with 3 offspring per pair."""
        # 6 females, 4 males → 4 pairs → 12 offspring
        n = 10
        m = 5
        rng = np.random.RandomState(0)
        geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
        sex = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
        samples = SampleMeta(iid=np.arange(n), sex=sex)
        variants = VariantMeta(vid=np.arange(m), af=np.full(m, 0.5))
        hap = DenseHaplotypeArray(genotypes=geno, samples=samples, variants=variants)
        mate = RandomMating(offspring_per_pair=3)
        assignment = mate.mate(hap.samples, rng=np.random.RandomState(1))
        assert assignment.n_offspring == 4 * 3  # 4 pairs * 3

    def test_large_population(self):
        """Large population should mate without error."""
        hap = _make_haplotypes(n=10000, seed=99)
        mate = RandomMating()
        assignment = mate.mate(hap.samples, rng=np.random.RandomState(0))
        assert assignment.n_offspring == 5000 * 2  # 5000 pairs * 2

    def test_single_pair(self):
        """Minimum population: 1 female + 1 male."""
        n = 2
        m = 5
        geno = np.zeros((n, m, 2), dtype=np.int8)
        sex = np.array([0, 1])
        samples = SampleMeta(iid=np.arange(n), sex=sex)
        variants = VariantMeta(vid=np.arange(m), af=np.full(m, 0.5))
        hap = DenseHaplotypeArray(genotypes=geno, samples=samples, variants=variants)
        mate = RandomMating()
        assignment = mate.mate(hap.samples, rng=np.random.RandomState(0))
        assert assignment.n_offspring == 2
        assert assignment.maternal_idx[0] == 0
        assert assignment.paternal_idx[0] == 1


class TestRandomMatingBehavior:
    """Additional tests for RandomMating behavior and edge cases."""

    def test_offspring_per_pair_1(self):
        """offspring_per_pair=1 should halve population from balanced parents."""
        hap = _make_haplotypes(n=100)
        mate = RandomMating(offspring_per_pair=1)
        assignment = mate.mate(hap.samples, rng=np.random.RandomState(0))
        assert assignment.n_offspring == 50  # 50 pairs × 1

    def test_offspring_per_pair_4(self):
        """offspring_per_pair=4 should double population from balanced parents."""
        hap = _make_haplotypes(n=100)
        mate = RandomMating(offspring_per_pair=4)
        assignment = mate.mate(hap.samples, rng=np.random.RandomState(0))
        assert assignment.n_offspring == 50 * 4

    def test_invalid_offspring_per_pair_raises(self):
        """offspring_per_pair=0 should raise."""
        with pytest.raises(ValueError, match="offspring_per_pair"):
            RandomMating(offspring_per_pair=0)

    def test_offspring_generation_increments(self):
        """Offspring should be in the next generation."""
        hap = _make_haplotypes(n=20)
        mate = RandomMating()
        assignment = mate.mate(hap.samples, rng=np.random.RandomState(0))
        assert assignment.offspring_samples.generation == hap.samples.generation + 1

    def test_offspring_fids_unique_per_family(self):
        """Each pair should produce offspring with the same FID."""
        hap = _make_haplotypes(n=100)
        mate = RandomMating(offspring_per_pair=3)
        assignment = mate.mate(hap.samples, rng=np.random.RandomState(0))
        # All siblings in a family share FID
        fids = assignment.offspring_samples.fid
        for pair_idx in range(50):
            start = pair_idx * 3
            family_fids = fids[start:start + 3]
            assert len(np.unique(family_fids)) == 1

    def test_offspring_sex_alternates(self):
        """Within each family, sex should alternate (0, 1, 0, ...)."""
        hap = _make_haplotypes(n=100)
        mate = RandomMating(offspring_per_pair=4)
        assignment = mate.mate(hap.samples, rng=np.random.RandomState(0))
        sex = assignment.offspring_samples.sex
        for pair_idx in range(50):
            start = pair_idx * 4
            family_sex = sex[start:start + 4]
            np.testing.assert_array_equal(family_sex, [0, 1, 0, 1])

    def test_maternal_idx_all_female(self):
        """All maternal indices should point to females."""
        hap = _make_haplotypes(n=100)
        mate = RandomMating()
        assignment = mate.mate(hap.samples, rng=np.random.RandomState(0))
        parent_sex = hap.samples.sex
        assert np.all(parent_sex[assignment.maternal_idx] == 0)

    def test_paternal_idx_all_male(self):
        """All paternal indices should point to males."""
        hap = _make_haplotypes(n=100)
        mate = RandomMating()
        assignment = mate.mate(hap.samples, rng=np.random.RandomState(0))
        parent_sex = hap.samples.sex
        assert np.all(parent_sex[assignment.paternal_idx] == 1)

    def test_no_females_raises(self):
        """All-male population should raise ValueError."""
        n = 10
        m = 5
        geno = np.zeros((n, m, 2), dtype=np.int8)
        sex = np.ones(n, dtype=np.int64)  # all male
        samples = SampleMeta(iid=np.arange(n), sex=sex)
        mate = RandomMating()
        with pytest.raises(ValueError, match="at least one female"):
            mate.mate(samples, rng=np.random.RandomState(0))

    def test_no_males_raises(self):
        """All-female population should raise ValueError."""
        n = 10
        m = 5
        geno = np.zeros((n, m, 2), dtype=np.int8)
        sex = np.zeros(n, dtype=np.int64)  # all female
        samples = SampleMeta(iid=np.arange(n), sex=sex)
        mate = RandomMating()
        with pytest.raises(ValueError, match="at least one female and one male"):
            mate.mate(samples, rng=np.random.RandomState(0))

    def test_determinism_same_seed(self):
        """Same seed should produce identical assignments."""
        hap = _make_haplotypes(n=100)
        mate = RandomMating()
        a1 = mate.mate(hap.samples, rng=np.random.RandomState(42))
        a2 = mate.mate(hap.samples, rng=np.random.RandomState(42))
        np.testing.assert_array_equal(a1.maternal_idx, a2.maternal_idx)
        np.testing.assert_array_equal(a1.paternal_idx, a2.paternal_idx)

    def test_different_seeds_differ(self):
        """Different seeds should usually produce different assignments."""
        hap = _make_haplotypes(n=100)
        mate = RandomMating()
        a1 = mate.mate(hap.samples, rng=np.random.RandomState(42))
        a2 = mate.mate(hap.samples, rng=np.random.RandomState(99))
        assert not np.array_equal(a1.maternal_idx, a2.maternal_idx)


class TestNMateAssignmentValidation:
    """Tests for MateAssignment validation."""

    def test_mismatched_maternal_length(self):
        """maternal_idx length mismatch should raise."""
        samples = SampleMeta(iid=np.arange(4), sex=np.array([0,1,0,1]))
        with pytest.raises(ValueError, match="maternal_idx length"):
            MateAssignment(
                offspring_samples=samples,
                maternal_idx=np.array([0, 0]),  # length 2 != 4
                paternal_idx=np.array([1, 1, 1, 1]),
            )

    def test_mismatched_paternal_length(self):
        """paternal_idx length mismatch should raise."""
        samples = SampleMeta(iid=np.arange(4), sex=np.array([0,1,0,1]))
        with pytest.raises(ValueError, match="paternal_idx length"):
            MateAssignment(
                offspring_samples=samples,
                maternal_idx=np.array([0, 0, 0, 0]),
                paternal_idx=np.array([1]),  # length 1 != 4
            )

    def test_negative_maternal_idx(self):
        """Negative maternal indices should raise."""
        samples = SampleMeta(iid=np.arange(2), sex=np.array([0,1]))
        with pytest.raises(ValueError, match="negative"):
            MateAssignment(
                offspring_samples=samples,
                maternal_idx=np.array([-1, 0]),
                paternal_idx=np.array([1, 1]),
            )

    def test_repr(self):
        """MateAssignment repr should not crash."""
        hap = _make_haplotypes(n=20)
        mate = RandomMating()
        assignment = mate.mate(hap.samples, rng=np.random.RandomState(0))
        r = repr(assignment)
        assert "MateAssignment" in r
        assert "n_offspring" in r
