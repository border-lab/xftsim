"""
Unit tests for SibPairFilter vectorized pairing logic.

Tests the internal vectorized pair generation across different family sizes,
ensuring correct pair counts, no duplicates, valid indices, and proper structure.

Coverage:
1. Family of size 2: produces exactly 1 pair
2. Family of size 3: produces exactly 3 pairs (upper triangle)
3. Family of size 4: produces exactly 6 pairs
4. Multiple families: total pairs = sum of C(n,2) for each family
5. Single-child families: produce 0 pairs
6. Mixed family sizes: correct total pair count
7. All pairs have different indices (no self-pairs)
8. Pair indices are valid (within bounds of offspring array)
9. Filter output has correct column structure (sib1/sib2 indices)
10. Large family (10+ siblings): correct pair count
"""
import sys
import os
import numpy as np
import pytest

# Add tests directory to path for testdata imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation

from xftsim.struct import SampleMeta, PhenotypeArray
from xftsim.filters import SibPairFilter
from xftsim.sim import Simulation


def _make_pheno_with_fids(n, fids, seed=42):
    """Helper to create PhenotypeArray with specified FIDs."""
    rng = np.random.RandomState(seed)
    sm = SampleMeta(iid=np.arange(n), fid=np.asarray(fids, dtype=np.int64))
    pheno = PhenotypeArray(samples=sm)
    pheno['Y'] = rng.normal(0, 1, n)
    return pheno


def _combinations(n, k=2):
    """Compute C(n, k) for pair counting."""
    if k > n or k < 0:
        return 0
    if k == 0 or k == n:
        return 1
    # C(n, 2) = n * (n - 1) / 2
    if k == 2:
        return n * (n - 1) // 2
    # General formula
    from math import factorial
    return factorial(n) // (factorial(k) * factorial(n - k))


class TestSibPairVectorizedPairCounts:
    """Test that pair counts match expected C(n, 2) for various family sizes."""

    def test_family_size_2_one_pair(self):
        """Family of size 2 produces exactly 1 pair."""
        fids = np.array([0, 0], dtype=np.int64)
        pheno = _make_pheno_with_fids(2, fids)

        filt = SibPairFilter()
        view = filt.apply(
            generation=1,
            phenotype_history={1: pheno},
            pedigree_history={},
        )

        assert view.n_pairs == 1
        assert len(view.sib1_idx) == 1
        assert len(view.sib2_idx) == 1

    def test_family_size_3_three_pairs(self):
        """Family of size 3 produces exactly 3 pairs (upper triangle)."""
        fids = np.array([0, 0, 0], dtype=np.int64)
        pheno = _make_pheno_with_fids(3, fids)

        filt = SibPairFilter()
        view = filt.apply(
            generation=1,
            phenotype_history={1: pheno},
            pedigree_history={},
        )

        expected = _combinations(3, 2)  # C(3, 2) = 3
        assert view.n_pairs == expected
        assert len(view.sib1_idx) == expected
        assert len(view.sib2_idx) == expected

    def test_family_size_4_six_pairs(self):
        """Family of size 4 produces exactly 6 pairs."""
        fids = np.array([0, 0, 0, 0], dtype=np.int64)
        pheno = _make_pheno_with_fids(4, fids)

        filt = SibPairFilter()
        view = filt.apply(
            generation=1,
            phenotype_history={1: pheno},
            pedigree_history={},
        )

        expected = _combinations(4, 2)  # C(4, 2) = 6
        assert view.n_pairs == expected
        assert len(view.sib1_idx) == expected
        assert len(view.sib2_idx) == expected

    def test_single_child_families_zero_pairs(self):
        """Single-child families produce 0 pairs."""
        fids = np.array([0, 1, 2, 3, 4], dtype=np.int64)
        pheno = _make_pheno_with_fids(5, fids)

        filt = SibPairFilter()
        view = filt.apply(
            generation=1,
            phenotype_history={1: pheno},
            pedigree_history={},
        )

        assert view.n_pairs == 0
        assert len(view.sib1_idx) == 0
        assert len(view.sib2_idx) == 0

    def test_multiple_families_sum_combinations(self):
        """Multiple families: total pairs = sum of C(n,2) for each family."""
        # 3 families: size 2, size 3, size 4
        # Expected: C(2,2) + C(3,2) + C(4,2) = 1 + 3 + 6 = 10
        fids = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=np.int64)
        pheno = _make_pheno_with_fids(9, fids)

        filt = SibPairFilter()
        view = filt.apply(
            generation=1,
            phenotype_history={1: pheno},
            pedigree_history={},
        )

        expected = _combinations(2, 2) + _combinations(3, 2) + _combinations(4, 2)
        assert view.n_pairs == expected

    def test_mixed_family_sizes(self):
        """Mixed family sizes: sizes 1, 2, 2, 3."""
        # Family 0: size 1 -> 0 pairs
        # Family 1: size 2 -> 1 pair
        # Family 2: size 2 -> 1 pair
        # Family 3: size 3 -> 3 pairs
        # Total: 5 pairs
        fids = np.array([0, 1, 1, 2, 2, 3, 3, 3], dtype=np.int64)
        pheno = _make_pheno_with_fids(8, fids)

        filt = SibPairFilter()
        view = filt.apply(
            generation=1,
            phenotype_history={1: pheno},
            pedigree_history={},
        )

        expected = 0 + 1 + 1 + 3
        assert view.n_pairs == expected

    def test_large_family_10_siblings(self):
        """Large family with 10 siblings: C(10,2) = 45 pairs."""
        fids = np.zeros(10, dtype=np.int64)
        pheno = _make_pheno_with_fids(10, fids)

        filt = SibPairFilter()
        view = filt.apply(
            generation=1,
            phenotype_history={1: pheno},
            pedigree_history={},
        )

        expected = _combinations(10, 2)  # 10 * 9 / 2 = 45
        assert view.n_pairs == expected

    def test_very_large_family_15_siblings(self):
        """Very large family with 15 siblings: C(15,2) = 105 pairs."""
        fids = np.zeros(15, dtype=np.int64)
        pheno = _make_pheno_with_fids(15, fids)

        filt = SibPairFilter()
        view = filt.apply(
            generation=1,
            phenotype_history={1: pheno},
            pedigree_history={},
        )

        expected = _combinations(15, 2)  # 15 * 14 / 2 = 105
        assert view.n_pairs == expected


class TestSibPairVectorizedValidity:
    """Test that generated pairs have valid, distinct indices."""

    def test_no_self_pairs(self):
        """All pairs should have different indices (no i == j)."""
        fids = np.array([0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        pheno = _make_pheno_with_fids(7, fids)

        filt = SibPairFilter()
        view = filt.apply(
            generation=1,
            phenotype_history={1: pheno},
            pedigree_history={},
        )

        # Verify no self-pairs
        for i in range(view.n_pairs):
            assert view.sib1_idx[i] != view.sib2_idx[i], \
                f"Self-pair found at position {i}: ({view.sib1_idx[i]}, {view.sib2_idx[i]})"

    def test_indices_within_bounds(self):
        """All pair indices should be valid (within array bounds)."""
        fids = np.array([0, 0, 0, 1, 1, 2, 2, 2], dtype=np.int64)
        pheno = _make_pheno_with_fids(8, fids)

        filt = SibPairFilter()
        view = filt.apply(
            generation=1,
            phenotype_history={1: pheno},
            pedigree_history={},
        )

        n = pheno.samples.n
        assert np.all(view.sib1_idx >= 0), "Found negative sib1_idx"
        assert np.all(view.sib1_idx < n), "Found sib1_idx >= n"
        assert np.all(view.sib2_idx >= 0), "Found negative sib2_idx"
        assert np.all(view.sib2_idx < n), "Found sib2_idx >= n"

    def test_no_duplicate_pairs(self):
        """No duplicate pairs should exist in the output."""
        fids = np.array([0, 0, 0, 0, 1, 1, 1], dtype=np.int64)
        pheno = _make_pheno_with_fids(7, fids)

        filt = SibPairFilter()
        view = filt.apply(
            generation=1,
            phenotype_history={1: pheno},
            pedigree_history={},
        )

        # Create set of normalized pairs (min, max)
        pair_set = set()
        for i in range(view.n_pairs):
            a, b = int(view.sib1_idx[i]), int(view.sib2_idx[i])
            pair = (min(a, b), max(a, b))
            assert pair not in pair_set, f"Duplicate pair found: {pair}"
            pair_set.add(pair)

    def test_pairs_share_same_fid(self):
        """Each pair should come from the same family (same FID)."""
        fids = np.array([0, 0, 1, 1, 1, 2, 2], dtype=np.int64)
        pheno = _make_pheno_with_fids(7, fids)

        filt = SibPairFilter()
        view = filt.apply(
            generation=1,
            phenotype_history={1: pheno},
            pedigree_history={},
        )

        for i in range(view.n_pairs):
            idx1 = view.sib1_idx[i]
            idx2 = view.sib2_idx[i]
            assert fids[idx1] == fids[idx2], \
                f"Pair {i} crosses families: FID[{idx1}]={fids[idx1]}, FID[{idx2}]={fids[idx2]}"


class TestSibPairVectorizedStructure:
    """Test the structure of the filter output."""

    def test_output_has_correct_fields(self):
        """SibPairView should have all expected fields."""
        fids = np.array([0, 0, 0], dtype=np.int64)
        pheno = _make_pheno_with_fids(3, fids)

        filt = SibPairFilter()
        view = filt.apply(
            generation=1,
            phenotype_history={1: pheno},
            pedigree_history={},
        )

        assert hasattr(view, 'sib1_phenotypes')
        assert hasattr(view, 'sib2_phenotypes')
        assert hasattr(view, 'n_pairs')
        assert hasattr(view, 'sib1_idx')
        assert hasattr(view, 'sib2_idx')

    def test_phenotype_dicts_have_correct_keys(self):
        """Phenotype dicts should contain all phenotype keys from input."""
        fids = np.array([0, 0], dtype=np.int64)
        rng = np.random.RandomState(42)
        sm = SampleMeta(iid=np.arange(2), fid=fids)
        pheno = PhenotypeArray(samples=sm)
        pheno['Y'] = rng.normal(0, 1, 2)
        pheno['Z'] = rng.normal(0, 1, 2)

        filt = SibPairFilter()
        view = filt.apply(
            generation=1,
            phenotype_history={1: pheno},
            pedigree_history={},
        )

        assert 'Y' in view.sib1_phenotypes
        assert 'Z' in view.sib1_phenotypes
        assert 'Y' in view.sib2_phenotypes
        assert 'Z' in view.sib2_phenotypes

    def test_phenotype_arrays_have_correct_length(self):
        """Phenotype arrays should have length n_pairs."""
        fids = np.array([0, 0, 0, 1, 1], dtype=np.int64)
        pheno = _make_pheno_with_fids(5, fids)

        filt = SibPairFilter()
        view = filt.apply(
            generation=1,
            phenotype_history={1: pheno},
            pedigree_history={},
        )

        for key in view.sib1_phenotypes:
            assert len(view.sib1_phenotypes[key]) == view.n_pairs
            assert len(view.sib2_phenotypes[key]) == view.n_pairs

    def test_empty_view_structure(self):
        """Empty view (no pairs) should still have correct structure."""
        fids = np.arange(5, dtype=np.int64)  # all singletons
        pheno = _make_pheno_with_fids(5, fids)

        filt = SibPairFilter()
        view = filt.apply(
            generation=1,
            phenotype_history={1: pheno},
            pedigree_history={},
        )

        assert view.n_pairs == 0
        assert len(view.sib1_idx) == 0
        assert len(view.sib2_idx) == 0
        for key in pheno.keys:
            assert key in view.sib1_phenotypes
            assert key in view.sib2_phenotypes
            assert len(view.sib1_phenotypes[key]) == 0
            assert len(view.sib2_phenotypes[key]) == 0


class TestSibPairVectorizedPhenotypeValues:
    """Test that phenotype values are correctly indexed."""

    def test_phenotype_values_match_indices(self):
        """Phenotype values should match the values at the specified indices."""
        fids = np.array([0, 0, 0], dtype=np.int64)
        vals = np.array([10.0, 20.0, 30.0])
        sm = SampleMeta(iid=np.arange(3), fid=fids)
        pheno = PhenotypeArray(samples=sm)
        pheno['Y'] = vals

        filt = SibPairFilter()
        view = filt.apply(
            generation=1,
            phenotype_history={1: pheno},
            pedigree_history={},
        )

        # Should have C(3,2) = 3 pairs
        assert view.n_pairs == 3

        # Check each pair's phenotype values match the original array
        for i in range(view.n_pairs):
            idx1 = view.sib1_idx[i]
            idx2 = view.sib2_idx[i]
            assert view.sib1_phenotypes['Y'][i] == vals[idx1]
            assert view.sib2_phenotypes['Y'][i] == vals[idx2]

    def test_multiple_phenotypes_aligned(self):
        """Multiple phenotypes should be correctly aligned across pairs."""
        fids = np.array([0, 0], dtype=np.int64)
        y_vals = np.array([100.0, 200.0])
        z_vals = np.array([1.0, 2.0])

        sm = SampleMeta(iid=np.arange(2), fid=fids)
        pheno = PhenotypeArray(samples=sm)
        pheno['Y'] = y_vals
        pheno['Z'] = z_vals

        filt = SibPairFilter()
        view = filt.apply(
            generation=1,
            phenotype_history={1: pheno},
            pedigree_history={},
        )

        # One pair
        assert view.n_pairs == 1
        idx1 = view.sib1_idx[0]
        idx2 = view.sib2_idx[0]

        assert view.sib1_phenotypes['Y'][0] == y_vals[idx1]
        assert view.sib2_phenotypes['Y'][0] == y_vals[idx2]
        assert view.sib1_phenotypes['Z'][0] == z_vals[idx1]
        assert view.sib2_phenotypes['Z'][0] == z_vals[idx2]


class TestSibPairVectorizedWithSimulation:
    """Test SibPairFilter with actual simulation output."""

    def test_filter_on_simulated_generation(self):
        """Run a minimal simulation and verify filter output."""
        # Create founders
        founders = TestSimulation.founder_haplotypes(n=20, m=50, seed=100)
        arch = TestSimulation.simple_architecture(m=50, h2=0.5, seed=200)
        recomb = TestSimulation.recombination_map(m=50, p=0.5)
        mating = TestSimulation.mating_regime(offspring_per_pair=3)

        sim = Simulation(
            founder_haplotypes=founders,
            architecture=arch,
            recombination_map=recomb,
            mating_regime=mating,
            seed=100,
        )
        # Run 2 generations (0 and 1) to produce offspring in generation 1
        sim.run(2)

        # Apply filter to generation 1
        filt = SibPairFilter()

        view = filt.apply(
            generation=1,
            phenotype_history=sim.phenotype_history,
            pedigree_history=sim.pedigree_history,
        )

        # With 20 founders and offspring_per_pair=3, we have 10 pairs * 3 = 30 offspring
        # All in different families -> 0 pairs... Wait, actually they should be grouped by FID
        # Let me check: random mating assigns FIDs based on parent pairs
        # Each pair produces 3 offspring -> 10 families of size 3
        # C(3,2) * 10 = 3 * 10 = 30 pairs

        assert view is not None, "View should not be None"
        assert view.n_pairs > 0, f"Should have some pairs, got {view.n_pairs}"

        # Check validity
        assert len(view.sib1_idx) == view.n_pairs
        assert len(view.sib2_idx) == view.n_pairs
        assert np.all(view.sib1_idx != view.sib2_idx)

    def test_filter_across_multiple_generations(self):
        """Run multiple generations and filter each one."""
        founders = TestSimulation.founder_haplotypes(n=10, m=50, seed=300)
        arch = TestSimulation.simple_architecture(m=50, h2=0.5, seed=400)
        recomb = TestSimulation.recombination_map(m=50, p=0.5)
        mating = TestSimulation.mating_regime(offspring_per_pair=2)

        sim = Simulation(
            founder_haplotypes=founders,
            architecture=arch,
            recombination_map=recomb,
            mating_regime=mating,
            seed=300,
        )
        # Run 3 generations (0, 1, 2) to have data for filtering
        sim.run(3)

        filt = SibPairFilter()

        # Filter gen 1
        view1 = filt.apply(1, sim.phenotype_history, sim.pedigree_history)
        assert view1 is not None
        assert view1.n_pairs > 0

        # Filter gen 2
        view2 = filt.apply(2, sim.phenotype_history, sim.pedigree_history)
        assert view2 is not None
        assert view2.n_pairs > 0


class TestSibPairVectorizedEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_missing_generation_returns_none(self):
        """Requesting a generation not in history returns None."""
        fids = np.array([0, 0], dtype=np.int64)
        pheno = _make_pheno_with_fids(2, fids)

        filt = SibPairFilter()
        view = filt.apply(
            generation=5,  # not in history
            phenotype_history={1: pheno},
            pedigree_history={},
        )

        assert view is None

    def test_empty_phenotype_history(self):
        """Empty phenotype history returns None."""
        filt = SibPairFilter()
        view = filt.apply(
            generation=1,
            phenotype_history={},
            pedigree_history={},
        )

        assert view is None

    def test_all_families_same_large_size(self):
        """Multiple families all with the same large size."""
        # 5 families, each with 6 members
        # C(6,2) * 5 = 15 * 5 = 75 pairs
        fids = np.repeat(np.arange(5), 6)
        pheno = _make_pheno_with_fids(30, fids)

        filt = SibPairFilter()
        view = filt.apply(
            generation=1,
            phenotype_history={1: pheno},
            pedigree_history={},
        )

        expected = _combinations(6, 2) * 5
        assert view.n_pairs == expected

    def test_families_with_mixed_sizes_including_large(self):
        """Mix of small, medium, and large families."""
        # Sizes: 1, 2, 5, 10, 15
        # Pairs: 0, 1, 10, 45, 105 = 161
        fids = np.concatenate([
            np.full(1, 0),
            np.full(2, 1),
            np.full(5, 2),
            np.full(10, 3),
            np.full(15, 4),
        ])
        pheno = _make_pheno_with_fids(33, fids)

        filt = SibPairFilter()
        view = filt.apply(
            generation=1,
            phenotype_history={1: pheno},
            pedigree_history={},
        )

        expected = 0 + 1 + 10 + 45 + 105
        assert view.n_pairs == expected
