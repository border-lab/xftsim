"""
Unit tests for TrioFilter and SibPairFilter edge cases.

Tests:
1. TrioFilter at gen 0 returns None
2. TrioFilter missing pedigree returns None
3. TrioFilter missing parent phenotypes returns None
4. TrioFilter aligns keys present in both generations
5. SibPairFilter all singletons returns empty
6. SibPairFilter mixed family sizes
7. SibPairFilter key propagation
8. FilteredView dataclasses
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, NPhenotypeArray, PedigreeArray
from xftsim.filters import TrioFilter, SibPairFilter, TrioView, SibPairView


def _make_pheno(n, keys_values, generation=0):
    sm = SampleMeta(iid=np.arange(n), generation=generation)
    pheno = NPhenotypeArray(samples=sm)
    for k, v in keys_values.items():
        pheno._values[k] = np.asarray(v, dtype=np.float64)
    return pheno


def _make_pedigree(n_offspring, parent_n):
    sm = SampleMeta(
        iid=np.arange(n_offspring),
        fid=np.repeat(np.arange(n_offspring // 2), 2)[:n_offspring],
        generation=1,
    )
    n_pairs = n_offspring // 2
    maternal_idx = np.repeat(np.arange(n_pairs), 2)[:n_offspring]
    paternal_idx = np.repeat(np.arange(n_pairs, 2 * n_pairs), 2)[:n_offspring]
    return PedigreeArray(
        offspring_samples=sm,
        maternal_idx=maternal_idx,
        paternal_idx=paternal_idx,
        parent_n=parent_n,
    )


class TestTrioFilterEdgeCases:
    def test_gen0_returns_none(self):
        filt = TrioFilter()
        pheno = _make_pheno(10, {'Y': np.arange(10, dtype=float)})
        result = filt.apply(0, {0: pheno}, {})
        assert result is None

    def test_missing_pedigree_returns_none(self):
        filt = TrioFilter()
        pheno0 = _make_pheno(10, {'Y': np.arange(10, dtype=float)}, generation=0)
        pheno1 = _make_pheno(10, {'Y': np.arange(10, dtype=float)}, generation=1)
        result = filt.apply(1, {0: pheno0, 1: pheno1}, {})
        assert result is None

    def test_missing_parent_phenotype_returns_none(self):
        """If parent generation phenotypes are pruned, returns None."""
        filt = TrioFilter()
        pheno1 = _make_pheno(10, {'Y': np.arange(10, dtype=float)}, generation=1)
        ped = _make_pedigree(10, parent_n=10)
        result = filt.apply(1, {1: pheno1}, {1: ped})
        assert result is None

    def test_key_alignment(self):
        """Keys present in offspring but not parent should be skipped in parent dicts."""
        filt = TrioFilter()
        pheno0 = _make_pheno(10, {'Y': np.arange(10, dtype=float)}, generation=0)
        pheno1 = _make_pheno(10, {
            'Y': np.arange(10, dtype=float),
            'Z': np.ones(10),
        }, generation=1)
        ped = _make_pedigree(10, parent_n=10)
        result = filt.apply(1, {0: pheno0, 1: pheno1}, {1: ped})
        assert result is not None
        assert 'Y' in result.offspring_phenotypes
        assert 'Y' in result.mother_phenotypes
        # Z is in offspring but not in parent → not in mother/father
        assert 'Z' not in result.mother_phenotypes
        # Z should still be in offspring
        assert 'Z' in result.offspring_phenotypes

    def test_n_trios_matches_offspring_count(self):
        filt = TrioFilter()
        pheno0 = _make_pheno(10, {'Y': np.arange(10, dtype=float)}, generation=0)
        pheno1 = _make_pheno(8, {'Y': np.arange(8, dtype=float)}, generation=1)
        ped = _make_pedigree(8, parent_n=10)
        result = filt.apply(1, {0: pheno0, 1: pheno1}, {1: ped})
        assert result.n_trios == 8

    def test_offspring_values_are_copies(self):
        """Offspring phenotypes in trio view should be copies, not references."""
        filt = TrioFilter()
        vals = np.arange(10, dtype=float)
        pheno0 = _make_pheno(10, {'Y': vals.copy()}, generation=0)
        pheno1 = _make_pheno(10, {'Y': vals.copy()}, generation=1)
        ped = _make_pedigree(10, parent_n=10)
        result = filt.apply(1, {0: pheno0, 1: pheno1}, {1: ped})
        # Modifying the trio view should not affect the original phenotypes
        result.offspring_phenotypes['Y'][0] = 999.0
        assert pheno1['Y'][0] != 999.0


class TestSibPairFilterEdgeCases:
    def test_all_singletons(self):
        """If every family has exactly 1 member, returns empty SibPairView."""
        filt = SibPairFilter()
        sm = SampleMeta(
            iid=np.arange(5),
            fid=np.arange(5),
            generation=1,
        )
        pheno = NPhenotypeArray(samples=sm)
        pheno._values['Y'] = np.arange(5, dtype=float)
        result = filt.apply(1, {1: pheno}, {})
        assert result.n_pairs == 0
        assert len(result.sib1_phenotypes['Y']) == 0

    def test_mixed_family_sizes(self):
        """Families of size 1, 2, and 3."""
        filt = SibPairFilter()
        fids = np.array([0, 1, 1, 2, 2, 2])
        sm = SampleMeta(iid=np.arange(6), fid=fids, generation=1)
        pheno = NPhenotypeArray(samples=sm)
        pheno._values['Y'] = np.arange(6, dtype=float)
        result = filt.apply(1, {1: pheno}, {})
        # Family 0: 0 pairs, Family 1: 1 pair, Family 2: C(3,2)=3 pairs
        assert result.n_pairs == 4

    def test_key_propagation(self):
        """All phenotype keys should appear in sib pair view."""
        filt = SibPairFilter()
        fids = np.array([0, 0, 1, 1])
        sm = SampleMeta(iid=np.arange(4), fid=fids, generation=1)
        pheno = NPhenotypeArray(samples=sm)
        pheno._values['A'] = np.array([1.0, 2.0, 3.0, 4.0])
        pheno._values['B'] = np.array([10.0, 20.0, 30.0, 40.0])
        result = filt.apply(1, {1: pheno}, {})
        assert 'A' in result.sib1_phenotypes
        assert 'B' in result.sib1_phenotypes
        assert result.n_pairs == 2

    def test_missing_generation_returns_none(self):
        filt = SibPairFilter()
        result = filt.apply(5, {}, {})
        assert result is None

    def test_sib_pair_indices_valid(self):
        """sib1_idx and sib2_idx should be valid sample indices."""
        filt = SibPairFilter()
        fids = np.array([0, 0, 0])
        sm = SampleMeta(iid=np.arange(3), fid=fids, generation=1)
        pheno = NPhenotypeArray(samples=sm)
        pheno._values['Y'] = np.array([10.0, 20.0, 30.0])
        result = filt.apply(1, {1: pheno}, {})
        assert result.n_pairs == 3
        assert len(result.sib1_idx) == 3
        assert len(result.sib2_idx) == 3
        assert np.all(result.sib1_idx >= 0)
        assert np.all(result.sib2_idx < 3)

    def test_large_family(self):
        """Family of size 10 → C(10,2) = 45 pairs."""
        filt = SibPairFilter()
        fids = np.zeros(10, dtype=int)
        sm = SampleMeta(iid=np.arange(10), fid=fids, generation=1)
        pheno = NPhenotypeArray(samples=sm)
        pheno._values['Y'] = np.arange(10, dtype=float)
        result = filt.apply(1, {1: pheno}, {})
        assert result.n_pairs == 45  # C(10,2)


class TestFilteredViewDataclasses:
    def test_trio_view_creation(self):
        tv = TrioView(
            offspring_phenotypes={'Y': np.array([1.0, 2.0])},
            mother_phenotypes={'Y': np.array([3.0, 4.0])},
            father_phenotypes={'Y': np.array([5.0, 6.0])},
            n_trios=2,
        )
        assert tv.n_trios == 2
        assert len(tv.offspring_phenotypes['Y']) == 2

    def test_sib_pair_view_creation(self):
        spv = SibPairView(
            sib1_phenotypes={'Y': np.array([1.0])},
            sib2_phenotypes={'Y': np.array([2.0])},
            n_pairs=1,
            sib1_idx=np.array([0]),
            sib2_idx=np.array([1]),
        )
        assert spv.n_pairs == 1

    def test_sib_pair_view_default_none_indices(self):
        spv = SibPairView(
            sib1_phenotypes={},
            sib2_phenotypes={},
            n_pairs=0,
        )
        assert spv.sib1_idx is None
        assert spv.sib2_idx is None
