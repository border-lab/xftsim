"""
Unit tests for _resolve_grouping and _ParentalComponent warning/fallback paths.

Tests:
1. _resolve_grouping: FID, sex, mother, father, extra fields, unknown variable
2. _ParentalComponent: gen-0 no founder (returns zeros + warning),
   pruned phenotype_history (returns zeros + warning), missing phenotype name
3. Sibling component edge cases: single-member groups, all-zero, all-positive
"""
import numpy as np
import pytest
import warnings

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray, NPhenotypeArray, PedigreeArray
from xftsim.narch import (
    _resolve_grouping,
    MotherComponent, FatherComponent, ParentComponent,
    NoiseComponent, AggregationComponent, GeneticComponent,
    SiblingMeanComponent, SiblingSumComponent, SiblingAnyComponent,
    SiblingCountComponent, SiblingEldestComponent, SiblingYoungestComponent,
    Architecture, ArchNode,
)
from xftsim.neffect import AdditiveEffects

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_haplotypes(n=20, m=10, seed=42, extra=None, fid=None, sex=None):
    rng = np.random.RandomState(seed)
    geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
    if sex is None:
        sex = np.tile([0, 1], (n + 1) // 2)[:n]
    if fid is None:
        fid = np.arange(n) // 2
    sm = SampleMeta(iid=np.arange(n), fid=fid, sex=sex, extra=extra or {})
    vm = VariantMeta(vid=np.arange(m))
    return DenseHaplotypeArray(genotypes=geno, samples=sm, variants=vm)


def _make_pedigree(n=20):
    """Make a simple pedigree (all offspring from first pair)."""
    sm = SampleMeta(iid=np.arange(n))
    return PedigreeArray(
        offspring_samples=sm,
        maternal_idx=np.zeros(n, dtype=np.int64),
        paternal_idx=np.ones(n, dtype=np.int64),
        parent_n=n,
    )


# ── _resolve_grouping ────────────────────────────────────────────────────────

class TestResolveGrouping:
    def test_none_returns_none(self):
        """grouping=None → per-individual, returns None."""
        hap = _make_haplotypes()
        result = _resolve_grouping(None, hap)
        assert result is None

    def test_fid(self):
        """grouping='FID' returns FID array."""
        fid = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4,
                         5, 5, 6, 6, 7, 7, 8, 8, 9, 9])
        hap = _make_haplotypes(fid=fid)
        result = _resolve_grouping('FID', hap)
        np.testing.assert_array_equal(result, fid)

    def test_sex(self):
        """grouping='sex' returns sex array."""
        sex = np.tile([0, 1], 10)
        hap = _make_haplotypes(sex=sex)
        result = _resolve_grouping('sex', hap)
        np.testing.assert_array_equal(result, sex)

    def test_mother_gen0_warns_and_returns_none(self):
        """grouping='mother' at gen 0 should warn and return None."""
        hap = _make_haplotypes()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _resolve_grouping('mother', hap, generation=0, pedigree_history={})
            assert result is None
            assert any("falling back to IID" in str(x.message) for x in w)

    def test_father_gen0_warns_and_returns_none(self):
        """grouping='father' at gen 0 should warn and return None."""
        hap = _make_haplotypes()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _resolve_grouping('father', hap, generation=0, pedigree_history={})
            assert result is None
            assert any("falling back to IID" in str(x.message) for x in w)

    def test_mother_no_pedigree_for_gen_warns(self):
        """grouping='mother' at gen 3 with no gen 3 pedigree → warn."""
        hap = _make_haplotypes()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _resolve_grouping('mother', hap, generation=3, pedigree_history={})
            assert result is None

    def test_mother_with_pedigree(self):
        """grouping='mother' at gen 1 with pedigree returns maternal_idx."""
        hap = _make_haplotypes()
        ped = _make_pedigree(n=20)
        result = _resolve_grouping('mother', hap, generation=1, pedigree_history={1: ped})
        np.testing.assert_array_equal(result, ped.maternal_idx)

    def test_father_with_pedigree(self):
        """grouping='father' at gen 1 with pedigree returns paternal_idx."""
        hap = _make_haplotypes()
        ped = _make_pedigree(n=20)
        result = _resolve_grouping('father', hap, generation=1, pedigree_history={1: ped})
        np.testing.assert_array_equal(result, ped.paternal_idx)

    def test_extra_field(self):
        """grouping by extra field on SampleMeta."""
        batch = np.array([1, 1, 2, 2, 3, 3, 1, 1, 2, 2,
                          3, 3, 1, 1, 2, 2, 3, 3, 1, 1])
        hap = _make_haplotypes(extra={'batch': batch})
        result = _resolve_grouping('batch', hap)
        np.testing.assert_array_equal(result, batch)

    def test_unknown_variable_raises(self):
        """Unknown grouping variable should raise ValueError."""
        hap = _make_haplotypes()
        with pytest.raises(ValueError, match="Unknown grouping variable"):
            _resolve_grouping('nonexistent_field', hap)

    def test_unknown_variable_message_lists_options(self):
        """Error message should list available options."""
        hap = _make_haplotypes()
        with pytest.raises(ValueError, match="FID.*sex.*mother.*father"):
            _resolve_grouping('xyz', hap)


# ── _ParentalComponent warning paths ────────────────────────────────────────

class TestParentalComponentWarnings:
    def _make_context(self, n=20, m=10):
        """Build haplotypes and basic context for parental testing."""
        hap = _make_haplotypes(n=n, m=m)
        return hap

    def test_mother_gen0_no_founder_warns_returns_zeros(self):
        """MotherComponent at gen 0 with no founder → zeros + warning."""
        hap = self._make_context()
        comp = MotherComponent('Y')
        node = ArchNode(
            outputs=['VT_mother'],
            component=comp,
            inputs=[],
        )
        pheno = NPhenotypeArray(samples=hap.samples)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = comp.compute(node, hap, pheno, generation=0, pedigree_history={})
            assert np.all(result == 0.0)
            assert len(result) == hap.n
            assert any("returning zeros" in str(x.message) for x in w)

    def test_father_gen0_no_founder_warns_returns_zeros(self):
        """FatherComponent at gen 0 with no founder → zeros + warning."""
        hap = self._make_context()
        comp = FatherComponent('Y')
        node = ArchNode(outputs=['VT_father'], component=comp, inputs=[])
        pheno = NPhenotypeArray(samples=hap.samples)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = comp.compute(node, hap, pheno, generation=0, pedigree_history={})
            assert np.all(result == 0.0)

    def test_parent_gen0_no_founder_warns_returns_zeros(self):
        """ParentComponent at gen 0 with no founder → zeros + warning."""
        hap = self._make_context()
        comp = ParentComponent('Y')
        node = ArchNode(outputs=['VT_parent'], component=comp, inputs=[])
        pheno = NPhenotypeArray(samples=hap.samples)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = comp.compute(node, hap, pheno, generation=0, pedigree_history={})
            assert np.all(result == 0.0)

    def test_mother_gen0_with_founder_uses_founder(self):
        """MotherComponent at gen 0 with founder_component → uses founder."""
        hap = self._make_context()
        founder = NoiseComponent(variance=1.0)
        comp = MotherComponent('Y', founder_component=founder)
        node = ArchNode(outputs=['VT_mother'], component=comp, inputs=[])
        pheno = NPhenotypeArray(samples=hap.samples)
        rng = np.random.RandomState(42)
        result = comp.compute(node, hap, pheno, generation=0, pedigree_history={}, rng=rng)
        # Should NOT be all zeros — founder noise produces non-zero values
        assert np.var(result) > 0

    def test_pruned_phenotype_history_warns_returns_zeros(self):
        """When prev_gen is pruned from phenotype_history → zeros + warning."""
        hap = self._make_context()
        ped = _make_pedigree(n=20)
        comp = MotherComponent('Y')
        node = ArchNode(outputs=['VT_mother'], component=comp, inputs=[])
        pheno = NPhenotypeArray(samples=hap.samples)
        # generation=2, pedigree exists, but phenotype_history has no gen 1
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = comp.compute(
                node, hap, pheno,
                generation=2,
                pedigree_history={2: ped},
                phenotype_history={},  # gen 1 pruned
            )
            assert np.all(result == 0.0)
            assert any("pruned by retention" in str(x.message) for x in w)

    def test_missing_phenotype_name_raises(self):
        """When phenotype name not in prev gen → ValueError."""
        hap = self._make_context()
        ped = _make_pedigree(n=20)
        comp = MotherComponent('nonexistent_trait')
        node = ArchNode(outputs=['VT_mother'], component=comp, inputs=[])
        pheno = NPhenotypeArray(samples=hap.samples)
        # prev gen exists but doesn't have the requested trait
        prev_pheno = NPhenotypeArray(
            samples=SampleMeta(iid=np.arange(20)),
            values={'other_trait': np.ones(20)},
        )
        with pytest.raises(ValueError, match="not found"):
            comp.compute(
                node, hap, pheno,
                generation=1,
                pedigree_history={1: ped},
                phenotype_history={0: prev_pheno},
            )


# ── Expression evaluator additional edge cases ──────────────────────────────

class TestExpressionEdgeCases:
    @pytest.fixture
    def haplotypes(self):
        return _make_haplotypes()

    def test_unary_minus_after_operator(self, haplotypes):
        """Unary minus after an operator: A + -B."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('B', NoiseComponent(variance=1.0))
        arch.add('Y', AggregationComponent('A + -B'))
        rng = np.random.RandomState(42)
        pheno = arch.compute(haplotypes, rng=rng)
        expected = pheno['A'] - pheno['B']
        np.testing.assert_allclose(pheno['Y'], expected, atol=1e-10)

    def test_unary_minus_after_multiply(self, haplotypes):
        """Unary minus after multiply: A * -2."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('Y', AggregationComponent('A * -2'))
        rng = np.random.RandomState(42)
        pheno = arch.compute(haplotypes, rng=rng)
        expected = pheno['A'] * -2.0
        np.testing.assert_allclose(pheno['Y'], expected, atol=1e-10)

    def test_unary_minus_in_parentheses(self, haplotypes):
        """Unary minus inside parens: (-A)."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('Y', AggregationComponent('(-A)'))
        rng = np.random.RandomState(42)
        pheno = arch.compute(haplotypes, rng=rng)
        np.testing.assert_allclose(pheno['Y'], -pheno['A'], atol=1e-10)

    def test_numeric_literal_decimal(self, haplotypes):
        """Decimal literal in expression: 0.001 * A."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('Y', AggregationComponent('0.001 * A'))
        rng = np.random.RandomState(42)
        pheno = arch.compute(haplotypes, rng=rng)
        np.testing.assert_allclose(pheno['Y'], 0.001 * pheno['A'], atol=1e-12)

    def test_numeric_literal_large(self, haplotypes):
        """Large number literal: 100 * A."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('Y', AggregationComponent('100 * A'))
        rng = np.random.RandomState(42)
        pheno = arch.compute(haplotypes, rng=rng)
        np.testing.assert_allclose(pheno['Y'], 100.0 * pheno['A'], atol=1e-8)

    def test_multiple_operations_precedence(self, haplotypes):
        """Check precedence: A + B * 2 == A + (B*2), not (A+B)*2."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('B', NoiseComponent(variance=1.0))
        arch.add('Y', AggregationComponent('A + B * 2'))
        rng = np.random.RandomState(42)
        pheno = arch.compute(haplotypes, rng=rng)
        expected = pheno['A'] + pheno['B'] * 2
        np.testing.assert_allclose(pheno['Y'], expected, atol=1e-10)

    def test_complex_expression(self, haplotypes):
        """Complex: (A + B) * 0.5 - C / 2."""
        arch = Architecture()
        arch.add('A', NoiseComponent(variance=1.0))
        arch.add('B', NoiseComponent(variance=1.0))
        arch.add('C', NoiseComponent(variance=1.0))
        arch.add('Y', AggregationComponent('(A + B) * 0.5 - C / 2'))
        rng = np.random.RandomState(42)
        pheno = arch.compute(haplotypes, rng=rng)
        expected = (pheno['A'] + pheno['B']) * 0.5 - pheno['C'] / 2
        np.testing.assert_allclose(pheno['Y'], expected, atol=1e-10)

    def test_closing_paren_before_opening_raises(self):
        """Closing paren without opening should raise."""
        from xftsim.narch import _shunting_yard, _tokenize
        with pytest.raises(ValueError, match="[Pp]arenthes"):
            _shunting_yard(_tokenize(')A + B'))

    def test_unclosed_open_paren_raises(self):
        """Unclosed opening paren should raise."""
        from xftsim.narch import _shunting_yard, _tokenize
        with pytest.raises(ValueError, match="[Pp]arenthes"):
            _shunting_yard(_tokenize('(A + B'))

    def test_trailing_unary_minus_raises(self):
        """Unary minus at end with no following token should raise."""
        from xftsim.narch import _shunting_yard, _tokenize
        tokens = _tokenize('A + -')
        with pytest.raises(ValueError, match="end of expression"):
            _shunting_yard(tokens)


# ── Sibling component edge cases ────────────────────────────────────────────

class TestSiblingComponentEdgeCases:
    def test_mean_single_member_groups(self):
        """SiblingMean with all single-member groups returns original values."""
        comp = SiblingMeanComponent('Y')
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        labels = np.array([0, 1, 2, 3, 4])  # all unique
        result = comp._aggregate_groups(values, labels)
        np.testing.assert_allclose(result, values)

    def test_mean_two_member_groups(self):
        """SiblingMean with pairs."""
        comp = SiblingMeanComponent('Y')
        values = np.array([1.0, 3.0, 5.0, 7.0])
        labels = np.array([0, 0, 1, 1])
        result = comp._aggregate_groups(values, labels)
        np.testing.assert_allclose(result, [2.0, 2.0, 6.0, 6.0])

    def test_sum_single_member(self):
        """SiblingSum with single-member groups."""
        comp = SiblingSumComponent('Y')
        values = np.array([10.0, 20.0, 30.0])
        labels = np.array([0, 1, 2])
        result = comp._aggregate_groups(values, labels)
        np.testing.assert_allclose(result, values)

    def test_any_all_zero(self):
        """SiblingAny with all-zero values returns all zeros."""
        comp = SiblingAnyComponent('Y')
        values = np.array([0.0, 0.0, 0.0, 0.0])
        labels = np.array([0, 0, 1, 1])
        result = comp._aggregate_groups(values, labels)
        np.testing.assert_allclose(result, [0.0, 0.0, 0.0, 0.0])

    def test_any_all_positive(self):
        """SiblingAny with all-positive values returns all ones."""
        comp = SiblingAnyComponent('Y')
        values = np.array([1.0, 2.0, 3.0, 4.0])
        labels = np.array([0, 0, 1, 1])
        result = comp._aggregate_groups(values, labels)
        np.testing.assert_allclose(result, [1.0, 1.0, 1.0, 1.0])

    def test_any_mixed(self):
        """SiblingAny: one positive in group → whole group is 1."""
        comp = SiblingAnyComponent('Y')
        values = np.array([0.0, 1.0, 0.0, 0.0])
        labels = np.array([0, 0, 1, 1])
        result = comp._aggregate_groups(values, labels)
        np.testing.assert_allclose(result, [1.0, 1.0, 0.0, 0.0])

    def test_count_unequal_groups(self):
        """SiblingCount with unequal group sizes."""
        comp = SiblingCountComponent('Y')
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        labels = np.array([0, 0, 0, 1, 1])
        result = comp._aggregate_groups(values, labels)
        np.testing.assert_allclose(result, [3.0, 3.0, 3.0, 2.0, 2.0])

    def test_eldest_returns_first_in_order(self):
        """SiblingEldest returns value of first member (array order)."""
        comp = SiblingEldestComponent('Y')
        values = np.array([10.0, 20.0, 30.0, 40.0])
        labels = np.array([0, 0, 1, 1])
        result = comp._aggregate_groups(values, labels)
        np.testing.assert_allclose(result, [10.0, 10.0, 30.0, 30.0])

    def test_youngest_returns_last_in_order(self):
        """SiblingYoungest returns value of last member (array order)."""
        comp = SiblingYoungestComponent('Y')
        values = np.array([10.0, 20.0, 30.0, 40.0])
        labels = np.array([0, 0, 1, 1])
        result = comp._aggregate_groups(values, labels)
        np.testing.assert_allclose(result, [20.0, 20.0, 40.0, 40.0])

    def test_eldest_single_member(self):
        """Eldest with single-member group returns that member's value."""
        comp = SiblingEldestComponent('Y')
        values = np.array([5.0, 10.0, 15.0])
        labels = np.array([0, 1, 2])
        result = comp._aggregate_groups(values, labels)
        np.testing.assert_allclose(result, values)

    def test_youngest_single_member(self):
        """Youngest with single-member group returns that member's value."""
        comp = SiblingYoungestComponent('Y')
        values = np.array([5.0, 10.0, 15.0])
        labels = np.array([0, 1, 2])
        result = comp._aggregate_groups(values, labels)
        np.testing.assert_allclose(result, values)

    def test_large_group(self):
        """SiblingMean with a single large group."""
        comp = SiblingMeanComponent('Y')
        values = np.arange(100, dtype=np.float64)
        labels = np.zeros(100, dtype=int)
        result = comp._aggregate_groups(values, labels)
        expected = np.full(100, np.mean(values))
        np.testing.assert_allclose(result, expected)
