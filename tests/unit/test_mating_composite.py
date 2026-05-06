"""
Unit tests for LinearAssortativeMating composite computation edge cases.

Tests:
1. All components have zero variance
2. No components found in phenotypes (all missing)
3. One component present, one missing
4. Zero-variance component skipped in standardization
5. Disassortative pairing: negative scores for males
6. Assortative pairing: spouse correlation positive
7. Composite with single component
8. Composite with many components
"""
import numpy as np
import pytest

from xftsim.mate import LinearAssortativeMating, RandomMating
from xftsim.struct import SampleMeta, NPhenotypeArray


def _make_phenotypes(n, component_values=None, seed=42):
    """Create sample metadata and phenotypes for mating tests."""
    rng = np.random.RandomState(seed)
    sex = np.tile([0, 1], n // 2)[:n]  # alternating 0, 1
    sm = SampleMeta(iid=np.arange(n), sex=sex)
    pheno = NPhenotypeArray(samples=sm)
    if component_values:
        for name, vals in component_values.items():
            pheno._values[name] = np.asarray(vals, dtype=np.float64)
    return sm, pheno


class TestCompositeZeroVariance:
    def test_all_zero_variance_components(self):
        """All components constant → composite is constant → essentially random mating."""
        n = 20
        sm, pheno = _make_phenotypes(n, component_values={
            'Y': np.ones(n),
            'Z': np.full(n, 5.0),
        })
        mating = LinearAssortativeMating(component_names=['Y', 'Z'], r=0.5)
        # Should not crash — constant composite means noise dominates
        assignment = mating.mate(sm, rng=np.random.RandomState(42), phenotypes=pheno)
        assert assignment.n_offspring == (n // 2) * 2

    def test_one_zero_variance_one_normal(self):
        """Mixed: one constant component, one variable."""
        n = 20
        rng = np.random.RandomState(42)
        sm, pheno = _make_phenotypes(n, component_values={
            'Y': np.ones(n),  # constant
            'Z': rng.normal(0, 1, n),  # variable
        })
        mating = LinearAssortativeMating(component_names=['Y', 'Z'], r=0.5)
        assignment = mating.mate(sm, rng=np.random.RandomState(42), phenotypes=pheno)
        assert assignment.n_offspring == (n // 2) * 2


class TestCompositeMissingComponents:
    def test_no_components_found(self):
        """No component_names found in phenotypes → composite is all zeros."""
        n = 20
        sm, pheno = _make_phenotypes(n, component_values={'OTHER': np.ones(n)})
        mating = LinearAssortativeMating(component_names=['MISSING1', 'MISSING2'], r=0.5)
        # Should still work — composite is zero, noise dominates
        assignment = mating.mate(sm, rng=np.random.RandomState(42), phenotypes=pheno)
        assert assignment.n_offspring == (n // 2) * 2

    def test_partial_components(self):
        """Some components found, some missing → uses available ones."""
        n = 20
        rng = np.random.RandomState(42)
        sm, pheno = _make_phenotypes(n, component_values={
            'Y': rng.normal(0, 1, n),
        })
        mating = LinearAssortativeMating(component_names=['Y', 'MISSING'], r=0.5)
        assignment = mating.mate(sm, rng=np.random.RandomState(42), phenotypes=pheno)
        assert assignment.n_offspring == (n // 2) * 2


class TestDisassortativeSpouseCorrelation:
    def test_negative_r_negates_male_scores(self):
        """With r < 0, high-scoring females pair with low-scoring males."""
        n = 100
        rng = np.random.RandomState(42)
        vals = rng.normal(0, 1, n)
        sm, pheno = _make_phenotypes(n, component_values={'Y': vals})

        mating = LinearAssortativeMating(component_names=['Y'], r=-0.5)
        assignment = mating.mate(sm, rng=np.random.RandomState(42), phenotypes=pheno)

        # Extract spouse phenotypes
        mom_vals = vals[assignment.maternal_idx[::2]]  # one per pair
        dad_vals = vals[assignment.paternal_idx[::2]]
        corr = np.corrcoef(mom_vals, dad_vals)[0, 1]
        # Should be negative (or near-zero) with disassortative mating
        assert corr < 0.3, f"Disassortative corr={corr} should be negative or near-zero"


class TestAssortativeSpouseCorrelation:
    def test_positive_r_positive_correlation(self):
        """With r > 0, high-scoring pairs together."""
        n = 200
        rng = np.random.RandomState(42)
        vals = rng.normal(0, 1, n)
        sm, pheno = _make_phenotypes(n, component_values={'Y': vals})

        mating = LinearAssortativeMating(component_names=['Y'], r=0.8)
        assignment = mating.mate(sm, rng=np.random.RandomState(42), phenotypes=pheno)

        mom_vals = vals[assignment.maternal_idx[::2]]
        dad_vals = vals[assignment.paternal_idx[::2]]
        corr = np.corrcoef(mom_vals, dad_vals)[0, 1]
        assert corr > 0.2, f"Assortative corr={corr} should be positive"


class TestCompositeSingleComponent:
    def test_single_component(self):
        """Single component → composite = standardized single component."""
        n = 20
        rng = np.random.RandomState(42)
        sm, pheno = _make_phenotypes(n, component_values={
            'Y': rng.normal(0, 1, n),
        })
        mating = LinearAssortativeMating(component_names=['Y'], r=0.5)
        assignment = mating.mate(sm, rng=np.random.RandomState(42), phenotypes=pheno)
        assert assignment.n_offspring > 0


class TestCompositeManyComponents:
    def test_many_components(self):
        """Many components → composite averages all."""
        n = 20
        rng = np.random.RandomState(42)
        vals = {}
        for i in range(10):
            vals[f'trait_{i}'] = rng.normal(0, 1, n)
        sm, pheno = _make_phenotypes(n, component_values=vals)

        comp_names = [f'trait_{i}' for i in range(10)]
        mating = LinearAssortativeMating(component_names=comp_names, r=0.5)
        assignment = mating.mate(sm, rng=np.random.RandomState(42), phenotypes=pheno)
        assert assignment.n_offspring > 0


class TestAssortativeOffspringStructure:
    def test_offspring_per_pair_respected(self):
        """offspring_per_pair is applied correctly."""
        n = 20
        rng = np.random.RandomState(42)
        sm, pheno = _make_phenotypes(n, component_values={
            'Y': rng.normal(0, 1, n),
        })

        for opp in [1, 2, 3]:
            mating = LinearAssortativeMating(component_names=['Y'], r=0.5,
                                             offspring_per_pair=opp)
            assignment = mating.mate(sm, rng=np.random.RandomState(42), phenotypes=pheno)
            n_pairs = n // 2
            assert assignment.n_offspring == n_pairs * opp

    def test_maternal_idx_are_females(self):
        """Maternal indices should point to females (sex=0)."""
        n = 20
        rng = np.random.RandomState(42)
        sm, pheno = _make_phenotypes(n, component_values={
            'Y': rng.normal(0, 1, n),
        })
        mating = LinearAssortativeMating(component_names=['Y'], r=0.5)
        assignment = mating.mate(sm, rng=np.random.RandomState(42), phenotypes=pheno)

        for idx in assignment.maternal_idx:
            assert sm.sex[idx] == 0, f"Maternal index {idx} is not female"

    def test_paternal_idx_are_males(self):
        """Paternal indices should point to males (sex=1)."""
        n = 20
        rng = np.random.RandomState(42)
        sm, pheno = _make_phenotypes(n, component_values={
            'Y': rng.normal(0, 1, n),
        })
        mating = LinearAssortativeMating(component_names=['Y'], r=0.5)
        assignment = mating.mate(sm, rng=np.random.RandomState(42), phenotypes=pheno)

        for idx in assignment.paternal_idx:
            assert sm.sex[idx] == 1, f"Paternal index {idx} is not male"
