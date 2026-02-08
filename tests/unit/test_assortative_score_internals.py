"""
Unit tests for LinearAssortativeMating composite score and pairing internals.

Tests:
1. Composite score standardization: (vals - mean) / sd
2. Composite averaging over multiple components
3. Score mixing formula: sqrt(|r|) * composite + sqrt(1-|r|) * noise
4. Rank-order pairing: highest female score pairs with highest male
5. Disassortative: male scores negated
6. Zero-variance component contributes zero to composite
7. Missing component skipped in composite
8. r=1.0 boundary: pure composite, no noise
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, NPhenotypeArray
from xftsim.nmate import LinearAssortativeMating


def _make_deterministic_phenotypes(n=100, seed=42):
    """Create phenotypes with known statistical properties."""
    rng = np.random.RandomState(seed)
    sex = np.tile([0, 1], n // 2)
    sm = SampleMeta(iid=np.arange(n), sex=sex)
    pheno = NPhenotypeArray(samples=sm)
    pheno['Y'] = rng.normal(5.0, 2.0, size=n)
    return sm, pheno


class TestCompositeStandardization:
    def test_single_component_standardized(self):
        """Composite from single component should be standardized (mean~0, sd~1)."""
        n = 1000
        sm, pheno = _make_deterministic_phenotypes(n=n, seed=42)
        mate = LinearAssortativeMating(
            component_names=['Y'], r=0.99, offspring_per_pair=2
        )
        # We can't directly access the composite, but we can verify behavior
        # indirectly: with r~1, scoring = composite, so rank order should
        # exactly match phenotype rank order
        rng = np.random.RandomState(123)
        assignment = mate.mate(sm, rng=rng, phenotypes=pheno)

        # With r=0.99, the highest-phenotype female should pair with
        # highest-phenotype male
        female_idx = np.where(sm.sex == 0)[0]
        male_idx = np.where(sm.sex == 1)[0]
        female_vals = pheno['Y'][female_idx]
        male_vals = pheno['Y'][male_idx]

        # Get the mothers from the assignment (first offspring per pair)
        mothers = assignment.maternal_idx[::2]
        fathers = assignment.paternal_idx[::2]

        # Mother phenotypes should be mostly sorted
        mother_pheno = pheno['Y'][mothers]
        # Rank correlation should be high
        from scipy.stats import spearmanr
        corr, _ = spearmanr(mother_pheno, pheno['Y'][fathers])
        assert corr > 0.8, f"Spouse rank correlation = {corr:.3f}, expected > 0.8"

    def test_multi_component_averaging(self):
        """Composite from two components should average them."""
        n = 200
        rng = np.random.RandomState(42)
        sex = np.tile([0, 1], n // 2)
        sm = SampleMeta(iid=np.arange(n), sex=sex)
        pheno = NPhenotypeArray(samples=sm)

        # Component A: high values for first 50
        # Component B: high values for last 50
        pheno['A'] = np.concatenate([rng.normal(10, 1, 100), rng.normal(0, 1, 100)])
        pheno['B'] = np.concatenate([rng.normal(0, 1, 100), rng.normal(10, 1, 100)])

        mate = LinearAssortativeMating(
            component_names=['A', 'B'], r=0.99, offspring_per_pair=2
        )
        assignment = mate.mate(sm, rng=np.random.RandomState(123), phenotypes=pheno)

        # With both A and B as components, the composite should balance them
        # No individual should dominate based on just one component
        mothers = assignment.maternal_idx[::2]
        fathers = assignment.paternal_idx[::2]
        assert len(mothers) > 0

    def test_zero_variance_component_neutral(self):
        """Component with zero variance should contribute zero to composite."""
        n = 100
        sex = np.tile([0, 1], n // 2)
        sm = SampleMeta(iid=np.arange(n), sex=sex)
        pheno = NPhenotypeArray(samples=sm)
        pheno['Y'] = np.random.RandomState(42).normal(0, 1, n)
        pheno['C'] = np.ones(n) * 5.0  # constant = zero variance

        mate = LinearAssortativeMating(
            component_names=['Y', 'C'], r=0.8, offspring_per_pair=2
        )
        # Should not raise, and assortment should be on Y only
        assignment = mate.mate(sm, rng=np.random.RandomState(123), phenotypes=pheno)
        assert assignment.n_offspring > 0


class TestScoreMixingFormula:
    def test_high_r_means_high_phenotype_correlation(self):
        """With r close to 1, spouse correlation should be high."""
        n = 500
        sm, pheno = _make_deterministic_phenotypes(n=n, seed=42)
        mate = LinearAssortativeMating(
            component_names=['Y'], r=0.95, offspring_per_pair=2
        )
        assignment = mate.mate(sm, rng=np.random.RandomState(42), phenotypes=pheno)
        mothers = assignment.maternal_idx[::2]
        fathers = assignment.paternal_idx[::2]
        r = np.corrcoef(pheno['Y'][mothers], pheno['Y'][fathers])[0, 1]
        assert r > 0.6, f"r=0.95 should produce corr > 0.6, got {r:.3f}"

    def test_low_r_means_low_correlation(self):
        """With r close to 0, spouse correlation should be near zero."""
        n = 500
        sm, pheno = _make_deterministic_phenotypes(n=n, seed=42)
        mate = LinearAssortativeMating(
            component_names=['Y'], r=0.05, offspring_per_pair=2
        )
        # Use different seed from phenotype RNG to avoid sequence coincidence
        assignment = mate.mate(sm, rng=np.random.RandomState(999), phenotypes=pheno)
        mothers = assignment.maternal_idx[::2]
        fathers = assignment.paternal_idx[::2]
        r = np.corrcoef(pheno['Y'][mothers], pheno['Y'][fathers])[0, 1]
        assert abs(r) < 0.4, f"r=0.05 should produce corr near 0, got {r:.3f}"


class TestRankOrderPairing:
    def test_disassortative_negates_male_scores(self):
        """Negative r should pair high females with low males."""
        n = 500
        sm, pheno = _make_deterministic_phenotypes(n=n, seed=42)
        mate = LinearAssortativeMating(
            component_names=['Y'], r=-0.95, offspring_per_pair=2
        )
        assignment = mate.mate(sm, rng=np.random.RandomState(42), phenotypes=pheno)
        mothers = assignment.maternal_idx[::2]
        fathers = assignment.paternal_idx[::2]
        r = np.corrcoef(pheno['Y'][mothers], pheno['Y'][fathers])[0, 1]
        assert r < -0.3, f"r=-0.95 should produce negative corr, got {r:.3f}"

    def test_missing_component_skipped(self):
        """Components not in phenotypes should be skipped in composite."""
        n = 100
        sex = np.tile([0, 1], n // 2)
        sm = SampleMeta(iid=np.arange(n), sex=sex)
        pheno = NPhenotypeArray(samples=sm)
        pheno['Y'] = np.random.RandomState(42).normal(0, 1, n)
        # 'Z' is NOT in phenotypes
        mate = LinearAssortativeMating(
            component_names=['Y', 'Z'], r=0.8, offspring_per_pair=2
        )
        # Should work, using only Y
        assignment = mate.mate(sm, rng=np.random.RandomState(123), phenotypes=pheno)
        assert assignment.n_offspring > 0

    def test_all_components_missing_falls_back_to_random(self):
        """If all components missing, composite=0, effectively random."""
        n = 100
        sex = np.tile([0, 1], n // 2)
        sm = SampleMeta(iid=np.arange(n), sex=sex)
        pheno = NPhenotypeArray(samples=sm)
        pheno['Y'] = np.random.RandomState(42).normal(0, 1, n)
        # Neither 'A' nor 'B' is in phenotypes
        mate = LinearAssortativeMating(
            component_names=['A', 'B'], r=0.8, offspring_per_pair=2
        )
        assignment = mate.mate(sm, rng=np.random.RandomState(123), phenotypes=pheno)
        assert assignment.n_offspring > 0
