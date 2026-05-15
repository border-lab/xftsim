"""Tests for LinearAssortativeMating."""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray, PhenotypeArray
from xftsim.mate import RandomMating, LinearAssortativeMating, MateAssignment
from xftsim.arch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.effect import AdditiveEffects
from xftsim.reproduce import RecombinationMap
from xftsim.sim import Simulation


def _make_pop(n=2000, m=50, seed=42):
    """Create a population with balanced sex and phenotypes."""
    rng = np.random.RandomState(seed)
    geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
    sex = np.tile([0, 1], (n + 1) // 2)[:n]
    samples = SampleMeta(iid=np.arange(n), sex=sex)
    variants = VariantMeta(vid=np.arange(m), af=np.full(m, 0.5))
    hap = DenseHaplotypeArray(genotypes=geno, samples=samples, variants=variants)
    # Compute phenotypes
    eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=123, standardized=False)
    pheno = PhenotypeArray(samples=samples)
    pheno._values['Y'] = (geno[:, :, 0] + geno[:, :, 1]).astype(np.float64) @ eff.effects
    pheno._values['Y'] += rng.normal(0, 0.5, size=n)
    return hap, pheno


class TestLinearAssortativeMating:
    def test_r_zero_behaves_like_random(self):
        """r=0 should produce a valid assignment (equivalent to random)."""
        hap, pheno = _make_pop(n=200)
        mate = LinearAssortativeMating(['Y'], r=0.0)
        rng = np.random.RandomState(99)
        assignment = mate.mate(hap.samples, rng=rng, phenotypes=pheno)
        assert isinstance(assignment, MateAssignment)
        assert assignment.n_offspring > 0

    def test_positive_r_produces_offspring(self):
        hap, pheno = _make_pop(n=200)
        mate = LinearAssortativeMating(['Y'], r=0.5)
        rng = np.random.RandomState(99)
        assignment = mate.mate(hap.samples, rng=rng, phenotypes=pheno)
        assert assignment.n_offspring == 100 * 2  # 100 pairs * 2 opp

    def test_negative_r_works(self):
        hap, pheno = _make_pop(n=200)
        mate = LinearAssortativeMating(['Y'], r=-0.3)
        rng = np.random.RandomState(99)
        assignment = mate.mate(hap.samples, rng=rng, phenotypes=pheno)
        assert assignment.n_offspring > 0

    def test_invalid_r_raises(self):
        with pytest.raises(ValueError, match="r must be"):
            LinearAssortativeMating(['Y'], r=1.0)
        with pytest.raises(ValueError, match="r must be"):
            LinearAssortativeMating(['Y'], r=-1.0)

    def test_positive_spouse_correlation(self):
        """With r=0.5, spouse phenotype correlation should be meaningfully positive."""
        hap, pheno = _make_pop(n=2000, seed=7)
        mate = LinearAssortativeMating(['Y'], r=0.5)
        rng = np.random.RandomState(42)
        assignment = mate.mate(hap.samples, rng=rng, phenotypes=pheno)
        # Compute spouse correlation on Y
        mother_y = pheno['Y'][assignment.maternal_idx[::2]]  # one per pair
        father_y = pheno['Y'][assignment.paternal_idx[::2]]
        corr = np.corrcoef(mother_y, father_y)[0, 1]
        assert corr > 0.1, f"Expected positive correlation, got {corr}"

    def test_negative_spouse_correlation(self):
        """With r=-0.5, spouse phenotype correlation should be negative."""
        hap, pheno = _make_pop(n=2000, seed=8)
        mate = LinearAssortativeMating(['Y'], r=-0.5)
        rng = np.random.RandomState(42)
        assignment = mate.mate(hap.samples, rng=rng, phenotypes=pheno)
        mother_y = pheno['Y'][assignment.maternal_idx[::2]]
        father_y = pheno['Y'][assignment.paternal_idx[::2]]
        corr = np.corrcoef(mother_y, father_y)[0, 1]
        assert corr < -0.1, f"Expected negative correlation, got {corr}"

    def test_sex_consistency(self):
        """Mothers should be female, fathers should be male."""
        hap, pheno = _make_pop(n=200)
        mate = LinearAssortativeMating(['Y'], r=0.5)
        assignment = mate.mate(hap.samples, rng=np.random.RandomState(0), phenotypes=pheno)
        assert np.all(hap.samples.sex[assignment.maternal_idx] == 0)
        assert np.all(hap.samples.sex[assignment.paternal_idx] == 1)

    def test_valid_indices(self):
        hap, pheno = _make_pop(n=200)
        mate = LinearAssortativeMating(['Y'], r=0.5)
        assignment = mate.mate(hap.samples, rng=np.random.RandomState(0), phenotypes=pheno)
        assert np.all(assignment.maternal_idx >= 0)
        assert np.all(assignment.maternal_idx < hap.n)
        assert np.all(assignment.paternal_idx >= 0)
        assert np.all(assignment.paternal_idx < hap.n)

    def test_offspring_count(self):
        hap, pheno = _make_pop(n=200)
        mate = LinearAssortativeMating(['Y'], r=0.3, offspring_per_pair=3)
        assignment = mate.mate(hap.samples, rng=np.random.RandomState(0), phenotypes=pheno)
        n_pairs = min(np.sum(hap.samples.sex == 0), np.sum(hap.samples.sex == 1))
        assert assignment.n_offspring == n_pairs * 3

    def test_deterministic_with_seed(self):
        hap, pheno = _make_pop(n=200)
        mate = LinearAssortativeMating(['Y'], r=0.5)
        a1 = mate.mate(hap.samples, rng=np.random.RandomState(42), phenotypes=pheno)
        a2 = mate.mate(hap.samples, rng=np.random.RandomState(42), phenotypes=pheno)
        np.testing.assert_array_equal(a1.maternal_idx, a2.maternal_idx)
        np.testing.assert_array_equal(a1.paternal_idx, a2.paternal_idx)

    def test_random_mating_accepts_phenotypes_kwarg(self):
        """RandomMating.mate() should accept phenotypes= without error."""
        hap, pheno = _make_pop(n=200)
        mate = RandomMating()
        assignment = mate.mate(hap.samples, rng=np.random.RandomState(0), phenotypes=pheno)
        assert isinstance(assignment, MateAssignment)

    def test_fallback_when_no_phenotypes(self):
        """With phenotypes=None, assortative should fall back to random."""
        hap, _ = _make_pop(n=200)
        mate = LinearAssortativeMating(['Y'], r=0.5)
        assignment = mate.mate(hap.samples, rng=np.random.RandomState(0), phenotypes=None)
        assert isinstance(assignment, MateAssignment)
        assert assignment.n_offspring > 0


class TestAssortativeEdgeCases:
    """Edge cases for LinearAssortativeMating."""

    def test_extreme_positive_r(self):
        """r=0.95 should work without error and produce strong positive correlation."""
        hap, pheno = _make_pop(n=2000, seed=10)
        mate = LinearAssortativeMating(['Y'], r=0.95)
        assignment = mate.mate(hap.samples, rng=np.random.RandomState(42), phenotypes=pheno)
        mother_y = pheno['Y'][assignment.maternal_idx[::2]]
        father_y = pheno['Y'][assignment.paternal_idx[::2]]
        corr = np.corrcoef(mother_y, father_y)[0, 1]
        assert corr > 0.3, f"Expected strong positive correlation with r=0.95, got {corr}"

    def test_extreme_negative_r(self):
        """r=-0.95 should work and produce strong negative correlation."""
        hap, pheno = _make_pop(n=2000, seed=11)
        mate = LinearAssortativeMating(['Y'], r=-0.95)
        assignment = mate.mate(hap.samples, rng=np.random.RandomState(42), phenotypes=pheno)
        mother_y = pheno['Y'][assignment.maternal_idx[::2]]
        father_y = pheno['Y'][assignment.paternal_idx[::2]]
        corr = np.corrcoef(mother_y, father_y)[0, 1]
        assert corr < -0.3, f"Expected strong negative correlation with r=-0.95, got {corr}"

    def test_multivariate_component_names(self):
        """Assortment on two phenotype components should work."""
        rng = np.random.RandomState(42)
        n, m = 500, 30
        geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
        sex = np.tile([0, 1], (n + 1) // 2)[:n]
        samples = SampleMeta(iid=np.arange(n), sex=sex)
        pheno = PhenotypeArray(samples=samples)
        pheno._values['Y1'] = rng.normal(0, 1, size=n)
        pheno._values['Y2'] = rng.normal(0, 1, size=n)
        mate = LinearAssortativeMating(['Y1', 'Y2'], r=0.5)
        assignment = mate.mate(samples, rng=np.random.RandomState(99), phenotypes=pheno)
        assert assignment.n_offspring == 250 * 2
        # Composite of Y1+Y2 should show some spouse correlation
        composite_m = (pheno['Y1'] + pheno['Y2'])[assignment.maternal_idx[::2]]
        composite_f = (pheno['Y1'] + pheno['Y2'])[assignment.paternal_idx[::2]]
        corr = np.corrcoef(composite_m, composite_f)[0, 1]
        assert corr > 0.05, f"Expected positive composite correlation, got {corr}"

    def test_missing_component_key_ignored(self):
        """Component names not in phenotypes should be silently skipped."""
        hap, pheno = _make_pop(n=200)
        mate = LinearAssortativeMating(['Y', 'MISSING_KEY'], r=0.5)
        # Should not raise — MISSING_KEY is simply skipped
        assignment = mate.mate(hap.samples, rng=np.random.RandomState(0), phenotypes=pheno)
        assert isinstance(assignment, MateAssignment)
        assert assignment.n_offspring > 0

    def test_all_components_missing_falls_back(self):
        """If no component_names match, should still produce offspring (noise-only scores)."""
        hap, pheno = _make_pop(n=200)
        mate = LinearAssortativeMating(['NONEXISTENT'], r=0.5)
        assignment = mate.mate(hap.samples, rng=np.random.RandomState(0), phenotypes=pheno)
        assert isinstance(assignment, MateAssignment)
        assert assignment.n_offspring > 0

    def test_zero_variance_phenotype(self):
        """All individuals with identical phenotype should still work."""
        rng = np.random.RandomState(42)
        n, m = 200, 10
        geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
        sex = np.tile([0, 1], (n + 1) // 2)[:n]
        samples = SampleMeta(iid=np.arange(n), sex=sex)
        pheno = PhenotypeArray(samples=samples)
        pheno._values['Y'] = np.ones(n)  # zero variance
        mate = LinearAssortativeMating(['Y'], r=0.5)
        assignment = mate.mate(samples, rng=np.random.RandomState(0), phenotypes=pheno)
        assert assignment.n_offspring > 0

    def test_repr(self):
        """LinearAssortativeMating repr should not crash."""
        mate = LinearAssortativeMating(['Y1', 'Y2'], r=0.3, offspring_per_pair=3)
        r = repr(mate)
        assert 'LinearAssortativeMating' in r
        assert 'r=0.3' in r
        assert 'offspring_per_pair=3' in r

    def test_boundary_r_values_raise(self):
        """r=1.0 and r=-1.0 should raise; r=0.99 and r=-0.99 should be fine."""
        with pytest.raises(ValueError):
            LinearAssortativeMating(['Y'], r=1.0)
        with pytest.raises(ValueError):
            LinearAssortativeMating(['Y'], r=-1.0)
        # These should NOT raise
        LinearAssortativeMating(['Y'], r=0.99)
        LinearAssortativeMating(['Y'], r=-0.99)

    def test_single_component_name(self):
        """Single component name (not a list) should be wrapped."""
        mate = LinearAssortativeMating('Y', r=0.5)
        assert mate.component_names == ['Y']

    def test_offspring_per_pair_validation(self):
        """offspring_per_pair < 1 should raise."""
        with pytest.raises(ValueError, match="offspring_per_pair"):
            LinearAssortativeMating(['Y'], r=0.5, offspring_per_pair=0)
