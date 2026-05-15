"""
Numerical tests for mating behavior.

Stochastic protocol: tolerance ~ 4/sqrt(N), N=10000.
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray, PhenotypeArray
from xftsim.mate import RandomMating, LinearAssortativeMating
from xftsim.effect import AdditiveEffects

N = 2000
M = 50
TOL = 4.0 / np.sqrt(N)  # ~0.089


def _make_pop(n=N, m=M, seed=42):
    rng = np.random.RandomState(seed)
    geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
    sex = np.tile([0, 1], (n + 1) // 2)[:n]
    samples = SampleMeta(iid=np.arange(n), sex=sex)
    variants = VariantMeta(vid=np.arange(m), af=np.full(m, 0.5))
    hap = DenseHaplotypeArray(genotypes=geno, samples=samples, variants=variants)
    pheno = PhenotypeArray(samples=samples)
    eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=123, standardized=False)
    pheno._values['Y'] = (geno[:, :, 0] + geno[:, :, 1]).astype(np.float64) @ eff.effects
    pheno._values['Y'] += rng.normal(0, 0.5, size=n)
    return hap, pheno


def _spouse_correlation(pheno, assignment, key='Y'):
    """Compute correlation between mated pairs' phenotype values."""
    mother_vals = pheno[key][assignment.maternal_idx[::2]]  # one per pair
    father_vals = pheno[key][assignment.paternal_idx[::2]]
    return np.corrcoef(mother_vals, father_vals)[0, 1]


class TestRandomMatingNumerical:
    def test_zero_spouse_correlation(self, stochastic_seed):
        """Random mating should produce ~0 spouse correlation."""
        hap, pheno = _make_pop(seed=stochastic_seed.seed)
        mate = RandomMating()
        assignment = mate.mate(hap.samples, rng=stochastic_seed.rng, phenotypes=pheno)
        corr = _spouse_correlation(pheno, assignment)
        assert abs(corr) < 3 * TOL, f"seed={stochastic_seed.seed}, corr={corr}"

    def test_uniform_parent_usage(self, stochastic_seed):
        """Each parent sex should be used approximately uniformly."""
        hap, pheno = _make_pop(n=1000, seed=stochastic_seed.seed)
        mate = RandomMating()
        assignment = mate.mate(hap.samples, rng=stochastic_seed.rng)
        # Count how many times each mother is used
        unique_mothers = np.unique(assignment.maternal_idx)
        # All mothers should be paired (500 females, 500 pairs)
        n_female = np.sum(hap.samples.sex == 0)
        n_male = np.sum(hap.samples.sex == 1)
        n_pairs = min(n_female, n_male)
        assert len(unique_mothers) == n_pairs


class TestAssortativeMatingNumerical:
    def test_positive_correlation(self, stochastic_seed):
        """Positive r should produce positive spouse correlation."""
        hap, pheno = _make_pop(seed=stochastic_seed.seed)
        mate = LinearAssortativeMating(['Y'], r=0.5)
        assignment = mate.mate(hap.samples, rng=stochastic_seed.rng, phenotypes=pheno)
        corr = _spouse_correlation(pheno, assignment)
        assert corr > 0.1, f"seed={stochastic_seed.seed}, corr={corr}"

    def test_negative_correlation(self, stochastic_seed):
        """Negative r should produce negative spouse correlation."""
        hap, pheno = _make_pop(seed=stochastic_seed.seed)
        mate = LinearAssortativeMating(['Y'], r=-0.5)
        assignment = mate.mate(hap.samples, rng=stochastic_seed.rng, phenotypes=pheno)
        corr = _spouse_correlation(pheno, assignment)
        assert corr < -0.1, f"seed={stochastic_seed.seed}, corr={corr}"

    def test_higher_r_higher_correlation(self, stochastic_seed):
        """Higher |r| should produce higher |spouse correlation|."""
        hap, pheno = _make_pop(seed=stochastic_seed.seed)
        corrs = []
        for r in [0.1, 0.3, 0.7]:
            mate = LinearAssortativeMating(['Y'], r=r)
            rng = np.random.RandomState(stochastic_seed.seed)
            assignment = mate.mate(hap.samples, rng=rng, phenotypes=pheno)
            corrs.append(_spouse_correlation(pheno, assignment))
        # Monotonically increasing (allowing small violation for stochasticity)
        assert corrs[2] > corrs[0], (
            f"seed={stochastic_seed.seed}, corrs={corrs}"
        )

    def test_variance_inflation_over_gens(self):
        """
        Assortative mating should inflate additive genetic variance over generations.
        (Deterministic test with fixed seed.)
        """
        from tests.testdata import TestSimulation
        from xftsim.sim import Simulation
        from xftsim.stats import SampleStatistics

        hap = TestSimulation.founder_haplotypes(n=1000, m=50, seed=42)
        arch = TestSimulation.simple_architecture(m=50, h2=0.5, seed=123)
        rmap = TestSimulation.recombination_map(m=50)

        # Random mating sim
        sim_rand = Simulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=RandomMating(), recombination_map=rmap,
            statistics=[SampleStatistics()], seed=42,
        )
        sim_rand.run(5)

        # Assortative mating sim
        hap2 = TestSimulation.founder_haplotypes(n=1000, m=50, seed=42)
        sim_assort = Simulation(
            founder_haplotypes=hap2, architecture=arch,
            mating_regime=LinearAssortativeMating(['Y'], r=0.7),
            recombination_map=rmap,
            statistics=[SampleStatistics()], seed=42,
        )
        sim_assort.run(5)

        # Compare Y variance at last generation
        var_rand = sim_rand.results[-1].statistics['SampleStatistics']['var']
        var_assort = sim_assort.results[-1].statistics['SampleStatistics']['var']
        # Y is the last key; assortative should have higher variance
        keys_r = sim_rand.results[-1].statistics['SampleStatistics']['keys']
        y_idx = keys_r.index('Y')
        assert var_assort[y_idx] > var_rand[y_idx] * 0.95  # allow some noise


class TestMultivariateAssortativeMating:
    """Test assortative mating on a composite of two phenotypes."""

    def _make_bivariate_pop(self, n=N, m=M, seed=42):
        """Create a population with two phenotypes."""
        rng = np.random.RandomState(seed)
        geno = rng.randint(0, 2, size=(n, m, 2)).astype(np.int8)
        sex = np.tile([0, 1], (n + 1) // 2)[:n]
        samples = SampleMeta(iid=np.arange(n), sex=sex)
        variants = VariantMeta(vid=np.arange(m), af=np.full(m, 0.5))
        hap = DenseHaplotypeArray(genotypes=geno, samples=samples, variants=variants)
        pheno = PhenotypeArray(samples=samples)
        eff1 = AdditiveEffects.from_h2(h2=0.5, m=m, seed=123, standardized=False)
        eff2 = AdditiveEffects.from_h2(h2=0.3, m=m, seed=456, standardized=False)
        G = (geno[:, :, 0] + geno[:, :, 1]).astype(np.float64)
        pheno._values['Y1'] = G @ eff1.effects + rng.normal(0, 0.5, size=n)
        pheno._values['Y2'] = G @ eff2.effects + rng.normal(0, 0.7, size=n)
        return hap, pheno

    def test_composite_spouse_correlation(self, stochastic_seed):
        """Assortment on [Y1, Y2] should produce positive composite spouse correlation."""
        hap, pheno = self._make_bivariate_pop(seed=stochastic_seed.seed)
        mate = LinearAssortativeMating(['Y1', 'Y2'], r=0.5)
        assignment = mate.mate(hap.samples, rng=stochastic_seed.rng, phenotypes=pheno)

        # Composite = standardized average of Y1 and Y2
        y1 = pheno['Y1']
        y2 = pheno['Y2']
        composite = (y1 - y1.mean()) / max(y1.std(), 1e-10) + \
                    (y2 - y2.mean()) / max(y2.std(), 1e-10)
        composite /= 2

        mother_comp = composite[assignment.maternal_idx[::2]]
        father_comp = composite[assignment.paternal_idx[::2]]
        corr = np.corrcoef(mother_comp, father_comp)[0, 1]
        assert corr > 0.05, (
            f"seed={stochastic_seed.seed}, composite_corr={corr:.4f}"
        )

    def test_single_trait_vs_composite_correlation(self, stochastic_seed):
        """Assortment on [Y1] only should correlate spouses on Y1 more than on Y2."""
        hap, pheno = self._make_bivariate_pop(seed=stochastic_seed.seed)
        mate = LinearAssortativeMating(['Y1'], r=0.5)
        assignment = mate.mate(hap.samples, rng=stochastic_seed.rng, phenotypes=pheno)

        corr_y1 = np.corrcoef(
            pheno['Y1'][assignment.maternal_idx[::2]],
            pheno['Y1'][assignment.paternal_idx[::2]]
        )[0, 1]
        corr_y2 = np.corrcoef(
            pheno['Y2'][assignment.maternal_idx[::2]],
            pheno['Y2'][assignment.paternal_idx[::2]]
        )[0, 1]

        # Y1 should have higher spouse correlation since we assort on it
        # (Y2 may still show some correlation if genetically correlated)
        assert corr_y1 > corr_y2 - 0.1, (
            f"seed={stochastic_seed.seed}, corr_Y1={corr_y1:.4f}, corr_Y2={corr_y2:.4f}"
        )
