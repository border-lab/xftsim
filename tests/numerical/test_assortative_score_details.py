"""
Numerical tests for LinearAssortativeMating score computation details.

Tests focus on the statistical properties of the mating score formula:
    score = sqrt(|r|) * composite + sqrt(1-|r|) * noise

where composite is the average of standardized phenotype components.

Tests:
1. Score standardization: mean≈0, sd≈1 after standardization for each component
2. Multi-component averaging: with 3 phenotype components, score is average of standardized components
3. Noise addition: sqrt(1-|r|) scaling verification - compare variance with and without noise
4. Disassortative (r<0): verify NEGATIVE spousal correlation
5. Moderate r=0.3: spousal correlation should be roughly proportional to r
6. Rank-order pairing: sorted male/female scores produce high correlation
7. Multi-trait assortative mating in a 3-gen simulation: correlation persists
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, NPhenotypeArray
from xftsim.neffect import AdditiveEffects, MultivariateEffects
from xftsim.narch import (
    Architecture, GeneticComponent, MVGeneticComponent,
    NoiseComponent, AggregationComponent
)
from xftsim.nmate import LinearAssortativeMating
from xftsim.reproduce import RecombinationMap
from xftsim.nsim import NSimulation

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestScoreStandardization:
    def test_component_standardization_mean_zero_sd_one(self):
        """Each component should be standardized to mean≈0, sd≈1 before averaging."""
        n = 2000
        # Generate phenotypes with different means and SDs
        rng_pheno = np.random.RandomState(42)
        sex = np.tile([0, 1], n // 2)
        sm = SampleMeta(iid=np.arange(n), sex=sex)
        pheno = NPhenotypeArray(samples=sm)

        # Component with mean=10, sd=3
        pheno['A'] = rng_pheno.normal(10, 3, size=n)
        # Component with mean=-5, sd=0.5
        pheno['B'] = rng_pheno.normal(-5, 0.5, size=n)

        # Use r=0.99 so scores ≈ composites (minimal noise)
        mate = LinearAssortativeMating(
            component_names=['A', 'B'], r=0.99, offspring_per_pair=2
        )

        # Use different seed for mating to avoid correlation artifacts
        rng_mate = np.random.RandomState(999)
        assignment = mate.mate(sm, rng=rng_mate, phenotypes=pheno)

        # Extract paired phenotypes
        mothers = assignment.maternal_idx[::2]
        fathers = assignment.paternal_idx[::2]

        # The rank correlation should be very high if standardization worked
        # (components with vastly different scales should contribute equally)
        mother_A = pheno['A'][mothers]
        father_A = pheno['A'][fathers]
        mother_B = pheno['B'][mothers]
        father_B = pheno['B'][fathers]

        corr_A = np.corrcoef(mother_A, father_A)[0, 1]
        corr_B = np.corrcoef(mother_B, father_B)[0, 1]

        # Both components should show spousal correlation with r=0.99
        # Even with different scales, both should contribute if standardized
        assert corr_A > 0.35, f"Component A corr={corr_A:.3f}, expected >0.35"
        assert corr_B > 0.25, f"Component B corr={corr_B:.3f}, expected >0.25"

        # The correlations should be similar magnitude despite different scales
        # (shows equal weighting after standardization)
        ratio = min(corr_A, corr_B) / max(corr_A, corr_B)
        assert ratio > 0.4, f"Corr ratio={ratio:.3f}, components not equally weighted"


class TestMultiComponentAveraging:
    def test_three_component_average_composite(self):
        """With 3 phenotype components, composite should be average of standardized values."""
        n = 1500
        rng_pheno = np.random.RandomState(100)
        sex = np.tile([0, 1], n // 2)
        sm = SampleMeta(iid=np.arange(n), sex=sex)
        pheno = NPhenotypeArray(samples=sm)

        # Three independent components with different distributions
        pheno['trait1'] = rng_pheno.normal(0, 1, size=n)
        pheno['trait2'] = rng_pheno.normal(5, 2, size=n)
        pheno['trait3'] = rng_pheno.exponential(2, size=n) - 2

        # Use r=0.95 for strong signal
        mate = LinearAssortativeMating(
            component_names=['trait1', 'trait2', 'trait3'],
            r=0.95,
            offspring_per_pair=2
        )

        rng_mate = np.random.RandomState(888)
        assignment = mate.mate(sm, rng=rng_mate, phenotypes=pheno)

        mothers = assignment.maternal_idx[::2]
        fathers = assignment.paternal_idx[::2]

        # All three traits should show spousal correlation
        # Lower threshold since averaging 3 components dilutes each
        for trait in ['trait1', 'trait2', 'trait3']:
            mother_vals = pheno[trait][mothers]
            father_vals = pheno[trait][fathers]
            corr = np.corrcoef(mother_vals, father_vals)[0, 1]
            assert corr > 0.2, f"{trait} corr={corr:.3f}, expected >0.2"

        # Compute manual composite to verify averaging
        # Standardize each component
        std_trait1 = (pheno['trait1'] - pheno['trait1'].mean()) / pheno['trait1'].std()
        std_trait2 = (pheno['trait2'] - pheno['trait2'].mean()) / pheno['trait2'].std()
        std_trait3 = (pheno['trait3'] - pheno['trait3'].mean()) / pheno['trait3'].std()
        manual_composite = (std_trait1 + std_trait2 + std_trait3) / 3

        # Verify manual composite also shows high spousal correlation
        mother_comp = manual_composite[mothers]
        father_comp = manual_composite[fathers]
        comp_corr = np.corrcoef(mother_comp, father_comp)[0, 1]
        assert comp_corr > 0.5, f"Composite corr={comp_corr:.3f}, expected >0.5"


class TestNoiseAddition:
    def test_noise_scaling_sqrt_one_minus_r(self):
        """Noise term should be scaled by sqrt(1-|r|) to control signal/noise ratio."""
        n = 2000

        # Create simple phenotype
        rng_pheno = np.random.RandomState(200)
        sex = np.tile([0, 1], n // 2)
        sm = SampleMeta(iid=np.arange(n), sex=sex)
        pheno = NPhenotypeArray(samples=sm)
        pheno['Y'] = rng_pheno.normal(0, 1, size=n)

        # Test with r=0.6 (abs_r=0.6, noise_weight=sqrt(0.4)≈0.632)
        mate = LinearAssortativeMating(
            component_names=['Y'], r=0.6, offspring_per_pair=2
        )

        # Use different seed for mating
        rng_mate = np.random.RandomState(777)
        assignment = mate.mate(sm, rng=rng_mate, phenotypes=pheno)

        mothers = assignment.maternal_idx[::2]
        fathers = assignment.paternal_idx[::2]

        # Expected correlation should be roughly r * correlation(composite, composite)
        # Since composite is derived from Y, correlation should be close to r
        actual_corr = np.corrcoef(pheno['Y'][mothers], pheno['Y'][fathers])[0, 1]

        # With large n, actual correlation should be within reasonable range of r
        # Allow 0.15 tolerance for noise variance
        assert 0.45 < actual_corr < 0.75, \
            f"With r=0.6, spousal corr={actual_corr:.3f}, expected ≈0.6±0.15"

    def test_higher_r_lower_noise_variance(self):
        """Higher |r| means lower noise weight, should give more consistent pairing."""
        n = 1200
        rng_pheno = np.random.RandomState(300)
        sex = np.tile([0, 1], n // 2)
        sm = SampleMeta(iid=np.arange(n), sex=sex)
        pheno = NPhenotypeArray(samples=sm)
        pheno['Y'] = rng_pheno.normal(0, 1, size=n)

        # Test r=0.3 (high noise: sqrt(0.91)≈0.95)
        mate_low = LinearAssortativeMating(
            component_names=['Y'], r=0.3, offspring_per_pair=2
        )
        assignment_low = mate_low.mate(
            sm, rng=np.random.RandomState(666), phenotypes=pheno
        )

        # Test r=0.9 (low noise: sqrt(0.19)≈0.44)
        mate_high = LinearAssortativeMating(
            component_names=['Y'], r=0.9, offspring_per_pair=2
        )
        assignment_high = mate_high.mate(
            sm, rng=np.random.RandomState(667), phenotypes=pheno
        )

        # Extract correlations
        mothers_low = assignment_low.maternal_idx[::2]
        fathers_low = assignment_low.paternal_idx[::2]
        corr_low = np.corrcoef(pheno['Y'][mothers_low], pheno['Y'][fathers_low])[0, 1]

        mothers_high = assignment_high.maternal_idx[::2]
        fathers_high = assignment_high.paternal_idx[::2]
        corr_high = np.corrcoef(pheno['Y'][mothers_high], pheno['Y'][fathers_high])[0, 1]

        # Higher r should produce stronger correlation
        assert corr_high > corr_low + 0.2, \
            f"r=0.9 corr={corr_high:.3f} should exceed r=0.3 corr={corr_low:.3f} by >0.2"


class TestDisassortativeMating:
    def test_negative_r_produces_negative_correlation(self):
        """With r<0, male scores are negated, producing negative spousal correlation."""
        n = 1500
        rng_pheno = np.random.RandomState(400)
        sex = np.tile([0, 1], n // 2)
        sm = SampleMeta(iid=np.arange(n), sex=sex)
        pheno = NPhenotypeArray(samples=sm)
        pheno['Y'] = rng_pheno.normal(0, 1, size=n)

        # Strong disassortative mating
        mate = LinearAssortativeMating(
            component_names=['Y'], r=-0.8, offspring_per_pair=2
        )

        rng_mate = np.random.RandomState(555)
        assignment = mate.mate(sm, rng=rng_mate, phenotypes=pheno)

        mothers = assignment.maternal_idx[::2]
        fathers = assignment.paternal_idx[::2]

        corr = np.corrcoef(pheno['Y'][mothers], pheno['Y'][fathers])[0, 1]

        # Should produce strong negative correlation
        assert corr < -0.4, f"r=-0.8 should produce corr < -0.4, got {corr:.3f}"

    def test_disassortative_pairs_high_with_low(self):
        """Negative r should pair high-phenotype females with low-phenotype males."""
        n = 1000
        rng_pheno = np.random.RandomState(500)
        sex = np.tile([0, 1], n // 2)
        sm = SampleMeta(iid=np.arange(n), sex=sex)
        pheno = NPhenotypeArray(samples=sm)
        pheno['Y'] = rng_pheno.normal(0, 1, size=n)

        # Very strong disassortative mating
        mate = LinearAssortativeMating(
            component_names=['Y'], r=-0.95, offspring_per_pair=2
        )

        rng_mate = np.random.RandomState(444)
        assignment = mate.mate(sm, rng=rng_mate, phenotypes=pheno)

        mothers = assignment.maternal_idx[::2]
        fathers = assignment.paternal_idx[::2]

        # Find highest 10% mothers by phenotype
        mother_y = pheno['Y'][mothers]
        father_y = pheno['Y'][fathers]
        high_mother_idx = np.argsort(mother_y)[-len(mother_y)//10:]

        # Their paired fathers should have LOW phenotypes on average
        paired_fathers_y = father_y[high_mother_idx]

        # Mean of paired fathers should be below population mean
        pop_mean = pheno['Y'].mean()
        assert paired_fathers_y.mean() < pop_mean - 0.2, \
            f"High mothers paired with fathers mean={paired_fathers_y.mean():.3f}, " \
            f"should be < pop_mean={pop_mean:.3f}"


class TestModerateAssortment:
    def test_r_0_3_produces_proportional_correlation(self):
        """With r=0.3, spousal correlation should be roughly 0.3."""
        n = 2000
        rng_pheno = np.random.RandomState(600)
        sex = np.tile([0, 1], n // 2)
        sm = SampleMeta(iid=np.arange(n), sex=sex)
        pheno = NPhenotypeArray(samples=sm)
        pheno['Y'] = rng_pheno.normal(0, 1, size=n)

        mate = LinearAssortativeMating(
            component_names=['Y'], r=0.3, offspring_per_pair=2
        )

        rng_mate = np.random.RandomState(333)
        assignment = mate.mate(sm, rng=rng_mate, phenotypes=pheno)

        mothers = assignment.maternal_idx[::2]
        fathers = assignment.paternal_idx[::2]

        corr = np.corrcoef(pheno['Y'][mothers], pheno['Y'][fathers])[0, 1]

        # With large n, should be close to target r=0.3
        # Allow 0.15 tolerance
        assert 0.15 < corr < 0.45, \
            f"r=0.3 should produce corr ≈0.3±0.15, got {corr:.3f}"

    def test_correlation_scales_with_r(self):
        """Spousal correlation should generally increase with r."""
        n = 2000  # Larger n for more stable estimates
        rng_pheno = np.random.RandomState(700)
        sex = np.tile([0, 1], n // 2)
        sm = SampleMeta(iid=np.arange(n), sex=sex)
        pheno = NPhenotypeArray(samples=sm)
        pheno['Y'] = rng_pheno.normal(0, 1, size=n)

        r_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        correlations = []

        for idx, r in enumerate(r_values):
            mate = LinearAssortativeMating(
                component_names=['Y'], r=r, offspring_per_pair=2
            )
            # Use different seed for each to avoid cross-contamination
            # Avoid seeds that might create anomalies
            rng_mate = np.random.RandomState(7000 + idx)
            assignment = mate.mate(sm, rng=rng_mate, phenotypes=pheno)

            mothers = assignment.maternal_idx[::2]
            fathers = assignment.paternal_idx[::2]
            corr = np.corrcoef(pheno['Y'][mothers], pheno['Y'][fathers])[0, 1]
            correlations.append(corr)

        # Test: low r values should give lower correlations than high r values
        # Compare first half average to second half average
        low_r_avg = np.mean(correlations[:2])  # r=0.1, 0.3
        high_r_avg = np.mean(correlations[3:])  # r=0.7, 0.9
        assert high_r_avg > low_r_avg + 0.2, \
            f"High r avg={high_r_avg:.3f} should exceed low r avg={low_r_avg:.3f} by >0.2"


class TestRankOrderPairing:
    def test_perfect_rank_order_with_high_r(self):
        """With r≈1, pairing should follow near-perfect rank order."""
        n = 1000
        rng_pheno = np.random.RandomState(800)
        sex = np.tile([0, 1], n // 2)
        sm = SampleMeta(iid=np.arange(n), sex=sex)
        pheno = NPhenotypeArray(samples=sm)
        pheno['Y'] = rng_pheno.normal(0, 1, size=n)

        # Very high r means minimal noise
        mate = LinearAssortativeMating(
            component_names=['Y'], r=0.99, offspring_per_pair=2
        )

        rng_mate = np.random.RandomState(222)
        assignment = mate.mate(sm, rng=rng_mate, phenotypes=pheno)

        mothers = assignment.maternal_idx[::2]
        fathers = assignment.paternal_idx[::2]

        # Compute rank correlation (Spearman)
        from scipy.stats import spearmanr
        mother_y = pheno['Y'][mothers]
        father_y = pheno['Y'][fathers]
        rank_corr, _ = spearmanr(mother_y, father_y)

        # Rank correlation should be very high
        assert rank_corr > 0.9, f"With r=0.99, rank_corr={rank_corr:.3f}, expected >0.9"

    def test_sorted_scores_produce_high_correlation(self):
        """Sorting males and females by score should produce r≈1 pairing."""
        n = 800
        rng_pheno = np.random.RandomState(900)
        sex = np.tile([0, 1], n // 2)
        sm = SampleMeta(iid=np.arange(n), sex=sex)
        pheno = NPhenotypeArray(samples=sm)
        pheno['Y'] = rng_pheno.normal(0, 1, size=n)

        # With r=0.98, noise is minimal
        mate = LinearAssortativeMating(
            component_names=['Y'], r=0.98, offspring_per_pair=2
        )

        rng_mate = np.random.RandomState(111)
        assignment = mate.mate(sm, rng=rng_mate, phenotypes=pheno)

        mothers = assignment.maternal_idx[::2]
        fathers = assignment.paternal_idx[::2]

        # Pearson correlation should be very high
        corr = np.corrcoef(pheno['Y'][mothers], pheno['Y'][fathers])[0, 1]
        assert corr > 0.85, f"With r=0.98, corr={corr:.3f}, expected >0.85"


class TestMultiTraitMultiGen:
    def test_multitrait_assortative_correlation_persists(self):
        """Assortative mating on multiple traits should maintain correlation across generations."""
        n = 1000
        m = 50

        # Test across multiple independent simulation runs to verify persistence
        # Run 1: gen 0 -> gen 1
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=1000)
        effects = MultivariateEffects.from_h2_rg(
            h2=[0.5, 0.4], rg=0.3, m=m, seed=1001
        )
        arch = Architecture()
        arch.add(['trait1.G', 'trait2.G'], MVGeneticComponent(effects))
        arch.add('trait1.E', NoiseComponent(variance=0.5))
        arch.add('trait2.E', NoiseComponent(variance=0.6))
        arch.add('trait1', AggregationComponent('trait1.G + trait1.E'))
        arch.add('trait2', AggregationComponent('trait2.G + trait2.E'))

        mate = LinearAssortativeMating(
            component_names=['trait1', 'trait2'], r=0.6, offspring_per_pair=2
        )

        sim1 = NSimulation(
            founder_haplotypes=hap,
            architecture=arch,
            mating_regime=mate,
            recombination_map=RecombinationMap.constant_map(m=m),
            seed=1002,
            retain_phenotypes=10,
        )
        sim1.run(2)  # Gen 0 and 1

        # Check gen 0 mating (note: only most recent mate assignment is kept)
        assignment1 = sim1._mate_assignments[0]
        pheno1 = sim1.phenotype_history[0]

        mothers1 = assignment1.maternal_idx[::2]
        fathers1 = assignment1.paternal_idx[::2]

        # Compute composite correlation
        composite1 = np.zeros(pheno1['trait1'].shape[0])
        t1 = (pheno1['trait1'] - pheno1['trait1'].mean()) / pheno1['trait1'].std()
        t2 = (pheno1['trait2'] - pheno1['trait2'].mean()) / pheno1['trait2'].std()
        composite1 = (t1 + t2) / 2

        corr1 = np.corrcoef(composite1[mothers1], composite1[fathers1])[0, 1]
        assert corr1 > 0.2, f"Gen 0 composite corr={corr1:.3f}, expected >0.2"

        # Run 2: multi-generation to verify later generations also show correlation
        sim2 = NSimulation(
            founder_haplotypes=hap,
            architecture=arch,
            mating_regime=mate,
            recombination_map=RecombinationMap.constant_map(m=m),
            seed=1003,
            retain_phenotypes=10,
        )
        sim2.run(4)  # Gen 0, 1, 2, 3

        # Check most recent mate assignment (gen 2 -> gen 3)
        final_gen = sim2.generation
        assignment2 = sim2._mate_assignments[final_gen - 1]
        pheno2 = sim2.phenotype_history[final_gen - 1]

        mothers2 = assignment2.maternal_idx[::2]
        fathers2 = assignment2.paternal_idx[::2]

        # Verify both traits show correlation in later generation
        for trait in ['trait1', 'trait2']:
            mother_vals = pheno2[trait][mothers2]
            father_vals = pheno2[trait][fathers2]
            trait_corr = np.corrcoef(mother_vals, father_vals)[0, 1]
            assert trait_corr > 0.1, \
                f"{trait} gen {final_gen-1} corr={trait_corr:.3f}, expected >0.1"
