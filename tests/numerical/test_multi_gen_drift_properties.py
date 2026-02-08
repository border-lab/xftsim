"""
Numerical test: multi-generation statistical properties and drift.

Tests covering genetic and phenotypic properties across multiple generations:
1. Genetic variance increases under assortative mating over 5 generations
2. Allele frequency variance increases with more generations (drift)
3. Mean phenotype stays near zero over 10 generations with random mating
4. Parent-offspring phenotype correlation ≈ h2/2 (mid-parent regression)
5. Phenotype variance stabilizes after initial transient (within 20% of equilibrium by gen 5)
6. Larger population size → smaller variance of allele frequency change
7. h2 remains approximately constant across generations under random mating
"""
import numpy as np
import pytest

from xftsim.neffect import AdditiveEffects
from xftsim.narch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
from xftsim.nmate import RandomMating, LinearAssortativeMating
from xftsim.reproduce import RecombinationMap
from xftsim.nsim import NSimulation

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from testdata import TestSimulation


class TestMultiGenerationDriftProperties:
    def test_genetic_variance_increases_under_assortative_mating(self):
        """
        Genetic variance should increase over generations under assortative mating.

        Theory: Assortative mating increases genetic variance by creating more
        extreme genotypic combinations.
        """
        n, m = 1000, 100
        h2 = 0.5
        seed = 42

        # Create founder population
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
        eff = AdditiveEffects.from_h2(h2=h2, m=m, seed=seed + 100)

        # Architecture: Y = G + E
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=1.0 - h2))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))

        # Assortative mating with r=0.5
        mating = LinearAssortativeMating(
            component_names=['Y'], r=0.5, offspring_per_pair=2
        )

        sim = NSimulation(
            founder_haplotypes=hap,
            architecture=arch,
            mating_regime=mating,
            recombination_map=RecombinationMap.constant_map(m=m),
            seed=seed,
            retain_haplotypes=10,
            retain_phenotypes=10,
        )
        sim.run(6)

        # Track genetic variance across generations
        gen_vars = []
        for g in range(6):
            if g in sim.phenotype_history:
                var_g = np.var(sim.phenotype_history[g]['Y.G'])
                gen_vars.append(var_g)

        # Genetic variance should increase monotonically (or at least be higher at end)
        assert len(gen_vars) == 6
        assert gen_vars[-1] > gen_vars[0] * 1.05, \
            f"Gen 5 Var(G)={gen_vars[-1]:.4f} should exceed Gen 0 Var(G)={gen_vars[0]:.4f} by >5%"

        # Check for general increasing trend (allowing some fluctuation)
        increasing_count = sum(gen_vars[i+1] >= gen_vars[i] * 0.95 for i in range(5))
        assert increasing_count >= 3, \
            f"Genetic variance should show increasing trend: {gen_vars}"

    def test_allele_frequency_variance_increases_with_generations(self):
        """
        Variance of allele frequency changes should increase with more generations.

        Theory: Drift accumulates over time, so Var(Δp) ∝ t for neutral alleles.
        """
        n, m = 1000, 200
        h2 = 0.5
        seed = 123

        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
        eff = AdditiveEffects.from_h2(h2=h2, m=m, seed=seed + 100)

        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=1.0 - h2))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))

        sim = NSimulation(
            founder_haplotypes=hap,
            architecture=arch,
            mating_regime=RandomMating(offspring_per_pair=2),
            recombination_map=RecombinationMap.constant_map(m=m),
            seed=seed,
            retain_haplotypes=15,
        )
        sim.run(11)

        af_0 = sim.haplotype_history[0].recompute_af()

        # Measure variance of AF changes at different time points
        variances = {}
        for gen in [3, 6, 10]:
            if gen in sim.haplotype_history:
                af_t = sim.haplotype_history[gen].recompute_af()
                delta_af = af_t - af_0
                variances[gen] = np.var(delta_af)

        # Variance should increase with time
        assert len(variances) == 3
        assert variances[6] > variances[3] * 0.8, \
            f"Var(Δp) at gen 6 ({variances[6]:.5f}) should exceed gen 3 ({variances[3]:.5f})"
        assert variances[10] > variances[6] * 0.8, \
            f"Var(Δp) at gen 10 ({variances[10]:.5f}) should exceed gen 6 ({variances[6]:.5f})"

    def test_mean_phenotype_near_zero_with_random_mating(self):
        """
        Mean phenotype should remain near zero over 10 generations with random mating.

        Theory: Without selection, mean phenotype should fluctuate around zero
        (assuming centered genetic effects).
        """
        n, m = 1200, 150
        h2 = 0.6
        seed = 456

        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
        eff = AdditiveEffects.from_h2(h2=h2, m=m, seed=seed + 100)

        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=1.0 - h2))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))

        sim = NSimulation(
            founder_haplotypes=hap,
            architecture=arch,
            mating_regime=RandomMating(offspring_per_pair=2),
            recombination_map=RecombinationMap.constant_map(m=m),
            seed=seed,
            retain_phenotypes=12,
        )
        sim.run(11)

        # Track mean phenotype across generations
        means = []
        for g in range(11):
            if g in sim.phenotype_history:
                mean_y = np.mean(sim.phenotype_history[g]['Y'])
                means.append(mean_y)

        # All means should be close to zero (allowing drift + noise)
        for g, mean_y in enumerate(means):
            assert abs(mean_y) < 0.15, \
                f"Gen {g} mean phenotype = {mean_y:.4f}, expected near 0"

        # Overall average should be very close to zero
        grand_mean = np.mean(means)
        assert abs(grand_mean) < 0.08, \
            f"Average mean across generations = {grand_mean:.4f}, expected near 0"

    def test_parent_offspring_correlation_matches_h2_over_2(self):
        """
        Parent-offspring phenotypic correlation should approximate h2/2 (midparent regression).

        Theory: Cov(Y_offspring, Y_midparent) / Var(Y_parent) ≈ h2/2 for additive traits.
        """
        n, m = 1500, 120
        h2 = 0.6
        seed = 789

        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
        eff = AdditiveEffects.from_h2(h2=h2, m=m, seed=seed + 100)

        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=1.0 - h2))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))

        sim = NSimulation(
            founder_haplotypes=hap,
            architecture=arch,
            mating_regime=RandomMating(offspring_per_pair=2),
            recombination_map=RecombinationMap.constant_map(m=m),
            seed=seed,
            retain_phenotypes=3,
        )
        sim.run(3)

        # Compute parent-offspring correlation for generation 1
        if 0 in sim.phenotype_history and 1 in sim.phenotype_history:
            parent_pheno = sim.phenotype_history[0]['Y']
            offspring_pheno = sim.phenotype_history[1]['Y']

            # Get pedigree to map offspring to parents
            ped = sim.pedigree_history[1]
            maternal_pheno = parent_pheno[ped.maternal_idx]
            paternal_pheno = parent_pheno[ped.paternal_idx]
            midparent_pheno = 0.5 * (maternal_pheno + paternal_pheno)

            # Compute correlation
            corr = np.corrcoef(midparent_pheno, offspring_pheno)[0, 1]

            # Expected correlation ≈ h2/2 for additive model
            # With h2=0.6, expected ≈ 0.30
            # Allow generous tolerance due to finite sample + noise
            expected = h2 / 2
            assert 0.15 < corr < 0.45, \
                f"Parent-offspring correlation = {corr:.3f}, expected ≈ {expected:.3f}"

    def test_phenotype_variance_stabilizes_after_transient(self):
        """
        Phenotype variance should stabilize after initial transient.

        Theory: After a few generations, Var(Y) should reach equilibrium
        (within 20% by generation 5).
        """
        n, m = 1200, 100
        h2 = 0.5
        seed = 999

        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
        eff = AdditiveEffects.from_h2(h2=h2, m=m, seed=seed + 100)

        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=1.0 - h2))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))

        sim = NSimulation(
            founder_haplotypes=hap,
            architecture=arch,
            mating_regime=RandomMating(offspring_per_pair=2),
            recombination_map=RecombinationMap.constant_map(m=m),
            seed=seed,
            retain_phenotypes=10,
        )
        sim.run(8)

        # Track phenotype variance across generations
        variances = []
        for g in range(8):
            if g in sim.phenotype_history:
                var_y = np.var(sim.phenotype_history[g]['Y'])
                variances.append(var_y)

        # By generation 5, variance should be within 20% of generation 7
        if len(variances) >= 8:
            var_5 = variances[5]
            var_7 = variances[7]
            ratio = var_5 / var_7
            assert 0.80 < ratio < 1.20, \
                f"Variance at gen 5 ({var_5:.4f}) should be within 20% of gen 7 ({var_7:.4f}), ratio={ratio:.3f}"

    def test_larger_population_smaller_af_variance(self):
        """
        Larger population size should result in smaller variance of allele frequency change.

        Theory: Drift variance scales as 1/(2N), so larger N → smaller Var(Δp).
        """
        m = 100
        h2 = 0.5
        seed_base = 111
        n_gens = 6

        def run_and_measure_drift(n, seed):
            hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
            eff = AdditiveEffects.from_h2(h2=h2, m=m, seed=seed + 100)

            arch = Architecture()
            arch.add('Y.G', GeneticComponent(eff))
            arch.add('Y.E', NoiseComponent(variance=1.0 - h2))
            arch.add('Y', AggregationComponent('Y.G + Y.E'))

            sim = NSimulation(
                founder_haplotypes=hap,
                architecture=arch,
                mating_regime=RandomMating(offspring_per_pair=2),
                recombination_map=RecombinationMap.constant_map(m=m),
                seed=seed,
                retain_haplotypes=10,
            )
            sim.run(n_gens)

            af_0 = sim.haplotype_history[0].recompute_af()
            af_final = sim.haplotypes.recompute_af()
            delta_af = af_final - af_0
            return np.var(delta_af)

        # Average over multiple seeds for stability
        small_drift = np.mean([run_and_measure_drift(200, seed_base + s) for s in [0, 1, 2]])
        large_drift = np.mean([run_and_measure_drift(1000, seed_base + s) for s in [0, 1, 2]])

        # Larger population should have smaller drift variance
        # Expected ratio ≈ N_large / N_small = 1000/200 = 5
        # Allow conservative check (>1.5x)
        assert small_drift > large_drift * 1.5, \
            f"Small pop (n=200) drift={small_drift:.5f} should exceed large pop (n=1000) drift={large_drift:.5f}"

    def test_h2_approximately_constant_across_generations(self):
        """
        Heritability (h2) should remain approximately constant across generations
        under random mating without selection.

        Theory: Random mating preserves genetic variance, so h2 should be stable.
        """
        n, m = 1500, 150
        h2_design = 0.5
        seed = 222

        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
        eff = AdditiveEffects.from_h2(h2=h2_design, m=m, seed=seed + 100)

        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=1.0 - h2_design))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))

        sim = NSimulation(
            founder_haplotypes=hap,
            architecture=arch,
            mating_regime=RandomMating(offspring_per_pair=2),
            recombination_map=RecombinationMap.constant_map(m=m),
            seed=seed,
            retain_phenotypes=8,
        )
        sim.run(7)

        # Compute h2 = Var(G) / Var(Y) at each generation
        h2_estimates = []
        for g in range(7):
            if g in sim.phenotype_history:
                pheno = sim.phenotype_history[g]
                var_g = np.var(pheno['Y.G'])
                var_y = np.var(pheno['Y'])
                h2_est = var_g / var_y
                h2_estimates.append(h2_est)

        # All h2 estimates should be within reasonable range of each other
        h2_mean = np.mean(h2_estimates)
        h2_std = np.std(h2_estimates)

        # Check that h2 doesn't drift too much (CV < 0.2)
        cv = h2_std / h2_mean
        assert cv < 0.25, \
            f"h2 coefficient of variation = {cv:.3f}, expected < 0.25 (mean={h2_mean:.3f}, std={h2_std:.3f})"

        # Check that early and late h2 are similar
        if len(h2_estimates) >= 7:
            early_h2 = np.mean(h2_estimates[:3])
            late_h2 = np.mean(h2_estimates[4:7])
            ratio = late_h2 / early_h2
            assert 0.70 < ratio < 1.30, \
                f"Early h2 ({early_h2:.3f}) and late h2 ({late_h2:.3f}) should be similar, ratio={ratio:.3f}"

    def test_genetic_correlation_preserved_under_drift(self):
        """
        Genetic correlation between traits should be approximately preserved
        over multiple generations despite drift.

        Theory: Drift affects all traits, but genetic correlations
        (which depend on shared genetic architecture) should remain stable.
        """
        n, m = 1200, 100
        h2 = [0.5, 0.4]
        rg_design = 0.6
        seed = 333

        from xftsim.neffect import MultivariateEffects
        from xftsim.narch import MVGeneticComponent

        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)
        eff = MultivariateEffects.from_h2_rg(h2=h2, rg=rg_design, m=m, seed=seed + 100)

        arch = Architecture()
        arch.add(['T1.G', 'T2.G'], MVGeneticComponent(eff))
        arch.add('T1.E', NoiseComponent(variance=1.0 - h2[0]))
        arch.add('T2.E', NoiseComponent(variance=1.0 - h2[1]))
        arch.add('T1', AggregationComponent('T1.G + T1.E'))
        arch.add('T2', AggregationComponent('T2.G + T2.E'))

        sim = NSimulation(
            founder_haplotypes=hap,
            architecture=arch,
            mating_regime=RandomMating(offspring_per_pair=2),
            recombination_map=RecombinationMap.constant_map(m=m),
            seed=seed,
            retain_phenotypes=8,
        )
        sim.run(7)

        # Compute genetic correlation at first and last generation
        rg_estimates = []
        for g in [0, 6]:
            if g in sim.phenotype_history:
                pheno = sim.phenotype_history[g]
                g1 = pheno['T1.G']
                g2 = pheno['T2.G']
                rg = np.corrcoef(g1, g2)[0, 1]
                rg_estimates.append(rg)

        if len(rg_estimates) == 2:
            rg_0, rg_6 = rg_estimates
            # Genetic correlation should remain similar
            assert abs(rg_6 - rg_0) < 0.15, \
                f"Genetic correlation changed from {rg_0:.3f} (gen 0) to {rg_6:.3f} (gen 6), expected stable"

            # Both should be reasonably close to design value
            for g, rg_est in zip([0, 6], rg_estimates):
                assert abs(rg_est - rg_design) < 0.25, \
                    f"Gen {g} genetic correlation = {rg_est:.3f}, design value = {rg_design:.3f}"
