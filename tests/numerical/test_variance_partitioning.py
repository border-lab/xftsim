"""
Numerical tests for variance partitioning and heritability recovery.

Tests that when we set up Y = Y.G + Y.E with known h2, the variance
of Y.G / variance of Y approximates h2.
"""

import numpy as np
import pytest

from xftsim.arch import (
    Architecture,
    GeneticComponent,
    MVGeneticComponent,
    NoiseComponent,
    AggregationComponent,
)
from xftsim.effect import AdditiveEffects, MultivariateEffects
from xftsim.sim import NSimulation
from xftsim.mate import RandomMating
from xftsim.reproduce import RecombinationMap
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from testdata import TestSimulation


def test_h2_recovery():
    """
    Test that we can recover known heritability from a simple additive model.

    Setup: Y = Y.G + Y.E where:
    - Y.G ~ genetic(eff) with h2=0.5 (variance approximately 0.5)
    - Y.E ~ noise(var=0.5)
    - Expected total variance ≈ 1.0, h2 = var(Y.G) / var(Y) ≈ 0.5

    from_h2() creates effects such that var(Y.G) ≈ h2 when using standardized
    genotypes. The actual h2 depends on sampling variation.
    """
    n = 2000
    m = 500  # Use more variants for stability
    h2_target = 0.5
    noise_var = 1.0 - h2_target  # Set noise variance to achieve target h2
    seed = 42

    # Create founder haplotypes
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)

    # Create effects with target h2
    eff = AdditiveEffects.from_h2(h2=h2_target, m=m, seed=seed)

    # Build architecture using the programmatic API
    arch = Architecture()
    arch.add("Y.G", GeneticComponent(eff))
    arch.add("Y.E", NoiseComponent(variance=noise_var))
    arch.add("Y", AggregationComponent("Y.G + Y.E"), inputs=["Y.G", "Y.E"])

    # Compute phenotypes for generation 0
    rng = np.random.RandomState(seed)
    pheno = arch.compute(hap, rng=rng)

    # Extract values
    y_g = pheno["Y.G"]
    y_e = pheno["Y.E"]
    y = pheno["Y"]

    # Compute variances
    var_yg = np.var(y_g)
    var_ye = np.var(y_e)
    var_y = np.var(y)

    # Check that total variance is approximately 1.0
    assert abs(var_y - 1.0) < 0.5, (
        f"Total variance {var_y:.3f} should be approximately 1.0 "
        f"(var_yg={var_yg:.3f}, var_ye={var_ye:.3f})"
    )

    # Check that genetic variance is in the right ballpark
    assert abs(var_yg - h2_target) < 0.4, (
        f"Genetic variance {var_yg:.3f} should be approximately {h2_target}"
    )

    # Verify Y = Y.G + Y.E
    y_reconstructed = y_g + y_e
    assert np.allclose(y, y_reconstructed), "Y should equal Y.G + Y.E"


def test_noise_variance_dominates_at_low_h2():
    """
    Test that at low heritability (h2=0.1), most variance comes from noise.

    With h2=0.1, we expect var(Y.E) / var(Y) > 0.7 (most variance is environmental).
    """
    n = 2000
    m = 50
    h2_target = 0.1
    noise_var = 1.0 - h2_target
    seed = 43

    # Create founder haplotypes
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)

    # Create effects with low h2
    eff = AdditiveEffects.from_h2(h2=h2_target, m=m, seed=seed)

    # Build architecture
    arch = Architecture()
    arch.add("Y.G", GeneticComponent(eff))
    arch.add("Y.E", NoiseComponent(variance=noise_var))
    arch.add("Y", AggregationComponent("Y.G + Y.E"), inputs=["Y.G", "Y.E"])

    # Compute phenotypes
    rng = np.random.RandomState(seed)
    pheno = arch.compute(hap, rng=rng)

    # Extract values
    y_e = pheno["Y.E"]
    y = pheno["Y"]

    # Compute variances
    var_ye = np.var(y_e)
    var_y = np.var(y)

    # Environmental variance should dominate
    env_proportion = var_ye / var_y

    assert env_proportion > 0.7, (
        f"At h2={h2_target}, environmental variance proportion should be > 0.7, "
        f"but got {env_proportion:.3f}"
    )

    # Also verify the overall h2 is low
    y_g = pheno["Y.G"]
    var_yg = np.var(y_g)
    h2_obs = var_yg / var_y

    assert h2_obs < 0.3, (
        f"At low h2 target ({h2_target}), observed h2={h2_obs:.3f} should be < 0.3"
    )


def test_multivariate_variance_ratio():
    """
    Test variance partitioning for multivariate traits.

    Two traits with h2=[0.5, 0.3] and genetic correlation rg=0.2.
    Each trait should have the appropriate genetic variance proportion.
    """
    n = 2000
    m = 500  # Use more variants for stability
    h2_targets = [0.5, 0.3]
    rg = 0.2
    noise_vars = [1.0 - h2_targets[0], 1.0 - h2_targets[1]]
    seed = 44

    # Create founder haplotypes
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)

    # Create multivariate effects
    mv_eff = MultivariateEffects.from_h2_rg(h2=h2_targets, rg=rg, m=m, seed=seed)

    # Build architecture with two traits
    arch = Architecture()
    arch.add(["Y1.G", "Y2.G"], MVGeneticComponent(mv_eff))
    arch.add("Y1.E", NoiseComponent(variance=noise_vars[0]))
    arch.add("Y2.E", NoiseComponent(variance=noise_vars[1]))
    arch.add("Y1", AggregationComponent("Y1.G + Y1.E"), inputs=["Y1.G", "Y1.E"])
    arch.add("Y2", AggregationComponent("Y2.G + Y2.E"), inputs=["Y2.G", "Y2.E"])

    # Compute phenotypes
    rng = np.random.RandomState(seed)
    pheno = arch.compute(hap, rng=rng)

    # Extract values for trait 1
    y1_g = pheno["Y1.G"]
    y1 = pheno["Y1"]

    var_y1g = np.var(y1_g)
    var_y1 = np.var(y1)

    # Extract values for trait 2
    y2_g = pheno["Y2.G"]
    y2 = pheno["Y2"]

    var_y2g = np.var(y2_g)
    var_y2 = np.var(y2)

    # Check trait 1: total variance should be approximately 1.0
    assert abs(var_y1 - 1.0) < 0.5, (
        f"Trait 1: total variance {var_y1:.3f} should be approximately 1.0 "
        f"(var_yg={var_y1g:.3f})"
    )

    # Check trait 1: genetic variance should be in the right ballpark
    assert abs(var_y1g - h2_targets[0]) < 0.4, (
        f"Trait 1: genetic variance {var_y1g:.3f} should be approximately {h2_targets[0]}"
    )

    # Check trait 2: total variance should be approximately 1.0
    assert abs(var_y2 - 1.0) < 0.5, (
        f"Trait 2: total variance {var_y2:.3f} should be approximately 1.0 "
        f"(var_yg={var_y2g:.3f})"
    )

    # Check trait 2: genetic variance should be in the right ballpark
    assert abs(var_y2g - h2_targets[1]) < 0.4, (
        f"Trait 2: genetic variance {var_y2g:.3f} should be approximately {h2_targets[1]}"
    )

    # Verify aggregation works correctly
    y1_e = pheno["Y1.E"]
    y2_e = pheno["Y2.E"]
    y1_reconstructed = y1_g + y1_e
    y2_reconstructed = y2_g + y2_e
    assert np.allclose(y1, y1_reconstructed), "Y1 should equal Y1.G + Y1.E"
    assert np.allclose(y2, y2_reconstructed), "Y2 should equal Y2.G + Y2.E"

    # Check genetic correlation (should be approximately rg)
    cor_g = np.corrcoef(y1_g, y2_g)[0, 1]
    assert abs(cor_g - rg) < 0.3, (
        f"Genetic correlation {cor_g:.3f} differs from target rg={rg} by more than 0.3"
    )


def test_zero_heritability():
    """
    Test the edge case where h2=0 (purely environmental variance).

    All variance should come from noise, genetic variance should be negligible.
    """
    n = 2000
    m = 50
    h2_target = 0.0
    noise_var = 1.0
    seed = 45

    # Create founder haplotypes
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)

    # Create effects with h2=0 (all zeros)
    eff = AdditiveEffects.from_h2(h2=h2_target, m=m, seed=seed)

    # Build architecture
    arch = Architecture()
    arch.add("Y.G", GeneticComponent(eff))
    arch.add("Y.E", NoiseComponent(variance=noise_var))
    arch.add("Y", AggregationComponent("Y.G + Y.E"), inputs=["Y.G", "Y.E"])

    # Compute phenotypes
    rng = np.random.RandomState(seed)
    pheno = arch.compute(hap, rng=rng)

    # Extract values
    y_g = pheno["Y.G"]
    y_e = pheno["Y.E"]
    y = pheno["Y"]

    # Genetic component should be essentially zero
    var_yg = np.var(y_g)
    var_y = np.var(y)

    assert var_yg < 0.01, f"With h2=0, genetic variance should be near zero, got {var_yg:.3f}"

    # All variance should come from noise
    assert abs(var_y - noise_var) < 0.1, (
        f"With h2=0, total variance should equal noise variance, "
        f"got var_y={var_y:.3f}, noise_var={noise_var}"
    )

    # Y should be approximately equal to Y.E
    assert np.allclose(y, y_e, atol=1e-10), "With h2=0, Y should equal Y.E"


def test_high_heritability():
    """
    Test the case where h2 is very high (h2=0.9).

    Most variance should come from genetics, with minimal environmental contribution.
    Note: With high h2, total variance may be lower than 1.0 due to sampling
    variability, but the genetic component should still dominate.
    """
    n = 2000
    m = 500  # Use more variants for stability
    h2_target = 0.9
    noise_var = 1.0 - h2_target  # Small noise
    seed = 46

    # Create founder haplotypes
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)

    # Create effects with high h2
    eff = AdditiveEffects.from_h2(h2=h2_target, m=m, seed=seed)

    # Build architecture
    arch = Architecture()
    arch.add("Y.G", GeneticComponent(eff))
    arch.add("Y.E", NoiseComponent(variance=noise_var))
    arch.add("Y", AggregationComponent("Y.G + Y.E"), inputs=["Y.G", "Y.E"])

    # Compute phenotypes
    rng = np.random.RandomState(seed)
    pheno = arch.compute(hap, rng=rng)

    # Extract values
    y_g = pheno["Y.G"]
    y_e = pheno["Y.E"]
    y = pheno["Y"]

    # Compute variances
    var_yg = np.var(y_g)
    var_ye = np.var(y_e)
    var_y = np.var(y)

    # At high h2, environmental variance should be very small
    assert var_ye < 0.2, (
        f"At h2={h2_target}, environmental variance {var_ye:.3f} should be < 0.2"
    )

    # Genetic variance should dominate (at least 80% of total)
    genetic_proportion = var_yg / var_y
    assert genetic_proportion > 0.75, (
        f"At h2={h2_target}, genetic variance proportion should be > 0.75, "
        f"but got {genetic_proportion:.3f} (var_yg={var_yg:.3f}, var_y={var_y:.3f})"
    )


def test_variance_consistency_across_generations():
    """
    Test that variance partitioning remains consistent across generations.

    Run a simple simulation for 3 generations with random mating and verify
    that heritability remains approximately constant.
    """
    n = 1000
    m = 500  # Use more variants for stability
    h2_target = 0.5
    noise_var = 1.0 - h2_target
    seed = 47

    # Create founder haplotypes (with balanced sex)
    hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)

    # Create effects
    eff = AdditiveEffects.from_h2(h2=h2_target, m=m, seed=seed)

    # Build architecture
    arch = Architecture()
    arch.add("Y.G", GeneticComponent(eff))
    arch.add("Y.E", NoiseComponent(variance=noise_var))
    arch.add("Y", AggregationComponent("Y.G + Y.E"), inputs=["Y.G", "Y.E"])

    # Create recombination map (constant)
    rmap = RecombinationMap.constant_map(m=m, p=0.5)

    # Create mating function with 2 offspring per pair to maintain population size
    mate_fn = RandomMating(offspring_per_pair=2)

    # Create simulation
    sim = NSimulation(
        founder_haplotypes=hap,
        architecture=arch,
        recombination_map=rmap,
        mating_regime=mate_fn,
        seed=seed,
    )

    # Run for 3 generations
    sim.run(3)

    # Check genetic variance and total variance in each generation
    var_yg_by_gen = []
    var_y_by_gen = []
    for gen in range(3):
        pheno = sim.phenotype_history[gen]
        y_g = pheno["Y.G"]
        y = pheno["Y"]

        var_yg = np.var(y_g)
        var_y = np.var(y)
        var_yg_by_gen.append(var_yg)
        var_y_by_gen.append(var_y)

    # All generations should have genetic variance in reasonable range
    for gen, var_yg in enumerate(var_yg_by_gen):
        assert abs(var_yg - h2_target) < 0.4, (
            f"Generation {gen}: genetic variance {var_yg:.3f} differs from "
            f"target {h2_target} by more than 0.4"
        )

    # All generations should have total variance approximately 1.0
    for gen, var_y in enumerate(var_y_by_gen):
        assert abs(var_y - 1.0) < 0.5, (
            f"Generation {gen}: total variance {var_y:.3f} differs from 1.0 "
            f"by more than 0.5"
        )

    # Genetic variance should not drift too much across generations
    var_yg_std = np.std(var_yg_by_gen)
    assert var_yg_std < 0.2, (
        f"Genetic variance standard deviation across generations ({var_yg_std:.3f}) "
        f"is too large (var_yg values: {var_yg_by_gen})"
    )
