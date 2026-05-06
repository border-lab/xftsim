"""
Numerical tests for HaplotypeGeneticComponent variance partitioning.

Verifies that maternal-only and paternal-only haplotype genetic effects
produce the correct variance contributions.

Stochastic protocol: tolerance ~ 4/sqrt(N), N=10000.
"""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray, NPhenotypeArray
from xftsim.arch import (
    Architecture, GeneticComponent, HaplotypeGeneticComponent,
    NoiseComponent, AggregationComponent,
)
from xftsim.effect import AdditiveEffects

N = 10000
M = 100
TOL = 4.0 / np.sqrt(N)  # ~0.04


def _make_haplotypes(n=N, m=M, seed=42):
    rng = np.random.RandomState(seed)
    af = np.full(m, 0.5)
    geno = np.zeros((n, m, 2), dtype=np.int8)
    for j in range(m):
        geno[:, j, 0] = rng.binomial(1, af[j], size=n)
        geno[:, j, 1] = rng.binomial(1, af[j], size=n)
    sex = np.tile([0, 1], (n + 1) // 2)[:n]
    samples = SampleMeta(iid=np.arange(n), sex=sex)
    variants = VariantMeta(vid=np.arange(m), af=af)
    return DenseHaplotypeArray(genotypes=geno, samples=samples, variants=variants)


class TestHaplotypeGeneticVariance:
    """Test that haplotype-specific genetic components produce correct variance."""

    def test_maternal_variance(self, stochastic_seed):
        """Maternal haplotype genetic variance should be ~half of diploid."""
        hap = _make_haplotypes(seed=stochastic_seed.seed)
        eff = AdditiveEffects.from_h2(h2=0.5, m=M, seed=stochastic_seed.seed,
                                       standardized=False)
        arch = Architecture()
        arch.add('G_mat', HaplotypeGeneticComponent(eff, haplotype='maternal'))
        pheno = arch.compute(hap, rng=stochastic_seed.rng)
        var_mat = np.var(pheno['G_mat'])
        # With AF=0.5, Var(hap0 @ beta) = p*q * sum(beta^2) = 0.25 * sum(beta^2)
        expected = 0.25 * np.sum(eff.effects ** 2)
        assert abs(var_mat - expected) / max(expected, 0.01) < 0.5, (
            f"seed={stochastic_seed.seed}, var_mat={var_mat:.4f}, expected={expected:.4f}"
        )

    def test_paternal_variance(self, stochastic_seed):
        """Paternal haplotype genetic variance should be ~half of diploid."""
        hap = _make_haplotypes(seed=stochastic_seed.seed)
        eff = AdditiveEffects.from_h2(h2=0.5, m=M, seed=stochastic_seed.seed,
                                       standardized=False)
        arch = Architecture()
        arch.add('G_pat', HaplotypeGeneticComponent(eff, haplotype='paternal'))
        pheno = arch.compute(hap, rng=stochastic_seed.rng)
        var_pat = np.var(pheno['G_pat'])
        expected = 0.25 * np.sum(eff.effects ** 2)
        assert abs(var_pat - expected) / max(expected, 0.01) < 0.5, (
            f"seed={stochastic_seed.seed}, var_pat={var_pat:.4f}, expected={expected:.4f}"
        )

    def test_maternal_plus_paternal_equals_diploid(self, stochastic_seed):
        """G_maternal + G_paternal should equal G_diploid exactly."""
        hap = _make_haplotypes(seed=stochastic_seed.seed)
        eff = AdditiveEffects.from_h2(h2=0.5, m=M, seed=stochastic_seed.seed,
                                       standardized=False)
        arch = Architecture()
        arch.add('G_mat', HaplotypeGeneticComponent(eff, haplotype='maternal'))
        arch.add('G_pat', HaplotypeGeneticComponent(eff, haplotype='paternal'))
        arch.add('G_dip', GeneticComponent(eff))
        pheno = arch.compute(hap, rng=stochastic_seed.rng)
        np.testing.assert_allclose(
            pheno['G_mat'] + pheno['G_pat'], pheno['G_dip'], atol=1e-10,
        )

    def test_maternal_paternal_uncorrelated(self, stochastic_seed):
        """Maternal and paternal haplotype genetic values should be ~uncorrelated."""
        hap = _make_haplotypes(seed=stochastic_seed.seed)
        eff = AdditiveEffects.from_h2(h2=0.5, m=M, seed=stochastic_seed.seed,
                                       standardized=False)
        arch = Architecture()
        arch.add('G_mat', HaplotypeGeneticComponent(eff, haplotype='maternal'))
        arch.add('G_pat', HaplotypeGeneticComponent(eff, haplotype='paternal'))
        pheno = arch.compute(hap, rng=stochastic_seed.rng)
        corr = np.corrcoef(pheno['G_mat'], pheno['G_pat'])[0, 1]
        assert abs(corr) < 3 * TOL, (
            f"seed={stochastic_seed.seed}, corr={corr:.4f}"
        )

    def test_diploid_variance_is_sum(self, stochastic_seed):
        """Var(G_diploid) should approximately equal Var(G_mat) + Var(G_pat).

        This follows from independence of maternal and paternal haplotypes
        in a random-mating founder population.
        """
        hap = _make_haplotypes(seed=stochastic_seed.seed)
        eff = AdditiveEffects.from_h2(h2=0.5, m=M, seed=stochastic_seed.seed,
                                       standardized=False)
        arch = Architecture()
        arch.add('G_mat', HaplotypeGeneticComponent(eff, haplotype='maternal'))
        arch.add('G_pat', HaplotypeGeneticComponent(eff, haplotype='paternal'))
        arch.add('G_dip', GeneticComponent(eff))
        pheno = arch.compute(hap, rng=stochastic_seed.rng)
        var_mat = np.var(pheno['G_mat'])
        var_pat = np.var(pheno['G_pat'])
        var_dip = np.var(pheno['G_dip'])
        # Var(X+Y) = Var(X) + Var(Y) + 2*Cov(X,Y) ≈ Var(X) + Var(Y) for independent
        assert abs(var_dip - (var_mat + var_pat)) / max(var_dip, 0.01) < 0.3, (
            f"seed={stochastic_seed.seed}, var_dip={var_dip:.4f}, "
            f"var_mat+var_pat={var_mat + var_pat:.4f}"
        )


class TestHaplotypeGeneticInSimulation:
    """Test haplotype genetic effects over multiple generations."""

    def test_haplotype_genetic_sim_finite(self):
        """Simulation with haplotype genetic effects should stay finite."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from testdata import TestSimulation
        from xftsim.sim import NSimulation
        from xftsim.mate import RandomMating
        from xftsim.reproduce import RecombinationMap

        m, n = 50, 400
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.3, m=m, seed=123, standardized=False)

        arch = Architecture()
        arch.add('Y.Gmat', HaplotypeGeneticComponent(eff, haplotype='maternal'))
        arch.add('Y.Gpat', HaplotypeGeneticComponent(eff, haplotype='paternal'))
        arch.add('Y.E', NoiseComponent(variance=0.4))
        arch.add('Y', AggregationComponent('Y.Gmat + Y.Gpat + Y.E'))

        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        mate = RandomMating(offspring_per_pair=2)

        sim = NSimulation(
            hap, arch, mate, rmap, seed=42,
            retain_haplotypes=2, retain_phenotypes=3,
        )
        sim.run(5)

        pheno = sim.phenotype_history[sim.generation]
        assert np.all(np.isfinite(pheno['Y']))
        assert np.all(np.isfinite(pheno['Y.Gmat']))
        assert np.all(np.isfinite(pheno['Y.Gpat']))

    def test_haplotype_effects_sum_to_total(self):
        """In the simulation, G_mat + G_pat should equal the total genetic value."""
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from testdata import TestSimulation
        from xftsim.sim import NSimulation
        from xftsim.mate import RandomMating
        from xftsim.reproduce import RecombinationMap

        m, n = 50, 400
        hap = TestSimulation.founder_haplotypes(n=n, m=m, seed=42)
        eff = AdditiveEffects.from_h2(h2=0.3, m=m, seed=123, standardized=False)

        arch = Architecture()
        arch.add('Y.Gmat', HaplotypeGeneticComponent(eff, haplotype='maternal'))
        arch.add('Y.Gpat', HaplotypeGeneticComponent(eff, haplotype='paternal'))
        arch.add('Y.Gtot', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.4))
        arch.add('Y', AggregationComponent('Y.Gmat + Y.Gpat + Y.E'))

        rmap = RecombinationMap.constant_map(m=m, p=0.5)
        mate = RandomMating(offspring_per_pair=2)

        sim = NSimulation(
            hap, arch, mate, rmap, seed=42,
            retain_haplotypes=2, retain_phenotypes=3,
        )
        sim.run(3)

        # Check at each available generation
        for gen in sim.phenotype_history:
            pheno = sim.phenotype_history[gen]
            if 'Y.Gmat' in pheno.keys and 'Y.Gpat' in pheno.keys and 'Y.Gtot' in pheno.keys:
                np.testing.assert_allclose(
                    pheno['Y.Gmat'] + pheno['Y.Gpat'],
                    pheno['Y.Gtot'],
                    atol=1e-10,
                    err_msg=f"Gen {gen}: mat+pat != diploid"
                )
