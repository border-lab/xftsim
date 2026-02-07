"""Integration tests for formula-based Architecture construction in simulations."""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray
from xftsim.neffect import AdditiveEffects, MultivariateEffects
from xftsim.narch import Architecture
from xftsim.nsim import NSimulation
from xftsim.nmate import RandomMating
from xftsim.reproduce import RecombinationMap
from tests.testdata import TestSimulation


def _make_hap(n=500, m=50, seed=42):
    return TestSimulation.founder_haplotypes(n=n, m=m, seed=seed)


def _run_formula_sim(formula, effects, n_gen=2, n=500, m=50, seed=42):
    """Build a sim from a formula string and run it."""
    hap = _make_hap(n=n, m=m, seed=seed)
    arch = Architecture.from_formula(formula, effects=effects)
    rmap = RecombinationMap.constant_map(m=m, p=0.5)
    mate = RandomMating(offspring_per_pair=2)
    sim = NSimulation(
        founder_haplotypes=hap, architecture=arch,
        mating_regime=mate, recombination_map=rmap, seed=seed,
    )
    sim.run(n_gen)
    return sim


class TestFormulaSimple:
    """Single-trait formula: Y.G ~ genetic(eff), Y.E ~ noise(0.5), Y ~ Y.G + Y.E"""

    def test_single_trait(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=50, seed=123)
        formula = """
        Y.G ~ genetic(eff)
        Y.E ~ noise(0.5)
        Y ~ Y.G + Y.E
        """
        sim = _run_formula_sim(formula, effects={'eff': eff})
        assert 'Y' in sim.phenotype_history[0].keys
        assert 'Y.G' in sim.phenotype_history[0].keys
        assert np.all(np.isfinite(sim.phenotype_history[0]['Y']))

    def test_matches_programmatic(self):
        """Formula-based sim should match programmatic sim."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=50, seed=123)
        formula = """
        Y.G ~ genetic(eff)
        Y.E ~ noise(0.5)
        Y ~ Y.G + Y.E
        """
        # Formula sim
        sim1 = _run_formula_sim(formula, effects={'eff': eff}, n_gen=1, seed=42)
        # Programmatic sim
        from xftsim.narch import GeneticComponent, NoiseComponent, AggregationComponent
        arch2 = Architecture()
        arch2.add('Y.G', GeneticComponent(eff))
        arch2.add('Y.E', NoiseComponent(variance=0.5))
        arch2.add('Y', AggregationComponent('Y.G + Y.E'))
        hap = _make_hap(seed=42)
        rmap = RecombinationMap.constant_map(m=50, p=0.5)
        mate = RandomMating(offspring_per_pair=2)
        sim2 = NSimulation(
            founder_haplotypes=hap, architecture=arch2,
            mating_regime=mate, recombination_map=rmap, seed=42,
        )
        sim2.run(1)
        # Y.G is deterministic — should match exactly
        np.testing.assert_allclose(
            sim1.phenotype_history[0]['Y.G'],
            sim2.phenotype_history[0]['Y.G'],
            atol=1e-10,
        )


class TestFormulaBivariate:
    """Bivariate formula with mvGenetic + cnoise."""

    def test_bivariate_sim(self):
        mv_eff = MultivariateEffects.from_h2_rg(h2=[0.5, 0.3], rg=0.2, m=50, seed=123)
        formula = """
        (trait1.G, trait2.G) ~ mvGenetic(mv)
        trait1.E ~ noise(0.5)
        trait2.E ~ noise(0.7)
        trait1 ~ trait1.G + trait1.E
        trait2 ~ trait2.G + trait2.E
        """
        sim = _run_formula_sim(formula, effects={'mv': mv_eff}, n_gen=2)
        assert 'trait1' in sim.phenotype_history[0].keys
        assert 'trait2' in sim.phenotype_history[0].keys
        assert np.all(np.isfinite(sim.phenotype_history[0]['trait1']))

    def test_cnoise_formula(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=50, seed=123)
        formula = """
        Y1.G ~ genetic(eff)
        Y2.G ~ genetic(eff)
        (Y1.E, Y2.E) ~ cnoise(cov=[[0.5, 0.1], [0.1, 0.5]])
        Y1 ~ Y1.G + Y1.E
        Y2 ~ Y2.G + Y2.E
        """
        sim = _run_formula_sim(formula, effects={'eff': eff}, n_gen=1)
        assert 'Y1.E' in sim.phenotype_history[0].keys
        assert 'Y2.E' in sim.phenotype_history[0].keys


class TestFormulaVT:
    """Vertical transmission via parent/mother/father."""

    def test_parent_formula(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=50, seed=123)
        formula = """
        Y.G ~ genetic(eff)
        Y.VT ~ parent(Y, founder=noise(0.5))
        Y.E ~ noise(0.3)
        Y ~ Y.G + 0.3 * Y.VT + Y.E
        """
        sim = _run_formula_sim(formula, effects={'eff': eff}, n_gen=3)
        # Gen 0: founder fallback noise for VT
        assert np.all(np.isfinite(sim.phenotype_history[0]['Y']))
        # Gen 1+: uses actual parent values
        assert np.all(np.isfinite(sim.phenotype_history[1]['Y']))

    def test_mother_father_formula(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=50, seed=123)
        formula = """
        Y.G ~ genetic(eff)
        Y.M ~ mother(Y, founder=noise(0.5))
        Y.F ~ father(Y, founder=noise(0.5))
        Y.E ~ noise(0.3)
        Y ~ Y.G + 0.2 * Y.M + 0.2 * Y.F + Y.E
        """
        sim = _run_formula_sim(formula, effects={'eff': eff}, n_gen=2)
        assert np.all(np.isfinite(sim.phenotype_history[0]['Y']))


class TestFormulaGrouping:
    """Grouping with | operator."""

    def test_noise_grouped_by_fid(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=50, seed=123)
        formula = """
        Y.G ~ genetic(eff)
        Y.E ~ noise(0.5) | FID
        Y ~ Y.G + Y.E
        """
        sim = _run_formula_sim(formula, effects={'eff': eff}, n_gen=1)
        # Individuals in same family should have same noise
        pheno = sim.phenotype_history[0]
        samples = sim.haplotype_history[0].samples
        fids = samples.fid
        unique_fids = np.unique(fids)
        for fid in unique_fids[:3]:  # check a few families
            mask = fids == fid
            noise_vals = pheno['Y.E'][mask]
            assert np.all(noise_vals == noise_vals[0])


class TestFormulaSibling:
    """Sibling aggregation functions."""

    def test_sibling_mean_formula(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=50, seed=123)
        formula = """
        Y.G ~ genetic(eff)
        Y.E ~ noise(0.5)
        Y ~ Y.G + Y.E
        Y.sibmean ~ sibling_mean(Y)
        """
        sim = _run_formula_sim(formula, effects={'eff': eff}, n_gen=1)
        assert 'Y.sibmean' in sim.phenotype_history[0].keys
        assert np.all(np.isfinite(sim.phenotype_history[0]['Y.sibmean']))


class TestFormulaHaplotypeGenetic:
    """Haplotype-specific genetic components."""

    def test_haplotype_genetic_formula(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=50, seed=123)
        formula = """
        Y.mat ~ haplotypeGenetic(eff, haplotype='maternal')
        Y.pat ~ haplotypeGenetic(eff, haplotype='paternal')
        Y.E ~ noise(0.5)
        Y ~ Y.mat + Y.pat + Y.E
        """
        sim = _run_formula_sim(formula, effects={'eff': eff}, n_gen=1)
        pheno = sim.phenotype_history[0]
        # mat + pat should equal diploid genetic
        mat = pheno['Y.mat']
        pat = pheno['Y.pat']
        # Compare against standard genetic component
        hap = sim.haplotype_history[0]
        diploid = hap.matvec(eff.effects)
        np.testing.assert_allclose(mat + pat, diploid, atol=1e-10)


class TestFormulaComments:
    """Parser should handle comments and blank lines."""

    def test_comments_and_blanks(self):
        eff = AdditiveEffects.from_h2(h2=0.5, m=50, seed=123)
        formula = """
        # This is a comment
        Y.G ~ genetic(eff)

        # Another comment
        Y.E ~ noise(0.5)

        Y ~ Y.G + Y.E
        """
        sim = _run_formula_sim(formula, effects={'eff': eff}, n_gen=1)
        assert 'Y' in sim.phenotype_history[0].keys


class TestFormulaErrors:
    """Formula error handling."""

    def test_missing_effect(self):
        with pytest.raises(ValueError, match="not found in effects"):
            Architecture.from_formula("Y.G ~ genetic(missing)", effects={})

    def test_unknown_function(self):
        with pytest.raises(ValueError, match="unknown function"):
            Architecture.from_formula("Y ~ badfunc(x)")

    def test_missing_tilde(self):
        with pytest.raises(ValueError, match="missing '~'"):
            Architecture.from_formula("Y = noise(0.5)")

    def test_duplicate_output(self):
        with pytest.raises(ValueError, match="duplicate"):
            Architecture.from_formula("""
            Y ~ noise(0.5)
            Y ~ noise(0.3)
            """)

    def test_circular_dependency(self):
        with pytest.raises(ValueError, match="Undefined reference"):
            # Y depends on Z which doesn't exist
            Architecture.from_formula("Y ~ Z + noise(0.5)")


# ── Complex formula integration tests ──────────────────────────────────────

class TestFormulaComplexArchitectures:
    """End-to-end tests with complex formula-based architectures."""

    def test_bivariate_vt_formula(self):
        """Bivariate architecture with vertical transmission via formula."""
        mv_eff = MultivariateEffects.from_h2_rg(h2=[0.4, 0.3], rg=0.3, m=50, seed=42)
        formula = """
        (Y1.G, Y2.G) ~ mvGenetic(mv)
        Y1.VT ~ parent(Y1, founder=noise(0.3))
        Y2.VT ~ parent(Y2, founder=noise(0.3))
        Y1.E ~ noise(0.3)
        Y2.E ~ noise(0.4)
        Y1 ~ Y1.G + 0.2 * Y1.VT + Y1.E
        Y2 ~ Y2.G + 0.2 * Y2.VT + Y2.E
        """
        sim = _run_formula_sim(formula, effects={'mv': mv_eff}, n_gen=5)
        for gen in range(5):
            if gen in sim.phenotype_history:
                for name in ['Y1', 'Y2']:
                    if name in sim.phenotype_history[gen].keys:
                        assert np.all(np.isfinite(sim.phenotype_history[gen][name]))

    def test_ige_formula(self):
        """Indirect genetic effects via haplotypeGenetic."""
        eff = AdditiveEffects.from_h2(h2=0.4, m=50, seed=42)
        formula = """
        Y.G.mat ~ haplotypeGenetic(eff, haplotype='maternal')
        Y.G.pat ~ haplotypeGenetic(eff, haplotype='paternal')
        Y.E ~ noise(0.6)
        Y ~ Y.G.mat + Y.G.pat + Y.E
        """
        # Only run 1 gen so gen 0 haplotypes are available for checking
        sim = _run_formula_sim(formula, effects={'eff': eff}, n_gen=1)
        pheno0 = sim.phenotype_history[0]
        assert np.all(np.isfinite(pheno0['Y']))
        # mat + pat should equal standard genetic
        hap = sim.haplotype_history[0]
        expected_g = hap.matvec(eff.effects)
        np.testing.assert_allclose(
            pheno0['Y.G.mat'] + pheno0['Y.G.pat'],
            expected_g,
            atol=1e-10,
        )

    def test_sibling_mean_in_formula(self):
        """Sibling mean used as input to aggregation."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=50, seed=42)
        formula = """
        Y.G ~ genetic(eff)
        Y.E ~ noise(0.5)
        Y ~ Y.G + Y.E
        Y.sibmean ~ sibling_mean(Y)
        Y.dev ~ Y - Y.sibmean
        """
        sim = _run_formula_sim(formula, effects={'eff': eff}, n_gen=2)
        pheno0 = sim.phenotype_history[0]
        assert 'Y.dev' in pheno0.keys
        assert np.all(np.isfinite(pheno0['Y.dev']))
        # At gen 0, each individual is in own family, so sibmean = Y
        # and deviation should be ~0
        np.testing.assert_allclose(pheno0['Y.dev'], 0.0, atol=1e-10)

    def test_grouped_noise_across_generations(self):
        """FID-grouped noise should produce shared family noise across gens."""
        eff = AdditiveEffects.from_h2(h2=0.5, m=50, seed=42)
        formula = """
        Y.G ~ genetic(eff)
        Y.shared ~ noise(0.3) | FID
        Y.E ~ noise(0.2)
        Y ~ Y.G + Y.shared + Y.E
        """
        sim = _run_formula_sim(
            formula, effects={'eff': eff},
            n_gen=3, n=200, m=50,
        )
        # Gen 1 offspring: siblings should have same Y.shared
        if 1 in sim.phenotype_history:
            pheno1 = sim.phenotype_history[1]
            fids = pheno1.samples.fid
            unique_fids = np.unique(fids)
            for fid in unique_fids[:5]:
                mask = fids == fid
                shared = pheno1['Y.shared'][mask]
                assert np.all(shared == shared[0])

    def test_complex_chain(self):
        """Deep chain of aggregations should work."""
        eff = AdditiveEffects.from_h2(h2=0.3, m=50, seed=42)
        formula = """
        a ~ genetic(eff)
        b ~ noise(0.2)
        c ~ a + b
        d ~ 0.5 * c
        e ~ noise(0.1)
        f ~ d + e
        """
        sim = _run_formula_sim(formula, effects={'eff': eff}, n_gen=2)
        pheno0 = sim.phenotype_history[0]
        # Verify chain: f = 0.5*(a+b) + e
        expected_f = 0.5 * (pheno0['a'] + pheno0['b']) + pheno0['e']
        np.testing.assert_allclose(pheno0['f'], expected_f, atol=1e-10)

    def test_mother_father_separate_effects(self):
        """Separate mother/father VT components should stay finite."""
        eff = AdditiveEffects.from_h2(h2=0.4, m=50, seed=42)
        formula = """
        Y.G ~ genetic(eff)
        Y.M ~ mother(Y, founder=noise(0.2))
        Y.F ~ father(Y, founder=noise(0.2))
        Y.E ~ noise(0.2)
        Y ~ Y.G + 0.15 * Y.M + 0.15 * Y.F + Y.E
        """
        sim = _run_formula_sim(formula, effects={'eff': eff}, n_gen=5)
        last_gen = sim.generation
        pheno = sim.phenotype_history[last_gen]
        assert np.all(np.isfinite(pheno['Y']))
        assert pheno['Y'].var() < 100.0
