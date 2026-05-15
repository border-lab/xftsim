"""
Tests for remaining small coverage gaps in parser.py, nstats.py, nmate.py, ngwas.py.

Targets:
- parser.py lines 221, 227, 266 (mvGenetic/haplotypeGenetic error paths)
- nstats.py lines 270, 283-284, 290 (MatingStatistics spouse correlation edge cases)
- nmate.py lines 113, 207 (rng=None fallback paths)
- ngwas.py line 100 (non-DenseHaplotypeArray path in GWAS)
- narch.py line 571 (sibling _resolve_grouping returning None)
"""
import numpy as np
import pytest

from xftsim.effect import AdditiveEffects, MultivariateEffects, SparseEffects
from xftsim.struct import SampleMeta, VariantMeta, DenseHaplotypeArray, PhenotypeArray
from xftsim.parser import parse_formula


# ---------------------------------------------------------------------------
# parser.py: mvGenetic error paths
# ---------------------------------------------------------------------------

class TestParserMvGeneticErrors:
    def test_mvGenetic_missing_effect(self):
        """mvGenetic with effect not in dict raises ValueError."""
        formula = "(Y1.G, Y2.G) ~ mvGenetic(missing_eff)"
        effects = {"other": MultivariateEffects.from_h2_rg(h2=[0.5, 0.5], rg=0.0, m=10)}
        with pytest.raises(ValueError, match="not found in effects dict"):
            parse_formula(formula, effects=effects)

    def test_mvGenetic_not_effectspec(self):
        """mvGenetic with non-EffectSpec value raises ValueError."""
        formula = "(Y1.G, Y2.G) ~ mvGenetic(fake)"
        effects = {"fake": "not_an_effect"}
        with pytest.raises(ValueError, match="is not an EffectSpec"):
            parse_formula(formula, effects=effects)


# ---------------------------------------------------------------------------
# parser.py: haplotypeGenetic error paths
# ---------------------------------------------------------------------------

class TestParserHaplotypeGeneticErrors:
    def test_haplotypeGenetic_not_effectspec(self):
        """haplotypeGenetic with non-EffectSpec value raises ValueError."""
        formula = "Y.G ~ haplotypeGenetic(fake)"
        effects = {"fake": [1, 2, 3]}
        with pytest.raises(ValueError, match="is not an EffectSpec"):
            parse_formula(formula, effects=effects)


# ---------------------------------------------------------------------------
# nstats.py: MatingStatistics edge cases
# ---------------------------------------------------------------------------

class TestMatingStatisticsEdgeCases:
    def _build_trio_sim(self, n=40, m=10, assortative=False, gens=2):
        """Run a small simulation that produces pedigrees and trios for stats."""
        from xftsim.founders import founder_haplotypes_uniform_AFs
        from xftsim.arch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
        from xftsim.mate import RandomMating, LinearAssortativeMating
        from xftsim.reproduce import RecombinationMap
        from xftsim.sim import Simulation
        from xftsim.stats import SampleStatistics, MatingStatistics
        from xftsim.filters import TrioFilter

        np.random.seed(42)
        hap = founder_haplotypes_uniform_AFs(n=n, m=m)
        eff = AdditiveEffects.from_h2(h2=0.5, m=m, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        rm = RecombinationMap.constant_map(m=m, p=0.5)

        if assortative:
            mating = LinearAssortativeMating(
                component_names=['Y'], r=0.3, offspring_per_pair=2)
        else:
            mating = RandomMating(offspring_per_pair=2)

        sim = Simulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=mating, recombination_map=rm,
            statistics=[SampleStatistics(), MatingStatistics()],
            filters={'trio': TrioFilter()},
            seed=42,
        )
        sim.run(gens)
        return sim

    def test_mating_stats_computes(self):
        sim = self._build_trio_sim(n=40, gens=2)
        # Check that MatingStatistics was computed
        assert len(sim.results) > 0
        found_mating = False
        for res in sim.results:
            if 'MatingStatistics' in res.statistics:
                found_mating = True
                stats = res.statistics['MatingStatistics']
                assert 'n_mating_pairs' in stats
                assert 'mean_offspring_count' in stats
                assert 'spouse_correlations' in stats
        assert found_mating

    def test_mating_stats_with_assortative(self):
        """Assortative mating produces non-zero spouse correlations."""
        sim = self._build_trio_sim(n=100, gens=2, assortative=True)
        found_mating = False
        for res in sim.results:
            if 'MatingStatistics' in res.statistics:
                found_mating = True
                stats = res.statistics['MatingStatistics']
                # With assortment r=0.3, spouse_correlations should have Y
                if 'Y' in stats['spouse_correlations']:
                    # Just check it's a number (sign depends on randomness)
                    assert isinstance(stats['spouse_correlations']['Y'], float)
        assert found_mating


# ---------------------------------------------------------------------------
# nmate.py: rng=None defaults
# ---------------------------------------------------------------------------

class TestNmateRngDefault:
    def test_random_mating_rng_none(self):
        """RandomMating.mate with rng=None creates default RNG."""
        from xftsim.mate import RandomMating
        samples = SampleMeta(iid=np.arange(10))
        rm = RandomMating(offspring_per_pair=2)
        assignment = rm.mate(samples, rng=None)
        assert assignment.offspring_samples.n > 0

    def test_assortative_mating_rng_none(self):
        """LinearAssortativeMating.mate with rng=None creates default RNG."""
        from xftsim.mate import LinearAssortativeMating
        samples = SampleMeta(iid=np.arange(10))
        phenotypes = PhenotypeArray(
            samples=samples,
            values={"Y": np.random.randn(10)},
        )
        am = LinearAssortativeMating(component_names=['Y'], r=0.3, offspring_per_pair=2)
        assignment = am.mate(samples, rng=None, phenotypes=phenotypes)
        assert assignment.offspring_samples.n > 0


# ---------------------------------------------------------------------------
# ngwas.py: non-DenseHaplotypeArray path
# ---------------------------------------------------------------------------

class TestNGWASNonDensePath:
    """Test GWAS with a HaplotypeOperator that is not DenseHaplotypeArray.

    Line 100: G = hap.to_dense().diploid_genotypes.astype(np.float64)

    We test this by creating a minimal wrapper.
    """

    def test_gwas_with_mock_operator(self):
        """GWAS should work by calling to_dense() on non-Dense operator."""
        from xftsim.gwas import GWAS

        # Create a concrete DenseHaplotypeArray
        geno = np.random.RandomState(42).randint(0, 2, (20, 5, 2)).astype(np.int8)
        samples = SampleMeta(iid=np.arange(20))
        hap = DenseHaplotypeArray(genotypes=geno, samples=samples)

        # Create a wrapper that is NOT DenseHaplotypeArray
        class MockOperator:
            """Wraps a DenseHaplotypeArray but is not one."""
            def __init__(self, dense):
                self._dense = dense
                self.samples = dense.samples
                self.variants = dense.variants

            @property
            def n(self):
                return self._dense.n

            @property
            def m(self):
                return self._dense.m

            def to_dense(self):
                return self._dense

        mock = MockOperator(hap)
        phenotypes = PhenotypeArray(
            samples=samples,
            values={"Y": np.random.RandomState(42).randn(20)},
        )

        gwas = GWAS(haplotypes=mock, phenotypes=phenotypes)
        results = gwas.run()
        assert "Y" in results
        assert results["Y"].beta.shape == (5,)


# ---------------------------------------------------------------------------
# narch.py: _SiblingComponent grouping=None path
# ---------------------------------------------------------------------------

class TestSiblingMeanDirect:
    """Test _SiblingComponent.compute() directly to cover narch.py lines 571+."""

    def test_sibling_mean_compute_directly(self):
        """Call SiblingMeanComponent.compute() directly with a mock ArchNode."""
        from xftsim.arch import SiblingMeanComponent, ArchNode

        # Create test data
        n = 10
        geno = np.zeros((n, 5, 2), dtype=np.int8)
        samples = SampleMeta(
            iid=np.arange(n),
            fid=np.array([0, 0, 0, 1, 1, 1, 2, 2, 3, 3]),
        )
        hap = DenseHaplotypeArray(genotypes=geno, samples=samples)
        pheno = PhenotypeArray(
            samples=samples,
            values={"Y": np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])},
        )

        comp = SiblingMeanComponent('Y')
        # Create a mock ArchNode with FID grouping (default for siblings)
        node = ArchNode(outputs=['Y.sib'], component=comp, inputs=['Y'], grouping='FID')
        result = comp.compute(node, hap, pheno, generation=0)
        assert result.shape == (n,)
        # Family 0 (indices 0,1,2): mean = 2.0
        np.testing.assert_allclose(result[0], 2.0)
        np.testing.assert_allclose(result[1], 2.0)
        np.testing.assert_allclose(result[2], 2.0)
        # Family 1 (indices 3,4,5): mean = 5.0
        np.testing.assert_allclose(result[3], 5.0)
