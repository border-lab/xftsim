"""
Tests for mating regimes.

Regression coverage for issue #18: LinearAssortativeMatingRegime silently
produced random mating for negative cross-mate correlations, because
np.sqrt(R) is NaN when R < 0 and np.argsort of an all-NaN array returns the
identity permutation.
"""
import numpy as np
import pytest


def _realized_mate_correlation(xft, r, seed=5, n=400, m=80, generations=3):
    """Run a short simulation and return the mean realized cross-mate correlation."""
    np.random.seed(seed)
    founders = xft.founders.founder_haplotypes_uniform_AFs(n=n, m=m)
    architecture = xft.arch.GCTA_Architecture(
        h2=[.5], phenotype_name=['pheno'], haplotypes=founders)
    recombination_map = xft.reproduce.RecombinationMap.constant_map_from_haplotypes(
        founders, p=.5)
    component_index = xft.index.ComponentIndex.from_product(['pheno'], ['phenotype'])
    regime = xft.mate.LinearAssortativeMatingRegime(
        r=r, component_index=component_index, offspring_per_pair=2)
    sim = xft.sim.Simulation(
        founder_haplotypes=founders,
        mating_regime=regime,
        recombination_map=recombination_map,
        architecture=architecture,
        statistics=[xft.stats.MatingStatistics()],
        post_processors=[xft.proc.LimitMemory(n_haplotype_generations=1)])
    sim.run(generations)
    correlations = [
        float(np.asarray(res['mating_statistics']['mate_correlations'])[0, 1])
        for res in list(sim.results_store.values())[1:]
    ]
    return float(np.mean(correlations))


class TestLinearAssortativeMatingRegime:
    """Cross-mate correlations must be realized with the requested sign."""

    @pytest.mark.timeout(120)
    def test_positive_correlation_is_realized(self, xft):
        assert _realized_mate_correlation(xft, 0.6) == pytest.approx(0.6, abs=0.12)

    @pytest.mark.timeout(120)
    def test_negative_correlation_is_realized(self, xft):
        """Regression test for issue #18.

        Before the fix this returned approximately zero rather than -0.6,
        because the mating scores were entirely NaN.
        """
        assert _realized_mate_correlation(xft, -0.6) == pytest.approx(-0.6, abs=0.12)

    @pytest.mark.timeout(120)
    def test_zero_correlation_is_realized(self, xft):
        assert _realized_mate_correlation(xft, 0.0) == pytest.approx(0.0, abs=0.12)

    @pytest.mark.timeout(120)
    def test_negative_and_positive_are_distinguishable(self, xft):
        """The specific failure mode was negative r behaving like r = 0."""
        negative = _realized_mate_correlation(xft, -0.6)
        zero = _realized_mate_correlation(xft, 0.0)
        assert negative < zero - 0.3

    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("r", [-0.8, -0.4, 0.4, 0.8])
    def test_correlation_is_realized_across_the_range(self, xft, r):
        """Both signs must be realized, not merely produce a finite number.

        Checking only for a finite result would not catch the original bug:
        the realized correlation was approximately zero, which is finite.
        """
        realized = _realized_mate_correlation(xft, r, generations=2)
        assert np.isfinite(realized)
        assert realized == pytest.approx(r, abs=0.15)
