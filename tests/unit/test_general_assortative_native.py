"""Tests for GeneralAssortativeMating with the native (non-Hexaly) solver."""
import numpy as np
import pytest

from xftsim.struct import SampleMeta, PhenotypeArray
from xftsim.mate import (BatchedMating, GeneralAssortativeMating,
                         MateAssignment, RandomMating)


def _make_pop(n=2000, K=3, seed=42):
    """Balanced-sex population with K correlated phenotype components."""
    rng = np.random.RandomState(seed)
    sex = np.tile([0, 1], (n + 1) // 2)[:n]
    samples = SampleMeta(iid=np.arange(n), sex=sex)
    pheno = PhenotypeArray(samples=samples)
    for k in range(K):
        pheno._values[f"Y{k}"] = rng.normal(size=n)
    return samples, pheno


def _observed_cross_corr(pheno, assignment, names):
    """Empirical cross-mate correlation of the realized pairing.

    Rows index the maternal (permuted) group, columns the paternal group,
    matching the ``cross_corr`` convention.
    """
    mat = assignment.maternal_idx
    pat = assignment.paternal_idx
    # One row per pair (offspring repeat their parents)
    _, first = np.unique(assignment.offspring_samples.fid, return_index=True)
    mat, pat = mat[first], pat[first]
    M = np.column_stack([pheno[nm][mat] for nm in names])
    P = np.column_stack([pheno[nm][pat] for nm in names])
    M = (M - M.mean(0)) / M.std(0)
    P = (P - P.mean(0)) / P.std(0)
    return M.T @ P / M.shape[0]


class TestNativeGeneralAssortativeMating:
    def test_constructs_without_hexaly(self):
        """The native path must not require the proprietary solver."""
        regime = GeneralAssortativeMating(['Y0', 'Y1'], np.eye(2) * 0.3)
        assert regime.solver == 'native'

    def test_achieves_target_cross_correlation(self):
        names = ['Y0', 'Y1', 'Y2']
        target = np.array([[0.40, 0.10, 0.00],
                           [0.10, 0.25, 0.05],
                           [0.00, 0.05, 0.15]])
        samples, pheno = _make_pop(n=2000, K=3)
        regime = GeneralAssortativeMating(names, target)
        assignment = regime.mate(samples, rng=np.random.RandomState(0),
                                 phenotypes=pheno)
        assert isinstance(assignment, MateAssignment)
        observed = _observed_cross_corr(pheno, assignment, names)
        np.testing.assert_allclose(observed, target, atol=0.01)

    def test_asymmetric_target_orientation(self):
        """cross_corr[i, j] is female component i against male component j."""
        names = ['Y0', 'Y1']
        target = np.array([[0.45, 0.05],
                           [0.05, 0.10]])
        samples, pheno = _make_pop(n=3000, K=2, seed=7)
        regime = GeneralAssortativeMating(names, target)
        assignment = regime.mate(samples, rng=np.random.RandomState(1),
                                 phenotypes=pheno)
        observed = _observed_cross_corr(pheno, assignment, names)
        assert abs(observed[0, 0] - 0.45) < 0.01
        assert abs(observed[1, 1] - 0.10) < 0.01

    def test_reproducible_given_rng_seed(self):
        names = ['Y0', 'Y1']
        target = np.eye(2) * 0.3
        samples, pheno = _make_pop(n=1000, K=2, seed=3)
        regime = GeneralAssortativeMating(names, target)
        a = regime.mate(samples, rng=np.random.RandomState(5), phenotypes=pheno)
        b = regime.mate(samples, rng=np.random.RandomState(5), phenotypes=pheno)
        np.testing.assert_array_equal(a.maternal_idx, b.maternal_idx)
        np.testing.assert_array_equal(a.paternal_idx, b.paternal_idx)

    def test_offspring_structure_matches_other_regimes(self):
        samples, pheno = _make_pop(n=500, K=2, seed=4)
        regime = GeneralAssortativeMating(['Y0', 'Y1'], np.eye(2) * 0.2,
                                          offspring_per_pair=3)
        assignment = regime.mate(samples, rng=np.random.RandomState(0),
                                 phenotypes=pheno)
        assert assignment.n_offspring == 250 * 3
        assert len(assignment.maternal_idx) == 250 * 3

    def test_unconverged_target_warns(self):
        """An unattainable target should be reported, not silently missed."""
        samples, pheno = _make_pop(n=400, K=2, seed=8)
        impossible = np.array([[1.2, 0.0], [0.0, 1.2]])
        regime = GeneralAssortativeMating(['Y0', 'Y1'], impossible,
                                          solver_params={'max_evals': 5_000})
        with pytest.warns(UserWarning):
            regime.mate(samples, rng=np.random.RandomState(0),
                        phenotypes=pheno)

    def test_last_result_exposes_diagnostics(self):
        samples, pheno = _make_pop(n=1000, K=2, seed=9)
        regime = GeneralAssortativeMating(['Y0', 'Y1'], np.eye(2) * 0.3)
        regime.mate(samples, rng=np.random.RandomState(0), phenotypes=pheno)
        assert regime.last_result.converged
        assert regime.last_result.max_abs_residual < 0.005

    def test_rejects_unknown_solver(self):
        with pytest.raises(ValueError, match="native.*hexaly"):
            GeneralAssortativeMating(['Y0'], np.eye(1), solver='gurobi')

    def test_rejects_wrong_solver_params_for_solver(self):
        """Hexaly keys must not silently pass through to the native solver."""
        with pytest.raises(ValueError, match="unknown solver_params"):
            GeneralAssortativeMating(['Y0'], np.eye(1),
                                     solver_params={'nb_threads': 4})

    def test_cross_corr_shape_validated(self):
        with pytest.raises(ValueError, match="does not match"):
            GeneralAssortativeMating(['Y0', 'Y1'], np.eye(3))


class TestSerializationRoundTrip:
    def test_native_regime_round_trips(self):
        from xftsim.io import (_serialize_mating_regime,
                               _deserialize_mating_regime)
        target = np.array([[0.3, 0.1], [0.1, 0.2]])
        regime = GeneralAssortativeMating(['Y0', 'Y1'], target,
                                          solver_params={'tol': 0.01})
        config = _serialize_mating_regime(regime)
        assert config['solver'] == 'native'
        restored = _deserialize_mating_regime(config)
        assert restored.solver == 'native'
        assert restored.solver_params['tol'] == 0.01
        np.testing.assert_array_equal(restored.cross_corr, target)

    def test_legacy_checkpoint_without_solver_key_resumes_on_hexaly(self):
        """Old checkpoints predate the native solver and must not be
        silently switched to a different solver on resume."""
        from xftsim.io import _deserialize_mating_regime
        legacy = {
            'type': 'GeneralAssortativeMating',
            'component_names': ['Y0', 'Y1'],
            'cross_corr': [[0.3, 0.0], [0.0, 0.3]],
            'offspring_per_pair': 2,
            'solver_params': {'nb_threads': 4, 'time_limit': 60},
        }
        pytest.importorskip('hexaly',
                            reason='legacy path constructs a hexaly regime')
        restored = _deserialize_mating_regime(legacy)
        assert restored.solver == 'hexaly'


class TestBatchedAutoSizing:
    """BatchedMating('auto') sizes batches to the accuracy floor and warns
    when the target is unreachable."""

    def test_auto_sizes_to_min_pairs_and_hits_tol(self):
        from xftsim.matchsolver import min_pairs_for_tol
        K = 5
        names = [f"Y{k}" for k in range(K)]
        target = 0.2 * np.eye(K)
        samples, pheno = _make_pop(n=40000, K=K, seed=0)
        reg = GeneralAssortativeMating(names, target, solver='native')
        bm = BatchedMating(reg)  # default 'auto'
        bs = bm._resolve_batch_size(samples)
        # ~2 individuals per pair, so batch ~ 2 x min_pairs.
        assert 1.5 * min_pairs_for_tol(0.005, K) <= bs <= 3 * min_pairs_for_tol(0.005, K)
        asg = bm.mate(samples, rng=np.random.RandomState(1), phenotypes=pheno)
        observed = _observed_cross_corr(pheno, asg, names)
        assert np.max(np.abs(observed - target)) < 0.006

    def test_unattainable_sample_warns(self):
        K = 10
        names = [f"Y{k}" for k in range(K)]
        target = 0.2 * np.eye(K)
        samples, _ = _make_pop(n=6000, K=K, seed=0)  # ~3000 pairs < 8000 needed
        reg = GeneralAssortativeMating(names, target, solver='native')
        bm = BatchedMating(reg)
        with pytest.warns(UserWarning, match="cannot reach"):
            bm._resolve_batch_size(samples)

    def test_explicit_undersized_batch_warns(self):
        K = 8
        names = [f"Y{k}" for k in range(K)]
        target = 0.15 * np.eye(K)
        samples, _ = _make_pop(n=20000, K=K, seed=0)
        reg = GeneralAssortativeMating(names, target, solver='native')
        bm = BatchedMating(reg, max_batch_size=1000)  # ~500 pairs < 2000 needed
        with pytest.warns(UserWarning, match="below the"):
            bm._resolve_batch_size(samples)

    def test_regime_without_target_uses_fallback(self):
        samples, _ = _make_pop(n=5000, K=2, seed=0)
        bm = BatchedMating(RandomMating(offspring_per_pair=2))
        # No cross-correlation target: fall back to the fixed default, no warn.
        assert bm._resolve_batch_size(samples) == BatchedMating._AUTO_FALLBACK

    def test_rejects_bad_max_batch_size(self):
        with pytest.raises(ValueError, match="positive integer or 'auto'"):
            BatchedMating(RandomMating(), max_batch_size=0)
        with pytest.raises(ValueError, match="positive integer or 'auto'"):
            BatchedMating(RandomMating(), max_batch_size="big")
