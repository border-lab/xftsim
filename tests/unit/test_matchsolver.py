"""Tests for the native cross-correlation matching solver."""
import numpy as np
import pytest

from xftsim.matchsolver import MatchResult, solve_cross_correlation


def _achievable(n, K, seed, rho_scale=1.0):
    """Y, Z and a cross-correlation target attained by a known permutation."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(2 * K, 2 * K))
    omega = A @ A.T + 0.5 * np.eye(2 * K)
    J = rng.multivariate_normal(np.zeros(2 * K), omega, size=n)
    Y, Z = J[:, :K], J[:, K:]
    zy = (Y - Y.mean(0)) / Y.std(0)
    zz = (Z - Z.mean(0)) / Z.std(0)
    R = rho_scale * (zy.T @ zz) / n
    return Y[rng.permutation(n)], Z, R


def _achieved(Y, Z, perm):
    """Empirical cross-correlation of the permuted pairing."""
    n = Y.shape[0]
    zy = (Y - Y.mean(0)) / Y.std(0)
    zz = (Z - Z.mean(0)) / Z.std(0)
    return zy[perm].T @ zz / n


class TestSolveCrossCorrelation:
    def test_hits_achievable_target(self):
        Y, Z, R = _achievable(2000, 4, seed=1)
        res = solve_cross_correlation(Y, Z, R, seed=0)
        assert isinstance(res, MatchResult)
        assert res.converged
        assert res.max_abs_residual < 0.005

    def test_returns_valid_permutation(self):
        Y, Z, R = _achievable(1000, 3, seed=2)
        res = solve_cross_correlation(Y, Z, R, seed=0)
        np.testing.assert_array_equal(np.sort(res.perm), np.arange(1000))
        assert res.perm.dtype == np.int64

    def test_reported_residual_matches_independent_recomputation(self):
        """The reported residual must be the real one, in caller orientation."""
        Y, Z, R = _achievable(1000, 3, seed=3)
        res = solve_cross_correlation(Y, Z, R, seed=0)
        independent = R - _achieved(Y, Z, res.perm)
        np.testing.assert_allclose(res.residual, independent, atol=1e-12)
        np.testing.assert_allclose(res.max_abs_residual,
                                   np.max(np.abs(independent)), rtol=1e-10)

    def test_orientation_is_not_transposed(self):
        """An asymmetric target catches a transposed cross-correlation."""
        rng = np.random.default_rng(11)
        n = 4000
        Z = rng.normal(size=(n, 2))
        Y = rng.normal(size=(n, 2))
        R = np.array([[0.45, 0.05], [0.05, 0.10]])  # asymmetric
        res = solve_cross_correlation(Y, Z, R, seed=0)
        ach = _achieved(Y, Z, res.perm)
        assert abs(ach[0, 0] - 0.45) < 0.01
        assert abs(ach[1, 1] - 0.10) < 0.01

    def test_creates_dependence_between_independent_groups(self):
        rng = np.random.default_rng(4)
        Y = rng.normal(size=(3000, 3))
        Z = rng.normal(size=(3000, 3))
        R = 0.3 * np.eye(3)
        res = solve_cross_correlation(Y, Z, R, seed=0)
        assert res.converged
        assert res.max_abs_residual < 0.005

    def test_determinism(self):
        Y, Z, R = _achievable(1000, 3, seed=5)
        a = solve_cross_correlation(Y, Z, R, seed=42)
        b = solve_cross_correlation(Y, Z, R, seed=42)
        np.testing.assert_array_equal(a.perm, b.perm)
        assert a.max_abs_residual == b.max_abs_residual

    def test_different_seeds_differ(self):
        Y, Z, R = _achievable(1000, 3, seed=6)
        a = solve_cross_correlation(Y, Z, R, seed=1)
        b = solve_cross_correlation(Y, Z, R, seed=2)
        assert not np.array_equal(a.perm, b.perm)

    def test_infeasible_target_warns_and_returns_best_effort(self):
        rng = np.random.default_rng(7)
        Y = rng.normal(size=(500, 2))
        Z = rng.normal(size=(500, 2))
        R = np.array([[1.4, 0.0], [0.0, 1.4]])  # |rho| > 1: impossible
        with pytest.warns(UserWarning, match="infeasible"):
            res = solve_cross_correlation(Y, Z, R, seed=0, max_evals=20_000)
        assert not res.converged
        assert np.isfinite(res.frobenius_residual)
        np.testing.assert_array_equal(np.sort(res.perm), np.arange(500))

    def test_constant_column_is_tolerated(self):
        """A degenerate phenotype carries no signal but must not crash."""
        rng = np.random.default_rng(8)
        Y = rng.normal(size=(400, 2))
        Y[:, 1] = 3.0
        Z = rng.normal(size=(400, 2))
        R = np.zeros((2, 2))
        res = solve_cross_correlation(Y, Z, R, seed=0, max_evals=20_000,
                                      warn_infeasible=False)
        np.testing.assert_array_equal(np.sort(res.perm), np.arange(400))

    def test_rectangular_component_counts(self):
        """Mate groups may carry different numbers of components."""
        rng = np.random.default_rng(9)
        Y = rng.normal(size=(2000, 4))
        Z = rng.normal(size=(2000, 2))
        R = 0.2 * np.ones((4, 2))
        res = solve_cross_correlation(Y, Z, R, seed=0)
        assert res.residual.shape == (4, 2)
        assert res.max_abs_residual < 0.005

    def test_shape_validation(self):
        Y = np.zeros((10, 2))
        Z = np.zeros((10, 2))
        with pytest.raises(ValueError, match="row counts"):
            solve_cross_correlation(np.zeros((5, 2)), Z, np.zeros((2, 2)))
        with pytest.raises(ValueError, match="does not match"):
            solve_cross_correlation(Y, Z, np.zeros((3, 2)))

    def test_tolerance_is_respected(self):
        """A loose tolerance should stop early; a tight one should do more."""
        Y, Z, R = _achievable(2000, 3, seed=10)
        loose = solve_cross_correlation(Y, Z, R, tol=0.05, seed=0)
        tight = solve_cross_correlation(Y, Z, R, tol=0.005, seed=0)
        assert loose.max_abs_residual < 0.05
        assert tight.max_abs_residual < 0.005
        assert tight.evals >= loose.evals
