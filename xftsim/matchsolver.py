"""Native cross-correlation matching solver for assortative mating.

Finds a permutation of one mate group that induces a prescribed K x K
cross-mate correlation matrix, replacing the proprietary Hexaly QAP solver.

The objective depends on the permutation only through the K x K statistic
``M(P) = Z' P Y``, so the whole search happens in K^2 dimensions and never
forms an n x n matrix. Two stages:

1. Greedy residual-tracking construction. Visit rows of the fixed group in
   random order; for each, pick the best of ``m`` sampled unmatched partners
   by how well the running cross-correlation tracks the target.
2. Swap local search. Swapping the partners of positions i and j changes M
   by the rank-1 matrix ``outer(a, b)`` with ``a = zt_i - zt_j`` and
   ``b = zy_pi[j] - zy_pi[i]``, so the exact objective change costs O(K^2):

       delta_f = -(2/n) a' R b + ||a||^2 ||b||^2 / n^2

   Greedy acceptance, with proposals mixing uniform pairs and pairs drawn
   from nearest-neighbor lists for fine adjustments near the target.

Both stages evaluate the true objective exactly at every step; the final
residual is recomputed from the permutation so no float drift leaks into
reported diagnostics.

Cost is O(n K^2) per stage-1 pass and O(K^2) per swap evaluation, versus the
O(n^2) memory of the Koopmans-Beckmann encoding the Hexaly path uses. Memory
is O(nK), so n in the hundreds of thousands is routine.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

try:  # numba is a hard dependency of xftsim; the fallback is insurance only
    from numba import njit
    _HAVE_NUMBA = True
except ImportError:  # pragma: no cover - exercised only without numba
    _HAVE_NUMBA = False

    def njit(*args, **kwargs):
        def _wrap(func):
            return func
        if args and callable(args[0]):
            return args[0]
        return _wrap


@dataclass
class MatchResult:
    """Outcome of a cross-correlation matching solve.

    Attributes
    ----------
    perm : np.ndarray
        (n,) int64 permutation of the mobile group's rows.
    residual : np.ndarray
        (K, K) target minus achieved cross-correlation, in the caller's
        orientation.
    max_abs_residual : float
        Largest absolute entry of ``residual``.
    frobenius_residual : float
        Frobenius norm of ``residual``.
    evals : int
        Swap evaluations performed in stage 2.
    converged : bool
        Whether ``max_abs_residual`` fell below the requested tolerance.
    """

    perm: np.ndarray
    residual: np.ndarray
    max_abs_residual: float
    frobenius_residual: float
    evals: int
    converged: bool


@njit(cache=True)
def _construct(Zy, Zt, C, order, m, seed):
    """Greedy residual-tracking construction (stage 1)."""
    np.random.seed(seed)
    n, Ky = Zy.shape
    Kt = Zt.shape[1]

    sqn = np.zeros(n)
    for i in range(n):
        acc = 0.0
        for b in range(Ky):
            acc += Zy[i, b] * Zy[i, b]
        sqn[i] = acc

    pool = np.arange(n)
    size = n
    M = np.zeros((Kt, Ky))
    pi = np.empty(n, dtype=np.int64)
    w = np.zeros(Ky)

    for t in range(n):
        j = order[t]
        uu = 0.0
        for a in range(Kt):
            uu += Zt[j, a] * Zt[j, a]
        # w = E' u with E = M - (t+1) C, the schedule drift before this pick
        for b in range(Ky):
            acc = 0.0
            for a in range(Kt):
                acc += (M[a, b] - (t + 1) * C[a, b]) * Zt[j, a]
            w[b] = acc

        k = m if m < size else size
        best_idx = -1
        best_slot = -1
        best_score = 1.0e300
        for _ in range(k):
            slot = np.random.randint(0, size)
            idx = pool[slot]
            acc = 0.0
            for b in range(Ky):
                acc += Zy[idx, b] * w[b]
            score = 2.0 * acc + uu * sqn[idx]
            if score < best_score:
                best_score = score
                best_idx = idx
                best_slot = slot

        pi[j] = best_idx
        for a in range(Kt):
            zta = Zt[j, a]
            for b in range(Ky):
                M[a, b] += zta * Zy[best_idx, b]
        size -= 1
        pool[best_slot] = pool[size]

    return pi


@njit(cache=True)
def _polish(Zy, Zt, C, pi, R, tol, max_evals, check_every, p_fine, nbrs,
            stall_evals, seed):
    """Swap local search with exact rank-1 delta evaluation (stage 2).

    Stops on ``tol``, on ``max_evals``, or after ``stall_evals`` evaluations
    with no relative improvement in the max absolute residual. The stall exit
    matters because the reachable residual has a floor around 0.5/sqrt(n):
    below roughly n = 5000 at K = 10 a 0.005 target is simply unreachable,
    and without this the solver would burn the whole budget finding out.
    """
    np.random.seed(seed)
    n, Ky = Zy.shape
    Kt = Zt.shape[1]

    inv = np.empty(n, dtype=np.int64)
    for i in range(n):
        inv[pi[i]] = i

    f = 0.0
    for a in range(Kt):
        for b in range(Ky):
            f += R[a, b] * R[a, b]

    av = np.empty(Kt)
    bv = np.empty(Ky)
    n_nb = nbrs.shape[1]
    use_fine = n_nb > 0
    evals = 0
    best_max_abs = np.inf
    evals_since_gain = 0

    while evals < max_evals:
        block = check_every
        if max_evals - evals < block:
            block = max_evals - evals
        for _ in range(block):
            evals += 1
            i = np.random.randint(0, n)
            if use_fine and np.random.random() < p_fine:
                j = inv[nbrs[pi[i], np.random.randint(0, n_nb)]]
            else:
                j = np.random.randint(0, n)
            if i == j:
                continue

            pii = pi[i]
            pij = pi[j]
            aa = 0.0
            for a in range(Kt):
                av[a] = Zt[i, a] - Zt[j, a]
                aa += av[a] * av[a]
            bb = 0.0
            for b in range(Ky):
                bv[b] = Zy[pij, b] - Zy[pii, b]
                bb += bv[b] * bv[b]

            arb = 0.0
            for a in range(Kt):
                acc = 0.0
                for b in range(Ky):
                    acc += R[a, b] * bv[b]
                arb += av[a] * acc
            delta = (-2.0 / n) * arb + aa * bb / (n * n)

            if delta < 0.0:
                for a in range(Kt):
                    ava = av[a]
                    for b in range(Ky):
                        R[a, b] -= ava * bv[b] / n
                f += delta
                pi[i] = pij
                pi[j] = pii
                inv[pij] = i
                inv[pii] = j

        max_abs = 0.0
        for a in range(Kt):
            for b in range(Ky):
                v = R[a, b] if R[a, b] >= 0.0 else -R[a, b]
                if v > max_abs:
                    max_abs = v
        if max_abs < tol:
            break

        # Relative-gain test: greedy descent makes the objective monotone,
        # but max_abs can tick up while f falls, so compare against the best
        # seen rather than the previous window.
        if max_abs < best_max_abs * (1.0 - 1.0e-3):
            best_max_abs = max_abs
            evals_since_gain = 0
        else:
            evals_since_gain += block
            if evals_since_gain >= stall_evals:
                break

    return pi, evals


def _standardize(X):
    """Center and scale columns; constant columns become all-zero.

    Matches the existing mating code's handling of degenerate phenotypes
    (a constant column carries no information a permutation can act on).
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    Z = X - X.mean(axis=0)
    sd = Z.std(axis=0)
    nz = sd > 0
    Z[:, nz] /= sd[nz]
    Z[:, ~nz] = 0.0
    return Z


def _neighbor_lists(Z, k):
    """k nearest neighbors of each row of Z, self excluded.

    Queried with ``workers=-1``: at K ~ 10 the tree degrades toward a linear
    scan, so the query dominates total solve time when run single-threaded
    (measured 93 s of a 99 s solve at n = 1e5, versus 14 s parallel).

    Exact neighbors are worth that cost. Approximate substitutes fail here:
    neighbors taken from sorted random 1-D projections cost 0.3 s but sit a
    mean 4.04 away from their source row, against 1.05 for exact neighbors
    and 4.33 for a random row, i.e. barely better than random. Fine moves
    need genuinely small ``||b||`` to overcome the swap penalty term.
    """
    n = Z.shape[0]
    k = min(k, n - 1)
    if k <= 0:
        return np.zeros((n, 0), dtype=np.int64)
    from scipy.spatial import cKDTree
    _, idx = cKDTree(Z).query(Z, k=k + 1, workers=-1)
    return np.ascontiguousarray(idx[:, 1:], dtype=np.int64)


def solve_cross_correlation(Y, Z, R, tol=0.005, max_evals=None,
                            stall_evals=None, m=64, n_neighbors=10,
                            p_fine=0.5, check_every=1000, seed=None,
                            warn_infeasible=True):
    """Permute the rows of ``Y`` to induce cross-correlation ``R`` with ``Z``.

    Finds ``perm`` minimizing ``||Y[perm]' Z / n - R||_F``, i.e. the
    permutation making the empirical cross-correlation between the two mate
    groups match the target.

    Parameters
    ----------
    Y : np.ndarray
        (n, K) phenotypes of the mobile group (permuted). Standardized
        internally; raw values are fine.
    Z : np.ndarray
        (n, K) phenotypes of the fixed group.
    R : np.ndarray
        (K, K) target cross-correlation. ``R[i, j]`` is the desired
        correlation between component i in the ``Y`` group and component j
        in the ``Z`` group.
    tol : float
        Convergence tolerance, max absolute entrywise error on the
        correlation scale.
    max_evals : int, optional
        Swap evaluation budget. Defaults to ``max(100 * n, 20_000_000)``.
        The total work needed is set by how far the residual must fall, not
        by ``n``: measured instances reaching a 0.005 target took 6-9 million
        evaluations whether n was 5,000 or 100,000 (so 1,600 evaluations per
        row at the low end and 60 at the high end). A budget proportional to
        ``n`` therefore starves small instances, which is why this is
        effectively a constant with a floor for very large ``n``.
    stall_evals : int, optional
        Give up after this many consecutive evaluations without relative
        improvement in the max absolute residual. Defaults to
        ``max(20 * n, 1_000_000)``. The attainable residual has a floor near
        ``0.5 / sqrt(n)`` — at K = 10 that is roughly 0.016 for n = 1,000 and
        0.011 for n = 2,000, both above the default ``tol`` — so a target
        below the floor can never be met, and this is what stops the solver
        from spending the full ``max_evals`` budget rediscovering that on
        every call.
    m : int
        Candidate sample size per greedy construction step.
    n_neighbors : int
        Neighbor-list size for fine swap proposals. 0 disables them.
    p_fine : float
        Probability a proposal is drawn from the neighbor lists.
    check_every : int
        Evaluations between convergence checks.
    seed : int, optional
        Seed for reproducibility. ``None`` draws one nondeterministically.
    warn_infeasible : bool
        Warn when the target is not attainable by any permutation because
        the implied joint correlation matrix is not positive definite.

    Returns
    -------
    MatchResult
    """
    Y = np.asarray(Y, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    if Y.ndim != 2 or Z.ndim != 2:
        raise ValueError("Y and Z must be 2-D arrays")
    if Y.shape[0] != Z.shape[0]:
        raise ValueError(
            f"Y and Z must have equal row counts; got {Y.shape[0]} and "
            f"{Z.shape[0]}")
    if R.shape != (Y.shape[1], Z.shape[1]):
        raise ValueError(
            f"R shape {R.shape} does not match (Y columns, Z columns) = "
            f"({Y.shape[1]}, {Z.shape[1]})")

    n = Y.shape[0]
    Zy = _standardize(Y)
    Zt = _standardize(Z)
    # Internally the fixed group is the row index of the statistic, so the
    # target transposes: we drive Zt' Zy[perm] / n toward C = R'.
    C = np.ascontiguousarray(R.T)

    if warn_infeasible and n > 1:
        Cy = (Zy.T @ Zy) / n
        Ct = (Zt.T @ Zt) / n
        omega = np.block([[Cy, C.T], [C, Ct]])
        min_eig = float(np.linalg.eigvalsh(omega)[0])
        if min_eig <= 0:
            warnings.warn(
                "target cross-correlation is infeasible: the implied joint "
                f"correlation matrix is not positive definite (min eigenvalue "
                f"{min_eig:.3e}); solving for the closest attainable "
                "cross-correlation instead",
                stacklevel=2)

    if n < 2:
        perm = np.arange(n, dtype=np.int64)
        resid = R - (Zy[perm].T @ Zt) / max(n, 1)
        return MatchResult(perm, resid, float(np.max(np.abs(resid))),
                           float(np.linalg.norm(resid)), 0, False)

    if max_evals is None:
        max_evals = max(100 * n, 20_000_000)
    if stall_evals is None:
        stall_evals = max(20 * n, 1_000_000)
    ss = np.random.SeedSequence(seed)
    rng = np.random.default_rng(ss)
    # numba's RNG is separate from numpy's, so kernels get explicit seeds
    k_seeds = ss.generate_state(2).astype(np.int64) % (2 ** 31 - 1)

    order = rng.permutation(n).astype(np.int64)
    perm = _construct(Zy, Zt, C, order, int(m), int(k_seeds[0]))

    R_int = np.ascontiguousarray(C - (Zt.T @ Zy[perm]) / n)
    if np.max(np.abs(R_int)) >= tol and max_evals > 0:
        nbrs = (_neighbor_lists(Zy, n_neighbors) if n_neighbors > 0
                else np.zeros((n, 0), dtype=np.int64))
        perm, evals = _polish(Zy, Zt, C, perm, R_int, float(tol),
                              int(max_evals), int(check_every), float(p_fine),
                              nbrs, int(stall_evals), int(k_seeds[1]))
    else:
        evals = 0

    # Recompute exactly from the permutation, in the caller's orientation
    resid = R - (Zy[perm].T @ Zt) / n
    max_abs = float(np.max(np.abs(resid)))
    return MatchResult(perm=perm, residual=resid, max_abs_residual=max_abs,
                       frobenius_residual=float(np.linalg.norm(resid)),
                       evals=int(evals), converged=bool(max_abs < tol))
