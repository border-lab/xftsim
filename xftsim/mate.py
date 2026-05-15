"""
New mate assignment and mating regimes for the refactored simulation loop.

MateAssignment: dataclass linking offspring to parents by index.
RandomMating: shuffles and pairs individuals to produce offspring.
LinearAssortativeMating: rank-order pairing on a phenotypic composite.
GeneralAssortativeMating: arbitrary K x K cross-mate correlation via QAP (Hexaly).
BatchedMating: wraps any mating regime, splitting individuals into batches.
"""
from __future__ import annotations

import math

import numpy as np
from dataclasses import dataclass
from typing import Dict

from xftsim.struct import SampleMeta, PhenotypeArray


@dataclass
class MateAssignment:
    """
    Links offspring to parents via integer indices into the parent generation.

    Parameters
    ----------
    offspring_samples : SampleMeta
        Metadata for the offspring (unique iids, fids, sex, generation).
    maternal_idx : np.ndarray
        (n_offspring,) indices into the parent generation for mothers.
    paternal_idx : np.ndarray
        (n_offspring,) indices into the parent generation for fathers.
    """
    offspring_samples: SampleMeta
    maternal_idx: np.ndarray
    paternal_idx: np.ndarray

    def __post_init__(self) -> None:
        self.maternal_idx = np.asarray(self.maternal_idx, dtype=np.int64)
        self.paternal_idx = np.asarray(self.paternal_idx, dtype=np.int64)
        n = self.offspring_samples.n
        if len(self.maternal_idx) != n:
            raise ValueError(
                f"maternal_idx length {len(self.maternal_idx)} != offspring n {n}"
            )
        if len(self.paternal_idx) != n:
            raise ValueError(
                f"paternal_idx length {len(self.paternal_idx)} != offspring n {n}"
            )
        if n > 0:
            if np.any(self.maternal_idx < 0):
                raise ValueError("maternal_idx contains negative indices")
            if np.any(self.paternal_idx < 0):
                raise ValueError("paternal_idx contains negative indices")

    @property
    def n_offspring(self) -> int:
        """Number of offspring in this assignment."""
        return self.offspring_samples.n

    def __repr__(self) -> str:
        return (f"MateAssignment(n_offspring={self.n_offspring}, "
                f"generation={self.offspring_samples.generation})")


class RandomMating:
    """Random mating: shuffle individuals, pair them up, produce offspring.

    Individuals are separated by sex (0=female, 1=male), each group is
    shuffled, and min(n_female, n_male) pairs are formed. Each pair
    produces ``offspring_per_pair`` offspring with sequential IIDs,
    pair-based FIDs, and alternating sex.

    Parameters
    ----------
    offspring_per_pair : int
        Number of offspring per mating pair. Default 2.

    Examples
    --------
    >>> from xftsim.struct import SampleMeta
    >>> import numpy as np
    >>> samples = SampleMeta(iid=np.arange(10))
    >>> mating = RandomMating(offspring_per_pair=2)
    >>> assignment = mating.mate(samples, rng=np.random.RandomState(0))
    >>> assignment.n_offspring
    10
    """

    def __init__(self, offspring_per_pair: int = 2) -> None:
        if offspring_per_pair < 1:
            raise ValueError("offspring_per_pair must be >= 1")
        self.offspring_per_pair: int = offspring_per_pair

    def mate(self, samples: SampleMeta,
             rng: np.random.RandomState | None = None,
             phenotypes: PhenotypeArray | None = None) -> MateAssignment:
        """
        Produce a mate assignment from the current generation.

        Algorithm:
        - Separate individuals by sex (0=female, 1=male).
        - Shuffle each group independently.
        - Pair up: min(n_female, n_male) pairs.
        - Each pair produces offspring_per_pair offspring.
        - Offspring get sequential iids, pair-based fids, alternating sex.

        Parameters
        ----------
        samples : SampleMeta
            Current generation's sample metadata.
        rng : np.random.RandomState, optional
            Random state for reproducibility.
        phenotypes : PhenotypeArray, optional
            Ignored by RandomMating (accepted for interface compatibility).

        Returns
        -------
        MateAssignment
        """
        if rng is None:
            rng = np.random.RandomState()

        female_idx = np.where(samples.sex == 0)[0]
        male_idx = np.where(samples.sex == 1)[0]

        if len(female_idx) == 0 or len(male_idx) == 0:
            raise ValueError("Need at least one female and one male for mating")

        # Shuffle each sex group
        rng.shuffle(female_idx)
        rng.shuffle(male_idx)

        # Number of pairs = min of the two groups
        n_pairs = min(len(female_idx), len(male_idx))
        mothers = female_idx[:n_pairs]
        fathers = male_idx[:n_pairs]

        opp = self.offspring_per_pair
        n_offspring = n_pairs * opp

        # Expand: each pair produces opp offspring
        maternal_idx = np.repeat(mothers, opp)
        paternal_idx = np.repeat(fathers, opp)

        # Offspring metadata
        iid = np.arange(n_offspring, dtype=np.int64)
        fid = np.repeat(np.arange(n_pairs, dtype=np.int64), opp)
        # Alternate sex within each family
        sex_pattern = np.tile(np.arange(opp, dtype=np.int64) % 2, n_pairs)
        generation = samples.generation + 1

        offspring_samples = SampleMeta(
            iid=iid, fid=fid, sex=sex_pattern, generation=generation,
        )

        return MateAssignment(
            offspring_samples=offspring_samples,
            maternal_idx=maternal_idx,
            paternal_idx=paternal_idx,
        )

    def __repr__(self) -> str:
        return f"RandomMating(offspring_per_pair={self.offspring_per_pair})"


class LinearAssortativeMating:
    """Assortative mating via rank-order pairing on a phenotypic composite.

    Algorithm:

    1. Standardize each component in ``component_names`` to mean 0, sd 1.
    2. Compute composite = average of standardized components.
    3. Mating score = ``sqrt(|r|) * composite + sqrt(1-|r|) * noise``.
       If ``r < 0``, negate the composite for one sex (disassortative).
    4. Sort each sex by mating score, pair in rank order.

    Parameters
    ----------
    component_names : list[str]
        Phenotype component names to use for assortment.
    r : float
        Target spousal correlation. -1 < r < 1. 0 = random mating.
    offspring_per_pair : int
        Number of offspring per mating pair.
    """

    def __init__(self, component_names: list[str], r: float = 0.0,
                 offspring_per_pair: int = 2) -> None:
        if not -1 < r < 1:
            raise ValueError(f"r must be in (-1, 1), got {r}")
        if offspring_per_pair < 1:
            raise ValueError("offspring_per_pair must be >= 1")
        self.component_names: list[str] = list(component_names)
        self.r: float = float(r)
        self.offspring_per_pair: int = offspring_per_pair

    def mate(self, samples: SampleMeta,
             rng: np.random.RandomState | None = None,
             phenotypes: PhenotypeArray | None = None) -> MateAssignment:
        """Produce a mate assignment with phenotypic assortment.

        Falls back to random mating if ``r == 0`` or ``phenotypes`` is None.

        Parameters
        ----------
        samples : SampleMeta
            Current generation's sample metadata.
        rng : np.random.RandomState, optional
            Random state for reproducibility.
        phenotypes : PhenotypeArray, optional
            Current phenotypes (needed for assortment scoring).

        Returns
        -------
        MateAssignment
        """
        if rng is None:
            rng = np.random.RandomState()

        # Fall back to random if no assortment needed
        if self.r == 0.0 or phenotypes is None:
            return RandomMating(self.offspring_per_pair).mate(samples, rng=rng)

        female_idx = np.where(samples.sex == 0)[0]
        male_idx = np.where(samples.sex == 1)[0]

        if len(female_idx) == 0 or len(male_idx) == 0:
            raise ValueError("Need at least one female and one male for mating")

        # Compute sum of standardized phenotype components (NOT mean).
        # This matches the legacy LinearAssortativeMatingRegime behavior.
        n_total = samples.n
        standardized_cols = []
        for name in self.component_names:
            if name in phenotypes:
                vals = phenotypes[name].copy().astype(np.float64)
                sd = vals.std()
                if sd > 0:
                    vals = (vals - vals.mean()) / sd
                standardized_cols.append(vals)

        K = len(standardized_cols)
        if K == 0:
            return RandomMating(self.offspring_per_pair).mate(samples, rng=rng)

        # Sum of standardized traits (not mean)
        trait_sum = np.sum(standardized_cols, axis=0)

        # Compute latent correlation R = K * |r|, matching legacy:
        #   cross_cov = K^2 * r
        #   R = K^2 * r / sqrt(sum(within_cov_f) * sum(within_cov_m))
        # At gen 0 with uncorrelated traits: sum(within_cov) = K, so R = K*r.
        # At later gens, within-person correlations adapt R automatically.
        # We compute R from the actual within-person covariance to handle
        # the case where traits become correlated over generations.
        if K > 1:
            data = np.column_stack(standardized_cols)
            # Separate by sex for within-cov computation
            f_data = data[np.where(samples.sex == 0)[0]]
            m_data = data[np.where(samples.sex == 1)[0]]
            within_cov_f = np.cov(f_data, rowvar=False)
            within_cov_m = np.cov(m_data, rowvar=False)
            sum_cov_f = np.sum(within_cov_f)
            sum_cov_m = np.sum(within_cov_m)
            denom = np.sqrt(sum_cov_f * sum_cov_m)
            if denom > 1e-15:
                R = K * K * abs(self.r) / denom
            else:
                R = abs(self.r)
        else:
            R = abs(self.r)

        R = min(R, 0.999)  # clamp to valid range

        # Mating score: sqrt(R) * sum + sqrt(1-R) * noise
        # Noise scaled to std of trait sum (matching legacy)
        noise = rng.normal(0, trait_sum.std(), size=n_total)
        scores = np.sqrt(R) * trait_sum + np.sqrt(1.0 - R) * noise

        # Disassortative: negate scores for one sex
        if self.r < 0:
            scores[male_idx] *= -1

        # Sort each sex by score, pair in rank order
        female_order = female_idx[np.argsort(scores[female_idx])]
        male_order = male_idx[np.argsort(scores[male_idx])]

        n_pairs = min(len(female_order), len(male_order))
        mothers = female_order[:n_pairs]
        fathers = male_order[:n_pairs]

        opp = self.offspring_per_pair
        n_offspring = n_pairs * opp

        maternal_idx = np.repeat(mothers, opp)
        paternal_idx = np.repeat(fathers, opp)

        iid = np.arange(n_offspring, dtype=np.int64)
        fid = np.repeat(np.arange(n_pairs, dtype=np.int64), opp)
        sex_pattern = np.tile(np.arange(opp, dtype=np.int64) % 2, n_pairs)
        generation = samples.generation + 1

        offspring_samples = SampleMeta(
            iid=iid, fid=fid, sex=sex_pattern, generation=generation,
        )

        return MateAssignment(
            offspring_samples=offspring_samples,
            maternal_idx=maternal_idx,
            paternal_idx=paternal_idx,
        )

    def __repr__(self) -> str:
        return (f"LinearAssortativeMating(components={self.component_names}, "
                f"r={self.r}, offspring_per_pair={self.offspring_per_pair})")


def _solve_qap_hexaly(Y: np.ndarray, Z: np.ndarray, R: np.ndarray,
                       nb_threads: int = 4, time_limit: int = 120,
                       tolerance: float = 1e-5, verbosity: int = 1,
                       time_between_displays: int = 15,
                       termination_interval: int = 15) -> np.ndarray:
    """Solve the Quadratic Assignment Problem using Hexaly Optimizer.

    Finds a permutation P* of females that minimizes
    ``||Y'[P*] Z / n  -  R||_F`` where Y, Z are standardized phenotype
    matrices and R is the target cross-mate correlation.

    Parameters
    ----------
    Y : np.ndarray
        (n, K) standardized phenotypes for the first mate group.
    Z : np.ndarray
        (n, K) standardized phenotypes for the second mate group.
    R : np.ndarray
        (K, K) target cross-correlation matrix.
    nb_threads : int
        Number of solver threads.
    time_limit : int
        Maximum solve time in seconds.
    tolerance : float
        Objective threshold for early termination.
    verbosity : int
        Hexaly output verbosity.
    time_between_displays : int
        Seconds between Hexaly status lines.
    termination_interval : int
        Stop if no improvement for this many seconds.

    Returns
    -------
    np.ndarray
        (n,) permutation array mapping female indices.
    """
    import hexaly.optimizer

    n = Y.shape[0]

    # Initial value: sort-of-linear heuristic (sort by row means)
    init_perm = np.argsort(Y.mean(axis=1))[np.argsort(np.argsort(Z.mean(axis=1)))]

    const = np.trace(R @ R.T)
    # Gram matrices
    YY = Y @ Y.T / n  # "flow"
    ZZ = Z @ Z.T / n  # "distance"
    W = Y @ R @ Z.T / n  # "cost"

    class _TerminateSolver:
        def __init__(self, interval: int):
            self.last_best_value = np.inf
            self.last_best_running_time = 0.0
            self.interval = interval

        def callback(self, optimizer, cb_type):
            stats = optimizer.statistics
            obj = optimizer.model.objectives[0]
            if obj.value < self.last_best_value:
                self.last_best_running_time = stats.running_time
                self.last_best_value = obj.value
            if stats.running_time - self.last_best_running_time > self.interval:
                optimizer.stop()

    with hexaly.optimizer.HexalyOptimizer() as optimizer:
        cb = _TerminateSolver(int(termination_interval))
        optimizer.add_callback(hexaly.optimizer.HxCallbackType.TIME_TICKED,
                               cb.callback)
        optimizer.param.time_limit = int(time_limit)
        optimizer.param.nb_threads = int(nb_threads)
        optimizer.param.verbosity = int(verbosity)
        optimizer.param.time_between_displays = int(time_between_displays)

        model = optimizer.model
        array_YY = model.array(model.array(YY[i, :]) for i in range(n))
        array_W = model.array(model.array(W[i, :]) for i in range(n))

        # Decision variable: permutation as a list
        p = model.list(n)
        model.constraint(model.eq(model.count(p), n))

        # Objective: sqrt( sum_ij YY[P[i],P[j]]*ZZ[i,j] - 2*sum_i W[P[i],i] + const )
        qobj = model.sum(
            model.at(array_YY, p[i], p[j]) * ZZ[i, j]
            for j in range(n) for i in range(n))
        lobj = model.sum(model.at(array_W, p[i], i) for i in range(n))
        obj = (qobj - 2 * lobj + const) ** 0.5
        model.minimize(obj)
        model.close()

        # Seed with heuristic
        p.value.clear()
        for pp in init_perm:
            p.value.add(int(pp))

        optimizer.param.set_objective_threshold(0, tolerance)
        optimizer.solve()

        return np.array([p.value.get(i) for i in range(n)])


class GeneralAssortativeMating:
    """Assortative mating with an arbitrary K x K cross-mate correlation target.

    Uses the Hexaly Optimizer to solve the Quadratic Assignment Problem:
    find a permutation P* of one sex that minimizes
    ``||Y'[P*] Z / n  -  Omega||_F`` where Omega is the target cross-mate
    cross-trait correlation matrix.

    Parameters
    ----------
    component_names : list[str]
        Phenotype component names (keys in PhenotypeArray) to use.
        Order must match the rows/columns of ``cross_corr``.
    cross_corr : np.ndarray
        (K, K) target cross-mate correlation matrix.
        ``cross_corr[i, j]`` is the desired correlation between component i
        in females and component j in males.
    offspring_per_pair : int
        Number of offspring per mating pair.
    solver_params : dict, optional
        Hexaly solver parameters. Keys: nb_threads, time_limit, tolerance,
        verbosity, time_between_displays, termination_interval.
    """

    def __init__(self, component_names: list[str],
                 cross_corr: np.ndarray,
                 offspring_per_pair: int = 2,
                 solver_params: Dict[str, int | float] | None = None) -> None:
        import hexaly.optimizer  # noqa: F401 — hard error if not installed

        if offspring_per_pair < 1:
            raise ValueError("offspring_per_pair must be >= 1")

        self.component_names: list[str] = list(component_names)
        K = len(self.component_names)
        self.cross_corr = np.asarray(cross_corr, dtype=np.float64)
        if self.cross_corr.shape != (K, K):
            raise ValueError(
                f"cross_corr shape {self.cross_corr.shape} does not match "
                f"{K} components — expected ({K}, {K})")
        self.offspring_per_pair: int = offspring_per_pair
        self.solver_params: dict = dict(
            nb_threads=4,
            time_limit=120,
            tolerance=1e-5,
            verbosity=1,
            time_between_displays=15,
            termination_interval=15,
        )
        if solver_params is not None:
            self.solver_params.update(solver_params)

    def mate(self, samples: SampleMeta,
             rng: np.random.RandomState | None = None,
             phenotypes: PhenotypeArray | None = None) -> MateAssignment:
        """Produce a mate assignment achieving the target cross-mate correlations.

        Parameters
        ----------
        samples : SampleMeta
            Current generation's sample metadata.
        rng : np.random.RandomState, optional
            Random state for reproducibility (used only for offspring metadata).
        phenotypes : PhenotypeArray
            Current phenotypes. Must contain all ``component_names``.

        Returns
        -------
        MateAssignment
        """
        if phenotypes is None:
            raise ValueError(
                "GeneralAssortativeMating requires phenotypes")

        if rng is None:
            rng = np.random.RandomState()

        female_idx = np.where(samples.sex == 0)[0]
        male_idx = np.where(samples.sex == 1)[0]

        if len(female_idx) == 0 or len(male_idx) == 0:
            raise ValueError("Need at least one female and one male for mating")

        # Balance sexes
        n_pairs = min(len(female_idx), len(male_idx))
        rng.shuffle(female_idx)
        rng.shuffle(male_idx)
        female_idx = female_idx[:n_pairs]
        male_idx = male_idx[:n_pairs]

        # Build (n_pairs, K) phenotype matrices
        K = len(self.component_names)
        Y = np.empty((n_pairs, K), dtype=np.float64)
        Z = np.empty((n_pairs, K), dtype=np.float64)
        for k, name in enumerate(self.component_names):
            vals = phenotypes[name].astype(np.float64)
            Y[:, k] = vals[female_idx]
            Z[:, k] = vals[male_idx]

        # Standardize columns
        for k in range(K):
            for arr in (Y, Z):
                mu = arr[:, k].mean()
                sd = arr[:, k].std()
                if sd > 0:
                    arr[:, k] = (arr[:, k] - mu) / sd
                else:
                    arr[:, k] = 0.0

        # Solve QAP — returns permutation of female indices
        perm = _solve_qap_hexaly(Y, Z, self.cross_corr, **self.solver_params)

        # Apply permutation to females; males stay in place
        mothers = female_idx[perm]
        fathers = male_idx

        opp = self.offspring_per_pair
        n_offspring = n_pairs * opp

        maternal_idx = np.repeat(mothers, opp)
        paternal_idx = np.repeat(fathers, opp)

        iid = np.arange(n_offspring, dtype=np.int64)
        fid = np.repeat(np.arange(n_pairs, dtype=np.int64), opp)
        sex_pattern = np.tile(np.arange(opp, dtype=np.int64) % 2, n_pairs)
        generation = samples.generation + 1

        offspring_samples = SampleMeta(
            iid=iid, fid=fid, sex=sex_pattern, generation=generation,
        )

        return MateAssignment(
            offspring_samples=offspring_samples,
            maternal_idx=maternal_idx,
            paternal_idx=paternal_idx,
        )

    def __repr__(self) -> str:
        K = len(self.component_names)
        return (f"GeneralAssortativeMating(components={self.component_names}, "
                f"cross_corr=({K}x{K}), "
                f"offspring_per_pair={self.offspring_per_pair})")


class BatchedMating:
    """Wraps any mating regime, splitting individuals into batches.

    Randomly partitions individuals into batches of at most
    ``max_batch_size`` individuals, runs the inner regime on each batch
    independently, then merges the resulting mate assignments.

    This is essential for GeneralAssortativeMating at large n, where the
    QAP solver scales quadratically. For example, n=8000 with
    max_batch_size=1000 yields 8 independent QAP solves of 500 pairs
    each, rather than one solve of 4000 pairs.

    Parameters
    ----------
    regime
        Any mating regime with a ``.mate(samples, rng, phenotypes)`` method.
    max_batch_size : int
        Maximum number of *individuals* (not pairs) per batch.
    """

    def __init__(self, regime, max_batch_size: int = 1000) -> None:
        self.regime = regime
        self.max_batch_size: int = max_batch_size

    def mate(self, samples: SampleMeta,
             rng: np.random.RandomState | None = None,
             phenotypes: PhenotypeArray | None = None) -> MateAssignment:
        if rng is None:
            rng = np.random.RandomState()

        n = samples.n
        num_batches = math.ceil(n / self.max_batch_size)

        # Random partition of all individuals
        perm = rng.permutation(n)
        batch_indices = np.array_split(perm, num_batches)

        all_maternal = []
        all_paternal = []
        all_n_offspring = 0
        all_fid_offset = 0

        for bi, batch_idx in enumerate(batch_indices):
            batch_idx = np.sort(batch_idx)

            batch_samples = samples.subset(batch_idx)
            batch_pheno = phenotypes.subset(batch_idx) if phenotypes is not None else None

            assignment = self.regime.mate(batch_samples, rng=rng,
                                         phenotypes=batch_pheno)

            # Map batch-local parent indices back to global indices
            all_maternal.append(batch_idx[assignment.maternal_idx])
            all_paternal.append(batch_idx[assignment.paternal_idx])
            all_n_offspring += assignment.n_offspring

        maternal_idx = np.concatenate(all_maternal)
        paternal_idx = np.concatenate(all_paternal)

        # Build offspring metadata
        opp = getattr(self.regime, 'offspring_per_pair', 2)
        n_pairs = all_n_offspring // opp
        iid = np.arange(all_n_offspring, dtype=np.int64)
        fid = np.repeat(np.arange(n_pairs, dtype=np.int64), opp)
        sex_pattern = np.tile(np.arange(opp, dtype=np.int64) % 2, n_pairs)
        generation = samples.generation + 1

        offspring_samples = SampleMeta(
            iid=iid, fid=fid, sex=sex_pattern, generation=generation,
        )

        return MateAssignment(
            offspring_samples=offspring_samples,
            maternal_idx=maternal_idx,
            paternal_idx=paternal_idx,
        )

    def __repr__(self) -> str:
        return (f"BatchedMating(regime={self.regime!r}, "
                f"max_batch_size={self.max_batch_size})")
