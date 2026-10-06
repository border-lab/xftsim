import warnings
import numpy as np
import pandas as pd
import dask.array as da
import nptyping as npt
import xarray as xr
from nptyping import NDArray, Int8, Int64, Float64, Bool, Shape
from typing import Any, Hashable, List, Iterable, Union
from functools import cached_property
import numba as nb
import math
import nptyping as npt

import xftsim as xft


class RecombinationMap:
    """A class to represent a diploid recombination map.

    Parameters
    ----------
    p : float or numpy.ndarray, optional
        Probabilities, either a float or a numpy.ndarray, default is None. A single value
        results in an exchangeable map, an array corresponds to probabilities of recombination
        between specified loci.
    m : int, optional
        Number of variants. Required if p is a float.
    vid : numpy.ndarray, optional
        Variant IDs, default is None.
    chrom : numpy.ndarray, optional
        Chromosomes, default is None.
    pos_bp : numpy.ndarray, optional
        Per-variant base-pair positions, default is None. When provided,
        any locus j where ``pos_bp[j] == pos_bp[j-1]`` has its
        recombination probability forced to 0: crossover cannot occur
        between two mutations at the same physical site (e.g.,
        multi-allelic sites where two ALT entries share a position, or
        msprime-discretized positions that collide). Without this guard,
        the GRG-native meiosis path's interval-based mutation filter
        (``bisect_left`` on positions) silently groups all same-position
        variants into whichever segment claims the position, producing
        wrong-hap assignments at the few duplicate-position cells.
    """

    def __init__(self,
                 p=None,
                 m: int = None,
                 vid: NDArray[Shape["*"], Any] = None,
                 chrom: NDArray[Shape["*"], Int64] = None,
                 pos_bp: NDArray[Shape["*"], Int64] = None,
                 ):
        if m is not None:
            self.m = m
        elif vid is not None:
            self.m = len(vid)
        else:
            raise ValueError("Must provide m or vid")

        if vid is None:
            vid = np.arange(self.m)
        if chrom is None:
            chrom = np.zeros(self.m, dtype=np.int64)

        self.vid = vid
        self.chrom = chrom
        self.pos_bp = pos_bp

        # Chromosome boundaries (where recombination probability is 0.5)
        self._chrom_boundary = np.concatenate(
            [[0], 1 + np.where(chrom[1:] != chrom[:-1])[0]])

        if isinstance(p, float):
            assert 0 <= p <= 1, "Provide a valid probability"
            self._probabilities = np.ones(self.m) * p
        elif isinstance(p, np.ndarray):
            assert p.shape[0] == self.m, "p and m must agree in length"
            self._probabilities = p.copy()
        else:
            # Default to 0.5 (free recombination)
            self._probabilities = np.ones(self.m) * 0.5

        # Force chromosome boundaries to have 0.5 probability
        self._probabilities[self._chrom_boundary] = 0.5

        # Force p=0 wherever consecutive variants share a bp position.
        # A crossover physically cannot land between two mutations at the
        # same site, and the GRG-native meiosis path cannot represent it
        # via interval-based segments (bisect_left collapses duplicates).
        # Forcing p=0 here makes the per-locus phase vector compatible
        # with the segment-based algorithm: same-position variants always
        # inherit from the same source hap. Only within a chromosome: the
        # first variant of a chromosome keeps 0.5 even if its pos_bp equals
        # the previous chromosome's last (e.g. placeholder positions).
        if pos_bp is not None:
            pos_bp_arr = np.asarray(pos_bp)
            assert pos_bp_arr.shape[0] == self.m, "pos_bp must have length m"
            same_pos_with_prev = np.concatenate(
                [[False], pos_bp_arr[1:] == pos_bp_arr[:-1]])
            same_pos_with_prev[self._chrom_boundary] = False
            self._probabilities[same_pos_with_prev] = 0.0

    @staticmethod
    def constant_map(m: int, p: float = 0.5,
                     pos_bp: NDArray[Shape["*"], Int64] = None,
                     ) -> "RecombinationMap":
        """
        Create a constant recombination map.

        Parameters
        ----------
        m : int
            Number of variants.
        p : float, optional
            Recombination probability, default is 0.5.
        pos_bp : numpy.ndarray, optional
            Per-variant base-pair positions; forwarded to
            ``RecombinationMap.__init__`` to suppress crossovers between
            same-position variants. See class docstring.

        Returns
        -------
        RecombinationMap
            A constant recombination map.
        """
        return RecombinationMap(p=p, m=m, pos_bp=pos_bp)

    @staticmethod
    def from_haplotypes(haplotypes,
                        p: float = 0.5) -> "RecombinationMap":
        """
        Create a constant recombination map from haplotypes.

        Threads ``haplotypes.variants.pos_bp`` into the constructor when
        available so that same-position variants are properly handled
        (see class docstring). Falls back to None when the haplotype
        object lacks position metadata, which preserves the historical
        behavior.

        Parameters
        ----------
        haplotypes : DenseHaplotypeArray or GraphHaplotypeOperator
            Haplotypes data. Any object with ``.m`` and ``.vid``; if it
            also exposes ``.variants.pos_bp``, that's used.
        p : float, optional
            Probability, default is 0.5.

        Returns
        -------
        RecombinationMap
            A constant recombination map.
        """
        pos_bp = None
        variants = getattr(haplotypes, "variants", None)
        if variants is not None:
            pos_bp = getattr(variants, "pos_bp", None)
        return RecombinationMap(p=p, m=haplotypes.m, vid=haplotypes.vid,
                                pos_bp=pos_bp)

    @staticmethod
    def from_haplotypes_with_cM(haplotypes) -> "RecombinationMap":
        """
        Create a variable recombination map from per-variant genetic positions.

        Converts the genetic distance between adjacent variants into the
        probability that the transmitted haplotype switches parental copies
        between them, using Haldane's map function (crossovers form a Poisson
        process along the genetic map, no interference). For adjacent
        variants ``j-1`` and ``j`` on the same chromosome separated by
        ``d = (pos_cM[j] - pos_cM[j-1]) / 100`` Morgans, the switch happens
        when an odd number of crossovers falls between them::

            p_j = (1 - exp(-2 d)) / 2

        The first variant of each chromosome gets ``p = 0.5`` (independent
        segregation). Variants at the same ``pos_bp`` get ``p = 0`` (see
        class docstring). This is the map to use for real or coalescent-
        simulated sequence data, where adjacent variants are tightly linked;
        a constant map such as ``from_haplotypes(haplotypes, p=0.5)`` treats
        every adjacent pair as unlinked.

        Parameters
        ----------
        haplotypes : DenseHaplotypeArray or GraphHaplotypeOperator
            Haplotypes whose ``variants`` carry ``pos_cM`` (and optionally
            ``chrom`` and ``pos_bp``), with variants ordered by position
            within each chromosome.

        Returns
        -------
        RecombinationMap
            A variable-rate recombination map.

        Raises
        ------
        ValueError
            If ``pos_cM`` is missing, non-finite, or decreases within a
            chromosome.
        """
        variants = haplotypes.variants
        if variants.pos_cM is None:
            raise ValueError(
                "haplotypes.variants.pos_cM is not set; a variable map needs "
                "per-variant genetic positions in centimorgans.")
        pos_cM = np.asarray(variants.pos_cM, dtype=np.float64)
        if not np.all(np.isfinite(pos_cM)):
            raise ValueError("haplotypes.variants.pos_cM contains NaN or inf values.")
        m = pos_cM.shape[0]
        chrom = variants.chrom
        if chrom is None:
            chrom = np.zeros(m, dtype=np.int64)

        same_chrom = chrom[1:] == chrom[:-1]
        d = np.diff(pos_cM) / 100.0
        if np.any(d[same_chrom] < 0):
            raise ValueError(
                "haplotypes.variants.pos_cM decreases within a chromosome; "
                "variants must be ordered by position.")
        # Chromosome boundaries are reset to 0.5 by the constructor; using
        # d = inf there keeps the arithmetic warning-free.
        d = np.where(same_chrom, d, np.inf)
        p = np.empty(m, dtype=np.float64)
        p[0] = 0.5
        p[1:] = -0.5 * np.expm1(-2.0 * d)
        return RecombinationMap(p=p, vid=variants.vid, chrom=chrom,
                                pos_bp=variants.pos_bp)

    def __repr__(self):
        df = pd.DataFrame.from_dict(dict(vid=self.vid,
                                         chrom=self.chrom,
                                         p=self._probabilities))
        return df.__repr__()


def _crossover_model(p):
    """
    Precompute the crossover sampler's inputs from per-locus switch probabilities.

    ``p[j]`` is the probability that the transmitted haplotype switches
    parental copies at locus ``j`` (for ``j = 0``: starts on the second copy).
    Switches at loci with ``p[j] < 0.5`` are generated by a Poisson crossover
    process: locus ``j`` gets rate ``lam[j] = -log(1 - 2 p[j]) / 2``, and it
    switches when an odd number of crossovers lands on it, which happens with
    probability ``(1 - exp(-2 lam[j])) / 2 = p[j]``. Disjoint loci receive
    independent Poisson counts, so the switches are independent Bernoulli(p)
    exactly as in a per-locus draw, but sampling costs
    O(number of crossovers x log m) instead of O(m). Loci with
    ``p[j] >= 0.5`` (chromosome starts, unlinked loci) have no finite rate
    and get an explicit Bernoulli draw.

    When the expected number of crossovers is high enough that the Poisson
    route would cost more than a per-locus pass (e.g. a constant map with
    intermediate p over many loci), every locus with ``p > 0`` gets an
    explicit Bernoulli draw instead. Both representations sample the same
    distribution; the choice depends only on ``p``, so dense and GRG
    meiosis always use the same one.

    Parameters
    ----------
    p : numpy.ndarray[float64]
        Per-locus switch probabilities in [0, 1].

    Returns
    -------
    cumlam : numpy.ndarray[float64]
        Cumulative Poisson rate, ``cumsum(lam)``.
    special_idx : numpy.ndarray[int64]
        Loci drawn as explicit Bernoullis.
    special_p : numpy.ndarray[float64]
        ``p[special_idx]``.
    """
    p = np.asarray(p, dtype=np.float64)
    if not np.all((p >= 0) & (p <= 1)):
        raise ValueError("recombination probabilities must lie in [0, 1]")
    m = p.shape[0]
    special = p >= 0.5
    lam = np.zeros(m, dtype=np.float64)
    lam[~special] = -0.5 * np.log1p(-2.0 * p[~special])
    # Each Poisson event costs a uniform draw plus a binary search over m
    # loci; a per-locus pass costs one uniform draw per locus.
    if lam.sum() * max(np.log2(m), 1.0) > m:
        special = p > 0
        lam[:] = 0.0
    special_idx = np.flatnonzero(special).astype(np.int64)
    return (np.cumsum(lam),
            special_idx,
            np.ascontiguousarray(p[special_idx], dtype=np.float64))


@nb.njit
def _draw_switches(cumlam, special_idx, special_p):
    """
    Draw the loci at which one transmitted haplotype switches parental copies.

    Uses the current (thread-local) numba RNG state; see
    ``_crossover_model`` for the model. Returns a sorted int64 array of the
    distinct loci with an odd number of switch events. A switch at locus 0
    means the haplotype starts on the second parental copy.
    """
    m = cumlam.shape[0]
    total = cumlam[m - 1] if m > 0 else 0.0
    n_xo = np.random.poisson(total) if total > 0.0 else 0
    events = np.empty(n_xo + special_idx.shape[0], dtype=np.int64)
    k = 0
    for _ in range(n_xo):
        # Locus j owns the half-open interval [cumlam[j-1], cumlam[j]);
        # zero-rate loci own empty intervals and are never hit. The redraw
        # guards against u rounding up to exactly `total`.
        j = m
        while j >= m:
            j = np.searchsorted(cumlam, np.random.random() * total, side='right')
        events[k] = j
        k += 1
    for s in range(special_idx.shape[0]):
        if np.random.random() < special_p[s]:
            events[k] = special_idx[s]
            k += 1
    if n_xo == 0:
        # special_idx is sorted and distinct, so the events already are.
        return events[:k]
    events = np.sort(events[:k])

    # Keep loci hit an odd number of times.
    out = np.empty(k, dtype=np.int64)
    c = 0
    i = 0
    while i < k:
        j = i + 1
        while j < k and events[j] == events[i]:
            j += 1
        if (j - i) % 2 == 1:
            out[c] = events[i]
            c += 1
        i = j
    return out[:c]


@nb.njit
def _meiosis_pair_switches_seeded(cumlam, special_idx, special_p, seed):
    """
    Seed numba's RNG and draw (maternal, paternal) switch loci.

    Exists because ``np.random.seed()`` at the Python level does NOT
    propagate to numba's internal RNG state — JIT'd functions like
    ``_draw_switches`` read a separate thread-local numba RNG. To make
    Python callers (e.g. the GRG-native meiosis path) deterministic, the
    seed and both draws must happen inside a single JIT call.

    Matches the dense ``_meiosis_3d`` kernel's per-offspring behavior
    exactly: one seed, then two consecutive ``_draw_switches`` draws against
    that just-seeded stream. So given the same per-offspring seed both the
    dense and GRG meiosis paths sample the same crossovers, guaranteeing
    cross-path determinism on top of within-path determinism.
    """
    np.random.seed(seed)
    mat = _draw_switches(cumlam, special_idx, special_p)
    pat = _draw_switches(cumlam, special_idx, special_p)
    return mat, pat


@nb.njit
def _switches_to_phase(switches, m):
    """Expand sorted switch loci to a per-locus 0/1 phase vector of length m."""
    phase = np.empty(m, dtype=np.int64)
    cur = 0
    s = 0
    for j in range(m):
        if s < switches.shape[0] and switches[s] == j:
            cur ^= 1
            s += 1
        phase[j] = cur
    return phase


def _meiosis_pair_seeded(p, seed):
    """
    Draw (maternal, paternal) per-locus phase vectors for one offspring.

    Convenience wrapper over ``_meiosis_pair_switches_seeded`` that takes
    raw per-locus probabilities and expands the switch loci to phase
    vectors (``phase[j]`` selects parental copy 0 or 1 at locus j). Given
    the same ``seed``, it reproduces the phases used by both meiosis paths.
    """
    cumlam, special_idx, special_p = _crossover_model(p)
    mat, pat = _meiosis_pair_switches_seeded(cumlam, special_idx, special_p, seed)
    m = cumlam.shape[0]
    return _switches_to_phase(mat, m), _switches_to_phase(pat, m)


@nb.njit("int8[:,:,:](int8[:,:,:], int8[:,:,:], int64, int64, float64[:], int64[:], float64[:], int64[:], int64[:], uint32[:])", parallel=True)
def _meiosis_3d(parental_genotypes,
                offspring_genotypes,
                n_offspring,
                m,
                cumlam,
                special_idx,
                special_p,
                maternal_inds,
                paternal_inds,
                seeds,
                ):
    """
    Performs meiosis on 3D genotype arrays.

    Parameters
    ----------
    parental_genotypes : numpy.ndarray[int8]
        3D array of parental genotypes (n_parents, m, 2).
    offspring_genotypes : numpy.ndarray[int8]
        3D array to store offspring genotypes (n_offspring, m, 2).
    n_offspring : int64
        The number of offspring.
    m : int64
        The number of variants.
    cumlam, special_idx, special_p : numpy.ndarray
        Crossover model from ``_crossover_model``.
    maternal_inds : numpy.ndarray[int64]
        An array of maternal parent indices.
    paternal_inds : numpy.ndarray[int64]
        An array of paternal parent indices.
    seeds : numpy.ndarray[uint32]
        Per-offspring RNG seeds. ``seeds[i]`` is applied to numba's
        thread-local RNG before drawing offspring i's maternal/paternal
        crossovers, so the resulting genotypes are deterministic given
        the seed array — independent of how ``nb.prange`` distributes
        offspring across threads. Length must equal ``n_offspring``.

    Returns
    -------
    numpy.ndarray[int8]
        3D array of offspring genotypes.
    """
    for i in nb.prange(n_offspring):
        # Re-seed the current thread's RNG with this offspring's seed.
        # Numba's `np.random.seed()` is per-thread inside `prange`, so this
        # localizes randomness to the offspring rather than the thread —
        # which is what makes the output independent of scheduling.
        np.random.seed(seeds[i])

        # Loci where the maternal / paternal transmitted haplotype switches
        # parental copy (sorted, distinct).
        mat_switches = _draw_switches(cumlam, special_idx, special_p)
        pat_switches = _draw_switches(cumlam, special_idx, special_p)

        mat_idx = maternal_inds[i]
        pat_idx = paternal_inds[i]

        mat_hap = 0
        pat_hap = 0
        ms = 0
        ps = 0
        for j in range(m):
            if ms < mat_switches.shape[0] and mat_switches[ms] == j:
                mat_hap ^= 1
                ms += 1
            if ps < pat_switches.shape[0] and pat_switches[ps] == j:
                pat_hap ^= 1
                ps += 1
            # Offspring's first haplotype comes from mother
            offspring_genotypes[i, j, 0] = parental_genotypes[mat_idx, j, mat_hap]
            # Offspring's second haplotype comes from father
            offspring_genotypes[i, j, 1] = parental_genotypes[pat_idx, j, pat_hap]

    return offspring_genotypes


def _spawn_meiosis_seeds(rng: np.random.RandomState, n: int) -> np.ndarray:
    """Derive ``n`` independent uint32 seeds from ``rng`` via SeedSequence.

    Centralized so the dense and GRG meiosis paths agree on how seeds are
    derived from the simulation's master RNG. Both consume exactly one
    ``rng.randint`` draw before delegating to ``SeedSequence.spawn`` —
    keeping the master rng state advance the same regardless of which
    path runs.
    """
    if rng is None:
        rng = np.random.RandomState()
    entropy = int(rng.randint(0, 2 ** 31 - 1))
    ss = np.random.SeedSequence(entropy)
    children = ss.spawn(n)
    return np.fromiter(
        (c.generate_state(1)[0] for c in children),
        dtype=np.uint32,
        count=n,
    )


def meiosis(parental_haplotypes: xft.struct.DenseHaplotypeArray,
            recombination_map: RecombinationMap,
            maternal_inds: NDArray[Shape["*"], Int64],
            paternal_inds: NDArray[Shape["*"], Int64],
            rng: np.random.RandomState = None,
            ) -> NDArray[Shape["*, *, *"], Int8]:
    """
    Performs meiosis on parental haplotypes.

    Parameters
    ----------
    parental_haplotypes : DenseHaplotypeArray
        Parental haplotype data.
    recombination_map : RecombinationMap
        Recombination probabilities.
    maternal_inds : numpy.ndarray[int64]
        An array of maternal parent indices.
    paternal_inds : numpy.ndarray[int64]
        An array of paternal parent indices.
    rng : numpy.random.RandomState, optional
        Master RNG. Used to derive per-offspring seeds via
        ``SeedSequence.spawn`` so that crossover sampling is deterministic
        given the master state (independent of how ``nb.prange`` schedules
        offspring across threads). If ``None`` (the default), a fresh
        ``RandomState`` is constructed, preserving the historical
        non-deterministic behavior for callers that don't pass an rng.

    Returns
    -------
    numpy.ndarray[int8]
        3D array of offspring genotypes (n_offspring, m, 2).
    """
    parental_genotypes = parental_haplotypes.genotypes
    m = parental_haplotypes.m
    recombination_p = recombination_map._probabilities

    assert parental_genotypes.shape[1] == m, "incompatible dimensions"
    assert paternal_inds.shape[0] == maternal_inds.shape[0], "incompatible dimensions"
    assert np.max(maternal_inds) < parental_genotypes.shape[0], "maternal index out of bounds"
    assert np.max(paternal_inds) < parental_genotypes.shape[0], "paternal index out of bounds"
    assert parental_genotypes.dtype == np.int8, "genotypes must be int8"
    assert recombination_p.dtype == np.float64, "recombination_p must be float64"
    assert recombination_p.shape[0] == m, "recombination map and haplotypes disagree on m"
    assert maternal_inds.dtype == paternal_inds.dtype == np.int64, "indices must be int64"

    n_offspring = maternal_inds.shape[0]
    offspring_genotypes = np.empty((n_offspring, m, 2), dtype=np.int8)
    seeds = _spawn_meiosis_seeds(rng, n_offspring)
    cumlam, special_idx, special_p = _crossover_model(recombination_p)

    return _meiosis_3d(parental_genotypes,
                       offspring_genotypes,
                       n_offspring,
                       m,
                       cumlam,
                       special_idx,
                       special_p,
                       maternal_inds,
                       paternal_inds,
                       seeds)

