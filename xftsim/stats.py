"""
Statistics computed per-generation in a simulation.

Statistic ABC and concrete implementations. Each statistic receives
the phenotype history and any filtered views, and returns a result
stored in GenerationResult.
"""
from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from xftsim.filters import TrioView, SibPairView, FilteredView
from xftsim.struct import PhenotypeArray, StandardizedHaplotypeOperator


@dataclass
class GenerationResult:
    """
    Results from a single generation of simulation.

    Parameters
    ----------
    generation : int
        Generation number.
    statistics : dict
        Name → value mapping of computed statistics.
    """
    generation: int
    statistics: Dict[str, Any] = field(default_factory=dict)


class Statistic(ABC):
    """Abstract base class for per-generation statistics.

    Subclasses implement ``estimate()`` to compute a summary statistic
    from phenotype history and filtered views each generation.
    """

    @abstractmethod
    def estimate(self, phenotype_history: dict[int, PhenotypeArray],
                 filtered_views: dict[str, FilteredView],
                 generation: int,
                 **kwargs: Any) -> Any:
        """
        Compute the statistic for a given generation.

        Parameters
        ----------
        phenotype_history : dict[int, PhenotypeArray]
            Generation → phenotypes mapping.
        filtered_views : dict[str, FilteredView]
            Named filtered views (from filters).
        generation : int
            Current generation number.
        **kwargs
            Additional context. May include:
            - haplotype_history: dict[int, HaplotypeOperator]

        Returns
        -------
        Any
            The computed statistic value.
        """
        ...


class SampleStatistics(Statistic):
    """
    Compute the sample covariance matrix across all phenotype components.

    Returns a dict with 'cov' (k x k matrix), 'var' (diagonal), and 'keys'.

    Parameters
    ----------
    filter_name : str, optional
        If given, compute on the ``phenotypes`` of the filtered view with
        this name (e.g. a ``MetadataFilter`` selecting one population)
        instead of the whole generation.
    """

    def __init__(self, filter_name: str | None = None) -> None:
        self.filter_name = filter_name

    def estimate(self, phenotype_history: dict[int, PhenotypeArray],
                 filtered_views: dict[str, FilteredView],
                 generation: int,
                 **kwargs: Any) -> dict[str, Any] | None:
        if generation not in phenotype_history:
            return None

        if self.filter_name is None:
            pheno = phenotype_history[generation]
        else:
            view = filtered_views.get(self.filter_name)
            pheno = getattr(view, 'phenotypes', None)
            if pheno is None:
                return None
        keys = list(pheno.keys)
        if not keys:
            return {'cov': np.array([[]]), 'var': np.array([]), 'keys': []}

        n = pheno.samples.n
        k = len(keys)
        data = np.column_stack([pheno[key] for key in keys])  # (n, k)
        cov = np.cov(data, rowvar=False)  # (k, k)
        # np.cov returns scalar for k=1
        if cov.ndim == 0:
            cov = cov.reshape(1, 1)

        return {
            'cov': cov,
            'var': np.diag(cov),
            'keys': keys,
        }


class HasemanElstonEstimator(Statistic):
    """
    GRM-based Haseman-Elston regression estimator.

    Estimates genetic covariance (and heritability) using the GRM
    (genomic relationship matrix) computed from standardized genotypes.
    Works with any sample — does not require siblings, trios, or any
    specific family structure. Works at generation 0 (founders).

    The estimator solves:
        cov_g = Y' (K Y - Y) / (tr(K^2) - n)
    where K = G G' / m is the GRM built from per-SNP standardized
    genotypes, and Y is the (n x k) phenotype matrix (standardized).

    This matches the legacy ``haseman_elston()`` function.

    Parameters
    ----------
    phenotype_keys : list[str], optional
        Phenotype names to estimate heritability for. If None,
        uses all phenotype keys that do NOT contain a '.'
        (i.e., top-level phenotypes like 'height', not sub-components
        like 'height.G').
    n_probe : int
        Number of random probes for stochastic trace estimation.
        Set to 0 for deterministic (exact) trace. Default 0.
    """

    def __init__(self, phenotype_keys: list[str] | None = None,
                 n_probe: int = 0) -> None:
        self.phenotype_keys = phenotype_keys
        self.n_probe = n_probe

    def estimate(self, phenotype_history: dict[int, PhenotypeArray],
                 filtered_views: dict[str, FilteredView],
                 generation: int,
                 **kwargs: Any) -> dict[str, dict[str, Any]] | None:
        haplotype_history = kwargs.get('haplotype_history')
        if haplotype_history is None or generation not in haplotype_history:
            return None
        if generation not in phenotype_history:
            return None

        hap = haplotype_history[generation]
        pheno = phenotype_history[generation]

        # Select phenotype keys
        if self.phenotype_keys is not None:
            keys = [k for k in self.phenotype_keys if k in pheno]
        else:
            keys = [k for k in pheno.keys if '.' not in k]
        if not keys:
            return None

        # Wrap raw haplotypes so matvec/rmatvec act on standardized G
        # without materializing the (n x m) standardized matrix.
        S = StandardizedHaplotypeOperator(hap)
        n, m = S.n, S.m

        # Build phenotype matrix (n x k), standardized
        Y = np.column_stack([pheno[k] for k in keys]).astype(np.float64)
        Y_mean = Y.mean(axis=0)
        Y_std = Y.std(axis=0)
        Y_std[Y_std < 1e-15] = 1.0
        Y = (Y - Y_mean) / Y_std

        # K Y = G (G' Y) / m  — computed without forming K or G explicitly
        GtY = S.rmatvec(Y)  # (m, k)
        KY = S.matvec(GtY) / m  # (n, k)

        # tr(K^2)
        if self.n_probe > 0 and n > 500:
            # Stochastic trace estimation via Hutchinson's method
            rng = np.random.RandomState()
            probes = rng.randn(n, self.n_probe)
            # tr(K^2) ≈ (1/l) * tr(P' K^2 P) = (1/l) * ||K P||_F^2
            KP = S.matvec(S.rmatvec(probes)) / m  # (n, n_probe)
            trK2 = np.sum(KP ** 2) / self.n_probe
        else:
            # Deterministic: tr(K^2) = ||G' G||_F^2 / m^2.
            # No cheap operator-only path for this; materialize standardized G.
            G = hap.to_diploid_standardized(scale=True).astype(np.float64)
            GtG = G.T @ G  # (m, m)
            trK2 = np.sum(GtG ** 2) / (m * m)

        denom = trK2 - n
        if abs(denom) < 1e-15:
            return {k: {'h2': np.nan, 'n': n} for k in keys}

        # HE estimate: cov_g = Y' (K Y - Y) / (tr(K^2) - n)
        cov_g = Y.T @ (KY - Y) / denom  # (k, k)

        results = {}
        for i, key in enumerate(keys):
            h2_est = float(cov_g[i, i])
            results[key] = {
                'h2': h2_est,
                'n': int(n),
            }
        # Also store the full genetic covariance matrix
        results['_cov_g'] = cov_g
        results['_keys'] = keys

        return results


class ParentOffspringRegression(Statistic):
    """
    Parent-offspring regression estimator of heritability.

    Regresses offspring phenotype on mid-parent value.
    Under an additive model the slope equals h2.

    Requires a TrioFilter (keyed by ``filter_name``) to be active.

    Parameters
    ----------
    filter_name : str
        Key in ``filtered_views`` that contains a TrioView.
        Default is ``'trio'``.
    """

    def __init__(self, filter_name: str = 'trio') -> None:
        self.filter_name: str = filter_name

    def estimate(self, phenotype_history: dict[int, PhenotypeArray],
                 filtered_views: dict[str, FilteredView],
                 generation: int,
                 **kwargs: Any) -> dict[str, dict[str, Any]] | None:
        view = filtered_views.get(self.filter_name)
        if view is None or not isinstance(view, TrioView):
            return None
        if view.n_trios == 0:
            return None

        results: dict[str, dict[str, Any]] = {}
        keys = list(view.offspring_phenotypes.keys())
        for key in keys:
            if key not in view.mother_phenotypes or key not in view.father_phenotypes:
                continue

            y_off = view.offspring_phenotypes[key]
            y_mom = view.mother_phenotypes[key]
            y_dad = view.father_phenotypes[key]
            midparent = 0.5 * (y_mom + y_dad)

            n = len(y_off)
            if n < 2:
                results[key] = {'h2': np.nan, 'slope': np.nan, 'intercept': np.nan,
                                'se': np.nan, 'n_trios': n}
                continue

            # OLS: y_off = intercept + slope * midparent
            mp_mean = midparent.mean()
            y_mean = y_off.mean()
            ss_mp = np.sum((midparent - mp_mean) ** 2)

            if ss_mp < 1e-15:
                results[key] = {'h2': np.nan, 'slope': np.nan, 'intercept': np.nan,
                                'se': np.nan, 'n_trios': n}
                continue

            slope = np.sum((midparent - mp_mean) * (y_off - y_mean)) / ss_mp
            intercept = y_mean - slope * mp_mean

            # Standard error of slope
            residuals = y_off - (intercept + slope * midparent)
            mse = np.sum(residuals ** 2) / (n - 2)
            se = np.sqrt(mse / ss_mp) if ss_mp > 0 else np.nan

            results[key] = {
                'h2': float(slope),
                'slope': float(slope),
                'intercept': float(intercept),
                'se': float(se),
                'n_trios': int(n),
            }

        return results


class MatingStatistics(Statistic):
    """
    Compute mating statistics from pedigree structure and parent phenotypes.

    Returns per-generation dict with:
    - n_mating_pairs: number of unique parent pairs
    - mean_offspring_count: mean offspring per pair
    - spouse_correlations: dict of phenotype name -> spousal Pearson r

    Requires a TrioFilter (keyed by ``filter_name``) to be active so that
    parent phenotypes are available, or works directly from pedigree if
    phenotype_history has the parent generation.

    Parameters
    ----------
    filter_name : str
        Key in ``filtered_views`` for a TrioView (used for spouse correlations).
        Default is ``'trio'``.
    """

    def __init__(self, filter_name: str = 'trio') -> None:
        self.filter_name: str = filter_name

    def estimate(self, phenotype_history: dict[int, PhenotypeArray],
                 filtered_views: dict[str, FilteredView],
                 generation: int,
                 **kwargs: Any) -> dict[str, Any] | None:
        if generation not in phenotype_history:
            return None

        pheno = phenotype_history[generation]

        # Get the TrioView to extract parent phenotypes
        view = filtered_views.get(self.filter_name)

        # Compute pair counts from FID structure of current generation
        fids = pheno.samples.fid
        _, counts = np.unique(fids, return_counts=True)
        n_mating_pairs = int(len(counts))
        mean_offspring_count = float(np.mean(counts))

        # Spouse correlations from TrioView
        spouse_correlations = {}
        if view is not None and isinstance(view, TrioView) and view.n_trios > 0:
            for key in view.mother_phenotypes:
                if key not in view.father_phenotypes:
                    continue
                y_mom = view.mother_phenotypes[key]
                y_dad = view.father_phenotypes[key]

                # Deduplicate: unique parent pairs (many offspring share same parents)
                # Use maternal + paternal phenotype values as proxy for pair identity
                # Stack and find unique pairs (row-wise)
                pairs = np.column_stack([y_mom, y_dad])
                unique_pairs = np.unique(pairs, axis=0)
                mom_unique = unique_pairs[:, 0]
                dad_unique = unique_pairs[:, 1]

                if len(mom_unique) < 2:
                    spouse_correlations[key] = np.nan
                    continue

                var_m = np.var(mom_unique, ddof=1)
                var_d = np.var(dad_unique, ddof=1)
                denom = np.sqrt(var_m * var_d)
                if denom < 1e-15:
                    spouse_correlations[key] = 0.0
                else:
                    cov_md = np.cov(mom_unique, dad_unique)[0, 1]
                    spouse_correlations[key] = float(cov_md / denom)

        return {
            'n_mating_pairs': n_mating_pairs,
            'mean_offspring_count': mean_offspring_count,
            'spouse_correlations': spouse_correlations,
        }


def _py_scalar(x: Any) -> Any:
    """numpy scalar -> Python scalar (object-array elements pass through)."""
    return x.item() if isinstance(x, np.generic) else x


class MetadataSummary(Statistic):
    """
    Summarize per-individual metadata fields (``SampleMeta.extra``) each
    generation.

    Numeric fields (e.g. ``ancestry_<POP>`` proportions) report ``mean`` and
    ``var``; other fields (e.g. population labels) report ``counts`` per
    value. Results are stored in ``sim.results`` every generation, so the
    full trajectory (such as the decay of ancestry variance under random
    mating) survives history pruning.

    Parameters
    ----------
    keys : list[str], optional
        Fields to summarize. Default: every field present (except ``by``).
    by : str, optional
        Categorical field to stratify by. The result then maps each group
        label to its own summary.

    Returns
    -------
    dict
        ``{field: {'n', 'mean', 'var'} or {'n', 'counts'}}``, or
        ``{group: {field: ...}}`` when ``by`` is set.
    """

    def __init__(self, keys: list[str] | None = None,
                 by: str | None = None) -> None:
        self.keys = None if keys is None else list(keys)
        self.by = by

    @property
    def metadata_keys(self) -> tuple[str, ...]:
        """Metadata fields this statistic reads."""
        return tuple(self.keys or ()) + ((self.by,) if self.by else ())

    @staticmethod
    def _summarize(extra: dict[str, np.ndarray], keys: list[str],
                   idx: np.ndarray) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for key in keys:
            vals = extra[key][idx]
            if np.issubdtype(vals.dtype, np.number) or vals.dtype == bool:
                vals = vals.astype(np.float64)
                mean, var = vals.mean(axis=0), vals.var(axis=0)
                out[key] = {
                    'n': int(len(idx)),
                    'mean': float(mean) if np.ndim(mean) == 0 else mean,
                    'var': float(var) if np.ndim(var) == 0 else var,
                }
            else:
                labels, counts = np.unique(vals, return_counts=True)
                out[key] = {
                    'n': int(len(idx)),
                    'counts': {_py_scalar(lab): int(c)
                               for lab, c in zip(labels, counts)},
                }
        return out

    def estimate(self, phenotype_history: dict[int, PhenotypeArray],
                 filtered_views: dict[str, FilteredView],
                 generation: int,
                 **kwargs: Any) -> dict[str, Any] | None:
        if generation not in phenotype_history:
            return None

        samples = phenotype_history[generation].samples
        extra = samples.extra
        if self.keys is None:
            keys = [k for k in extra if k != self.by]
        else:
            missing = [k for k in self.keys if k not in extra]
            if missing:
                raise KeyError(
                    f"MetadataSummary: generation {generation} has no metadata "
                    f"field(s) {missing}; available: {sorted(extra)}"
                )
            keys = self.keys

        if self.by is None:
            return self._summarize(extra, keys, np.arange(samples.n))

        if self.by not in extra:
            raise KeyError(
                f"MetadataSummary: generation {generation} has no metadata "
                f"field '{self.by}' to stratify by; available: {sorted(extra)}"
            )
        labels = extra[self.by]
        try:
            groups, group_idx = np.unique(labels, return_inverse=True)
        except TypeError as err:
            raise TypeError(
                f"MetadataSummary: labels in '{self.by}' cannot be sorted "
                f"(mixed types such as None and str?): {err}") from err
        return {
            _py_scalar(label): self._summarize(extra, keys,
                                               np.flatnonzero(group_idx == g))
            for g, label in enumerate(groups)
        }
