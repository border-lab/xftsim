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
from xftsim.struct import NPhenotypeArray


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
    def estimate(self, phenotype_history: dict[int, NPhenotypeArray],
                 filtered_views: dict[str, FilteredView],
                 generation: int,
                 **kwargs: Any) -> Any:
        """
        Compute the statistic for a given generation.

        Parameters
        ----------
        phenotype_history : dict[int, NPhenotypeArray]
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
    """

    def estimate(self, phenotype_history: dict[int, NPhenotypeArray],
                 filtered_views: dict[str, FilteredView],
                 generation: int,
                 **kwargs: Any) -> dict[str, Any] | None:
        if generation not in phenotype_history:
            return None

        pheno = phenotype_history[generation]
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

    def estimate(self, phenotype_history: dict[int, NPhenotypeArray],
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

        # Build standardized genotype matrix (n x m), per-SNP standardized
        G = hap.to_diploid_standardized(scale=True).astype(np.float64)
        n, m = G.shape

        # Build phenotype matrix (n x k), standardized
        Y = np.column_stack([pheno[k] for k in keys]).astype(np.float64)
        Y_mean = Y.mean(axis=0)
        Y_std = Y.std(axis=0)
        Y_std[Y_std < 1e-15] = 1.0
        Y = (Y - Y_mean) / Y_std

        # K Y = G (G' Y) / m  — computed without forming K explicitly
        GtY = G.T @ Y  # (m, k)
        KY = G @ GtY / m  # (n, k)

        # tr(K^2)
        if self.n_probe > 0 and n > 500:
            # Stochastic trace estimation via Hutchinson's method
            rng = np.random.RandomState()
            probes = rng.randn(n, self.n_probe)
            # tr(K^2) ≈ (1/l) * tr(P' K^2 P) = (1/l) * ||K P||_F^2
            KP = G @ (G.T @ probes) / m  # (n, n_probe)
            trK2 = np.sum(KP ** 2) / self.n_probe
        else:
            # Deterministic: tr(K^2) = tr(G G' G G' / m^2)
            # = ||G' G||_F^2 / m^2 (avoids forming n x n matrix)
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

    def estimate(self, phenotype_history: dict[int, NPhenotypeArray],
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

    def estimate(self, phenotype_history: dict[int, NPhenotypeArray],
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
