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

from xftsim.nfilter import TrioView, SibPairView, FilteredView
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
                 generation: int) -> Any:
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
                 generation: int) -> dict[str, Any] | None:
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
    Sibling-based Haseman-Elston estimator of heritability.

    Uses sibling pair covariance to estimate h2 per phenotype.
    For full sibs under an additive model:  Cov(sib1, sib2) / Var(Y) ~ h2/2,
    so h2 ~ 2 * r_sib where r_sib is the sibling intraclass correlation.

    Requires a SibPairFilter (keyed by ``filter_name``) to be active
    in the simulation's filters dict.

    Parameters
    ----------
    filter_name : str
        Key in the ``filtered_views`` dict that contains a SibPairView.
        Default is ``'sibpair'``.
    """

    def __init__(self, filter_name: str = 'sibpair') -> None:
        self.filter_name: str = filter_name

    def estimate(self, phenotype_history: dict[int, NPhenotypeArray],
                 filtered_views: dict[str, FilteredView],
                 generation: int) -> dict[str, dict[str, Any]] | None:
        view = filtered_views.get(self.filter_name)
        if view is None or not isinstance(view, SibPairView):
            return None
        if view.n_pairs == 0:
            return None

        results = {}
        keys = list(view.sib1_phenotypes.keys())
        for key in keys:
            y1 = view.sib1_phenotypes[key]
            y2 = view.sib2_phenotypes[key]
            if len(y1) < 2:
                results[key] = {'h2': np.nan, 'sib_r': np.nan, 'n_pairs': len(y1)}
                continue

            # Sibling intraclass correlation via Pearson r
            # (equivalent to ICC for paired data)
            m1 = y1.mean()
            m2 = y2.mean()
            cov_12 = np.mean((y1 - m1) * (y2 - m2))
            var1 = np.var(y1, ddof=0)
            var2 = np.var(y2, ddof=0)
            denom = np.sqrt(var1 * var2)
            if denom < 1e-15:
                sib_r = 0.0
            else:
                sib_r = cov_12 / denom

            # h2 = 2 * sib_r for full sibs (additive model)
            h2 = 2.0 * sib_r
            results[key] = {
                'h2': float(h2),
                'sib_r': float(sib_r),
                'n_pairs': int(len(y1)),
            }

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
                 generation: int) -> dict[str, dict[str, Any]] | None:
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
                 generation: int) -> dict[str, Any] | None:
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
