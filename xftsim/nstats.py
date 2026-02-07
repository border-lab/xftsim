"""
Statistics computed per-generation in a simulation.

Statistic ABC and concrete implementations. Each statistic receives
the phenotype history and any filtered views, and returns a result
stored in GenerationResult.
"""
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


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
    """Abstract base class for per-generation statistics."""

    @abstractmethod
    def estimate(self, phenotype_history: dict,
                 filtered_views: dict,
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

    def estimate(self, phenotype_history, filtered_views, generation):
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
