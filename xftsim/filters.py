"""
Filters for extracting structured views from simulation history.

Filters produce FilteredViews (trios, sib-pairs, etc.) from phenotype
and pedigree histories, used by statistics modules.
"""
from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

from xftsim.struct import PhenotypeArray, PedigreeArray


@dataclass
class FilteredView:
    """Base class for filtered data views produced by Filter.apply()."""
    pass


@dataclass
class TrioView(FilteredView):
    """
    Aligned trio data: offspring, mother, and father phenotypes.

    All dicts map phenotype name -> (n_trios,) array.
    """
    offspring_phenotypes: Dict[str, np.ndarray]
    mother_phenotypes: Dict[str, np.ndarray]
    father_phenotypes: Dict[str, np.ndarray]
    n_trios: int


@dataclass
class SibPairView(FilteredView):
    """
    Sibling pair data: two aligned sets of sibling phenotypes.

    All dicts map phenotype name -> (n_pairs,) array.
    sib1_idx and sib2_idx are the original sample indices.
    """
    sib1_phenotypes: Dict[str, np.ndarray]
    sib2_phenotypes: Dict[str, np.ndarray]
    n_pairs: int
    sib1_idx: np.ndarray = None
    sib2_idx: np.ndarray = None


class Filter(ABC):
    """
    Abstract base class for filters.

    Filters extract structured views from simulation history.
    """

    @abstractmethod
    def apply(self, generation: int,
              phenotype_history: dict,
              pedigree_history: dict) -> Optional[FilteredView]:
        """
        Apply the filter to extract a view.

        Parameters
        ----------
        generation : int
            Current generation number.
        phenotype_history : dict[int, PhenotypeArray]
            Generation -> phenotypes mapping.
        pedigree_history : dict[int, PedigreeArray]
            Generation -> pedigree mapping.

        Returns
        -------
        FilteredView or None
            The filtered view, or None if not applicable.
        """
        ...


class TrioFilter(Filter):
    """
    Extract complete trios (offspring + both parents) from adjacent generations.

    At generation 0, returns None (no parents).
    At generation > 0, indexes parent phenotypes from gen-1 by pedigree indices.
    """

    def apply(self, generation: int,
              phenotype_history: dict[int, PhenotypeArray],
              pedigree_history: dict[int, PedigreeArray]) -> TrioView | None:
        if generation == 0 or generation not in pedigree_history:
            return None

        prev_gen = generation - 1
        if prev_gen not in phenotype_history:
            return None

        ped = pedigree_history[generation]
        offspring_pheno = phenotype_history[generation]
        parent_pheno = phenotype_history[prev_gen]

        n = offspring_pheno.samples.n
        keys = list(offspring_pheno.keys)

        offspring_dict: dict[str, np.ndarray] = {}
        mother_dict: dict[str, np.ndarray] = {}
        father_dict: dict[str, np.ndarray] = {}

        for key in keys:
            offspring_dict[key] = offspring_pheno[key].copy()
            if key in parent_pheno:
                mother_dict[key] = parent_pheno[key][ped.maternal_idx]
                father_dict[key] = parent_pheno[key][ped.paternal_idx]

        return TrioView(
            offspring_phenotypes=offspring_dict,
            mother_phenotypes=mother_dict,
            father_phenotypes=father_dict,
            n_trios=n,
        )


class SibPairFilter(Filter):
    """
    Extract sibling pairs (individuals sharing the same FID).

    Groups offspring by FID and forms all unique within-family pairs.
    """

    def apply(self, generation: int,
              phenotype_history: dict[int, PhenotypeArray],
              pedigree_history: dict[int, PedigreeArray]) -> SibPairView | None:
        if generation not in phenotype_history:
            return None

        pheno = phenotype_history[generation]
        fids = pheno.samples.fid
        keys = list(pheno.keys)

        # Sort by FID for contiguous groups
        sort_idx = np.argsort(fids)
        sorted_fids = fids[sort_idx]
        _, start_idx, counts = np.unique(
            sorted_fids, return_index=True, return_counts=True
        )

        # Only families with 2+ members
        multi_mask = counts >= 2
        starts = start_idx[multi_mask]
        sizes = counts[multi_mask]

        if len(starts) == 0:
            empty = np.array([], dtype=np.int64)
            return SibPairView(
                sib1_phenotypes={k: np.array([]) for k in keys},
                sib2_phenotypes={k: np.array([]) for k in keys},
                n_pairs=0,
                sib1_idx=empty,
                sib2_idx=empty,
            )

        # Vectorized pair generation: batch by family size
        sib1_parts = []
        sib2_parts = []
        for sz in np.unique(sizes):
            mask_sz = sizes == sz
            fam_starts = starts[mask_sz]
            n_fam = len(fam_starts)
            # Gather original indices for all families of this size
            offsets = np.arange(sz)
            all_idx = sort_idx[
                (fam_starts[:, None] + offsets[None, :]).ravel()
            ].reshape(n_fam, sz)
            # Upper-triangle pairs within each family
            i_tri, j_tri = np.triu_indices(sz, k=1)
            sib1_parts.append(all_idx[:, i_tri].ravel())
            sib2_parts.append(all_idx[:, j_tri].ravel())

        idx1 = np.concatenate(sib1_parts)
        idx2 = np.concatenate(sib2_parts)

        sib1_dict = {key: pheno[key][idx1] for key in keys}
        sib2_dict = {key: pheno[key][idx2] for key in keys}

        return SibPairView(
            sib1_phenotypes=sib1_dict,
            sib2_phenotypes=sib2_dict,
            n_pairs=len(idx1),
            sib1_idx=idx1,
            sib2_idx=idx2,
        )


# ---------------------------------------------------------------------------
# Unrelated sample filter
# ---------------------------------------------------------------------------

@dataclass
class UnrelatedView(FilteredView):
    """
    View of one individual per family (unrelated subsample).

    Attributes
    ----------
    indices : np.ndarray
        Indices into the original sample array (one per family).
    phenotypes : PhenotypeArray
        Subset of phenotypes for the selected individuals.
    """
    indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.intp))
    phenotypes: PhenotypeArray = None


class UnrelatedFilter(Filter):
    """
    Select one individual per family (first occurrence per FID).

    Produces an UnrelatedView with the first individual encountered
    for each unique FID value.
    """

    def apply(self, generation: int,
              phenotype_history: dict[int, PhenotypeArray],
              pedigree_history: dict[int, PedigreeArray]) -> UnrelatedView | None:
        if generation not in phenotype_history:
            return None

        pheno = phenotype_history[generation]
        fids = pheno.samples.fid

        # np.unique with return_index gives the first occurrence of each FID
        _, first_idx = np.unique(fids, return_index=True)
        first_idx = np.sort(first_idx)

        return UnrelatedView(
            indices=first_idx,
            phenotypes=pheno.subset(first_idx),
        )


# ---------------------------------------------------------------------------
# Ascertainment filter
# ---------------------------------------------------------------------------

@dataclass
class AscertainedView(FilteredView):
    """
    View of individuals passing an ascertainment threshold.

    Attributes
    ----------
    indices : np.ndarray
        Indices into the original sample array.
    phenotypes : PhenotypeArray
        Subset of phenotypes for selected individuals.
    ascertainment_key : str
        The phenotype key used for ascertainment.
    threshold : float
        The quantile threshold value(s) used.
    """
    indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.intp))
    phenotypes: PhenotypeArray = None
    ascertainment_key: str = ""
    threshold: float = 0.0


class AscertainmentFilter(Filter):
    """
    Select individuals from the tails of a phenotype distribution.

    Parameters
    ----------
    phenotype_key : str
        Which phenotype to ascertain on (e.g. 'Y', 'Y.G').
    quantile : float
        Proportion of the distribution to select. E.g. 0.1 selects
        the top 10%, bottom 10%, or both (depending on ``tail``).
    tail : str
        Which tail(s) to select: 'upper', 'lower', or 'both'.
        - 'upper': individuals above the (1 - quantile) percentile
        - 'lower': individuals below the quantile percentile
        - 'both': individuals in either tail (union of upper and lower)
    """

    def __init__(self, phenotype_key: str, quantile: float,
                 tail: str = 'both') -> None:
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"quantile must be in (0, 1), got {quantile}")
        if tail not in ('upper', 'lower', 'both'):
            raise ValueError(f"tail must be 'upper', 'lower', or 'both', got '{tail}'")
        self.phenotype_key: str = phenotype_key
        self.quantile: float = quantile
        self.tail: str = tail

    def apply(self, generation: int,
              phenotype_history: dict[int, PhenotypeArray],
              pedigree_history: dict[int, PedigreeArray]) -> AscertainedView | None:
        if generation not in phenotype_history:
            return None

        pheno = phenotype_history[generation]

        if self.phenotype_key not in pheno:
            return None

        values = pheno[self.phenotype_key]

        if self.tail == 'upper':
            threshold = np.quantile(values, 1.0 - self.quantile)
            mask = values >= threshold
        elif self.tail == 'lower':
            threshold = np.quantile(values, self.quantile)
            mask = values <= threshold
        else:  # 'both'
            lower_thresh = np.quantile(values, self.quantile)
            upper_thresh = np.quantile(values, 1.0 - self.quantile)
            mask = (values <= lower_thresh) | (values >= upper_thresh)
            threshold = self.quantile  # store the quantile itself for 'both'

        indices = np.where(mask)[0]

        return AscertainedView(
            indices=indices,
            phenotypes=pheno.subset(indices),
            ascertainment_key=self.phenotype_key,
            threshold=float(threshold),
        )


# ---------------------------------------------------------------------------
# Subsample filter
# ---------------------------------------------------------------------------

@dataclass
class SubsampleView(FilteredView):
    """
    View of a random subsample of individuals.

    Attributes
    ----------
    indices : np.ndarray
        Indices into the original sample array.
    phenotypes : PhenotypeArray
        Subset of phenotypes for selected individuals.
    n_subsample : int
        Number of individuals in the subsample.
    """
    indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.intp))
    phenotypes: PhenotypeArray = None
    n_subsample: int = 0


class SubsampleFilter(Filter):
    """
    Randomly subsample individuals.

    Exactly one of ``n`` or ``fraction`` must be provided.

    Parameters
    ----------
    n : int, optional
        Exact number of individuals to sample. If larger than the
        population, all individuals are returned.
    fraction : float, optional
        Fraction of individuals to sample, in (0, 1].
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, n: int | None = None, fraction: float | None = None,
                 seed: int | None = None) -> None:
        if n is not None and fraction is not None:
            raise ValueError("Specify exactly one of 'n' or 'fraction', not both")
        if n is None and fraction is None:
            raise ValueError("Must specify one of 'n' or 'fraction'")
        if n is not None and n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if fraction is not None and not (0.0 < fraction <= 1.0):
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        self._n = n
        self._fraction = fraction
        self._seed = seed

    def apply(self, generation: int,
              phenotype_history: dict[int, PhenotypeArray],
              pedigree_history: dict[int, PedigreeArray]) -> SubsampleView | None:
        if generation not in phenotype_history:
            return None

        pheno = phenotype_history[generation]
        n_total = pheno.samples.n

        if self._n is not None:
            n_sub = min(self._n, n_total)
        else:
            n_sub = max(1, int(np.round(self._fraction * n_total)))

        rng = np.random.RandomState(self._seed)
        indices = np.sort(rng.choice(n_total, size=n_sub, replace=False))

        return SubsampleView(
            indices=indices,
            phenotypes=pheno.subset(indices),
            n_subsample=n_sub,
        )
