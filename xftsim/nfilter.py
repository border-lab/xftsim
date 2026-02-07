"""
Filters for extracting structured views from simulation history.

Filters produce FilteredViews (trios, sib-pairs, etc.) from phenotype
and pedigree histories, used by statistics modules.
"""
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

from xftsim.struct import NPhenotypeArray, PedigreeArray


@dataclass
class FilteredView:
    """Base class for filtered data views."""
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
        phenotype_history : dict[int, NPhenotypeArray]
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

    def apply(self, generation, phenotype_history, pedigree_history):
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

        offspring_dict = {}
        mother_dict = {}
        father_dict = {}

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

    def apply(self, generation, phenotype_history, pedigree_history):
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
