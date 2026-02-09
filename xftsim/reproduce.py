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

from xftsim.mate import MateAssignment


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
    """

    def __init__(self,
                 p=None,
                 m: int = None,
                 vid: NDArray[Shape["*"], Any] = None,
                 chrom: NDArray[Shape["*"], Int64] = None,
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

    @staticmethod
    def constant_map(m: int, p: float = 0.5) -> "RecombinationMap":
        """
        Create a constant recombination map.

        Parameters
        ----------
        m : int
            Number of variants.
        p : float, optional
            Recombination probability, default is 0.5.

        Returns
        -------
        RecombinationMap
            A constant recombination map.
        """
        return RecombinationMap(p=p, m=m)

    @staticmethod
    def from_haplotypes(haplotypes: xft.struct.NHaplotypeArray,
                        p: float = 0.5) -> "RecombinationMap":
        """
        Create a constant recombination map from haplotypes.

        Parameters
        ----------
        haplotypes : NHaplotypeArray
            Haplotypes data.
        p : float, optional
            Probability, default is 0.5.

        Returns
        -------
        RecombinationMap
            A constant recombination map.
        """
        return RecombinationMap(p=p, m=haplotypes.m, vid=haplotypes.vid)

    def __repr__(self):
        df = pd.DataFrame.from_dict(dict(vid=self.vid,
                                         chrom=self.chrom,
                                         p=self._probabilities))
        return df.__repr__()


@nb.njit("int64[:](float64[:])")
def _meiosis_i(p):
    """
    Maps recombination probabilities to haploid indices (0 or 1).

    Parameters
    ----------
    p : numpy.ndarray[float64]
        An array of recombination probabilities.

    Returns
    -------
    numpy.ndarray[int64]
        An array of haploid indices (0 or 1).
    """
    m = p.shape[0]
    output = np.empty(m, dtype=np.int64)
    for j in range(m):
        output[j] = np.random.binomial(1, p[j])
    output = np.cumsum(output) % 2
    return output


@nb.njit("int8[:,:,:](int8[:,:,:], int8[:,:,:], int64, int64, float64[:], int64[:], int64[:])", parallel=True)
def _meiosis_3d(parental_genotypes,
                offspring_genotypes,
                n_offspring,
                m,
                recombination_p,
                maternal_inds,
                paternal_inds,
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
    recombination_p : numpy.ndarray[float64]
        An array of recombination probabilities.
    maternal_inds : numpy.ndarray[int64]
        An array of maternal parent indices.
    paternal_inds : numpy.ndarray[int64]
        An array of paternal parent indices.

    Returns
    -------
    numpy.ndarray[int8]
        3D array of offspring genotypes.
    """
    for i in nb.prange(n_offspring):
        # Maternal meiosis: select which haplotype (0 or 1) at each locus
        mat_hap_select = _meiosis_i(recombination_p)
        # Paternal meiosis: select which haplotype (0 or 1) at each locus
        pat_hap_select = _meiosis_i(recombination_p)

        mat_idx = maternal_inds[i]
        pat_idx = paternal_inds[i]

        for j in range(m):
            # Offspring's first haplotype comes from mother
            offspring_genotypes[i, j, 0] = parental_genotypes[mat_idx, j, mat_hap_select[j]]
            # Offspring's second haplotype comes from father
            offspring_genotypes[i, j, 1] = parental_genotypes[pat_idx, j, pat_hap_select[j]]

    return offspring_genotypes


def meiosis(parental_haplotypes: xft.struct.NHaplotypeArray,
            recombination_map: RecombinationMap,
            maternal_inds: NDArray[Shape["*"], Int64],
            paternal_inds: NDArray[Shape["*"], Int64],
            ) -> NDArray[Shape["*, *, *"], Int8]:
    """
    Performs meiosis on parental haplotypes.

    Parameters
    ----------
    parental_haplotypes : NHaplotypeArray
        Parental haplotype data.
    recombination_map : RecombinationMap
        Recombination probabilities.
    maternal_inds : numpy.ndarray[int64]
        An array of maternal parent indices.
    paternal_inds : numpy.ndarray[int64]
        An array of paternal parent indices.

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
    assert maternal_inds.dtype == paternal_inds.dtype == np.int64, "indices must be int64"

    n_offspring = maternal_inds.shape[0]
    offspring_genotypes = np.empty((n_offspring, m, 2), dtype=np.int8)

    return _meiosis_3d(parental_genotypes,
                       offspring_genotypes,
                       n_offspring,
                       m,
                       recombination_p,
                       maternal_inds,
                       paternal_inds)


class Meiosis:
    """
    A class representing the process of meiosis.

    Attributes
    ----------
    recombination_map : RecombinationMap, optional
        A pre-defined recombination map.
    p : float, optional
        A probability used when generating an exchangeable recombination map on the fly.

    Methods
    -------
    get_recombination_map(haplotypes):
        Returns the recombination map, either pre-defined or generated on the fly.
    reproduce(parental_haplotypes, mating):
        Returns a NHaplotypeArray representing the offspring after meiosis.
    """

    def __init__(self,
                 rmap: RecombinationMap = None,
                 p: float = None):
        assert (p is not None) ^ (rmap is not None), "Provide p XOR rmap"
        self.recombination_map = rmap
        self._p = p

    def get_recombination_map(self, haplotypes: xft.struct.NHaplotypeArray) -> RecombinationMap:
        """
        Get the recombination map, either pre-defined or generated on the fly.

        Parameters
        ----------
        haplotypes : NHaplotypeArray
            The haplotype data.

        Returns
        -------
        RecombinationMap
            The recombination map.
        """
        if self.recombination_map is not None:
            return self.recombination_map
        else:
            return RecombinationMap.from_haplotypes(haplotypes, p=self._p)

    def reproduce(self,
                  parental_haplotypes: xft.struct.NHaplotypeArray,
                  mating: MateAssignment,
                  ) -> xft.struct.NHaplotypeArray:
        """
        Return a NHaplotypeArray representing the offspring after meiosis.

        Parameters
        ----------
        parental_haplotypes : NHaplotypeArray
            The parental haplotype data.
        mating : MateAssignment
            The mate assignment object.

        Returns
        -------
        NHaplotypeArray
            The NHaplotypeArray representing the offspring after meiosis.
        """
        rmap = self.get_recombination_map(parental_haplotypes)

        # Get parent indices
        maternal_inds = mating.reproducing_maternal_index_array
        paternal_inds = mating.reproducing_paternal_index_array

        # Perform meiosis
        offspring_genotypes = meiosis(
            parental_haplotypes,
            rmap,
            maternal_inds,
            paternal_inds,
        )

        # Create offspring sample metadata
        n_offspring = offspring_genotypes.shape[0]
        offspring_iid = np.arange(n_offspring, dtype=np.int64)
        offspring_samples = xft.struct.SampleMeta(iid=offspring_iid)

        # Inherit variant metadata from parents (variants don't change)
        offspring_variants = parental_haplotypes.variants

        return xft.struct.NHaplotypeArray(
            genotypes=offspring_genotypes,
            generation=parental_haplotypes.generation + 1,
            samples=offspring_samples,
            variants=offspring_variants,
        )


def transmit_parental_phenotypes(
    mating: MateAssignment,
    parental_phenotypes: xr.DataArray,
    offspring_phenotypes: xr.DataArray,
    control: dict = None,
) -> None:
    """
    Transmits parental phenotypes to offspring.

    Parameters
    ----------
    mating : MateAssignment
        An object representing mating assignments.
    parental_phenotypes : xr.DataArray
        A data array containing parental phenotypes.
    offspring_phenotypes : xr.DataArray
        A data array containing offspring phenotypes.
    control : dict, optional
        A dictionary containing additional control parameters, default is None.

    Returns
    -------
    None
    """
    # sample indexes (wrt to previous generation) for parents
    parent_gen_mat_sample_ind = mating.reproducing_maternal_index
    parent_gen_pat_sample_ind = mating.reproducing_paternal_index

    # component index of current generation
    offspring_component_index = parental_phenotypes.xft.get_component_indexer()
    # component indexes (in current generation) for inherited phenotypes
    offspring_gen_maternal_component_ind = offspring_component_index[dict(
        vorigin_relative=0)]
    offspring_gen_paternal_component_ind = offspring_component_index[dict(
        vorigin_relative=1)]
    # component indexes (in previous generation) for inherited phenotypes
    parent_gen_maternal_component_ind = offspring_gen_maternal_component_ind.to_proband()
    parent_gen_paternal_component_ind = offspring_gen_paternal_component_ind.to_proband()

    # transmit maternal components
    maternal_data = parental_phenotypes.xft[parent_gen_mat_sample_ind,
                                            parent_gen_maternal_component_ind].data
    offspring_phenotypes.xft[None,
                             offspring_gen_maternal_component_ind] = maternal_data

    # transmit paternal components
    paternal_data = parental_phenotypes.xft[parent_gen_pat_sample_ind,
                                            parent_gen_paternal_component_ind].data
    offspring_phenotypes.xft[None,
                             offspring_gen_paternal_component_ind] = paternal_data
