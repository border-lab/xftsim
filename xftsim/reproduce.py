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

