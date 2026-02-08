import xftsim as xft
import warnings
import functools
import numpy as np
import numba as nb
import pandas as pd
import nptyping as npt
import scipy as sp
import xarray as xr
from nptyping import NDArray, Int8, Int64, Float64, Bool, Shape
from typing import Any, Hashable, List, Iterable, Callable, Union, Dict, final
from numpy.typing import ArrayLike
from functools import cached_property
import dask.array as da
# from warnings import deprecated





class SampleFilter:
    """
    Base class for sample filters for estimators

    Attributes
    ----------
    name : str
        The name of the filter.

    Methods
    -------
    filter(phenotypes) -> np.ndarray(int):
        Generate indices of rows to sample
    """
    def __init__(self, name: str = None):
        self.name = name

    def filter(self, phenotypes):
        pass


# @deprecated("Composition is now built into AscertainmentFilter class")
class ComposedFilter(SampleFilter):
    """
    DEPRECATED
    Composed sample filters for estimators

    Attributes
    ----------
    outer_filter : SampleFilter
    inner_filter : SampleFilter

    Methods
    -------
    filter(phenotypes) -> np.ndarray(int):
        Generate indices of rows to sample by apply outer_filter to result of 
        inner_filtersd
    """
    def __init__(self, outer_filter : SampleFilter, inner_filter : SampleFilter):
        # if inner_filter.nsub > outer_filter.nsub and outer_filter.nsub!=0:
            # raise RuntimeError('inner filter is finer than outer filter')
        self.inner_filter = inner_filter
        self.outer_filter = outer_filter

    @property
    def nsub(self):
        return self.outer_filter.nsub

    @nsub.setter
    def nsub(self, value):
        self.outer_filter.nsub = value 

    def filter(self, phenotypes):
        subinds1 = self.inner_filter.filter(phenotypes)
        subinds0 = self.outer_filter.filter(phenotypes[subinds1,:])
        return subinds1[subinds0]


class AscertainmentFilter(SampleFilter):
    """
    Base class for sample that ascertains from random sample filters for estimators

    Attributes
    ----------
    name : str
        The name of the filter.
    nsub_random : int
        The size of the random subsample 
    nsub_ascertained : int
        The size of the ascertained sub-subsample 
 
    Methods
    -------
    filter(phenotypes) -> np.ndarray(int):
        Generate indices of rows to sample
    """

    def __init__(self, name: str,
                 nsub_random: int,
                 nsub_ascertained: int,
                 ):
        self.name = name
        self.nsub_random = nsub_random
        self.nsub_ascertained = nsub_ascertained

    @property
    def nsub(self):
        return self.nsub_ascertained

    @nsub.setter
    def nsub(self, value):
        self.nsub_ascertained = value 
    
    def filter(self, phenotypes):
        pass



class PassFilter(SampleFilter):
    """
    Default non-filter filter for estimators

    Methods
    -------
    filter(sim: xft.sim.Simulation) -> np.ndarray(int):
        Returns all sample indices
    """
    def __init__(self, name: str = 'pass'):
        self.name = name

    def filter(self, phenotypes):
        subinds = np.arange(phenotypes.xft.n)
        return subinds

class UnrelatedSampleFilter(SampleFilter):
    """
    Take a sample of unrelated individuals

    Attributes
    ----------
    nsub : int
        The number of individuals to take.

    Methods
    -------
    filter(sim: xft.sim.Simulation) -> np.ndarray(int):
        Generate indices of rows to sample
    """
    def __init__(self, nsub: int = 0, name: str = 'UnrelatedSampleFilter'):
        self.name = name
        self.nsub = nsub

    def filter(self, phenotypes):
        sind = phenotypes.xft.get_sample_indexer()
        if self.nsub==0:
            nsub=sind.n_fam
        else:
            nsub = self.nsub
        subinds = xft.utils.hierarchical_subsample(a1=sind.fid, a2=sind.iid, 
                                                   n1=nsub, n2=1)
        return subinds


class FirstUnrelatedSampleFilter(SampleFilter):
    """
    Take the first individual in each family. Fast.

    Attributes
    ----------
    nsub : int
        The number of individuals to take.

    Methods
    -------
    filter(sim: xft.sim.Simulation) -> np.ndarray(int):
        Generate indices of rows to sample
    """
    def __init__(self, nsub: int = 0, name: str = 'FirstUnrelatedSampleFilter'):
        self.name = name
        self.nsub = nsub

    def filter(self, phenotypes):
        sind = phenotypes.xft.get_sample_indexer()
        nfam=sind.n_fam
        if self.nsub==0:
            nsub = nfam
        else:
            nsub = np.min([self.nsub, nfam])
        null,subinds = np.unique(sind.fid, return_index=True) 
        return np.sort(np.random.permutation(subinds)[:nsub])



class SibpairSampleFilter(SampleFilter):
    """
    Take a sample of unrelated individuals

    Attributes
    ----------
    nsub : int
        The number of sibpairs to take.

    Methods
    -------
    filter(sim: xft.sim.Simulation) -> np.ndarray(int):
        Generate indices of rows to sample
    """
    def __init__(self, nsub: int = 0, name: str = 'SibpairSampleFilter'):
        self.name = name
        self.nsub = nsub

    def filter(self, phenotypes):
        sind = phenotypes.xft.get_sample_indexer()
        if self.nsub==0:
            nsub=sind.n_fam
        else:
            nsub = self.nsub
        subinds = xft.utils.hierarchical_subsample(a1=sind.fid, a2=sind.iid, 
                                                   n1=nsub, n2=2)
        return subinds



class UnrelatedAscertainmentFilter(AscertainmentFilter):
    """
    Take top/bottom `nsub_ascertained` from `nsub_random` sample of unrelated individuals
    ## TODO docstring

    Attributes
    ----------
    name : str
        The name of the filter.
    nsub_random : int
        The size of the random subsample 
    nsub_ascertained : int
        The size of the ascertained sub-subsample .


    Methods
    -------
    filter(sim: xft.sim.Simulation) -> np.ndarray(int):
        Generate indices of rows to sample
    """
    def __init__(self, 
                 nsub_random: int = 0, 
                 nsub_ascertained: int = 0, 
                 component_index: xft.index.ComponentIndex = None,
                 coef: np.ndarray = None,
                 coef_noise: float = 0,
                 standardize: bool = True,
                 name: str = 'UnrelatedAscertainmentFilter',
                 ):
        self.name = name
        self.nsub_random = nsub_random
        self.nsub_ascertained = nsub_ascertained
        self.component_index = component_index
        self.coef = coef
        self.coef_noise = coef_noise
        self.standardize = standardize
        self.polarity = -1

    def filter(self, phenotypes):
        sind = phenotypes.xft.get_sample_indexer()
        ## set (sub)subsample sizes
        if self.nsub_random==0:
            nsub_random=sind.n_fam
        else:
            nsub_random = self.nsub_random
        if self.nsub_ascertained==0:
            nsub_ascertained=sind.n_fam//2
        else:
            nsub_ascertained = self.nsub_ascertained
        ## default to ascertainment on phenotypes
        if self.component_index is None:
            component_index = phenotypes.xft.get_component_indexer()[{'vorigin_relative':-1,'component_name':'phenotype'}]
        else:
            component_index = self.component_index
        ## default to sorting by sum of components
        if self.coef is None:
            coef = self.polarity * np.ones(component_index.k_total)
        else:
            coef = self.polarity * self.coef

        ## get random subsample
        unrel_inds = xft.utils.hierarchical_subsample(a1=sind.fid, a2=sind.iid, 
                                                      n1=nsub_random, n2=1)
        pheno_sub = phenotypes[unrel_inds, :].xft[None,component_index].data
        ## score random subsample
        if self.standardize:
            pheno_sub=xft.utils.standardize_array(pheno_sub)
        ascertainment_score = pheno_sub @ coef 
        ascertainment_score += np.random.randn(pheno_sub.shape[0]) * self.coef_noise
        ## order by score
        ascertained_inds = unrel_inds[np.argsort(ascertainment_score)]
        ## take first nsub_ascertained
        nsub_ascertained = np.min([self.nsub_ascertained, ascertained_inds.shape[0]])
        subinds = ascertained_inds[:nsub_ascertained]
        return subinds


class SibpairAscertainmentFilter(AscertainmentFilter):
    """
    Take top/bottom `nsub_ascertained` from `nsub_random` sample of sibpairs individuals
    ## TODO docstring

    Attributes
    ----------
    name : str
        The name of the filter.
    nsub_random : int
        The size of the random subsample 
    nsub_ascertained : int
        The size of the ascertained sub-subsample .




    Methods
    -------
    filter(sim: xft.sim.Simulation) -> np.ndarray(int):
        Generate indices of rows to sample
    """
    def __init__(self, 
                 nsub_random: int = 0, 
                 nsub_ascertained: int = 0, 
                 component_index: xft.index.ComponentIndex = None,
                 coef: np.ndarray = None,
                 coef_noise: float = 0,
                 standardize: bool = True,
                 name: str = 'UnrelatedAscertainmentFilter',
                 combine: str = 'mean',
                 ):
        self.name = name
        self.nsub_random = nsub_random
        self.nsub_ascertained = nsub_ascertained
        self.component_index = component_index
        self.coef = coef
        self.coef_noise = coef_noise
        self.standardize = standardize
        self.combine = combine
        self.polarity = -1

    def score(self, phenotypes):
        ## score sibpairs for ascertainment
        if self.coef is None:
            coef = 1 * np.ones(phenotypes.shape[1])
        else:
            coef = self.coef
        if self.standardize:
            pheno_sub=xft.utils.standardize_array(phenotypes)
        if self.combine == 'mean':
            pheno_sum = (pheno_sub[0::2,:] + pheno_sub[1::2,:])/2
            ascertainment_score = pheno_sum @ coef 
        elif self.combine == 'max':
            s1 = pheno_sub[0::2,:] @ coef
            s2 = pheno_sub[1::2,:] @ coef
            ss = np.hstack([s1.reshape((s1.shape[0],1)),s2.reshape((s2.shape[0],1))])
            ascertainment_score = np.max(ss,1)
        elif self.combine == 'min':
            s1 = pheno_sub[0::2,:] @ coef
            s2 = pheno_sub[1::2,:] @ coef
            ss = np.hstack([s1.reshape((s1.shape[0],1)),s2.reshape((s2.shape[0],1))])
            ascertainment_score = np.min(ss,1)
        ascertainment_score = ascertainment_score/np.std(ascertainment_score) + np.random.randn(ascertainment_score.shape[0]) * self.coef_noise
        return np.repeat(ascertainment_score,2)

    def filter(self, phenotypes):
        sind = phenotypes.xft.get_sample_indexer()
         ## set (sub)subsample sizes
        if self.nsub_random==0:
            nsub_random=sind.n_fam
        else:
            nsub_random = self.nsub_random
        if self.nsub_ascertained==0:
            nsub_ascertained=sind.n_fam//2
        else:
            nsub_ascertained = self.nsub_ascertained
        ## default to ascertainment on phenotypes
        if self.component_index is None:
            component_index = phenotypes.xft.get_component_indexer()[{'vorigin_relative':-1,'component_name':'phenotype'}]
        else:
            component_index = self.component_index
        sib_inds = xft.utils.hierarchical_subsample(a1=sind.fid, a2=sind.iid, 
                                                    n1=nsub_random, n2=2)
        pheno_sub = phenotypes[sib_inds, :].xft[None,component_index].data

        ascertainment_score = self.polarity*self.score(pheno_sub)
        ascertained_inds = np.argsort(ascertainment_score)
        nsub_ascertained = np.min([nsub_ascertained, 2*ascertained_inds.shape[0]])
        subinds = np.sort(sib_inds[ascertained_inds[:(nsub_ascertained*2)]])
        return subinds



