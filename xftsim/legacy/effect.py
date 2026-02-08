"""Summary
"""
import sys
import warnings
import numpy as np
import pandas as pd
import nptyping as npt
import xarray as xr
from nptyping import NDArray, Int8, Int64, Float64, Bool, Shape
from typing import Any, Hashable, List, Iterable, Callable, Union, Dict
from functools import cached_property
from dataclasses import dataclass, field

import xftsim as xft

# simple additive effects that are standardized and/or scaled


class AdditiveEffects:
    """Additive genetic effects object. Given matrix / vector of effects will provide various scalings / offsets for computation
    
    Parameters
    ----------
    beta : NDArray[Any, Any]
        Vector of diploid effects
    variant_indexer : xft.index.HaploidVariantIndex | xft.index.DiploidVariantIndex
        Variant indexer, will determine ploidy automatically
    component_indexer : xft.index.ComponentIndex, optional
        Phenotype component indexer, defaults to xft.index.ComponentIndex.RangeIndex if not provided
    standardized : bool, optional
        True implies these are effects of standardized variants, by default True
    scaled : bool, optional
        True implies these are effects of variants * sqrt(m_causal), by default True
    
    Attributes
    ----------
    AF : NDArray
        diploid allele frequencies
    beta_scaled_standardized_diploid : NDArray
        Diploid effects scaled of standardized variants multiplied by number of causal variants per phenotype
    beta_scaled_standardized_haploid : NDArray
        haploid variant of above
    beta_scaled_unstandardized_diploid : NDArray
        Diploid effects scaled of unstandardized variants multiplied by number of causal variants per phenotype
    beta_scaled_unstandardized_haploid : NDArray
        haploid variant of above
    beta_unscaled_standardized_diploid : NDArray
        Diploid effects scaled of standardized variants unscaled by number of causal variants per phenotype
    beta_unscaled_standardized_haploid : NDArray
        haploid variant of above
    beta_unscaled_unstandardized_diploid : NDArray
        Diploid effects scaled of unstandardized variants unscaled by number of causal variants
        Multiply these against (0,1,2) raw genotypes and add offset to obtain phenotypes
    beta_unscaled_unstandardized_haploid : NDArray
        Haploid variant of above
    beta_raw_diploid : NDArray
        Alias for beta_unscaled_unstandardized_diploid
    beta_raw_haploid : NDArray
        Alias for beta_unscaled_unstandardized_haploid
    component_indexer : xft.index.ComponentIndex
    k : int
        Number of phenotypes (columns of effect matrix)
    m : int
        Number of diploid variants
    offset : NDArray
        To compute phenotypes, add offset after multiplying by beta_raw_* to mean deviate under HWE
    variant_indexer : xft.index.HaploidVariantIndex
    """

    def __init__(self,
        beta: NDArray[Any, Any],
        variant_indexer: Union[xft.index.HaploidVariantIndex,
            xft.index.DiploidVariantIndex] = None,
        component_indexer: xft.index.ComponentIndex = None,
        standardized: bool = True,
        scaled: bool = True,
        ):
        unscaled = not scaled
        unstandardized = not standardized
        if isinstance(variant_indexer, xft.index.HaploidVariantIndex):
            self.component_indexer = component_indexer
        else:
            self.component_indexer = component_indexer.to_haploid()

        self.variant_indexer = variant_indexer

        if isinstance(variant_indexer, xft.index.HaploidVariantIndex):
            self._tiledAF = xft.utils.ensure2D(self.variant_indexer.af)
            self.AF = self._tiledAF[::2, :]
        else:
            self.AF = xft.utils.ensure2D(self.variant_indexer.af)
            self._tiledAF = np.repeat(self.AF, 2, axis=0)
        if np.any(np.isnan(self.AF)):
            raise RuntimeError('Must provided allele frequencies')

        # ensure shape
        beta = xft.utils.ensure2D(beta)
        m, k = beta.shape
        self.m, self.k = variant_indexer.m, component_indexer.k_phenotypes
        if (self.m != m) or (self.k != k):
            raise RuntimeError('beta and indexers nonconformable')
        self._beta = beta

        if unscaled and standardized:
            self.beta_unscaled_unstandardized_diploid = beta / np.sqrt(2 * self.AF * (1 - self.AF))
            self.beta_unscaled_standardized_diploid = beta 
            self.beta_scaled_unstandardized_diploid = beta / np.sqrt(2 * self.AF * (1 - self.AF)) * np.sqrt(self.m_causal)
            self.beta_scaled_standardized_diploid = beta * np.sqrt(self.m_causal)
        elif scaled and unstandardized:
            self.beta_unscaled_unstandardized_diploid = beta / np.sqrt(self.m_causal)
            self.beta_unscaled_standardized_diploid = beta * np.sqrt(2 * self.AF * (1 - self.AF)) / np.sqrt(self.m_causal)
            self.beta_scaled_unstandardized_diploid = beta
            self.beta_scaled_standardized_diploid = beta * np.sqrt(2 * self.AF * (1 - self.AF))
        elif scaled and standardized:
            self.beta_unscaled_unstandardized_diploid = beta / np.sqrt(2 * self.AF * (1 - self.AF)) / np.sqrt(self.m_causal)
            self.beta_unscaled_standardized_diploid = beta / np.sqrt(self.m_causal)
            self.beta_scaled_unstandardized_diploid = beta / np.sqrt(2 * self.AF * (1 - self.AF))
            self.beta_scaled_standardized_diploid = beta
        elif unscaled and unstandardized:
            self.beta_unscaled_unstandardized_diploid = beta
            self.beta_unscaled_standardized_diploid = beta * np.sqrt(2 * self.AF * (1 - self.AF))
            self.beta_scaled_unstandardized_diploid = beta * np.sqrt(self.m_causal)
            self.beta_scaled_standardized_diploid = beta * np.sqrt(2 * self.AF * (1 - self.AF)) * np.sqrt(self.m_causal)
        else:
            raise TypeError("standardized, scaled must be bool")

        if component_indexer is None:
            self.component_indexer = xft.index.ComponentIndex.range_index(c)

    @property
    def m_causal(self) -> NDArray:  # number of non-zero betas per phenotype
        return (np.sum(self._beta != 0, axis=0))
    @property
    def offset(self) -> NDArray:  # add to raw multiple against raw (0,1,2)
        return np.sum(-2 * self.AF * self.beta_raw_diploid, axis=0)
    @property
    def beta_unscaled_unstandardized_haploid(self):
        return np.repeat(self.beta_unscaled_unstandardized_diploid, 2, axis=0)
    @property
    def beta_unscaled_standardized_haploid(self):
        return np.repeat(self.beta_unscaled_standardized_diploid, 2, axis=0)
    @property
    def beta_scaled_unstandardized_haploid(self):
        return np.repeat(self.beta_scaled_unstandardized_diploid, 2, axis=0)
    @property
    def beta_scaled_standardized_haploid(self):
        return np.repeat(self.beta_scaled_standardized_diploid, 2, axis=0)
    @property
    def beta_raw_diploid(self):
        return self.beta_unscaled_unstandardized_diploid
    @property
    def beta_raw_haploid(self):
        return self.beta_unscaled_unstandardized_haploid

    def corr(self):
        rr = np.corrcoef(self.beta_scaled_standardized_diploid, rowvar=False)
        names = self.component_indexer.phenotype_name.values
        return pd.DataFrame(rr, index=names, columns=names)


class GCTAEffects(AdditiveEffects):
    """Additive genetic effects object under GCTA infinitessimal model <CITE>

    Under this genetic architecture, all variants are causal and standardized genetic variants / sqrt(m) 
    have the user specified (possibly diagonal) variance covariance matrix

    
    Parameters
    ----------
    vg : Iterable | NDArray
        Vector of genetic variances or genetic variance/covariance matrix
    variant_indexer : xft.index.HaploidVariantIndex | xft.index.DiploidVariantIndex
        Variant indexer, will determine ploidy automatically
    component_indexer : xft.index.ComponentIndex, optional
        Phenotype component indexer, defaults to xft.index.ComponentIndex.RangeIndex if not provided
    """
    def __init__(self,
        vg: Union[Iterable, NDArray],  # either genetic variances or genetic variance/covariance matrix
        variant_indexer: Union[xft.index.HaploidVariantIndex,
            xft.index.DiploidVariantIndex] = None,
        component_indexer: xft.index.ComponentIndex = None,
        ):

        vg=np.array(vg)
        
        if len(vg.shape) == 1:
            vg=np.diag(vg)
        elif len(vg.shape) != 2:
            raise TypeError("vg must be a vector or matrix")
      
        m=variant_indexer.m
        k=component_indexer.k_phenotypes
        beta=np.random.multivariate_normal(np.zeros(k), vg, m)

        super().__init__(beta=beta,
                         variant_indexer=variant_indexer,
                         component_indexer=component_indexer,
                         scaled=True,
                         standardized=True)



class NonOverlappingEffects(AdditiveEffects):
    """Additive genetic effects object under non-infinitessimal model with no pleoitropy

    Under this genetic architecture, the genome is partitioned into k+1 components corresponding
    to k sets of variants corresponding to those causal for each trait together with a final
    set of variants not causal for any traits. Within each kth set of causal variants, 
    standardized variants are Gaussian with variance vg[k] / sqrt(proportions[k])
    
    Parameters
    ----------
    vg : Iterable
        Vector of genetic variances or genetic variance/covariance matrix
    proportions : Iterable
        Proportion of variants causal for each trait. If an extra value is provided, this will
        be the number of variants that are noncausal for all traits. Defaults to an equal
        number of variants per trait
    permute : bool
        Permute variants? If False, causal variants for each phenotype will fall into contiguous 
        blocks, defaults to True
    variant_indexer : xft.index.HaploidVariantIndex | xft.index.DiploidVariantIndex
        Variant indexer, will determine ploidy automatically
    component_indexer : xft.index.ComponentIndex, optional
        Phenotype component indexer, defaults to xft.index.ComponentIndex.RangeIndex if not provided
    """
    def __init__(self,
        vg: Iterable,
        proportions: Iterable = None,
        variant_indexer: Union[xft.index.HaploidVariantIndex,
            xft.index.DiploidVariantIndex] = None,
        component_indexer: xft.index.ComponentIndex = None,
        permute: bool = True,
        ):
     
        vg=np.array(vg).ravel()
   
        m=variant_indexer.m
        k=component_indexer.k_phenotypes   
        
        if len(vg) != k:
            raise ValueError("vg must have the same number of elements as there are phenotypes")
        
        if proportions is None:
            proportions = np.ones(k)
        elif not (len(proportions) in (k, k + 1)):
            raise ValueError("proportions must have the same number of elements as there are phenotypes or one more than that quantity")

        if permute:
            causal_inds = np.random.permutation(m)
        else:
            causal_inds = np.arange(m)

        m_causal = np.floor(xft.utils.to_simplex(proportions)*m).astype(int)
        ## if numbers per variant don't added up round by adding to biggest block
        if np.sum(m_causal) < m:
            max_ind = np.where(m_causal==np.max(m_causal))[0][0]
            m_causal[max_ind] += int(m - (np.sum(m_causal)))
        start_inds = np.cumsum(np.concatenate([[0], m_causal[:-1]]))
        stop_inds = np.cumsum(m_causal)

        beta = np.zeros((m,k))
        for j in range(k):
            beta[causal_inds[start_inds[j]:stop_inds[j]],j] = np.random.randn(m_causal[j]) * np.sqrt(vg[j])

        super().__init__(beta=beta,
                         variant_indexer=variant_indexer,
                         component_indexer=component_indexer,
                         scaled=True,
                         standardized=True)