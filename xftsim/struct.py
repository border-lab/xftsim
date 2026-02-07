import warnings
import numpy as np
import pandas as pd
import nptyping as npt
import xarray as xr
import dask.array as da
from dataclasses import dataclass, field
from nptyping import NDArray, Int8, Int64, Float64, Bool, Shape, Float, Int
from abc import ABC, abstractmethod
from typing import Any, Hashable, List, Iterable, Callable, Union, Dict, Tuple, Optional
from functools import cached_property
from scipy import interpolate
import xftsim as xft


class GeneticMap:
    """
    Map between physical and genetic distances.

    Parameters
    ----------
    chrom : Iterable
        Chromsomes variants are located on
    pos_bp : Iterable
        Physical positions of variants
    pos_cM : Iterable
        Map distances in cM

    Attributes
    __________
    frame : pd.DataFrame
        Pandas DataFrame with the above columns
    chroms : np.ndarray
        Unique chromosomes present in map
    """
    def __init__(self,
                 chrom: Iterable,
                 pos_bp: Iterable,
                 pos_cM: Iterable):
        self.frame = pd.DataFrame.from_dict(dict(chrom = chrom,
                                                 pos_bp = pos_bp,
                                                 pos_cM = pos_cM))
        self.chroms = np.unique(self.frame.chrom.values.astype(int)).astype(str)

    @classmethod
    def from_pyrho_maps(cls, paths: Iterable, sep='\t', **kwargs) -> "GeneticMap":
        """Construct genetic map objects from maps provided at https://github.com/popgenmethods/pyrho
        Please cite their work if you use their maps.
        
        Parameters
        ----------
        paths : Iterable
            Paths for each chromosome
        sep : str, optional
            Passed to pd.read_csv()
        **kwargs
            Additional arguments to pd.read_csv()
        
        Returns
        -------
        GeneticMap
        """
        gmap = pd.concat([pd.read_csv(path, sep='\t', **kwargs) for path in paths])
        chrom = np.char.lstrip(gmap.Chromosome.values.astype(str),'chr')
        pos_bp = gmap['Position(bp)'].values
        pos_cM = gmap['Map(cM)'].values
        return cls(chrom, pos_bp, pos_cM)


    def interpolate_cM_chrom(self, pos_bp: Iterable, chrom: str, **kwargs):
        """
        Interpolate cM values in a specified chromosome based on genetic map information.

        Parameters
        ----------
        pos_bp : Iterable
            Physical positions for which to interpolate cM values
        chrom : str
            Chromosome on which to interpolate
        **kwargs
            Additional keyword arguments to be passed to scipy.interpolate.interp1d.
        """ 
        subset = self.frame[self.frame.chrom==chrom]
        interpolator = interpolate.interp1d(x = subset.pos_bp.values,
                                            y = subset.pos_cM.values,
                                            **kwargs)
        return interpolator(pos_bp)


        if self._col_dim != 'variant':
            raise TypeError
        chroms = np.unique(self._obj.chrom.values.astype(int)).astype(str)
        for chrom in chroms:  
            rmap_chrom = rmap_df[rmap_df['Chromosome']=='chr'+chrom]
            interpolator = interpolate.interp1d(x = rmap_chrom['Position(bp)'].values,
                                                y = rmap_chrom['Map(cM)'].values,
                                                **kwargs)
            self._obj.pos_cM[self._obj.chrom==chrom] = interpolator(self._obj.pos_bp[self._obj.chrom==chrom])

 
                 

@xr.register_dataarray_accessor("xft")
class XftAccessor:
    """
    Accessor for Xarray DataArrays with specialized functionality for HaplotypeArray
    and PhenotypeArray objects.
    
    Parameters
    ----------
    xarray_obj : xarray.DataArray
        The DataArray to be accessed.
    
    Attributes
    ----------
    _obj : xarray.DataArray
        The DataArray to be accessed.
    _array_type : str
        The type of the DataArray, either 'HaplotypeArray' or 'componentArray'.
    _non_annotation_vars : list of str
        The non-annotation variables in the DataArray.
    _variant_vars : list of str
        The variant annotation variables in the DataArray.
    _sample_vars : list of str
        The sample annotation variables in the DataArray.
    _component_vars : list of str
        The component annotation variables in the DataArray.
    _row_dim : str
        The label of the row dimension.
    _col_dim : str
        The label of the column dimension.
    shape : tuple
        The shape of the DataArray.
    n : int
        The number of rows in the DataArray.
    data : numpy.ndarray
        The data in the DataArray.
    row_vars: list
        List of coordinate variable names for the row dimension.
    column_vars: list
        List of coordinate variable names for the column dimension.
    sample_mindex: pd.MultiIndex
        MultiIndex object for the 'sample' dimension, containing iid, fid, and sex columns.
    component_mindex: pd.MultiIndex
        MultiIndex object for the 'component' dimension, containing phenotype_name, component_name, and vorigin_relative columns.


    Raises
    ------
    NotImplementedError
        If the DataArray dimensions are not ('sample', 'variant') or ('sample', 'component').
    """
    
    def __init__(self, xarray_obj):
        self._obj = xarray_obj
        if self._obj.dims == ('sample', 'variant'):
            self._array_type = 'HaplotypeArray'
            self._non_annotation_vars = [
                'variant', 'vid', 'chrom', 'zero_allele', 'one_allele', 'af', 'hcopy', 'pos_bp', 'pos_cM']
            self._variant_vars = ['vid', 'chrom', 'zero_allele', 'one_allele', 'af']
            self._sample_vars = ['iid', 'fid', 'sex']
        elif self._obj.dims == ('sample', 'component'):
            self._array_type = 'componentArray'
            self._component_vars = ['phenotype_name', 'component_name', 'vorigin_relative']
            self._sample_vars = ['iid', 'fid', 'sex']
        else:
            raise NotImplementedError("Unsupported dimensions for DataArray.")
    
    @property
    def _row_dim(self):
        """
        The label of the row dimension.
        
        Returns
        -------
        str
            The label of the row dimension.
        """
        return self._obj.dims[0]

    @property
    def _col_dim(self):
        """
        The label of the column dimension.
        
        Returns
        -------
        str
            The label of the column dimension.
        """
        return self._obj.dims[1]

    def _check_sample_indexed(self):
        if self._row_dim != 'sample':
            raise TypeError("Array must have sample index")

    def _check_variant_indexed(self):
        if self._col_dim != 'variant':
            raise TypeError("Array must have variant index")

    @property
    def shape(self):
        """
        The shape of the DataArray.
        
        Returns
        -------
        tuple
            The shape of the DataArray.
        """
        return self._obj.shape

    @property
    def n(self):
        """
        The number of rows in the DataArray.
        
        Returns
        -------
        int
            The number of rows in the DataArray.
        """
        return self._obj.shape[0]

    @property
    def data(self):
        """
        The data in the DataArray.
        
        Returns
        -------
        numpy.ndarray
            The data in the DataArray.
        """
        return self._obj.data

    #### functions for constructing XftIndex objects ####
    def get_sample_indexer(self):
        """
        Returns an instance of `xft.index.SampleIndex` representing the sample indexer
        constructed from the input data.

        Raises
        ------
        NotImplementedError
            If `_row_dim` is not `'sample'`.

        Returns
        -------
        SampleIndex
            An instance of `xft.index.SampleIndex` constructed from the sample data in the
            input object.
        """
        if self._row_dim != 'sample':
            raise NotImplementedError
        return xft.index.SampleIndex(
            iid=self._obj.coords['sample'].iid.data,
            fid=self._obj.coords['sample'].fid.data,
            sex=self._obj.coords['sample'].sex.data,
            generation=self._obj.generation
        )

    def set_sample_indexer(self, value):
        raise NotImplementedError

    def get_variant_indexer(self):
        """
        Get the variant indexer of a HaplotypeArray.

        Returns
        -------
        xft.index.HaploidVariantIndex
            A HaploidVariantIndex object.
        """
        if self._col_dim != 'variant':
            raise NotImplementedError
        annotations = self.get_annotation_dict()
        if len(annotations) == 0:
            annotation_array = None
            annotation_names = None
        else:
            annotation_array, annotation_names = annotations.items()
        return xft.index.HaploidVariantIndex(
            vid=self._obj.coords['variant'].vid.data,
            chrom=self._obj.coords['variant'].chrom.data,
            h_copy=self._obj.coords['variant'].hcopy.data,
            zero_allele=self._obj.coords['variant'].zero_allele.data,
            one_allele=self._obj.coords['variant'].one_allele.data,
            af=self._obj.coords['variant'].af.data,
            pos_bp=self._obj.coords['variant'].pos_bp.data,
            pos_cM=self._obj.coords['variant'].pos_cM.data,
            annotation_array=annotation_array,
            annotation_names=annotation_names)

    def set_variant_indexer(self, value):
        raise NotImplementedError

    def get_component_indexer(self):
        """
        Get the component indexer of a PhenotypeArray.

        Returns
        -------
        xft.index.ComponentIndex
            A ComponentIndex object.
        """
        if self._col_dim != 'component':
            raise NotImplementedError
        return xft.index.ComponentIndex(
            phenotype_name=self._obj.coords['component'].phenotype_name.data,
            component_name=self._obj.coords['component'].component_name.data,
            vorigin_relative=self._obj.coords['component'].vorigin_relative.data,
            comp_type=self._obj.coords['component'].comp_type.data,
        )

    def reindex_components(self, value):
        """
        Reindex the components.

        Parameters
        ----------
        value : xft.index.ComponentIndex
            A ComponentIndex object.

        Returns
        -------
        PhenotypeArray
            A new PhenotypeArray object.

        """
        # ugly as hell, works for now
        return PhenotypeArray(self._obj.data,
                              component_indexer=value,
                              sample_indexer=self.get_sample_indexer(),
                              )
        # self._obj['phenotype_name'] = value.phenotype_name
        # self._obj['component_name'] = value.component_name
        # self._obj['vorigin_relative'] = value.vorigin_relative

    def get_row_indexer(self):
        """
        Get the row indexer.

        Returns
        -------
        xft.index.SampleIndex
            A SampleIndex object.

        Raises
        ------
        TypeError
            If the row dimension is not 'sample'.

        """
        if self._row_dim == 'sample':
            return self.get_sample_indexer()
        else:
            raise TypeError

    def set_row_indexer(self):
        raise NotImplementedError

    def get_column_indexer(self):
        """
        Get the column indexer object for the PhenotypeArray object.

        Returns
        -------
        xft.index.Indexer
            The indexer object based on the current column dimension.

        Raises
        ------
        TypeError
            If the current column dimension is not recognized.
        """
        if self._col_dim == 'variant':
            return self.get_variant_indexer()
        elif self._col_dim == 'component':
            return self.get_component_indexer()
        else:
            raise TypeError

    def set_column_indexer(self, value):
        """
        Set the column indexer object for the PhenotypeArray object.

        Parameters
        ----------
        value : xft.index.Indexer
            The new indexer object for the PhenotypeArray object.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If the current column dimension is not recognized.
        """
        if self._col_dim == 'variant':
            return self.set_variant_indexer(value)
        elif self._col_dim == 'component':
            return self.set_component_indexer(value)
        else:
            raise TypeError

    @property
    def row_vars(self):
        """
        Get the row coordinate variables for the PhenotypeArray object.

        Returns
        -------
        XftIndex
            The row coordinate variables of the row dimension.
        """
        return self.get_row_indexer()._coord_variables

    @property
    def column_vars(self):
        """
        Get the column coordinate variables for the DataArray object.

        Returns
        -------
        XftIndex
            The column coordinate variables of the current column dimension.
        """
        return self.get_column_indexer()._coord_variables

    # accessors for pd.MultiIndex objects
    @property
    def sample_mindex(self):
        """
        Get the sample multi-index for the PhenotypeArray object.

        Returns
        -------
        pd.MultiIndex
            A multi-index object containing sample IDs, family IDs, and sex information.
        
        Raises
        ------
        NotImplementedError
            If the current row dimension is not 'sample'.
        """
        if self._row_dim != 'sample':
            raise NotImplementedError
        df = pd.DataFrame.from_dict(dict(
            iid=self._obj.coords['sample'].iid.data,
            fid=self._obj.coords['sample'].fid.data,
            sex=self._obj.coords['sample'].sex.data,
        ))
        return pd.MultiIndex.from_frame(df)

    @property
    def component_mindex(self):
        """
        Get a Pandas MultiIndex object for the component dimension.

        Returns
        -------
        pandas.MultiIndex
            MultiIndex object with phenotype_name, component_name, and vorigin_relative
            as index levels.

        Raises
        ------
        NotImplementedError
            If the column dimension is not 'component'.
        """ 
        if self._col_dim != 'component':
            raise NotImplementedError
        df = pd.DataFrame.from_dict(dict(
            phenotype_name=self._obj.coords['component'].phenotype_name.data,
            component_name=self._obj.coords['component'].component_name.data,
            vorigin_relative=self._obj.coords['component'].vorigin_relative.data
        ))
        return pd.MultiIndex.from_frame(df)

    def standardize(self):
        out = self._obj.copy()
        out.data = xft.utils.standardize_array(self._obj.data)
        return out

    ############ HaplotypeArray properties ############

    def interpolate_cM(self,
                       gmap: GeneticMap,
                       # rmap_df: pd.DataFrame = xft.data.get_ceu_map(),
                       **kwargs):
        """
        Interpolate cM values based on genetic map information.
        Specific to HaplotypeArray objects.

        Parameters
        ----------
        gmap : GeneticMap
            Genetic map data
        **kwargs
            Additional keyword arguments to be passed to scipy.interpolate.interp1d.

        Raises
        ------
        TypeError
            If the column dimension is not 'variant'.
        ValueError
            If not all chromosomes required are present in the genetic map
        """
        if self._col_dim != 'variant':
            raise TypeError
        chroms = np.unique(self._obj.chrom.values.astype(int)).astype(str)
        if not (set(chroms) <= set(gmap.chroms)):
            raise ValueError('Not all chromosomes are present on specified genetic Map')
        for chrom in chroms:  
            self._obj.pos_cM[self._obj.chrom==chrom] = gmap.interpolate_cM_chrom(self._obj.pos_bp[self._obj.chrom==chrom], 
                                      chrom=chrom,
                                      **kwargs)
            # self._obj.pos_cM[self._obj.chrom==chrom] = interpolator(self._obj.pos_bp[self._obj.chrom==chrom])

    def use_empirical_afs(self):
        """
        Sets allele frequencies to the empirical frequencies.
        Specific to HaplotypeArray objects.
        
        Raises
        ------
        TypeError
            If `_col_dim` is not 'variant'.
        """
        if self._col_dim != 'variant':
            raise TypeError
        self._obj.af.values = np.repeat(self.af_empirical,2)

    @property
    def diploid_vid(self):
        """
        Diploid variant ID.
        Specific to HaplotypeArray objects.
        
        Returns
        -------
        numpy.ndarray
            Diploid variant IDs.
        
        Raises
        ------
        TypeError
            If `_col_dim` is not 'variant'.
        """
        if self._col_dim != 'variant':
            raise TypeError
        return self._obj.vid[::2]

    @property
    def diploid_chrom(self):
        """
        Diploid chromosome numbers.
        Specific to HaplotypeArray objects.
        
        Returns
        -------
        numpy.ndarray
            Diploid chromosome numbers.
        
        Raises
        ------
        TypeError
            If `_col_dim` is not 'variant'.
        """
        if self._col_dim != 'variant':
            raise TypeError
        return self._obj.chrom[::2]

    @property
    def generation(self):
        """
        Generation of the data.
        Specific to HaplotypeArray objects.
        
        Returns
        -------
        int
            Generation attribute.
        
        Raises
        ------
        TypeError
            If `_col_dim` is not 'variant'.
        """
        if self._col_dim != 'variant':
            raise TypeError
        return self.attrs['generation']

    @property
    def af_empirical(self):
        """
        Empirical allele frequencies.
        Specific to HaplotypeArray objects.
        
        Returns
        -------
        numpy.ndarray
            Empirical allele frequencies.
        
        Raises
        ------
        TypeError
            If `_col_dim` is not 'variant'.
        """
        if self._col_dim != 'variant':
            raise TypeError
        hap_AF = self._obj.mean(axis=0)
        AF = np.asarray(hap_AF.data[0::2] + hap_AF.data[1::2]) / 2.
        return AF

    @property
    def maf_empirical(self):
        """
        Empirical minor allele frequencies.
        Specific to HaplotypeArray objects.
        
        Returns
        -------
        numpy.ndarray
            Empirical minor allele frequencies.
        
        Raises
        ------
        TypeError
            If `_col_dim` is not 'variant'.
        """
        if self._col_dim != 'variant':
            raise TypeError
        tmp = self.af_empirical
        tmp2 = 1 - tmp
        return np.where(tmp < tmp2, tmp, tmp2)

    @property
    def m(self):
        """
        Return the number of distinct diploid variants.
        Specific to HaplotypeArray objects.
        
        Returns
        -------
        int:
            The number of distinct diploid variants in the array.
            
        Raises
        ------
        TypeError:
            If the `_col_dim` attribute is not equal to 'variant'.
        """
        if self._col_dim != 'variant':
            raise TypeError
        return self._obj.shape[1] // 2

    def to_diploid(self):
        """
        Convert the object to a diploid representation by adding the two haplotypes for each variant.
        Specific to HaplotypeArray objects.
        
        Raises
        ------
        TypeError:
            If the `_col_dim` attribute is not equal to 'variant'.
        """
        if self._col_dim != 'variant':
            raise TypeError
        data = self._obj[:, 0::2].data + self._obj[:, 1::2].data
        vind = self.get_variant_indexer().to_diploid()
        sind = self.get_sample_indexer()

        coord_dict = sind.coord_dict.copy()
        coord_dict.update(vind.coord_dict)
        ## convert to dask array if necessary
        return xr.DataArray(data=data,
                            dims=['sample', 'variant'],
                            coords=coord_dict,
                            name='GenotypeArray',
                            attrs={
                                'generation': self._obj.generation,
                            })

        # tmp = haplo[:,0::2].data + haplo[:,1::2].data
        # ind = haplo.xft.get_variant_indexer().to_diploid()
        # xft.struct.HaplotypeArray(tmp,sample_indexer=haplo.xft.get_sample_indexer(),
        #                           variant_indexer=ind)
        raise NotImplementedError()


    def to_diploid_standardized(self, af: NDArray = None, scale: bool = False):
        """
        Standardize the HaplotypeArray object and convert it to a diploid representation.
        Specific to HaplotypeArray objects.
        
        Parameters
        ----------
        af: NDArray, optional
            An array containing the allele frequencies of each variant. If not provided, empirical afs
            will with used
        scale: bool, optional
            Whether or not to scale the standardized array by the square root of the number of variants.
            
        Returns
        -------
        ndarray:
            A standardized diploid array where each variant is represented as the sum of two haplotypes.
            
        Raises
        ------
        TypeError:
            If the `_col_dim` attribute is not equal to 'variant'.
        """
        if self._col_dim != 'variant':
            raise TypeError
        if af is None:
            af = self.af_empirical
        if scale:
            return xft.utils.standardize_array_hw(self._obj.data, af) / np.sqrt(self.m)
        else:
            return xft.utils.standardize_array_hw(self._obj.data, af)

    def get_annotation_dict(self):
        """
        Return a dictionary of all annotation variables associated with the variants in the object.
        Specific to HaplotypeArray objects.
        
        Returns
        -------
        dict:
            A dictionary where the keys are the annotation variable names and the values are the corresponding arrays.
            
        Raises
        ------
        TypeError:
            If the `_col_dim` attribute is not equal to 'variant'.
        """
        if self._col_dim != 'variant':
            raise TypeError
        return {x[0]: x[1].values for x in self._obj.coords.variables.items() if 'variant' in x[1].dims and x[0] not in self._non_annotation_vars}

    def get_non_annotation_dict(self):
        """
        Return a dictionary of all non-annotation variables associated with the variants in the object.
        Specific to HaplotypeArray objects.
        
        Returns
        -------
        dict:
            A dictionary where the keys are the non-annotation variable names and the values are the corresponding arrays.
            
        Raises
        ------
        TypeError:
            If the `_col_dim` attribute is not equal to 'variant'.
        """
        if self._col_dim != 'variant':
            raise TypeError
        return {x[0]: x[1].values for x in self._obj.coords.variables.items() if 'variant' in x[1].dims and x[0] in self._variant_vars}

    # component index properties / methods
    def grep_component_index(self, keyword: str = 'phenotype'):
        """
        Returns the index array of components whose names contain the given keyword.
        Specific to PhenotypeArray objects.

        Parameters
        ----------
        keyword : str, optional
            The keyword to search for in component names, by default 'phenotype'.

        Returns
        -------
        XftIndex
            The index of components that match the given keyword.

        Raises
        ------
        TypeError
            If the column dimension is not 'component'.
        """
        if self._col_dim != 'component':
            raise TypeError
        pheno_cols = self._obj.component_name.values[self._obj.component_name.str.contains(
            keyword).values]
        component_index = self._obj.xft.get_component_indexer()[
            dict(component_name=pheno_cols)]
        return component_index


    def get_comp_type(self, ctype='intermediate'):
        """
        Returns the index array of components with comp_type==ctype
        Specific to PhenotypeArray objects.

        Returns
        -------
        XftIndex
            The index of components that match the given keyword.

        Raises
        ------
        TypeError
            If the column dimension is not 'component'.
        """
        if self._col_dim != 'component':
            raise TypeError
        pheno_cols = self._obj.component_name.values[self._obj.comp_type.str.contains(
            ctype).values]
        component_index = self._obj.xft.get_component_indexer()[
            dict(component_name=pheno_cols)]
        return component_index

    def get_intermediate_components(self):
        """
        Returns the index array of components with comp_type=='intermediate'
        Specific to PhenotypeArray objects.

        Returns
        -------
        XftIndex
            The index of components that match the given keyword.

        Raises
        ------
        TypeError
            If the column dimension is not 'component'.
        """
        return self.get_comp_type('intermediate')

    def get_outcome_components(self):
        """
        Returns the index array of components with comp_type=='outcome'
        Specific to PhenotypeArray objects.

        Returns
        -------
        XftIndex
            The index of components that match the given keyword.

        Raises
        ------
        TypeError
            If the column dimension is not 'component'.
        """
        return self.get_comp_type('outcome')


    @property
    def k_total(self):
        """
        Returns the total number of components.
        Specific to PhenotypeArray objects.

        Returns
        -------
        int
            The total number of components.

        Raises
        ------
        TypeError
            If the column dimension is not 'component'.
        """
        if self._col_dim != 'component':
            raise TypeError
        return self.shape[1]  # number of all phenotype components

    @property
    def k_phenotypes(self):
        """
        Returns the number of unique phenotype components.
        Specific to PhenotypeArray objects.

        Returns
        -------
        int
            The number of unique phenotype components.

        Raises
        ------
        TypeError
            If the column dimension is not 'component'.
        """
        if self._col_dim != 'component':
            raise TypeError
        return np.unique(self._obj.phenotype_name).shape[0]

    @property
    def all_phenotypes(self) -> np.ndarray:
        """
        Returns an array of all the unique phenotype component names.
        Specific to PhenotypeArray objects.

        Returns
        -------
        numpy.ndarray
            An array of all the unique phenotype component names.

        Raises
        ------
        TypeError
            If the column dimension is not 'component'.
        """

    @property
    def k_components(self) -> int:
        """
        Returns the number of unique component names.
        Specific to PhenotypeArray objects.

        Returns
        -------
        int
            The number of unique component names.

        Raises
        ------
        TypeError
            If the column dimension is not 'component'.
        """
        if self._col_dim != 'component':
            raise TypeError
        return np.unique(self._obj.component_name).shape[0]

    @property
    def all_components(self) -> np.ndarray:
        """
        Returns an array of all the unique component names.
        Specific to PhenotypeArray objects.

        Returns
        -------
        numpy.ndarray
            An array of all the unique component names.

        Raises
        ------
        TypeError
            If the column dimension is not 'component'.
        """
        if self._col_dim != 'component':
            raise TypeError
        return np.unique(self._obj.component_name)

    @property
    def k_relative(self) -> int:
        """
        Returns the number of unique origin relative values.
        Specific to PhenotypeArray objects.
        
        Returns
        -------
        int
            The number of unique origin relative values.
        
        Raises
        ------
        TypeError
            If the column dimension is not 'component'.
        """
        if self._col_dim != 'component':
            raise TypeError
        return np.unique(self._obj.vorigin_relative).shape[0]

    @property
    def all_relatives(self) -> np.ndarray:
        """
        Returns an array of all the unique origin relative values.
        Specific to PhenotypeArray objects.

        Returns
        -------
        numpy.ndarray
            An array of all the unique origin relative values.

        Raises
        ------
        TypeError
            If the column dimension is not 'component'.
        """
        if self._col_dim != 'component':
            raise TypeError
        return np.unique(self._obj.vorigin_relative)

    @property
    def k_current(self) -> int:
        """Returns the number of all current-gen specific components.
        Specific to PhenotypeArray objects.
        
        Raises
        ------
        TypeError
            If the column dimension is not 'component'.
        
        Returns
        -------
        int
            The number of all current-gen specific components.
        """
        if self._col_dim != 'component':
            raise TypeError
        return np.sum(self._obj.vorigin_relative == -1)

    def get_k_rel(self, rel) -> int:
        """Returns the number of components with the given relative origin.
        Specific to PhenotypeArray objects.
        
        Args:
            rel (int): The relative origin of the components.
        
        Raises:
            TypeError: If the column dimension is not 'component'.
            
        Returns:
            int: The number of components with the given relative origin.
        """
        if self._col_dim != 'component':
            raise TypeError
        return np.sum(self._obj.vorigin_relative == rel)

    @property
    def depth(self):
        """Returns the generational depth from binary relative encoding.
        Specific to PhenotypeArray objects.
        
        Raises
        ------
        TypeError
            If the column dimension is not 'component'.
        
        Returns
        -------
        Union[float, np.nan]
            The generational depth from binary relative encoding, or NaN if the relative origin is empty.
        """
        if self._col_dim != 'component':
            raise TypeError
        if len(self.vorigin_relative) != 0:
            return math.floor(math.log2(np.max(self._obj.vorigin_relative) + 2)) + 1
        else:
            return np.NaN

    def split_by_phenotype(self) -> Dict[str, pd.DataFrame]:
        """Splits the data by phenotype name.
        Specific to PhenotypeArray objects.
        
        Raises
        ------
        TypeError
            If the column dimension is not 'component'.
        
        Returns
        -------
        Dict[str, pd.DataFrame]
            A dictionary of dataframes, where the keys are the unique phenotype names and the values are dataframes containing the data for each phenotype.
        """
        if self._col_dim != 'component':
            raise TypeError
        return {phenotype: pheno.loc[:, pheno.phenotype_name == phenotype] for phenotype in self.all_phenotypes}

    def split_by_component(self):
        """Splits the data by component name.
        Specific to PhenotypeArray objects.
        
        Raises
        ------
        TypeError
            If the column dimension is not 'component'.
        
        Returns
        -------
        Dict[str, pd.DataFrame]
            A dictionary of dataframes, where the keys are the unique component names and the values are dataframes containing the data for each component.
        """
        if self._col_dim != 'component':
            raise TypeError
        return {component: pheno.loc[:, pheno.component_name == component] for component in self.all_components}

    def split_by_vorigin(self) -> Dict[int, pd.DataFrame]:
        """Splits the data by relative origin.
        Specific to PhenotypeArray objects.
        
        Raises
        ------
        TypeError
            If the column dimension is not 'component'.
        
        Returns
        -------
        Dict[int, pd.DataFrame]
            A dictionary of dataframes, where the keys are the unique relative origins and the values are dataframes containing the data for each relative origin.
        """
        if self._col_dim != 'component':
            raise TypeError
        return {vorigin: pheno.loc[:, pheno.vorigin_relative == vorigin] for vorigin in self.all_relatives}

    def split_by_phenotype_vorigin(self) -> Dict[Tuple[str, int], pd.DataFrame]:
        """Splits the data by phenotype name and relative origin.
        Specific to PhenotypeArray objects.
        
        Raises
        ------
        TypeError
        If the column dimension is not 'component'.
        
        Returns
        -------
        Dict[Tuple[str, int], pd.DataFrame]
            A dictionary of dataframes, where the keys are tuples of phenotype name and relative origin and the values are dataframes containing the data for each combination of phenotype name and relative origin.
        """
        if self._col_dim != 'component':
            raise TypeError
        return {(phenotype, vorigin): pheno.loc[:, (pheno.phenotype_name == phenotype) ^ (pheno.vorigin_relative == vorigin)] for phenotype in self.all_phenotypes for vorigin in self.all_relatives}

    def as_pd(self, prettify: bool = True):
        """Returns the data as a Pandas DataFrame.
        Specific to PhenotypeArray objects.
        
        Parameters
        ----------
        prettify : bool, optional
            If True, the multi-index columns will be prettified by replacing -1, 0, 1 with 'proband', 'mother', 'father', respectively.
        
        Raises
        ------
        TypeError
            If the column dimension is not 'component'.
        
        Returns
        -------
        pd.DataFrame
            A Pandas DataFrame representing the data.
        """
        if self._col_dim != 'component':
            raise TypeError
        component_mind = self.component_mindex
        if prettify:
            fr = component_mind.to_frame()
            fr['vorigin_relative'].replace(
                [-1, 0, 1], ['proband', 'mother', 'father'], inplace=True)
            component_mind = pd.MultiIndex.from_frame(fr)
        return xr.DataArray(self._obj.data,
                            dims=('sample', 'component'),
                            coords={'sample': self.sample_mindex,
                                    'component': component_mind},
                            ).to_pandas()

    def __getitem__(self, *args):
        """
        Retrieve a subset of the data with the given indices.

        Parameters
        ----------
        args : tuple
            Tuple of indices to retrieve the subset of data. It can be one of the following:
            
            - A dictionary where the keys are column names and the values are the indices of the columns to retrieve.
            - Two positional arguments representing row and column indices, respectively.

        Returns
        -------
        xr.DataArray
            The subset of data corresponding to the given indices.

        Raises
        ------
        KeyError
            If any of the indices provided are invalid.

        """
        # indexing with a dict
        argv = args[0]
        # argv = args[0]
        if isinstance(argv, dict):
            row_dict = {key: value for (
                key, value) in argv.items() if key in self.row_vars}
            col_dict = {key: value for (
                key, value) in argv.items() if key in self.column_vars}
            # TODO possible gotcha with extra args
            if len(row_dict) == 0:
                row_dict = slice(None)
            if len(col_dict) == 0:
                col_dict = slice(None)

            row_indices = self.get_row_indexer()[row_dict].unique_identifier
            column_indices = self.get_column_indexer()[
                col_dict].unique_identifier
        # indexing with two postional args
        else:
            # argv = tuple(*(args[0]))
            assert len(argv) == 2, "provide 2 positional arguments"
            # row index
            if argv[0] is None or argv[0] is slice(None):
                row_indices = slice(None)
            elif not isinstance(argv[0], xft.index.SampleIndex):
                raise KeyError
            else:
                row_indices = argv[0].unique_identifier
            # col index
            isvarindex_1 = isinstance(argv[1], xft.index.HaploidVariantIndex)
            isvarindex_2 = isinstance(argv[1], xft.index.DiploidVariantIndex)
            isvarindex = (isvarindex_1 or isvarindex_2) and (
                self._col_dim == 'variant')
            iscomindex = (isinstance(
                argv[1], xft.index.ComponentIndex) and self._col_dim == 'component')
            # print(argv)
            # print(not iscomindex)
            # print(not isinstance(argv[1], xft.index.ComponentIndex))
            if argv[1] is None or argv[1] is slice(None) or argv[1] is slice(None, None, None):
                column_indices = slice(None)
            elif (not iscomindex) and (not isvarindex):
                raise KeyError
            else:
                column_indices = argv[1].unique_identifier
            # indexing by XftIndex
        return self._obj.loc[row_indices, column_indices]

    def __setitem__(self, args, data):
        """
        Set the values of a subset of the data with the given indices.

        Parameters
        ----------
        args : tuple
            Tuple of indices to set the values of the subset of data. It can be one of the following:
            
            - A dictionary where the keys are column names and the values are the indices of the columns to set the values.
            - Two positional arguments representing row and column indices, respectively.
        data : xr.DataArray
            The data to set in the specified subset of data.

        Raises
        ------
        KeyError
            If any of the indices provided are invalid.
        """
        # indexing with a dict
        argv = args  # [0]
        # argv = args[0]
        if isinstance(argv, dict):
            row_dict = {key: value for (
                key, value) in argv.items() if key in self.row_vars}
            col_dict = {key: value for (
                key, value) in argv.items() if key in self.column_vars}
            # TODO possible gotcha with extra args
            if len(row_dict) == 0:
                row_dict = slice(None)
            if len(col_dict) == 0:
                col_dict = slice(None)

            row_indices = self.get_row_indexer()[row_dict].unique_identifier
            column_indices = self.get_column_indexer()[
                col_dict].unique_identifier
        # indexing with two postional args
        else:
            # argv = tuple(*(args[0]))
            assert len(argv) == 2, "provide 2 positional arguments"
            # row index
            if argv[0] is None or argv[0] is slice(None):
                row_indices = slice(None)
            elif not isinstance(argv[0], xft.index.SampleIndex):
                raise KeyError
            else:
                row_indices = argv[0].unique_identifier
            # col index
            isvarindex_1 = isinstance(argv[1], xft.index.HaploidVariantIndex)
            isvarindex_2 = isinstance(argv[1], xft.index.DiploidVariantIndex)
            isvarindex = (isvarindex_1 or isvarindex_2) and (
                self._col_dim == 'variant')
            iscomindex = (isinstance(
                argv[1], xft.index.ComponentIndex) and self._col_dim == 'component')

            if argv[1] is None or argv[1] is slice(None) or argv[1] is slice(None, None, None):
                column_indices = slice(None)
            elif (not iscomindex) and (not isvarindex):
                raise KeyError
            else:
                column_indices = argv[1].unique_identifier
            # indexing by XftIndex
        self._obj.loc[row_indices, column_indices] = data


class HaplotypeArray:
    """
    Base class for haplotype arrays (legacy).

    This is a minimal base class that defines the common interface for all
    haplotype array implementations. Subclasses should implement the abstract
    properties and methods.

    Attributes
    ----------
    n : int
        Number of samples.
    m : int
        Number of diploid variants.
    generation : int
        Generation number.
    """

    def __init__(self, generation: int = 0):
        self._generation = generation

    @property
    def n(self) -> int:
        """Number of samples."""
        raise NotImplementedError("Subclasses must implement n")

    @property
    def m(self) -> int:
        """Number of diploid variants."""
        raise NotImplementedError("Subclasses must implement m")

    @property
    def generation(self) -> int:
        """Generation number."""
        return self._generation

    @generation.setter
    def generation(self, value: int):
        self._generation = value

    @property
    def diploid_genotypes(self) -> np.ndarray:
        """
        Return diploid genotype counts (0, 1, or 2) as 2D array (n, m).
        Subclasses must implement this.
        """
        raise NotImplementedError("Subclasses must implement diploid_genotypes")

    def matvec(self, v: np.ndarray) -> np.ndarray:
        """
        Matrix-vector product: G @ v where G is the (n, m) diploid genotype matrix.

        Parameters
        ----------
        v : np.ndarray
            Vector of shape (m,) or (m, k) to multiply.

        Returns
        -------
        np.ndarray
            Result of shape (n,) or (n, k).
        """
        return self.diploid_genotypes @ v

    def rmatvec(self, v: np.ndarray) -> np.ndarray:
        """
        Right matrix-vector product: G.T @ v where G is the (n, m) diploid genotype matrix.

        Parameters
        ----------
        v : np.ndarray
            Vector of shape (n,) or (n, k) to multiply.

        Returns
        -------
        np.ndarray
            Result of shape (m,) or (m, k).
        """
        return self.diploid_genotypes.T @ v

    def standardized_matvec(self, v: np.ndarray, af: np.ndarray = None) -> np.ndarray:
        """
        Standardized matrix-vector product using centered (and optionally scaled) genotypes.

        Parameters
        ----------
        v : np.ndarray
            Vector of shape (m,) or (m, k) to multiply.
        af : np.ndarray, optional
            Allele frequencies for centering. If None, uses empirical frequencies.

        Returns
        -------
        np.ndarray
            Result of shape (n,) or (n, k).
        """
        G_std = self.to_diploid_standardized(af=af, scale=False)
        return G_std @ v

    def standardized_rmatvec(self, v: np.ndarray, af: np.ndarray = None) -> np.ndarray:
        """
        Standardized right matrix-vector product.

        Parameters
        ----------
        v : np.ndarray
            Vector of shape (n,) or (n, k) to multiply.
        af : np.ndarray, optional
            Allele frequencies for centering. If None, uses empirical frequencies.

        Returns
        -------
        np.ndarray
            Result of shape (m,) or (m, k).
        """
        G_std = self.to_diploid_standardized(af=af, scale=False)
        return G_std.T @ v

    @property
    def af_empirical(self) -> np.ndarray:
        """
        Compute empirical allele frequencies.
        Subclasses must implement this.
        """
        raise NotImplementedError("Subclasses must implement af_empirical")

    def to_diploid_standardized(self, af: np.ndarray = None, scale: bool = False) -> np.ndarray:
        """
        Return standardized diploid genotypes.
        Subclasses must implement this.
        """
        raise NotImplementedError("Subclasses must implement to_diploid_standardized")


class XarrayHaplotypeArray(HaplotypeArray):
    """
    Haplotype array backed by xarray DataArray.

    Stores haplotypes as a 2D array with shape (n_samples, 2*m_variants)
    where haplotypes are interleaved.

    Parameters
    ----------
    haplotypes : np.ndarray, optional
        A 2D array of binary haplotypes. If not provided, default to None.
    variant_indexer : xft.index.HaploidVariantIndex, optional
        A haploid variant indexer. If not provided, default to None.
    sample_indexer : xft.index.SampleIndex, optional
        A sample indexer. If not provided, default to None.
    generation : int, optional
        The generation number associated with the haplotypes. Default to 0.
    n : int, optional
        The number of samples. Required if `sample_indexer` is not provided.
    m : int, optional
        The number of variants. Required if `variant_indexer` is not provided.
    dask : bool, optional
        Create a Dask array?
    """

    def __new__(cls,
                haplotypes: NDArray[Shape["*, *"], Int8] = None,
                variant_indexer: "xft.index.HaploidVariantIndex" = None,
                sample_indexer: "xft.index.SampleIndex" = None,
                generation: int = 0,
                n: int = None,
                m: int = None,
                dask: bool = False,
                **kwargs,
                ) -> xr.DataArray:
        # obtain n,m if missing
        if haplotypes is not None:
            assert haplotypes.shape[1] % 2 == 0
            n, m = haplotypes.shape
            m = m // 2
        # populate variant_indexer if missing
        if variant_indexer is None:
            variant_indexer = xft.index.DiploidVariantIndex(
                m=m, n_chrom=np.min([22, m])).to_haploid()
            if haplotypes is not None:
                warnings.warn(
                    'Using empirical allele frequencies as variant_indexer not provided')
                tmp = np.mean(haplotypes, 0)
                variant_indexer.af = np.repeat((tmp[0::2] + tmp[1::2]) * .5, 2)

        # populate sample_indexer if missing
        if sample_indexer is None:
            sample_indexer = xft.index.SampleIndex(n=n, generation=generation)
        # populate haplotypes with NaN if not provided
        if haplotypes is None:
            warnings.warn('Defaulting allele counts to -1', stacklevel=2)
            data = np.full((sample_indexer.n, variant_indexer.m * 2),
                           fill_value=-1, dtype=np.int8)
        else:
            data = haplotypes.astype(np.int8)

        coord_dict = sample_indexer.coord_dict.copy()
        coord_dict.update(variant_indexer.coord_dict)
        # convert to dask array if necessary
        if dask and not isinstance(data, da.Array):
            data = da.asarray(data)
        return xr.DataArray(data=data,
                            dims=['sample', 'variant'],
                            coords=coord_dict,
                            name='HaplotypeArray',
                            attrs={
                                'generation': generation,
                            })


class PhenotypeArray:    
    """
    An array that stores phenotypes for a set of individuals.
    Dummy class used for generation of DataArrays and static methods 

    Parameters
    ----------
    components : ndarray, optional
        n x 2m array of binary haplotypes.
    component_indexer : xft.index.ComponentIndex, optional
        Indexer for components.
    sample_indexer : xft.index.SampleIndex, optional
        Indexer for samples.
    generation : int, optional
        The generation this PhenotypeArray belongs to.
    n : int, optional
        The number of samples.
    k_total : int, optional
        The total number of components.

    Returns
    -------
    xr.DataArray
        The initialized PhenotypeArray.

    Raises
    ------
    AssertionError
        If `components` is provided, then `n` and `k_total` must not be provided.
        If `component_indexer` is provided, then `k_total` must not be provided.
        If `sample_indexer` is provided, then `n` must not be provided.
        If `components` is provided and `sample_indexer` is provided, then the shape of
        `components` must match the size of the sample dimension of `sample_indexer`.
        If `components` is provided and `component_indexer` is provided, then the shape
        of `components` must match the size of the component dimension of `component_indexer`.
        If `component_indexer` is provided, then the size of the component dimension of
        `component_indexer` must match `k_total`.
    """
    def __new__(cls,
                # n x 2m array of binary haplotypes
                components: NDArray[Shape["*, *"], Float] = None,
                component_indexer: xft.index.ComponentIndex = None,
                sample_indexer: xft.index.SampleIndex = None,
                generation: int = 0,
                n: int = None,
                k_total: int = None,
                ):
        # ensure components is conformable with indexers
        if components is not None:
            assert n is None, "Provide n OR components"
            assert k_total is None, "Provide k_total OR components"
            # todo verify this doesn't induce copy
            components = np.array(components)
            if sample_indexer is not None:
                assert components.shape[0] == sample_indexer.n, "Noncomformable sample_indexer"
            if component_indexer is not None:
                assert components.shape[1] == component_indexer.k_total, "Noncomformable component_indexer"
        # obtain dimensions if necessary
        if k_total is not None:
            assert component_indexer is None, "Provide k_total OR component_indexer"
            component_indexer = xft.index.ComponentIndex(k_total=k_total)
        if n is not None:
            assert sample_indexer is None, "Provide n OR sample_indexer"
            sample_indexer = xft.index.SampleIndex(n=n)
        k_total, n = component_indexer.k_total, sample_indexer.n
        # initialize component array if necessary
        if components is None:
            components = np.full((n, k_total), fill_value=np.NaN)

        coord_dict = sample_indexer.coord_dict.copy()
        coord_dict.update(component_indexer.coord_dict)
        return xr.DataArray(data=components,
                            dims=['sample', 'component'],
                            coords=coord_dict,
                            name='PhenotypeArray',
                            attrs={
                                'generation': generation,
                            })

    @staticmethod
    def from_product(
        phenotype_name: Iterable,
        component_name: Iterable,
        vorigin_relative: Iterable,
        components: xr.DataArray = None,
        sample_indexer: xft.index.SampleIndex = None,
        generation: int = None,
        haplotypes: xr.DataArray = None,
        n: int = None,
    ) -> xr.DataArray:
        """
        Create a PhenotypeArray from a product of names.

        Parameters
        ----------
        phenotype_name : iterable
            The names of the phenotypes.
        component_name : iterable
            The names of the components.
        vorigin_relative : iterable
            The relative origins of each component.
        components : xr.DataArray, optional
            The array to use as the components.
        sample_indexer : xft.index.SampleIndex, optional
            The sample indexer to use.
        generation : int, optional
            The generation of the PhenotypeArray.
        haplotypes : xr.DataArray, optional
            The haplotypes to use.
        n : int, optional
            The number of samples to use.

        Returns
        -------
        xr.DataArray
            The new PhenotypeArray.

        Raises
        ------
        AssertionError
            If exactly one of `generation` and `sample_indexer` is provided, or exactly one
            of `haplotypes` and `sample_indexer`/`generation` or `n`/`generation` is provided.
        """
        # use either haplotypes xOR sample_indexer/generation xOR n/generation
        bool_gsi = bool(generation is not None and sample_indexer is not None)
        bool_h = bool(haplotypes is not None)
        bool_n = bool(n is not None and generation is not None)
        assert bool_gsi ^ bool_h ^ bool_n
        if bool_n:
            sample_indexer = xft.index.SampleIndex(n=n, generation=generation)
        elif bool_h:
            generation = haplotypes.xft.generation
            sample_indexer = haplotypes.xft.get_sample_indexer()
        component_indexer = xft.index.ComponentIndex.from_product(
            phenotype_name, component_name, vorigin_relative)
        return PhenotypeArray(
            components=components,
            component_indexer=component_indexer,
            sample_indexer=sample_indexer,
            generation=generation,
        )

    @staticmethod
    def _test():
        generation = 0
        n = 3
        m = 10
        n_chrom = 10
        haplotypes = np.full((n, m * 2), fill_value=-1, dtype=np.int8)
        variant_indexer = xft.index.DiploidVariantIndex(
            m=m, n_chrom=n_chrom).to_haploid()
        sample_indexer = xft.index.SampleIndex(n=n, generation=generation)
        XarrayHaplotypeArray(haplotypes, generation=generation)


@dataclass(frozen=True)
class SampleMeta:
    """
    Immutable metadata for samples/individuals.

    Parameters
    ----------
    iid : np.ndarray
        Individual IDs (required).
    fid : np.ndarray, optional
        Family IDs. Defaults to iid if not provided.
    sex : np.ndarray, optional
        Biological sex (0=female, 1=male). Defaults to alternating 0,1.
    generation : int, optional
        Generation number. Default is 0.
    extra : dict, optional
        Arbitrary metadata arrays (ancestry PCs, batch IDs, etc.).

    Attributes
    ----------
    n : int
        Number of individuals.
    n_fam : int
        Number of unique families.
    n_female : int
        Number of biological females (sex=0).
    n_male : int
        Number of biological males (sex=1).
    """
    iid: np.ndarray
    fid: np.ndarray = None
    sex: np.ndarray = None
    generation: int = 0
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        # Use object.__setattr__ since dataclass is frozen
        iid = np.asarray(self.iid)
        object.__setattr__(self, 'iid', iid)

        if self.fid is None:
            object.__setattr__(self, 'fid', iid.copy())
        else:
            object.__setattr__(self, 'fid', np.asarray(self.fid))

        if self.sex is None:
            n = len(iid)
            sex = np.tile([0, 1], (n + 1) // 2)[:n]
            object.__setattr__(self, 'sex', sex)
        else:
            object.__setattr__(self, 'sex', np.asarray(self.sex))

        # Validate and convert extra arrays
        n = len(iid)
        validated = {}
        for k, v in self.extra.items():
            arr = np.asarray(v)
            if len(arr) != n:
                raise ValueError(
                    f"extra['{k}'] has length {len(arr)}, expected {n}"
                )
            validated[k] = arr
        object.__setattr__(self, 'extra', validated)

    @property
    def n(self) -> int:
        """Number of individuals."""
        return len(self.iid)

    @property
    def n_fam(self) -> int:
        """Number of unique families."""
        return len(np.unique(self.fid))

    @property
    def n_female(self) -> int:
        """Number of biological females (sex=0)."""
        return int(np.sum(self.sex == 0))

    @property
    def n_male(self) -> int:
        """Number of biological males (sex=1)."""
        return int(np.sum(self.sex == 1))

    @property
    def unique_identifier(self) -> np.ndarray:
        """
        Unique identifier for each sample, combining generation, iid, and fid.
        Format: '{generation}.{iid}.{fid}'
        """
        return np.array([
            f"{self.generation}.{i}.{f}"
            for i, f in zip(self.iid, self.fid)
        ])

    def subset(self, idx) -> "SampleMeta":
        """Return a new SampleMeta with a subset of samples."""
        return SampleMeta(
            iid=self.iid[idx].copy(),
            fid=self.fid[idx].copy(),
            sex=self.sex[idx].copy(),
            generation=self.generation,
            extra={k: v[idx].copy() for k, v in self.extra.items()},
        )

    def with_generation(self, generation: int) -> "SampleMeta":
        """Return a new SampleMeta with a different generation."""
        return SampleMeta(
            iid=self.iid,
            fid=self.fid,
            sex=self.sex,
            generation=generation,
            extra=self.extra,
        )

    def to_sample_index(self) -> "xft.index.SampleIndex":
        """
        Convert to legacy SampleIndex for compatibility with PhenotypeArray.

        Returns
        -------
        xft.index.SampleIndex
            A SampleIndex with the same data.
        """
        return xft.index.SampleIndex(
            iid=self.iid.astype(str),
            fid=self.fid.astype(str),
            sex=self.sex,
            generation=self.generation,
        )

    def __repr__(self) -> str:
        return (f"SampleMeta(n={self.n}, n_fam={self.n_fam}, "
                f"n_female={self.n_female}, n_male={self.n_male}, "
                f"generation={self.generation})")


@dataclass(frozen=True)
class VariantMeta:
    """
    Immutable metadata for genetic variants.

    Parameters
    ----------
    vid : np.ndarray
        Variant IDs (required).
    chrom : np.ndarray, optional
        Chromosome for each variant.
    pos_bp : np.ndarray, optional
        Base pair position.
    pos_cM : np.ndarray, optional
        Centimorgan position.
    af : np.ndarray, optional
        Allele frequencies.
    zero_allele : np.ndarray, optional
        Reference allele (e.g., 'A').
    one_allele : np.ndarray, optional
        Alternate allele (e.g., 'G').
    extra : dict, optional
        Arbitrary metadata arrays (annotation flags, etc.).

    Attributes
    ----------
    m : int
        Number of variants.
    """
    vid: np.ndarray
    chrom: np.ndarray = None
    pos_bp: np.ndarray = None
    pos_cM: np.ndarray = None
    af: np.ndarray = None
    zero_allele: np.ndarray = None
    one_allele: np.ndarray = None
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        # Use object.__setattr__ since dataclass is frozen
        vid = np.asarray(self.vid)
        object.__setattr__(self, 'vid', vid)
        m = len(vid)

        if self.chrom is not None:
            object.__setattr__(self, 'chrom', np.asarray(self.chrom))
        if self.pos_bp is not None:
            object.__setattr__(self, 'pos_bp', np.asarray(self.pos_bp))
        if self.pos_cM is not None:
            object.__setattr__(self, 'pos_cM', np.asarray(self.pos_cM))
        if self.af is not None:
            object.__setattr__(self, 'af', np.asarray(self.af))
        if self.zero_allele is not None:
            object.__setattr__(self, 'zero_allele', np.asarray(self.zero_allele))
        if self.one_allele is not None:
            object.__setattr__(self, 'one_allele', np.asarray(self.one_allele))

        # Validate and convert extra arrays
        validated = {}
        for k, v in self.extra.items():
            arr = np.asarray(v)
            if len(arr) != m:
                raise ValueError(
                    f"extra['{k}'] has length {len(arr)}, expected {m}"
                )
            validated[k] = arr
        object.__setattr__(self, 'extra', validated)

    def __getitem__(self, key: str) -> np.ndarray:
        """Access core fields or extras by name: variants['coding']."""
        core = {'vid', 'chrom', 'pos_bp', 'pos_cM', 'af', 'zero_allele', 'one_allele'}
        if key in core:
            val = getattr(self, key)
            if val is None:
                raise KeyError(f"Field '{key}' is None")
            return val
        return self.extra[key]

    @property
    def m(self) -> int:
        """Number of variants."""
        return len(self.vid)

    def subset(self, idx) -> "VariantMeta":
        """Return a new VariantMeta with a subset of variants."""
        return VariantMeta(
            vid=self.vid[idx].copy(),
            chrom=self.chrom[idx].copy() if self.chrom is not None else None,
            pos_bp=self.pos_bp[idx].copy() if self.pos_bp is not None else None,
            pos_cM=self.pos_cM[idx].copy() if self.pos_cM is not None else None,
            af=self.af[idx].copy() if self.af is not None else None,
            zero_allele=self.zero_allele[idx].copy() if self.zero_allele is not None else None,
            one_allele=self.one_allele[idx].copy() if self.one_allele is not None else None,
            extra={k: v[idx].copy() for k, v in self.extra.items()},
        )

    def to_variant_index(self, af: np.ndarray = None) -> "xft.index.DiploidVariantIndex":
        """
        Convert to legacy DiploidVariantIndex for compatibility.

        Parameters
        ----------
        af : np.ndarray, optional
            Allele frequencies. If not provided, uses stored af or NaN.

        Returns
        -------
        xft.index.DiploidVariantIndex
            A DiploidVariantIndex with the same data.
        """
        return xft.index.DiploidVariantIndex(
            vid=self.vid,
            chrom=self.chrom,
            pos_bp=self.pos_bp,
            pos_cM=self.pos_cM,
            af=af if af is not None else self.af,
            zero_allele=self.zero_allele,
            one_allele=self.one_allele,
        )

    def __repr__(self) -> str:
        fields = [f"m={self.m}"]
        if self.chrom is not None:
            n_chrom = len(np.unique(self.chrom))
            fields.append(f"n_chrom={n_chrom}")
        if self.af is not None:
            fields.append("af=True")
        return f"VariantMeta({', '.join(fields)})"


# ---------------------------------------------------------------------------
# HaplotypeOperator ABC (new — typed against by Architecture, EffectSpec, etc.)
# ---------------------------------------------------------------------------

class HaplotypeOperator(ABC):
    """
    Abstract base class for all genotype representations.

    Concrete implementations:
    - DenseHaplotypeArray — NumPy-backed (n, m, 2) array
    - GraphHaplotypeOperator — GRG wrapper (Phase 4)
    """

    # samples: SampleMeta  — set by concrete implementations
    # variants: VariantMeta — set by concrete implementations

    @property
    @abstractmethod
    def n(self) -> int:
        ...

    @property
    @abstractmethod
    def m(self) -> int:
        ...

    @abstractmethod
    def matvec(self, v: np.ndarray) -> np.ndarray:
        """Diploid G @ v."""
        ...

    @abstractmethod
    def rmatvec(self, v: np.ndarray) -> np.ndarray:
        """G.T @ v."""
        ...

    @abstractmethod
    def matvec_maternal(self, v: np.ndarray) -> np.ndarray:
        """hap[:,:,0] @ v."""
        ...

    @abstractmethod
    def matvec_paternal(self, v: np.ndarray) -> np.ndarray:
        """hap[:,:,1] @ v."""
        ...

    @abstractmethod
    def standardized_matvec(self, v: np.ndarray, af: np.ndarray = None) -> np.ndarray:
        """Standardized diploid matvec."""
        ...

    @abstractmethod
    def recompute_af(self) -> np.ndarray:
        """Recompute empirical allele frequencies from current data."""
        ...

    @abstractmethod
    def to_dense(self) -> "DenseHaplotypeArray":
        """Materialize as a DenseHaplotypeArray."""
        ...

    @abstractmethod
    def meiosis(self, assignment, recombination_map) -> "HaplotypeOperator":
        """
        Perform meiosis to produce offspring haplotypes.

        Parameters
        ----------
        assignment : NMateAssignment
            Mate assignment with maternal/paternal indices and offspring metadata.
        recombination_map : RecombinationMap
            Recombination probabilities between loci.

        Returns
        -------
        HaplotypeOperator
            Offspring haplotypes.
        """
        ...

    @abstractmethod
    def __getitem__(self, key) -> "HaplotypeOperator":
        """Subset by sample/variant indices."""
        ...


class DenseHaplotypeArray(HaplotypeArray, HaplotypeOperator):
    """
    Dense numpy-backed haplotype array implementing both old HaplotypeArray
    interface and new HaplotypeOperator ABC.

    Stores haplotypes as a 3D array with shape (n_samples, m_variants, 2)
    where the last dimension represents the two haplotype copies.
    Convention: genotypes[:,:,0] = maternal, genotypes[:,:,1] = paternal.

    Parameters
    ----------
    genotypes : np.ndarray
        3D array of shape (n, m, 2) containing haplotype data.
    generation : int, optional
        Generation number. Default is 0.
    samples : SampleMeta, optional
        Sample metadata (iid, fid, sex).
    variants : VariantMeta, optional
        Variant metadata (vid, chrom, pos_bp, pos_cM, af, alleles).
    """

    def __init__(
        self,
        genotypes: NDArray[Shape["*, *, *"], Int8],
        generation: int = 0,
        samples: Optional[SampleMeta] = None,
        variants: Optional[VariantMeta] = None,
    ) -> None:
        # call base class init
        super().__init__(generation=generation)

        # Validate genotypes
        if genotypes is None:
            genotypes = np.empty((0, 0, 2), dtype=np.int8)
        else:
            genotypes = np.asarray(genotypes, dtype=np.int8)

        if genotypes.ndim != 3 or genotypes.shape[2] != 2:
            raise ValueError(
                f"genotypes must be 3-D with last dim = 2, got shape {genotypes.shape}"
            )

        self.genotypes = genotypes
        n, m, _ = genotypes.shape

        # Create default SampleMeta if not provided
        if samples is None:
            iid = np.arange(n, dtype=np.int64)
            self.samples = SampleMeta(iid=iid, generation=generation)
        else:
            if samples.n != n:
                raise ValueError(
                    f"samples.n ({samples.n}) must match genotypes.shape[0] ({n})"
                )
            # Ensure SampleMeta has correct generation
            if samples.generation != generation:
                self.samples = samples.with_generation(generation)
            else:
                self.samples = samples

        # Create default VariantMeta if not provided
        if variants is None:
            vid = np.arange(m, dtype=np.int64)
            self.variants = VariantMeta(vid=vid)
        else:
            if variants.m != m:
                raise ValueError(
                    f"variants.m ({variants.m}) must match genotypes.shape[1] ({m})"
                )
            self.variants = variants

    # --- implement base class properties ---

    @property
    def n(self) -> int:
        """Number of samples."""
        return self.genotypes.shape[0]

    @property
    def m(self) -> int:
        """Number of diploid variants."""
        return self.genotypes.shape[1]

    @property
    def diploid_genotypes(self) -> np.ndarray:
        """Return diploid genotype counts (0, 1, or 2) as 2D array (n, m)."""
        return self.genotypes[:, :, 0] + self.genotypes[:, :, 1]

    # --- sample metadata accessors ---

    @property
    def iid(self) -> np.ndarray:
        """Individual IDs."""
        return self.samples.iid

    @property
    def fid(self) -> np.ndarray:
        """Family IDs."""
        return self.samples.fid

    @property
    def sex(self) -> np.ndarray:
        """Biological sex (0=female, 1=male)."""
        return self.samples.sex

    @property
    def n_fam(self) -> int:
        """Number of unique families."""
        return self.samples.n_fam

    @property
    def n_female(self) -> int:
        """Number of biological females."""
        return self.samples.n_female

    @property
    def n_male(self) -> int:
        """Number of biological males."""
        return self.samples.n_male

    # --- variant metadata accessors ---

    @property
    def vid(self) -> np.ndarray:
        """Variant IDs."""
        return self.variants.vid

    @property
    def chrom(self) -> Optional[np.ndarray]:
        """Chromosome for each variant."""
        return self.variants.chrom

    @property
    def pos_bp(self) -> Optional[np.ndarray]:
        """Base pair position."""
        return self.variants.pos_bp

    @property
    def pos_cM(self) -> Optional[np.ndarray]:
        """Centimorgan position."""
        return self.variants.pos_cM

    @property
    def af(self) -> Optional[np.ndarray]:
        """Stored allele frequencies (from VariantMeta)."""
        return self.variants.af

    # --- matrix–vector operations ---

    def diploid_matvec(self, u: np.ndarray) -> np.ndarray:
        """(G[:, :, 0] + G[:, :, 1]) @ u."""
        return (self.genotypes[:, :, 0] @ u) + (self.genotypes[:, :, 1] @ u)

    def standardized_haploid_matvec(self, u: np.ndarray, haploid: int) -> np.ndarray:
        """
        Standardized matvec for one haplotype (0 or 1):
        center & scale each variant column, then multiply by u.
        """
        H = self.genotypes[:, :, haploid]
        col_mean = H.mean(axis=0)
        col_std = H.std(axis=0, ddof=1)
        col_std[col_std == 0] = 1.0  # avoid divide-by-zero
        H_std = (H - col_mean) / col_std
        return H_std @ u

    # --- subsetting ---

    def subset(
        self,
        sample_idx=None,
        variant_idx=None,
        copy: bool = True,
    ) -> "DenseHaplotypeArray":
        """
        Return a new NHaplotypeArray with a subset of samples and/or variants.

        Parameters
        ----------
        sample_idx : array-like or slice, optional
            Indices of samples to keep.
        variant_idx : array-like or slice, optional
            Indices of variants to keep.
        copy : bool, optional
            If True, copy the data. Default True.

        Returns
        -------
        NHaplotypeArray
            Subsetted haplotype array.
        """
        if sample_idx is None:
            sample_idx = slice(None)
        if variant_idx is None:
            variant_idx = slice(None)

        new_genotypes = self.genotypes[sample_idx, :, :][:, variant_idx, :]
        if copy:
            new_genotypes = new_genotypes.copy()

        new_samples = self.samples.subset(sample_idx)
        new_variants = self.variants.subset(variant_idx)

        return DenseHaplotypeArray(
            genotypes=new_genotypes,
            generation=self.generation,
            samples=new_samples,
            variants=new_variants,
        )

    def __getitem__(self, key) -> "DenseHaplotypeArray":
        """
        Support numpy/xarray-style indexing: haplotypes[sample_idx] or haplotypes[sample_idx, variant_idx].

        Parameters
        ----------
        key : int, slice, array, or tuple
            Index or indices for samples (and optionally variants).

        Returns
        -------
        NHaplotypeArray
            Subsetted haplotype array.
        """
        if isinstance(key, tuple):
            if len(key) == 1:
                sample_idx = key[0]
                variant_idx = slice(None)
            elif len(key) == 2:
                sample_idx, variant_idx = key
            else:
                raise IndexError(f"Too many indices: {len(key)}")
        else:
            sample_idx = key
            variant_idx = slice(None)

        return self.subset(sample_idx=sample_idx, variant_idx=variant_idx, copy=False)

    def drop_isel(self, sample=None, variant=None) -> "DenseHaplotypeArray":
        """
        Drop samples or variants by index (xarray-style compatibility).

        Parameters
        ----------
        sample : array-like, optional
            Indices of samples to drop.
        variant : array-like, optional
            Indices of variants to drop.

        Returns
        -------
        NHaplotypeArray
            Haplotype array with specified samples/variants removed.
        """
        sample_idx = slice(None)
        variant_idx = slice(None)

        if sample is not None:
            keep_samples = np.ones(self.n, dtype=bool)
            keep_samples[sample] = False
            sample_idx = np.where(keep_samples)[0]

        if variant is not None:
            keep_variants = np.ones(self.m, dtype=bool)
            keep_variants[variant] = False
            variant_idx = np.where(keep_variants)[0]

        return self.subset(sample_idx=sample_idx, variant_idx=variant_idx, copy=True)

    # --- Compatibility methods for xarray-style interface ---

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape as (n_samples, 2*m_variants) for compatibility with 2D expectations."""
        return (self.n, 2 * self.m)

    @property
    def data(self) -> np.ndarray:
        """Return genotypes in 2D interleaved format (n, 2m) for compatibility."""
        # Interleave the two haplotypes: [h0_v0, h1_v0, h0_v1, h1_v1, ...]
        n, m, _ = self.genotypes.shape
        interleaved = np.empty((n, 2 * m), dtype=np.int8)
        interleaved[:, 0::2] = self.genotypes[:, :, 0]
        interleaved[:, 1::2] = self.genotypes[:, :, 1]
        return interleaved

    @property
    def values(self) -> np.ndarray:
        """Alias for data property."""
        return self.data

    @property
    def attrs(self) -> dict:
        """Return attributes dict for compatibility with xarray interface."""
        return {'generation': self.generation}

    @property
    def af_empirical(self) -> np.ndarray:
        """Compute empirical allele frequencies from genotype data."""
        # Mean across samples for each haplotype, then average the two haplotypes
        af_hap0 = self.genotypes[:, :, 0].mean(axis=0)
        af_hap1 = self.genotypes[:, :, 1].mean(axis=0)
        return (af_hap0 + af_hap1) / 2

    def to_diploid_standardized(self, af: np.ndarray = None, scale: bool = False) -> np.ndarray:
        """
        Return standardized diploid genotypes.

        Parameters
        ----------
        af : np.ndarray, optional
            Allele frequencies to use for standardization. If None, uses empirical.
        scale : bool, optional
            If True, scale by sqrt(2*p*(1-p)). Default False.

        Returns
        -------
        np.ndarray
            Standardized diploid genotypes (n, m).
        """
        G = self.diploid_genotypes.astype(np.float64)
        if af is None:
            af = self.af_empirical
        # Center: subtract 2*p (expected value under HWE)
        G_centered = G - 2 * af
        if scale:
            # Scale by sqrt(2*p*(1-p))
            denom = np.sqrt(2 * af * (1 - af))
            denom[denom == 0] = 1.0  # avoid divide by zero
            G_centered = G_centered / denom
        return G_centered

    # --- Deprecated methods for backward compatibility ---

    def get_sample_indexer(self) -> "xft.index.SampleIndex":
        """
        Create a SampleIndex from this haplotype array.

        .. deprecated::
            Use the `samples` attribute directly instead.

        Returns
        -------
        xft.index.SampleIndex
            Sample indexer with data from this array.
        """
        warnings.warn(
            "get_sample_indexer() is deprecated. Use the 'samples' attribute directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        return xft.index.SampleIndex(
            iid=self.samples.iid.astype(str),
            fid=self.samples.fid.astype(str),
            sex=self.samples.sex,
            generation=self.generation,
        )

    def get_variant_indexer(self) -> "xft.index.DiploidVariantIndex":
        """
        Create a DiploidVariantIndex from this haplotype array.

        .. deprecated::
            Use the `variants` attribute directly instead.

        Returns
        -------
        xft.index.DiploidVariantIndex
            Variant indexer with data from this array.
        """
        warnings.warn(
            "get_variant_indexer() is deprecated. Use the 'variants' attribute directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        return xft.index.DiploidVariantIndex(
            vid=self.variants.vid,
            chrom=self.variants.chrom,
            pos_bp=self.variants.pos_bp,
            pos_cM=self.variants.pos_cM,
            af=self.variants.af if self.variants.af is not None else self.af_empirical,
            zero_allele=self.variants.zero_allele,
            one_allele=self.variants.one_allele,
        )

    # --- HaplotypeOperator ABC implementation ---

    def matvec_maternal(self, v: np.ndarray) -> np.ndarray:
        """Maternal haplotype matvec: hap[:,:,0] @ v."""
        return self.genotypes[:, :, 0] @ v

    def matvec_paternal(self, v: np.ndarray) -> np.ndarray:
        """Paternal haplotype matvec: hap[:,:,1] @ v."""
        return self.genotypes[:, :, 1] @ v

    def recompute_af(self) -> np.ndarray:
        """Recompute and return empirical allele frequencies."""
        return self.af_empirical

    def to_dense(self) -> "DenseHaplotypeArray":
        """Return self (already dense)."""
        return self

    def meiosis(self, assignment, recombination_map) -> "DenseHaplotypeArray":
        """
        Perform meiosis to produce offspring haplotypes.

        Delegates to the existing numba-jitted _meiosis_3d kernel in reproduce.py.

        Parameters
        ----------
        assignment : NMateAssignment
            Mate assignment with maternal/paternal indices and offspring metadata.
        recombination_map : RecombinationMap
            Recombination probabilities between loci.

        Returns
        -------
        DenseHaplotypeArray
            Offspring haplotypes with inherited VariantMeta.
        """
        from xftsim.reproduce import meiosis as _meiosis_fn

        offspring_genotypes = _meiosis_fn(
            self,
            recombination_map,
            assignment.maternal_idx,
            assignment.paternal_idx,
        )

        return DenseHaplotypeArray(
            genotypes=offspring_genotypes,
            generation=assignment.offspring_samples.generation,
            samples=assignment.offspring_samples,
            variants=self.variants,
        )

    @property
    def xft(self) -> "NHaplotypeArrayAccessor":
        """Return accessor object for compatibility with xarray .xft interface."""
        return NHaplotypeArrayAccessor(self)

    def __repr__(self) -> str:
        parts = [f"n={self.n}", f"m={self.m}", f"generation={self.generation}"]
        if self.samples.n_fam != self.n:
            parts.append(f"n_fam={self.n_fam}")
        return f"DenseHaplotypeArray({', '.join(parts)})"


def _extract_variant_meta_from_grg(grg) -> VariantMeta:
    """Extract VariantMeta from a pygrgl GRG object.

    Constructs variant IDs as "{position}:{ref}:{alt}".
    Chromosome information is not available from GRG metadata.
    """
    m = grg.num_mutations
    positions = np.empty(m, dtype=np.int64)
    ref_alleles = np.empty(m, dtype=object)
    alt_alleles = np.empty(m, dtype=object)
    vids = np.empty(m, dtype=object)

    for i in range(m):
        mut = grg.get_mutation_by_id(i)
        positions[i] = int(mut.position)
        ref_alleles[i] = str(mut.ref_allele)
        alt_alleles[i] = str(mut.allele)
        vids[i] = f"{int(mut.position)}:{mut.ref_allele}:{mut.allele}"

    return VariantMeta(
        vid=vids,
        pos_bp=positions,
        zero_allele=ref_alleles,
        one_allele=alt_alleles,
    )


class GraphHaplotypeOperator(HaplotypeOperator):
    """GRG-backed haplotype operator using pygrgl graph traversal.

    Provides O(nodes)-per-variant matvec without materializing the full
    genotype matrix.  After meiosis, offspring revert to DenseHaplotypeArray
    since GRG has no native recombination support.

    Parameters
    ----------
    grg : pygrgl.GRG
        Loaded GRG object (via ``pygrgl.load_immutable_grg``).
    generation : int
        Generation number (default 0).
    samples : SampleMeta, optional
        Sample metadata.  If None, extracted from GRG individual IDs.
    variants : VariantMeta, optional
        Variant metadata.  If None, extracted from GRG mutation data.
    """

    def __init__(self, grg, generation: int = 0,
                 samples: Optional[SampleMeta] = None,
                 variants: Optional[VariantMeta] = None):
        self._grg = grg
        self._generation = generation
        self._cached_af = None

        # --- Samples ---
        if samples is None:
            n = grg.num_individuals
            if grg.has_individual_ids:
                iid = np.array([grg.get_individual_id(i) for i in range(n)])
            else:
                iid = np.arange(n, dtype=np.int64)
            self.samples = SampleMeta(iid=iid, generation=generation)
        else:
            if samples.n != grg.num_individuals:
                raise ValueError(
                    f"samples.n ({samples.n}) != grg.num_individuals ({grg.num_individuals})"
                )
            if samples.generation != generation:
                self.samples = samples.with_generation(generation)
            else:
                self.samples = samples

        # --- Variants ---
        if variants is None:
            self.variants = _extract_variant_meta_from_grg(grg)
        else:
            if variants.m != grg.num_mutations:
                raise ValueError(
                    f"variants.m ({variants.m}) != grg.num_mutations ({grg.num_mutations})"
                )
            self.variants = variants

    # --- properties ---

    @property
    def n(self) -> int:
        return self._grg.num_individuals

    @property
    def m(self) -> int:
        return self._grg.num_mutations

    @property
    def generation(self) -> int:
        return self._generation

    @generation.setter
    def generation(self, value: int):
        self._generation = value

    # --- matrix-vector operations ---

    def matvec(self, v: np.ndarray) -> np.ndarray:
        """Diploid G @ v via GRG DOWN traversal (by_individual=True)."""
        import pygrgl
        v = np.asarray(v, dtype=np.float64)
        inp = np.atleast_2d(v.T)  # (K, m)
        out = pygrgl.matmul(self._grg, inp, pygrgl.TraversalDirection.DOWN,
                            by_individual=True)  # (K, n)
        result = out.T  # (n, K)
        if v.ndim == 1:
            return result.ravel()
        return result

    def rmatvec(self, v: np.ndarray) -> np.ndarray:
        """G.T @ v via GRG UP traversal (by_individual=True)."""
        import pygrgl
        v = np.asarray(v, dtype=np.float64)
        inp = np.atleast_2d(v.T)  # (K, n)
        out = pygrgl.matmul(self._grg, inp, pygrgl.TraversalDirection.UP,
                            by_individual=True)  # (K, m)
        result = out.T  # (m, K)
        if v.ndim == 1:
            return result.ravel()
        return result

    def matvec_maternal(self, v: np.ndarray) -> np.ndarray:
        """Maternal haplotype matvec via GRG DOWN haploid, even indices."""
        import pygrgl
        v = np.asarray(v, dtype=np.float64)
        inp = np.atleast_2d(v.T)  # (K, m)
        out = pygrgl.matmul(self._grg, inp, pygrgl.TraversalDirection.DOWN,
                            by_individual=False)  # (K, 2n)
        maternal = out[:, 0::2]  # even = maternal
        result = maternal.T  # (n, K)
        if v.ndim == 1:
            return result.ravel()
        return result

    def matvec_paternal(self, v: np.ndarray) -> np.ndarray:
        """Paternal haplotype matvec via GRG DOWN haploid, odd indices."""
        import pygrgl
        v = np.asarray(v, dtype=np.float64)
        inp = np.atleast_2d(v.T)  # (K, m)
        out = pygrgl.matmul(self._grg, inp, pygrgl.TraversalDirection.DOWN,
                            by_individual=False)  # (K, 2n)
        paternal = out[:, 1::2]  # odd = paternal
        result = paternal.T  # (n, K)
        if v.ndim == 1:
            return result.ravel()
        return result

    def standardized_matvec(self, v: np.ndarray, af: np.ndarray = None) -> np.ndarray:
        """Centered diploid matvec: G@v - 2*af@v (no materialization)."""
        v = np.asarray(v, dtype=np.float64)
        if af is None:
            af = self.recompute_af()
        raw = self.matvec(v)
        # centering: E[G_ij] = 2*af_j, so (G - 2p)@v = G@v - 2p@v
        correction = 2.0 * (af @ v)
        return raw - correction

    def recompute_af(self) -> np.ndarray:
        """Compute allele frequencies via GRG UP traversal: G.T @ 1 / (2n)."""
        import pygrgl
        if self._cached_af is not None:
            return self._cached_af
        ones = np.ones((1, self.n), dtype=np.float64)
        counts = pygrgl.matmul(self._grg, ones, pygrgl.TraversalDirection.UP,
                               by_individual=True)  # (1, m)
        self._cached_af = (counts.ravel() / (2.0 * self.n))
        return self._cached_af

    def to_dense(self) -> DenseHaplotypeArray:
        """Materialize full genotype matrix via identity matvec."""
        import pygrgl
        eye = np.eye(self.m, dtype=np.float64)  # (m, m)
        # DOWN haploid: (m, m) -> (m, 2n)
        haploid = pygrgl.matmul(self._grg, eye, pygrgl.TraversalDirection.DOWN,
                                by_individual=False)  # (m, 2n)
        # Reshape to (n, m, 2): maternal=even, paternal=odd
        maternal = haploid[:, 0::2].T  # (n, m)
        paternal = haploid[:, 1::2].T  # (n, m)
        genotypes = np.stack([maternal, paternal], axis=2).astype(np.int8)
        return DenseHaplotypeArray(
            genotypes=genotypes,
            generation=self._generation,
            samples=self.samples,
            variants=self.variants,
        )

    def meiosis(self, assignment, recombination_map) -> DenseHaplotypeArray:
        """Materialize to dense, then perform meiosis."""
        return self.to_dense().meiosis(assignment, recombination_map)

    def __getitem__(self, key) -> DenseHaplotypeArray:
        """Materialize to dense and subset."""
        return self.to_dense()[key]

    def __repr__(self) -> str:
        return (f"GraphHaplotypeOperator(n={self.n}, m={self.m}, "
                f"generation={self.generation})")


class NHaplotypeArrayAccessor:
    """
    Accessor class that mimics the xarray .xft interface for DenseHaplotypeArray.
    Provides compatibility with code expecting xarray-style access.
    """

    def __init__(self, haplotypes: "DenseHaplotypeArray"):
        self._haplotypes = haplotypes

    @property
    def n(self) -> int:
        """Number of samples."""
        return self._haplotypes.n

    @property
    def m(self) -> int:
        """Number of variants."""
        return self._haplotypes.m

    @property
    def generation(self) -> int:
        """Generation number."""
        return self._haplotypes.generation

    @property
    def samples(self) -> SampleMeta:
        """Sample metadata."""
        return self._haplotypes.samples

    @property
    def variants(self) -> VariantMeta:
        """Variant metadata."""
        return self._haplotypes.variants

    @property
    def af_empirical(self) -> np.ndarray:
        """Empirical allele frequencies."""
        return self._haplotypes.af_empirical

    def get_sample_indexer(self) -> "xft.index.SampleIndex":
        """Get sample indexer (deprecated)."""
        return self._haplotypes.get_sample_indexer()

    def get_variant_indexer(self) -> "xft.index.DiploidVariantIndex":
        """Get variant indexer (deprecated)."""
        return self._haplotypes.get_variant_indexer()

    def to_diploid(self) -> np.ndarray:
        """Return diploid genotype counts (0, 1, or 2) as 2D array (n, m)."""
        return self._haplotypes.diploid_genotypes

    def to_diploid_standardized(self, af: np.ndarray = None, scale: bool = False) -> np.ndarray:
        """Return standardized diploid genotypes."""
        return self._haplotypes.to_diploid_standardized(af=af, scale=scale)


# Backward-compatible alias
NHaplotypeArray = DenseHaplotypeArray



# ---------------------------------------------------------------------------
# NPhenotypeArray — new numpy-backed phenotype container
# ---------------------------------------------------------------------------

class NPhenotypeArray:
    """
    Thin wrapper around a flat dict of named 1-D arrays.

    Each key is a component/phenotype name (e.g. 'height.G', 'height').
    The dot is purely a human convention — not parsed.

    Parameters
    ----------
    samples : SampleMeta
        Sample metadata that travels with the data.
    values : dict, optional
        Initial name → (n,) array mapping.
    """

    def __init__(self, samples: SampleMeta, values: Optional[Dict[str, np.ndarray]] = None):
        self.samples = samples
        self._values: Dict[str, np.ndarray] = {}
        if values is not None:
            for k, v in values.items():
                self[k] = v

    def __getitem__(self, key: str) -> np.ndarray:
        return self._values[key]

    def __setitem__(self, key: str, val: np.ndarray):
        val = np.asarray(val, dtype=np.float64)
        if val.shape != (self.samples.n,):
            raise ValueError(
                f"Value for '{key}' has shape {val.shape}, expected ({self.samples.n},)"
            )
        if key in self._values:
            warnings.warn(f"Overwriting existing key '{key}' in NPhenotypeArray")
        self._values[key] = val

    def __contains__(self, key: str) -> bool:
        return key in self._values

    @property
    def keys(self):
        """Return the names of all stored components."""
        return self._values.keys()

    def subset(self, idx) -> "NPhenotypeArray":
        """Return a new NPhenotypeArray with a subset of samples."""
        new_samples = self.samples.subset(idx)
        new_values = {k: v[idx].copy() for k, v in self._values.items()}
        return NPhenotypeArray(samples=new_samples, values=new_values)

    def __repr__(self) -> str:
        return (f"NPhenotypeArray(n={self.samples.n}, "
                f"keys={list(self._values.keys())})")


# ---------------------------------------------------------------------------
# PedigreeArray
# ---------------------------------------------------------------------------

@dataclass
class PedigreeArray:
    """
    Integer index arrays linking offspring to parents.

    Produced at reproduction time; consumed by parent/mother/father references
    and by filters (TrioFilter, SibPairFilter).

    Parameters
    ----------
    offspring_samples : SampleMeta
        Metadata for the offspring generation.
    maternal_idx : np.ndarray
        (n,) indices into the *parent* generation's SampleMeta for each offspring's mother.
    paternal_idx : np.ndarray
        (n,) indices into the *parent* generation's SampleMeta for each offspring's father.
    parent_n : int
        Number of individuals in the parent generation (for bounds checking).
    """
    offspring_samples: SampleMeta
    maternal_idx: np.ndarray
    paternal_idx: np.ndarray
    parent_n: int

    def __post_init__(self):
        self.maternal_idx = np.asarray(self.maternal_idx, dtype=np.intp)
        self.paternal_idx = np.asarray(self.paternal_idx, dtype=np.intp)
        n = self.offspring_samples.n
        if len(self.maternal_idx) != n:
            raise ValueError(
                f"maternal_idx length {len(self.maternal_idx)} != offspring n {n}"
            )
        if len(self.paternal_idx) != n:
            raise ValueError(
                f"paternal_idx length {len(self.paternal_idx)} != offspring n {n}"
            )
        if n > 0:
            if np.any(self.maternal_idx < 0) or np.any(self.maternal_idx >= self.parent_n):
                raise ValueError(
                    f"maternal_idx out of bounds [0, {self.parent_n})"
                )
            if np.any(self.paternal_idx < 0) or np.any(self.paternal_idx >= self.parent_n):
                raise ValueError(
                    f"paternal_idx out of bounds [0, {self.parent_n})"
                )