import warnings
import numpy as np
import pandas as pd
import math
import functools

from typing import Iterable, Union, Dict
from nptyping import NDArray, Shape, Float, Int, Object
from nptyping import NDArray, Int8, Int64, Float64, Bool, Shape, Float, Int
from typing import Any, Hashable, List, Iterable, Callable, Union, Dict, Tuple
from collections.abc import Sequence


import xftsim as xft


import traceback
import warnings
import sys

# def warn_with_traceback(message, category, filename, lineno, file=None, line=None):

#     log = file if hasattr(file,'write') else sys.stderr
#     traceback.print_stack(file=log)
#     log.write(warnings.formatwarning(message, category, filename, lineno, line))

# warnings.showwarning = warn_with_traceback

from xftsim.utils import ensure2D, cartesian_product, ids_from_n_generation, paste, merge_duplicates
## superclass not to be used
class XftIndex:
    """
    XftIndex is a class representing an index for the XftSim simulation model. Super class not for direct
    use by the user.

    Attributes:
    -----------
    _coord_variables: List[str]
        List of names of the coordinate variables.
    _index_variables: List[str]
        List of names of the index variables.
    _dimension: str
        Name of the dimension variable.
    _frame: pandas.DataFrame
        Dataframe representing the index.

    Methods:
    --------
    validate():
        Validates the index by checking if the `_coord_variables`, `_index_variables`, and `_dimension` attributes are not None.
        Raises an AssertionError if any of these attributes is None.

    frame:
        Property representing the `_frame` attribute.
        Getter: Returns the `_frame` attribute.
        Setter: Sets the `_frame` attribute and generates a new index using the `unique_identifier` property.
    
    frame_copy():
        Returns a copy of the `_frame` attribute.

    unique_identifier:
        Property representing the unique identifier of the index.
        Returns a string representing the concatenation of all index variables, separated by a period.
    
    coords:
        Property representing the coordinates of the index.
        Returns a dictionary where the keys are the coordinate variables and the values are the corresponding values in the `_frame` attribute.
    
    coord_dict:
        Property representing the coordinate dictionary of the index.
        Returns a dictionary where the keys are the variables and the values are tuples representing the `(dimension, value)` of each coordinate.
    
    coord_frame:
        Property representing the coordinate frame of the index.
        Returns a dataframe where the columns are the coordinate variables and the rows correspond to each row in the `_frame` attribute.
    
    coord_mindex:
        Property representing the coordinate multi-index of the index.
        Returns a multi-index where the levels correspond to the coordinate variables and the values correspond to the corresponding values in the `_frame` attribute.
    
    coord_index:
        Property representing the coordinate index of the index.
        Returns an index representing the unique identifier of the index.
    
    __getitem__(arg):
        Returns a new instance of the XftIndex class, corresponding to a subset of the `_frame` attribute.
        If `arg` is a dictionary, returns the rows where the values of the keys in the dictionary match the corresponding values in the `_frame` attribute.
        If `arg` is an integer or slice, returns the row(s) at the corresponding index in the `_frame` attribute.

    iloc(key):
        Returns a new instance of the XftIndex class, corresponding to a subset of the `_frame` attribute.
        Returns the row(s) at the corresponding index in the `_frame` attribute.

    merge(other):
        Merges the `_frame` attribute of the current instance with another instance of the XftIndex class.
        If the two instances have a different `_dimension` attribute or a different class type, raises a TypeError.
        Returns a new instance of the XftIndex class representing the merged index.

    reduce_merge(args):
        Static method that reduces the list of `args` by calling the `merge` method on each pair of consecutive elements.
        Returns the final merged index.

    stack(other):
        Stacks the `_frame` attribute of the current instance with another instance of the XftIndex class.
        If the two instances have a different `_dimension` attribute or a different class type, raises a TypeError.
        Returns a new instance of the XftIndex class representing the stacked index.

    at_most(n_new):
        Downsamples the `_frame` attribute at random to contain at most `n_new` rows.
        If the number of rows in the `_frame` attribute is already less than or equal to `n_new`, returns a copy of the current instance.
        Returns a new instance of the XftIndex class representing the downsampled index.
    """
    def validate(self):
        assert self._coord_variables is not None
        assert self._index_variables is not None
        assert self._dimension is not None

    @property
    def frame(self):
        # self._frame[self._dimension] = self.unique_identifier
        # self._frame.set_index(self._dimension, inplace=True)
        return(self._frame)
    @frame.setter
    def frame(self, dataframe):
        # dataframe[self._dimension] = xft.utils.unique_identifier(dataframe, self._index_variables)
        # dataframe.set_index(self._dimension, inplace=True)
        self._frame = dataframe
        new_index = pd.Series(data=self.unique_identifier,
                              name=self._dimension)
        self._frame = dataframe.set_index(new_index)
        # self._frame[self._dimension] = self.unique_identifier
        # self._frame.set_index(self._dimension, inplace=True)
        # return(self._frame)        

    def frame_copy(self):
        return(self.frame.copy())

    @property
    def unique_identifier(self):
        return xft.utils.unique_identifier(self.frame, self._index_variables)
        # return paste([self._frame[variable].values for variable in self._index_variables], sep=".")
    
    @property
    def coords(self):
        coords = {self._dimension:np.asarray(self.unique_identifier)}
        coords.update({variable:np.asarray(self.frame[variable].values) for variable in self._coord_variables})
        return coords

    @property
    def coord_dict(self):
        return {key:(self._dimension, value) for (key,value) in self.coords.items()}

    @property
    def coord_frame(self):
        frame = pd.DataFrame.from_dict(self.coords)
        frame.set_index(self._dimension, drop=False, inplace=True)
        return frame

    @property
    def coord_mindex(self):
        cm = pd.MultiIndex.from_frame(self.coord_frame)
        return cm

    @property
    def coord_index(self):
        return pd.Index(self.unique_identifier)

    def __getitem__(self, arg): ## TODO: optimize to avoid reconstruction
        frame = self.frame
        ## subsetting for dict
        if isinstance(arg, dict):
            subset = np.full(frame.shape[0], fill_value=True, dtype=bool)
            for (key,value) in arg.items():
                if not key in frame.columns:
                    raise KeyError
                elif pd.api.types.is_list_like(value):            
                    subset *= frame[key].isin(value).values
                else:                 
                    subset *= (frame[key].values == value)
            return self.__class__(frame=frame.loc[subset,:])
        else:
            return self.__class__(frame=frame.loc[arg])

    def iloc(self, key): ## TODO: optimize to avoid reconstruction
        return self.__class__(frame=self.frame.iloc[key])

    def merge(self, other, deduplicate=True):
        if self.__class__ != other.__class__: raise TypeError()
        if self._dimension != other._dimension: raise TypeError()
        frame = pd.concat([self.frame, other.frame])
        if deduplicate:
            frame = frame[~frame.duplicated()]
        if hasattr(self, 'generation'):
            return self.__class__(frame=frame, generation=self.generation)
        else: 
            return self.__class__(frame=frame)

    @staticmethod
    def reduce_merge(args,deduplicate=True):
        return functools.reduce(lambda a,b: XftIndex.merge(a,b,deduplicate=deduplicate), args)

    def stack(self, other):
        if self.__class__ != other.__class__: raise TypeError()
        if self._dimension != other._dimension: raise TypeError()
        frame = pd.concat([self.frame, other.frame])
        if hasattr(self, 'generation'):
            return self.__class__(frame=frame, generation=self.generation)
        else: 
            return self.__class__(frame=frame)

    ## downsample at random
    def at_most(self, n_new):
        if self.n > n_new:
            # frame = self.frame.iloc[np.sort(np.random.choice(self.n, n_new))]
            frame = self.frame.iloc[:n_new]
        else: 
            frame = self.frame
        if hasattr(self, 'generation'):
            return self.__class__(frame=frame, generation=self.generation)
        else: 
            return self.__class__(frame=frame)

    def __len__(self):
        return self.frame.shape[0]


class SampleIndex(XftIndex):
    """Index for individual samples.

    This class is used to keep track of information for individual samples.

    Parameters
    ----------
    iid : Iterable, optional
        Iterable of individual IDs.
    fid : Iterable, optional
        Iterable of family IDs.
    sex : Iterable, optional
        Iterable of biological sexes.
    frame : pd.DataFrame, optional
        Dataframe containing information for each sample.
    n : int, optional
        Number of samples to generate a random ID set for.
    generation : int, optional
        Generation number for samples.

    Attributes
    ----------
    n : int
        Number of individuals.
    n_fam : int
        Number of families.
    n_female : int
        Number of biological females.
    n_male : int
        Number of biological males.
    iid : ndarray
        Array of individual IDs.
    fid : ndarray
        Array of family IDs.
    sex : ndarray
        Array of biological sexes.
    """
    def __init__(self,
                 iid: Iterable = None, ## individual id
                 fid: Iterable = None, ## family id
                 sex: Iterable = None, ## biological sex
                 frame: pd.DataFrame = None,
                 n: int = None,
                 generation: int = 0,
                 ):
        self.generation = generation
        self._dimension = "sample"
        self._index_variables = np.array(['iid', 'fid'])
        self._coord_variables = np.array(['iid', 'fid', 'sex'])

        ## either provide iid XOR n XOR frame
        assert (iid is not None) ^ (frame is not None) ^ (n is not None)

        ## if iid provided, populate fid/sex if needed, generate frame
        if iid is not None:
            iid = np.array(iid)
            n = iid.shape[0]
            if fid is None:
                fid = iid
            if sex is None:
                sex = np.tile([0,1], math.ceil(n/2))[:n] ## ensure proper length
            self.frame = pd.DataFrame.from_dict({'iid':iid, 'fid':fid, 'sex':sex})
        ## if n provided, populate iid/fid/sex, generate frame
        elif n is not None:
            iid = ids_from_n_generation(n, generation)
            fid = iid
            sex = np.tile([0,1], math.ceil(n/2))[:n]
            self.frame = pd.DataFrame.from_dict({'iid':iid, 'fid':fid, 'sex':sex})
        ## otherwise just use provided frame
        elif frame is not None:
            self.frame = frame


    @property
    def unique_identifier(self):
        return xft.utils.unique_identifier(self.frame, self._index_variables, prefix=self.generation)

    ## access data frame columns as properitie[s
    @property
    def iid(self):
        return self.frame['iid'].values
    # @iid.setter
    # def iid(self, value):
        # self._frame['iid']=value
    @property
    def fid(self):
        return self.frame['fid'].values
    # @fid.setter
    # def fid(self, value):
        # self._frame['fid']=value
    @property
    def sex(self):
        return self.frame['sex'].values
    # @sex.setter
    # def sex(self, value):
        # self._frame['sex']=value
        
    @property
    def n(self):
        return self.iid.shape[0]
    
    @property
    def n_fam(self):
        return np.unique(self.fid).shape[0]
    
    @property
    def n_female(self):
        return np.sum(self.sex==0).astype(int)
    
    @property
    def n_male(self):
        return np.sum(self.sex==1).astype(int)

    def __repr__(self, short=False):
        cf = self.frame
        mi_repr = cf.__repr__().split("\n")
        if len(mi_repr) > 10:
            mi_repr = mi_repr[:3] + ["            ..."] + mi_repr[-3:]
        output = [
                  "<SampleIndex>",
                  f"  Generation {self.generation}",
                  f"  {self.n} indviduals from {self.n_fam} families",
                  f"  {self.n_female} biological females",
                  f"  {self.n_male} biological males",
                  ] + mi_repr
        if short:
            return output[1:4]
        else:
            return "\n".join(output)

    def __eq__(self, other):
        return np.all(self.frame == other.frame)

    ## redefine to include generation
    def __getitem__(self, arg): ## TODO: optimize to avoid reconstruction
        frame = self.frame
        ## subsetting for dict
        if isinstance(arg, dict):
            subset = np.full(frame.shape[0], fill_value=True, dtype=bool)
            for (key,value) in arg.items():
                if not key in frame.columns:
                    raise KeyError
                elif pd.api.types.is_list_like(value):            
                    subset *= frame[key].isin(value).values
                else:                 
                    subset *= (frame[key].values == value)
            return self.__class__(frame=frame.loc[subset,:], generation=self.generation)
        else:
            return self.__class__(frame=frame.loc[arg], generation=self.generation)


    ## redefine to include generation
    def iloc(self, key): ## TODO: optimize to avoid reconstruction
        return self.__class__(frame=self.frame.iloc[key], generation=self.generation)


class DiploidVariantIndex(XftIndex):
    """
    This class is used to index diploid genetic variants. Variants are defined by a set of unique IDs and may have
    additional annotations. Each variant is associated with two alleles, represented as strings.

    Parameters
    ----------
    vid: NDArray[Shape["*"], Object], optional
        Variant IDs, by default None.
    chrom: NDArray[Shape["*"], Int], optional
        Chromosome of variant, by default None.
    zero_allele: NDArray[Shape["*"], Object], optional
        First allele of variant, by default None.
    one_allele: NDArray[Shape["*"], Object], optional
        Second allele of variant, by default None.
    af: Iterable, optional
        Allele frequency of variant, by default None.
    annotation_array: Union[NDArray, pd.DataFrame], optional
        Additional variant annotations, by default None.
    annotation_names: Iterable, optional
        Names of the additional variant annotations, by default None.
    frame: pd.DataFrame, optional
        A pandas DataFrame containing variant information, by default None.
    m: int, optional
        The number of variants, by default None.
    n_chrom: int, optional
        The number of chromosomes, by default 1.
    h_copy: NDArray[Shape["*"], Object], optional
        A string indicating the haplotype of each variant, by default None.
    pos_bp: Iterable, optional
        Base-pair positions of the variant, by default None.
    pos_cM: Iterable, optional
        Centimorgan positions of the variant, by default None.

    Attributes
    ----------
    vid: ndarray
        Variant IDs.
    chrom: ndarray
        Chromosome of variant.
    zero_allele: ndarray
        First allele of variant.
    one_allele: ndarray
        Second allele of variant.
    hcopy: ndarray
        A string indicating the copy of each variant.
    af: ndarray
        Allele frequency of variant.
    pos_bp: ndarray
        Base-pair positions of the variant.
    pos_cM: ndarray
        Centimorgan positions of the variant.
    ploidy: str
        A string indicating the ploidy of the variant (always "Diploid" for this class).
    annotation: pd.DataFrame
        A pandas DataFrame containing additional variant annotations.
    annotation_array: Union[ndarray, None]
        A numpy array containing additional variant annotations.
    annotation_names: ndarray
        An array containing names of additional variant annotations.
    m: int
        The number of variants.
    n_chrom: int
        The number of chromosomes.
    n_annotations: int
        The number of additional variant annotations.
    maf: ndarray
        Minor allele frequency of variant.
        
    Raises
    ------
    AssertionError
        If vid, m, or frame is not provided.
        If both zero_allele and one_allele are not provided.
    """
    def __init__(self,
                 vid: NDArray[Shape["*"], Object] = None, ## variant id
                 chrom: NDArray[Shape["*"], Int] = None, ## chromosome
                 zero_allele: NDArray[Shape["*"], Object] = None, ## allele coding
                 one_allele: NDArray[Shape["*"], Object] = None, 
                 af: Iterable = None,
                 annotation_array: Union[NDArray, pd.DataFrame] = None,
                 annotation_names: Iterable = None,
                 frame: pd.DataFrame = None,
                 m: int = None,
                 n_chrom: int = 1,
                 h_copy: NDArray[Shape["*"], Object] = None,
                 pos_bp: Iterable = None,
                 pos_cM: Iterable = None,                 
                 ):
        self._dimension = "variant"
        self._index_variables = ["vid", "hcopy"]
        self._coord_variables = ["vid", "chrom","hcopy","zero_allele",
                                 "one_allele", "af", "pos_bp", "pos_cM"]

        ## provide vid XOR m XOR frame
        assert (vid is not None) ^ (m is not None) ^ (frame is not None)
        if frame is None:
            ## set generic vids if not provided
            if m is not None:
                vid = np.arange(m).astype(str)
            else: 
                vid = np.array(vid)
                m = vid.shape[0]
                if self.ploidy =="Diploid":
                    assert np.unique(vid).shape[0] == vid.shape[0], "Duplicate variant IDs"
            ## set generic chromosomes if missing
            if chrom is None:
                chrom = np.repeat(np.arange(n_chrom), 
                                  math.ceil(m/n_chrom))[:m]
            ## set generic zero/one alleles if either is missing
            if zero_allele is None or one_allele is None:
                zero_allele = np.repeat(['A'], m)
                one_allele = np.repeat(['G'], m)
            ## set afs to NaN if missing
            if af is not None:
                # assert len(af) == m, "number of variants and afs do not match"
                pass
            else:
                af = np.full(m, fill_value=np.NaN)
            ## populate positions if needed
            if pos_bp is not None:
                assert len(pos_bp) == m, "number of variants and pos_bp do not match"
            else:
                pos_bp = np.full(m, fill_value=np.NaN)
            if pos_cM is not None:
                assert len(pos_cM) == m, "number of variants and pos_cM do not match"
            else:
                pos_cM = np.full(m, fill_value=np.NaN)
            ## populate hcopy if needed
            if h_copy is None:
                h_copy = np.repeat("d",m)
            self.frame = pd.DataFrame.from_dict({
                       "vid":vid, 
                       "chrom":chrom,
                       "zero_allele":zero_allele, 
                       "one_allele":one_allele,
                       "af":af,
                       "hcopy":h_copy,
                       "pos_bp":pos_bp,
                       "pos_cM":pos_cM,
                       })
        ## otherwise use provided data frame
        else:
            self.frame = frame
            ## TODO verify df is valid
            ## populate possibly missing elements
            if not "hcopy" in frame.columns.values:
                self.frame["hcopy"]=np.repeat("d",m)
            if not "pos_bp" in frame.columns.values:
                self.frame["pos_bp"]=np.repeat(np.NaN,m)
            if not "pos_cM" in frame.columns.values:
                self.frame["pos_cM"]=np.repeat(np.NaN,m)
        self._annotation_array = annotation_array
        self._annotation_names = annotation_names
        if annotation_array is not None:
            if annotation_names is None:
                self._annotation_names = np.char.add('a_',np.arange(annotation_array.shape[1]).astype(str))
            self._annotation = pd.DataFrame.from_dict({key:value for (key,value) in zip(self._annotation_names, annotation_array.T)})
            self.frame = pd.concat([self.frame, self._annotation],1)
            self._coord_variables = np.concatenate(self._coord_variables,
                                                   self._annotation_names)
        else:
            self._annotation = None


    @property
    def vid(self):
        return self.frame['vid'].values
    # @vid.setter
    # def vid(self, value):
        # self._frame['vid'] = value
    @property
    def chrom(self):
        return self.frame['chrom'].values
    # @chrom.setter
    # def chrom(self, value):
        # self._frame['chrom'] = value
    @property
    def one_allele(self):
        return self.frame['one_allele'].values
    # @one_allele.setter
    # def one_allele(self, value):
        # self._frame['one_allele'] = value
    @property
    def zero_allele(self):
        return self.frame['zero_allele'].values
    # @zero_allele.setter
    # def zero_allele(self, value):
        # self._frame['zero_allele'] = value
    @property
    def hcopy(self):
        return self.frame['hcopy'].values
    # @hcopy.setter
    # def hcopy(self, value):
        # self._frame['hcopy'] = value
    @property
    def af(self):
        return self.frame['af'].values
    @af.setter
    def af(self, value):
        self._frame['af'] = value
    
    @property
    def pos_bp(self):
        return self.frame['pos_bp'].values
    
    @property
    def pos_cM(self):
        return self.frame['pos_cM'].values
    
    @property
    def ploidy(self):
        return "Diploid"

    @property
    def annotation(self):
        return self._annotation
    
    @property
    def annotation_array(self):
        return self._annotation_array

    @property
    def annotation_names(self):
        return self._annotation_names

    @property
    def m(self):
        return np.unique(self.vid).shape[0]

    @property
    def n_chrom(self):
        return np.unique(self.chrom).shape[0]

    @property
    def n_annotations(self):
        if self.annotation is not None:
            return self.annotation.shape[1]
        else:
            return 0

    @property
    def maf(self):
        if not np.any(self.af==np.NaN):
            return np.full(self.m, fill_value=np.NaN)
        else:
            tmp = 1 - self.af
            return np.where(tmp <= self.af, tmp, self.af)
    

    def __repr__(self):
        cf = self.frame
        mi_repr = cf.__repr__().split("\n")
        if np.any(self.af == np.NaN) is None:
            af_str = "  allele frequencies unknown"
        else:
            af_str = f"  MAF ranges from {min(self.maf)} to {max(self.maf)}"
        output = [
               "<"+self.ploidy+"VariantIndex>",
               f"  {self.m} diploid variants on {self.n_chrom} chromosome(s)",
               af_str,
               f"  {self.n_annotations} annotation(s) ",
               ] + mi_repr
        return "\n".join(output)

    def __eq__(self, other):
        return np.all(self.coord_frame == other.coord_frame)

    def to_haploid(self):
        hframe = pd.DataFrame.from_dict({
           "vid":np.repeat(self.vid, 2), 
           "chrom":np.repeat(self.chrom, 2),
           "zero_allele":np.repeat(self.zero_allele, 2), 
           "one_allele":np.repeat(self.one_allele, 2),
           "af":np.repeat(self.af, 2),
           "hcopy":np.tile(['0','1'],self.vid.shape[0]),
           "pos_bp":np.repeat(self.pos_bp, 2),
           "pos_cM":np.repeat(self.pos_cM, 2),
           })
        if self._annotation_array is not None:
            self._annotation_array = np.repeat(self._annotation_array,2,axis=1)
            self._annotation_names = np.repeat(self._annotation_array,2)

        return HaploidVariantIndex(frame=hframe,
                            annotation_array=self.annotation_array,
                            annotation_names=self.annotation_names)

    def annotate(self):
        raise NotImplementedError ## TODO

class HaploidVariantIndex(DiploidVariantIndex):
    """
    A class representing a haploid variant index.


    Attributes
    ----------
    vid : numpy.ndarray
        Variant IDs.
    chrom : numpy.ndarray
        Chromosome numbers.
    zero_allele : numpy.ndarray
        Alleles with value zero.
    one_allele : numpy.ndarray
        Alleles with value one.
    af : numpy.ndarray
        Allele frequencies.
    pos_bp : numpy.ndarray
        Positions of variants in base pairs.
    pos_cM : numpy.ndarray
        Positions of variants in centiMorgans.
    m : int
        Number of unique variant IDs.
    n_chrom : int
        Number of unique chromosome numbers.
    n_annotations : int
        Number of annotations.
    maf : numpy.ndarray
        Minor allele frequencies.
    ploidy : str
        The ploidy of the variant index. In this case, "Haploid".
    hcopy: ndarray
        A string indicating the copy of each variant.

    Methods
    -------
    to_diploid()
        Converts the haploid variant index to diploid.
    """
    @property
    def ploidy(self):
        return "Haploid"

    def to_diploid(self):
        dframe = pd.DataFrame.from_dict({
           "vid":self.vid[::2],
           "chrom":self.chrom[::2],
           "zero_allele":self.zero_allele[::2],
           "one_allele":self.one_allele[::2],
           "af":self.af[::2],
           "hcopy":np.repeat(['d'],self.vid.shape[0]//2),
           "pos_bp":self.pos_bp[::2],
           "pos_cM":self.pos_cM[::2],
           })
        if self._annotation_array is not None:
            self._annotation_array = self._annotation_array[:,::2]
            self._annotation_names = self._annotation_array[::2]

        return DiploidVariantIndex(frame=dframe,
                            annotation_array=self.annotation_array,
                            annotation_names=self.annotation_names)


class ComponentIndex(XftIndex):
    """
    Index object for phenotype components, including origin relative to proband.

    Parameters
    ----------
    phenotype_name : iterable, optional
        Names of phenotypes. Either `phenotype_name`, `frame`, or `k_total` must be provided.
    component_name : iterable, optional
        Names of phenotype components.
    vorigin_relative : iterable, optional
        Relative origin of phenotype component. -1 for proband, 0 for mother, 1 for father.
    comp_type: iterable, optional
        Elements are either 'intermediate' or 'outcome' to distinguish between phenotype components
        versus phenotypes themselves
    frame : pandas.DataFrame, optional
        Pre-existing frame to initialize index.
    k_total : int, optional
        Total number of phenotypes to generate generic names.

    Attributes
    ----------
    phenotype_name : numpy.ndarray
        Names of phenotypes.
    component_name : numpy.ndarray
        Names of phenotype components.
    vorigin_relative : numpy.ndarray
        Relative origin of phenotype component. -1 for proband, 0 for mother, 1 for father.
    k_total : int
        Total number of phenotypes.
    k_phenotypes : int
        Number of unique phenotypes.
    k_components : int
        Number of unique phenotype components.
    k_relative : int
        Number of unique relative origins.
    depth : float
        Generational depth from binary relative encoding.
    unique_identifier : numpy.ndarray
        Unique identifier for the index.

    Methods
    -------
    to_vorigin(origin)
        Returns a new ComponentIndex with all vorigin_relative set to `origin`.
    to_proband()
        Returns a new ComponentIndex with all vorigin_relative set to -1 (proband).
    from_frame(df)
        Returns a new ComponentIndex initialized from a Pandas DataFrame.
    from_arrays(phenotype_name, component_name, vorigin_relative=None)
        Returns a new ComponentIndex initialized from numpy arrays.
    from_product(phenotype_name, component_name, vorigin_relative=None)
        Returns a new ComponentIndex initialized from a Cartesian product of phenotype_name, component_name, and vorigin_relative.
    range_index(c, component_name=['generic'], vorigin_relative=[-1], prefix='phenotype')
        Returns a new ComponentIndex with generic phenotype names.

    """
    def __init__(self,
                 phenotype_name: Iterable = None, # names of phenotypes
                 component_name: Iterable = None, # names of phenotype components
                 vorigin_relative: Iterable = None, # relative of phenotype origin
                 comp_type: Iterable = None, # elements are either 'intermediate' or 'outcome'
                 comp_type_map: Dict = dict(phenotype='outcome'),
                 frame: pd.DataFrame = None,
                 k_total: int = None,
                 ):
        self._dimension = "component"
        self._index_variables = np.array(['phenotype_name', 'component_name', 'vorigin_relative'])
        self._coord_variables = np.array(['phenotype_name', 'component_name', 'vorigin_relative', 'comp_type'])

        ## provide phenotype_name XOR frame XOR k_total
        assert (phenotype_name is not None) ^ (k_total is not None)^ (frame is not None), "provide phenotype_name OR k_total OR frame"

        ## set generic phenotype names if not provided
        if frame is None:
            if phenotype_name is None:
                assert component_name is None and vorigin_relative is None, "provide phenotype_name with component_name / vorigin_relative"
                phenotype_name = np.arange(k_total).astype(str)
            else:
                phenotype_name = np.array(phenotype_name).astype(str)
                k_total = phenotype_name.shape[0]
            if component_name is None:
                component_name = np.repeat("generic", k_total)
            else:
                component_name = np.array(component_name).astype(str)
            if vorigin_relative is None:
                vorigin_relative = np.repeat(-1, k_total).astype(int)
            else:
                vorigin_relative = np.array(vorigin_relative)
            if comp_type is None:
                comp_type = np.repeat('intermediate', k_total)
                for (key,value) in comp_type_map.items():
                    comp_type[component_name==key] = value
            else:
                comp_type = np.array(comp_type)
            ## check for duplicates:
            # phenotype_name,component_name,vorigin_relative = merge_duplicates([phenotype_name,
            #                                                                    component_name,
            #                                                                    vorigin_relative])
            self.frame = pd.DataFrame.from_dict({'phenotype_name':phenotype_name,
                                                 'component_name':component_name,
                                                 'vorigin_relative':vorigin_relative.astype(int),
                                                 'comp_type':comp_type})
        elif frame is not None:
            self.frame = frame

    ## function to return component index with vorigin_relative = -1
    ## only intended for use when all vorigin_relative == x != -1
    def to_vorigin(self, origin):
        frame = self.frame_copy()
        frame.vorigin_relative = origin
        return ComponentIndex.from_frame(frame)

    def to_haploid(self):
        """No-op for ComponentIndex (phenotype components have no ploidy concept)."""
        return self

    def to_proband(self):
        if len(np.unique(self.vorigin_relative)) > 1 or np.any(self.vorigin_relative==-1):
            raise RuntimeError
        else:
            return self.to_vorigin(-1)
            # frame = self.frame_copy()
            # frame.vorigin_relative = -1
            # return ComponentIndex.from_frame(frame)
            
    @property
    def phenotype_name(self):
        return self.frame['phenotype_name']
    @phenotype_name.setter
    def phenotype_name(self, value):
        self._frame['phenotype_name'] = value

    @property
    def unique_identifier(self):
        fr = self.frame.loc[:,['phenotype_name','component_name']]
        # print(fr)
        fr['vorigin_relative'] = self.frame.vorigin_relative.values.astype(int).astype(str)
        # fr.vorigin_relative.values=fr['vorigin_relative'].replace([-1,0,1],['proband','mother','father'])
        # fr.vorigin_relative.values[fr.vorigin_relative=='-1'] = 'proband'
        fr['vorigin_relative'].replace(['-1','0','1'],['proband','mother','father'], inplace=True)
        return xft.utils.unique_identifier(fr, ['phenotype_name','component_name','vorigin_relative'])

    @property
    def comp_type(self):
        return self.frame['comp_type']
    @comp_type.setter
    def comp_type(self, value):
        self._frame['comp_type'] = value

    @property
    def component_name(self):
        return self.frame['component_name']
    @component_name.setter
    def component_name(self, value):
        self._frame['component_name'] = value

    @property
    def vorigin_relative(self):
        return self.frame['vorigin_relative']
    @vorigin_relative.setter
    def vorigin_relative(self, value):
        self._frame['vorigin_relative'] = value

    @property
    def k_total(self):
        return self.phenotype_name.shape[0]

    @property
    def k_phenotypes(self):
        return np.unique(self.phenotype_name).shape[0]

    @property
    def k_components(self):
        return np.unique(self.component_name).shape[0]

    @property
    def k_relative(self):
        return np.unique(self.vorigin_relative).shape[0]


    @property
    def depth(self): ## generational depth from binary relative encoding
        if len(self.vorigin_relative) > 0:
            return math.floor(math.log2(np.max(self.vorigin_relative)+2)) + 1
        else:
            return np.NaN 

    def __repr__(self):
        if self.frame.shape[0] == 0:
            return "<Empty ComponentIndex>"
        else:
            mi_repr = self.frame.__repr__().split("\n")
            if len(mi_repr) > 50:
                mi_repr = mi_repr[:4] + ["            ..."] + mi_repr[-2:]
            if self.depth == 1:
                depth_str = f"{self.depth} generation"
            else:
                depth_str = f"{self.depth} generations"
            if self.k_components == 1:
                kcomp_str = f"{self.k_components} component"
            else:
                kcomp_str = f"{self.k_components} components"
            if self.k_phenotypes == 1:
                kphen_str = f"{self.k_phenotypes} phenotype"
            else:
                kphen_str = f"{self.k_phenotypes} phenotypes"      
            return "\n".join([
                             "<ComponentIndex>",
                             f"  {kcomp_str} of {kphen_str} spanning {depth_str}",
                             ] + mi_repr)

    def __eq__(self, other):
        return np.all(self.coord_frame == other.coord_frame)

    @staticmethod
    def from_frame(df: pd.DataFrame):
        phenotype_name = df.phenotype_name.values
        component_name = df.component_name.values
        vorigin_relative = df.vorigin_relative.values
        if 'comp_type' in df.columns:
            comp_type = df.comp_type.values
        else:
            comp_type = None
        return ComponentIndex(phenotype_name, component_name, vorigin_relative, comp_type)

    @staticmethod
    def from_arrays(phenotype_name: Iterable,
                                   component_name: Iterable,
                                   vorigin_relative: Iterable = None,
                                   comp_type: Iterable = None,
                                   ):
        if vorigin_relative is None:
            vorigin_relative = [-1]
        if comp_type is None:
            return ComponentIndex(phenotype_name, component_name, vorigin_relative)
        else:
            return ComponentIndex(phenotype_name, component_name, vorigin_relative, comp_type)


    @staticmethod
    def from_product(phenotype_name: Iterable,
                     component_name: Iterable,
                     vorigin_relative: Iterable = None,
                     comp_type_map: Dict = dict(phenotype='outcome')
                                   ):
        if vorigin_relative is None:
            vorigin_relative = [-1]
        phenotype_name, component_name, vorigin_relative = cartesian_product(phenotype_name, 
                                                                               component_name, 
                                                                               vorigin_relative)
        comp_type = np.repeat('intermediate', len(phenotype_name))
        if len(component_name)>0:
            for (key,value) in comp_type_map.items():
                comp_type[component_name==key] = value
        return ComponentIndex(phenotype_name, component_name, vorigin_relative, comp_type=comp_type)

    @staticmethod
    def range_index(c: int,
                    component_name: Iterable = ['generic'],
                    vorigin_relative: Iterable = [-1],
                    prefix: str = 'phenotype',):
        phenotype_name = np.char.add(prefix, np.arange(c).astype(str))
        return ComponentIndex.from_product(phenotype_name, component_name, vorigin_relative)


    @property
    def _pretty_vorigin_relative(self):
        output = np.repeat('proband', self.vorigin_relative.shape[0]).astype('<U8')
        output[self.vorigin_relative==0] = 'maternal' 
        output[self.vorigin_relative==1] = 'paternal' 
        return output
    @property
    def _nodes(self):
        return xft.utils.paste([self._pretty_vorigin_relative,
                               self.phenotype_name.values,
                               self.component_name.values], sep ='\n')



## TODO
def sampleIndex_from_plink():
    raise NotImplementedError

## TODO
def sampleIndex_from_VCF():
    raise NotImplementedError

## TODO
def variantIndex_from_plink():
    raise NotImplementedError

## TODO
def variantIndex_from_VCF():
    raise NotImplementedError

## tests
def _test_SampleIndex():
    generation = 0
    n=10
    iid = np.char.add(str(generation)+"_",np.arange(n).astype(str))
    fid = iid
    sex = np.tile(np.arange(2),math.ceil(n/2))[:n]
    ## should work
    default = SampleIndex(n=n)
    assert default == SampleIndex(n=n, generation = generation)
    assert default == SampleIndex(iid=iid)
    assert default == SampleIndex(iid=iid, generation = generation)
    assert default == SampleIndex(iid=iid, fid=fid)
    assert default == SampleIndex(iid=iid, fid=fid, generation = generation)
    assert default == SampleIndex(iid=iid, fid=fid, sex=sex)
    assert default == SampleIndex(iid=iid, fid=fid, sex=sex, generation = generation)
    return 0 ## TODO error cases

def _test_VariantIndex():
    pass

def _test_DiploidVariantIndex():
    pass

def _test_HaploidVariantIndex():
    pass

def _test_ComponentIndex():
    pass

def _test_ComponentIndex_from_arrays():
    pass

def _test_ComponentIndex_from_product():
    pass

def _test_sampleIndex_from_plink():
    pass

def _test_sampleIndex_from_VCF():
    pass

def _test_variantIndex_from_plink():
    pass

def _test_variantIndex_from_VCF():
    pass

class SampleFilter:
    def __init__(self, 
                 filter_function: Callable,
                 filter_name: str = None,
                 metadata: Dict = {}):
        """Class of functions that subsample fullSample.SampleIndex objects
        
        Parameters
        ----------
        filter_function : Callable
            Maps an SampleIndex and other metadata to a set of integer indices.
        filter_name : str, optional
            Name of filter applied
        metadata : Dict, optional
            Dictionary containing any additional metadata
        """
        self._filter_function = filter_function
        self.filter_name = filter_name
        self.metadata = metadata

    def filter(self, sindex: SampleIndex, **kwargs):
        if self._filter_function is not None:
            return self._filter_function(sindex, **kwargs)
        else:
            return np.arange(sindex.n)

class NullFilter(SampleFilter):
    def __init__(self):
        super().__init__(filter_function = None,
                         filter_name = 'NullFilter')

class RandomSiblingFilter(SampleFilter):

    """Randomly select one sibling per family
    """
    
    @staticmethod
    def _sib_random_sample(sindex: SampleIndex):
        selection = np.sort(pd.Series(range(sindex.n)).groupby(sindex.fid).apply(np.random.choice))
        return selection

    def __init__(self):
        super().__init__(filter_function = RandomSiblingFilter._sib_random_sample,
                         filter_name = 'RandomSibling')

class RandomSubsampleFilter(SampleFilter):
    """Randomly subsample `k` individuals
    """
    def _random_subsample(self,
                          sindex: SampleIndex):
        k = self.k
        n = sindex.n
        if k is None:
            k = n
        selection = np.sort(np.random.permutation(n)[:k])
        return selection
    def __init__(self, k: int):
        self.k = k
        super().__init__(filter_function = self._random_subsample,
                         filter_name = 'RandomSibling')



class RandomSiblingSubsampleFilter(SampleFilter):
    
    """Randomly subsample `k` families, choosing one offspring per family
    """
    def _sib_random_subsample(self, sindex: SampleIndex):
        selection = pd.Series(range(sindex.n)).groupby(sindex.fid).apply(np.random.choice)
        k = self.k
        if k is None or len(selection) < k:
            k = len(selection)
        selection = np.sort(selection[:k])
        return selection
    def __init__(self, k: int):
        self.k = k
        super().__init__(filter_function = self._sib_random_subsample,
                         filter_name = 'RandomSiblingSubsample')


class SiblingPairFilter(SampleFilter):
    
    """
    Subsample 2 siblings each from k families with at least two siblings
    """
    def _sibpair_subsample(self, sindex: SampleIndex):
        if self.k is None:
            k = len(family_subset)
        else:
            k = self.k
        family_subset = np.random.permutation(np.unique(sindex.fid[pd.Series(sindex.fid).duplicated()]))[:k]
        sub_inds = np.where(pd.Series(sindex.fid).isin(family_subset))[0]
        selection = pd.Series(sub_inds).groupby(sindex.fid[sub_inds]).apply(np.random.choice, size=2, replace = False)
        selection = np.sort(np.concatenate(selection.values))
        return selection
    def __init__(self, k: int = None):
        self.k = k
        super().__init__(filter_function = self._sibpair_subsample,
                         filter_name = 'RandomSiblingSubsample')
