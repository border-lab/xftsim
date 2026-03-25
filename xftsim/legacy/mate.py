"""
This module contains functions and classes for implementing different mating regimes in the context of forward time genetics simulations.

Functions:

    _solve_qap_ls: Private function that solves the Quadratic Assignment Problem using LocalSolver.

Classes:

    MatingRegime: Base class for defining mating regimes.
    RandomMatingRegime: A class for implementing random mating.
    LinearAssortativeMatingRegime: A class for implementing linear assortative mating.
    KAssortativeMatingRegime: A class for implementing k-assortative mating.
    BatchedMatingRegime: A class for batching individuals to improve mating regime performance.
"""
import warnings
import numpy as np
import pandas as pd
import nptyping as npt
import xarray as xr
from nptyping import NDArray, Int8, Int64, Float64, Bool, Shape
from typing import Any, Hashable, List, Iterable, Callable, Union, Dict
from functools import cached_property
from dataclasses import dataclass, field
import math
import xftsim as xft


## todo add methods for offspring per iid etc
## todo add methods for checking validity
## TODO properties
class MateAssignment:
    """
    Represents a mate assignment for a given generation of individuals.

    Parameters
    ----------
    generation : int
        The generation number.
    maternal_sample_index : xft.index.SampleIndex
        The sample index for the maternal individuals.
    paternal_sample_index : xft.index.SampleIndex
        The sample index for the paternal individuals.
    previous_generation_sample_index : xft.index.SampleIndex
        The sample index for the previous generation.
    n_offspring_per_pair : NDArray[Shape["*"], Int64]
        An array containing the number of offspring per mating pair.
    n_females_per_pair : NDArray[Shape["*"], Int64]
        An array containing the number of female offspring per mating pair.
    sex_aware : bool, optional (default=False)
        Whether the mate assignment is sex-aware.
    """
    def __init__(self,
                 generation: int,
                 maternal_sample_index: xft.index.SampleIndex, 
                 paternal_sample_index: xft.index.SampleIndex, 
                 previous_generation_sample_index: xft.index.SampleIndex, 
                 n_offspring_per_pair: NDArray[Shape["*"], Int64],
                 n_females_per_pair: NDArray[Shape["*"], Int64],
                 sex_aware: bool=False,
                 ):
        ## catch duplicates ## TODO
        # maternal_iids,paternal_iids,n_offspring_per_pair = xft.utils.merge_duplicate_pairs(maternal_iids,
                                                                                  # paternal_iids,
                                                                                  # n_offspring_per_pair,
                                                                                  # sort = not sex_aware)
        self._maternal_sample_index = maternal_sample_index
        self._paternal_sample_index = paternal_sample_index
        self.generation = generation
        self.n_offspring_per_pair = np.array(n_offspring_per_pair)
        self.n_females_per_pair = np.array(n_females_per_pair)
        self.n_males_per_pair = self.n_offspring_per_pair - self.n_females_per_pair
        self._family_inds = np.arange(maternal_sample_index.n)
        self.previous_generation_sample_index = previous_generation_sample_index
        self.sex_aware = sex_aware

    def get_mate_phenotypes(self, 
                        phenotypes: xr.DataArray, 
                        component_index: xft.index.ComponentIndex = None,
                        full: bool = True):
        """
        Retrieves mate phenotypes based on the given phenotypes data.

        Parameters
        ----------
        phenotypes : xr.DataArray
            The phenotypes data array.
        component_index : xft.index.ComponentIndex, optional
            The component index for the phenotypes data array.
        full : bool
            Ignore component_index and get all components.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the mate phenotypes.
        """
        if full:
            component_index = phenotypes.xft.get_component_indexer()
        elif component_index is None:
            component_index = phenotypes.xft.grep_component_index('phenotype')[dict(vorigin_relative=-1)]
        Y_maternal = phenotypes.xft[self._maternal_sample_index, component_index]
        Y_paternal = phenotypes.xft[self._paternal_sample_index, component_index]
        new_index = xft.utils.paste([Y_maternal.iid.values,
                                     Y_paternal.iid.values], sep = ":")        
        Y_maternal = Y_maternal.xft.reindex_components(Y_maternal.xft.get_component_indexer().to_vorigin(0)).to_pandas().set_index(new_index)
        Y_paternal = Y_paternal.xft.reindex_components(Y_paternal.xft.get_component_indexer().to_vorigin(1)).to_pandas().set_index(new_index)
        return pd.concat([Y_maternal, Y_paternal], axis=1)


    @staticmethod
    def reduce_merge(assignments: Iterable):
        """
        Merges a list of MateAssignment objects into a single MateAssignment object.

        Parameters
        ----------
        assignments : Iterable
            An iterable of MateAssignment objects to be merged.

        Returns
        -------
        MateAssignment
            A new MateAssignment object resulting from the merge of the input assignments.
        """
        return MateAssignment(
                              generation = int(np.max([x.generation for x in assignments])),
                              maternal_sample_index = xft.index.XftIndex.reduce_merge([x._maternal_sample_index for x in assignments], deduplicate=False),
                              paternal_sample_index = xft.index.XftIndex.reduce_merge([x._paternal_sample_index for x in assignments], deduplicate=False), 
                              n_offspring_per_pair = np.concatenate([x.n_offspring_per_pair for x in assignments]),
                              n_females_per_pair = np.concatenate([x.n_females_per_pair for x in assignments]),
                              # sex_aware = bool(np.prod([x.sex_aware for x in assignments])),      
                              previous_generation_sample_index = xft.index.XftIndex.reduce_merge([x.previous_generation_sample_index for x in assignments]), 
                              )
    @property
    def _expanded_indices(self):
        """
        A NumPy array containing the expanded indices for the offspring.

        Returns
        -------
        np.ndarray
            An array containing the expanded indices for the offspring.
        """
        return np.repeat(range(self._maternal_sample_index.n),
                         self.n_offspring_per_pair)

    @property
    def reproducing_maternal_index(self):
        """
        The maternal index for reproducing individuals.

        Returns
        -------
        xft.index.SampleIndex
            The maternal index for reproducing individuals.
        """
        return self._maternal_sample_index.iloc(self._expanded_indices)
    
    @property
    def reproducing_paternal_index(self):
        """
        The paternal index for reproducing individuals.

        Returns
        -------
        xft.index.SampleIndex
            The paternal index for reproducing individuals.
        """
        return self._paternal_sample_index.iloc(self._expanded_indices)

    @property
    def n_females(self):
        """
        The total number of female offspring.

        Returns
        -------
        int
            The total number of female offspring.
        """
        return np.sum(self.n_females_per_pair)

    @property
    def n_males(self):
        """
        The total number of male offspring.

        Returns
        -------
        int
            The total number of male offspring.
        """
        return np.sum(self.n_males_per_pair)

    @property
    def n_reproducing_pairs(self):
        """
        The total number of reproducing pairs.

        Returns
        -------
        int
            The total number of reproducing pairs.
        """
        return np.sum(self.n_offspring_per_pair>=1)
    
    @property
    def n_total_offspring(self):
        """
        The total number of offspring.

        Returns
        -------
        int
            The total number of offspring.
        """
        return np.sum(self.n_offspring_per_pair)

    @property
    def offspring_iids(self):
        """
        The unique identifiers for the offspring.

        Returns
        -------
        np.ndarray
            An array containing the unique identifiers for the offspring.
        """
        return xft.utils.ids_from_generation_range(self.generation+1, self.n_total_offspring) ## TODO possible gotcha with chunking?
    @property
    def offspring_fids(self):
        """
        The family identifiers for the offspring.

        Returns
        -------
        np.ndarray
            An array containing the family identifiers for the offspring.
        """
        return xft.utils.ids_from_generation(self.generation+1, self._family_inds)[self._expanded_indices]
    @property
    def offspring_sex(self):
        """
        The sex of the offspring.

        Returns
        -------
        np.ndarray
            An array containing the sex of the offspring.
        """
        return np.concatenate([np.repeat([0,1],[x,y]) for (x,y) in zip(self.n_females_per_pair,
                                                                       self.n_males_per_pair)])

    @property
    def is_constant_population(self):
        """
        TODO property to determine if the population is constant or not.

        Returns
        -------
        bool
            True if the population is constant, False otherwise.
        """
        raise NotImplementedError ## TODO

    # @cached_property## TODO possible gotcha with chunking?
    @property## TODO possible gotcha with chunking?
    def offspring_sample_index(self):
        """
        The sample index for the offspring.

        Returns
        -------
        xft.index.SampleIndex
            The sample index for the offspring.
        """
        return xft.index.SampleIndex(iid=self.offspring_iids,
                                    fid=self.offspring_fids,
                                    sex=self.offspring_sex,
                                    generation=self._maternal_sample_index.generation+1)


    def get_mating_frame(self):
        """
        Constructs a DataFrame containing mate phenotypes regardless of reproductive success.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing mating information.
        """
        frame = np.hstack([self._maternal_sample_index.coord_frame.to_numpy(),
                           self._paternal_sample_index.coord_frame.to_numpy(),
                           xft.utils.ensure2D(self.n_offspring_per_pair),
                           xft.utils.ensure2D(self.n_females_per_pair),
                           ])
        frame = pd.DataFrame(frame, columns=[
                                            'maternal_sample', 'maternal_iid', 'maternal_fid', 'maternal_sex',
                                            'paternal_sample', 'paternal_iid', 'paternal_fid', 'paternal_sex',
                                            'n_offspring',
                                            'n_female_spring',
                                            ])
        return frame

    def get_reproduction_frame(self):
        """
        Constructs a DataFrame containing information relating to mates and offspring.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing reproduction information.
        """
        frame = np.hstack([self.offspring_sample_index.coord_frame.to_numpy(),
                           self.reproducing_maternal_index.coord_frame.to_numpy(),
                           self.reproducing_paternal_index.coord_frame.to_numpy(),
                           ])
        frame = pd.DataFrame(frame, columns=['sample', 'iid', 'fid', 'sex',
                                            'maternal_sample', 'maternal_iid', 'maternal_fid', 'maternal_sex',
                                            'paternal_sample', 'paternal_iid', 'paternal_fid', 'paternal_sex',
                                            ])
        return frame.set_index('sample')

    @property
    def maternal_integer_index(self):
        """
        The integer index for the maternal individuals.

        Returns
        -------
        np.ndarray
            An array containing the integer index for the maternal individuals.
        """
        return xft.utils.match(self.reproducing_maternal_index.unique_identifier, 
                              self.previous_generation_sample_index.unique_identifier)
    @property
    def paternal_integer_index(self):
        """
        The integer index for the paternal individuals.

        Returns
        -------
        np.ndarray
            An array containing the integer index for the paternal individuals.
        """
        return xft.utils.match(self.reproducing_paternal_index.unique_identifier, 
                              self.previous_generation_sample_index.unique_identifier)

    def update_pedigree(self, pedigree):
        raise NotImplementedError ## TODO

    def trio_view(self, pheno_parental, pheno_offspring):
        """
        Returns an array with the phenotypes of offspring, followed by the phenotypes of their parents
        in the same order as the order of offspring in this MateAssignment.

        Parameters
        ----------
        pheno_parental : xr.DataArray
            An xarray DataArray containing the phenotypes of the parents.
        pheno_offspring : xr.DataArray
            An xarray DataArray containing the phenotypes of the offspring.

        Returns
        -------
        np.ndarray
            An array with the phenotypes of offspring, followed by the phenotypes of their parents.

        """
        return np.hstack([pheno_offspring.data, 
                          pheno_parental.data[self.maternal_integer_index],
                          pheno_parental.data[self.paternal_integer_index]])





class MatingRegime:
    """
    A class for defining a mating regime to simulate the reproductive behavior of a population.

    Parameters
    ----------
    mateFunction : Callable, optional
        A function that specifies how the mating process is carried out. Default is None.
    offspring_per_pair : Union[Callable, int, xft.utils.VariableCount], optional
        The number of offspring per mating pair. This can be a callable function, an integer, or a VariableCount object. Default is xft.utils.ConstantCount(1).
    mates_per_female : Union[Callable, int, xft.utils.VariableCount], optional
        The number of mating partners each female has. This can be a callable function, an integer, or a VariableCount object. Default is xft.utils.ConstantCount(1).
    female_offspring_per_pair : Union[Callable, str, int, xft.utils.VariableCount], optional
        The number of female offspring per mating pair. This can be a callable function, a string, an integer, or a VariableCount object. If set to 'balanced', the number of female offspring will be randomly assigned from a balanced range (0, ..., total_offspring). Default is 'balanced'.
    sex_aware : bool, optional
        Whether the mating process should take sex into account. If True, females and males will be paired up based on their sex. If False, the pairs will be randomly assigned. Default is False.
    exhaustive : bool, optional
        Whether the mating pairs should be enumerated exhaustively or randomly. If True, all possible pairings will be enumerated before repeating. If False, the pairings will be randomly assigned with replacement. Default is True.
    component_index : xft.index.ComponentIndex, optional
        Which phenotype components (if any) are used in assigning mates
    haplotypes : bool, optional
        Flag indeicating if haplotype data is used to assign mates (defaults to False)

    Attributes
    ----------
    sex_aware : bool
        Whether the mating process should take sex into account.
    offspring_per_pair : Union[Callable, int, xft.utils.VariableCount]
        The number of offspring per mating pair.
    mates_per_female : Union[Callable, int, xft.utils.VariableCount]
        The number of mating partners each female has.
    female_offspring_per_pair : Union[Callable, str, int, xft.utils.VariableCount]
        The number of female offspring per mating pair.
    exhaustive : bool
        Whether the mating pairs should be enumerated exhaustively or randomly.
    mateFunction : Callable
        A function that specifies how the mating process is carried out.
    expected_offspring_per_pair : float
        The expected number of offspring per mating pair.
    expected_mates_per_female : float
        The expected number of mating partners each female has.
    expected_female_offspring_per_pair : float
        The expected number of female offspring per mating pair.
    population_growth_factor : float
        The population growth factor.

    Methods
    -------
    get_potential_mates(haplotypes: xr.DataArray = None, phenotypes: xr.DataArray = None) -> (NDArray, NDArray)
        Returns the potential female and male mating partners based on the sex awareness parameter.
    enumerate_assignment(female_indices: NDArray, male_indices: NDArray, haplotypes: xr.DataArray = None, phenotypes: xr.DataArray = None) -> MateAssignment
        Enumerates the mating assignments.
    mate(haplotypes: xr.DataArray = None, phenotypes: xr.DataArray = None, control: dict = None) -> MateAssignment
        Calls the mateFunction to perform the mating process.
    """
    def __init__(self, 
                 mateFunction: Callable = None,
                 offspring_per_pair: Union[Callable, int, xft.utils.VariableCount] = xft.utils.ConstantCount(1),
                 mates_per_female: Union[Callable, int, xft.utils.VariableCount] =  xft.utils.ConstantCount(1),
                 female_offspring_per_pair: Union[Callable, str, int, xft.utils.VariableCount] = 'balanced', ## doesn't make total sense
                 sex_aware: bool = False,
                 exhaustive: bool = True,
                 component_index: xft.index.ComponentIndex = None,
                 haplotypes: bool = False,
                 ):
        ## replace integer counts with ConstantCount objects
        if isinstance(offspring_per_pair, int):
            offspring_per_pair = xft.utils.ConstantCount(offspring_per_pair)
        if isinstance(mates_per_female, int):
            mates_per_female = xft.utils.ConstantCount(mates_per_female)
        if isinstance(female_offspring_per_pair, int):
            female_offspring_per_pair = xft.utils.ConstantCount(female_offspring_per_pair)
        ## set attributes
        self.sex_aware = sex_aware
        self.offspring_per_pair = offspring_per_pair
        self.mates_per_female = mates_per_female
        self.female_offspring_per_pair = female_offspring_per_pair
        self.exhaustive = exhaustive
        self._mateFunction = mateFunction
        self._component_index = component_index
        self._haplotypes = haplotypes

    def get_potential_mates(self,
                            haplotypes = None,
                            phenotypes: xr.DataArray = None,
                            ) -> (NDArray, NDArray):
        """
        Return potential mating pairs.

        Parameters
        ----------
        haplotypes: NHaplotypeArray
            The haplotypes to use for mating.
        phenotypes: xr.DataArray
            The phenotypes to use for mating.

        Returns
        -------
        (NDArray, NDArray)
            The potential female and male mating indices.
        """
        # Get sample identifiers and sex from haplotypes
        sample_ids = haplotypes.samples.unique_identifier
        sex = haplotypes.sex

        if self.sex_aware:
            female_indices = sample_ids[sex == 0]
            male_indices = sample_ids[sex == 1]
            assert len(female_indices) == len(male_indices), "Unbalanced population"
        else:
            permuted_indices = np.random.permutation(sample_ids)
            female_indices = np.sort(permuted_indices[0::2])
            male_indices = np.sort(permuted_indices[1::2])

        return (female_indices, male_indices)

    def enumerate_assignment(self,
                             female_indices: NDArray,
                             male_indices: NDArray,
                             haplotypes: xr.DataArray = None, 
                             phenotypes: xr.DataArray = None,
                             ) -> MateAssignment:
        """
        Enumerate the mate assignments.

        Parameters
        ----------
        female_indices: NDArray
            The indices of the females to mate.
        male_indices: NDArray
            The indices of the males to mate.
        haplotypes: xr.DataArray
            The haplotypes to use for mating.
        phenotypes: xr.DataArray
            The phenotypes to use for mating.

        Returns
        -------
        MateAssignment
            The mate assignments.
        """
        matings_per_female = self.mates_per_female.draw(len(female_indices))
        n_matings = np.sum(matings_per_female)

        offspring_per_mate_pair = self.offspring_per_pair.draw(n_matings)

        female_mate_indices = xft.utils.exhaustive_enumerate(female_indices, matings_per_female)
        if self.exhaustive:
            male_mate_indices = xft.utils.exhaustive_enumerate(male_indices, matings_per_female)
        else:
            raise NotImplementedError#male_mate_indices = np.random.choice(male_indices, n_matings)

        if not isinstance(self.female_offspring_per_pair, str):
            raise NotImplementedError
        elif self.female_offspring_per_pair == 'balanced':
                n_females_per_pair = np.apply_along_axis(lambda x: np.random.randint(x+1), 0, offspring_per_mate_pair)
        else:            
            raise NotImplementedError
        if self.exhaustive:
            male_mate_indices = xft.utils.exhaustive_enumerate(male_indices, matings_per_female)
        else:
            raise NotImplementedError
            # male_mate_indices = np.random.choice(male_indices, n_matings)

        return MateAssignment(generation = haplotypes.generation,
                              maternal_sample_index = self._sample_indexer[female_mate_indices],
                              paternal_sample_index = self._sample_indexer[male_mate_indices],
                              previous_generation_sample_index = self._sample_indexer,
                              n_offspring_per_pair = offspring_per_mate_pair,
                              n_females_per_pair = n_females_per_pair,
                              sex_aware = self.sex_aware,
                              )

    @property
    def mateFunction(self):
        if self._mateFunction is not None:
            return self._mateFunction
        else:
            raise NotImplementedError("'mateFunction' not implemented")


    @mateFunction.setter
    def mateFunction(self, value):
        self._mateFunction = value

    def mate(self,
            haplotypes: xr.DataArray = None,
            phenotypes: xr.DataArray = None,
            control: dict = None,
             ) -> MateAssignment:
        """
        Mate individuals.

        Parameters
        ----------
        haplotypes : xarray.DataArray, optional
            The haplotypes of the individuals, by default None.
        phenotypes : xarray.DataArray, optional
            The phenotypes of the individuals, by default None.
        control : dict, optional
            The mating control parameters, by default None.

        Returns
        -------
        MateAssignment
            The mate assignment result.
        """
        return self._mateFunction(haplotypes, phenotypes)

    @property
    def expected_offspring_per_pair(self) -> float:
        """
        Get the expected offspring per pair.

        Returns
        -------
        float
            The expected offspring per pair.

        Raises
        ------
        NotImplementedError
            If the offspring count is not an integer or a VariableCount.
        """
        if type(self.offspring_per_pair) is int:
            return self.offspring_per_pair
        elif isinstance(self.offspring_per_pair, xft.utils.VariableCount):
            return self.offspring_per_pair.expectation
        else:
            raise NotImplementedError
    
    @property
    def expected_mates_per_female(self) -> float:
        """
        Get the expected mates per female.

        Returns
        -------
        float
            The expected mates per female.

        Raises
        ------
        NotImplementedError
            If the mates count is not an integer or a VariableCount.
        """
        if type(self.mates_per_female) is int:
            return self.mates_per_female
        elif isinstance(self.mates_per_female, xft.utils.VariableCount):
            return self.mates_per_female.expectation
        else:
            raise NotImplementedError

    @property
    def expected_female_offspring_per_pair(self) -> float:
        """
        Get the expected female offspring per pair.

        Returns
        -------
        float
            The expected female offspring per pair.

        Raises
        ------
        NotImplementedError
            If the female offspring count is not an integer or a VariableCount.
        """
        if type(self.female_offspring_per_pair) is int:
            return self.female_offspring_per_pair
        elif isinstance(self.female_offspring_per_pair, xft.utils.VariableCount):
            return self.female_offspring_per_pair.expectation
        else:
            raise NotImplementedError

    @property
    def population_growth_factor(self) -> float:
        """
        Get the population growth factor.

        Returns
        -------
        float
            The population growth factor.
        """
        return .5 * self.expected_offspring_per_pair * self.expected_mates_per_female

    def _dependency_graph(self):
        edge_list = []
        if self._haplotypes:
            edge_list.append(('proband\nhaplotypes', 'Mating\nregime'))
        if self._component_index is not None:
            for input_node in self._component_index._nodes:
                edge_list.append((input_node, 'Mating\nregime'))
        return edge_list

    @property
    def dependency_graph_edges(self):
        return self._dependency_graph()

    @property
    def dependency_graph(self):
        import networkx as nx
        G = nx.DiGraph()
        G.add_edges_from(self.dependency_graph_edges)
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        return (G,pos)

    def draw_dependency_graph(self, 
                              node_color='none', 
                              node_size = 1500, 
                              arrowsize = 7,
                              font_size=6, 
                              margins=.1, 
                              **kwargs):
        import networkx as nx
        G,pos = self.dependency_graph
        nx.draw_networkx(G,pos, 
                         node_color=node_color, 
                         node_size = node_size, 
                         font_size=font_size,
                         margins=margins,
                         arrowsize=arrowsize,
                         **kwargs)


class RandomMatingRegime(MatingRegime):
    """
    A mating regime that randomly pairs individuals and produces offspring with balanced numbers of males and females.

    Parameters
    ----------
    offspring_per_pair : xft.utils.VariableCount, optional
        Number of offspring produced per mating pair, by default xft.utils.ConstantCount(1)
    mates_per_female : xft.utils.VariableCount, optional
        Number of males that mate with each female, by default xft.utils.ConstantCount(1)
    female_offspring_per_pair : Union[str, xft.utils.VariableCount], optional
        The number of female offspring per mating pair. If "balanced", the number is balanced with
        the number of male offspring. By default, "balanced".
    sex_aware : bool, optional
        If True, randomly paired individuals are selected so that there is an equal number of males and females.
        Otherwise, random pairing is performed. By default, False.
    exhaustive : bool, optional
        If True, perform exhaustive enumeration of potential mates. If False, perform random sampling. By default, True.
    """
    def __init__(self, 
                 offspring_per_pair: xft.utils.VariableCount = xft.utils.ConstantCount(1),
                 mates_per_female: xft.utils.VariableCount =  xft.utils.ConstantCount(2),
                 female_offspring_per_pair: Union[str, xft.utils.VariableCount] = 'balanced', ## doesn't make total sense
                 sex_aware: bool = False,
                 exhaustive: bool = True,
                 ):
        super().__init__(self,
                         offspring_per_pair=offspring_per_pair,
                         mates_per_female=mates_per_female,
                         female_offspring_per_pair=female_offspring_per_pair,
                         sex_aware=sex_aware,
                         exhaustive=exhaustive,
                         component_index = None,
                         haplotypes = False,
                         )

    def mate(self, 
             haplotypes: xr.DataArray = None, 
             phenotypes: xr.DataArray = None,
             control: dict = None,
             ):
        """
        Mate individuals randomly with balanced numbers of males and females.

        Parameters
        ----------
        haplotypes : xr.DataArray, optional
            Array containing haplotypes, by default None
        phenotypes : xr.DataArray, optional
            Array containing phenotypes, by default None
        control : dict, optional
            Control dictionary, by default None

        Returns
        -------
        MateAssignment
            An object containing the maternal and paternal sample indices, the number of offspring per pair,
            and the number of female offspring per pair.
        """
        self._sample_indexer = haplotypes.samples.to_sample_index()
        female_indices, male_indices = self.get_potential_mates(haplotypes, phenotypes)

        female_indices = np.random.permutation(female_indices)
        male_indices = np.random.permutation(male_indices)

        return self.enumerate_assignment(female_indices=female_indices,
                                         male_indices=male_indices,
                                         haplotypes=haplotypes,
                                         phenotypes=phenotypes,
                                         )

# mr =RandomMatingRegime()





class LinearAssortativeMatingRegime(MatingRegime):
    """
    A linear assortative mating regime that performs mate selection based on a specified component index. 
    Speifically, individuals are mated such that the cross-mate correlations across all specified components
    are equal to `r`. This reflects mating on a linear combination of phenotypes and does not generalize to
    many cross-mate correlation stuctures observed in practice, but is more efficient. 

    Parameters
    ----------
    component_index : xft.index.ComponentIndex
        The component index used to select mating pairs based on the correlation between the phenotype values.
    r : float, optional
        The correlation coefficient, a value between -1 and 1. Defaults to 0.
    offspring_per_pair : Union[int, xft.utils.VariableCount], optional
        The number of offspring per pair. If int, it will be converted to a ConstantCount object. Defaults to 1.
    mates_per_female : Union[int, xft.utils.VariableCount], optional
        The number of mates per female. If int, it will be converted to a ConstantCount object. Defaults to 1.
    female_offspring_per_pair : Union[str, int, xft.utils.VariableCount], optional
        The number of female offspring per mating pair. If 'balanced', the number of females is randomly selected 
        for each pair to balance the sex ratio. If int, it will be converted to a ConstantCount object. Defaults 
        to 'balanced'.
    sex_aware : bool, optional
        If True, only mating pairs with different sex are allowed. Defaults to False.
    exhaustive : bool, optional
        If True, all possible mating pairs will be enumerated. If False, pairs will be randomly selected.
        Defaults to True.
    
    Raises
    ------
    AssertionError
        If r is not between -1 and 1.
        If the correlation r is not feasible for the number of phenotypes in the component index.


    TODO: see also
    
    """
    def __init__(self, 
                 component_index: xft.index.ComponentIndex,
                 r: float = 0,
                 offspring_per_pair: Union[int, xft.utils.VariableCount] = xft.utils.ConstantCount(1),
                 mates_per_female: Union[int, xft.utils.VariableCount] =  xft.utils.ConstantCount(2),
                 female_offspring_per_pair: Union[str, int, xft.utils.VariableCount] = 'balanced', ## doesn't make total sense
                 sex_aware: bool = False,
                 exhaustive: bool = True,
                 ):
        super().__init__(self,
                         offspring_per_pair=offspring_per_pair,
                         mates_per_female=mates_per_female,
                         female_offspring_per_pair=female_offspring_per_pair,
                         sex_aware=sex_aware,
                         exhaustive=exhaustive,
                         component_index=component_index,
                         haplotypes=False,
                         )
        ## Linear regime attributes
        self.r = r
        self.component_index = component_index
        self.K = component_index.k_total
        assert (-1. <= r <= 1.), "Invalid correlation"
        if r > 1/self.K or r < -1/self.K:
            warnings.warn(f"Linear regime with {self.K} phenotypes only feasible for -1/{self.K} <= r <= 1/{self.K}")

    def mate(self, 
             haplotypes: xr.DataArray = None, 
             phenotypes: xr.DataArray = None,
             control: dict = None,
             ):
        female_indices, male_indices = self.get_potential_mates(haplotypes, phenotypes)

        sample_indexer = phenotypes.xft.get_sample_indexer()
        female_indexer = sample_indexer[female_indices] 
        male_indexer = sample_indexer[male_indices] 

        ## downsample for balance
        n_new = np.min([female_indexer.n, male_indexer.n])
        female_indexer = female_indexer.at_most(n_new)
        male_indexer = male_indexer.at_most(n_new)

        
        female_components = xft.utils.standardize_array(phenotypes.xft[female_indexer,
                                                        self.component_index].data)
        male_components = xft.utils.standardize_array(phenotypes.xft[male_indexer,
                                                      self.component_index].data)

        n=male_components.shape[0]
        sum_scaled_mate1 = np.sum(female_components, axis=1)
        sum_scaled_mate2 = np.sum(male_components, axis=1)
        within_cov1 = np.cov(female_components, rowvar=False)
        within_cov2 = np.cov(male_components, rowvar=False)
        cross_cov = self.K**2 * self.r
        R = np.sum(cross_cov)/ np.sqrt(np.sum(within_cov1)*(np.sum(within_cov2)))
        mating_score1 = sum_scaled_mate1 * np.sqrt(R) + np.sqrt(np.abs(1-R))*np.random.normal(0,np.std(sum_scaled_mate1),n)#sum_scaled_mate1.shape[0])
        mating_score2 = sum_scaled_mate2 * np.sqrt(R) + np.sqrt(np.abs(1-R))*np.random.normal(0,np.std(sum_scaled_mate2),n)#sum_scaled_mate2.shape[0])

        return self.enumerate_assignment(female_indices=female_indices[np.argsort(mating_score1)[:n_new]],
                                         male_indices=male_indices[np.argsort(mating_score2)[:n_new]],
                                         haplotypes=haplotypes,
                                         phenotypes=phenotypes,
                                         )



class BatchedMatingRegime(MatingRegime):
    """
    BatchedMatingRegime class that batches mating assignments, either for the sake of efficiency or to
    simulate stratification.

    Parameters
    ----------
    regime : MatingRegime
        The mating regime object.
    max_batch_size : int
        Maximum size of each batch.

    Attributes
    ----------
    regime : MatingRegime
        The mating regime object.
    max_batch_size : int
        Maximum size of each batch.

    Methods
    -------
    batch(haplotypes, phenotypes, control)
        Split samples into batches.
    mate(haplotypes, phenotypes, control)
        Generate mating assignments in batches.

    """
    def __init__(self, 
                 regime: MatingRegime, 
                 max_batch_size: int,
                 ):
        self.regime= regime
        self.max_batch_size = max_batch_size

    def batch(self, 
              haplotypes: xr.DataArray = None, 
              phenotypes: xr.DataArray = None,
              control: dict = None,
              ):
        """
        Split samples into batches.

        Parameters
        ----------
        haplotypes : xarray.DataArray, optional
            Haplotypes array.
        phenotypes : xarray.DataArray, optional
            Phenotypes array.
        control : dict, optional
            Control parameters.

        Returns
        -------
        batches : list
            List of batches of samples.
        num_batches : int
            Number of batches.

        """
        N = haplotypes.n
        num_batches = math.ceil(N / self.max_batch_size)
        batches = [np.sort(x) for x in np.array_split(np.random.permutation(N), num_batches)]
        return (batches, num_batches)

    def mate(self,
             haplotypes: xr.DataArray = None,
             phenotypes: xr.DataArray = None,
             control: dict = None,
             ):
        """
        Generate mating assignments in batches and merge into single assignment object.

        Parameters
        ----------
        haplotypes : xarray.DataArray, optional
            Haplotypes array.
        phenotypes : xarray.DataArray, optional
            Phenotypes array.
        control : dict, optional
            Control parameters.

        Returns
        -------
        mate_assignments : MateAssignment
            Mating assignments.

        """
        batches, num_batches = self.batch(haplotypes,
                                          phenotypes,
                                          control)
        assignments = []
        for tt,batch in enumerate(batches):
            print(f"Batch {tt}")
            assignments += [self.regime.mate(haplotypes[batch],
                                             phenotypes[batch],
                                             control)]
        return MateAssignment.reduce_merge(assignments)




class FilteredMatingRegime(MatingRegime):
    """
    Applies filter to sample prior to commencing mating

    Parameters
    ----------
    regime : MatingRegime
        The mating regime to be filtered.
    sample_filter : xft.filters.SampleFilter
        The sample filter to apply

    mate(haplotypes, phenotypes, control)
        Generate mating assignments in batches.

    """
    def __init__(self, 
                 regime: MatingRegime, 
                 sample_filter: xft.filters.SampleFilter,
                 ):
        self.regime = regime
        self.sample_filter = sample_filter


    def mate(self,
             haplotypes: xr.DataArray = None,
             phenotypes: xr.DataArray = None,
             control: dict = None,
             ):
        mate_inds = np.sort(self.sample_filter.filter(phenotypes))
        return self.regime.mate(haplotypes[mate_inds,:],
                                phenotypes[mate_inds,:],
                                control)


def _solve_qap_ls(Y, Z, R, nb_threads=6, time_limit=30, tolerance=.001, 
                  verbosity=1, time_between_displays=1, termination_interval=15):
    """
    Solves the Quadratic Assignment Problem (QAP) using the LocalSolver optimization solver.
    
    Parameters
    ----------
    Y : numpy.ndarray
        The first set of mate phenotypes.
    Z : numpy.ndarray
        The second set of mate phenotypes.
    R : numpy.ndarray
        The target cross-correlation matrix.
    nb_threads : int, optional
        The number of threads to be used for the optimization.
    time_limit : int, optional
        The amount of time (in seconds) to run the optimization passed to `.param.set_time_limit()`
    tolerance : float, optional
        The minimum threshold for the objective function to terminate optimization.
    verbosity : int, optional
        The amount of time (in seconds) to run the optimization.

    Returns
    -------
    P : numpy.ndarray
        A permutation matrix.
    """
    import localsolver
    class TerminateSolver:
        def __init__(self, interval: int = 20):
            self.last_best_value = 100.0
            self.last_best_running_time = 0
            self.interval = interval

        def callback(self, ls, cb_type):
            stats = ls.statistics
            obj = ls.model.objectives[0]
            if obj.value < self.last_best_value:
                self.last_best_running_time = stats.running_time
                self.last_best_value = obj.value
            if stats.running_time - self.last_best_running_time > self.interval:
                print(f">>> No improvement during {self.interval} seconds: resolution is stopped")
                ls.stop()
    Y = np.apply_along_axis(lambda x: (x-np.mean(x))/np.std(x),0,Y)
    Z = np.apply_along_axis(lambda x: (x-np.mean(x))/np.std(x),0,Z)
    n = Y.shape[0]
    # for later use as initial value
    tmp = np.argsort(np.apply_along_axis(np.mean, 1, Y))[
        np.argsort(np.argsort(np.apply_along_axis(np.mean, 1, Z)))]
    const = np.trace(R @ R.T)
    # flows
    YY = Y @ Y.T / n
    # distance
    ZZ = Z @ Z.T / n
    # cost
    W = Y @ R @ Z.T / n

    with localsolver.LocalSolver() as ls:
        cb = TerminateSolver(int(termination_interval))
        ls.add_callback(localsolver.LSCallbackType.TIME_TICKED, cb.callback)
        ls.param.set_time_limit(int(time_limit))
        ls.param.set_nb_threads(int(nb_threads))
        ls.param.set_verbosity(verbosity)
        ls.param.set_time_between_displays(time_between_displays)

        model = ls.model
        array_YY = model.array(model.array(YY[i, :]) for i in range(n))
        array_W = model.array(model.array(W[i, :]) for i in range(n))
        # permutation
        p = model.list(n)
        model.constraint(model.eq(model.count(p), n))
        # objective
        qobj = model.sum(
            model.at(array_YY, p[i], p[j]) * ZZ[i, j] for j in range(n) for i in range(n))
        lobj = model.sum(model.at(array_W, p[i], i) for i in range(n))
        obj = (qobj - 2 * lobj + const)**.5
        model.minimize(obj)
        model.close()
        # set initial value of permutation
        p.value.clear()
        for pp in tmp:
            p.value.add(pp)
        # solve
        ls.param.set_objective_threshold(0,tolerance)
        ls.solve()

        # solution
        P = np.array([p.value.get(i) for i in range(n)])

        return P


##TODO add seed
class GeneralAssortativeMatingRegime(MatingRegime):
    """
    A class that implements the general assortative mating regimes. I.e., matches two sets of individuals with
    K phenotypes to achieve an arbitrary K x K cross-mate cross-correlation structure.

    Parameters
    ----------
    component_index : xft.index.ComponentIndex
        An object containing information about the components.
    cross_corr : ndarray
        The cross-correlation matrix of size K x K.
    offspring_per_pair : Union[int, xft.utils.VariableCount], optional
        The number of offspring per mating pair. Default is 1.
    mates_per_female : Union[int, xft.utils.VariableCount], optional
        The number of mates for each female. Default is 2.
    female_offspring_per_pair : Union[str, int, xft.utils.VariableCount], optional
        The number of offspring per mating pair for females. Default is 'balanced'.
    sex_aware : bool, optional
        Whether to consider sex in mating pairs. Default is False.
    exhaustive : bool, optional
        Whether to enumerate all possible pairs. Default is True.
    control : dict, optional
        A dictionary of control parameters passed to LocalSolver. Defaults are as follows: 
        nb_threads=4, time_limit=120, tolerance=1e-5, verbosity=1, time_between_displays=15

    Attributes
    ----------
    cross_corr : ndarray
        The cross-correlation matrix of size K x K.
    component_index : xft.index.ComponentIndex
        An object containing information about the components.
    K : int
        The total number of components.

    Methods
    -------
    mate(haplotypes: xr.DataArray = None, phenotypes: xr.DataArray = None, control: dict = None) -> xft.mate.MateAssignment:
        Mate haplotypes and phenotypes based on the K-assortative mating regime.
    """

    def __init__(self,
                 component_index: xft.index.ComponentIndex,
                 cross_corr: NDArray,
                 offspring_per_pair: Union[int, xft.utils.VariableCount] = xft.utils.ConstantCount(
                     1),
                 mates_per_female: Union[int, xft.utils.VariableCount] = xft.utils.ConstantCount(
                     2),
                 # doesn't make total sense
                 female_offspring_per_pair: Union[str, int,
                                                  xft.utils.VariableCount] = 'balanced',
                 sex_aware: bool = False,
                 exhaustive: bool = True,
                 control: dict = {},

                 ):
        super().__init__(self,
                         offspring_per_pair=offspring_per_pair,
                         mates_per_female=mates_per_female,
                         female_offspring_per_pair=female_offspring_per_pair,
                         sex_aware=sex_aware,
                         exhaustive=exhaustive,
                         )
        # Linear regime attributes
        self.cross_corr = cross_corr
        self.component_index = component_index
        self.K = component_index.k_total
        self.control = dict(nb_threads=4, time_limit=120, tolerance=1e-5, verbosity=1, time_between_displays=15)
        self.control.update(control)

    def mate(self,
             haplotypes: xr.DataArray = None,
             phenotypes: xr.DataArray = None,
             control: dict = {},
             ):
        """
        Mate haplotypes and phenotypes based on the K-assortative mating regime.

        Parameters
        ----------
        haplotypes : xr.DataArray, optional
            The haplotype data to be mated. Default is None.
        phenotypes : xr.DataArray, optional
            The phenotype data to be mated. Default is None.

        Returns
        -------
        assignment : xft.mate.MateAssignment
            The assignment of haplotypes to parents.
        """
        female_indices, male_indices = self.get_potential_mates(
            haplotypes, phenotypes)

        sample_indexer = phenotypes.xft.get_sample_indexer()
        female_indexer = sample_indexer[female_indices]
        male_indexer = sample_indexer[male_indices]

        # downsample for balance
        n_new = np.min([female_indexer.n, male_indexer.n])
        female_indexer = female_indexer.at_most(n_new)
        male_indexer = male_indexer.at_most(n_new)

        female_components = xft.utils.standardize_array(phenotypes.xft[female_indexer,
                                                                       self.component_index].data)
        male_components = xft.utils.standardize_array(phenotypes.xft[male_indexer,
                                                                     self.component_index].data)

        perm = _solve_qap_ls(
            female_components, male_components, self.cross_corr, 
                nb_threads=self.control['nb_threads'],
                time_limit=self.control['time_limit'],
                tolerance=self.control['tolerance'],
                verbosity=self.control['verbosity'],
                time_between_displays=self.control['time_between_displays'],
                )

        return self.enumerate_assignment(female_indices=female_indices[perm],
                                         male_indices=male_indices,
                                         haplotypes=haplotypes,
                                         phenotypes=phenotypes,
                                         )



#     couplings <- rbind(mate1_inds[order(mating_score1)],
#                        mate2_inds[order(mating_score2)])
#     couplings <- rbind(mate1_inds[order(mating_score1)],
#                        mate2_inds[order(mating_score2)])
#     return(couplings)







#     def _mateFunction(self, 
#                       haplotypes: xr.DataArray = None, 
#                       phenotypes: xr.DataArray = None):

#         sample_indexer = haplotypes.xft.get_sample_indexer()

#         if self.sex_aware:
#             female_indices = haplotypes.sample[haplotypes.sex==0]
#             male_indices = haplotypes.sample[haplotypes.sex==1]
#             assert len(female_indices) == len(male_indices), "Unbalanced population"
#         elif not self.sex_aware:
#             permuted_indices = np.random.permutation(haplotypes.sample)
#             female_indices = np.sort(permuted_indices[0::2])
#             male_indices = np.sort(permuted_indices[1::2])

#         matings_per_female = self.mates_per_female.draw(len(female_indices))
#         n_matings = np.sum(matings_per_female)

#         offspring_per_mate_pair = self.offspring_per_pair.draw(n_matings)

#         female_mate_indices = xft.utils.exhaustive_enumerate(female_indices, matings_per_female)

#         if not isinstance(self.female_offspring_per_pair, str):
#             raise NotImplementedError
#         elif self.female_offspring_per_pair == 'balanced':
#                 n_females_per_pair = np.apply_along_axis(lambda x: np.random.randint(x+1), 0, offspring_per_mate_pair)
#         else:            
#             raise NotImplementedError
#         if self.exhaustive:
#             male_mate_indices = xft.utils.exhaustive_enumerate(male_indices, matings_per_female)
#         else:
#             male_mate_indices = np.random.choice(male_indices, n_matings)
#         # print(f"male_indices: {male_indices}, n_matings: {n_matings}, male_mate_indices: {male_mate_indices}")

#         return MateAssignment(generation = haplotypes.generation,
#                               maternal_sample_index = sample_indexer[female_mate_indices],
#                               paternal_sample_index = sample_indexer[male_mate_indices],
#                               previous_generation_sample_index = sample_indexer,
#                               n_offspring_per_pair = offspring_per_mate_pair,
#                               n_females_per_pair = n_females_per_pair,
#                               sex_aware = self.sex_aware,
#                               )




#     def _mateFunction(self, 
#                       haplotypes: xr.DataArray = None, 
#                       phenotypes: xr.DataArray = None):

#         sample_indexer = haplotypes.xft.get_sample_indexer()

#         if self.sex_aware:
#             female_indices = haplotypes.sample[haplotypes.sex==0]
#             male_indices = haplotypes.sample[haplotypes.sex==1]
#             assert len(female_indices) == len(male_indices), "Unbalanced population"
#         elif not self.sex_aware:
#             permuted_indices = np.random.permutation(haplotypes.sample)
#             female_indices = permuted_indices[0::2]
#             male_indices = permuted_indices[1::2]

#         matings_per_female = self.mates_per_female.draw(len(female_indices))
#         n_matings = np.sum(matings_per_female)

#         offspring_per_mate_pair = self.offspring_per_pair.draw(n_matings)


#         female_mate_indices = xft.utils.exhaustive_enumerate(female_indices, matings_per_female)

#         if not isinstance(self.female_offspring_per_pair, str):
#             raise NotImplementedError
#         elif self.female_offspring_per_pair == 'balanced':
#                 n_females_per_pair = np.apply_along_axis(lambda x: np.random.randint(x+1), 0, offspring_per_mate_pair)
#         else:            
#             raise NotImplementedError
#         if self.exhaustive:
#             male_mate_indices = xft.utils.exhaustive_enumerate(male_indices, matings_per_female)
#         else:
#             male_mate_indices = np.random.choice(male_indices, n_matings)
#         # print(f"male_indices: {male_indices}, n_matings: {n_matings}, male_mate_indices: {male_mate_indices}")
#     pass

# class KtraitAssortativeMatingRegime(MatingRegime):
#     ## TODO
#     pass

# class ChunkedMatingRegime(MatingRegime):
#     ## TODO
#     pass


#     #     if (not cp) and (not sa) and (opp==1) 

#     # def mate(self, 
#     #          phenotypes: Phenotypes = None,
#     #          pedigree: Pedigree = None,
#     #          ) -> MateAssignment:
#     #     raise NotImplementedError
# # class SuperClass:
# #     def __init__(self):
# #         pass
# #     def mate(self, *args):
# #         raise NotImplementedError

# # class SubClass(SuperClass):
# #     def __init__(self, params):
# #         self.params = params

# #     def mate(self):
# #         return self.params

# mate_linear_order <- function(yz, r) {
#     ## shuffle mates to avoid sex differences
#     n <- 2*(nrow(yz)%/%2)
#     K = ncol(yz)
#     permuted_inds <- sample.int(nrow(yz),n)
#     mate1_inds <- sort(permuted_inds[1:(n/2)])
#     mate2_inds <- sort(permuted_inds[(n/2+1):n])
#     scaled_mate1 <- (apply(yz[mate1_inds,,drop=FALSE ],2,scale))
#     scaled_mate2 <- (apply(yz[mate2_inds,,drop=FALSE ],2,scale))
#     sum_scaled_mate1 <- rowSums(scaled_mate1)
#     sum_scaled_mate2 <- rowSums(scaled_mate2)
#     within_cov1 <- cov(scaled_mate1)
#     within_cov2 <- cov(scaled_mate2)
#     cross_cov <- K^2 * r
#     R = sum(cross_cov)/ sqrt(sum(within_cov1)*(sum(within_cov2)))
#                                         #     print(R)
#     mating_score1 <- sum_scaled_mate1 * sqrt(R) + sqrt(abs(1-R))*rnorm(n/2,0,sd(sum_scaled_mate1))
#     mating_score2 <- sum_scaled_mate2 * sqrt(R) + sqrt(abs(1-R))*rnorm(n/2,0,sd(sum_scaled_mate2))
#     couplings <- rbind(mate1_inds[order(mating_score1)],
#                        mate2_inds[order(mating_score2)])
#     couplings <- rbind(mate1_inds[order(mating_score1)],
#                        mate2_inds[order(mating_score2)])
#     return(couplings)
# }