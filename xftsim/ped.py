import warnings
import numpy as np
import pandas as pd
import nptyping as npt
import xarray as xr
from nptyping import NDArray, Int8, Int64, Float64, Bool, Shape
from typing import Any, Hashable, List, Iterable, Callable, Union, Dict
from functools import cached_property
# from dataclasses import dataclass, field
import networkx as nx

import xftsim as xft


class Pedigree:
    """
    A class representing a pedigree as a graph.

    Attributes
    ----------
    G : nx.DiGraph
        The directed graph representing the pedigree.
    _generation : dict
        A dictionary containing node generations.
    _fid : dict
        A dictionary containing node family IDs.
    _generational_depth : int
        The generational depth of the tree.

    Methods
    -------
    generation(K: int):
        Returns the subgraph of nodes with generation K.
    generations(gens):
        Returns the subgraph of nodes with generations in the given iterable.
    current_generation(K):
        Returns the subgraph of nodes in the current generation.
    most_recent_K_generations():
        Returns the subgraph of nodes in the most recent K generations.
    _add_edges_from_arrays(x, y):
        Adds edges from arrays x and y.
    add_offspring(mating: xft.mate.MateAssignment):
        Adds offspring nodes and edges to the pedigree based on a MateAssignment object.
    _get_trios():
        TODO
    """
    def __init__(self,
                 founder_samples: "xft.struct.SampleMeta",
                 ):
        """
        Initialize the Pedigree object with founder samples and their generation.

        Parameters
        ----------
        founder_samples : xft.struct.SampleMeta
            The founder samples metadata.
        """
        self.G = nx.DiGraph()
        self._generation_dict = {}
        self._fid_dict = {}
        self.founder_generation = founder_samples.generation
        self.current_generation = self.founder_generation

        self._add_nodes(founder_samples)

    def _add_nodes(self, samples: "xft.struct.SampleMeta"):
        uid = samples.unique_identifier

        self.current_generation = max(self.current_generation, samples.generation)
        self._generation_dict.update({node: samples.generation for node in uid})
        self._fid_dict.update({node: fid for (node, fid) in zip(uid, samples.fid)})

        self.G.add_nodes_from(uid)


    @property
    def generational_depth(self):
        return self.current_generation - self.founder_generation
    
    def generation(self, K: int):
        """
        Returns the subgraph of nodes with generation K.

        Parameters
        ----------
        K : int
            The generation number.

        Returns
        -------
        nx.subgraph_view
            The subgraph of nodes with generation K.
        """
        return nx.subgraph_view(self.G, filter_node=lambda x: self._generation_dict[x] == K)

    def generations(self, gens):
        """
        Returns the subgraph of nodes with generations in the given iterable.

        Parameters
        ----------
        gens : iterable
            An iterable containing generations.

        Returns
        -------
        nx.subgraph_view
            The subgraph of nodes with generations in the given iterable.
        """
        return nx.subgraph_view(self.G, filter_node=lambda x: self._generation_dict[x] in gens)

    def get_current_generation(self):
        """
        Returns the subgraph of nodes in the current generation.

        Parameters
        ----------
        K : int
            The generation number.

        Returns
        -------
        nx.subgraph_view
            The subgraph of nodes in the current generation.
        """
        return self.generation(self.current_generation)

    def get_most_recent_K_generations(self, K):
        """
        Returns the subgraph of nodes in the most recent K generations.

        Returns
        -------
        nx.subgraph_view
            The subgraph of nodes in the most recent K generations.
        """
        if K > self.generational_depth:
            raise ValueError('K exceeds generational depth of pedigree')
        return self.generations(range(self.current_generation, self.current_generation - K, -1))

    # def _add_edges_from_arrays(self, x, y):
    #     """
    #     Adds edges from arrays x and y.

    #     Parameters
    #     ----------
    #     x : array-like
    #         The source nodes for the edges.
    #     y : array-like
    #         The target nodes for the edges.
    #     """
    #     self._add_nodes(x)
    #     self.G.add_edges_from([(xx, yy) for (xx, yy) in zip(x, y)])

    def _add_edges_from_indexes(self, x, y):
        """
        Adds edges from arrays x and y.

        Parameters
        ----------
        x : xft.index.SampleIndex
            The sample index objections corresponing to source nodes
        y : xft.index.SampleIndex
            The sample index objections corresponing to target nodes
        """
        self._add_nodes(x)
        self._add_nodes(y)
        self.G.add_edges_from([(xx, yy) for (xx, yy) in zip(x.unique_identifier, y.unique_identifier)])

    def _add_offspring(self,
                      mating: xft.mate.MateAssignment,
                      ):
        """
        Adds offspring nodes and edges to the pedigree based on a MateAssignment object.

        Parameters
        ----------
        mating : xft.mate.MateAssignment
            The MateAssignment object containing mating information.
        """
        self._add_edges_from_indexes(
            mating.reproducing_maternal_index, mating.offspring_sample_index)
        self._add_edges_from_indexes(
            mating.reproducing_paternal_index, mating.offspring_sample_index)

    def _get_trios(self):
        """
        TODO: Implement this method.
        """
        raise NotImplementedError


# NN=100
# test_iids = np.array(['0_' + str(i) for i in range(NN)])
# ped = Pedigree(test_iids)

# mothers = np.random.permutation(np.repeat(test_iids[0::2],2))
# fathers = np.random.permutation(np.repeat(test_iids[1::2],2))
# offspring = np.array(['1_' + str(i) for i in range(NN)])
# mating = MateAssignment(0, mothers, fathers, offspring)
# ped.add_offspring(mating)


# mothers = np.random.permutation(np.repeat(offspring[0::2],2))
# fathers = np.random.permutation(np.repeat(offspring[1::2],2))
# offspring = np.array(['2_' + str(i) for i in range(NN)])
# mating = MateAssignment(0, mothers, fathers, offspring)
# ped.add_offspring(mating)
