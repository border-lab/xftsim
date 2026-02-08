"""
Module to define classes for post-processing xft simulation data.
Classes:

PostProcessor:
Base class for defining post-processing operations on xft simulation data.

LimitMemory(PostProcessor):
Class to limit the amount of memory used by the simulation by deleting old haplotype and/or phenotype data.

WriteToDisk(PostProcessor):
Class to write simulation data to disk.
"""

import xftsim as xft
import warnings
import functools
import numpy as np
import pandas as pd
import nptyping as npt
import xarray as xr
from nptyping import NDArray, Int8, Int64, Float64, Bool, Shape
from typing import Any, Hashable, List, Iterable, Callable, Union, Dict, final
from numpy.typing import ArrayLike
from functools import cached_property

import numpy as np
import pandas as pd


class PostProcessor:
    """
    Base class for defining post-processing operations on XFT simulation data.
    Parameters:
    -----------
    processor: Callable
        A callable object that takes a single argument of type xft.sim.Simulation and performs some post-processing operation on it.
    name: str
        A name for the post-processing operation being defined.

    Methods:
    --------
    process(sim: xft.sim.Simulation) -> None:
        Applies the post-processing operation to the given simulation.
    """
    def __init__(self,
                 processor: Callable,
                 name: str,
                 ):
        self.name = name
        self.processor = processor

    def process(self,
                sim: xft.sim.Simulation) -> None:
        """
        Applies the post-processing operation to the given simulation.
        
        Parameters:
        -----------
        sim: xft.sim.Simulation
            The simulation to apply the post-processing operation to.
        
        Returns:
        --------
        None
        """
        self.processor(sim)


class LimitMemory(PostProcessor):
    """
    Class to limit the amount of memory used by the simulation by deleting old haplotype and/or phenotype data.
    Parameters:
    -----------
    n_haplotype_generations: int, optional
        The number of haplotype generations to keep. If -1, keep all generations. Default is -1.
    n_phenotype_generations: int, optional
        The number of phenotype generations to keep. If -1, keep all generations. Default is -1.

    Methods:
    --------
    processor(sim: xft.sim.Simulation) -> None:
        Deletes old haplotype and/or phenotype data from the simulation.
    """
    def __init__(self,
                 n_haplotype_generations: int = -1,
                 n_phenotype_generations: int = -1,
                 ):
        assert (n_haplotype_generations == -1) | (n_haplotype_generations >=
                                                  1), "valid inputs are -1 (keep all generations) OR n>=1 (keep n most recent generations)"
        assert (n_phenotype_generations == -1) | (n_phenotype_generations >=
                                                  1), "valid inputs are -1 (keep all generations) OR n>=1 (keep n most recent generations)"
        self.n_haplotype_generations = n_haplotype_generations
        self.n_phenotype_generations = n_phenotype_generations

    def processor(self,
                  sim: xft.sim.Simulation) -> None:
        """
        Deletes old haplotype and/or phenotype data from the simulation.
        
        Parameters:
        -----------
        sim: xft.sim.Simulation
            The simulation to delete old data from.
        
        Returns:
        --------
        None
        """
        # delete all but last
        if self.n_haplotype_generations >= 1:
            for k in range(0, sim.generation - (self.n_haplotype_generations - 1)):
                if k >= 0:
                    sim.haplotype_store[k] = None
        if self.n_phenotype_generations >= 1:
            for k in range(0, sim.generation - (self.n_phenotype_generations - 1)):
                if k >= 0:
                    sim.phenotype_store[k] = None
        return None


class WriteToDisk(PostProcessor):
    """docstring for PostProcess"""

    def __init__(self, arg):
        self.arg = arg
