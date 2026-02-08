import xftsim as xft


import warnings
import pickle
import functools
import numpy as np
import pandas as pd
import nptyping as npt
import xarray as xr
from nptyping import NDArray, Int8, Int64, Float64, Bool, Shape
from typing import Any, Hashable, List, Iterable, Callable, Union, Dict, final
from numpy.typing import ArrayLike
from functools import cached_property
import funcy



class Simulation():
    """
    A class for running an xft simulation.

    Attributes
    ----------
    mating_regime : xft.mate.MatingRegime
        Mating regime.
    recombination_map : xft.reproduce.RecombinationMap
        Recombination map.
    architecture : xft.arch.Architecture
        Phenogenetic architecture.
    statistics : Iterable, optional
        Iterable of statistics to compute each generation, by default empty list.
    post_processors : Iterable, optional
        Iterable of post processors to apply each generation, by default empty list.
    generation : int, optional
        Initial generation, by default -1, corresponding to an uninitialized simulation
    control : Dict, optional
        Control parameters for the simulation, by default an empty dictionary.
    reproduction_method : xft.reproduce.ReproductionMethod, optional
        Reproduction method for the simulation, by default xft.reproduce.Meiosis.
    control : dict
        Control parameters for the simulation
    haplotypes : xr.DataArray
        Haplotypes for the current generation.
    phenotypes : xr.DataArray
        Phenotypes for the current generation.
    mating : xr.DataArray
        Mating information for the current generation.
    parent_mating : xr.DataArray
        Mating information for the previous generation.
    parent_haplotypes : xr.DataArray
        Haplotypes for the previous generation.
    parent_phenotypes : xr.DataArray
        Phenotypes for the previous generation.
    results : xr.DataArray
        Results for the current generation.
    current_afs_empirical : xr.DataArray
        Current empirical allele frequencies.
    current_std_genotypes : xr.DataArray
        Current standardized genotypes.
    current_std_phenotypes : xr.DataArray
        Current standardized phenotypes.
    phenotype_store : Dict[int, xr.DataArray]
        Dictionary storing phenotypes for each generation.
    haplotype_store : Dict[int, xr.DataArray]
        Dictionary storing haplotypes for each generation.
    mating_store : Dict[int, xr.DataArray]
        Dictionary storing mating information for each generation.
    results_store : Dict[int, xr.DataArray]
        Dictionary storing results for each generation.
    pedigree : Any
        Pedigree information for the simulation (currently not implemented).
    metadata : Dict
        Dictionary containing user specified metadata

    Methods
    -------
    run(n_generations: int):
        Run the simulation for a specified number of generations.
    run_generation():
        Run a single generation of the simulation.
    compute_phenotypes():
        Compute phenotypes for the current generation.
    mate():
        Perform mating for the current generation.
    reproduce():
        Perform reproduction for the current generation.
    estimate_statistics():
        Estimate statistics for the current generation.
    process():
        Process the current generation using post-processors.
    update_pedigree():
        Update pedigree information for the current generation.
    increment_generation():
        Increment the current generation.
    move_forward(n_generations: int):
        Move the simulation forward by a specified number of generations.
    """
    def __init__(self,
                 founder_haplotypes: xr.DataArray,
                 mating_regime: xft.mate.MatingRegime,
                 recombination_map: xft.reproduce.RecombinationMap,
                 architecture: xft.arch.Architecture,
                 statistics: Iterable = [],
                 post_processors: Iterable = [],
                 generation: int = -1,
                 control={},
                 reproduction_method=xft.reproduce.Meiosis,
                 metadata={},
                 filter_sample: bool = False,
                 sample_filter: xft.index.SampleFilter = None,
                 ):
        # attributes
        self.mating_regime = mating_regime
        self.recombination_map = recombination_map
        self.architecture = architecture
        self.statistics = statistics
        self.post_processors = post_processors
        self.reproduction_method = reproduction_method
        self.reproduction_regime = self.reproduction_method(
            self.recombination_map)
        self.metadata = metadata
        self.filter_sample = filter_sample
        self.sample_filter = sample_filter
        # default control parameters:
        ctrl = Simulation._default_control()
        ctrl.update(control)
        self.control = ctrl
        # properties
        self._generation = generation
        # initialize data stores
        # for proper initialization
        self.haplotype_store = {np.max([generation, 0]): founder_haplotypes}
        self.phenotype_store = dict()
        self.mating_store = dict()
        self.results_store = dict()
        self.pedigree = xft.ped.Pedigree(founder_haplotypes.samples)

        #### generation specific cached properties ####
        # computed once per generation, deleted next generation
        # by call to increment_generation
        self._current_afs_empirical = None
        self._current_std_genotypes = None
        self._current_std_phenotypes = None
        self._current_std_genotypes_filtered = None
        self._current_std_phenotypes_filtered = None

    @property
    def control(self):
        return self._control

    @control.setter
    def control(self, value):
        self._control = value
        self._validate_control()

    @staticmethod
    def _default_control():
        return dict(
            standardization='hardy_weinberg_empirical',
        )

    def _validate_control(self):
        if self._control['standardization'] not in [
            'hardy_weinberg_empirical',
            # 'hardy_weinberg_reference',
        ]:
            raise RuntimeError()

    @xft.utils.profiled(level=0)
    def run(self, n_generations: int):
        """
        Run the simulation for a specified number of generations.

        Parameters
        ----------
        n_generations : int
            Number of generations to run the simulation.
        """
        for generation in range(n_generations):
            self.run_generation()

    @xft.utils.profiled(level=0)
    def run_generation(self):
        """
        Run a single generation of the simulation.
        """
        self.increment_generation()
        self.reproduce()
        self.compute_phenotypes()
        self.mate()
        self.update_pedigree()
        self.apply_filter()
        self.estimate_statistics()
        self.process()

    @xft.utils.profiled()
    def compute_phenotypes(self):
        """
        Compute phenotypes for the current generation.
        """
        # generation zero
        if self.generation == 0:
            self.phenotypes = self.architecture.initialize_founder_phenotype_array(
                self.haplotypes, self.control)
            self.architecture.compute_phenotypes(
                self.haplotypes, self.phenotypes, self.control)
        elif self.generation >= 1:
            self.phenotypes = self.architecture.initialize_phenotype_array(
                self.haplotypes, self.control)
            xft.reproduce.transmit_parental_phenotypes(self.parent_mating,
                                                       self.parent_phenotypes,
                                                       self.phenotypes,
                                                       self.control)
            self.architecture.compute_phenotypes(
                self.haplotypes, self.phenotypes, self.control)

    @xft.utils.profiled()
    def mate(self):
        """
        Perform mating for the current generation.
        """
        self.mating = self.mating_regime.mate(
            self.haplotypes, self.phenotypes, self.control)

    @xft.utils.profiled()
    def reproduce(self):
        """
        Perform reproduction for the current generation.
        """
        if self.generation == 0:
            pass
        elif self.generation >= 1:
            self.haplotypes = self.reproduction_regime.reproduce(self.parent_haplotypes,
                                                                 self.parent_mating,
                                                                 self.control)

    @xft.utils.profiled()
    def estimate_statistics(self):
        """
        Estimate statistics for the current generation.
        """
        for stat in self.statistics:
            stat.estimate(self)

    def apply_filter(self):
        """
        Apply sample filters to the current generation
        """
        if self.sample_filter is None:
            pass
        else:
            self.filter_indices = self.sample_filter.filter(self.phenotypes.xft.get_sample_indexer())

    @xft.utils.profiled()
    def process(self):
        """
        Apply post-processors to the current generation.
        """
        for ind, proc in enumerate(self.post_processors):
            # allow callables that map xft.Simulation -> None
            if isinstance(proc, Callable):
                proc = xft.proc.PostProcessor(
                    proc, ' '.join(['process', str(ind)]))
            proc.process(self)

    @xft.utils.profiled()
    def update_pedigree(self):
        """
        Update pedigree information (NOT IMPLEMENTED).
        """
        # warnings.warn('update_pedigree() not implemented')
        self.pedigree._add_offspring(self.mating)

    # generation is immutable except through Simulation().increment_generation()
    @property
    def generation(self):
        return self._generation

    # erase current-generation specific cached properties
    @xft.utils.profiled()
    def increment_generation(self):
        self._current_afs_empirical = None
        self._current_std_genotypes = None
        self._current_std_haplotypes = None
        self._generation += 1

    @property
    def haplotypes(self):
        if self.generation in self.haplotype_store.keys():
            return self.haplotype_store[self.generation]
        else:
            return None

    @property
    def haplotypes_filtered(self):
        if self.filter_sample:
            return self.haplotypes[self.filter_indices,:]
        else:
            return self.haplotypes

    @haplotypes.setter
    def haplotypes(self, value):
        self.haplotype_store[self.generation] = value

    @property
    def phenotypes(self):
        if self.generation in self.phenotype_store.keys():
            return self.phenotype_store[self.generation]
        else:
            return None

    @phenotypes.setter
    def phenotypes(self, value):
        self.phenotype_store[self.generation] = value

    @property
    def phenotypes_filtered(self):
        if self.filter_sample:
            return self.phenotypes[self.filter_indices,:]
        else:
            return self.phenotypes


    @property
    def mating(self):
        if self.generation in self.mating_store.keys():
            return self.mating_store[self.generation]
        else:
            return None

    @mating.setter
    def mating(self, value):
        self.mating_store[self.generation] = value

    @property
    def parent_mating(self):
        if (self.generation - 1) in self.mating_store.keys():
            return self.mating_store[(self.generation - 1)]
        else:
            return None

    @property
    def parent_haplotypes(self):
        if (self.generation - 1) in self.haplotype_store.keys():
            return self.haplotype_store[(self.generation - 1)]
        else:
            return None

    @property
    def parent_phenotypes(self):
        if (self.generation - 1) in self.phenotype_store.keys():
            return self.phenotype_store[(self.generation - 1)]
        else:
            return None

    def move_forward(self, n_generations):
        pass

    @property
    def results(self):
        if self.generation in self.results_store.keys():
            return self.results_store[self.generation]
        else:
            return None

    # generation specific properties that are overwritten
    # by increment_generation()
    @property
    def current_afs_empirical(self):
        if self._current_afs_empirical is None:
            return self.haplotypes.xft.af_empirical
        else:
            return self._current_afs_empirical

    @property
    def current_std_genotypes(self):
        if self._current_std_genotypes is None:
            if self.control['standardization'] == 'hardy_weinberg_empirical':
                return self.haplotypes.xft.to_diploid_standardized(af=None,
                                                                   scale=False)
            else:
                raise NotImplementedError()
        else:
            return self._current_std_genotypes

    @property
    def current_std_genotypes_filtered(self):
        if self.filter_sample is False:
            return self.current_std_genotypes
        elif self._current_std_genotypes_filtered is None:
            if self.control['standardization'] == 'hardy_weinberg_empirical':
                return self.haplotypes[self.filter_indices,:].xft.to_diploid_standardized(af=None,
                                                                   scale=False)
            else:
                raise NotImplementedError()
        else:
            return self._current_std_genotypes_filtered

    @property
    def current_std_phenotypes(self):
        if self._current_std_phenotypes is None:
            return self.phenotypes.xft.standardize()
        else:
            return self._current_std_phenotypes

    @property
    def current_std_phenotypes_filtered(self):
        if self.filter_sample is False:
            return self.current_std_phenotypes
        elif self._current_std_phenotypes_filtered is None:
            return self.phenotypes[self.filter_indices,:].xft.standardize()
        else:
            return self._current_std_phenotypes_filtered

    @property
    def dependency_graph_edges(self):
        edges = []
        colors = []
        edge_labels = []
        for i,component in enumerate(self.architecture.components):
            edges += component.dependency_graph_edges
            colors += [i+1 for j in range(len(component.dependency_graph_edges))]
            edge_labels += [component._component_name for j in range(len(component.dependency_graph_edges))]
        if 'proband\nhaplotypes' in [edge[0] for edge in edges]:
            edges += [('maternal\nhaplotypes','proband\nhaplotypes'),
                      ('paternal\nhaplotypes','proband\nhaplotypes')]
            colors += [0, 0]
            edge_labels += ['meiosis', 'meiosis']
        ## hack to ensure noise nodes (labelled #\d+) have unique blank labels
        import re
        noise_nodes = [x[0] for x in [re.findall('^#\\d+$', node,) for node in [edge[0] for edge in edges]] if x != []]
        noise_dict = {x:' '*(i+1) for i,x in enumerate(np.unique(noise_nodes))}
        for i in range(len(edges)):
            edge = edges[i]
            if edge[0] in noise_nodes:
                edges[i] = (noise_dict[edge[0]],edge[1])
        ## add mating
        edges += self.mating_regime.dependency_graph_edges
        colors += [np.max(colors)+1 for edge in self.mating_regime.dependency_graph_edges]
        edge_labels += ['mating' for edge in self.mating_regime.dependency_graph_edges]
        return (edges,np.array(colors), edge_labels)

    @property
    def dependency_graph(self):
        import networkx as nx
        G = nx.DiGraph()
        edges, colors, edge_labels = self.dependency_graph_edges
        G.add_edges_from(edges)
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
        return (G,pos,colors,edges,edge_labels)

    def draw_dependency_graph(self, 
                              node_color='none', 
                              node_size = 1200, 
                              font_size=5, 
                              margins=.1,
                              edge_color="#222222", 
                              arrowsize = 6,
                              number_edges: bool = True,
                              **kwargs):
        import networkx as nx
        G,pos,colors,edges,edge_labels = self.dependency_graph
        if number_edges:
            edge_labels = xft.utils.paste([colors.astype(str),edge_labels],sep=':')
        color_dict = {a:b for a,b in zip(edges, colors)}
        label_dict = {a:b for a,b in zip(edges, edge_labels)}
        new_colors = [color_dict[edge] for edge in G.edges]
        # new_labels = [label_dict[edge] for edge in G.edges]
        # labels = {key:str(value) for (key,value) in zip(G.edges, colors)}
        nx.draw_networkx(G,pos, 
                         node_color=node_color, 
                         node_size = node_size, 
                         font_size=font_size,
                         margins=margins,
                         edge_color=new_colors,
                         arrowsize=arrowsize,
                         **kwargs)
        nx.draw_networkx_edge_labels(G,pos, 
                 font_size=font_size//1.3,
                 edge_labels=label_dict,
                 rotate=False,
                 bbox={'boxstyle':'round', 'ec':'none', 'fc':(1.0, 1.0, 1.0)},
                 # font_color=colors/np.max(colors),
                 )

    @xft.utils.profiled()
    def pickle_results(self, 
                       path: str, 
                       metadata: dict = dict(),
                       results_store: bool = True,
                       architecture: bool = True,
                       mating_store: bool = True,
                       phenotype_store: bool = True,
                       mating_regime: bool = False,
                       haplotype_store: bool = False,
                       ):
        _metadata = self.metadata
        _metadata.update(metadata); 
        _architecture = None; _mating_store = None; _phenotype_store = None; 
        _mating_regime = None; _haplotype_store = None; _results_store = None;
        if results_store:
            _results_store = self.results_store
        if architecture:
            _architecture = self.architecture.components[0].effects
        if mating_store:
            _mating_store = self.mating_store
        if phenotype_store:
            _phenotype_store = self.phenotype_store
        if mating_regime:
            _mating_regime = self.mating_regime
        if haplotype_store:
            _haplotype_store = self.haplotype_store
        to_pickle=dict(metadata=_metadata,
                       results_store=_results_store,
                       architecture=_architecture,
                       mating_store=_mating_store,
                       phenotype_store=_phenotype_store,
                       mating_regime=_mating_regime,
                       haplotype_store=_haplotype_store)
        with open(path+'.pickle', 'wb') as f:
            pickle.dump(obj=to_pickle, file=f)

class DemoSimulation:
    """Demo simulation using the new NSimulation system."""
    demo_routines = dict(UGRM='Univariate GCTA with balanced random mating demo',
                         BGRM='Bivariate GCTA with balanced random mating demo')
    _demo_long = dict(BGRM='Two phenotypes, height and bone mineral density (BMD)\n'
                            'assuming bivariate GCTA infinitesimal architecture\n'
                            'with h2 values set to 0.5 and 0.4 for height and BMD\n'
                            'respectively and a genetic effect correlation of 0.0.',
                      UGRM='A single phenotype, height, assuming univariate GCTA\n'
                            'infinitesimal architecture with h2 = 0.5.')

    def __init__(self,
                 routine: str = 'BGRM',
                 n: int = 2000,
                 m: int = 400) -> None:
        if routine not in self.demo_routines:
            raise NotImplementedError(
                f"Routine `{routine}` not implemented. "
                f"See `DemoSimulation.demo_routines` for supported routines"
            )

        founder_haplotypes = xft.founders.founder_haplotypes_uniform_AFs(n=n, m=m)
        recombination_map = xft.reproduce.RecombinationMap.from_haplotypes(
            founder_haplotypes, p=.1)
        mating_regime = xft.nmate.RandomMating(offspring_per_pair=2)

        if routine == 'UGRM':
            effects = {'eff': xft.neffect.AdditiveEffects.from_h2(h2=0.5, m=m)}
            architecture = xft.narch.Architecture(
                formula="height.G ~ genetic(eff)\nheight.E ~ noise(0.5)\nheight ~ height.G + height.E",
                effects=effects,
            )
        elif routine == 'BGRM':
            effects = {'eff': xft.neffect.MultivariateEffects.from_h2_rg(
                h2=np.array([0.5, 0.4]), rg=np.eye(2), m=m)}
            architecture = xft.narch.Architecture(
                formula=(
                    "(height.G, BMD.G) ~ mvGenetic(eff)\n"
                    "height.E ~ noise(0.5)\n"
                    "BMD.E ~ noise(0.6)\n"
                    "height ~ height.G + height.E\n"
                    "BMD ~ BMD.G + BMD.E"
                ),
                effects=effects,
            )

        self._sim = xft.nsim.NSimulation(
            founder_haplotypes=founder_haplotypes,
            architecture=architecture,
            mating_regime=mating_regime,
            recombination_map=recombination_map,
            statistics=[xft.nstats.SampleStatistics()],
            seed=42,
        )
        self._sim.run(2)

        self.routine = routine
        self.routine_title = self.demo_routines[routine]
        self.routine_long = self._demo_long[routine]
        self._n = n
        self._m = m

    @property
    def generation(self):
        return self._sim.generation

    @property
    def haplotypes(self):
        return self._sim.haplotypes

    @property
    def phenotypes(self):
        return self._sim.phenotypes

    @property
    def results(self):
        return self._sim.results

    def run(self, n_generations: int):
        self._sim.continue_run(n_generations)

    def __repr__(self):
        return '\n'.join(["<DemoSimulation>",
                         self.routine_title,
                         f"n = {self._n}; m = {self._m}",
                         self.routine_long,
                         ])



