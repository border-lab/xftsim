"""
New simulation loop for the refactored xftsim.

NSimulation: forward-time genetics simulation using the new data structures,
architecture DAG, and mate assignment system.
"""
import numpy as np

from xftsim.struct import (
    HaplotypeOperator, NPhenotypeArray, PedigreeArray,
)
from xftsim.narch import Architecture
from xftsim.nmate import NMateAssignment, RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.nstats import GenerationResult


class NSimulation:
    """
    Forward-time genetics simulation.

    Parameters
    ----------
    founder_haplotypes : HaplotypeOperator
        Generation-0 haplotypes.
    architecture : Architecture
        Phenogenetic architecture (DAG of ArchNodes).
    mating_regime : RandomMating
        Mating strategy that produces NMateAssignment.
    recombination_map : RecombinationMap
        Recombination probabilities for meiosis.
    retain_haplotypes : int
        How many past generations of haplotypes to keep. Default 1.
    retain_phenotypes : int
        How many past generations of phenotypes to keep. Default 2.
    callbacks : list[callable], optional
        Functions called after each generation with ``callback(sim)``.
        Set ``sim.stop = True`` inside a callback for early stopping.
    filters : dict[str, Filter], optional
        Named filters to run after each generation's phenotype computation.
    statistics : list[Statistic], optional
        Statistics to compute after each generation.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        founder_haplotypes: HaplotypeOperator,
        architecture: Architecture,
        mating_regime,
        recombination_map: RecombinationMap,
        retain_haplotypes: int = 1,
        retain_phenotypes: int = 2,
        callbacks=None,
        filters=None,
        statistics=None,
        seed=None,
    ):
        self.architecture = architecture
        self.mating_regime = mating_regime
        self.recombination_map = recombination_map
        self.retain_haplotypes = retain_haplotypes
        self.retain_phenotypes = retain_phenotypes
        self.callbacks = callbacks or []
        self.filters = filters or {}
        self.statistics = statistics or []
        self.rng = np.random.RandomState(seed)
        self.stop = False

        # History dicts keyed by generation
        self.haplotype_history: dict[int, HaplotypeOperator] = {
            0: founder_haplotypes,
        }
        self.phenotype_history: dict[int, NPhenotypeArray] = {}
        self.pedigree_history: dict[int, PedigreeArray] = {}
        self._mate_assignments: dict[int, NMateAssignment] = {}

        # Results from statistics
        self.results: list[GenerationResult] = []

        self.generation = 0

    @property
    def haplotypes(self) -> HaplotypeOperator:
        """Current generation's haplotypes."""
        return self.haplotype_history[self.generation]

    @property
    def phenotypes(self) -> NPhenotypeArray:
        """Current generation's phenotypes."""
        return self.phenotype_history[self.generation]

    def run(self, n_generations: int):
        """
        Run the simulation for n_generations.

        Generation 0: compute phenotypes from founder haplotypes, assign mates.
        Generation t>0: meiosis -> compute phenotypes -> assign mates.

        Parameters
        ----------
        n_generations : int
            Number of generations to simulate (including gen 0).
        """
        # --- Generation 0: founders ---
        hap = self.haplotype_history[0]
        pheno = self.architecture.compute(
            hap, rng=self.rng,
            phenotype_history=self.phenotype_history,
            pedigree_history=self.pedigree_history,
            generation=0,
        )
        self.phenotype_history[0] = pheno

        self._run_filters_and_stats(0)

        if n_generations > 1:
            assignment = self.mating_regime.mate(
                hap.samples, rng=self.rng,
                phenotypes=self.phenotype_history.get(0),
            )
            self._mate_assignments[0] = assignment

        self._run_callbacks()
        if self.stop:
            return

        # --- Generations 1..n_generations-1 ---
        for gen in range(1, n_generations):
            prev_assignment = self._mate_assignments[gen - 1]
            prev_hap = self.haplotype_history[gen - 1]

            # Meiosis: produce offspring haplotypes
            offspring_hap = prev_hap.meiosis(
                prev_assignment, self.recombination_map
            )
            self.haplotype_history[gen] = offspring_hap
            self.generation = gen

            # Build PedigreeArray
            ped = PedigreeArray(
                offspring_samples=prev_assignment.offspring_samples,
                maternal_idx=prev_assignment.maternal_idx,
                paternal_idx=prev_assignment.paternal_idx,
                parent_n=prev_hap.n,
            )
            self.pedigree_history[gen] = ped

            # Compute phenotypes
            pheno = self.architecture.compute(
                offspring_hap, rng=self.rng,
                phenotype_history=self.phenotype_history,
                pedigree_history=self.pedigree_history,
                generation=gen,
            )
            self.phenotype_history[gen] = pheno

            self._run_filters_and_stats(gen)

            # Assign mates for next generation (unless this is the last gen)
            if gen < n_generations - 1:
                assignment = self.mating_regime.mate(
                    offspring_hap.samples, rng=self.rng,
                    phenotypes=self.phenotype_history.get(gen),
                )
                self._mate_assignments[gen] = assignment

            # Enforce retention policy
            self._enforce_retention(gen)

            self._run_callbacks()
            if self.stop:
                return

    def _run_filters_and_stats(self, gen: int):
        """Run filters and statistics for the given generation."""
        # Run filters
        filtered_views = {}
        for name, filt in self.filters.items():
            view = filt.apply(gen, self.phenotype_history, self.pedigree_history)
            if view is not None:
                filtered_views[name] = view

        # Run statistics
        if self.statistics:
            stats = {}
            name_counts = {}
            for stat in self.statistics:
                result = stat.estimate(
                    self.phenotype_history, filtered_views, gen
                )
                base_key = type(stat).__name__
                count = name_counts.get(base_key, 0)
                name_counts[base_key] = count + 1
                key = base_key if count == 0 else f"{base_key}_{count}"
                stats[key] = result
            self.results.append(GenerationResult(generation=gen, statistics=stats))

    def _enforce_retention(self, current_gen: int):
        """Drop old generations from history dicts per retention policy."""
        # Haplotypes
        for g in list(self.haplotype_history.keys()):
            if g < current_gen - self.retain_haplotypes:
                del self.haplotype_history[g]

        # Phenotypes
        for g in list(self.phenotype_history.keys()):
            if g < current_gen - self.retain_phenotypes:
                del self.phenotype_history[g]

        # Pedigrees: keep same as phenotypes
        for g in list(self.pedigree_history.keys()):
            if g < current_gen - self.retain_phenotypes:
                del self.pedigree_history[g]

        # Mate assignments: only need the most recent
        for g in list(self._mate_assignments.keys()):
            if g < current_gen - 1:
                del self._mate_assignments[g]

    def _run_callbacks(self):
        """Execute all registered callbacks."""
        for cb in self.callbacks:
            cb(self)

    def __repr__(self):
        return (f"NSimulation(generation={self.generation}, "
                f"n={self.haplotypes.n}, m={self.haplotypes.m})")
