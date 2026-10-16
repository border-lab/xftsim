"""
New simulation loop for the refactored xftsim.

Simulation: forward-time genetics simulation using the new data structures,
architecture DAG, and mate assignment system.
"""
from __future__ import annotations

import copy
import dataclasses
import warnings
from typing import Callable

import numpy as np

from xftsim import inherit
from xftsim.inherit import InheritanceRule
from xftsim.struct import (
    DenseHaplotypeArray, HaplotypeOperator, PhenotypeArray, PedigreeArray,
    SampleMeta,
)
from xftsim.arch import Architecture
from xftsim.mate import MateAssignment, RandomMating
from xftsim.reproduce import RecombinationMap
from xftsim.filters import Filter, FilteredView
from xftsim.stats import GenerationResult, Statistic


_STANDARDIZE_WITH = ('founders', 'current')


class Simulation:
    """Forward-time genetics simulation.

    Orchestrates the simulation loop: for each generation, performs meiosis
    (recombination), computes phenotypes via the architecture DAG, applies
    filters, computes statistics, and runs callbacks. History dicts are
    pruned according to configurable retention policies.

    Parameters
    ----------
    founder_haplotypes : HaplotypeOperator
        Generation-0 haplotypes (DenseHaplotypeArray or GraphHaplotypeOperator).
    architecture : Architecture
        Phenogenetic architecture (DAG of ArchNodes).
    mating_regime : RandomMating or LinearAssortativeMating
        Mating strategy that produces MateAssignment each generation.
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
    metadata_rules : list[InheritanceRule], optional
        Rules from ``xftsim.inherit`` that carry per-individual metadata
        (``SampleMeta.extra``, e.g. population or ancestry) from parents to
        offspring each generation. If None (default), metadata is not
        inherited. Every metadata field used after generation 0 by the
        architecture (``| field`` grouping, ``meta(field)``), the mating
        regime, filters or statistics must be produced by a rule; otherwise
        ``run()`` raises before generation 0.
    standardize_with : {'founders', 'current'}
        Allele frequencies used to standardize genotypes for genetic effects
        on the standardized scale. ``'founders'`` (default) uses the
        founders' empirical frequencies in every generation, so each
        allele's effect stays fixed and genetic values change only through
        genotypes, as in xftsim 0.3. ``'current'`` re-standardizes with each
        generation's own frequencies, which rescales every allele's effect
        as its frequency drifts (the v0.9 behavior before this option).
        Generation 0 is identical either way. Estimators such as HE always
        standardize with the current sample.

    Attributes
    ----------
    generation : int
        Current generation number.
    haplotype_history : dict[int, HaplotypeOperator]
        Generation -> haplotype data mapping (pruned by retention policy).
    phenotype_history : dict[int, PhenotypeArray]
        Generation -> phenotype data mapping (pruned by retention policy).
    pedigree_history : dict[int, PedigreeArray]
        Generation -> pedigree mapping (pruned by retention policy).
    results : list[GenerationResult]
        Statistics results for each completed generation.
    stop : bool
        Set to True inside a callback to halt the simulation.
    founder_af : np.ndarray or None
        Empirical allele frequencies of generation 0, set by ``run()``.

    Examples
    --------
    >>> from xftsim.founders import founder_haplotypes_uniform_AFs
    >>> from xftsim.effect import AdditiveEffects
    >>> from xftsim.arch import Architecture, GeneticComponent, NoiseComponent
    >>> from xftsim.mate import RandomMating
    >>> from xftsim.reproduce import RecombinationMap
    >>> import numpy as np
    >>> hap = founder_haplotypes_uniform_AFs(n=100, m=50)
    >>> eff = AdditiveEffects.from_h2(h2=0.5, m=50, seed=42)
    >>> arch = Architecture()
    >>> arch.add('Y.G', GeneticComponent(eff))
    >>> arch.add('Y.E', NoiseComponent(0.5))
    >>> arch.add('Y', AggregationComponent('Y.G + Y.E'))
    >>> rmap = RecombinationMap.uniform(m=50, p=0.01)
    >>> sim = Simulation(hap, arch, RandomMating(), rmap, seed=1)
    >>> sim.run(n_generations=3)
    >>> sim.generation
    2
    """

    def __init__(
        self,
        founder_haplotypes: HaplotypeOperator,
        architecture: Architecture,
        mating_regime: RandomMating,
        recombination_map: RecombinationMap,
        retain_haplotypes: int = 1,
        retain_phenotypes: int = 2,
        callbacks: list[Callable[[Simulation], None]] | None = None,
        filters: dict[str, Filter] | None = None,
        statistics: list[Statistic] | None = None,
        seed: int | None = None,
        metadata_rules: list[InheritanceRule] | None = None,
        standardize_with: str = 'founders',
    ) -> None:
        if standardize_with not in _STANDARDIZE_WITH:
            raise ValueError(
                f"standardize_with must be one of {_STANDARDIZE_WITH}, "
                f"got {standardize_with!r}"
            )
        self.architecture: Architecture = architecture
        self.mating_regime: RandomMating = mating_regime
        self.recombination_map: RecombinationMap = recombination_map
        self.retain_haplotypes: int = retain_haplotypes
        self.retain_phenotypes: int = retain_phenotypes
        self.callbacks: list[Callable[[Simulation], None]] = callbacks or []
        self.filters: dict[str, Filter] = filters or {}
        self.statistics: list[Statistic] = statistics or []
        self.rng: np.random.RandomState = np.random.RandomState(seed)
        self.stop: bool = False
        self.metadata_rules: list[InheritanceRule] | None = (
            list(metadata_rules) if metadata_rules is not None else None
        )
        self._warned_dropped_metadata: set[str] = set()
        self.standardize_with: str = standardize_with
        self.founder_af: np.ndarray | None = None

        # History dicts keyed by generation
        self.haplotype_history: dict[int, HaplotypeOperator] = {
            0: founder_haplotypes,
        }
        self.phenotype_history: dict[int, PhenotypeArray] = {}
        self.pedigree_history: dict[int, PedigreeArray] = {}
        self._mate_assignments: dict[int, MateAssignment] = {}

        # Results from statistics
        self.results: list[GenerationResult] = []

        self.generation: int = 0

    @classmethod
    def from_checkpoint(cls, dir_path: str,
                        mating_regime: RandomMating | None = None,
                        recombination_map: RecombinationMap | None = None,
                        callbacks: list[Callable[[Simulation], None]] | None = None,
                        filters: dict[str, Filter] | None = None,
                        statistics: list[Statistic] | None = None,
                        metadata_rules: list[InheritanceRule] | None = None,
                        standardize_with: str | None = None,
                        ) -> Simulation:
        """
        Reconstruct a simulation from a checkpoint directory.

        Parameters
        ----------
        dir_path : str
            Path to checkpoint directory (created by save_simulation_checkpoint).
        mating_regime : optional
            Override the saved mating regime. If None, uses the saved one.
        recombination_map : RecombinationMap, optional
            Override the saved recombination map. If None, uses the saved one.
        callbacks : list[callable], optional
            Callbacks for continued execution.
        filters : dict, optional
            Filters for continued execution.
        statistics : list, optional
            Statistics for continued execution.
        metadata_rules : list[InheritanceRule], optional
            Override the saved metadata inheritance rules. Required if the
            checkpoint used a ``Custom`` rule.
        standardize_with : {'founders', 'current'}, optional
            Override the saved standardization mode. Checkpoints written
            before this option existed were simulated with ``'current'`` and
            resume that way (with a warning) unless overridden.

        Returns
        -------
        Simulation
            A simulation ready for continued execution via run().
        """
        from xftsim.io import load_simulation_checkpoint

        checkpoint = load_simulation_checkpoint(dir_path)

        # Use saved values unless overridden
        if mating_regime is None:
            mating_regime = checkpoint['mating_regime']
        if recombination_map is None:
            recombination_map = checkpoint['recombination_map']
        if metadata_rules is None:
            metadata_rules = checkpoint.get('metadata_rules')
            unrestored = [r for r in metadata_rules or []
                          if isinstance(r, inherit.UnrestoredCustom)]
            if unrestored:
                raise ValueError(
                    f"The checkpoint used Custom metadata rule(s) {unrestored}, "
                    f"whose functions are not saved. Pass all metadata rules "
                    f"again via from_checkpoint(..., metadata_rules=[...])."
                )

        if standardize_with is None:
            standardize_with = checkpoint.get('standardize_with')
            if standardize_with is None:
                warnings.warn(
                    "This checkpoint predates the standardize_with option and "
                    "was simulated by re-standardizing genotypes every "
                    "generation; resuming with standardize_with='current' to "
                    "match. Pass standardize_with='founders' to fix allele "
                    "effects at the founders' frequencies instead.",
                    stacklevel=2,
                )
                standardize_with = 'current'

        if mating_regime is None:
            raise ValueError(
                "No mating regime in checkpoint and none provided. "
                "Pass mating_regime= explicitly."
            )
        if recombination_map is None:
            raise ValueError(
                "No recombination map in checkpoint and none provided. "
                "Pass recombination_map= explicitly."
            )

        # Find the earliest generation haplotypes as "founders" for construction
        min_gen = min(checkpoint['haplotype_history'].keys())
        founder = checkpoint['haplotype_history'][min_gen]

        sim = cls(
            founder_haplotypes=founder,
            architecture=checkpoint['architecture'],
            mating_regime=mating_regime,
            recombination_map=recombination_map,
            retain_haplotypes=checkpoint['retain_haplotypes'],
            retain_phenotypes=checkpoint['retain_phenotypes'],
            callbacks=callbacks,
            filters=filters,
            statistics=statistics,
            metadata_rules=metadata_rules,
            standardize_with=standardize_with,
        )

        # Restore state
        sim.haplotype_history = checkpoint['haplotype_history']
        sim.phenotype_history = checkpoint['phenotype_history']
        sim.pedigree_history = checkpoint['pedigree_history']
        sim.rng = checkpoint['rng']
        sim.generation = checkpoint['generation']
        sim.results = checkpoint.get('results', [])

        founder_af = checkpoint.get('founder_af')
        hap0 = sim.haplotype_history.get(0)
        # A retained generation-0 GraphHaplotypeOperator is stale once meiosis
        # has run (it shares the mutated graph), so only dense generation-0
        # haplotypes, or a checkpoint taken at generation 0, can stand in.
        if founder_af is None and hap0 is not None and (
                sim.generation == 0 or isinstance(hap0, DenseHaplotypeArray)):
            founder_af = hap0.recompute_af()
        if founder_af is not None:
            founder_af = np.array(founder_af, dtype=np.float64)
            if founder_af.shape != (founder.m,):
                raise ValueError(
                    f"Checkpoint founder allele frequencies have shape "
                    f"{founder_af.shape}, expected ({founder.m},)."
                )
        elif standardize_with == 'founders':
            raise ValueError(
                "standardize_with='founders' needs the founders' allele "
                "frequencies, but this checkpoint has neither saved "
                "frequencies nor usable generation-0 haplotypes. Resume with "
                "standardize_with='current'."
            )
        sim.founder_af = founder_af

        return sim

    @property
    def haplotypes(self) -> HaplotypeOperator:
        """Current generation's haplotypes."""
        return self.haplotype_history[self.generation]

    @property
    def phenotypes(self) -> PhenotypeArray:
        """Current generation's phenotypes."""
        return self.phenotype_history[self.generation]

    def _validate(self) -> None:
        """Check that architecture effect dimensions match haplotype dimensions."""
        from xftsim.arch import GeneticComponent, MVGeneticComponent, HaplotypeGeneticComponent
        hap = self.haplotype_history[0]
        m = hap.m
        for node in self.architecture.nodes:
            comp = node.component
            if isinstance(comp, (GeneticComponent, MVGeneticComponent, HaplotypeGeneticComponent)):
                eff_m = comp.effects.m
                if eff_m != m:
                    raise ValueError(
                        f"Effect dimension mismatch for node {node.outputs}: "
                        f"effects have m={eff_m} but founder haplotypes have m={m}"
                    )

    def run(self, n_generations: int) -> None:
        """
        Run the simulation for n_generations.

        Generation 0: compute phenotypes from founder haplotypes, assign mates.
        Generation t>0: meiosis -> compute phenotypes -> assign mates.

        Parameters
        ----------
        n_generations : int
            Number of generations to simulate (including gen 0).
        """
        self._initialize_metadata()
        self._validate()
        self._validate_metadata(self.haplotype_history[0].samples, n_generations)
        self.founder_af = np.array(self.haplotype_history[0].recompute_af(),
                                   dtype=np.float64)

        # --- Generation 0: founders ---
        hap = self.haplotype_history[0]
        pheno = self.architecture.compute(
            hap, rng=self.rng,
            phenotype_history=self.phenotype_history,
            pedigree_history=self.pedigree_history,
            generation=0,
            reference_af=self._reference_af(),
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
            prev_hap = self.haplotype_history[gen - 1]
            prev_assignment = self._inherit_metadata(
                self._mate_assignments[gen - 1], prev_hap.samples,
            )

            # Meiosis: produce offspring haplotypes. Pass self.rng so that
            # crossover sampling is driven by the simulation's seeded RNG
            # rather than numba's per-thread state (which is non-deterministic
            # across runs and not affected by sim seed).
            offspring_hap = prev_hap.meiosis(
                prev_assignment, self.recombination_map, rng=self.rng,
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
                reference_af=self._reference_af(),
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

    def continue_run(self, n_additional: int) -> None:
        """
        Continue a simulation for n_additional generations from current state.

        Used after loading from a checkpoint via from_checkpoint().

        Parameters
        ----------
        n_additional : int
            Number of additional generations to simulate.
        """
        start_gen = self.generation
        self.stop = False
        if n_additional > 0:
            self._validate_metadata(
                self.haplotype_history[start_gen].samples, n_additional + 1,
            )

        # Mate assignment for current generation (needed for meiosis to next gen)
        if n_additional > 0 and start_gen not in self._mate_assignments:
            hap = self.haplotype_history[start_gen]
            assignment = self.mating_regime.mate(
                hap.samples, rng=self.rng,
                phenotypes=self.phenotype_history.get(start_gen),
            )
            self._mate_assignments[start_gen] = assignment

        for gen in range(start_gen + 1, start_gen + 1 + n_additional):
            prev_hap = self.haplotype_history[gen - 1]
            prev_assignment = self._inherit_metadata(
                self._mate_assignments[gen - 1], prev_hap.samples,
            )

            offspring_hap = prev_hap.meiosis(
                prev_assignment, self.recombination_map, rng=self.rng,
            )
            self.haplotype_history[gen] = offspring_hap
            self.generation = gen

            ped = PedigreeArray(
                offspring_samples=prev_assignment.offspring_samples,
                maternal_idx=prev_assignment.maternal_idx,
                paternal_idx=prev_assignment.paternal_idx,
                parent_n=prev_hap.n,
            )
            self.pedigree_history[gen] = ped

            pheno = self.architecture.compute(
                offspring_hap, rng=self.rng,
                phenotype_history=self.phenotype_history,
                pedigree_history=self.pedigree_history,
                generation=gen,
                reference_af=self._reference_af(),
            )
            self.phenotype_history[gen] = pheno

            self._run_filters_and_stats(gen)

            if gen < start_gen + n_additional:
                assignment = self.mating_regime.mate(
                    offspring_hap.samples, rng=self.rng,
                    phenotypes=self.phenotype_history.get(gen),
                )
                self._mate_assignments[gen] = assignment

            self._enforce_retention(gen)

            self._run_callbacks()
            if self.stop:
                return

    def _reference_af(self) -> np.ndarray | None:
        """Allele frequencies genetic components standardize with (None = current)."""
        if self.standardize_with == 'current':
            return None
        if self.founder_af is None:
            raise RuntimeError(
                "standardize_with='founders' but the founders' allele "
                "frequencies are unknown: start with run(), or resume with "
                "Simulation.from_checkpoint()."
            )
        return self.founder_af

    def _initialize_metadata(self) -> None:
        """Run the rules' generation-0 hooks on the founders.

        Derived fields are added to a shallow copy of the founder operator,
        so the caller's object is never modified.
        """
        if not self.metadata_rules:
            return
        hap = self.haplotype_history[0]
        new = inherit.initialize_metadata(self.metadata_rules, hap.samples)
        if not new:
            return
        hap = copy.copy(hap)
        hap.samples = hap.samples.with_extra(new, replace=False)
        self.haplotype_history[0] = hap

    def _required_metadata_keys(self) -> dict[str, list[str]]:
        """Metadata field -> descriptions of the components that read it."""
        needed: dict[str, list[str]] = {}

        def add(keys, who):
            for k in keys:
                needed.setdefault(k, []).append(who)

        for node in self.architecture.nodes:
            g = node.grouping
            if g is not None and g not in ('FID', 'sex', 'mother', 'father'):
                add([g], f"architecture node {node.outputs} (| {g})")
            add(getattr(node.component, 'metadata_keys', ()),
                f"architecture node {node.outputs}")
        regime = self.mating_regime
        while regime is not None:
            add(getattr(regime, 'metadata_keys', ()),
                f"mating regime {type(regime).__name__}")
            regime = getattr(regime, 'regime', None)
        for name, filt in self.filters.items():
            add(getattr(filt, 'metadata_keys', ()), f"filter '{name}'")
        for stat in self.statistics:
            add(getattr(stat, 'metadata_keys', ()),
                f"statistic {type(stat).__name__}")
        return needed

    def _validate_metadata(self, samples: SampleMeta, n_generations: int) -> None:
        """Check that every metadata field a component reads will be inherited."""
        if n_generations < 2:
            return
        rules = self.metadata_rules or []
        inherit.check_rules(rules, samples)
        produced = inherit.output_keys(rules, samples)
        missing = {k: who for k, who in self._required_metadata_keys().items()
                   if k not in produced}
        if missing:
            lines = [f"  '{k}': used by {', '.join(who)}"
                     for k, who in sorted(missing.items())]
            raise ValueError(
                "These metadata fields are read after generation 0 but no "
                "metadata rule produces them:\n" + "\n".join(lines)
                + "\nAdd a rule for each to Simulation(metadata_rules=...), "
                f"e.g. xftsim.inherit.Maternal({sorted(missing)[0]!r})."
            )

    def _inherit_metadata(self, assignment: MateAssignment,
                          parents: SampleMeta) -> MateAssignment:
        """Return ``assignment`` with offspring metadata filled in by the rules."""
        if self.metadata_rules is None:
            return assignment
        extra = inherit.inherit_metadata(
            self.metadata_rules, parents,
            assignment.maternal_idx, assignment.paternal_idx, self.rng,
        )
        dropped = (set(parents.extra)
                   - inherit.handled_keys(self.metadata_rules, parents)
                   - set(extra) - self._warned_dropped_metadata)
        if dropped:
            self._warned_dropped_metadata |= dropped
            warnings.warn(
                f"Metadata field(s) {sorted(dropped)} have no inheritance rule "
                f"and are not carried past generation {parents.generation}. "
                f"Add a rule from xftsim.inherit to keep them.",
                stacklevel=3,
            )
        offspring = assignment.offspring_samples.with_extra(extra, replace=False)
        return dataclasses.replace(assignment, offspring_samples=offspring)

    def _run_filters_and_stats(self, gen: int) -> None:
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
                    self.phenotype_history, filtered_views, gen,
                    haplotype_history=self.haplotype_history,
                )
                base_key = type(stat).__name__
                count = name_counts.get(base_key, 0)
                name_counts[base_key] = count + 1
                key = base_key if count == 0 else f"{base_key}_{count}"
                stats[key] = result
            self.results.append(GenerationResult(generation=gen, statistics=stats))

    def _enforce_retention(self, current_gen: int) -> None:
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

    def _run_callbacks(self) -> None:
        """Execute all registered callbacks."""
        for cb in self.callbacks:
            cb(self)

    def __repr__(self) -> str:
        return (f"Simulation(generation={self.generation}, "
                f"n={self.haplotypes.n}, m={self.haplotypes.m})")
