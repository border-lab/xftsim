"""
New mate assignment and random mating for the refactored simulation loop.

NMateAssignment: dataclass linking offspring to parents by index.
RandomMating: shuffles and pairs individuals to produce offspring.
"""
import numpy as np
from dataclasses import dataclass

from xftsim.struct import SampleMeta


@dataclass
class NMateAssignment:
    """
    Links offspring to parents via integer indices into the parent generation.

    Parameters
    ----------
    offspring_samples : SampleMeta
        Metadata for the offspring (unique iids, fids, sex, generation).
    maternal_idx : np.ndarray
        (n_offspring,) indices into the parent generation for mothers.
    paternal_idx : np.ndarray
        (n_offspring,) indices into the parent generation for fathers.
    """
    offspring_samples: SampleMeta
    maternal_idx: np.ndarray
    paternal_idx: np.ndarray

    def __post_init__(self):
        self.maternal_idx = np.asarray(self.maternal_idx, dtype=np.int64)
        self.paternal_idx = np.asarray(self.paternal_idx, dtype=np.int64)
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
            if np.any(self.maternal_idx < 0):
                raise ValueError("maternal_idx contains negative indices")
            if np.any(self.paternal_idx < 0):
                raise ValueError("paternal_idx contains negative indices")

    @property
    def n_offspring(self) -> int:
        return self.offspring_samples.n

    def __repr__(self):
        return (f"NMateAssignment(n_offspring={self.n_offspring}, "
                f"generation={self.offspring_samples.generation})")


class RandomMating:
    """
    Random mating: shuffle individuals, pair them up, produce offspring.

    Parameters
    ----------
    offspring_per_pair : int
        Number of offspring per mating pair. Default 2.
    """

    def __init__(self, offspring_per_pair: int = 2):
        if offspring_per_pair < 1:
            raise ValueError("offspring_per_pair must be >= 1")
        self.offspring_per_pair = offspring_per_pair

    def mate(self, samples: SampleMeta, rng=None, phenotypes=None) -> NMateAssignment:
        """
        Produce a mate assignment from the current generation.

        Algorithm:
        - Separate individuals by sex (0=female, 1=male).
        - Shuffle each group independently.
        - Pair up: min(n_female, n_male) pairs.
        - Each pair produces offspring_per_pair offspring.
        - Offspring get sequential iids, pair-based fids, alternating sex.

        Parameters
        ----------
        samples : SampleMeta
            Current generation's sample metadata.
        rng : np.random.RandomState, optional
            Random state for reproducibility.
        phenotypes : NPhenotypeArray, optional
            Ignored by RandomMating (accepted for interface compatibility).

        Returns
        -------
        NMateAssignment
        """
        if rng is None:
            rng = np.random.RandomState()

        female_idx = np.where(samples.sex == 0)[0]
        male_idx = np.where(samples.sex == 1)[0]

        if len(female_idx) == 0 or len(male_idx) == 0:
            raise ValueError("Need at least one female and one male for mating")

        # Shuffle each sex group
        rng.shuffle(female_idx)
        rng.shuffle(male_idx)

        # Number of pairs = min of the two groups
        n_pairs = min(len(female_idx), len(male_idx))
        mothers = female_idx[:n_pairs]
        fathers = male_idx[:n_pairs]

        opp = self.offspring_per_pair
        n_offspring = n_pairs * opp

        # Expand: each pair produces opp offspring
        maternal_idx = np.repeat(mothers, opp)
        paternal_idx = np.repeat(fathers, opp)

        # Offspring metadata
        iid = np.arange(n_offspring, dtype=np.int64)
        fid = np.repeat(np.arange(n_pairs, dtype=np.int64), opp)
        # Alternate sex within each family
        sex_pattern = np.tile(np.arange(opp, dtype=np.int64) % 2, n_pairs)
        generation = samples.generation + 1

        offspring_samples = SampleMeta(
            iid=iid, fid=fid, sex=sex_pattern, generation=generation,
        )

        return NMateAssignment(
            offspring_samples=offspring_samples,
            maternal_idx=maternal_idx,
            paternal_idx=paternal_idx,
        )

    def __repr__(self):
        return f"RandomMating(offspring_per_pair={self.offspring_per_pair})"


class LinearAssortativeMating:
    """
    Assortative mating via rank-order pairing on a phenotypic composite.

    Algorithm:
    1. Standardize each component in ``component_names`` to mean 0, sd 1.
    2. Compute composite = average of standardized components.
    3. Mating score = sqrt(|r|) * composite + sqrt(1-|r|) * noise.
       If r < 0, negate the composite for one sex (disassortative).
    4. Sort each sex by mating score, pair in rank order.

    Parameters
    ----------
    component_names : list[str]
        Phenotype component names to use for assortment.
    r : float
        Target spousal correlation. -1 < r < 1. 0 = random mating.
    offspring_per_pair : int
        Number of offspring per mating pair.
    """

    def __init__(self, component_names, r=0.0, offspring_per_pair=2):
        if not -1 < r < 1:
            raise ValueError(f"r must be in (-1, 1), got {r}")
        if offspring_per_pair < 1:
            raise ValueError("offspring_per_pair must be >= 1")
        self.component_names = list(component_names)
        self.r = float(r)
        self.offspring_per_pair = offspring_per_pair

    def mate(self, samples: SampleMeta, rng=None, phenotypes=None) -> NMateAssignment:
        """
        Produce a mate assignment with phenotypic assortment.

        Falls back to random mating if r==0 or phenotypes is None.
        """
        if rng is None:
            rng = np.random.RandomState()

        # Fall back to random if no assortment needed
        if self.r == 0.0 or phenotypes is None:
            return RandomMating(self.offspring_per_pair).mate(samples, rng=rng)

        female_idx = np.where(samples.sex == 0)[0]
        male_idx = np.where(samples.sex == 1)[0]

        if len(female_idx) == 0 or len(male_idx) == 0:
            raise ValueError("Need at least one female and one male for mating")

        # Compute composite score from phenotype components
        composites = np.zeros(samples.n, dtype=np.float64)
        n_comp = 0
        for name in self.component_names:
            if name in phenotypes:
                vals = phenotypes[name].copy()
                sd = vals.std()
                if sd > 0:
                    vals = (vals - vals.mean()) / sd
                composites += vals
                n_comp += 1
        if n_comp > 0:
            composites /= n_comp

        # Mating score: sqrt(|r|) * composite + sqrt(1-|r|) * noise
        abs_r = abs(self.r)
        noise = rng.normal(0, 1, size=samples.n)
        scores = np.sqrt(abs_r) * composites + np.sqrt(1.0 - abs_r) * noise

        # Disassortative: negate scores for one sex
        if self.r < 0:
            scores[male_idx] *= -1

        # Sort each sex by score, pair in rank order
        female_order = female_idx[np.argsort(scores[female_idx])]
        male_order = male_idx[np.argsort(scores[male_idx])]

        n_pairs = min(len(female_order), len(male_order))
        mothers = female_order[:n_pairs]
        fathers = male_order[:n_pairs]

        opp = self.offspring_per_pair
        n_offspring = n_pairs * opp

        maternal_idx = np.repeat(mothers, opp)
        paternal_idx = np.repeat(fathers, opp)

        iid = np.arange(n_offspring, dtype=np.int64)
        fid = np.repeat(np.arange(n_pairs, dtype=np.int64), opp)
        sex_pattern = np.tile(np.arange(opp, dtype=np.int64) % 2, n_pairs)
        generation = samples.generation + 1

        offspring_samples = SampleMeta(
            iid=iid, fid=fid, sex=sex_pattern, generation=generation,
        )

        return NMateAssignment(
            offspring_samples=offspring_samples,
            maternal_idx=maternal_idx,
            paternal_idx=paternal_idx,
        )

    def __repr__(self):
        return (f"LinearAssortativeMating(components={self.component_names}, "
                f"r={self.r}, offspring_per_pair={self.offspring_per_pair})")
