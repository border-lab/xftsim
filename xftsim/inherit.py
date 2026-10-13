"""
Inheritance rules for per-individual metadata (``SampleMeta.extra``).

Each generation, the simulation hands every rule the parent generation's
``SampleMeta`` together with the mate assignment's maternal/paternal indices,
and the rule returns the offspring's values for the fields it owns. Fields
without a rule are not carried forward.

InheritanceRule: abstract base class.
Midparent: numeric fields, offspring = (mother + father) / 2.
Maternal / Paternal: copy one parent's value.
RandomParent: copy a uniformly chosen parent's value.
Custom: user-supplied function.
AncestryProportions: one-hot founder population labels, then midparent.

Rules are stateless: everything a rule needs is read from the parent
``SampleMeta``, so a rule behaves identically after resuming from a
checkpoint.

Examples
--------
>>> import xftsim as xft
>>> rules = [xft.inherit.AncestryProportions('population'),
...          xft.inherit.Maternal('population')]
>>> sim = xft.sim.Simulation(founders, arch, mating, rmap,
...                          metadata_rules=rules)  # doctest: +SKIP
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Callable, Iterable

import numpy as np

from xftsim.struct import SampleMeta

_IDENT_RE = re.compile(r'^[A-Za-z_]\w*$')


def _require(samples: SampleMeta, keys: Iterable[str], rule: InheritanceRule) -> None:
    missing = [k for k in keys if k not in samples.extra]
    if missing:
        raise KeyError(
            f"{rule!r} needs metadata field(s) {missing} but generation "
            f"{samples.generation} only has {sorted(samples.extra)}. If this "
            f"simulation was resumed from a checkpoint written before these "
            f"fields existed, rerun it from generation 0 with metadata_rules."
        )


def _require_numeric(samples: SampleMeta, keys: Iterable[str],
                     rule: InheritanceRule) -> None:
    for k in keys:
        arr = samples.extra[k]
        if not (np.issubdtype(arr.dtype, np.number) or arr.dtype == bool):
            raise TypeError(
                f"{rule!r} averages '{k}', which has non-numeric dtype "
                f"{arr.dtype}. Use Maternal, Paternal or RandomParent for "
                f"categorical fields."
            )


class InheritanceRule(ABC):
    """Abstract base class for metadata inheritance rules.

    Subclasses set ``keys`` and implement ``inherit()``. Override
    ``initialize()`` to validate or derive fields at generation 0, and
    ``output_keys()`` / ``handled_keys()`` when the fields written are not
    simply ``keys``.

    Attributes
    ----------
    keys : tuple[str, ...]
        Metadata fields this rule owns.
    """

    keys: tuple[str, ...] = ()

    def initialize(self, samples: SampleMeta) -> dict[str, np.ndarray]:
        """Validate founder metadata and return fields to add at generation 0.

        Parameters
        ----------
        samples : SampleMeta
            Founder sample metadata.

        Returns
        -------
        dict[str, np.ndarray]
            New fields to add to the founders (empty by default).
        """
        _require(samples, self.keys, self)
        return {}

    def output_keys(self, samples: SampleMeta) -> set[str]:
        """Fields this rule writes for the offspring of ``samples``."""
        return set(self.keys)

    def handled_keys(self, samples: SampleMeta) -> set[str]:
        """Parent fields this rule accounts for.

        Parent fields that no rule handles are dropped with a warning.
        """
        return self.output_keys(samples)

    @abstractmethod
    def inherit(self, parents: SampleMeta, maternal_idx: np.ndarray,
                paternal_idx: np.ndarray,
                rng: np.random.RandomState) -> dict[str, np.ndarray]:
        """Compute offspring values from the parent generation.

        Parameters
        ----------
        parents : SampleMeta
            Parent generation's sample metadata.
        maternal_idx, paternal_idx : np.ndarray
            (n_offspring,) indices into ``parents``.
        rng : np.random.RandomState
            Simulation RNG. Deterministic rules must not draw from it.

        Returns
        -------
        dict[str, np.ndarray]
            Field name -> (n_offspring,) array.
        """
        ...

    def to_config(self) -> dict[str, object]:
        """JSON-serializable description, used by checkpoints."""
        return {'type': type(self).__name__, 'keys': list(self.keys)}

    def __repr__(self) -> str:
        return f"{type(self).__name__}({', '.join(map(repr, self.keys))})"


class Midparent(InheritanceRule):
    """Offspring value = mean of the two parents' values (numeric fields).

    Parameters
    ----------
    *keys : str
        Numeric metadata fields to average.
    """

    def __init__(self, *keys: str) -> None:
        if not keys:
            raise ValueError("Midparent needs at least one key")
        self.keys = tuple(keys)

    def initialize(self, samples: SampleMeta) -> dict[str, np.ndarray]:
        _require(samples, self.keys, self)
        _require_numeric(samples, self.keys, self)
        return {}

    def inherit(self, parents, maternal_idx, paternal_idx, rng):
        _require(parents, self.keys, self)
        return {
            k: 0.5 * (parents.extra[k][maternal_idx].astype(np.float64)
                      + parents.extra[k][paternal_idx].astype(np.float64))
            for k in self.keys
        }


class Maternal(InheritanceRule):
    """Offspring copies the mother's value (e.g. mtDNA lineage, deme).

    Parameters
    ----------
    *keys : str
        Metadata fields to copy.
    """

    def __init__(self, *keys: str) -> None:
        if not keys:
            raise ValueError("Maternal needs at least one key")
        self.keys = tuple(keys)

    def inherit(self, parents, maternal_idx, paternal_idx, rng):
        _require(parents, self.keys, self)
        return {k: parents.extra[k][maternal_idx].copy() for k in self.keys}


class Paternal(InheritanceRule):
    """Offspring copies the father's value (e.g. Y lineage, surname).

    Parameters
    ----------
    *keys : str
        Metadata fields to copy.
    """

    def __init__(self, *keys: str) -> None:
        if not keys:
            raise ValueError("Paternal needs at least one key")
        self.keys = tuple(keys)

    def inherit(self, parents, maternal_idx, paternal_idx, rng):
        _require(parents, self.keys, self)
        return {k: parents.extra[k][paternal_idx].copy() for k in self.keys}


class RandomParent(InheritanceRule):
    """Offspring copies the value of a uniformly chosen parent.

    One parent is drawn per offspring and used for all of this rule's keys.
    This is the only built-in rule that draws from the simulation RNG, so
    adding it changes all later random draws for a given seed.

    Parameters
    ----------
    *keys : str
        Metadata fields to copy.
    """

    def __init__(self, *keys: str) -> None:
        if not keys:
            raise ValueError("RandomParent needs at least one key")
        self.keys = tuple(keys)

    def inherit(self, parents, maternal_idx, paternal_idx, rng):
        _require(parents, self.keys, self)
        from_father = rng.randint(0, 2, size=len(maternal_idx)).astype(bool)
        return {
            k: np.where(from_father,
                        parents.extra[k][paternal_idx],
                        parents.extra[k][maternal_idx])
            for k in self.keys
        }


class Custom(InheritanceRule):
    """User-defined inheritance.

    Parameters
    ----------
    keys : str or list[str]
        Fields ``fn`` returns.
    fn : callable
        ``fn(mother, father, rng) -> dict``. ``mother`` and ``father`` map
        every parent metadata field to an (n_offspring,) array aligned with
        the offspring; the result must map each of ``keys`` to an
        (n_offspring,) array.

    Notes
    -----
    Functions are not saved in checkpoints. Pass the rule again via
    ``Simulation.from_checkpoint(..., metadata_rules=...)`` when resuming.
    """

    def __init__(self, keys: str | Iterable[str],
                 fn: Callable[[dict, dict, np.random.RandomState], dict]) -> None:
        self.keys = (keys,) if isinstance(keys, str) else tuple(keys)
        if not self.keys:
            raise ValueError("Custom needs at least one key")
        self.fn = fn

    def initialize(self, samples: SampleMeta) -> dict[str, np.ndarray]:
        return {}

    def inherit(self, parents, maternal_idx, paternal_idx, rng):
        mother = {k: v[maternal_idx] for k, v in parents.extra.items()}
        father = {k: v[paternal_idx] for k, v in parents.extra.items()}
        out = self.fn(mother, father, rng)
        n_off = len(maternal_idx)
        if set(out) != set(self.keys):
            raise ValueError(
                f"{self!r}: fn returned keys {sorted(out)}, expected "
                f"{sorted(self.keys)}"
            )
        result = {}
        for k in self.keys:
            arr = np.asarray(out[k])
            if arr.shape[:1] != (n_off,):
                raise ValueError(
                    f"{self!r}: fn returned '{k}' with shape {arr.shape}, "
                    f"expected ({n_off},)"
                )
            result[k] = arr
        return result

    def to_config(self) -> dict[str, object]:
        return {'type': 'Custom', 'keys': list(self.keys)}


class AncestryProportions(InheritanceRule):
    """Expected (global) ancestry proportions per founder population.

    At generation 0, the categorical ``source`` labels are one-hot encoded
    into one float column per population, ``<prefix><label>``. Founders that
    already carry fractional ``<prefix>*`` columns (e.g. admixed real data)
    are used as-is when ``source`` is absent. In every later generation each
    column is the mean of the two parents' values, so each individual's
    proportions sum to 1.

    These are expected proportions given the pedigree; realized genome-wide
    ancestry also varies with Mendelian segregation, which is not tracked.

    The rule owns every field whose name starts with ``prefix``: founders
    may not carry other fields with that prefix.

    Parameters
    ----------
    source : str
        Founder field holding population labels. Default ``'population'``
        (as set by ``founders.founder_haplotypes_from_stdpopsim``).
    prefix : str
        Prefix for the ancestry columns. Default ``'ancestry_'``.
    """

    def __init__(self, source: str = 'population',
                 prefix: str = 'ancestry_') -> None:
        if not prefix:
            raise ValueError("prefix must be non-empty")
        self.source = source
        self.prefix = prefix
        self.keys = ()

    def _columns(self, samples: SampleMeta) -> list[str]:
        return sorted(k for k in samples.extra if k.startswith(self.prefix))

    def initialize(self, samples: SampleMeta) -> dict[str, np.ndarray]:
        if self.source not in samples.extra:
            cols = self._columns(samples)
            if cols:
                _require_numeric(samples, cols, self)
                total = np.sum([samples.extra[k] for k in cols], axis=0)
                if not np.allclose(total, 1.0):
                    raise ValueError(
                        f"{self!r}: founder fields {cols} are used as ancestry "
                        f"proportions but do not sum to 1 for every "
                        f"individual. Every field starting with "
                        f"'{self.prefix}' must be an ancestry proportion."
                    )
                return {}
            raise KeyError(
                f"{self!r} needs founder field '{self.source}' (population "
                f"labels) or existing '{self.prefix}*' columns; founders have "
                f"{sorted(samples.extra)}."
            )

        labels = np.asarray(samples.extra[self.source]).astype(str)
        new: dict[str, np.ndarray] = {}
        origin: dict[str, str] = {}
        for label in np.unique(labels):
            key = self.prefix + re.sub(r'\W', '_', label)
            if not _IDENT_RE.match(key):
                raise ValueError(
                    f"{self!r}: column name '{key}' for label '{label}' is "
                    f"not a valid identifier; choose a different prefix."
                )
            if key in origin:
                raise ValueError(
                    f"{self!r}: labels '{origin[key]}' and '{label}' both map "
                    f"to column '{key}'."
                )
            origin[key] = label
            new[key] = (labels == label).astype(np.float64)

        for key, col in new.items():
            if key in samples.extra and not np.array_equal(samples.extra[key], col):
                raise ValueError(
                    f"{self!r}: founders already have a different '{key}' field."
                )
        foreign = sorted(set(self._columns(samples)) - set(new))
        if foreign:
            raise ValueError(
                f"{self!r}: founder field(s) {foreign} start with "
                f"'{self.prefix}' but are not population labels in "
                f"'{self.source}'. This rule averages every field with its "
                f"prefix, so rename them or choose another prefix."
            )
        return new

    def output_keys(self, samples: SampleMeta) -> set[str]:
        return set(self._columns(samples))

    def handled_keys(self, samples: SampleMeta) -> set[str]:
        return self.output_keys(samples) | {self.source}

    def inherit(self, parents, maternal_idx, paternal_idx, rng):
        cols = self._columns(parents)
        if not cols:
            raise KeyError(
                f"{self!r}: parent generation {parents.generation} has no "
                f"'{self.prefix}*' fields. If this simulation was resumed from "
                f"a checkpoint written before they existed, rerun it from "
                f"generation 0 with metadata_rules."
            )
        return {
            k: 0.5 * (parents.extra[k][maternal_idx] + parents.extra[k][paternal_idx])
            for k in cols
        }

    def to_config(self) -> dict[str, object]:
        return {'type': 'AncestryProportions', 'source': self.source,
                'prefix': self.prefix}

    def __repr__(self) -> str:
        return f"AncestryProportions(source={self.source!r}, prefix={self.prefix!r})"


class UnrestoredCustom(InheritanceRule):
    """Placeholder for a ``Custom`` rule loaded from a checkpoint.

    Checkpoints record only a Custom rule's keys, not its function, so a
    loaded checkpoint can be inspected but not resumed with this rule.
    """

    def __init__(self, *keys: str) -> None:
        self.keys = tuple(keys)

    def initialize(self, samples: SampleMeta) -> dict[str, np.ndarray]:
        return {}

    def inherit(self, parents, maternal_idx, paternal_idx, rng):
        raise ValueError(
            f"{self!r}: this Custom rule's function was not saved in the "
            f"checkpoint. Pass the rules again via "
            f"Simulation.from_checkpoint(..., metadata_rules=[...])."
        )

    def to_config(self) -> dict[str, object]:
        return {'type': 'Custom', 'keys': list(self.keys)}


# ---------------------------------------------------------------------------
# Applying rule sets
# ---------------------------------------------------------------------------

def check_rules(rules: list[InheritanceRule], samples: SampleMeta) -> None:
    """Raise if two rules write the same field for the offspring of ``samples``."""
    owner: dict[str, InheritanceRule] = {}
    for rule in rules:
        for k in rule.output_keys(samples):
            if k in owner:
                raise ValueError(
                    f"Metadata field '{k}' is written by both {owner[k]!r} "
                    f"and {rule!r}."
                )
            owner[k] = rule


def initialize_metadata(rules: list[InheritanceRule],
                        samples: SampleMeta) -> dict[str, np.ndarray]:
    """Run every rule's generation-0 hook.

    Returns
    -------
    dict[str, np.ndarray]
        Fields to add to the founders.
    """
    new: dict[str, np.ndarray] = {}
    for rule in rules:
        new.update(rule.initialize(samples))
    check_rules(rules, samples.with_extra(new, replace=False))
    return new


def inherit_metadata(rules: list[InheritanceRule], parents: SampleMeta,
                     maternal_idx: np.ndarray, paternal_idx: np.ndarray,
                     rng: np.random.RandomState) -> dict[str, np.ndarray]:
    """Apply every rule and merge the offspring fields."""
    out: dict[str, np.ndarray] = {}
    for rule in rules:
        out.update(rule.inherit(parents, maternal_idx, paternal_idx, rng))
    return out


def output_keys(rules: list[InheritanceRule], samples: SampleMeta) -> set[str]:
    """Union of the fields the rules write for the offspring of ``samples``."""
    keys: set[str] = set()
    for rule in rules:
        keys |= rule.output_keys(samples)
    return keys


def handled_keys(rules: list[InheritanceRule], samples: SampleMeta) -> set[str]:
    """Union of the parent fields the rules account for."""
    keys: set[str] = set()
    for rule in rules:
        keys |= rule.handled_keys(samples)
    return keys


_RULES_BY_NAME = {
    'Midparent': Midparent,
    'Maternal': Maternal,
    'Paternal': Paternal,
    'RandomParent': RandomParent,
}


def rule_from_config(config: dict[str, object]) -> InheritanceRule:
    """Inverse of ``InheritanceRule.to_config()`` for built-in rules."""
    kind = config.get('type')
    if kind in _RULES_BY_NAME:
        return _RULES_BY_NAME[kind](*config['keys'])
    if kind == 'AncestryProportions':
        return AncestryProportions(source=config['source'], prefix=config['prefix'])
    if kind == 'Custom':
        return UnrestoredCustom(*config['keys'])
    raise ValueError(f"Unknown metadata rule type {kind!r}")
