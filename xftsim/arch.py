"""
Architecture system for the new xftsim design.

ArchComponent ABC, concrete components, ArchNode, and Architecture class.
Supports both programmatic construction (arch.add()) and formula parsing.
"""
from __future__ import annotations

import re
import warnings
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union
from collections import OrderedDict

from xftsim.struct import HaplotypeOperator, PhenotypeArray, PedigreeArray
from xftsim.effect import EffectSpec


# ---------------------------------------------------------------------------
# ArchComponent ABC + concrete components
# ---------------------------------------------------------------------------

class ArchComponent(ABC):
    """
    Abstract base class for architecture components (DSL built-in functions).

    Attributes
    ----------
    name : str
        Component name (e.g. 'genetic', 'noise').
    kind : str
        One of 'genetic', 'generative', 'aggregating'.
    accepts_grouping : bool
        Whether this component can use the | operator.
    """

    name: str = ""
    kind: str = ""
    accepts_grouping: bool = False

    @abstractmethod
    def compute(self, node: "ArchNode", haplotypes: HaplotypeOperator,
                phenotypes: PhenotypeArray, **kwargs) -> np.ndarray:
        """
        Execute this component and return the result array.

        Parameters
        ----------
        node : ArchNode
            The node being executed (provides inputs, outputs, grouping).
        haplotypes : HaplotypeOperator
            Current generation's haplotype data.
        phenotypes : PhenotypeArray
            Current phenotype array (may already have upstream values).
        **kwargs
            Additional context: phenotype_history, pedigree_history, generation.

        Returns
        -------
        np.ndarray
            Result array of shape (n,) or (n, k) for multi-output.
        """
        ...


class GeneticComponent(ArchComponent):
    """Univariate genetic component: computes diploid G @ effects.

    Uses ``standardized_matvec`` when ``effects.standardized`` is True,
    otherwise plain ``matvec``.

    Parameters
    ----------
    effects : EffectSpec
        Effect sizes (shape (m,) for univariate). Typically an
        ``AdditiveEffects`` or ``SparseEffects`` instance.
    """
    name = "genetic"
    kind = "genetic"
    accepts_grouping = False

    def __init__(self, effects: EffectSpec):
        self.effects = effects

    def compute(self, node: ArchNode, haplotypes: HaplotypeOperator,
                phenotypes: PhenotypeArray, **kwargs: object) -> np.ndarray:
        if self.effects.standardized:
            return haplotypes.standardized_matvec(self.effects.effects)
        else:
            return haplotypes.matvec(self.effects.effects)

    def __repr__(self) -> str:
        return f"GeneticComponent(effects={self.effects})"


class MVGeneticComponent(GeneticComponent):
    """Multivariate genetic component: computes G @ effects for k traits.

    Inherits ``compute()`` from GeneticComponent since numpy's matvec
    handles both 1D and 2D effect arrays.

    Parameters
    ----------
    effects : EffectSpec
        Effect sizes with shape (m, k). Typically a
        ``MultivariateEffects`` instance.
    """
    name = "mvGenetic"

    def __repr__(self) -> str:
        return f"MVGeneticComponent(effects={self.effects})"


class HaplotypeGeneticComponent(ArchComponent):
    """Haplotype-specific genetic component.

    Computes ``hap[:,:,0] @ effects`` (maternal) or
    ``hap[:,:,1] @ effects`` (paternal). Enables indirect genetic
    effects (IGE) formulas where maternal and paternal contributions
    are modeled separately.

    Parameters
    ----------
    effects : EffectSpec
        Effect sizes (shape (m,)).
    haplotype : str
        Which haplotype copy to use: ``'maternal'`` or ``'paternal'``.

    Raises
    ------
    ValueError
        If ``haplotype`` is not ``'maternal'`` or ``'paternal'``.
    """
    name = "haplotypeGenetic"
    kind = "genetic"
    accepts_grouping = False

    def __init__(self, effects: EffectSpec, haplotype: str = 'maternal'):
        if haplotype not in ('maternal', 'paternal'):
            raise ValueError(
                f"haplotype must be 'maternal' or 'paternal', got '{haplotype}'"
            )
        self.effects = effects
        self.haplotype = haplotype

    def compute(self, node: ArchNode, haplotypes: HaplotypeOperator,
                phenotypes: PhenotypeArray, **kwargs: object) -> np.ndarray:
        if self.haplotype == 'maternal':
            return haplotypes.matvec_maternal(self.effects.effects)
        else:
            return haplotypes.matvec_paternal(self.effects.effects)

    def __repr__(self) -> str:
        return f"HaplotypeGeneticComponent(effects={self.effects}, haplotype='{self.haplotype}')"


def _resolve_grouping(grouping: str | None, haplotypes: HaplotypeOperator,
                      **kwargs: object) -> np.ndarray | None:
    """
    Resolve a grouping variable to an (n,) label array.

    Parameters
    ----------
    grouping : str or None
        Grouping variable name. Special values: 'FID', 'sex', 'mother', 'father'.
        None means per-individual (IID).
    haplotypes : HaplotypeOperator
        Current generation's haplotype data.
    **kwargs
        Context: pedigree_history, generation, etc.

    Returns
    -------
    np.ndarray or None
        Label array of shape (n,), or None if no grouping (per-individual).
    """
    if grouping is None:
        return None

    generation = kwargs.get('generation', 0)
    pedigree_history = kwargs.get('pedigree_history', {})

    if grouping == 'FID':
        return haplotypes.samples.fid
    elif grouping == 'sex':
        return haplotypes.samples.sex
    elif grouping in ('mother', 'father'):
        if generation == 0 or generation not in pedigree_history:
            warnings.warn(
                f"Grouping by '{grouping}' at generation {generation} has no pedigree; "
                f"falling back to IID grouping."
            )
            return None
        ped = pedigree_history[generation]
        if grouping == 'mother':
            return ped.maternal_idx
        else:
            return ped.paternal_idx
    else:
        # Try extra fields on SampleMeta
        if grouping in haplotypes.samples.extra:
            return haplotypes.samples.extra[grouping]
        raise ValueError(
            f"Unknown grouping variable '{grouping}'. "
            f"Available: FID, sex, mother, father, or extra fields."
        )


class NoiseComponent(ArchComponent):
    """Univariate noise component.

    Draws iid N(0, variance) per individual. When grouping is active
    (e.g., ``| FID``), draws one shared value per group and broadcasts
    to all members.

    Parameters
    ----------
    variance : float
        Noise variance (used as the variance of the normal distribution).
    """
    name = "noise"
    kind = "generative"
    accepts_grouping = True

    def __init__(self, variance: float):
        self.variance = float(variance)

    def compute(self, node: ArchNode, haplotypes: HaplotypeOperator,
                phenotypes: PhenotypeArray, **kwargs: object) -> np.ndarray:
        n = haplotypes.n
        rng = kwargs.get('rng', np.random.RandomState())
        labels = _resolve_grouping(node.grouping, haplotypes, **kwargs)
        if labels is None:
            return rng.normal(0, np.sqrt(self.variance), size=n)
        # Grouped: draw one value per unique group, broadcast
        unique_labels, inverse = np.unique(labels, return_inverse=True)
        group_values = rng.normal(0, np.sqrt(self.variance), size=len(unique_labels))
        return group_values[inverse]

    def __repr__(self) -> str:
        return f"NoiseComponent(variance={self.variance})"


class CNoiseComponent(ArchComponent):
    """Correlated multivariate noise component.

    Draws N(0, cov) per individual. When grouping is active, draws one
    shared vector per group and broadcasts. Returns an (n, k) array.

    Parameters
    ----------
    cov : np.ndarray
        (k, k) covariance matrix. Must be square.

    Raises
    ------
    ValueError
        If ``cov`` is not a square matrix.
    """
    name = "cnoise"
    kind = "generative"
    accepts_grouping = True

    def __init__(self, cov: np.ndarray):
        self.cov = np.asarray(cov, dtype=np.float64)
        if self.cov.ndim != 2 or self.cov.shape[0] != self.cov.shape[1]:
            raise ValueError(f"cov must be a square matrix, got shape {self.cov.shape}")

    @property
    def k(self) -> int:
        """Number of correlated traits."""
        return self.cov.shape[0]

    def compute(self, node: ArchNode, haplotypes: HaplotypeOperator,
                phenotypes: PhenotypeArray, **kwargs: object) -> np.ndarray:
        n = haplotypes.n
        rng = kwargs.get('rng', np.random.RandomState())
        labels = _resolve_grouping(node.grouping, haplotypes, **kwargs)
        if labels is None:
            return rng.multivariate_normal(np.zeros(self.k), self.cov, size=n)
        # Grouped: draw one vector per unique group, broadcast
        unique_labels, inverse = np.unique(labels, return_inverse=True)
        group_values = rng.multivariate_normal(
            np.zeros(self.k), self.cov, size=len(unique_labels)
        )
        return group_values[inverse]

    def __repr__(self) -> str:
        return f"CNoiseComponent(cov={self.cov.shape})"


class ThresholdComponent(ArchComponent):
    """Binarizing threshold component: returns 1 where input exceeds threshold.

    Implements the liability threshold model: given a continuous phenotype
    (liability) and a threshold, produces a binary indicator (diagnosis).

    Parameters
    ----------
    source : str
        Name of the input phenotype to threshold.
    threshold : float
        Threshold value. Output is 1.0 where input > threshold, else 0.0.
    """
    name = "threshold"
    kind = "aggregating"
    accepts_grouping = False

    def __init__(self, source: str, threshold: float):
        self.source = source
        self.threshold = float(threshold)

    def compute(self, node: "ArchNode", haplotypes: HaplotypeOperator,
                phenotypes: PhenotypeArray, **kwargs: object) -> np.ndarray:
        y = phenotypes[self.source]
        return (y > self.threshold).astype(np.float64)

    def __repr__(self) -> str:
        return f"ThresholdComponent('{self.source}', {self.threshold})"


class AggregationComponent(ArchComponent):
    """Aggregation component: evaluates arithmetic expressions over phenotype values.

    Uses a custom tokenizer + shunting-yard evaluator (no ``eval()``).
    Supports ``+``, ``-``, ``*``, ``/``, scalar multiplication,
    parentheses, and dotted names (e.g., ``'Y.G + Y.E'``).

    Parameters
    ----------
    expression : str
        Arithmetic expression referencing phenotype names, e.g.
        ``'height.G + height.E'`` or ``'0.5 * (Y.G + Y.E)'``.
    """
    name = "aggregation"
    kind = "aggregating"
    accepts_grouping = False

    def __init__(self, expression: str):
        self.expression = expression
        # Extract input names (dotted identifiers, not numbers)
        self._input_names = self._extract_names(expression)

    @staticmethod
    def _extract_names(expr: str) -> list[str]:
        """Extract variable names from an arithmetic expression."""
        # Match identifiers (possibly dotted like height.G) but not pure numbers
        tokens = re.findall(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*', expr)
        # Deduplicate preserving order
        seen = set()
        result = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                result.append(t)
        return result

    def compute(self, node: ArchNode, haplotypes: HaplotypeOperator,
                phenotypes: PhenotypeArray, **kwargs: object) -> np.ndarray:
        n = haplotypes.n
        result = _evaluate_expression(self.expression, phenotypes, n)
        return result

    def __repr__(self) -> str:
        return f"AggregationComponent('{self.expression}')"


# ---------------------------------------------------------------------------
# Shunting-yard expression evaluator (no eval())
# ---------------------------------------------------------------------------

_PRECEDENCE = {'+': 1, '-': 1, '*': 2, '/': 2}
_TOKEN_RE = re.compile(
    r'(\d+\.?\d*(?:[eE][+-]?\d+)?)'   # numbers (int, float, scientific)
    r'|([A-Za-z_]\w*(?:\.\w+)*)'       # identifiers (possibly dotted)
    r'|([()+\-*/])'                      # operators and parens
)


def _tokenize(expr: str) -> list[tuple[str, object]]:
    """Tokenize an arithmetic expression into (type, value) pairs."""
    tokens = []
    for m in _TOKEN_RE.finditer(expr):
        if m.group(1):
            tokens.append(('NUM', float(m.group(1))))
        elif m.group(2):
            tokens.append(('NAME', m.group(2)))
        elif m.group(3):
            tokens.append(('OP', m.group(3)))
    return tokens


def _shunting_yard(tokens: list[tuple[str, object]]) -> list[tuple[str, object]]:
    """Convert infix tokens to postfix (Reverse Polish Notation)."""
    output = []
    op_stack = []

    i = 0
    while i < len(tokens):
        ttype, tval = tokens[i]

        if ttype in ('NUM', 'NAME'):
            output.append((ttype, tval))
        elif ttype == 'OP' and tval in _PRECEDENCE:
            # Handle unary minus: if '-' appears at start or after '(' or another operator
            if tval == '-' and (i == 0 or
                    (tokens[i-1][0] == 'OP' and tokens[i-1][1] in '(+-*/')):
                # Unary minus: read next token, negate it
                i += 1
                if i >= len(tokens):
                    raise ValueError("Unexpected end of expression after unary '-'")
                ntype, nval = tokens[i]
                if ntype == 'NUM':
                    output.append(('NUM', -nval))
                elif ntype == 'NAME':
                    output.append(('NUM', -1.0))
                    output.append(('NAME', nval))
                    output.append(('OP', '*'))
                elif ntype == 'OP' and nval == '(':
                    output.append(('NUM', -1.0))
                    op_stack.append(('OP', '*'))
                    op_stack.append(('OP', '('))
                else:
                    raise ValueError(f"Unexpected token after unary '-': {tokens[i]}")
            else:
                while (op_stack and op_stack[-1][1] != '(' and
                       op_stack[-1][1] in _PRECEDENCE and
                       _PRECEDENCE[op_stack[-1][1]] >= _PRECEDENCE[tval]):
                    output.append(op_stack.pop())
                op_stack.append((ttype, tval))
        elif ttype == 'OP' and tval == '(':
            op_stack.append((ttype, tval))
        elif ttype == 'OP' and tval == ')':
            while op_stack and op_stack[-1][1] != '(':
                output.append(op_stack.pop())
            if not op_stack:
                raise ValueError("Mismatched parentheses")
            op_stack.pop()  # remove '('
        i += 1

    while op_stack:
        if op_stack[-1][1] in ('(', ')'):
            raise ValueError("Mismatched parentheses")
        output.append(op_stack.pop())

    return output


def _evaluate_expression(expr: str, phenotypes: PhenotypeArray, n: int) -> np.ndarray:
    """Evaluate an arithmetic expression using phenotype values."""
    tokens = _tokenize(expr)
    rpn = _shunting_yard(tokens)

    stack = []
    for ttype, tval in rpn:
        if ttype == 'NUM':
            stack.append(np.full(n, tval, dtype=np.float64))
        elif ttype == 'NAME':
            if tval not in phenotypes:
                raise ValueError(f"Undefined reference '{tval}' in expression '{expr}'")
            stack.append(phenotypes[tval].copy())
        elif ttype == 'OP':
            if len(stack) < 2:
                raise ValueError(f"Invalid expression: not enough operands for '{tval}'")
            b = stack.pop()
            a = stack.pop()
            if tval == '+':
                stack.append(a + b)
            elif tval == '-':
                stack.append(a - b)
            elif tval == '*':
                stack.append(a * b)
            elif tval == '/':
                stack.append(a / b)

    if len(stack) != 1:
        raise ValueError(f"Invalid expression '{expr}': stack has {len(stack)} values")
    return stack[0]


# ---------------------------------------------------------------------------
# ArchNode
# ---------------------------------------------------------------------------

class _ParentalComponent(ArchComponent):
    """Base class for vertical transmission components.

    Handles gen-0 founder fallback and phenotype_history lookup.
    Subclasses implement ``_extract_values()`` to select the parent
    index(es).

    Parameters
    ----------
    phenotype_name : str
        Name of the phenotype to look up in the previous generation's
        phenotype array.
    founder_component : ArchComponent, optional
        Component to use at generation 0 when no parents exist.
        If None, returns zeros at gen 0.
    normalize : bool
        If True, standardize (mean=0, sd=1) the parental phenotype values
        before indexing by pedigree. This matches the legacy
        ``LinearVerticalComponent(normalize=True)`` behavior, which prevents
        variance from growing unboundedly across generations under VT
        feedback loops. Default False.
    """
    kind = "reference"
    accepts_grouping = False

    def __init__(self, phenotype_name: str,
                 founder_component: ArchComponent | None = None,
                 normalize: bool = False) -> None:
        self.phenotype_name: str = phenotype_name
        self.founder_component: ArchComponent | None = founder_component
        self.normalize: bool = normalize

    def compute(self, node: ArchNode, haplotypes: HaplotypeOperator,
                phenotypes: PhenotypeArray, **kwargs: object) -> np.ndarray:
        generation = kwargs.get('generation', 0)
        phenotype_history = kwargs.get('phenotype_history', {})
        pedigree_history = kwargs.get('pedigree_history', {})
        n = haplotypes.n

        if generation == 0 or generation not in pedigree_history:
            if self.founder_component is not None:
                return self.founder_component.compute(node, haplotypes, phenotypes, **kwargs)
            warnings.warn(
                f"{type(self).__name__}('{self.phenotype_name}'): no pedigree at "
                f"generation {generation}, returning zeros."
            )
            return np.zeros(n, dtype=np.float64)

        prev_gen = generation - 1
        if prev_gen not in phenotype_history:
            warnings.warn(
                f"{type(self).__name__}('{self.phenotype_name}'): generation {prev_gen} "
                f"not in phenotype_history (may be pruned by retention policy), "
                f"returning zeros."
            )
            return np.zeros(n, dtype=np.float64)

        prev_pheno = phenotype_history[prev_gen]
        if self.phenotype_name not in prev_pheno:
            raise ValueError(
                f"{type(self).__name__}: phenotype '{self.phenotype_name}' not found "
                f"in generation {prev_gen}. Available: {list(prev_pheno.keys)}"
            )
        parent_values = prev_pheno[self.phenotype_name]
        ped = pedigree_history[generation]
        result = self._extract_values(parent_values, ped)
        if self.normalize:
            # Normalize AFTER pedigree indexing, matching the legacy
            # LinearVerticalComponent behavior. Under assortative mating,
            # the offspring-weighted distribution of parental values differs
            # from the parental population distribution, so normalizing the
            # mapped values (what offspring actually see) is correct.
            sd = np.std(result)
            if sd > 1e-15:
                result = (result - np.mean(result)) / sd
            else:
                result = result - np.mean(result)
        return result

    @abstractmethod
    def _extract_values(self, parent_values: np.ndarray, ped: PedigreeArray) -> np.ndarray:
        """Select parent values using pedigree indices."""
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}('{self.phenotype_name}')"


class MotherComponent(_ParentalComponent):
    """Vertical transmission: mother's phenotype from previous generation.

    Parameters
    ----------
    phenotype_name : str
        Phenotype to look up in previous generation.
    founder_component : ArchComponent, optional
        Fallback at generation 0.
    normalize : bool
        Standardize parental values before lookup. Default False.
    """
    name = "mother"

    def _extract_values(self, parent_values: np.ndarray, ped: PedigreeArray) -> np.ndarray:
        return parent_values[ped.maternal_idx]


class FatherComponent(_ParentalComponent):
    """Vertical transmission: father's phenotype from previous generation.

    Parameters
    ----------
    phenotype_name : str
        Phenotype to look up in previous generation.
    founder_component : ArchComponent, optional
        Fallback at generation 0.
    normalize : bool
        Standardize parental values before lookup. Default False.
    """
    name = "father"

    def _extract_values(self, parent_values: np.ndarray, ped: PedigreeArray) -> np.ndarray:
        return parent_values[ped.paternal_idx]


class ParentComponent(_ParentalComponent):
    """Vertical transmission: midparent (average of mother and father).

    Parameters
    ----------
    phenotype_name : str
        Phenotype to look up in previous generation.
    founder_component : ArchComponent, optional
        Fallback at generation 0.
    normalize : bool
        Standardize parental values before lookup. Default False.
    """
    name = "parent"

    def _extract_values(self, parent_values: np.ndarray, ped: PedigreeArray) -> np.ndarray:
        return 0.5 * (parent_values[ped.maternal_idx] + parent_values[ped.paternal_idx])


# ---------------------------------------------------------------------------
# Sibling aggregation components
# ---------------------------------------------------------------------------

class _SiblingComponent(ArchComponent):
    """Base class for sibling aggregation components.

    Groups by FID (default) or explicit grouping via the ``|`` operator,
    aggregates a source phenotype within each group, and broadcasts the
    result back to all group members.

    Parameters
    ----------
    source_name : str
        Name of the source phenotype to aggregate (must already be
        computed by the time this node executes).
    """
    kind = "aggregating"
    accepts_grouping = True

    def __init__(self, source_name: str):
        self.source_name = source_name

    def compute(self, node: ArchNode, haplotypes: HaplotypeOperator,
                phenotypes: PhenotypeArray, **kwargs: object) -> np.ndarray:
        if self.source_name not in phenotypes:
            raise ValueError(
                f"{type(self).__name__}: source '{self.source_name}' not found "
                f"in phenotypes. Available: {list(phenotypes.keys)}"
            )
        values = phenotypes[self.source_name]
        labels = _resolve_grouping(node.grouping or 'FID', haplotypes, **kwargs)
        if labels is None:
            # Per-individual: aggregation is trivially itself
            return values.copy()
        return self._aggregate_groups(values, labels)

    @abstractmethod
    def _aggregate_groups(self, values: np.ndarray, labels: np.ndarray) -> np.ndarray:
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}('{self.source_name}')"


class SiblingMeanComponent(_SiblingComponent):
    """Sibling mean: average of source phenotype within group."""
    name = "sibling_mean"

    def _aggregate_groups(self, values: np.ndarray, labels: np.ndarray) -> np.ndarray:
        unique, inverse = np.unique(labels, return_inverse=True)
        sums = np.bincount(inverse, weights=values, minlength=len(unique))
        counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
        means = sums / np.maximum(counts, 1)
        return means[inverse]


class SiblingSumComponent(_SiblingComponent):
    """Sibling sum: sum of source phenotype within group."""
    name = "sibling_sum"

    def _aggregate_groups(self, values: np.ndarray, labels: np.ndarray) -> np.ndarray:
        unique, inverse = np.unique(labels, return_inverse=True)
        sums = np.bincount(inverse, weights=values, minlength=len(unique))
        return sums[inverse]


class SiblingAnyComponent(_SiblingComponent):
    """Sibling any: 1.0 if any member in group has value > 0, else 0.0."""
    name = "sibling_any"

    def _aggregate_groups(self, values: np.ndarray, labels: np.ndarray) -> np.ndarray:
        unique, inverse = np.unique(labels, return_inverse=True)
        # any: max > 0 within group
        pos = (values > 0).astype(np.float64)
        any_vals = np.zeros(len(unique), dtype=np.float64)
        np.maximum.at(any_vals, inverse, pos)
        return any_vals[inverse]


class SiblingCountComponent(_SiblingComponent):
    """Sibling count: number of individuals in each group."""
    name = "sibling_count"

    def _aggregate_groups(self, values: np.ndarray, labels: np.ndarray) -> np.ndarray:
        unique, inverse = np.unique(labels, return_inverse=True)
        counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
        return counts[inverse]


class SiblingEldestComponent(_SiblingComponent):
    """Sibling eldest: value of the first (lowest IID) member in each group."""
    name = "sibling_eldest"

    def _aggregate_groups(self, values: np.ndarray, labels: np.ndarray) -> np.ndarray:
        unique, inverse = np.unique(labels, return_inverse=True)
        # First occurrence in array order (lowest index = eldest)
        eldest_vals = np.empty(len(unique), dtype=np.float64)
        seen = np.zeros(len(unique), dtype=bool)
        for i, g in enumerate(inverse):
            if not seen[g]:
                eldest_vals[g] = values[i]
                seen[g] = True
        return eldest_vals[inverse]


class SiblingYoungestComponent(_SiblingComponent):
    """Sibling youngest: value of the last (highest IID) member in each group."""
    name = "sibling_youngest"

    def _aggregate_groups(self, values: np.ndarray, labels: np.ndarray) -> np.ndarray:
        unique, inverse = np.unique(labels, return_inverse=True)
        # Last occurrence in array order (highest index = youngest)
        youngest_vals = np.empty(len(unique), dtype=np.float64)
        for i, g in enumerate(inverse):
            youngest_vals[g] = values[i]
        return youngest_vals[inverse]


_SIBLING_COMPONENTS = {
    'sibling_mean': SiblingMeanComponent,
    'sibling_sum': SiblingSumComponent,
    'sibling_any': SiblingAnyComponent,
    'sibling_count': SiblingCountComponent,
    'sibling_eldest': SiblingEldestComponent,
    'sibling_youngest': SiblingYoungestComponent,
}


@dataclass
class ArchNode:
    """
    A single node in the architecture DAG.

    Parameters
    ----------
    outputs : list[str]
        Names written to PhenotypeArray.
    component : ArchComponent
        The computation to perform.
    inputs : list[str]
        Names read from PhenotypeArray (for aggregation) or [] (for generative).
    grouping : str or None
        Grouping variable for | operator, or None (implicit | IID).
    """
    outputs: list[str]
    component: ArchComponent
    inputs: list[str] = field(default_factory=list)
    grouping: str | None = None

    def __repr__(self) -> str:
        return (f"ArchNode(outputs={self.outputs}, component={self.component}, "
                f"inputs={self.inputs}, grouping={self.grouping})")


# ---------------------------------------------------------------------------
# BUILTINS registry
# ---------------------------------------------------------------------------

BUILTINS: dict[str, type[ArchComponent]] = {
    'genetic': GeneticComponent,
    'mvGenetic': MVGeneticComponent,
    'haplotypeGenetic': HaplotypeGeneticComponent,
    'noise': NoiseComponent,
    'cnoise': CNoiseComponent,
    'threshold': ThresholdComponent,
    'parent': ParentComponent,
    'mother': MotherComponent,
    'father': FatherComponent,
    **_SIBLING_COMPONENTS,
}
"""Registry mapping DSL function names to ArchComponent subclasses.

Used by the formula parser to resolve function calls like ``genetic(eff)``
to the corresponding component class.
"""


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class Architecture:
    """Phenogenetic architecture: a DAG of ArchNodes executed each generation.

    Can be constructed programmatically via ``add()`` or from a formula
    string (parsed by the ``parser`` module). Nodes are topologically
    sorted so that dependencies are resolved before dependents.

    Parameters
    ----------
    formula : str, optional
        Multi-line formula string (parsed into ArchNodes). See
        ``parser.parse_formula`` for the grammar.
    effects : dict, optional
        Name -> EffectSpec mapping for resolving effect references
        in the formula.

    Examples
    --------
    Programmatic construction:

    >>> from xftsim.effect import AdditiveEffects
    >>> eff = AdditiveEffects.from_h2(h2=0.5, m=100, seed=1)
    >>> arch = Architecture()
    >>> arch.add('Y.G', GeneticComponent(eff))
    >>> arch.add('Y.E', NoiseComponent(0.5))
    >>> arch.add('Y', AggregationComponent('Y.G + Y.E'))

    Formula construction:

    >>> arch = Architecture(
    ...     formula=\"\"\"
    ...     Y.G ~ genetic(eff)
    ...     Y.E ~ noise(0.5)
    ...     Y ~ Y.G + Y.E
    ...     \"\"\",
    ...     effects={'eff': eff},
    ... )
    """

    def __init__(self, formula: str | None = None,
                 effects: dict[str, EffectSpec] | None = None) -> None:
        self._nodes: list[ArchNode] = []
        self._sorted: list[ArchNode] | None = None
        self._output_map: dict[str, ArchNode] = {}

        if formula is not None:
            from xftsim.parser import parse_formula
            nodes = parse_formula(formula, effects or {})
            for node in nodes:
                self._register_node(node)
            self._sorted = self._toposort()

    @classmethod
    def from_formula(cls, formula: str,
                     effects: dict[str, EffectSpec] | None = None) -> Architecture:
        """Construct an Architecture from a DSL formula string.

        Parameters
        ----------
        formula : str
            Multi-line formula string (see parser module for grammar).
        effects : dict, optional
            Name → EffectSpec mapping for resolving effect references.

        Returns
        -------
        Architecture
        """
        return cls(formula=formula, effects=effects)

    def add(self, outputs: str | list[str], component: ArchComponent,
            inputs: list[str] | None = None, grouping: str | None = None) -> None:
        """
        Programmatically add a node to the architecture.

        Parameters
        ----------
        outputs : str or list[str]
            Output name(s).
        component : ArchComponent
            The component to execute.
        inputs : list[str], optional
            Input names (for aggregation). Auto-detected for AggregationComponent.
        grouping : str, optional
            Grouping variable.
        """
        if isinstance(outputs, str):
            outputs = [outputs]

        if inputs is None:
            if isinstance(component, AggregationComponent):
                inputs = component._input_names
            else:
                inputs = []

        node = ArchNode(
            outputs=outputs,
            component=component,
            inputs=inputs,
            grouping=grouping,
        )
        self._register_node(node)
        self._sorted = None  # invalidate cache

    def _register_node(self, node: ArchNode) -> None:
        """Register a node, checking for duplicate outputs."""
        for out in node.outputs:
            if out in self._output_map:
                raise ValueError(f"Duplicate output name '{out}'")
            self._output_map[out] = node
        self._nodes.append(node)

    @property
    def nodes(self) -> list[ArchNode]:
        """Return the topologically sorted node list."""
        if self._sorted is None:
            self._sorted = self._toposort()
        return self._sorted

    def _toposort(self) -> list[ArchNode]:
        """
        Topological sort via Kahn's algorithm.
        Validates no cycles and no undefined references.
        """
        # Build adjacency: node → set of nodes it depends on
        node_set = set(id(n) for n in self._nodes)
        id_to_node = {id(n): n for n in self._nodes}

        # Map output name → node
        output_to_node = {}
        for n in self._nodes:
            for out in n.outputs:
                output_to_node[out] = n

        # Check for undefined references
        all_outputs = set(output_to_node.keys())
        for n in self._nodes:
            for inp in n.inputs:
                if inp not in all_outputs:
                    raise ValueError(f"Undefined reference '{inp}' in node {n.outputs}")

        # Build in-degree map
        in_edges = {id(n): set() for n in self._nodes}
        for n in self._nodes:
            for inp in n.inputs:
                dep_node = output_to_node[inp]
                if id(dep_node) != id(n):
                    in_edges[id(n)].add(id(dep_node))

        # Kahn's algorithm
        in_degree = {nid: len(deps) for nid, deps in in_edges.items()}
        queue = [nid for nid, deg in in_degree.items() if deg == 0]
        sorted_ids = []

        while queue:
            nid = queue.pop(0)
            sorted_ids.append(nid)
            # For each node that depends on nid, decrement in-degree
            for other_id, deps in in_edges.items():
                if nid in deps:
                    in_degree[other_id] -= 1
                    if in_degree[other_id] == 0:
                        queue.append(other_id)

        if len(sorted_ids) != len(self._nodes):
            raise ValueError("Cycle detected in architecture DAG")

        return [id_to_node[nid] for nid in sorted_ids]

    def compute(self, haplotypes: HaplotypeOperator,
                phenotypes: PhenotypeArray | None = None,
                rng: np.random.RandomState | None = None,
                **kwargs: object) -> PhenotypeArray:
        """
        Execute all nodes in topological order.

        Parameters
        ----------
        haplotypes : HaplotypeOperator
            Current generation's haplotype data.
        phenotypes : PhenotypeArray, optional
            Existing phenotype array to write into. Created if None.
        rng : np.random.RandomState, optional
            Random state for noise components.
        **kwargs
            Additional context (phenotype_history, pedigree_history, generation).

        Returns
        -------
        PhenotypeArray
            The phenotype array with all computed values.
        """
        if phenotypes is None:
            phenotypes = PhenotypeArray(samples=haplotypes.samples)

        if rng is None:
            rng = np.random.RandomState()

        for node in self.nodes:
            result = node.component.compute(
                node, haplotypes, phenotypes, rng=rng, **kwargs
            )
            # Write result to phenotype array
            if len(node.outputs) == 1:
                phenotypes._values[node.outputs[0]] = np.asarray(result, dtype=np.float64)
            else:
                # Multi-output node (e.g. mvGenetic)
                for i, out_name in enumerate(node.outputs):
                    phenotypes._values[out_name] = np.asarray(result[:, i], dtype=np.float64)

        return phenotypes

    def __repr__(self) -> str:
        n_nodes = len(self._nodes)
        outputs = [out for n in self._nodes for out in n.outputs]
        return f"Architecture(nodes={n_nodes}, outputs={outputs})"
