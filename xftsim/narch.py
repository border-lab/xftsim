"""
Architecture system for the new xftsim design.

ArchComponent ABC, concrete components, ArchNode, and Architecture class.
Supports both programmatic construction (arch.add()) and formula parsing.
"""
import re
import warnings
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union
from collections import OrderedDict

from xftsim.struct import HaplotypeOperator, NPhenotypeArray
from xftsim.neffect import EffectSpec


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
                phenotypes: NPhenotypeArray, **kwargs) -> np.ndarray:
        """
        Execute this component and return the result array.

        Parameters
        ----------
        node : ArchNode
            The node being executed (provides inputs, outputs, grouping).
        haplotypes : HaplotypeOperator
            Current generation's haplotype data.
        phenotypes : NPhenotypeArray
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
    """
    Genetic component: computes G @ effects (or standardized_matvec).
    """
    name = "genetic"
    kind = "genetic"
    accepts_grouping = False

    def __init__(self, effects: EffectSpec):
        self.effects = effects

    def compute(self, node, haplotypes, phenotypes, **kwargs):
        if self.effects.standardized:
            return haplotypes.standardized_matvec(self.effects.effects)
        else:
            return haplotypes.matvec(self.effects.effects)

    def __repr__(self):
        return f"GeneticComponent(effects={self.effects})"


class NoiseComponent(ArchComponent):
    """
    Noise component: draws iid N(0, variance) per individual.
    """
    name = "noise"
    kind = "generative"
    accepts_grouping = True

    def __init__(self, variance: float):
        self.variance = float(variance)

    def compute(self, node, haplotypes, phenotypes, **kwargs):
        n = haplotypes.n
        rng = kwargs.get('rng', np.random.RandomState())
        return rng.normal(0, np.sqrt(self.variance), size=n)

    def __repr__(self):
        return f"NoiseComponent(variance={self.variance})"


class AggregationComponent(ArchComponent):
    """
    Aggregation component: evaluates arithmetic expressions over phenotype values.

    Uses a custom tokenizer + shunting-yard evaluator (no eval()).
    Supports: +, -, *, /, scalar multiplication, dotted names.
    """
    name = "aggregation"
    kind = "aggregating"
    accepts_grouping = False

    def __init__(self, expression: str):
        self.expression = expression
        # Extract input names (dotted identifiers, not numbers)
        self._input_names = self._extract_names(expression)

    @staticmethod
    def _extract_names(expr: str) -> list:
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

    def compute(self, node, haplotypes, phenotypes, **kwargs):
        n = haplotypes.n
        result = _evaluate_expression(self.expression, phenotypes, n)
        return result

    def __repr__(self):
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


def _tokenize(expr: str) -> list:
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


def _shunting_yard(tokens: list) -> list:
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


def _evaluate_expression(expr: str, phenotypes: NPhenotypeArray, n: int) -> np.ndarray:
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
    outputs: list
    component: ArchComponent
    inputs: list = field(default_factory=list)
    grouping: Optional[str] = None

    def __repr__(self):
        return (f"ArchNode(outputs={self.outputs}, component={self.component}, "
                f"inputs={self.inputs}, grouping={self.grouping})")


# ---------------------------------------------------------------------------
# BUILTINS registry
# ---------------------------------------------------------------------------

BUILTINS = {
    'genetic': GeneticComponent,
    'noise': NoiseComponent,
}


# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------

class Architecture:
    """
    Phenogenetic architecture: a DAG of ArchNodes executed each generation.

    Can be constructed programmatically via add() or from a formula string.

    Parameters
    ----------
    formula : str, optional
        Formula string (parsed into ArchNodes).
    effects : dict, optional
        Name → EffectSpec mapping for formula resolution.
    """

    def __init__(self, formula: str = None, effects: dict = None):
        self._nodes: list[ArchNode] = []
        self._sorted: list[ArchNode] = None
        self._output_map: dict[str, ArchNode] = {}

        if formula is not None:
            from xftsim.parser import parse_formula
            nodes = parse_formula(formula, effects or {})
            for node in nodes:
                self._register_node(node)
            self._sorted = self._toposort()

    def add(self, outputs: Union[str, list], component: ArchComponent,
            inputs: list = None, grouping: str = None):
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

    def _register_node(self, node: ArchNode):
        """Register a node, checking for duplicate outputs."""
        for out in node.outputs:
            if out in self._output_map:
                raise ValueError(f"Duplicate output name '{out}'")
            self._output_map[out] = node
        self._nodes.append(node)

    @property
    def nodes(self) -> list:
        """Return the topologically sorted node list."""
        if self._sorted is None:
            self._sorted = self._toposort()
        return self._sorted

    def _toposort(self) -> list:
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
                phenotypes: NPhenotypeArray = None,
                rng: np.random.RandomState = None,
                **kwargs) -> NPhenotypeArray:
        """
        Execute all nodes in topological order.

        Parameters
        ----------
        haplotypes : HaplotypeOperator
            Current generation's haplotype data.
        phenotypes : NPhenotypeArray, optional
            Existing phenotype array to write into. Created if None.
        rng : np.random.RandomState, optional
            Random state for noise components.
        **kwargs
            Additional context (phenotype_history, pedigree_history, generation).

        Returns
        -------
        NPhenotypeArray
            The phenotype array with all computed values.
        """
        if phenotypes is None:
            phenotypes = NPhenotypeArray(samples=haplotypes.samples)

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

    def __repr__(self):
        n_nodes = len(self._nodes)
        outputs = [out for n in self._nodes for out in n.outputs]
        return f"Architecture(nodes={n_nodes}, outputs={outputs})"
