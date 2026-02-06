"""
Minimal formula parser for Phase 1.

Parses formula strings into a list of ArchNode objects.

Phase 1 grammar handles:
- LHS ~ function(args)         e.g. height.G ~ genetic(eff)
- LHS ~ arithmetic_expression  e.g. height ~ height.G + height.E
- Functions: genetic(effect_name), noise(variance)
- Scalar multiplication: height ~ 0.3 * height.G + height.E

NOT handled in Phase 1:
- | grouping operator
- founder= kwarg
- tuple LHS (multivariate output)
- cnoise
- sibling references
"""
import re
from typing import Optional

from xftsim.narch import (
    ArchNode, ArchComponent, GeneticComponent, NoiseComponent,
    AggregationComponent, BUILTINS,
)
from xftsim.neffect import EffectSpec


def parse_formula(formula: str, effects: dict = None) -> list:
    """
    Parse a formula string into a list of ArchNode objects.

    Parameters
    ----------
    formula : str
        Multi-line formula string. Each line is one statement: LHS ~ RHS.
        Lines starting with # are comments. Empty lines are skipped.
    effects : dict, optional
        Name → EffectSpec mapping for resolving effect references.

    Returns
    -------
    list[ArchNode]
        Parsed nodes in declaration order.

    Raises
    ------
    ValueError
        On parse errors (unknown function, missing effect, etc.).
    """
    if effects is None:
        effects = {}

    nodes = []
    seen_outputs = set()

    lines = formula.strip().split('\n')
    for lineno, raw_line in enumerate(lines, 1):
        line = raw_line.strip()
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            continue

        # Split on ~
        if '~' not in line:
            raise ValueError(f"Line {lineno}: missing '~' in '{line}'")

        lhs, rhs = line.split('~', 1)
        lhs = lhs.strip()
        rhs = rhs.strip()

        if not lhs:
            raise ValueError(f"Line {lineno}: missing LHS in '{line}'")
        if not rhs:
            raise ValueError(f"Line {lineno}: missing RHS in '{line}'")

        # Check for duplicate outputs
        outputs = [lhs]
        for out in outputs:
            if out in seen_outputs:
                raise ValueError(f"Line {lineno}: duplicate output name '{out}'")
            seen_outputs.add(out)

        # Try to parse RHS as a function call
        node = _try_parse_function(outputs, rhs, effects, lineno)
        if node is None:
            # Otherwise treat as aggregation expression
            node = _parse_aggregation(outputs, rhs, lineno)

        nodes.append(node)

    return nodes


# Regex for function calls: name(args)
_FUNC_RE = re.compile(
    r'^([A-Za-z_]\w*)\s*\(\s*(.*?)\s*\)$',
    re.DOTALL
)


def _try_parse_function(outputs: list, rhs: str, effects: dict,
                        lineno: int) -> Optional[ArchNode]:
    """
    Try to parse RHS as a function call.

    Returns ArchNode if successful, None if RHS is not a function call.
    """
    match = _FUNC_RE.match(rhs)
    if not match:
        return None

    func_name = match.group(1)
    args_str = match.group(2).strip()

    if func_name not in BUILTINS:
        raise ValueError(
            f"Line {lineno}: unknown function '{func_name}'. "
            f"Available: {list(BUILTINS.keys())}"
        )

    if func_name == 'genetic':
        return _parse_genetic(outputs, args_str, effects, lineno)
    elif func_name == 'noise':
        return _parse_noise(outputs, args_str, lineno)
    else:
        raise ValueError(f"Line {lineno}: unhandled function '{func_name}'")


def _parse_genetic(outputs: list, args_str: str, effects: dict,
                   lineno: int) -> ArchNode:
    """Parse genetic(effect_name) → GeneticComponent."""
    effect_name = args_str.strip()
    if not effect_name:
        raise ValueError(f"Line {lineno}: genetic() requires an effect name")
    if effect_name not in effects:
        raise ValueError(
            f"Line {lineno}: effect '{effect_name}' not found in effects dict. "
            f"Available: {list(effects.keys())}"
        )
    effect = effects[effect_name]
    if not isinstance(effect, EffectSpec):
        raise ValueError(
            f"Line {lineno}: effects['{effect_name}'] is not an EffectSpec"
        )
    component = GeneticComponent(effects=effect)
    return ArchNode(outputs=outputs, component=component, inputs=[], grouping=None)


def _parse_noise(outputs: list, args_str: str, lineno: int) -> ArchNode:
    """Parse noise(variance) → NoiseComponent."""
    try:
        variance = float(args_str)
    except ValueError:
        raise ValueError(
            f"Line {lineno}: noise() requires a numeric variance, got '{args_str}'"
        )
    component = NoiseComponent(variance=variance)
    return ArchNode(outputs=outputs, component=component, inputs=[], grouping=None)


def _parse_aggregation(outputs: list, rhs: str, lineno: int) -> ArchNode:
    """Parse an arithmetic expression → AggregationComponent."""
    component = AggregationComponent(expression=rhs)
    inputs = component._input_names
    return ArchNode(outputs=outputs, component=component, inputs=inputs, grouping=None)
