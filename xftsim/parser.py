"""
Formula parser for xftsim architecture DSL.

Parses formula strings into a list of ArchNode objects.

Grammar:
- LHS ~ function(args)             e.g. height.G ~ genetic(eff)
- LHS ~ function(args) | GROUPING  e.g. height.E ~ noise(0.3) | FID
- LHS ~ arithmetic_expression      e.g. height ~ height.G + height.E
- (a, b) ~ mvGenetic(eff)          tuple LHS for multi-output
- (a, b) ~ cnoise(cov=[[1,0.2],[0.2,1]])  multivariate correlated noise
- Functions: genetic, mvGenetic, noise, cnoise, parent, mother, father
"""
from __future__ import annotations

import ast
import re
from typing import Optional

from xftsim.narch import (
    ArchNode, ArchComponent, GeneticComponent, MVGeneticComponent,
    HaplotypeGeneticComponent,
    NoiseComponent, CNoiseComponent, ThresholdComponent, AggregationComponent,
    MotherComponent, FatherComponent, ParentComponent,
    _SIBLING_COMPONENTS,
    BUILTINS,
)
from xftsim.neffect import EffectSpec


def parse_formula(formula: str,
                  effects: dict[str, EffectSpec] | None = None) -> list[ArchNode]:
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

        # Parse LHS: tuple syntax (a, b) or single name
        if lhs.startswith('(') and ')' in lhs:
            inner = lhs[1:lhs.index(')')].strip()
            outputs = [s.strip() for s in inner.split(',') if s.strip()]
            if len(outputs) < 2:
                raise ValueError(f"Line {lineno}: tuple LHS requires at least 2 names")
        else:
            outputs = [lhs]

        for out in outputs:
            if out in seen_outputs:
                raise ValueError(f"Line {lineno}: duplicate output name '{out}'")
            seen_outputs.add(out)

        # Extract trailing | GROUPING from RHS (but not inside parens)
        grouping = None
        rhs, grouping = _extract_grouping(rhs)

        # Try to parse RHS as a function call
        node = _try_parse_function(outputs, rhs, effects, lineno, grouping)
        if node is None:
            if grouping is not None:
                raise ValueError(
                    f"Line {lineno}: | grouping is only valid on function calls, "
                    f"not aggregation expressions"
                )
            # Otherwise treat as aggregation expression
            node = _parse_aggregation(outputs, rhs, lineno)

        nodes.append(node)

    return nodes


def _extract_grouping(rhs: str) -> tuple[str, str | None]:
    """
    Extract trailing | IDENTIFIER from RHS, respecting parentheses.

    Returns (rhs_without_pipe, grouping_str_or_None).
    """
    # Find the last | that is NOT inside parentheses
    depth = 0
    last_pipe = -1
    for i, ch in enumerate(rhs):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == '|' and depth == 0:
            last_pipe = i

    if last_pipe == -1:
        return rhs, None

    grouping = rhs[last_pipe + 1:].strip()
    rhs_part = rhs[:last_pipe].strip()
    if not grouping:
        return rhs_part, None
    # Validate grouping is a simple identifier
    if not re.match(r'^[A-Za-z_]\w*$', grouping):
        raise ValueError(f"Invalid grouping variable: '{grouping}'")
    return rhs_part, grouping


# Regex for function calls: name(args)
_FUNC_RE = re.compile(
    r'^([A-Za-z_]\w*)\s*\(\s*(.*?)\s*\)$',
    re.DOTALL
)


def _try_parse_function(outputs: list[str], rhs: str,
                        effects: dict[str, EffectSpec],
                        lineno: int,
                        grouping: str | None = None) -> ArchNode | None:
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

    # Validate grouping
    if grouping is not None:
        cls = BUILTINS[func_name]
        if not cls.accepts_grouping:
            raise ValueError(
                f"Line {lineno}: {func_name} does not accept | grouping"
            )

    if func_name == 'genetic':
        return _parse_genetic(outputs, args_str, effects, lineno, grouping)
    elif func_name == 'mvGenetic':
        return _parse_mvGenetic(outputs, args_str, effects, lineno, grouping)
    elif func_name == 'haplotypeGenetic':
        return _parse_haplotypeGenetic(outputs, args_str, effects, lineno, grouping)
    elif func_name == 'noise':
        return _parse_noise(outputs, args_str, lineno, grouping)
    elif func_name == 'threshold':
        return _parse_threshold(outputs, args_str, lineno)
    elif func_name == 'cnoise':
        return _parse_cnoise(outputs, args_str, lineno, grouping)
    elif func_name in ('parent', 'mother', 'father'):
        return _parse_parental(func_name, outputs, args_str, effects, lineno)
    elif func_name in _SIBLING_COMPONENTS:
        return _parse_sibling(func_name, outputs, args_str, lineno, grouping)
    else:
        raise ValueError(f"Line {lineno}: unhandled function '{func_name}'")


def _parse_genetic(outputs: list[str], args_str: str,
                   effects: dict[str, EffectSpec],
                   lineno: int, grouping: str | None = None) -> ArchNode:
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
    return ArchNode(outputs=outputs, component=component, inputs=[], grouping=grouping)


def _parse_mvGenetic(outputs: list[str], args_str: str,
                     effects: dict[str, EffectSpec],
                     lineno: int, grouping: str | None = None) -> ArchNode:
    """Parse mvGenetic(effect_name) → MVGeneticComponent."""
    effect_name = args_str.strip()
    if not effect_name:
        raise ValueError(f"Line {lineno}: mvGenetic() requires an effect name")
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
    if effect.k != len(outputs):
        raise ValueError(
            f"Line {lineno}: mvGenetic effect has k={effect.k} but "
            f"LHS has {len(outputs)} outputs"
        )
    component = MVGeneticComponent(effects=effect)
    return ArchNode(outputs=outputs, component=component, inputs=[], grouping=grouping)


def _parse_haplotypeGenetic(outputs: list[str], args_str: str,
                            effects: dict[str, EffectSpec],
                            lineno: int, grouping: str | None = None) -> ArchNode:
    """Parse haplotypeGenetic(eff) or haplotypeGenetic(eff, haplotype='maternal')."""
    # Split args on comma, respecting that haplotype= value has quotes
    parts = [p.strip() for p in args_str.split(',')]
    effect_name = parts[0].strip()
    haplotype = 'maternal'  # default

    for part in parts[1:]:
        part = part.strip()
        if part.startswith('haplotype='):
            val = part[len('haplotype='):].strip().strip("'\"")
            haplotype = val
        elif part:
            raise ValueError(
                f"Line {lineno}: unexpected argument '{part}' in haplotypeGenetic()"
            )

    if not effect_name:
        raise ValueError(f"Line {lineno}: haplotypeGenetic() requires an effect name")
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
    component = HaplotypeGeneticComponent(effects=effect, haplotype=haplotype)
    return ArchNode(outputs=outputs, component=component, inputs=[], grouping=grouping)


def _parse_noise(outputs: list[str], args_str: str, lineno: int,
                 grouping: str | None = None) -> ArchNode:
    """Parse noise(variance) → NoiseComponent."""
    try:
        variance = float(args_str)
    except ValueError:
        raise ValueError(
            f"Line {lineno}: noise() requires a numeric variance, got '{args_str}'"
        )
    component = NoiseComponent(variance=variance)
    return ArchNode(outputs=outputs, component=component, inputs=[], grouping=grouping)


def _parse_threshold(outputs: list[str], args_str: str,
                     lineno: int) -> ArchNode:
    """Parse threshold(source, value) → ThresholdComponent."""
    parts = [p.strip() for p in args_str.split(',')]
    if len(parts) != 2:
        raise ValueError(
            f"Line {lineno}: threshold() requires 2 arguments: "
            f"threshold(source, value), got {len(parts)}"
        )
    source = parts[0].strip()
    if not source:
        raise ValueError(f"Line {lineno}: threshold() requires a source name")
    try:
        thresh_val = float(parts[1])
    except ValueError:
        raise ValueError(
            f"Line {lineno}: threshold() second argument must be numeric, "
            f"got '{parts[1]}'"
        )
    component = ThresholdComponent(source=source, threshold=thresh_val)
    return ArchNode(outputs=outputs, component=component,
                    inputs=[source], grouping=None)


def _parse_cnoise(outputs: list[str], args_str: str, lineno: int,
                  grouping: str | None = None) -> ArchNode:
    """Parse cnoise(cov=[[...]]) → CNoiseComponent."""
    import numpy as np

    # Parse cov= kwarg
    args_str = args_str.strip()
    if args_str.startswith('cov='):
        matrix_str = args_str[4:].strip()
    else:
        matrix_str = args_str

    try:
        cov_list = ast.literal_eval(matrix_str)
    except (ValueError, SyntaxError):
        raise ValueError(
            f"Line {lineno}: cnoise() requires a matrix literal, got '{args_str}'"
        )
    cov = np.asarray(cov_list, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(
            f"Line {lineno}: cnoise() cov must be a square matrix, got shape {cov.shape}"
        )
    if len(outputs) != cov.shape[0]:
        raise ValueError(
            f"Line {lineno}: cnoise cov has k={cov.shape[0]} but "
            f"LHS has {len(outputs)} outputs"
        )
    component = CNoiseComponent(cov=cov)
    return ArchNode(outputs=outputs, component=component, inputs=[], grouping=grouping)


def _parse_parental(func_name: str, outputs: list[str], args_str: str,
                    effects: dict[str, EffectSpec], lineno: int) -> ArchNode:
    """Parse parent(phenotype, founder=..., normalize=true), mother(...), father(...).

    Optional kwargs:
        founder=noise(var)  — fallback component at generation 0
        normalize=true      — standardize parental values before lookup
                              (matches legacy LinearVerticalComponent behavior)
    """
    args_str = args_str.strip()
    founder_component = None
    normalize = False

    # Extract normalize= kwarg
    norm_match = re.search(r',\s*normalize\s*=\s*(true|false|True|False|1|0)', args_str)
    if norm_match:
        normalize = norm_match.group(1).lower() in ('true', '1')
        args_str = args_str[:norm_match.start()] + args_str[norm_match.end():]

    # Extract founder= kwarg
    founder_match = re.search(r',\s*founder\s*=\s*(.+)$', args_str)
    if founder_match:
        founder_str = founder_match.group(1).strip()
        phenotype_name = args_str[:founder_match.start()].strip()
        founder_component = _parse_founder_component(founder_str, lineno)
    else:
        phenotype_name = args_str.strip()

    if not phenotype_name:
        raise ValueError(
            f"Line {lineno}: {func_name}() requires a phenotype name"
        )

    comp_map = {
        'parent': ParentComponent,
        'mother': MotherComponent,
        'father': FatherComponent,
    }
    component = comp_map[func_name](
        phenotype_name, founder_component=founder_component, normalize=normalize,
    )
    return ArchNode(outputs=outputs, component=component, inputs=[], grouping=None)


def _parse_founder_component(founder_str: str, lineno: int) -> ArchComponent:
    """Parse a founder= value like noise(0.3) into a component."""
    match = _FUNC_RE.match(founder_str)
    if not match:
        raise ValueError(
            f"Line {lineno}: founder= requires a function call, got '{founder_str}'"
        )
    func_name = match.group(1)
    args_str = match.group(2).strip()

    if func_name == 'noise':
        try:
            variance = float(args_str)
        except ValueError:
            raise ValueError(
                f"Line {lineno}: noise() in founder= requires numeric variance, "
                f"got '{args_str}'"
            )
        return NoiseComponent(variance=variance)
    else:
        raise ValueError(
            f"Line {lineno}: unsupported function '{func_name}' in founder= "
            f"(currently only noise is supported)"
        )


def _parse_sibling(func_name: str, outputs: list[str], args_str: str,
                   lineno: int, grouping: str | None = None) -> ArchNode:
    """Parse sibling_mean(source_name), etc."""
    source_name = args_str.strip()
    if not source_name:
        raise ValueError(
            f"Line {lineno}: {func_name}() requires a source component name"
        )
    cls = _SIBLING_COMPONENTS[func_name]
    component = cls(source_name)
    return ArchNode(
        outputs=outputs, component=component,
        inputs=[source_name], grouping=grouping,
    )


def _parse_aggregation(outputs: list[str], rhs: str, lineno: int) -> ArchNode:
    """Parse an arithmetic expression → AggregationComponent."""
    component = AggregationComponent(expression=rhs)
    inputs = component._input_names
    return ArchNode(outputs=outputs, component=component, inputs=inputs, grouping=None)
