"""Shared helpers for the example-notebook integration tests.

Each test module here exercises one notebook in ``docs/examples/``: it
runs the notebook's code cells in a fresh namespace, then asserts that
the resulting variables are within tolerance of the expected scientific
targets.

The runner deliberately does NOT use a Jupyter kernel — the example
notebooks are all plain numpy/pandas code without magics or rich
display, so a direct ``exec`` is faster and gives us direct access to
the resulting globals for downstream assertions.
"""
from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest


EXAMPLES_DIR = pathlib.Path(__file__).resolve().parents[2] / "docs" / "examples"


def run_notebook(path: str | pathlib.Path) -> dict[str, Any]:
    """Execute the code cells of a notebook in a fresh namespace.

    Parameters
    ----------
    path : str or Path
        Path to the .ipynb file.

    Returns
    -------
    dict[str, Any]
        The post-execution module namespace.
    """
    path = pathlib.Path(path)
    with path.open() as f:
        nb = json.load(f)
    ns: dict[str, Any] = {"__name__": "__main__", "__file__": str(path)}
    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        code = compile(src, f"{path.name}#cell-{i}", "exec")
        exec(code, ns)
    return ns


@pytest.fixture(scope="module")
def examples_dir() -> pathlib.Path:
    """Path to the docs/examples notebook directory."""
    return EXAMPLES_DIR
