"""
Command-line interface for xftsim.

Provides commands for running simulations, resuming from checkpoints,
inspecting checkpoint data, and running built-in demos.

Usage:
    xftsim run config.yaml
    xftsim resume checkpoint_dir/
    xftsim info checkpoint_dir/
    xftsim demo UGRM

To install as a console script, add to setup.py / pyproject.toml:
    [project.scripts]
    xftsim = "xftsim.cli:app"
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import typer

app = typer.Typer(
    name="xftsim",
    help="xftsim: forward-time genetics simulator",
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Output mode helpers
# ---------------------------------------------------------------------------

def _detect_output_mode(plain: bool = False, rich_flag: bool = False) -> str:
    """Determine output mode: 'rich' or 'plain'.

    Parameters
    ----------
    plain : bool
        Force plain text output.
    rich_flag : bool
        Force rich output even when piped.

    Returns
    -------
    str
        'rich' or 'plain'
    """
    if plain:
        return "plain"
    if rich_flag:
        return "rich"
    if sys.stdout.isatty():
        return "rich"
    return "plain"


class _Output:
    """Unified output helper that dispatches to rich or plain text."""

    def __init__(self, mode: str = "plain", quiet: bool = False,
                 verbose: bool = False):
        self.mode = mode
        self.quiet = quiet
        self.verbose = verbose
        self._console = None
        if mode == "rich":
            try:
                from rich.console import Console
                self._console = Console()
            except ImportError:
                self.mode = "plain"

    @property
    def console(self):
        return self._console

    def info(self, msg: str):
        if self.quiet:
            return
        if self.mode == "rich" and self._console:
            self._console.print(msg)
        else:
            print(msg, flush=True)

    def error(self, msg: str):
        if self.mode == "rich" and self._console:
            self._console.print(f"[bold red]Error:[/bold red] {msg}")
        else:
            print(f"Error: {msg}", file=sys.stderr, flush=True)

    def debug(self, msg: str):
        if not self.verbose or self.quiet:
            return
        if self.mode == "rich" and self._console:
            self._console.print(f"[dim]{msg}[/dim]")
        else:
            print(msg, flush=True)

    def generation_line(self, gen: int, total: int, elapsed: float,
                        var_y: Optional[float] = None):
        """Print a per-generation status line (plain mode)."""
        if self.quiet:
            return
        parts = [f"[gen {gen}/{total}]", f"elapsed={elapsed:.2f}s"]
        if var_y is not None:
            parts.append(f"var(Y)={var_y:.4f}")
        print(" ".join(parts), flush=True)

    def summary_table(self, rows: list, headers: list):
        """Print a summary table."""
        if self.quiet:
            return
        if self.mode == "rich" and self._console:
            from rich.table import Table
            table = Table(title="Simulation Summary")
            for h in headers:
                table.add_column(h)
            for row in rows:
                table.add_row(*[str(x) for x in row])
            self._console.print(table)
        else:
            # Plain text table
            col_widths = [max(len(str(h)), max((len(str(r[i])) for r in rows),
                          default=0)) for i, h in enumerate(headers)]
            header_line = "  ".join(
                str(h).ljust(w) for h, w in zip(headers, col_widths)
            )
            sep = "  ".join("-" * w for w in col_widths)
            print(header_line, flush=True)
            print(sep, flush=True)
            for row in rows:
                print("  ".join(
                    str(x).ljust(w) for x, w in zip(row, col_widths)
                ), flush=True)


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def _load_config_file(path: str) -> dict:
    """Load a YAML or JSON config file.

    Parameters
    ----------
    path : str
        Path to config file (.yaml, .yml, or .json).

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    typer.BadParameter
        If the file cannot be loaded.
    """
    path = str(path)
    if not os.path.isfile(path):
        raise typer.BadParameter(f"Config file not found: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise typer.BadParameter(
                "pyyaml is required for YAML config files. "
                "Install with: pip install pyyaml"
            )
        with open(path, "r") as f:
            config = yaml.safe_load(f)
    elif ext == ".json":
        with open(path, "r") as f:
            config = json.load(f)
    else:
        # Try YAML first, fallback to JSON
        try:
            import yaml
            with open(path, "r") as f:
                config = yaml.safe_load(f)
        except Exception:
            with open(path, "r") as f:
                config = json.load(f)

    if not isinstance(config, dict):
        raise typer.BadParameter(f"Config file must be a YAML/JSON dict, got {type(config).__name__}")

    return config


def build_simulation_from_config(config: dict):
    """Build an Simulation from a parsed config dict.

    Parameters
    ----------
    config : dict
        Configuration dictionary with keys: founder, effects, architecture,
        mating, simulation. Optional: recombination, statistics, filters, output.

    Returns
    -------
    tuple[Simulation, dict]
        The configured simulation and output settings dict.

    Raises
    ------
    ValueError
        On invalid configuration.
    """
    from xftsim.founders import founder_haplotypes_uniform_AFs
    from xftsim.effect import AdditiveEffects, MultivariateEffects, SparseEffects
    from xftsim.arch import Architecture
    from xftsim.mate import RandomMating, LinearAssortativeMating
    from xftsim.reproduce import RecombinationMap
    from xftsim.stats import SampleStatistics
    from xftsim.filters import TrioFilter, SibPairFilter
    from xftsim.sim import Simulation

    # --- Founders ---
    founder_cfg = config.get("founder", {})
    n = founder_cfg.get("n", 100)
    m = founder_cfg.get("m", 50)
    min_maf = founder_cfg.get("min_maf", 0.1)

    sim_cfg = config.get("simulation", {})
    seed = sim_cfg.get("seed", None)

    # Set seed before founder generation for reproducibility
    if seed is not None:
        np.random.seed(seed)

    founder_hap = founder_haplotypes_uniform_AFs(n=n, m=m, minMAF=min_maf)

    # --- Effects ---
    effects_cfg = config.get("effects", {})
    effects = {}
    for eff_name, eff_spec in effects_cfg.items():
        eff_type = eff_spec.get("type", "additive")
        if eff_type == "additive":
            h2 = eff_spec.get("h2", 0.5)
            standardized = eff_spec.get("standardized", True)
            eff_seed = eff_spec.get("seed", None)
            effects[eff_name] = AdditiveEffects.from_h2(
                h2=h2, m=m, standardized=standardized, seed=eff_seed
            )
        elif eff_type == "sparse":
            h2 = eff_spec.get("h2", 0.5)
            k_causal = eff_spec.get("k_causal", max(1, m // 10))
            standardized = eff_spec.get("standardized", True)
            eff_seed = eff_spec.get("seed", None)
            effects[eff_name] = SparseEffects.from_h2(
                h2=h2, m=m, k_causal=k_causal,
                standardized=standardized, seed=eff_seed
            )
        elif eff_type == "multivariate":
            h2_list = eff_spec.get("h2", [0.5, 0.5])
            rg = eff_spec.get("rg", 0.0)
            standardized = eff_spec.get("standardized", True)
            eff_seed = eff_spec.get("seed", None)
            effects[eff_name] = MultivariateEffects.from_h2_rg(
                h2=h2_list, rg=rg, m=m,
                standardized=standardized, seed=eff_seed
            )
        else:
            raise ValueError(f"Unknown effect type: {eff_type}")

    # --- Architecture ---
    arch_cfg = config.get("architecture", {})
    formula = arch_cfg.get("formula", None)
    if formula is None:
        raise ValueError("Config must specify architecture.formula")
    architecture = Architecture(formula=formula, effects=effects)

    # --- Mating ---
    mating_cfg = config.get("mating", {})
    mating_type = mating_cfg.get("type", "random")
    offspring_per_pair = mating_cfg.get("offspring_per_pair", 2)

    if mating_type == "random":
        mating_regime = RandomMating(offspring_per_pair=offspring_per_pair)
    elif mating_type == "assortative":
        component_names = mating_cfg.get("component_names", ["Y"])
        r = mating_cfg.get("r", 0.0)
        mating_regime = LinearAssortativeMating(
            component_names=component_names, r=r,
            offspring_per_pair=offspring_per_pair,
        )
    else:
        raise ValueError(f"Unknown mating type: {mating_type}")

    # --- Recombination ---
    # Thread pos_bp through when the founders carry it, so RecombinationMap
    # can suppress crossovers between same-position variants (see
    # RecombinationMap docstring).
    recom_cfg = config.get("recombination", {})
    recom_type = recom_cfg.get("type", "constant")
    recom_p = recom_cfg.get("p", 0.5)
    founder_pos_bp = None
    founder_variants = getattr(founder_hap, "variants", None)
    if founder_variants is not None:
        founder_pos_bp = getattr(founder_variants, "pos_bp", None)
    if recom_type == "constant":
        recombination_map = RecombinationMap.constant_map(
            m=m, p=recom_p, pos_bp=founder_pos_bp)
    else:
        recombination_map = RecombinationMap.constant_map(
            m=m, p=recom_p, pos_bp=founder_pos_bp)

    # --- Statistics ---
    stats_cfg = config.get("statistics", [])
    statistics = []
    for stat_name in stats_cfg:
        if stat_name in ("sample_statistics", "SampleStatistics"):
            statistics.append(SampleStatistics())
        else:
            raise ValueError(f"Unknown statistic: {stat_name}")

    # --- Filters ---
    filters_cfg = config.get("filters", {})
    filters = {}
    for name, filt_type in filters_cfg.items():
        if isinstance(filt_type, str):
            filt_type_str = filt_type
        elif isinstance(filt_type, dict):
            filt_type_str = filt_type.get("type", name)
        else:
            filt_type_str = str(filt_type)

        if filt_type_str in ("trio", "TrioFilter"):
            filters[name] = TrioFilter()
        elif filt_type_str in ("sibpair", "SibPairFilter"):
            filters[name] = SibPairFilter()
        else:
            raise ValueError(f"Unknown filter type: {filt_type_str}")

    # --- Simulation params ---
    retain_haplotypes = sim_cfg.get("retain_haplotypes", 1)
    retain_phenotypes = sim_cfg.get("retain_phenotypes", 2)

    sim = Simulation(
        founder_haplotypes=founder_hap,
        architecture=architecture,
        mating_regime=mating_regime,
        recombination_map=recombination_map,
        retain_haplotypes=retain_haplotypes,
        retain_phenotypes=retain_phenotypes,
        statistics=statistics,
        filters=filters,
        seed=seed,
    )

    # --- Output settings ---
    output_cfg = config.get("output", {})

    return sim, output_cfg


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def run(
    config_path: str = typer.Argument(..., help="Path to YAML/JSON config file"),
    generations: Optional[int] = typer.Option(
        None, "--generations", "-g",
        help="Override number of generations from config"
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", "-s",
        help="Override random seed from config"
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", "-o",
        help="Directory for saving results"
    ),
    checkpoint_every: Optional[int] = typer.Option(
        None, "--checkpoint-every",
        help="Save checkpoint every N generations"
    ),
    plain: bool = typer.Option(False, "--plain", help="Force plain text output"),
    rich_flag: bool = typer.Option(False, "--rich", help="Force rich output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output except errors"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Extra detail"),
):
    """Run a simulation from a YAML/JSON config file."""
    mode = _detect_output_mode(plain, rich_flag)
    out = _Output(mode=mode, quiet=quiet, verbose=verbose)

    try:
        config = _load_config_file(config_path)
    except (typer.BadParameter, Exception) as e:
        out.error(str(e))
        raise typer.Exit(code=1)

    # Apply CLI overrides
    if seed is not None:
        config.setdefault("simulation", {})["seed"] = seed
    if generations is not None:
        config.setdefault("simulation", {})["generations"] = generations

    try:
        sim, output_cfg = build_simulation_from_config(config)
    except (ValueError, KeyError) as e:
        out.error(f"Config error: {e}")
        raise typer.Exit(code=1)

    n_gen = config.get("simulation", {}).get("generations", 10)
    if generations is not None:
        n_gen = generations
    out_dir = output_dir or output_cfg.get("dir", None)
    ckpt_every = checkpoint_every or output_cfg.get("checkpoint_every", None)

    out.info(f"Starting simulation: n={sim.haplotypes.n}, m={sim.haplotypes.m}, "
             f"generations={n_gen}")
    out.debug(f"Architecture: {sim.architecture}")
    out.debug(f"Mating: {sim.mating_regime}")

    _run_simulation(sim, n_gen, out, out_dir, ckpt_every)


@app.command()
def resume(
    checkpoint_dir: str = typer.Argument(..., help="Path to checkpoint directory"),
    generations: int = typer.Option(
        5, "--generations", "-g",
        help="Number of additional generations to run"
    ),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", "-o",
        help="Directory for saving results"
    ),
    checkpoint_every: Optional[int] = typer.Option(
        None, "--checkpoint-every",
        help="Save checkpoint every N generations"
    ),
    plain: bool = typer.Option(False, "--plain", help="Force plain text output"),
    rich_flag: bool = typer.Option(False, "--rich", help="Force rich output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output except errors"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Extra detail"),
):
    """Resume a simulation from a checkpoint directory."""
    from xftsim.sim import Simulation
    from xftsim.stats import SampleStatistics

    mode = _detect_output_mode(plain, rich_flag)
    out = _Output(mode=mode, quiet=quiet, verbose=verbose)

    if not os.path.isdir(checkpoint_dir):
        out.error(f"Checkpoint directory not found: {checkpoint_dir}")
        raise typer.Exit(code=1)

    out.info(f"Resuming from checkpoint: {checkpoint_dir}")

    try:
        sim = Simulation.from_checkpoint(
            checkpoint_dir,
            statistics=[SampleStatistics()],
        )
    except Exception as e:
        out.error(f"Failed to load checkpoint: {e}")
        raise typer.Exit(code=1)

    out.info(f"Loaded at generation {sim.generation}, "
             f"n={sim.haplotypes.n}, m={sim.haplotypes.m}")
    out.info(f"Running {generations} additional generations...")

    _run_simulation(
        sim, generations, out, output_dir, checkpoint_every,
        use_continue=True,
    )


@app.command()
def info(
    checkpoint_dir: str = typer.Argument(..., help="Path to checkpoint directory"),
    plain: bool = typer.Option(False, "--plain", help="Force plain text output"),
    rich_flag: bool = typer.Option(False, "--rich", help="Force rich output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output except errors"),
):
    """Show info about a checkpoint or saved simulation."""
    mode = _detect_output_mode(plain, rich_flag)
    out = _Output(mode=mode, quiet=quiet)

    if not os.path.isdir(checkpoint_dir):
        out.error(f"Checkpoint directory not found: {checkpoint_dir}")
        raise typer.Exit(code=1)

    meta_path = os.path.join(checkpoint_dir, "meta.json")
    if not os.path.isfile(meta_path):
        out.error(f"No meta.json found in {checkpoint_dir}")
        raise typer.Exit(code=1)

    with open(meta_path, "r") as f:
        meta = json.load(f)

    generation = meta.get("generation", "?")
    retain_hap = meta.get("retain_haplotypes", "?")
    retain_phen = meta.get("retain_phenotypes", "?")
    mating_info = meta.get("mating", {})
    mating_type = mating_info.get("type", "?")

    # Count haplotype files to get n/m
    n_samples = "?"
    n_variants = "?"
    hap_dir = os.path.join(checkpoint_dir, "haplotypes")
    if os.path.isdir(hap_dir):
        hap_files = sorted(Path(hap_dir).glob("gen_*.npz"))
        if hap_files:
            try:
                data = np.load(str(hap_files[-1]), allow_pickle=True)
                geno = data["genotypes"]
                n_samples = geno.shape[0]
                n_variants = geno.shape[1]
            except Exception:
                pass

    # Architecture summary
    arch_summary = "?"
    arch_dir = os.path.join(checkpoint_dir, "architecture")
    arch_json = os.path.join(arch_dir, "architecture.json")
    if os.path.isfile(arch_json):
        try:
            with open(arch_json, "r") as f:
                nodes = json.load(f)
            n_nodes = len(nodes)
            outputs = []
            for node in nodes:
                outputs.extend(node.get("outputs", []))
            arch_summary = f"{n_nodes} nodes, outputs: {outputs}"
        except Exception:
            pass

    rows = [
        ("Generation", generation),
        ("N samples", n_samples),
        ("N variants", n_variants),
        ("Retain haplotypes", retain_hap),
        ("Retain phenotypes", retain_phen),
        ("Mating type", mating_type),
        ("Architecture", arch_summary),
    ]

    if mode == "rich":
        try:
            from rich.table import Table
            table = Table(title=f"Checkpoint: {checkpoint_dir}")
            table.add_column("Property", style="bold")
            table.add_column("Value")
            for prop, val in rows:
                table.add_row(prop, str(val))
            out.console.print(table)
        except ImportError:
            for prop, val in rows:
                print(f"{prop}: {val}", flush=True)
    else:
        for prop, val in rows:
            print(f"{prop}: {val}", flush=True)


@app.command()
def demo(
    name: str = typer.Argument(
        ...,
        help="Demo name: UGRM (univariate genetic + noise) or BGRM (bivariate)",
    ),
    n: int = typer.Option(200, "--n", help="Number of founder individuals"),
    m: int = typer.Option(100, "--m", help="Number of variants"),
    generations: int = typer.Option(5, "--generations", "-g", help="Number of generations"),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    plain: bool = typer.Option(False, "--plain", help="Force plain text output"),
    rich_flag: bool = typer.Option(False, "--rich", help="Force rich output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output except errors"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Extra detail"),
):
    """Run a built-in demo simulation."""
    mode = _detect_output_mode(plain, rich_flag)
    out = _Output(mode=mode, quiet=quiet, verbose=verbose)

    name_upper = name.upper()
    if name_upper == "UGRM":
        config = _demo_ugrm(n, m, generations, seed)
    elif name_upper == "BGRM":
        config = _demo_bgrm(n, m, generations, seed)
    else:
        out.error(f"Unknown demo: {name}. Available: UGRM, BGRM")
        raise typer.Exit(code=1)

    try:
        sim, output_cfg = build_simulation_from_config(config)
    except Exception as e:
        out.error(f"Demo setup failed: {e}")
        raise typer.Exit(code=1)

    n_gen = config["simulation"]["generations"]
    out.info(f"Running demo '{name_upper}': n={n}, m={m}, generations={n_gen}")

    _run_simulation(sim, n_gen, out, None, None)


# ---------------------------------------------------------------------------
# Demo configs
# ---------------------------------------------------------------------------

def _demo_ugrm(n: int, m: int, generations: int, seed: int) -> dict:
    """Config for univariate additive genetic + noise model."""
    return {
        "founder": {"n": n, "m": m},
        "effects": {
            "eff1": {"type": "additive", "h2": 0.5},
        },
        "architecture": {
            "formula": "Y.G ~ genetic(eff1)\nY.E ~ noise(0.5)\nY ~ Y.G + Y.E",
        },
        "mating": {"type": "random", "offspring_per_pair": 2},
        "recombination": {"type": "constant"},
        "simulation": {
            "generations": generations,
            "seed": seed,
            "retain_haplotypes": 1,
            "retain_phenotypes": 2,
        },
        "statistics": ["sample_statistics"],
        "filters": {},
        "output": {},
    }


def _demo_bgrm(n: int, m: int, generations: int, seed: int) -> dict:
    """Config for bivariate correlated genetic model."""
    return {
        "founder": {"n": n, "m": m},
        "effects": {
            "mveff": {
                "type": "multivariate",
                "h2": [0.4, 0.6],
                "rg": 0.5,
            },
        },
        "architecture": {
            "formula": (
                "(Y1.G, Y2.G) ~ mvGenetic(mveff)\n"
                "Y1.E ~ noise(0.6)\n"
                "Y2.E ~ noise(0.4)\n"
                "Y1 ~ Y1.G + Y1.E\n"
                "Y2 ~ Y2.G + Y2.E"
            ),
        },
        "mating": {"type": "random", "offspring_per_pair": 2},
        "recombination": {"type": "constant"},
        "simulation": {
            "generations": generations,
            "seed": seed,
            "retain_haplotypes": 1,
            "retain_phenotypes": 2,
        },
        "statistics": ["sample_statistics"],
        "filters": {},
        "output": {},
    }


# ---------------------------------------------------------------------------
# Simulation runner (shared logic for run/resume/demo)
# ---------------------------------------------------------------------------

def _run_simulation(sim, n_gen: int, out: _Output,
                    output_dir: Optional[str],
                    checkpoint_every: Optional[int],
                    use_continue: bool = False):
    """Execute a simulation with progress tracking and optional checkpointing.

    Parameters
    ----------
    sim : Simulation
        The simulation to run.
    n_gen : int
        Number of generations.
    out : _Output
        Output helper.
    output_dir : str or None
        Where to save results/checkpoints.
    checkpoint_every : int or None
        Checkpoint interval.
    use_continue : bool
        If True, use sim.continue_run() instead of sim.run().
    """
    from xftsim.io import save_simulation_checkpoint

    start_time = time.time()

    # Callback for per-generation progress
    gen_start = [time.time()]
    start_gen = sim.generation

    def _progress_callback(s):
        elapsed = time.time() - gen_start[0]
        current_gen = s.generation

        if use_continue:
            gen_display = current_gen - start_gen
        else:
            gen_display = current_gen

        # Try to get variance of 'Y' if available
        var_y = None
        if current_gen in s.phenotype_history:
            pheno = s.phenotype_history[current_gen]
            if "Y" in pheno:
                var_y = float(np.var(pheno["Y"]))

        if out.mode == "rich" and not out.quiet and out._console:
            parts = [f"[green]gen {gen_display}/{n_gen}[/green]",
                     f"elapsed={elapsed:.2f}s"]
            if var_y is not None:
                parts.append(f"var(Y)={var_y:.4f}")
            out._console.print("  ".join(parts))
        else:
            out.generation_line(gen_display, n_gen, elapsed, var_y)

        # Checkpointing
        if (checkpoint_every and output_dir and
                current_gen > 0 and current_gen % checkpoint_every == 0):
            ckpt_dir = os.path.join(output_dir, f"checkpoint_gen{current_gen}")
            out.debug(f"Saving checkpoint to {ckpt_dir}")
            os.makedirs(ckpt_dir, exist_ok=True)
            save_simulation_checkpoint(s, ckpt_dir)

        gen_start[0] = time.time()

    sim.callbacks.append(_progress_callback)

    try:
        if use_continue:
            sim.continue_run(n_gen)
        else:
            sim.run(n_gen)
    except Exception as e:
        out.error(f"Simulation failed: {e}")
        raise typer.Exit(code=1)
    finally:
        # Remove our callback
        if _progress_callback in sim.callbacks:
            sim.callbacks.remove(_progress_callback)

    total_elapsed = time.time() - start_time

    # Final checkpoint
    if output_dir:
        final_dir = os.path.join(output_dir, "final")
        out.info(f"Saving final state to {final_dir}")
        os.makedirs(final_dir, exist_ok=True)
        save_simulation_checkpoint(sim, final_dir)

    # Summary
    out.info(f"Completed in {total_elapsed:.2f}s")

    if sim.results:
        rows = []
        for res in sim.results:
            gen = res.generation
            stats = res.statistics
            sample_stats = stats.get("SampleStatistics", {})
            if sample_stats and "var" in sample_stats:
                var_vals = sample_stats["var"]
                keys = sample_stats.get("keys", [])
                for k, v in zip(keys, var_vals):
                    rows.append((gen, k, f"{v:.4f}"))
        if rows:
            out.summary_table(rows, ["Generation", "Component", "Variance"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Entry point for console_scripts."""
    app()


if __name__ == "__main__":
    main()
