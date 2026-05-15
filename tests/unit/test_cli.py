"""
Tests for xftsim.cli — config parsing, output modes, demo commands, error handling.
"""
import json
import os
import sys
import tempfile

import numpy as np
import pytest

from xftsim.cli import (
    _detect_output_mode,
    _load_config_file,
    _Output,
    _demo_ugrm,
    _demo_bgrm,
    build_simulation_from_config,
    app,
)


# ---------------------------------------------------------------------------
# Output mode detection
# ---------------------------------------------------------------------------

class TestOutputModeDetection:
    """Tests for _detect_output_mode."""

    def test_plain_flag_forces_plain(self):
        assert _detect_output_mode(plain=True, rich_flag=False) == "plain"

    def test_rich_flag_forces_rich(self):
        assert _detect_output_mode(plain=False, rich_flag=True) == "rich"

    def test_plain_overrides_rich(self):
        # plain=True takes priority
        assert _detect_output_mode(plain=True, rich_flag=True) == "plain"

    def test_default_depends_on_tty(self):
        # We cannot control isatty from a test, but we can verify
        # that neither flag active gives a valid result
        result = _detect_output_mode(plain=False, rich_flag=False)
        assert result in ("plain", "rich")


class TestOutputHelper:
    """Tests for the _Output helper class."""

    def test_plain_mode(self):
        out = _Output(mode="plain", quiet=False, verbose=False)
        assert out.mode == "plain"
        assert out.console is None

    def test_rich_mode(self):
        out = _Output(mode="rich", quiet=False, verbose=False)
        assert out.mode == "rich"
        assert out.console is not None

    def test_quiet_suppresses_info(self, capsys):
        out = _Output(mode="plain", quiet=True, verbose=False)
        out.info("should not appear")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_quiet_does_not_suppress_error(self, capsys):
        out = _Output(mode="plain", quiet=True, verbose=False)
        out.error("bad thing")
        captured = capsys.readouterr()
        assert "bad thing" in captured.err

    def test_verbose_debug(self, capsys):
        out = _Output(mode="plain", quiet=False, verbose=True)
        out.debug("debug info")
        captured = capsys.readouterr()
        assert "debug info" in captured.out

    def test_non_verbose_hides_debug(self, capsys):
        out = _Output(mode="plain", quiet=False, verbose=False)
        out.debug("hidden")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_generation_line(self, capsys):
        out = _Output(mode="plain", quiet=False)
        out.generation_line(3, 10, 1.23, var_y=0.987)
        captured = capsys.readouterr()
        assert "gen 3/10" in captured.out
        assert "elapsed=1.23s" in captured.out
        assert "var(Y)=0.9870" in captured.out

    def test_generation_line_no_var(self, capsys):
        out = _Output(mode="plain", quiet=False)
        out.generation_line(1, 5, 0.5)
        captured = capsys.readouterr()
        assert "gen 1/5" in captured.out
        assert "var(Y)" not in captured.out

    def test_summary_table_plain(self, capsys):
        out = _Output(mode="plain", quiet=False)
        rows = [(0, "Y", "1.0000"), (1, "Y", "1.0500")]
        out.summary_table(rows, ["Gen", "Component", "Var"])
        captured = capsys.readouterr()
        assert "Gen" in captured.out
        assert "1.0500" in captured.out

    def test_summary_table_rich(self):
        out = _Output(mode="rich", quiet=False)
        rows = [(0, "Y", "1.0000")]
        # Should not raise
        out.summary_table(rows, ["Gen", "Component", "Var"])


# ---------------------------------------------------------------------------
# Config file loading
# ---------------------------------------------------------------------------

class TestLoadConfigFile:
    """Tests for _load_config_file."""

    def test_load_json(self, tmp_path):
        config = {"founder": {"n": 50, "m": 25}}
        path = tmp_path / "config.json"
        path.write_text(json.dumps(config))
        loaded = _load_config_file(str(path))
        assert loaded["founder"]["n"] == 50

    def test_load_yaml(self, tmp_path):
        yaml_content = "founder:\n  n: 60\n  m: 30\n"
        path = tmp_path / "config.yaml"
        path.write_text(yaml_content)
        loaded = _load_config_file(str(path))
        assert loaded["founder"]["n"] == 60

    def test_load_yml_extension(self, tmp_path):
        yaml_content = "founder:\n  n: 70\n"
        path = tmp_path / "config.yml"
        path.write_text(yaml_content)
        loaded = _load_config_file(str(path))
        assert loaded["founder"]["n"] == 70

    def test_missing_file_raises(self):
        import typer
        with pytest.raises(typer.BadParameter, match="not found"):
            _load_config_file("/nonexistent/config.yaml")

    def test_non_dict_raises(self, tmp_path):
        import typer
        path = tmp_path / "bad.json"
        path.write_text('"just a string"')
        with pytest.raises(typer.BadParameter, match="must be a YAML/JSON dict"):
            _load_config_file(str(path))


# ---------------------------------------------------------------------------
# Config → Simulation
# ---------------------------------------------------------------------------

class TestBuildSimulationFromConfig:
    """Tests for build_simulation_from_config."""

    @pytest.fixture
    def minimal_config(self):
        return {
            "founder": {"n": 50, "m": 20},
            "effects": {
                "eff1": {"type": "additive", "h2": 0.5},
            },
            "architecture": {
                "formula": "Y.G ~ genetic(eff1)\nY.E ~ noise(0.5)\nY ~ Y.G + Y.E",
            },
            "mating": {"type": "random", "offspring_per_pair": 2},
            "simulation": {"generations": 3, "seed": 42},
        }

    @pytest.fixture
    def full_config(self):
        return {
            "founder": {"n": 100, "m": 50, "min_maf": 0.05},
            "effects": {
                "eff1": {"type": "additive", "h2": 0.5, "standardized": True, "seed": 7},
            },
            "architecture": {
                "formula": "Y.G ~ genetic(eff1)\nY.E ~ noise(0.5)\nY ~ Y.G + Y.E",
            },
            "mating": {"type": "random", "offspring_per_pair": 2},
            "recombination": {"type": "constant", "p": 0.5},
            "simulation": {
                "generations": 10,
                "seed": 42,
                "retain_haplotypes": 2,
                "retain_phenotypes": 5,
            },
            "statistics": ["sample_statistics"],
            "filters": {"trio": "trio"},
            "output": {"dir": "./results", "checkpoint_every": 5},
        }

    def test_minimal_config_builds(self, minimal_config):
        sim, output_cfg = build_simulation_from_config(minimal_config)
        assert sim.haplotypes.n == 50
        assert sim.haplotypes.m == 20
        assert sim.generation == 0

    def test_full_config_builds(self, full_config):
        sim, output_cfg = build_simulation_from_config(full_config)
        assert sim.haplotypes.n == 100
        assert sim.haplotypes.m == 50
        assert sim.retain_haplotypes == 2
        assert sim.retain_phenotypes == 5
        assert len(sim.statistics) == 1
        assert len(sim.filters) == 1
        assert "trio" in sim.filters
        assert output_cfg["dir"] == "./results"
        assert output_cfg["checkpoint_every"] == 5

    def test_minimal_can_run(self, minimal_config):
        sim, _ = build_simulation_from_config(minimal_config)
        sim.run(3)
        assert sim.generation == 2  # 0, 1, 2

    def test_sparse_effects(self):
        config = {
            "founder": {"n": 50, "m": 30},
            "effects": {
                "sp": {"type": "sparse", "h2": 0.3, "k_causal": 5},
            },
            "architecture": {
                "formula": "Y.G ~ genetic(sp)\nY.E ~ noise(0.7)\nY ~ Y.G + Y.E",
            },
            "mating": {"type": "random"},
            "simulation": {"seed": 1},
        }
        sim, _ = build_simulation_from_config(config)
        sim.run(2)
        assert sim.generation == 1

    def test_multivariate_effects(self):
        config = {
            "founder": {"n": 60, "m": 40},
            "effects": {
                "mv": {"type": "multivariate", "h2": [0.4, 0.6], "rg": 0.3},
            },
            "architecture": {
                "formula": (
                    "(Y1.G, Y2.G) ~ mvGenetic(mv)\n"
                    "Y1.E ~ noise(0.6)\n"
                    "Y2.E ~ noise(0.4)\n"
                    "Y1 ~ Y1.G + Y1.E\n"
                    "Y2 ~ Y2.G + Y2.E"
                ),
            },
            "mating": {"type": "random"},
            "simulation": {"seed": 5},
        }
        sim, _ = build_simulation_from_config(config)
        sim.run(2)
        assert "Y1" in sim.phenotype_history[1]
        assert "Y2" in sim.phenotype_history[1]

    def test_assortative_mating(self):
        config = {
            "founder": {"n": 80, "m": 20},
            "effects": {
                "eff": {"type": "additive", "h2": 0.5},
            },
            "architecture": {
                "formula": "Y.G ~ genetic(eff)\nY.E ~ noise(0.5)\nY ~ Y.G + Y.E",
            },
            "mating": {
                "type": "assortative",
                "component_names": ["Y"],
                "r": 0.5,
                "offspring_per_pair": 2,
            },
            "simulation": {"seed": 10},
        }
        sim, _ = build_simulation_from_config(config)
        sim.run(3)
        assert sim.generation == 2

    def test_missing_formula_raises(self):
        config = {
            "founder": {"n": 50, "m": 20},
            "effects": {},
            "architecture": {},
            "mating": {"type": "random"},
            "simulation": {},
        }
        with pytest.raises(ValueError, match="formula"):
            build_simulation_from_config(config)

    def test_unknown_effect_type_raises(self):
        config = {
            "founder": {"n": 50, "m": 20},
            "effects": {
                "bad": {"type": "unknown_type"},
            },
            "architecture": {
                "formula": "Y ~ noise(1.0)",
            },
            "mating": {"type": "random"},
            "simulation": {},
        }
        with pytest.raises(ValueError, match="Unknown effect type"):
            build_simulation_from_config(config)

    def test_unknown_mating_type_raises(self):
        config = {
            "founder": {"n": 50, "m": 20},
            "effects": {},
            "architecture": {
                "formula": "Y ~ noise(1.0)",
            },
            "mating": {"type": "unknown_mating"},
            "simulation": {},
        }
        with pytest.raises(ValueError, match="Unknown mating type"):
            build_simulation_from_config(config)

    def test_unknown_statistic_raises(self):
        config = {
            "founder": {"n": 50, "m": 20},
            "effects": {},
            "architecture": {
                "formula": "Y ~ noise(1.0)",
            },
            "mating": {"type": "random"},
            "simulation": {},
            "statistics": ["unknown_stat"],
        }
        with pytest.raises(ValueError, match="Unknown statistic"):
            build_simulation_from_config(config)

    def test_unknown_filter_raises(self):
        config = {
            "founder": {"n": 50, "m": 20},
            "effects": {},
            "architecture": {
                "formula": "Y ~ noise(1.0)",
            },
            "mating": {"type": "random"},
            "simulation": {},
            "filters": {"bad": "unknown_filter"},
        }
        with pytest.raises(ValueError, match="Unknown filter type"):
            build_simulation_from_config(config)

    def test_sibpair_filter(self):
        config = {
            "founder": {"n": 50, "m": 20},
            "effects": {
                "eff": {"type": "additive", "h2": 0.5},
            },
            "architecture": {
                "formula": "Y.G ~ genetic(eff)\nY.E ~ noise(0.5)\nY ~ Y.G + Y.E",
            },
            "mating": {"type": "random"},
            "simulation": {"seed": 42},
            "filters": {"sib": "sibpair"},
        }
        sim, _ = build_simulation_from_config(config)
        assert "sib" in sim.filters

    def test_defaults_for_missing_sections(self):
        """Minimal config with only architecture.formula."""
        config = {
            "architecture": {
                "formula": "Y ~ noise(1.0)",
            },
        }
        sim, output_cfg = build_simulation_from_config(config)
        assert sim.haplotypes.n == 100  # default
        assert sim.haplotypes.m == 50   # default
        assert output_cfg == {}


# ---------------------------------------------------------------------------
# Demo configs
# ---------------------------------------------------------------------------

class TestDemoConfigs:
    """Tests for built-in demo configurations."""

    def test_ugrm_demo_builds(self):
        config = _demo_ugrm(n=50, m=20, generations=3, seed=42)
        sim, _ = build_simulation_from_config(config)
        assert sim.haplotypes.n == 50
        assert sim.haplotypes.m == 20

    def test_ugrm_demo_runs(self):
        config = _demo_ugrm(n=50, m=20, generations=3, seed=42)
        sim, _ = build_simulation_from_config(config)
        sim.run(3)
        assert sim.generation == 2

    def test_bgrm_demo_builds(self):
        config = _demo_bgrm(n=60, m=30, generations=3, seed=7)
        sim, _ = build_simulation_from_config(config)
        assert sim.haplotypes.n == 60
        assert sim.haplotypes.m == 30

    def test_bgrm_demo_runs(self):
        config = _demo_bgrm(n=60, m=30, generations=2, seed=7)
        sim, _ = build_simulation_from_config(config)
        sim.run(2)
        assert sim.generation == 1
        assert "Y1" in sim.phenotype_history[1]
        assert "Y2" in sim.phenotype_history[1]


# ---------------------------------------------------------------------------
# Integration: run from YAML config file
# ---------------------------------------------------------------------------

class TestRunFromConfigFile:
    """Test loading a config file and running the simulation."""

    def test_yaml_config_end_to_end(self, tmp_path):
        yaml_content = """
founder:
  n: 50
  m: 20

effects:
  eff1:
    type: additive
    h2: 0.5

architecture:
  formula: |
    Y.G ~ genetic(eff1)
    Y.E ~ noise(0.5)
    Y ~ Y.G + Y.E

mating:
  type: random
  offspring_per_pair: 2

simulation:
  generations: 3
  seed: 42
"""
        path = tmp_path / "sim.yaml"
        path.write_text(yaml_content)

        config = _load_config_file(str(path))
        sim, _ = build_simulation_from_config(config)
        n_gen = config["simulation"]["generations"]
        sim.run(n_gen)
        assert sim.generation == n_gen - 1

    def test_json_config_end_to_end(self, tmp_path):
        config = {
            "founder": {"n": 40, "m": 15},
            "effects": {"e": {"type": "additive", "h2": 0.4}},
            "architecture": {
                "formula": "Y.G ~ genetic(e)\nY.E ~ noise(0.6)\nY ~ Y.G + Y.E",
            },
            "mating": {"type": "random"},
            "simulation": {"generations": 2, "seed": 99},
        }
        path = tmp_path / "sim.json"
        path.write_text(json.dumps(config))

        loaded = _load_config_file(str(path))
        sim, _ = build_simulation_from_config(loaded)
        sim.run(loaded["simulation"]["generations"])
        assert sim.generation == 1


# ---------------------------------------------------------------------------
# CLI app import
# ---------------------------------------------------------------------------

class TestAppImport:
    """Test that the CLI app is importable."""

    def test_app_exists(self):
        assert app is not None

    def test_from_xftsim_import(self):
        from xftsim.cli import app as cli_app
        assert cli_app is not None

    def test_main_entry_point(self):
        from xftsim.cli import main
        assert callable(main)
