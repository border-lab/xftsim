"""
Tests covering cli.py edge cases and code paths.

Targets uncovered lines in cli.py: _detect_output_mode, _Output, _load_config_file,
build_simulation_from_config, run/resume/info/demo commands, _run_simulation,
demo configs, and summary table rendering.
"""
import json
import os
import sys
import tempfile

import numpy as np
import pytest
import typer

from xftsim.cli import (
    _detect_output_mode,
    _Output,
    _load_config_file,
    build_simulation_from_config,
    _demo_ugrm,
    _demo_bgrm,
    _run_simulation,
)


# ---------------------------------------------------------------------------
# _detect_output_mode
# ---------------------------------------------------------------------------

class TestDetectOutputMode:
    def test_plain_flag(self):
        assert _detect_output_mode(plain=True) == "plain"

    def test_rich_flag(self):
        assert _detect_output_mode(rich_flag=True) == "rich"

    def test_plain_overrides_rich(self):
        # plain takes priority
        assert _detect_output_mode(plain=True, rich_flag=True) == "plain"

    def test_default_not_tty(self, monkeypatch):
        # When stdout is not a tty, default is plain
        monkeypatch.setattr(sys.stdout, 'isatty', lambda: False)
        assert _detect_output_mode() == "plain"

    def test_default_tty(self, monkeypatch):
        monkeypatch.setattr(sys.stdout, 'isatty', lambda: True)
        assert _detect_output_mode() == "rich"


# ---------------------------------------------------------------------------
# _Output
# ---------------------------------------------------------------------------

class TestOutput:
    def test_plain_info(self, capsys):
        out = _Output(mode="plain")
        out.info("hello world")
        captured = capsys.readouterr()
        assert "hello world" in captured.out

    def test_quiet_suppresses_info(self, capsys):
        out = _Output(mode="plain", quiet=True)
        out.info("should not print")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_plain_error(self, capsys):
        out = _Output(mode="plain")
        out.error("something broke")
        captured = capsys.readouterr()
        assert "Error: something broke" in captured.err

    def test_plain_debug_not_verbose(self, capsys):
        out = _Output(mode="plain", verbose=False)
        out.debug("debug msg")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_plain_debug_verbose(self, capsys):
        out = _Output(mode="plain", verbose=True)
        out.debug("debug msg")
        captured = capsys.readouterr()
        assert "debug msg" in captured.out

    def test_debug_quiet_overrides_verbose(self, capsys):
        out = _Output(mode="plain", verbose=True, quiet=True)
        out.debug("debug msg")
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_generation_line_plain(self, capsys):
        out = _Output(mode="plain")
        out.generation_line(gen=2, total=10, elapsed=1.23, var_y=0.5678)
        captured = capsys.readouterr()
        assert "[gen 2/10]" in captured.out
        assert "elapsed=1.23s" in captured.out
        assert "var(Y)=0.5678" in captured.out

    def test_generation_line_no_var(self, capsys):
        out = _Output(mode="plain")
        out.generation_line(gen=1, total=5, elapsed=0.5)
        captured = capsys.readouterr()
        assert "var(Y)" not in captured.out

    def test_generation_line_quiet(self, capsys):
        out = _Output(mode="plain", quiet=True)
        out.generation_line(gen=1, total=5, elapsed=0.5, var_y=1.0)
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_summary_table_plain(self, capsys):
        out = _Output(mode="plain")
        rows = [(1, "Y", "0.5000"), (2, "Y", "0.4500")]
        headers = ["Gen", "Comp", "Var"]
        out.summary_table(rows, headers)
        captured = capsys.readouterr()
        assert "Gen" in captured.out
        assert "0.5000" in captured.out
        assert "---" in captured.out

    def test_summary_table_quiet(self, capsys):
        out = _Output(mode="plain", quiet=True)
        out.summary_table([(1, "Y", "0.5")], ["Gen", "Comp", "Var"])
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_console_property(self):
        out = _Output(mode="plain")
        assert out.console is None

    def test_rich_fallback_if_not_installed(self, monkeypatch):
        """If rich import fails, mode falls back to plain."""
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "rich.console" or name == "rich":
                raise ImportError("no rich")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', mock_import)
        out = _Output(mode="rich")
        assert out.mode == "plain"


# ---------------------------------------------------------------------------
# _load_config_file
# ---------------------------------------------------------------------------

class TestLoadConfigFile:
    def test_file_not_found(self):
        with pytest.raises(typer.BadParameter, match="Config file not found"):
            _load_config_file("/nonexistent/path.yaml")

    def test_json_config(self, tmp_path):
        cfg = {"founder": {"n": 10}, "simulation": {"seed": 1}}
        path = tmp_path / "config.json"
        path.write_text(json.dumps(cfg))
        result = _load_config_file(str(path))
        assert result["founder"]["n"] == 10

    def test_yaml_config(self, tmp_path):
        pytest.importorskip("yaml")
        import yaml
        cfg = {"founder": {"n": 20}, "simulation": {"seed": 2}}
        path = tmp_path / "config.yaml"
        path.write_text(yaml.dump(cfg))
        result = _load_config_file(str(path))
        assert result["founder"]["n"] == 20

    def test_yml_extension(self, tmp_path):
        pytest.importorskip("yaml")
        import yaml
        cfg = {"founder": {"n": 30}}
        path = tmp_path / "config.yml"
        path.write_text(yaml.dump(cfg))
        result = _load_config_file(str(path))
        assert result["founder"]["n"] == 30

    def test_unknown_extension_tries_yaml_then_json(self, tmp_path):
        """File with unknown extension (.toml) should try YAML, then JSON."""
        cfg = {"founder": {"n": 40}}
        path = tmp_path / "config.toml"
        path.write_text(json.dumps(cfg))
        result = _load_config_file(str(path))
        assert result["founder"]["n"] == 40

    def test_non_dict_config_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps([1, 2, 3]))
        with pytest.raises(typer.BadParameter, match="must be a YAML/JSON dict"):
            _load_config_file(str(path))


# ---------------------------------------------------------------------------
# build_simulation_from_config
# ---------------------------------------------------------------------------

class TestBuildSimulationFromConfig:
    @pytest.fixture
    def base_config(self):
        return {
            "founder": {"n": 50, "m": 20, "min_maf": 0.1},
            "effects": {
                "eff1": {"type": "additive", "h2": 0.5},
            },
            "architecture": {
                "formula": "Y.G ~ genetic(eff1)\nY.E ~ noise(0.5)\nY ~ Y.G + Y.E",
            },
            "mating": {"type": "random", "offspring_per_pair": 2},
            "recombination": {"type": "constant", "p": 0.5},
            "simulation": {"seed": 42, "retain_haplotypes": 1, "retain_phenotypes": 2},
            "statistics": ["sample_statistics"],
            "filters": {},
            "output": {},
        }

    def test_basic_config(self, base_config):
        sim, output_cfg = build_simulation_from_config(base_config)
        assert sim.haplotypes.n == 50
        assert sim.haplotypes.m == 20
        assert isinstance(output_cfg, dict)

    def test_sparse_effects(self, base_config):
        base_config["effects"]["eff1"] = {
            "type": "sparse", "h2": 0.3, "k_causal": 5, "seed": 10,
        }
        sim, _ = build_simulation_from_config(base_config)
        assert sim is not None

    def test_multivariate_effects(self, base_config):
        base_config["effects"] = {
            "mv": {"type": "multivariate", "h2": [0.4, 0.6], "rg": 0.3, "seed": 5},
        }
        base_config["architecture"]["formula"] = (
            "(Y1.G, Y2.G) ~ mvGenetic(mv)\n"
            "Y1.E ~ noise(0.6)\n"
            "Y2.E ~ noise(0.4)\n"
            "Y1 ~ Y1.G + Y1.E\n"
            "Y2 ~ Y2.G + Y2.E"
        )
        sim, _ = build_simulation_from_config(base_config)
        assert sim is not None

    def test_unknown_effect_type_raises(self, base_config):
        base_config["effects"]["eff1"]["type"] = "unknown_type"
        with pytest.raises(ValueError, match="Unknown effect type"):
            build_simulation_from_config(base_config)

    def test_missing_formula_raises(self, base_config):
        del base_config["architecture"]["formula"]
        with pytest.raises(ValueError, match="must specify architecture.formula"):
            build_simulation_from_config(base_config)

    def test_assortative_mating(self, base_config):
        base_config["mating"] = {
            "type": "assortative",
            "component_names": ["Y"],
            "r": 0.3,
            "offspring_per_pair": 2,
        }
        sim, _ = build_simulation_from_config(base_config)
        assert sim is not None

    def test_unknown_mating_type_raises(self, base_config):
        base_config["mating"]["type"] = "fantasy"
        with pytest.raises(ValueError, match="Unknown mating type"):
            build_simulation_from_config(base_config)

    def test_unknown_statistic_raises(self, base_config):
        base_config["statistics"] = ["bogus_stat"]
        with pytest.raises(ValueError, match="Unknown statistic"):
            build_simulation_from_config(base_config)

    def test_trio_filter(self, base_config):
        base_config["filters"] = {"trio": "TrioFilter"}
        sim, _ = build_simulation_from_config(base_config)
        assert "trio" in sim.filters

    def test_sibpair_filter_dict(self, base_config):
        base_config["filters"] = {"sib": {"type": "SibPairFilter"}}
        sim, _ = build_simulation_from_config(base_config)
        assert "sib" in sim.filters

    def test_unknown_filter_raises(self, base_config):
        base_config["filters"] = {"bad": "NotAFilter"}
        with pytest.raises(ValueError, match="Unknown filter type"):
            build_simulation_from_config(base_config)

    def test_filter_with_non_string_type(self, base_config):
        """Filter type given as int gets converted to str."""
        base_config["filters"] = {"bad": 42}
        with pytest.raises(ValueError, match="Unknown filter type"):
            build_simulation_from_config(base_config)

    def test_no_seed(self, base_config):
        del base_config["simulation"]["seed"]
        sim, _ = build_simulation_from_config(base_config)
        assert sim is not None

    def test_output_cfg_returned(self, base_config):
        base_config["output"] = {"dir": "/tmp/out", "checkpoint_every": 5}
        _, output_cfg = build_simulation_from_config(base_config)
        assert output_cfg["dir"] == "/tmp/out"
        assert output_cfg["checkpoint_every"] == 5


# ---------------------------------------------------------------------------
# Demo configs
# ---------------------------------------------------------------------------

class TestDemoConfigs:
    def test_ugrm_config_structure(self):
        cfg = _demo_ugrm(100, 50, 5, 42)
        assert cfg["founder"]["n"] == 100
        assert cfg["founder"]["m"] == 50
        assert cfg["simulation"]["generations"] == 5
        assert "architecture" in cfg

    def test_bgrm_config_structure(self):
        cfg = _demo_bgrm(100, 50, 5, 42)
        assert "mveff" in cfg["effects"]
        assert cfg["effects"]["mveff"]["type"] == "multivariate"

    def test_ugrm_builds(self):
        cfg = _demo_ugrm(50, 20, 3, 42)
        sim, _ = build_simulation_from_config(cfg)
        assert sim.haplotypes.n == 50

    def test_bgrm_builds(self):
        cfg = _demo_bgrm(50, 20, 3, 42)
        sim, _ = build_simulation_from_config(cfg)
        assert sim.haplotypes.n == 50


# ---------------------------------------------------------------------------
# _run_simulation
# ---------------------------------------------------------------------------

class TestRunSimulation:
    @pytest.fixture
    def sim_and_output(self):
        """Minimal simulation for testing the runner."""
        from xftsim.founders import founder_haplotypes_uniform_AFs
        from xftsim.neffect import AdditiveEffects
        from xftsim.narch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
        from xftsim.nmate import RandomMating
        from xftsim.reproduce import RecombinationMap
        from xftsim.nstats import SampleStatistics
        from xftsim.nsim import NSimulation

        np.random.seed(42)
        hap = founder_haplotypes_uniform_AFs(n=30, m=10)
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        rm = RecombinationMap.constant_map(m=10, p=0.5)
        mating = RandomMating(offspring_per_pair=2)
        sim = NSimulation(
            founder_haplotypes=hap,
            architecture=arch,
            mating_regime=mating,
            recombination_map=rm,
            statistics=[SampleStatistics()],
            seed=42,
        )
        out = _Output(mode="plain", quiet=True)
        return sim, out

    def test_run_without_checkpoint(self, sim_and_output):
        sim, out = sim_and_output
        _run_simulation(sim, 2, out, None, None)
        # run(2) runs gen 0 and gen 1
        assert sim.generation == 1

    def test_run_with_checkpoint(self, sim_and_output, tmp_path):
        sim, out = sim_and_output
        out_dir = str(tmp_path / "output")
        _run_simulation(sim, 2, out, out_dir, 1)
        # Should have checkpoints
        assert os.path.exists(os.path.join(out_dir, "final", "meta.json"))

    def test_run_use_continue(self, sim_and_output):
        sim, out = sim_and_output
        sim.run(2)  # run 2 gens first (gen 0 and 1)
        _run_simulation(sim, 2, out, None, None, use_continue=True)
        # continue_run(2) runs 2 additional gens: 2, 3
        assert sim.generation == 3


# ---------------------------------------------------------------------------
# CLI info command (unit-testing the internals)
# ---------------------------------------------------------------------------

class TestInfoCommand:
    def test_info_missing_dir(self):
        """info with nonexistent dir raises Exit."""
        from xftsim.cli import info
        from typer.testing import CliRunner
        from xftsim.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["info", "/nonexistent/path"])
        assert result.exit_code != 0

    def test_info_no_meta_json(self, tmp_path):
        """info with dir but no meta.json raises Exit."""
        from typer.testing import CliRunner
        from xftsim.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["info", str(tmp_path)])
        assert result.exit_code != 0

    def test_info_valid_checkpoint(self, tmp_path):
        """info with valid checkpoint shows properties."""
        from xftsim.cli import app
        from xftsim.founders import founder_haplotypes_uniform_AFs
        from xftsim.neffect import AdditiveEffects
        from xftsim.narch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
        from xftsim.nmate import RandomMating
        from xftsim.reproduce import RecombinationMap
        from xftsim.nsim import NSimulation
        from xftsim.io import save_simulation_checkpoint
        from typer.testing import CliRunner

        np.random.seed(42)
        hap = founder_haplotypes_uniform_AFs(n=20, m=10)
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        rm = RecombinationMap.constant_map(m=10, p=0.5)
        mating = RandomMating(offspring_per_pair=2)
        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=mating, recombination_map=rm, seed=42,
        )
        sim.run(2)
        ckpt_dir = str(tmp_path / "ckpt")
        save_simulation_checkpoint(sim, ckpt_dir)

        runner = CliRunner()
        result = runner.invoke(app, ["info", ckpt_dir, "--plain"])
        assert result.exit_code == 0
        assert "Generation" in result.output


# ---------------------------------------------------------------------------
# CLI demo command
# ---------------------------------------------------------------------------

class TestDemoCommand:
    def test_demo_ugrm(self):
        from typer.testing import CliRunner
        from xftsim.cli import app
        runner = CliRunner()
        result = runner.invoke(app, [
            "demo", "UGRM", "--n", "30", "--m", "10",
            "--generations", "2", "--seed", "42", "--plain", "--quiet"
        ])
        assert result.exit_code == 0

    def test_demo_bgrm(self):
        from typer.testing import CliRunner
        from xftsim.cli import app
        runner = CliRunner()
        result = runner.invoke(app, [
            "demo", "BGRM", "--n", "30", "--m", "10",
            "--generations", "2", "--seed", "42", "--plain", "--quiet"
        ])
        assert result.exit_code == 0

    def test_demo_unknown(self):
        from typer.testing import CliRunner
        from xftsim.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["demo", "BOGUS", "--plain"])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# CLI run command
# ---------------------------------------------------------------------------

class TestRunCommand:
    def test_run_json_config(self, tmp_path):
        from typer.testing import CliRunner
        from xftsim.cli import app

        cfg = _demo_ugrm(30, 10, 2, 42)
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(cfg))

        runner = CliRunner()
        result = runner.invoke(app, [
            "run", str(cfg_path), "--plain", "--quiet"
        ])
        assert result.exit_code == 0

    def test_run_with_overrides(self, tmp_path):
        from typer.testing import CliRunner
        from xftsim.cli import app

        cfg = _demo_ugrm(30, 10, 5, 42)
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(cfg))

        runner = CliRunner()
        result = runner.invoke(app, [
            "run", str(cfg_path), "--generations", "2", "--seed", "99",
            "--plain", "--quiet"
        ])
        assert result.exit_code == 0

    def test_run_nonexistent_config(self):
        from typer.testing import CliRunner
        from xftsim.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["run", "/no/such/file.json", "--plain"])
        assert result.exit_code != 0

    def test_run_with_output_dir(self, tmp_path):
        from typer.testing import CliRunner
        from xftsim.cli import app

        cfg = _demo_ugrm(30, 10, 2, 42)
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps(cfg))
        out_dir = str(tmp_path / "results")

        runner = CliRunner()
        result = runner.invoke(app, [
            "run", str(cfg_path), "--output-dir", out_dir,
            "--checkpoint-every", "1", "--plain", "--quiet"
        ])
        assert result.exit_code == 0
        assert os.path.exists(os.path.join(out_dir, "final", "meta.json"))


# ---------------------------------------------------------------------------
# CLI resume command
# ---------------------------------------------------------------------------

class TestResumeCommand:
    def test_resume_nonexistent_dir(self):
        from typer.testing import CliRunner
        from xftsim.cli import app
        runner = CliRunner()
        result = runner.invoke(app, ["resume", "/no/such/dir", "--plain"])
        assert result.exit_code != 0

    def test_resume_valid_checkpoint(self, tmp_path):
        from typer.testing import CliRunner
        from xftsim.cli import app
        from xftsim.founders import founder_haplotypes_uniform_AFs
        from xftsim.neffect import AdditiveEffects
        from xftsim.narch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
        from xftsim.nmate import RandomMating
        from xftsim.reproduce import RecombinationMap
        from xftsim.nsim import NSimulation
        from xftsim.io import save_simulation_checkpoint

        np.random.seed(42)
        hap = founder_haplotypes_uniform_AFs(n=20, m=10)
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        rm = RecombinationMap.constant_map(m=10, p=0.5)
        mating = RandomMating(offspring_per_pair=2)
        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=mating, recombination_map=rm, seed=42,
        )
        sim.run(2)
        ckpt_dir = str(tmp_path / "ckpt")
        save_simulation_checkpoint(sim, ckpt_dir)

        runner = CliRunner()
        result = runner.invoke(app, [
            "resume", ckpt_dir, "--generations", "1",
            "--plain", "--quiet"
        ])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Additional tests targeting uncovered lines
# ---------------------------------------------------------------------------


class TestOutputRichMode:
    """Tests for _Output methods in rich mode (lines 86, 92, 100)."""

    def test_rich_info(self):
        """Line 86: _Output.info() with rich console."""
        out = _Output(mode="rich", quiet=False)
        assert out.mode == "rich"
        assert out._console is not None
        # Should not raise; output goes to rich console
        out.info("rich info message")

    def test_rich_error(self):
        """Line 92: _Output.error() with rich console."""
        out = _Output(mode="rich", quiet=False)
        out.error("rich error message")

    def test_rich_debug(self):
        """Line 100: _Output.debug() with rich console."""
        out = _Output(mode="rich", quiet=False, verbose=True)
        out.debug("rich debug message")

    def test_rich_debug_not_verbose(self):
        """Debug in rich mode but verbose=False should still suppress."""
        out = _Output(mode="rich", quiet=False, verbose=False)
        out.debug("should not print")

    def test_rich_summary_table(self):
        """Lines 118-125: summary_table with rich mode."""
        out = _Output(mode="rich", quiet=False)
        rows = [(1, "Y", "0.5000"), (2, "Y", "0.4500")]
        headers = ["Gen", "Comp", "Var"]
        # Should not raise; output goes to rich console
        out.summary_table(rows, headers)


class TestLoadConfigYamlImportError:
    """Line 173-174: YAML import fails for .yaml/.yml extension."""

    def test_yaml_import_error(self, tmp_path, monkeypatch):
        """When yaml is not installed, loading .yaml raises BadParameter."""
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "yaml":
                raise ImportError("no yaml")
            return original_import(name, *args, **kwargs)

        # Write a valid yaml file
        path = tmp_path / "config.yaml"
        path.write_text('{"founder": {"n": 10}}')

        monkeypatch.setattr(builtins, '__import__', mock_import)
        with pytest.raises(typer.BadParameter, match="pyyaml is required"):
            _load_config_file(str(path))


class TestLoadConfigUnknownExtFallback:
    """Lines 189-191: Unknown extension, YAML fails, falls back to JSON."""

    def test_unknown_ext_yaml_fails_json_succeeds(self, tmp_path, monkeypatch):
        """File with unknown extension where YAML parse fails, JSON succeeds."""
        # Write valid JSON that is not valid YAML (actually all valid JSON
        # is valid YAML, so we mock yaml.safe_load to fail)
        cfg = {"founder": {"n": 40}}
        path = tmp_path / "config.cfg"
        path.write_text(json.dumps(cfg))

        import yaml
        original_safe_load = yaml.safe_load

        def broken_safe_load(*args, **kwargs):
            raise yaml.YAMLError("mock YAML parse failure")

        monkeypatch.setattr(yaml, 'safe_load', broken_safe_load)
        result = _load_config_file(str(path))
        assert result["founder"]["n"] == 40


class TestBuildSimRecombinationNonConstant:
    """Line 306: Non-constant recombination type falls through to default."""

    def test_non_constant_recombination(self):
        config = {
            "founder": {"n": 30, "m": 10},
            "effects": {
                "eff1": {"type": "additive", "h2": 0.5},
            },
            "architecture": {
                "formula": "Y.G ~ genetic(eff1)\nY.E ~ noise(0.5)\nY ~ Y.G + Y.E",
            },
            "mating": {"type": "random", "offspring_per_pair": 2},
            "recombination": {"type": "genetic_map", "p": 0.5},
            "simulation": {"seed": 42},
        }
        sim, _ = build_simulation_from_config(config)
        assert sim is not None


class TestRunCommandConfigError:
    """Lines 403-405: run command handles ValueError from build_simulation_from_config."""

    def test_run_invalid_config_values(self, tmp_path):
        from typer.testing import CliRunner
        from xftsim.cli import app

        # Config with missing formula triggers ValueError
        cfg = {
            "founder": {"n": 30, "m": 10},
            "effects": {"eff1": {"type": "additive", "h2": 0.5}},
            "architecture": {},
            "mating": {"type": "random"},
            "simulation": {"seed": 42},
        }
        cfg_path = tmp_path / "bad.json"
        cfg_path.write_text(json.dumps(cfg))

        runner = CliRunner()
        result = runner.invoke(app, ["run", str(cfg_path), "--plain"])
        assert result.exit_code != 0


class TestResumeCommandLoadFailure:
    """Lines 459-461: resume command handles checkpoint load failure."""

    def test_resume_corrupt_checkpoint(self, tmp_path):
        from typer.testing import CliRunner
        from xftsim.cli import app

        # Create a directory that looks like a checkpoint but has corrupt data
        ckpt_dir = tmp_path / "bad_ckpt"
        ckpt_dir.mkdir()
        # Write minimal meta.json but no other files
        meta = {"generation": 0, "retain_haplotypes": 1, "retain_phenotypes": 2}
        (ckpt_dir / "meta.json").write_text(json.dumps(meta))

        runner = CliRunner()
        result = runner.invoke(app, [
            "resume", str(ckpt_dir), "--generations", "1", "--plain"
        ])
        assert result.exit_code != 0


class TestInfoHaplotypeExceptionPath:
    """Lines 514-515: info command handles npz read exception."""

    def test_info_corrupt_haplotype_files(self, tmp_path):
        from typer.testing import CliRunner
        from xftsim.cli import app

        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()

        # Write valid meta.json
        meta = {
            "generation": 1,
            "retain_haplotypes": 1,
            "retain_phenotypes": 2,
            "mating": {"type": "RandomMating", "offspring_per_pair": 2},
        }
        (ckpt_dir / "meta.json").write_text(json.dumps(meta))

        # Write corrupt haplotype file
        hap_dir = ckpt_dir / "haplotypes"
        hap_dir.mkdir()
        (hap_dir / "gen_1.npz").write_text("not a real npz file")

        runner = CliRunner()
        result = runner.invoke(app, ["info", str(ckpt_dir), "--plain"])
        # Should succeed but show "?" for n_samples/n_variants
        assert result.exit_code == 0
        assert "?" in result.output


class TestInfoArchExceptionPath:
    """Lines 530-531: info command handles architecture JSON parse failure."""

    def test_info_corrupt_architecture(self, tmp_path):
        from typer.testing import CliRunner
        from xftsim.cli import app

        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()

        meta = {
            "generation": 1,
            "retain_haplotypes": 1,
            "retain_phenotypes": 2,
            "mating": {"type": "RandomMating", "offspring_per_pair": 2},
        }
        (ckpt_dir / "meta.json").write_text(json.dumps(meta))

        # Write corrupt architecture
        arch_dir = ckpt_dir / "architecture"
        arch_dir.mkdir()
        (arch_dir / "architecture.json").write_text("NOT VALID JSON{{{")

        runner = CliRunner()
        result = runner.invoke(app, ["info", str(ckpt_dir), "--plain"])
        assert result.exit_code == 0
        # Architecture should be "?" because JSON parse failed
        assert "?" in result.output


class TestInfoRichMode:
    """Lines 544-554: info command with rich output mode."""

    def test_info_rich_output(self, tmp_path):
        from typer.testing import CliRunner
        from xftsim.cli import app
        from xftsim.founders import founder_haplotypes_uniform_AFs
        from xftsim.neffect import AdditiveEffects
        from xftsim.narch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
        from xftsim.nmate import RandomMating
        from xftsim.reproduce import RecombinationMap
        from xftsim.nsim import NSimulation
        from xftsim.io import save_simulation_checkpoint

        np.random.seed(42)
        hap = founder_haplotypes_uniform_AFs(n=20, m=10)
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        rm = RecombinationMap.constant_map(m=10, p=0.5)
        mating = RandomMating(offspring_per_pair=2)
        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=mating, recombination_map=rm, seed=42,
        )
        sim.run(2)
        ckpt_dir = str(tmp_path / "ckpt")
        save_simulation_checkpoint(sim, ckpt_dir)

        runner = CliRunner()
        result = runner.invoke(app, ["info", ckpt_dir, "--rich"])
        assert result.exit_code == 0
        assert "Generation" in result.output


class TestInfoRichImportError:
    """Lines 552-554: info rich mode falls back when rich import fails."""

    def test_info_rich_fallback_no_rich(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner
        from xftsim.cli import app

        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()

        meta = {
            "generation": 2,
            "retain_haplotypes": 1,
            "retain_phenotypes": 2,
            "mating": {"type": "RandomMating", "offspring_per_pair": 2},
        }
        (ckpt_dir / "meta.json").write_text(json.dumps(meta))

        # We can test the fallback by invoking with --rich but the actual
        # rich import in the info command body is a separate try/except.
        # The CliRunner doesn't have a real tty so the output might not
        # use rich anyway. Just verify it succeeds.
        runner = CliRunner()
        result = runner.invoke(app, ["info", str(ckpt_dir), "--rich"])
        assert result.exit_code == 0
        assert "Generation" in result.output


class TestDemoSetupFailure:
    """Lines 590-592: demo command handles exception from build_simulation_from_config."""

    def test_demo_setup_failure(self, monkeypatch):
        from typer.testing import CliRunner
        from xftsim.cli import app
        import xftsim.cli as cli_module

        # Monkey-patch build_simulation_from_config to raise
        original_build = cli_module.build_simulation_from_config

        def broken_build(config):
            raise RuntimeError("simulated build failure")

        monkeypatch.setattr(cli_module, 'build_simulation_from_config', broken_build)

        runner = CliRunner()
        result = runner.invoke(app, [
            "demo", "UGRM", "--n", "30", "--m", "10",
            "--generations", "2", "--seed", "42", "--plain"
        ])
        assert result.exit_code != 0


class TestRunSimulationRichProgress:
    """Lines 712-716: rich mode progress callback in _run_simulation."""

    def test_run_simulation_rich_mode(self):
        from xftsim.founders import founder_haplotypes_uniform_AFs
        from xftsim.neffect import AdditiveEffects
        from xftsim.narch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
        from xftsim.nmate import RandomMating
        from xftsim.reproduce import RecombinationMap
        from xftsim.nstats import SampleStatistics
        from xftsim.nsim import NSimulation

        np.random.seed(42)
        hap = founder_haplotypes_uniform_AFs(n=30, m=10)
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        rm = RecombinationMap.constant_map(m=10, p=0.5)
        mating = RandomMating(offspring_per_pair=2)
        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=mating, recombination_map=rm,
            statistics=[SampleStatistics()],
            seed=42,
        )

        out = _Output(mode="rich", quiet=False, verbose=True)
        # Run in rich mode -- exercises lines 711-716
        _run_simulation(sim, 2, out, None, None)
        assert sim.generation == 1


class TestRunSimulationFailure:
    """Lines 737-739: simulation failure during _run_simulation."""

    def test_run_simulation_exception(self, monkeypatch):
        from xftsim.founders import founder_haplotypes_uniform_AFs
        from xftsim.neffect import AdditiveEffects
        from xftsim.narch import Architecture, GeneticComponent, NoiseComponent, AggregationComponent
        from xftsim.nmate import RandomMating
        from xftsim.reproduce import RecombinationMap
        from xftsim.nsim import NSimulation

        np.random.seed(42)
        hap = founder_haplotypes_uniform_AFs(n=30, m=10)
        eff = AdditiveEffects.from_h2(h2=0.5, m=10, seed=42)
        arch = Architecture()
        arch.add('Y.G', GeneticComponent(eff))
        arch.add('Y.E', NoiseComponent(variance=0.5))
        arch.add('Y', AggregationComponent('Y.G + Y.E'))
        rm = RecombinationMap.constant_map(m=10, p=0.5)
        mating = RandomMating(offspring_per_pair=2)
        sim = NSimulation(
            founder_haplotypes=hap, architecture=arch,
            mating_regime=mating, recombination_map=rm, seed=42,
        )

        # Monkey-patch sim.run to raise an exception
        def broken_run(n_gen):
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(sim, 'run', broken_run)

        out = _Output(mode="plain", quiet=True)
        with pytest.raises(typer.Exit):
            _run_simulation(sim, 2, out, None, None)


class TestMainEntryPoint:
    """Lines 778, 782: main() and __main__ entry point."""

    def test_main_callable(self):
        from xftsim.cli import main
        assert callable(main)

    def test_main_invokes_app(self, monkeypatch):
        """main() calls app(), which is the typer application."""
        from xftsim.cli import main, app
        called = []

        def mock_app(*args, **kwargs):
            called.append(True)

        monkeypatch.setattr('xftsim.cli.app', mock_app)
        main()
        assert len(called) == 1

    def test_dunder_main(self):
        """Line 782: __main__ block invoked via subprocess."""
        import subprocess
        result = subprocess.run(
            [sys.executable, "-c",
             "import runpy; runpy.run_module('xftsim.cli', run_name='__main__')"],
            capture_output=True, text=True, timeout=30,
        )
        # Without arguments, typer shows help or usage info
        # The important thing is it doesn't crash on import
        assert result.returncode in (0, 2)  # 2 = missing required arg


class TestInfoRichImportFallback:
    """Lines 552-554: info command rich mode with rich.table import failure."""

    def test_info_rich_table_import_error(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner
        from xftsim.cli import app

        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        meta = {
            "generation": 3,
            "retain_haplotypes": 1,
            "retain_phenotypes": 2,
            "mating": {"type": "RandomMating", "offspring_per_pair": 2},
        }
        (ckpt_dir / "meta.json").write_text(json.dumps(meta))

        # We need to make the import of rich.table fail inside the info
        # command's body. Use monkeypatch on builtins.__import__.
        import builtins
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "rich.table":
                raise ImportError("no rich.table")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', mock_import)

        runner = CliRunner()
        result = runner.invoke(app, ["info", str(ckpt_dir), "--rich"])
        assert result.exit_code == 0
        assert "Generation" in result.output
