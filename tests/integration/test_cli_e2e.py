"""
End-to-end CLI tests for xftsim.

These tests invoke the CLI via subprocess to verify that commands work
correctly when run as external processes, including proper exit codes,
output formatting, checkpointing, and error handling.
"""
import json
import os
import re
import subprocess
import sys

import pytest

PYTHON = sys.executable


def _run_cli(*args, timeout=60, cwd=None):
    """Run the xftsim CLI via subprocess and return the CompletedProcess."""
    cmd = [PYTHON, "-m", "xftsim.cli"] + list(args)
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
    )


def _minimal_yaml_config():
    """Return a minimal YAML config string for a univariate simulation."""
    return """\
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


def _minimal_json_config():
    """Return a minimal JSON config dict for a univariate simulation."""
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


# ---------------------------------------------------------------------------
# Demo commands
# ---------------------------------------------------------------------------


class TestDemoSubprocess:
    """Tests for the 'demo' command run via subprocess."""

    def test_demo_ugrm_subprocess(self):
        """Run demo UGRM and check exit code and generation output."""
        result = _run_cli(
            "demo", "UGRM",
            "--n", "200", "--m", "50", "--generations", "3", "--plain",
        )
        assert result.returncode == 0, (
            f"Expected exit code 0, got {result.returncode}.\n"
            f"stderr: {result.stderr}"
        )
        output = result.stdout
        assert "UGRM" in output
        assert "gen" in output.lower()
        # Should have generation progress lines
        assert "[gen 0/3]" in output or "gen 0" in output
        assert "Completed" in output

    def test_demo_bgrm_subprocess(self):
        """Run demo BGRM and check exit code and generation output."""
        result = _run_cli(
            "demo", "BGRM",
            "--n", "200", "--m", "50", "--generations", "3", "--plain",
        )
        assert result.returncode == 0, (
            f"Expected exit code 0, got {result.returncode}.\n"
            f"stderr: {result.stderr}"
        )
        output = result.stdout
        assert "BGRM" in output
        assert "Completed" in output
        # BGRM should produce Y1 and Y2 in the summary table
        assert "Y1" in output
        assert "Y2" in output


# ---------------------------------------------------------------------------
# Run from config files
# ---------------------------------------------------------------------------


class TestRunFromConfig:
    """Tests for the 'run' command with YAML and JSON configs."""

    def test_run_from_yaml_config(self, tmp_path):
        """Write a YAML config, run it, check success and output."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(_minimal_yaml_config())

        result = _run_cli(
            "run", str(config_file), "--plain",
        )
        assert result.returncode == 0, (
            f"Expected exit code 0, got {result.returncode}.\n"
            f"stderr: {result.stderr}\nstdout: {result.stdout}"
        )
        output = result.stdout
        assert "Starting simulation" in output
        assert "Completed" in output
        # Should mention generation progress
        assert "gen" in output.lower()

    def test_run_from_json_config(self, tmp_path):
        """Write a JSON config, run it, check success and output."""
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(_minimal_json_config(), indent=2))

        result = _run_cli(
            "run", str(config_file), "--plain",
        )
        assert result.returncode == 0, (
            f"Expected exit code 0, got {result.returncode}.\n"
            f"stderr: {result.stderr}\nstdout: {result.stdout}"
        )
        output = result.stdout
        assert "Starting simulation" in output
        assert "Completed" in output


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


class TestCheckpointing:
    """Tests for checkpoint creation, resume, and info."""

    def test_run_with_checkpoint(self, tmp_path):
        """Run with --checkpoint-every and verify checkpoint dirs are created."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(_minimal_yaml_config())
        output_dir = tmp_path / "output"

        result = _run_cli(
            "run", str(config_file),
            "--plain",
            "--checkpoint-every", "1",
            "--output-dir", str(output_dir),
        )
        assert result.returncode == 0, (
            f"Expected exit code 0, got {result.returncode}.\n"
            f"stderr: {result.stderr}\nstdout: {result.stdout}"
        )

        # Check that checkpoint directories were created
        assert output_dir.is_dir(), "Output directory should exist"

        # The run creates checkpoint_gen<N> dirs for each generation that is
        # a multiple of checkpoint_every (and > 0), plus a 'final' dir.
        final_dir = output_dir / "final"
        assert final_dir.is_dir(), "Final checkpoint directory should exist"
        assert (final_dir / "meta.json").is_file(), "meta.json should exist in final"

        # At least one intermediate checkpoint should exist (gen 1 or gen 2)
        checkpoint_dirs = [
            d for d in output_dir.iterdir()
            if d.is_dir() and d.name.startswith("checkpoint_gen")
        ]
        assert len(checkpoint_dirs) >= 1, (
            f"Expected at least 1 intermediate checkpoint, "
            f"found: {[d.name for d in output_dir.iterdir()]}"
        )

    def test_resume_from_checkpoint(self, tmp_path):
        """Create a checkpoint then resume from it."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(_minimal_yaml_config())
        output_dir = tmp_path / "output"

        # Step 1: Initial run with checkpointing
        result1 = _run_cli(
            "run", str(config_file),
            "--plain",
            "--checkpoint-every", "1",
            "--output-dir", str(output_dir),
        )
        assert result1.returncode == 0, (
            f"Initial run failed. stderr: {result1.stderr}"
        )

        # Find the final checkpoint
        checkpoint_dir = output_dir / "final"
        assert checkpoint_dir.is_dir(), (
            f"Checkpoint dir not found. Contents: "
            f"{[d.name for d in output_dir.iterdir()]}"
        )

        # Step 2: Resume from checkpoint
        result2 = _run_cli(
            "resume", str(checkpoint_dir),
            "--generations", "2",
            "--plain",
        )
        assert result2.returncode == 0, (
            f"Resume failed with code {result2.returncode}.\n"
            f"stderr: {result2.stderr}\nstdout: {result2.stdout}"
        )
        output = result2.stdout
        assert "Resuming from checkpoint" in output
        assert "Loaded at generation" in output
        assert "Completed" in output

    def test_info_command(self, tmp_path):
        """Create a checkpoint, then run info on it."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(_minimal_yaml_config())
        output_dir = tmp_path / "output"

        # Create checkpoint
        result1 = _run_cli(
            "run", str(config_file),
            "--plain",
            "--checkpoint-every", "1",
            "--output-dir", str(output_dir),
        )
        assert result1.returncode == 0, (
            f"Initial run failed. stderr: {result1.stderr}"
        )

        checkpoint_dir = output_dir / "final"

        # Run info
        result2 = _run_cli(
            "info", str(checkpoint_dir), "--plain",
        )
        assert result2.returncode == 0, (
            f"Info command failed with code {result2.returncode}.\n"
            f"stderr: {result2.stderr}\nstdout: {result2.stdout}"
        )
        output = result2.stdout
        # The info command outputs "Property: value" lines
        assert "Generation" in output
        assert "N samples" in output
        assert "N variants" in output
        # Check that actual values appear (the simulation had n=50, m=20)
        assert "50" in output
        assert "20" in output


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


class TestOutputFormatting:
    """Tests for --plain, --quiet, and ANSI escape handling."""

    def test_plain_output_no_ansi(self):
        """Verify --plain produces no ANSI escape codes."""
        result = _run_cli(
            "demo", "UGRM",
            "--n", "100", "--m", "30", "--generations", "2", "--plain",
        )
        assert result.returncode == 0
        # ANSI escape codes start with \x1b[
        assert "\x1b[" not in result.stdout, (
            "Plain output should not contain ANSI escape codes"
        )
        assert "\x1b[" not in result.stderr, (
            "Plain stderr should not contain ANSI escape codes"
        )

    def test_quiet_suppresses_output(self):
        """Verify --quiet suppresses stdout output."""
        result = _run_cli(
            "demo", "UGRM",
            "--n", "100", "--m", "30", "--generations", "2", "--quiet",
        )
        assert result.returncode == 0
        # Quiet mode should suppress all info/generation output
        # stdout should be empty or nearly empty
        stdout_stripped = result.stdout.strip()
        assert stdout_stripped == "", (
            f"Expected empty stdout with --quiet, got: {repr(stdout_stripped)}"
        )


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for error conditions and non-zero exit codes."""

    def test_invalid_config_exits_nonzero(self, tmp_path):
        """A config missing the required formula should fail."""
        bad_config = {
            "founder": {"n": 50, "m": 20},
            "effects": {},
            "architecture": {},
            "mating": {"type": "random"},
            "simulation": {"generations": 3, "seed": 42},
        }
        config_file = tmp_path / "bad_config.yaml"
        # Write as JSON with .yaml extension -- still valid YAML
        config_file.write_text(json.dumps(bad_config, indent=2))

        result = _run_cli(
            "run", str(config_file), "--plain",
        )
        assert result.returncode != 0, (
            f"Expected non-zero exit code for invalid config, "
            f"got {result.returncode}.\nstdout: {result.stdout}"
        )
        # Error message should mention formula or config
        combined = result.stdout + result.stderr
        assert "formula" in combined.lower() or "error" in combined.lower(), (
            f"Expected error message about formula. Output: {combined}"
        )

    def test_missing_config_file_exits_nonzero(self):
        """Running with a nonexistent config file should fail."""
        result = _run_cli(
            "run", "/nonexistent/path/config.yaml", "--plain",
        )
        assert result.returncode != 0

    def test_missing_checkpoint_dir_exits_nonzero(self):
        """Resume from a nonexistent checkpoint should fail."""
        result = _run_cli(
            "resume", "/nonexistent/checkpoint/dir", "--plain",
        )
        assert result.returncode != 0

    def test_info_missing_dir_exits_nonzero(self):
        """Info on a nonexistent directory should fail."""
        result = _run_cli(
            "info", "/nonexistent/checkpoint/dir", "--plain",
        )
        assert result.returncode != 0

    def test_unknown_demo_exits_nonzero(self):
        """Requesting an unknown demo name should fail."""
        result = _run_cli(
            "demo", "NONEXISTENT_DEMO", "--plain",
        )
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "unknown" in combined.lower() or "error" in combined.lower()
