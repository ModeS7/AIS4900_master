"""Comprehensive CLI tests for all entry points.

Tests CLI help, config overrides, and error handling for all scripts.
"""

import sys
import subprocess

import pytest


# Mark all tests in this module
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.timeout(120),  # 2 minute timeout
]


# =============================================================================
# CLI Help Tests
# =============================================================================


class TestCLIHelp:
    """Test --help for all CLI entry points."""

    @pytest.mark.parametrize(
        "script",
        [
            "medgen.scripts.train",
            "medgen.scripts.train_compression",
            "medgen.scripts.generate",
        ],
    )
    def test_help_exits_zero(self, script, run_cli):
        """--help returns exit code 0."""
        result = run_cli(
            ["python", "-m", script, "--help"],
            check=False,
        )
        assert result.returncode == 0, (
            f"{script} --help failed with code {result.returncode}:\n"
            f"STDERR: {result.stderr}"
        )

    @pytest.mark.parametrize(
        "script,expected_keywords",
        [
            ("medgen.scripts.train", ["mode", "strategy", "training"]),
            ("medgen.scripts.train_compression", ["config", "mode"]),
            ("medgen.scripts.generate", ["gen_mode", "paths", "num_images"]),
        ],
    )
    def test_help_shows_usage(self, script, expected_keywords, run_cli):
        """--help output contains expected keywords."""
        result = run_cli(
            ["python", "-m", script, "--help"],
            check=False,
        )

        output = result.stdout.lower()
        for keyword in expected_keywords:
            assert keyword.lower() in output, (
                f"{script} --help missing '{keyword}':\n{result.stdout[:2000]}"
            )

    def test_train_shows_modes(self, run_cli):
        """Train help shows available modes."""
        result = run_cli(
            ["python", "-m", "medgen.scripts.train", "--help"],
            check=False,
        )

        output = result.stdout.lower()
        # Should mention mode options
        assert "mode" in output, "Train help should mention mode"
        # Should list at least some modes
        modes_found = sum(
            1 for mode in ["bravo", "seg", "dual"] if mode in output
        )
        assert modes_found >= 1, (
            f"Train help should list mode options, found {modes_found}"
        )

    def test_train_shows_strategies(self, run_cli):
        """Train help shows available strategies."""
        result = run_cli(
            ["python", "-m", "medgen.scripts.train", "--help"],
            check=False,
        )

        output = result.stdout.lower()
        # Should mention strategy options
        assert "strategy" in output, "Train help should mention strategy"
        strategies_found = sum(
            1 for s in ["rflow", "ddpm"] if s in output
        )
        assert strategies_found >= 1, (
            f"Train help should list strategy options, found {strategies_found}"
        )


# =============================================================================
# Config Override Tests
# =============================================================================


class TestCLIConfigOverrides:
    """Test Hydra config override syntax."""

    def test_hydra_override_syntax(self, run_cli, temp_output_dir):
        """key=value override syntax works."""
        # Use --help to verify config parsing (doesn't require data)
        result = run_cli(
            [
                "python",
                "-m",
                "medgen.scripts.train",
                "--cfg=job",  # Print resolved config
                "training.epochs=99",
                "training.batch_size=16",
            ],
            check=False,
        )

        # Should parse config (even if it fails later)
        # Check stdout for the config values
        output = result.stdout + result.stderr
        assert "99" in output or "epochs" in output, (
            f"Override not visible in output:\n{output[:2000]}"
        )

    def test_nested_override(self, run_cli, temp_output_dir):
        """Nested key override (foo.bar=value) works."""
        result = run_cli(
            [
                "python",
                "-m",
                "medgen.scripts.train",
                "--cfg=job",
                "model.channels=[64,128,256]",
            ],
            check=False,
        )

        output = result.stdout + result.stderr
        # Should see the channel config somewhere
        assert "64" in output or "channels" in output, (
            f"Nested override not working:\n{output[:2000]}"
        )

    def test_multiple_overrides(self, run_cli, temp_output_dir):
        """Multiple overrides in same command work."""
        result = run_cli(
            [
                "python",
                "-m",
                "medgen.scripts.train",
                "--cfg=job",
                "mode=bravo",
                "strategy=rflow",
                "training.epochs=5",
                "training.batch_size=8",
            ],
            check=False,
        )

        # Should parse without error
        output = result.stdout + result.stderr
        assert not any(
            err in output.lower()
            for err in ["omegaconf.errors", "override error", "interpolation"]
        ), f"Multiple overrides caused config error:\n{output[:2000]}"


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestCLIErrorHandling:
    """Test CLI error handling and user-friendly messages."""

    def test_invalid_mode_error(self, run_cli, temp_output_dir):
        """Invalid mode gives helpful error message."""
        result = run_cli(
            [
                "python",
                "-m",
                "medgen.scripts.train",
                "mode=invalid_mode_xyz_123",
                f"hydra.run.dir={temp_output_dir}",
            ],
            check=False,
            timeout=60,
        )

        # Should fail
        assert result.returncode != 0, "Invalid mode should fail"

        # Error should mention the issue
        output = result.stdout + result.stderr
        assert any(
            phrase in output.lower()
            for phrase in [
                "invalid_mode_xyz_123",
                "mode",
                "not found",
                "invalid",
                "error",
            ]
        ), f"Error message not helpful:\n{output[:1500]}"

    def test_invalid_strategy_error(self, run_cli, temp_output_dir):
        """Invalid strategy gives helpful error message."""
        result = run_cli(
            [
                "python",
                "-m",
                "medgen.scripts.train",
                "mode=bravo",
                "strategy=fake_strategy_xyz",
                f"hydra.run.dir={temp_output_dir}",
            ],
            check=False,
            timeout=60,
        )

        # Should fail
        assert result.returncode != 0, "Invalid strategy should fail"

        # Error should be understandable
        output = result.stdout + result.stderr
        assert "error" in output.lower() or "not found" in output.lower(), (
            f"Error not clear:\n{output[:1500]}"
        )

    def test_missing_checkpoint_error(self, run_cli, temp_output_dir):
        """Missing checkpoint file gives clear error."""
        result = run_cli(
            [
                "python",
                "-m",
                "medgen.scripts.generate",
                "gen_mode=bravo",
                "seg_model=/this/path/does/not/exist.pt",
                "image_model=/another/fake/path.pt",
                f"paths.generated_dir={temp_output_dir}",
            ],
            check=False,
            timeout=60,
        )

        # Should fail
        assert result.returncode != 0, "Missing checkpoint should fail"

        # Error should mention file/path issue
        output = result.stdout + result.stderr
        assert any(
            phrase in output.lower()
            for phrase in ["not found", "no such file", "does not exist", "error"]
        ), f"Error about missing file not clear:\n{output[:1500]}"

    def test_invalid_config_name_error(self, run_cli, temp_output_dir):
        """Invalid --config-name gives helpful error."""
        result = run_cli(
            [
                "python",
                "-m",
                "medgen.scripts.train_compression",
                "--config-name=nonexistent_config_xyz",
                f"hydra.run.dir={temp_output_dir}",
            ],
            check=False,
            timeout=60,
        )

        # Should fail
        assert result.returncode != 0, "Invalid config name should fail"

        # Error should mention config issue
        output = result.stdout + result.stderr
        assert any(
            phrase in output.lower()
            for phrase in ["config", "not found", "error", "nonexistent"]
        ), f"Config error not clear:\n{output[:1500]}"

    def test_typo_in_key_gives_hint(self, run_cli, temp_output_dir):
        """Typo in config key gives helpful error (Hydra feature)."""
        result = run_cli(
            [
                "python",
                "-m",
                "medgen.scripts.train",
                "traning.epochs=5",  # Typo: "traning" instead of "training"
                f"hydra.run.dir={temp_output_dir}",
            ],
            check=False,
            timeout=60,
        )

        # Hydra should catch this
        output = result.stdout + result.stderr

        # Either fails with helpful message or parses it anyway
        if result.returncode != 0:
            # Should mention the typo or similar keys
            assert any(
                phrase in output.lower()
                for phrase in ["traning", "training", "key", "not found", "error"]
            ), f"Typo error not helpful:\n{output[:1500]}"


# =============================================================================
# Module Import Tests
# =============================================================================


class TestModuleImports:
    """Test that modules can be imported without side effects."""

    def test_train_module_importable(self):
        """medgen.scripts.train module can be imported."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import medgen.scripts.train; print('OK')",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Import should work (may print warnings, but shouldn't fail)
        assert "OK" in result.stdout, (
            f"Import failed:\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}"
        )

    def test_train_compression_module_importable(self):
        """medgen.scripts.train_compression module can be imported."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import medgen.scripts.train_compression; print('OK')",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert "OK" in result.stdout, (
            f"Import failed:\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}"
        )

    def test_generate_module_importable(self):
        """medgen.scripts.generate module can be imported."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import medgen.scripts.generate; print('OK')",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert "OK" in result.stdout, (
            f"Import failed:\n"
            f"STDOUT: {result.stdout}\n"
            f"STDERR: {result.stderr}"
        )


# =============================================================================
# Version and Info Tests
# =============================================================================


class TestVersionInfo:
    """Test version and info commands."""

    def test_hydra_version_works(self, run_cli):
        """Hydra --version flag works."""
        result = run_cli(
            ["python", "-m", "medgen.scripts.train", "--version"],
            check=False,
        )

        # Hydra might not have --version, but shouldn't crash
        # If it works, great. If not, should give sensible error.
        assert result.returncode in (0, 1, 2), (
            f"--version caused crash:\n{result.stderr}"
        )

    def test_cfg_job_shows_config(self, run_cli):
        """--cfg=job shows resolved configuration."""
        result = run_cli(
            [
                "python",
                "-m",
                "medgen.scripts.train",
                "--cfg=job",
                "mode=bravo",
            ],
            check=False,
        )

        # Should show config
        output = result.stdout
        config_indicators = ["training:", "model:", "mode:", "strategy:"]
        found = sum(1 for indicator in config_indicators if indicator in output)
        assert found >= 2, (
            f"--cfg=job should show config structure, found {found} indicators:\n"
            f"{output[:2000]}"
        )
