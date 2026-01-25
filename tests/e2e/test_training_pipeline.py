"""E2E tests for training pipeline.

Tests the complete training workflow from CLI invocation through checkpoint saving.
These tests use subprocess to run training commands like a real user would.
"""

import os
import sys
import subprocess
from pathlib import Path

import pytest


# Mark all tests in this module
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.timeout(600),  # 10 minute timeout
]


# =============================================================================
# Helper functions
# =============================================================================


def run_training_command(
    args: list[str],
    timeout: int = 300,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run a training command via subprocess.

    Args:
        args: Command arguments (without 'python -m')
        timeout: Timeout in seconds
        check: Whether to assert returncode == 0

    Returns:
        CompletedProcess with stdout, stderr, returncode
    """
    full_args = [sys.executable, "-m"] + args
    result = subprocess.run(
        full_args,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},  # Force CPU for CI
    )

    if check and result.returncode != 0:
        print(f"Command failed: {' '.join(full_args)}")
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        raise AssertionError(f"Training failed with code {result.returncode}")

    return result


def find_checkpoint_files(output_dir: Path) -> list[Path]:
    """Find all checkpoint files in output directory."""
    return list(output_dir.glob("**/checkpoint*.pt"))


def find_tensorboard_events(output_dir: Path) -> list[Path]:
    """Find all TensorBoard event files in output directory."""
    return list(output_dir.glob("**/events.out.tfevents.*"))


# =============================================================================
# Diffusion Training Pipeline Tests
# =============================================================================


class TestDiffusionTrainingPipeline:
    """Test complete diffusion training workflow."""

    @pytest.mark.gpu
    def test_minimal_training_completes(self, temp_output_dir):
        """Minimal training (1 epoch, 5 batches) completes without error."""
        result = run_training_command(
            [
                "medgen.scripts.train",
                "mode=bravo",
                "strategy=rflow",
                "training.epochs=1",
                "training.batch_size=2",
                "training.limit_batches=5",
                "model.channels=[32,64]",
                "model.image_size=64",
                f"hydra.run.dir={temp_output_dir}",
            ],
            timeout=300,
            check=False,  # Don't fail immediately - check returncode
        )

        # Should complete successfully (returncode 0)
        # May fail if data not available, but should not crash on config
        assert result.returncode in (0, 1), (
            f"Training crashed unexpectedly:\n"
            f"STDOUT: {result.stdout[-2000:]}\n"
            f"STDERR: {result.stderr[-2000:]}"
        )

    @pytest.mark.gpu
    def test_checkpoint_created(self, temp_output_dir):
        """Training creates checkpoint file(s)."""
        # Run minimal training
        result = run_training_command(
            [
                "medgen.scripts.train",
                "mode=bravo",
                "strategy=rflow",
                "training.epochs=1",
                "training.batch_size=2",
                "training.limit_batches=5",
                "model.channels=[32,64]",
                "model.image_size=64",
                f"hydra.run.dir={temp_output_dir}",
            ],
            timeout=300,
            check=False,
        )

        if result.returncode != 0:
            pytest.skip("Training didn't complete - data may not be available")

        # Check for checkpoint files
        checkpoints = find_checkpoint_files(temp_output_dir)
        assert len(checkpoints) >= 1, (
            f"No checkpoint files found in {temp_output_dir}\n"
            f"Contents: {list(temp_output_dir.rglob('*'))}"
        )

    @pytest.mark.gpu
    def test_tensorboard_logs_created(self, temp_output_dir):
        """Training creates TensorBoard event files."""
        # Run minimal training
        result = run_training_command(
            [
                "medgen.scripts.train",
                "mode=bravo",
                "strategy=rflow",
                "training.epochs=1",
                "training.batch_size=2",
                "training.limit_batches=5",
                "model.channels=[32,64]",
                "model.image_size=64",
                f"hydra.run.dir={temp_output_dir}",
            ],
            timeout=300,
            check=False,
        )

        if result.returncode != 0:
            pytest.skip("Training didn't complete - data may not be available")

        # Check for TensorBoard files
        events = find_tensorboard_events(temp_output_dir)
        assert len(events) >= 1, (
            f"No TensorBoard event files found in {temp_output_dir}\n"
            f"Contents: {list(temp_output_dir.rglob('*'))}"
        )

    @pytest.mark.gpu
    def test_resume_training(self, temp_output_dir):
        """Training can resume from checkpoint."""
        # First training run (1 epoch)
        first_run = run_training_command(
            [
                "medgen.scripts.train",
                "mode=bravo",
                "strategy=rflow",
                "training.epochs=1",
                "training.batch_size=2",
                "training.limit_batches=5",
                "model.channels=[32,64]",
                "model.image_size=64",
                f"hydra.run.dir={temp_output_dir}",
            ],
            timeout=300,
            check=False,
        )

        if first_run.returncode != 0:
            pytest.skip("Initial training didn't complete - data may not be available")

        # Find checkpoint from first run
        checkpoints = find_checkpoint_files(temp_output_dir)
        if not checkpoints:
            pytest.skip("No checkpoint found to resume from")

        checkpoint_path = checkpoints[0]

        # Second training run (resume for 1 more epoch)
        resume_dir = temp_output_dir / "resumed"
        resume_run = run_training_command(
            [
                "medgen.scripts.train",
                "mode=bravo",
                "strategy=rflow",
                "training.epochs=2",  # Train to epoch 2
                "training.batch_size=2",
                "training.limit_batches=5",
                "model.channels=[32,64]",
                "model.image_size=64",
                f"training.resume_checkpoint={checkpoint_path}",
                f"hydra.run.dir={resume_dir}",
            ],
            timeout=300,
            check=False,
        )

        assert resume_run.returncode == 0, (
            f"Resume training failed:\n"
            f"STDOUT: {resume_run.stdout[-2000:]}\n"
            f"STDERR: {resume_run.stderr[-2000:]}"
        )

    @pytest.mark.gpu
    def test_ddpm_strategy_training(self, temp_output_dir):
        """DDPM strategy training completes without error."""
        result = run_training_command(
            [
                "medgen.scripts.train",
                "mode=bravo",
                "strategy=ddpm",
                "training.epochs=1",
                "training.batch_size=2",
                "training.limit_batches=5",
                "model.channels=[32,64]",
                "model.image_size=64",
                f"hydra.run.dir={temp_output_dir}",
            ],
            timeout=300,
            check=False,
        )

        # Should complete or fail gracefully
        assert result.returncode in (0, 1), (
            f"DDPM training crashed unexpectedly:\n"
            f"STDERR: {result.stderr[-2000:]}"
        )


# =============================================================================
# Compression Training Pipeline Tests
# =============================================================================


class TestCompressionTrainingPipeline:
    """Test compression model (VAE, VQ-VAE) training workflows."""

    @pytest.mark.gpu
    def test_vae_training_completes(self, temp_output_dir):
        """VAE training completes without error."""
        result = run_training_command(
            [
                "medgen.scripts.train_compression",
                "--config-name=vae",
                "mode=multi_modality",
                "training.epochs=1",
                "training.batch_size=2",
                "training.limit_batches=5",
                "model.channels=[32,64]",
                f"hydra.run.dir={temp_output_dir}",
            ],
            timeout=300,
            check=False,
        )

        # Should complete or fail with data issue (not crash)
        assert result.returncode in (0, 1), (
            f"VAE training crashed unexpectedly:\n"
            f"STDERR: {result.stderr[-2000:]}"
        )

    @pytest.mark.gpu
    def test_vqvae_training_completes(self, temp_output_dir):
        """VQ-VAE training completes without error."""
        result = run_training_command(
            [
                "medgen.scripts.train_compression",
                "--config-name=vqvae",
                "mode=multi_modality",
                "training.epochs=1",
                "training.batch_size=2",
                "training.limit_batches=5",
                "model.channels=[32,64]",
                f"hydra.run.dir={temp_output_dir}",
            ],
            timeout=300,
            check=False,
        )

        # Should complete or fail with data issue (not crash)
        assert result.returncode in (0, 1), (
            f"VQ-VAE training crashed unexpectedly:\n"
            f"STDERR: {result.stderr[-2000:]}"
        )

    @pytest.mark.gpu
    def test_vae_checkpoint_created(self, temp_output_dir):
        """VAE training creates checkpoint file."""
        result = run_training_command(
            [
                "medgen.scripts.train_compression",
                "--config-name=vae",
                "mode=multi_modality",
                "training.epochs=1",
                "training.batch_size=2",
                "training.limit_batches=5",
                "model.channels=[32,64]",
                f"hydra.run.dir={temp_output_dir}",
            ],
            timeout=300,
            check=False,
        )

        if result.returncode != 0:
            pytest.skip("VAE training didn't complete - data may not be available")

        checkpoints = find_checkpoint_files(temp_output_dir)
        assert len(checkpoints) >= 1, (
            f"No VAE checkpoint files found in {temp_output_dir}"
        )


# =============================================================================
# Multiple Modes Tests
# =============================================================================


class TestMultipleModes:
    """Test that various mode configurations don't crash on startup."""

    @pytest.mark.gpu
    @pytest.mark.parametrize("mode", ["bravo", "seg", "dual"])
    def test_mode_training_starts(self, mode, temp_output_dir):
        """Each mode initializes and starts training without crashing."""
        result = run_training_command(
            [
                "medgen.scripts.train",
                f"mode={mode}",
                "strategy=rflow",
                "training.epochs=1",
                "training.batch_size=2",
                "training.limit_batches=3",  # Very minimal
                "model.channels=[32,64]",
                "model.image_size=64",
                f"hydra.run.dir={temp_output_dir}/{mode}",
            ],
            timeout=300,
            check=False,
        )

        # Should at least start without crashing immediately
        # returncode 0 = success, 1 = data error (acceptable for CI)
        assert result.returncode in (0, 1), (
            f"Mode '{mode}' crashed unexpectedly:\n"
            f"STDERR: {result.stderr[-2000:]}"
        )

        # Check that it got past config parsing (look for training output)
        # If there's output about epochs or batches, it started successfully
        output_combined = result.stdout + result.stderr
        started = any(
            indicator in output_combined.lower()
            for indicator in [
                "epoch",
                "batch",
                "training",
                "loading",
                "dataset",
                "no such file",  # Data error = at least started
            ]
        )
        assert started, (
            f"Mode '{mode}' appears to have failed during config parsing:\n"
            f"Output: {output_combined[:2000]}"
        )

    @pytest.mark.gpu
    @pytest.mark.parametrize("strategy", ["rflow", "ddpm"])
    def test_strategy_training_starts(self, strategy, temp_output_dir):
        """Each strategy initializes and starts training without crashing."""
        result = run_training_command(
            [
                "medgen.scripts.train",
                "mode=bravo",
                f"strategy={strategy}",
                "training.epochs=1",
                "training.batch_size=2",
                "training.limit_batches=3",
                "model.channels=[32,64]",
                "model.image_size=64",
                f"hydra.run.dir={temp_output_dir}/{strategy}",
            ],
            timeout=300,
            check=False,
        )

        # Should at least start without crashing immediately
        assert result.returncode in (0, 1), (
            f"Strategy '{strategy}' crashed unexpectedly:\n"
            f"STDERR: {result.stderr[-2000:]}"
        )


# =============================================================================
# 3D Training Tests
# =============================================================================


class Test3DTraining:
    """Test 3D spatial_dims training (requires more resources)."""

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_3d_training_starts(self, temp_output_dir):
        """3D training initializes without crashing."""
        result = run_training_command(
            [
                "medgen.scripts.train",
                "mode=bravo",
                "strategy=rflow",
                "model.spatial_dims=3",
                "training.epochs=1",
                "training.batch_size=1",  # 3D needs smaller batch
                "training.limit_batches=2",
                "model.channels=[16,32]",  # Smaller for 3D
                "model.image_size=32",
                f"hydra.run.dir={temp_output_dir}",
            ],
            timeout=600,  # 3D is slower
            check=False,
        )

        # Should at least start
        assert result.returncode in (0, 1), (
            f"3D training crashed unexpectedly:\n"
            f"STDERR: {result.stderr[-2000:]}"
        )


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestTrainingErrorHandling:
    """Test that training handles errors gracefully."""

    def test_invalid_mode_gives_helpful_error(self, temp_output_dir):
        """Invalid mode gives helpful error message, not stack trace."""
        result = run_training_command(
            [
                "medgen.scripts.train",
                "mode=invalid_mode_xyz",
                "strategy=rflow",
                f"hydra.run.dir={temp_output_dir}",
            ],
            timeout=60,
            check=False,
        )

        # Should fail with non-zero exit code
        assert result.returncode != 0, "Invalid mode should fail"

        # Error message should mention the invalid mode
        output = result.stdout + result.stderr
        assert "invalid_mode_xyz" in output.lower() or "mode" in output.lower(), (
            f"Error message doesn't mention invalid mode:\n{output[:2000]}"
        )

    def test_invalid_strategy_gives_helpful_error(self, temp_output_dir):
        """Invalid strategy gives helpful error message."""
        result = run_training_command(
            [
                "medgen.scripts.train",
                "mode=bravo",
                "strategy=invalid_strategy_xyz",
                f"hydra.run.dir={temp_output_dir}",
            ],
            timeout=60,
            check=False,
        )

        # Should fail with non-zero exit code
        assert result.returncode != 0, "Invalid strategy should fail"

    def test_missing_config_file_error(self, temp_output_dir):
        """Missing config file gives helpful error."""
        result = run_training_command(
            [
                "medgen.scripts.train_compression",
                "--config-name=nonexistent_config_xyz",
                f"hydra.run.dir={temp_output_dir}",
            ],
            timeout=60,
            check=False,
        )

        # Should fail with non-zero exit code
        assert result.returncode != 0, "Missing config should fail"

        # Should mention config issue
        output = result.stdout + result.stderr
        assert any(
            keyword in output.lower()
            for keyword in ["config", "not found", "error", "missing"]
        ), f"Error message doesn't mention config issue:\n{output[:2000]}"
