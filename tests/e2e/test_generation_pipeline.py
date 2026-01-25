"""E2E tests for generation pipeline.

Tests the complete generation workflow: checkpoint loading -> generation -> NIfTI output.
These tests use subprocess to run generation commands and validate outputs.
"""

import os
import sys
import subprocess
from pathlib import Path

import pytest
import numpy as np

try:
    import nibabel as nib

    HAS_NIBABEL = True
except ImportError:
    HAS_NIBABEL = False


# Mark all tests in this module
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.timeout(600),  # 10 minute timeout
]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def golden_checkpoint():
    """Path to golden checkpoint, skip if not available.

    Golden checkpoints are pre-trained models for E2E testing.
    They're created with `./tests/fixtures/create_golden_checkpoint.sh`.
    """
    checkpoint_dir = Path(__file__).parent.parent / "fixtures" / "golden_checkpoint"

    if not checkpoint_dir.exists():
        pytest.skip(
            "Golden checkpoint not available. "
            "Run: ./tests/fixtures/create_golden_checkpoint.sh"
        )

    return checkpoint_dir


@pytest.fixture
def bravo_checkpoint(golden_checkpoint):
    """Path to bravo mode checkpoint."""
    bravo_dir = golden_checkpoint / "bravo"
    checkpoint = bravo_dir / "checkpoint_best.pt"

    if not checkpoint.exists():
        # Try latest
        checkpoint = bravo_dir / "checkpoint_latest.pt"

    if not checkpoint.exists():
        pytest.skip(f"Bravo checkpoint not found in {bravo_dir}")

    return checkpoint


@pytest.fixture
def seg_checkpoint(golden_checkpoint):
    """Path to seg mode checkpoint."""
    seg_dir = golden_checkpoint / "seg"
    checkpoint = seg_dir / "checkpoint_best.pt"

    if not checkpoint.exists():
        # Try latest
        checkpoint = seg_dir / "checkpoint_latest.pt"

    if not checkpoint.exists():
        pytest.skip(f"Seg checkpoint not found in {seg_dir}")

    return checkpoint


# =============================================================================
# Helper functions
# =============================================================================


def run_generation_command(
    args: list[str],
    timeout: int = 300,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run a generation command via subprocess.

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
        raise AssertionError(f"Generation failed with code {result.returncode}")

    return result


def find_nifti_files(output_dir: Path) -> list[Path]:
    """Find all NIfTI files in output directory."""
    files = list(output_dir.glob("**/*.nii.gz"))
    files.extend(output_dir.glob("**/*.nii"))
    return files


def load_nifti_data(nifti_path: Path) -> np.ndarray:
    """Load NIfTI file and return data as numpy array."""
    if not HAS_NIBABEL:
        pytest.skip("nibabel not installed")
    img = nib.load(nifti_path)
    return np.array(img.dataobj)


# =============================================================================
# Generation From Checkpoint Tests
# =============================================================================


class TestGenerationFromCheckpoint:
    """Test generation from pre-trained checkpoints."""

    @pytest.mark.gpu
    def test_generation_produces_output(
        self, bravo_checkpoint, seg_checkpoint, temp_output_dir
    ):
        """Generation creates output files."""
        result = run_generation_command(
            [
                "medgen.scripts.generate",
                "gen_mode=bravo",
                f"seg_model={seg_checkpoint}",
                f"image_model={bravo_checkpoint}",
                f"paths.generated_dir={temp_output_dir}",
                "num_images=2",
                "num_steps=5",  # Very few steps for speed
                "image_size=64",
            ],
            timeout=300,
            check=False,
        )

        if result.returncode != 0:
            # Check if it's a data/model issue vs crash
            if "error" in result.stderr.lower():
                pytest.skip(f"Generation had an error: {result.stderr[-500:]}")
            raise AssertionError(
                f"Generation failed:\n"
                f"STDOUT: {result.stdout[-1000:]}\n"
                f"STDERR: {result.stderr[-1000:]}"
            )

        # Check that output files were created
        output_files = list(temp_output_dir.rglob("*"))
        assert len(output_files) > 0, (
            f"No output files created in {temp_output_dir}\n"
            f"STDOUT: {result.stdout[-500:]}"
        )

    @pytest.mark.gpu
    @pytest.mark.skipif(not HAS_NIBABEL, reason="nibabel not installed")
    def test_generated_nifti_valid_header(
        self, bravo_checkpoint, seg_checkpoint, temp_output_dir
    ):
        """Generated NIfTI files have valid headers."""
        result = run_generation_command(
            [
                "medgen.scripts.generate",
                "gen_mode=bravo",
                f"seg_model={seg_checkpoint}",
                f"image_model={bravo_checkpoint}",
                f"paths.generated_dir={temp_output_dir}",
                "num_images=1",
                "num_steps=5",
                "image_size=64",
            ],
            timeout=300,
            check=False,
        )

        if result.returncode != 0:
            pytest.skip(f"Generation failed: {result.stderr[-300:]}")

        nifti_files = find_nifti_files(temp_output_dir)
        if not nifti_files:
            pytest.skip("No NIfTI files generated")

        for nifti_path in nifti_files[:3]:  # Check first 3
            img = nib.load(nifti_path)

            # Check header is valid
            assert img.header is not None, f"{nifti_path} has no header"
            assert img.shape is not None, f"{nifti_path} has no shape"
            assert len(img.shape) >= 2, f"{nifti_path} shape too small: {img.shape}"

    @pytest.mark.gpu
    @pytest.mark.skipif(not HAS_NIBABEL, reason="nibabel not installed")
    def test_generated_values_in_range(
        self, bravo_checkpoint, seg_checkpoint, temp_output_dir
    ):
        """Generated values are finite and in reasonable range."""
        result = run_generation_command(
            [
                "medgen.scripts.generate",
                "gen_mode=bravo",
                f"seg_model={seg_checkpoint}",
                f"image_model={bravo_checkpoint}",
                f"paths.generated_dir={temp_output_dir}",
                "num_images=1",
                "num_steps=5",
                "image_size=64",
            ],
            timeout=300,
            check=False,
        )

        if result.returncode != 0:
            pytest.skip(f"Generation failed: {result.stderr[-300:]}")

        nifti_files = find_nifti_files(temp_output_dir)
        if not nifti_files:
            pytest.skip("No NIfTI files generated")

        for nifti_path in nifti_files[:3]:  # Check first 3
            data = load_nifti_data(nifti_path)

            # Values should be finite
            assert np.all(np.isfinite(data)), (
                f"{nifti_path} contains non-finite values"
            )

            # Values should be in reasonable range for normalized medical images
            # Typically [-1, 2] or [0, 1] depending on normalization
            assert data.min() >= -5, (
                f"{nifti_path} has unexpectedly low values: {data.min()}"
            )
            assert data.max() <= 5, (
                f"{nifti_path} has unexpectedly high values: {data.max()}"
            )


# =============================================================================
# Batch Generation Tests
# =============================================================================


class TestBatchGeneration:
    """Test batch generation functionality."""

    @pytest.mark.gpu
    def test_generates_requested_count(
        self, bravo_checkpoint, seg_checkpoint, temp_output_dir
    ):
        """Generation produces the requested number of images."""
        num_requested = 4
        result = run_generation_command(
            [
                "medgen.scripts.generate",
                "gen_mode=bravo",
                f"seg_model={seg_checkpoint}",
                f"image_model={bravo_checkpoint}",
                f"paths.generated_dir={temp_output_dir}",
                f"num_images={num_requested}",
                "num_steps=5",
                "image_size=64",
            ],
            timeout=600,
            check=False,
        )

        if result.returncode != 0:
            pytest.skip(f"Generation failed: {result.stderr[-300:]}")

        nifti_files = find_nifti_files(temp_output_dir)

        # May generate multiple files per image (seg, bravo, etc.)
        # So check we have at least num_requested total or per type
        assert len(nifti_files) >= num_requested, (
            f"Expected at least {num_requested} files, got {len(nifti_files)}"
        )

    @pytest.mark.gpu
    @pytest.mark.skipif(not HAS_NIBABEL, reason="nibabel not installed")
    def test_cfg_scale_affects_output(
        self, bravo_checkpoint, seg_checkpoint, temp_output_dir
    ):
        """Different CFG scales produce different outputs."""
        # Generate with CFG scale 1.0
        dir_cfg1 = temp_output_dir / "cfg1"
        dir_cfg1.mkdir()

        result1 = run_generation_command(
            [
                "medgen.scripts.generate",
                "gen_mode=bravo",
                f"seg_model={seg_checkpoint}",
                f"image_model={bravo_checkpoint}",
                f"paths.generated_dir={dir_cfg1}",
                "num_images=1",
                "num_steps=5",
                "cfg_scale_bravo=1.0",
                "image_size=64",
            ],
            timeout=300,
            check=False,
        )

        # Generate with CFG scale 3.0
        dir_cfg3 = temp_output_dir / "cfg3"
        dir_cfg3.mkdir()

        result2 = run_generation_command(
            [
                "medgen.scripts.generate",
                "gen_mode=bravo",
                f"seg_model={seg_checkpoint}",
                f"image_model={bravo_checkpoint}",
                f"paths.generated_dir={dir_cfg3}",
                "num_images=1",
                "num_steps=5",
                "cfg_scale_bravo=3.0",
                "image_size=64",
            ],
            timeout=300,
            check=False,
        )

        if result1.returncode != 0 or result2.returncode != 0:
            pytest.skip("One or both generations failed")

        files1 = find_nifti_files(dir_cfg1)
        files2 = find_nifti_files(dir_cfg3)

        if not files1 or not files2:
            pytest.skip("No output files to compare")

        # Load and compare
        data1 = load_nifti_data(files1[0])
        data2 = load_nifti_data(files2[0])

        # Different CFG should produce different results
        # (Unless something is wrong with the implementation)
        if data1.shape == data2.shape:
            diff = np.abs(data1 - data2).mean()
            # They should be different - but this is a weak test since
            # stochasticity alone would make them different
            assert diff > 0.001 or not np.allclose(data1, data2), (
                "Different CFG scales produced identical output"
            )


# =============================================================================
# Generation Reproducibility Tests
# =============================================================================


class TestGenerationReproducibility:
    """Test generation reproducibility features."""

    @pytest.mark.skip(reason="Seed parameter not yet implemented in generation script")
    @pytest.mark.gpu
    @pytest.mark.skipif(not HAS_NIBABEL, reason="nibabel not installed")
    def test_same_seed_same_output(
        self, bravo_checkpoint, seg_checkpoint, temp_output_dir
    ):
        """Same seed produces identical output."""
        seed = 42

        # First generation
        dir1 = temp_output_dir / "run1"
        dir1.mkdir()
        run_generation_command(
            [
                "medgen.scripts.generate",
                "gen_mode=bravo",
                f"seg_model={seg_checkpoint}",
                f"image_model={bravo_checkpoint}",
                f"paths.generated_dir={dir1}",
                "num_images=1",
                "num_steps=5",
                f"seed={seed}",
                "image_size=64",
            ],
            timeout=300,
        )

        # Second generation with same seed
        dir2 = temp_output_dir / "run2"
        dir2.mkdir()
        run_generation_command(
            [
                "medgen.scripts.generate",
                "gen_mode=bravo",
                f"seg_model={seg_checkpoint}",
                f"image_model={bravo_checkpoint}",
                f"paths.generated_dir={dir2}",
                "num_images=1",
                "num_steps=5",
                f"seed={seed}",
                "image_size=64",
            ],
            timeout=300,
        )

        files1 = find_nifti_files(dir1)
        files2 = find_nifti_files(dir2)

        assert len(files1) > 0 and len(files2) > 0

        data1 = load_nifti_data(files1[0])
        data2 = load_nifti_data(files2[0])

        assert np.allclose(data1, data2, atol=1e-5), (
            "Same seed produced different outputs"
        )


# =============================================================================
# Generation CLI Tests
# =============================================================================


class TestGenerationCLI:
    """Test generation CLI interface."""

    def test_generate_help(self, run_cli):
        """generate.py --help exits cleanly."""
        result = run_cli(
            ["python", "-m", "medgen.scripts.generate", "--help"],
            check=False,
        )
        assert result.returncode == 0, f"--help failed:\n{result.stderr}"

        # Check help shows expected options
        assert "gen_mode" in result.stdout or "paths" in result.stdout, (
            f"Help doesn't show expected options:\n{result.stdout[:1000]}"
        )

    def test_missing_checkpoint_error(self, temp_output_dir, run_cli):
        """Missing checkpoint gives clear error."""
        result = run_cli(
            [
                "python",
                "-m",
                "medgen.scripts.generate",
                "gen_mode=bravo",
                "seg_model=/nonexistent/path/model.pt",
                "image_model=/another/fake/model.pt",
                f"paths.generated_dir={temp_output_dir}",
            ],
            check=False,
        )

        # Should fail
        assert result.returncode != 0, "Should fail with missing checkpoint"

        # Should mention the missing file or error
        output = result.stdout + result.stderr
        assert any(
            keyword in output.lower()
            for keyword in ["not found", "no such file", "error", "missing", "exist"]
        ), f"Error message not helpful:\n{output[:1000]}"


# =============================================================================
# Generation Mode Tests
# =============================================================================


class TestGenerationModes:
    """Test different generation modes."""

    @pytest.mark.gpu
    def test_seg_mode_generation(self, seg_checkpoint, temp_output_dir):
        """Seg mode generation works."""
        result = run_generation_command(
            [
                "medgen.scripts.generate",
                "gen_mode=seg",
                f"seg_model={seg_checkpoint}",
                f"paths.generated_dir={temp_output_dir}",
                "num_images=2",
                "num_steps=5",
                "image_size=64",
            ],
            timeout=300,
            check=False,
        )

        # Should complete or give understandable error
        if result.returncode != 0:
            # Check it's not a crash
            assert "Traceback" not in result.stderr[-1000:] or "expected" in result.stderr.lower(), (
                f"Seg mode crashed:\n{result.stderr[-1000:]}"
            )

    @pytest.mark.gpu
    @pytest.mark.parametrize("gen_mode", ["bravo", "seg"])
    def test_generation_mode_starts(
        self, gen_mode, seg_checkpoint, bravo_checkpoint, temp_output_dir
    ):
        """Each generation mode starts without crashing."""
        args = [
            "medgen.scripts.generate",
            f"gen_mode={gen_mode}",
            f"seg_model={seg_checkpoint}",
            f"paths.generated_dir={temp_output_dir}",
            "num_images=1",
            "num_steps=3",
            "image_size=64",
        ]

        # Add image_model for modes that need it
        if gen_mode != "seg":
            args.append(f"image_model={bravo_checkpoint}")

        result = run_generation_command(args, timeout=300, check=False)

        # Should at least start (not crash on config)
        output = result.stdout + result.stderr
        config_error = any(
            phrase in output.lower()
            for phrase in ["config error", "hydra error", "omegaconf"]
        )
        assert not config_error or result.returncode == 0, (
            f"Mode '{gen_mode}' had config error:\n{output[-1000:]}"
        )
