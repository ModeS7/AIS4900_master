"""Test full generation pipeline: checkpoint -> generate -> save NIfTI."""
import pytest


@pytest.mark.timeout(300)
@pytest.mark.e2e
@pytest.mark.slow
@pytest.mark.gpu
class TestGenerationPipeline:
    """Test complete generation workflow."""

    def test_generate_help(self, run_cli):
        """generate.py --help exits cleanly."""
        result = run_cli([
            'python', '-m', 'medgen.scripts.generate', '--help'
        ], check=False)
        assert result.returncode == 0

    def test_generate_with_checkpoint(self, temp_output_dir):
        """Generate samples from checkpoint."""
        pytest.skip("Requires checkpoint - implement when available")

    def test_generated_nifti_valid(self, temp_output_dir):
        """Generated NIfTI files are valid."""
        pytest.skip("Requires generation output - implement when available")
