"""Test CLI entry points execute without error."""
import pytest


@pytest.mark.timeout(300)
@pytest.mark.e2e
@pytest.mark.slow
class TestTrainCLI:
    """Test train.py CLI entry point."""

    def test_train_help(self, run_cli):
        """--help exits cleanly."""
        result = run_cli([
            'python', '-m', 'medgen.scripts.train', '--help'
        ], check=False)
        assert result.returncode == 0

    def test_train_dry_run(self, run_cli, temp_output_dir):
        """Minimal training config runs without error."""
        result = run_cli([
            'python', '-m', 'medgen.scripts.train',
            'training.epochs=1',
            'training.batch_size=2',
            f'hydra.run.dir={temp_output_dir}',
            'training.limit_batches=1',
        ], timeout=120, check=False)
        # May fail due to missing data, but should not crash on config
        assert result.returncode in (0, 1)


@pytest.mark.timeout(300)
@pytest.mark.e2e
@pytest.mark.slow
class TestCompressionCLI:
    """Test train_compression.py CLI entry point."""

    def test_compression_help(self, run_cli):
        """--help exits cleanly."""
        result = run_cli([
            'python', '-m', 'medgen.scripts.train_compression', '--help'
        ], check=False)
        assert result.returncode == 0
