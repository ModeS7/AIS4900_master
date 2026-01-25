"""E2E test fixtures - full configs, real models."""
import pytest
import tempfile
from pathlib import Path
import subprocess


@pytest.fixture(scope="module")
def project_root():
    """Project root directory."""
    return Path(__file__).parent.parent.parent


@pytest.fixture
def temp_output_dir():
    """Temporary directory for E2E test outputs."""
    with tempfile.TemporaryDirectory(prefix="medgen_e2e_") as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def run_cli():
    """Factory for running CLI commands."""
    def _run(args: list, timeout: int = 300, check: bool = True):
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if check and result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
        return result
    return _run
