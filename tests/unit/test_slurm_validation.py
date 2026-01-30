"""Test SLURM files reference valid Python modules.

This catches deployment errors where SLURM scripts reference non-existent
Python modules (e.g., train_dcae instead of train_compression).
"""

import pytest
import re
from pathlib import Path
import importlib.util


class TestSlurmScriptValidation:
    """Validate SLURM scripts reference existing modules."""

    @pytest.fixture
    def slurm_files(self):
        """Find all SLURM files in IDUN directory."""
        project_root = Path(__file__).parent.parent.parent
        idun_dir = project_root / "IDUN"
        if not idun_dir.exists():
            pytest.skip("IDUN directory not found")
        return list(idun_dir.rglob("*.slurm"))

    def test_slurm_files_exist(self, slurm_files):
        """Verify we found SLURM files to test."""
        assert len(slurm_files) > 0, "No SLURM files found in IDUN/"

    def test_python_modules_exist(self, slurm_files):
        """Verify all python -m commands reference existing modules.

        REGRESSION: Catches errors like 'medgen.scripts.train_dcae' when the
        actual module is 'medgen.scripts.train_compression'.
        """
        # Pattern to match: python -m module.name
        pattern = r"python\s+-m\s+([\w.]+)"
        errors = []

        for slurm_file in slurm_files:
            content = slurm_file.read_text()
            matches = re.findall(pattern, content)

            for module_name in matches:
                # Check if module exists
                spec = importlib.util.find_spec(module_name)
                if spec is None:
                    # Get relative path for cleaner error message
                    rel_path = slurm_file.relative_to(slurm_file.parent.parent.parent)
                    errors.append(f"{rel_path}: Module '{module_name}' not found")

        if errors:
            pytest.fail(
                f"Found {len(errors)} SLURM scripts with invalid module references:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    def test_config_name_references_exist(self, slurm_files):
        """Verify --config-name= references point to existing config files.

        REGRESSION: Catches errors where config names reference non-existent
        Hydra config files.
        """
        project_root = Path(__file__).parent.parent.parent
        configs_dir = project_root / "configs"

        if not configs_dir.exists():
            pytest.skip("configs directory not found")

        # Pattern to match: --config-name=something or --config-name something
        pattern = r"--config-name[=\s](\w+)"
        errors = []

        for slurm_file in slurm_files:
            content = slurm_file.read_text()
            matches = re.findall(pattern, content)

            for config_name in matches:
                # Check common config locations
                found = False
                for config_path in [
                    configs_dir / f"{config_name}.yaml",
                    configs_dir / config_name / "config.yaml",
                    configs_dir / "experiment" / f"{config_name}.yaml",
                ]:
                    if config_path.exists():
                        found = True
                        break

                if not found:
                    rel_path = slurm_file.relative_to(slurm_file.parent.parent.parent)
                    errors.append(f"{rel_path}: Config '{config_name}' not found")

        if errors:
            pytest.fail(
                f"Found {len(errors)} SLURM scripts with invalid config references:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )

    def test_mode_references_have_configs(self, slurm_files):
        """Verify mode= parameters reference existing config files.

        REGRESSION: Catches errors where mode=X is used but configs/mode/X.yaml
        doesn't exist, which causes Hydra to fail at runtime.
        """
        project_root = Path(__file__).parent.parent.parent
        configs_dir = project_root / "configs" / "mode"

        if not configs_dir.exists():
            pytest.skip("configs/mode directory not found")

        # Pattern to match: mode=something (alphanumeric + underscore)
        pattern = r"(?<![.\w])mode=([a-z0-9_]+)"
        errors = []

        for slurm_file in slurm_files:
            content = slurm_file.read_text()
            matches = re.findall(pattern, content)

            for mode_name in matches:
                # Check if config file exists
                config_path = configs_dir / f"{mode_name}.yaml"
                if not config_path.exists():
                    rel_path = slurm_file.relative_to(slurm_file.parent.parent.parent)
                    errors.append(
                        f"{rel_path}: mode='{mode_name}' has no config file. "
                        f"Expected: configs/mode/{mode_name}.yaml"
                    )

        if errors:
            pytest.fail(
                f"Found {len(errors)} SLURM scripts with missing mode configs:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )
