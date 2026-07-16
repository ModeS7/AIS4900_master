"""Test SLURM files reference valid Python modules.

This catches deployment errors where SLURM scripts reference non-existent
Python modules (e.g., train_dcae instead of train_compression).
"""

import importlib.util
import re
import subprocess
from pathlib import Path

import pytest


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


class TestFixedGeneratorPanelSlurm:
    """Lock the leak-free 14-generator panel's cluster protocol."""

    EXPECTED = {
        "eval_gen_exp1_1_1000.slurm": (
            "exp1_1_1000",
            "exp1_1_1000_pixel_bravo_20260402-121556",
            None,
        ),
        "eval_gen_exp1_1_1000plus.slurm": (
            "exp1_1_1000plus",
            "exp1_1_1000plus_pixel_bravo_20260411-235425",
            None,
        ),
        "eval_gen_exp32_2_1000.slurm": (
            "exp32_2_1000",
            "exp32_2_1000_pixel_bravo_lpips_lowt_20260412-153027",
            None,
        ),
        "eval_gen_exp32_3_1000.slurm": (
            "exp32_3_1000",
            "exp32_3_1000_pixel_bravo_pseudo_huber_20260415-041057",
            None,
        ),
        "eval_gen_exp47a.slurm": ("exp47a", "exp47a_lpips_strong_20260425-055252", None),
        "eval_gen_exp47b.slurm": ("exp47b", "exp47b_huber_lpips_lowt_20260425-212907", None),
        "eval_gen_exp47c.slurm": ("exp47c", "exp47c_lpips_huber_builtin_20260425-223819", None),
        "eval_gen_exp47d.slurm": ("exp47d", "exp47d_compression_recipe_20260426-022039", None),
        "eval_gen_exp47e.slurm": ("exp47e", "exp47e_perceptual_only_lowt_20260427-074551", None),
        "eval_gen_exp48a.slurm": (
            "exp1_to_exp48a_t025",
            "exp48a_lowt_only_lpips_strong_20260425-160342",
            "exp1_1_1000_pixel_bravo_20260402-121556",
        ),
        "eval_gen_exp48b.slurm": (
            "exp1_to_exp48b_t025",
            "exp48b_lowt_only_l1_lpips_20260425-202256",
            "exp1_1_1000_pixel_bravo_20260402-121556",
        ),
        "eval_gen_exp48c.slurm": (
            "exp1_to_exp48c_t025",
            "exp48c_lowt_only_lpips_huber_20260425-162615",
            "exp1_1_1000_pixel_bravo_20260402-121556",
        ),
        "eval_gen_exp48d.slurm": (
            "exp1_to_exp48d_t025",
            "exp48d_lowt_only_huber_lpips_20260425-172756",
            "exp1_1_1000_pixel_bravo_20260402-121556",
        ),
        "eval_gen_exp48e.slurm": (
            "exp1_to_exp48e_t025",
            "exp48e_lowt_only_perceptual_floor_20260427-014900",
            "exp1_1_1000_pixel_bravo_20260402-121556",
        ),
    }

    @pytest.fixture
    def generate_dir(self):
        return Path(__file__).parent.parent.parent / "IDUN" / "generate"

    def test_exactly_14_thin_wrappers_pin_the_declared_runs(self, generate_dir):
        wrappers = {path.name: path for path in generate_dir.glob("eval_gen_exp*.slurm")}
        assert set(wrappers) == set(self.EXPECTED)

        invocation = re.compile(
            r'run_eval_generator\.sh"\s+\\\n\s+(\S+)\s+\\\n\s+(\S+)'
            r"(?:\s+\\\n\s+(\S+)\s+\\\n\s+(0\.25))?\s*$"
        )
        for name, (label, low_run, high_run) in self.EXPECTED.items():
            content = wrappers[name].read_text()
            assert content.startswith("#!/usr/bin/env bash\n")
            assert "set -Eeuo pipefail" in content
            assert content.count("run_eval_generator.sh") == 1
            assert "medgen.scripts.generate" not in content
            assert "medgen.scripts.eval_all_metrics" not in content
            match = invocation.search(content)
            assert match, f"Could not parse thin-wrapper invocation in {name}"
            assert match.group(1) == label
            assert match.group(2) == low_run
            assert match.group(3) == high_run
            assert match.group(4) == ("0.25" if high_run else None)

    def test_shared_runner_locks_protocol_and_fails_closed(self, generate_dir):
        content = (generate_dir / "run_eval_generator.sh").read_text()
        required = (
            "set -Eeuo pipefail",
            "/checkpoint_latest.pt",
            "expected_strategy=rflow",
            "expected_real_cases=105",
            "expected_real_depth=150",
            "seed=42",
            "num_images=105",
            "current_image=0",
            "num_steps_bravo=100",
            "trim_slices=10",
            "fov_mm=240.0",
            "ode_solver=euler",
            "shift_ratio_bravo=1.0",
            "cfg_scale_bravo=1.0",
            "validate_size_bins=false",
            "validate_brain_mask=false",
            "brain_atlas_path=null",
            "brain_pca_path=null",
            "diffrs_checkpoint=null",
            "mask_outside_brain=false",
            "mask_outside_brain_dilate_pixels=0",
            "require_real_bravo_pairs=true",
            "validate_real_seg_masks=true",
            "provenance_hash_checkpoints=true",
            '[[ ! -e "$FINAL_DIR" ]]',
            'mv -- "$STAGING_DIR" "$FINAL_DIR"',
            "panel_job_manifest.json",
            "generation_manifest.json",
            "git status --porcelain --untracked-files=all",
            "--git-dirty false",
            "PANEL_SOURCE_COMMIT must be exported by the panel launcher",
            "verify_source_tree",
        )
        for token in required:
            assert token in content
        for forbidden in (
            "checkpoint_best.pt",
            "ls -td",
            "SIGUSR1",
            "sbatch",
            "-delete",
            "eval_all_metrics",
        ):
            assert forbidden not in content

    def test_combined_metric_job_is_train_only_and_checks_manifests(self, generate_dir):
        content = (generate_dir / "eval_generator_panel_metrics.slurm").read_text()
        assert content.startswith("#!/usr/bin/env bash\n")
        assert "set -Eeuo pipefail" in content
        assert content.count("python -m medgen.scripts.eval_all_metrics") == 1
        assert '--reference "train:${TRAIN_DIR}"' in content
        assert "--pca-model none" in content
        assert "--seed 42" in content
        assert "--pool-cap 0" in content
        assert "validate --panel-root" in content
        assert "validate-report --report" in content
        assert content.count("--expected-git-commit") == 2
        assert '--source-git-commit "$SOURCE_COMMIT"' in content
        assert "git status --porcelain --untracked-files=all" in content
        assert "PANEL_SOURCE_COMMIT must be exported by the panel launcher" in content
        assert content.count("verify_source_tree") >= 3
        assert content.index("validate --panel-root") < content.index(
            "medgen.scripts.eval_all_metrics"
        )
        assert content.index("validate-report --report") > content.index(
            "medgen.scripts.eval_all_metrics"
        )
        for label, _, _ in self.EXPECTED.values():
            assert re.search(rf"^\s*{re.escape(label)}\s*$", content, flags=re.MULTILINE)
        assert "test1" not in content
        assert "checkpoint_best" not in content

    def test_launcher_submits_afterok_dependency_and_dry_runs(self, generate_dir):
        launcher = generate_dir / "submit_eval_generator_panel.sh"
        content = launcher.read_text()
        assert "sbatch --parsable" in content
        assert '--dependency="afterok:${dependency}"' in content
        assert "eval_generator_panel_metrics.slurm" in content
        assert "git status --porcelain --untracked-files=all" in content
        assert "export PANEL_SOURCE_COMMIT" in content
        for wrapper_name in self.EXPECTED:
            assert content.count(f"IDUN/generate/{wrapper_name}") == 1

        result = subprocess.run(
            ["bash", str(launcher), "--dry-run"],
            cwd=generate_dir.parent.parent,
            check=True,
            capture_output=True,
            text=True,
        )
        assert result.stdout.count("sbatch --parsable IDUN/generate/eval_gen_") == 14
        assert "afterok:<14-colon-separated-job-ids>" in result.stdout

    def test_panel_shell_files_pass_bash_syntax_check(self, generate_dir):
        files = [
            *generate_dir.glob("eval_gen_exp*.slurm"),
            generate_dir / "run_eval_generator.sh",
            generate_dir / "eval_generator_panel_metrics.slurm",
            generate_dir / "submit_eval_generator_panel.sh",
        ]
        subprocess.run(["bash", "-n", *map(str, files)], check=True)
