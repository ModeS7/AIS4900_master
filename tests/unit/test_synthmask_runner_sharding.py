"""Guardrails for fixed-mask generator shard submission."""

import os
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[2]
RUNNER = PROJECT_ROOT / "IDUN/generate/run_eval_generator_synthmask.sh"


def test_partial_range_cannot_write_to_canonical_label() -> None:
    env = os.environ.copy()
    env["SYNTHMASK_START_CANDIDATE"] = "300"
    env.pop("SYNTHMASK_OUTPUT_LABEL", None)

    result = subprocess.run(
        ["bash", str(RUNNER), "exp1_to_exp48a_t025", "placeholder_low_run"],
        cwd=PROJECT_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "a partial candidate range must use a separate SYNTHMASK_OUTPUT_LABEL" in result.stderr
