"""Static contracts for exp17 evaluation using the proven exp16_2 workflow."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SLURM_DIR = ROOT / "IDUN" / "train" / "downstream" / "nnunet"

EVALUATION_JOBS = {
    "eval_5fold_exp17_1_hybrid_25syn_exp1_to_exp48c_t025_d663.slurm": (
        25,
        "exp17_1_hybrid_25syn_exp1_to_exp48c_t025_d663",
        "eval17_h25",
    ),
    "eval_5fold_exp17_2_hybrid_50syn_exp1_to_exp48c_t025_d663.slurm": (
        50,
        "exp17_2_hybrid_50syn_exp1_to_exp48c_t025_d663",
        "eval17_h50",
    ),
    "eval_5fold_exp17_3_hybrid_105syn_exp1_to_exp48c_t025_d663.slurm": (
        105,
        "exp17_3_hybrid_105syn_exp1_to_exp48c_t025_d663",
        "eval17_h105",
    ),
    "eval_5fold_exp17_4_hybrid_210syn_exp1_to_exp48c_t025_d663.slurm": (
        210,
        "exp17_4_hybrid_210syn_exp1_to_exp48c_t025_d663",
        "eval17_h210",
    ),
}
METRICS_JOB = SLURM_DIR / "recompute_exp17_hybrid_all_metrics.slurm"


def _read(filename: str) -> str:
    path = SLURM_DIR / filename
    assert path.is_file(), f"missing exp17 evaluator: {path}"
    return path.read_text()


def _assert_bash_syntax(path: Path) -> None:
    result = subprocess.run(
        ["bash", "-n", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_exp17_has_four_independent_exp16_style_evaluation_jobs() -> None:
    assert len(EVALUATION_JOBS) == 4

    for filename, (dose, experiment, job_name) in EVALUATION_JOBS.items():
        path = SLURM_DIR / filename
        text = _read(filename)
        _assert_bash_syntax(path)

        assert text.startswith("#!/usr/bin/env bash\n")
        assert "set -Eeuo pipefail" in text
        assert "#SBATCH --time=0-00:30:00" in text
        assert '#SBATCH --constraint="a100|h100|gpu80g"' in text
        assert "#SBATCH --array" not in text
        assert f'#SBATCH --job-name="{job_name}"' in text
        assert f"readonly N_SYNTHETIC={dose}" in text
        assert "readonly DATASET_ID=663" in text
        assert f'readonly EXPERIMENT_NAME="{experiment}"' in text
        assert f"{experiment}_%j.out" in text
        assert f"{experiment}_%j.err" in text
        assert "--experiment mixed" in text
        assert '--n-synthetic "$N_SYNTHETIC"' in text


def test_exp17_evaluators_require_the_exact_complete_five_fold_model() -> None:
    for filename in EVALUATION_JOBS:
        text = _read(filename)

        assert 'readonly TRAINER="nnUNetTrainerBrainMets"' in text
        assert 'readonly PLANS_NAME="nnUNetResEncUNetLPlansD600"' in text
        assert 'readonly CONFIGURATION="3d_fullres"' in text
        assert "for fold in 0 1 2 3 4; do" in text
        assert '[[ -s "${fold_dir}/checkpoint_best.pth" ]]' in text
        assert '[[ -s "${fold_dir}/checkpoint_final.pth" ]]' in text
        assert "--folds 0 1 2 3 4" in text
        assert '--trainer "$TRAINER"' in text
        assert '--plans "$PLANS_NAME"' in text


def test_exp17_evaluators_use_the_same_direct_canonical_output_contract() -> None:
    for filename in EVALUATION_JOBS:
        text = _read(filename)

        assert 'readonly EVAL_DIR="${EXPERIMENT_RESULTS}/eval_${EXPERIMENT_NAME}"' in text
        assert 'readonly PREDICTIONS_DIR="${EVAL_DIR}/predictions"' in text
        assert (
            'readonly EVAL_JSON="${EXPERIMENT_RESULTS}/eval_${EXPERIMENT_NAME}.json"'
            in text
        )
        assert 'readonly COMPLETE_MARKER="${EVAL_DIR}/.exp17_5fold_eval_complete"' in text
        assert 'mkdir -p "$EVAL_DIR"' in text
        assert 'rm -f -- "$COMPLETE_MARKER"' in text
        assert "--output-dir" not in text
        assert ".incomplete_" not in text
        assert 'mv -f -- "$marker_tmp" "$COMPLETE_MARKER"' in text


def test_exp17_evaluators_verify_the_exact_official_51_ids() -> None:
    for filename in EVALUATION_JOBS:
        text = _read(filename)

        assert "readonly EXPECTED_CASES=51" in text
        assert "Dataset600_BrainMet/labelsTs" in text
        assert '"$RAW_DIR/imagesTs" "$RAW_DIR/labelsTs" "$OFFICIAL_LABELS_DIR"' in text
        assert "if image_ids != label_ids:" in text
        assert "label_ids != official_label_ids" in text
        assert '"$PREDICTIONS_DIR" "$OFFICIAL_LABELS_DIR"' in text
        assert "if prediction_ids != label_ids:" in text
        assert '[[ "$prediction_count" == "$EXPECTED_CASES" ]]' in text
        assert '[[ -s "$EVAL_JSON" ]]' in text

        inference = text.index("time python -m medgen.scripts.eval_nnunet")
        prediction_check = text.index("if prediction_ids != label_ids:")
        marker_write = text.index('marker_tmp="${COMPLETE_MARKER}.tmp_${JOB_ID}"')
        assert inference < prediction_check < marker_write


def test_exp17_hybrid_metrics_uses_all_four_test_arms_as_one_paired_family() -> None:
    text = METRICS_JOB.read_text()
    flat = " ".join(text.replace("\\\n", " ").split())
    _assert_bash_syntax(METRICS_JOB)

    assert "#SBATCH --partition=CPUQ" in text
    assert "#SBATCH --gres" not in text
    assert "nvidia-smi" not in text
    assert "medgen.scripts.eval_nnunet" not in text
    assert "medgen.scripts.recompute_nnunet_metrics" in text
    assert 'readonly BASELINE_EXPERIMENT="exp3_baseline_v2_d600"' in text
    assert 'readonly MARKER_NAME=".exp17_5fold_eval_complete"' in text
    assert "readonly EXPECTED_CASES=51" in text
    assert 'readonly EXPECTED_FOLDS="0,1,2,3,4"' in text
    assert "--baseline real_reference" in flat
    assert "--min-component-voxels 5" in flat
    assert "--detection-threshold 0.1" in flat
    assert "--bootstrap-draws 10000" in flat
    assert "--seed 0" in flat
    assert 'COMPARE_ARGS+=(--compare "$condition_label")' in text

    for label, (dose, experiment, _job_name) in zip(
        ("hybrid_25", "hybrid_50", "hybrid_105", "hybrid_210"),
        EVALUATION_JOBS.values(),
        strict=True,
    ):
        assert label in text
        assert str(dose) in text
        assert experiment in text


def test_exp17_hybrid_metrics_exports_all_audit_tables() -> None:
    text = METRICS_JOB.read_text()

    for output in (
        "summary.json",
        "summary.csv",
        "per_case.csv",
        "per_lesion.csv",
        "paired_statistics.csv",
    ):
        assert output in text
    for key in (
        "experiment",
        "dataset",
        "folds",
        "prediction_count",
        "stdout",
        "stderr",
    ):
        assert f'marker_value "$marker" {key}' in text
