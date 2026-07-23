"""Static contracts for strict exp16_2 five-fold evaluation jobs."""

import re
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
SLURM_DIR = PROJECT_ROOT / "IDUN/train/downstream/nnunet"

EVALUATION_JOBS = {
    "eval_5fold_exp16_2_synthetic_105_common105_exp1_1_1000_d650.slurm": (
        "exp1_1_1000",
        650,
        "eval16_c105_e1",
    ),
    "eval_5fold_exp16_2_synthetic_105_common105_exp1_1_1000plus_d651.slurm": (
        "exp1_1_1000plus",
        651,
        "eval16_c105_e1p",
    ),
    "eval_5fold_exp16_2_synthetic_105_common105_exp32_2_1000_d652.slurm": (
        "exp32_2_1000",
        652,
        "eval16_c105_e32",
    ),
    "eval_5fold_exp16_2_synthetic_105_common105_exp47a_d654.slurm": (
        "exp47a",
        654,
        "eval16_c105_e47a",
    ),
    "eval_5fold_exp16_2_synthetic_105_common105_exp47c_d656.slurm": (
        "exp47c",
        656,
        "eval16_c105_e47c",
    ),
    "eval_5fold_exp16_2_synthetic_105_common105_exp1_to_exp48c_t025_d661.slurm": (
        "exp1_to_exp48c_t025",
        661,
        "eval16_c105_48c",
    ),
    "eval_5fold_exp16_2_synthetic_105_common105_exp1_to_exp48d_t025_d662.slurm": (
        "exp1_to_exp48d_t025",
        662,
        "eval16_c105_48d",
    ),
}

MARKER_KEYS = [
    "job_id",
    "stdout",
    "stderr",
    "experiment",
    "dataset",
    "folds",
    "prediction_count",
]
PANEL_METRICS_JOB = "recompute_exp16_2_synthetic_only_all_metrics.slurm"


def _read(filename: str) -> str:
    path = SLURM_DIR / filename
    assert path.is_file(), f"missing exp16_2 evaluator: {path}"
    return path.read_text()


def test_exp16_2_evaluation_jobs_have_exact_identities_and_shell_syntax() -> None:
    assert len(EVALUATION_JOBS) == 7

    for filename, (label, dataset_id, job_name) in EVALUATION_JOBS.items():
        path = SLURM_DIR / filename
        text = _read(filename)

        result = subprocess.run(
            ["bash", "-n", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        assert text.startswith("#!/usr/bin/env bash\n")
        assert "set -Eeuo pipefail" in text
        assert '#SBATCH --constraint="a100|h100|gpu80g"' in text
        assert "#SBATCH --array" not in text
        assert f'#SBATCH --job-name="{job_name}"' in text
        assert f"readonly DATASET_ID={dataset_id}" in text
        assert f'readonly LABEL="{label}"' in text
        assert (
            'readonly EXPERIMENT_NAME="exp16_2_synthetic_105_common105_${LABEL}_d${DATASET_ID}"'
        ) in text
        assert f"{filename.removeprefix('eval_5fold_').removesuffix('.slurm')}_%j.out" in text
        assert f"{filename.removeprefix('eval_5fold_').removesuffix('.slurm')}_%j.err" in text


def test_exp16_2_evaluation_jobs_require_the_exact_complete_five_fold_model() -> None:
    for filename in EVALUATION_JOBS:
        text = _read(filename)

        assert 'readonly TRAINER="nnUNetTrainerBrainMets"' in text
        assert 'readonly PLANS_NAME="nnUNetResEncUNetLPlansD600"' in text
        assert 'readonly CONFIGURATION="3d_fullres"' in text
        assert (
            'readonly MODEL_DIR="${EXPERIMENT_RESULTS}/Dataset${DATASET_ID}_BrainMet/'
            '${TRAINER}__${PLANS_NAME}__${CONFIGURATION}"'
        ) in text
        assert "for fold in 0 1 2 3 4; do" in text
        assert '[[ -s "${fold_dir}/checkpoint_best.pth" ]]' in text
        assert '[[ -s "${fold_dir}/checkpoint_final.pth" ]]' in text
        assert "--folds 0 1 2 3 4" in text
        assert '--trainer "$TRAINER"' in text
        assert '--plans "$PLANS_NAME"' in text
        assert "--experiment synthetic" in text
        assert "--n-synthetic 105" in text


def test_exp16_2_evaluation_jobs_verify_the_official_51_ids_before_marking() -> None:
    for filename in EVALUATION_JOBS:
        text = _read(filename)

        assert "readonly EXPECTED_CASES=51" in text
        assert (
            'readonly OFFICIAL_LABELS_DIR="${NNUNET_BASE}/nnUNet_raw/Dataset600_BrainMet/labelsTs"'
        ) in text
        assert '"$RAW_DIR/imagesTs" "$RAW_DIR/labelsTs" "$OFFICIAL_LABELS_DIR"' in text
        assert "if image_ids != label_ids:" in text
        assert "label_ids != official_label_ids" in text
        assert '"$PREDICTIONS_DIR" "$OFFICIAL_LABELS_DIR"' in text
        assert "if len(prediction_ids) != expected_cases:" in text
        assert "if prediction_ids != label_ids:" in text
        assert '[[ "$prediction_count" == "$EXPECTED_CASES" ]]' in text
        assert '[[ -s "$EVAL_JSON" ]]' in text

        eval_call = text.index("time python -m medgen.scripts.eval_nnunet")
        prediction_check = text.index("if prediction_ids != label_ids:")
        marker_write = text.index('marker_tmp="${COMPLETE_MARKER}.tmp_${JOB_ID}"')
        assert eval_call < prediction_check < marker_write


def test_exp16_2_evaluation_marker_is_atomic_and_has_only_stable_schema() -> None:
    for filename in EVALUATION_JOBS:
        text = _read(filename)

        assert 'readonly EVAL_DIR="${EXPERIMENT_RESULTS}/eval_${EXPERIMENT_NAME}"' in text
        assert ('readonly COMPLETE_MARKER="${EVAL_DIR}/.exp16_2_5fold_eval_complete"') in text
        assert 'rm -f -- "$COMPLETE_MARKER"' in text
        assert 'mv -f -- "$marker_tmp" "$COMPLETE_MARKER"' in text

        marker_tail = text.split(
            'marker_tmp="${COMPLETE_MARKER}.tmp_${JOB_ID}"',
            maxsplit=1,
        )[1]
        marker_block = marker_tail.split('} > "$marker_tmp"', maxsplit=1)[0]
        keys = re.findall(r'^\s*echo "([a-z_]+)=', marker_block, flags=re.MULTILINE)
        assert keys == MARKER_KEYS


def test_exp16_2_complete_metric_panel_is_cpu_only_and_uses_all_seven_markers() -> None:
    path = SLURM_DIR / PANEL_METRICS_JOB
    text = path.read_text()
    flat = " ".join(text.replace("\\\n", " ").split())

    result = subprocess.run(
        ["bash", "-n", str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "#SBATCH --partition=CPUQ" in text
    assert "#SBATCH --gres" not in text
    assert "nvidia-smi" not in text
    assert "medgen.scripts.eval_nnunet" not in text
    assert "medgen.scripts.recompute_nnunet_metrics" in text
    assert 'readonly MARKER_NAME=".exp16_2_5fold_eval_complete"' in text
    assert "readonly EXPECTED_CASES=51" in text
    assert 'readonly EXPECTED_FOLDS="0,1,2,3,4"' in text
    assert 'readonly BASELINE_EXPERIMENT="exp3_baseline_v2_d600"' in text
    assert "--baseline real_reference" in flat
    assert "--min-component-voxels 5" in flat
    assert "--detection-threshold 0.1" in flat
    assert "--bootstrap-draws 10000" in flat

    for _filename, (generator, dataset_id, _) in EVALUATION_JOBS.items():
        assert generator in text
        assert str(dataset_id) in text

    for label in (
        "original_mse",
        "extended_mse",
        "perceptual_continuation",
        "strong_perceptual_continuation",
        "weighted_huber_transition",
        "weighted_huber_handoff",
        "pseudo_huber_perceptual_handoff",
    ):
        assert label in text


def test_exp16_2_complete_metric_panel_exports_every_audit_table() -> None:
    text = _read(PANEL_METRICS_JOB)

    for output in (
        "summary.json",
        "summary.csv",
        "per_case.csv",
        "per_lesion.csv",
        "paired_statistics.csv",
    ):
        assert output in text
    for marker_key in MARKER_KEYS:
        assert f'marker_value "$marker" {marker_key}' in text or marker_key == "job_id"
