"""Static contracts for the controlled exp16_2 common-105 nnU-Net panel."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[2]
NNUNET_SLURM_DIR = PROJECT_ROOT / "IDUN/train/downstream/nnunet"
METRICS_SLURM = (
    PROJECT_ROOT
    / "IDUN/generate/eval_generator_synthmask_metrics_all14_common96.slurm"
)
CORE5_METRICS_SLURM = (
    PROJECT_ROOT / "IDUN/generate/eval_generator_synthmask_metrics.slurm"
)

CORE5_TRAINING_JOBS = {
    "exp16_2_synthetic_105_common105_exp1_1_1000_d650.slurm": (
        "exp1_1_1000",
        650,
    ),
    "exp16_2_synthetic_105_common105_exp1_1_1000plus_d651.slurm": (
        "exp1_1_1000plus",
        651,
    ),
    "exp16_2_synthetic_105_common105_exp32_2_1000_d652.slurm": (
        "exp32_2_1000",
        652,
    ),
    "exp16_2_synthetic_105_common105_exp1_to_exp48c_t025_d661.slurm": (
        "exp1_to_exp48c_t025",
        661,
    ),
    "exp16_2_synthetic_105_common105_exp1_to_exp48d_t025_d662.slurm": (
        "exp1_to_exp48d_t025",
        662,
    ),
}
ADDITIONAL_TRAINING_JOBS = {
    "exp16_2_synthetic_105_common105_exp47a_d654.slurm": (
        "exp47a",
        654,
    ),
    "exp16_2_synthetic_105_common105_exp47c_d656.slurm": (
        "exp47c",
        656,
    ),
}
TRAINING_JOBS = {**CORE5_TRAINING_JOBS, **ADDITIONAL_TRAINING_JOBS}

MASK_MARKER_NAME = ".brain_mask_stage3_all14_complete"
PREPROCESS_MARKER_NAME = ".exp16_2_d600_preprocess_complete"
CONTROLLED_PLANS = "nnUNetResEncUNetLPlansD600"


def _read(path: Path) -> str:
    assert path.is_file(), f"missing SLURM file: {path}"
    return path.read_text()


def _flatten_shell(text: str) -> str:
    """Make a continued shell command suitable for exact token assertions."""

    return " ".join(text.replace("\\\n", " ").split())


def _assert_strict_mask_gate(text: str, marker_variable: str) -> None:
    marker_reference = f'"${marker_variable}"'
    assert MASK_MARKER_NAME in text
    assert f'[[ -s {marker_reference} ]]' in text
    for key, value in (
        ("mode", "in_place_atomic"),
        ("threshold", "0.05"),
        ("fill_holes", "true"),
        ("dilate_pixels", "2"),
    ):
        require_call = f"require_marker_value {marker_reference} {key} {value}"
        grep_call = f"grep -Fxq '{key}={value}' {marker_reference}"
        assert require_call in text or grep_call in text


def test_exp16_2_training_jobs_keep_the_controlled_d600_contract() -> None:
    assert len(CORE5_TRAINING_JOBS) == 5
    assert len(TRAINING_JOBS) == 7

    for filename, (label, dataset_id) in TRAINING_JOBS.items():
        text = _read(NNUNET_SLURM_DIR / filename)

        assert "#SBATCH --time=0-12:00:00" in text
        assert '#SBATCH --constraint="a100|h100|gpu80g"' in text
        assert "#SBATCH --array=0-4%5" in text
        assert f"readonly DATASET_ID={dataset_id}" in text
        assert f'readonly LABEL="{label}"' in text
        assert (
            f'readonly SCRIPT_RELATIVE="IDUN/train/downstream/nnunet/{filename}"'
            in text
        )
        assert f'readonly PLANS_NAME="{CONTROLLED_PLANS}"' in text
        assert (
            'readonly MODEL_NAME="nnUNetTrainerBrainMets__'
            '${PLANS_NAME}__3d_fullres"'
        ) in text
        assert (
            'readonly PREPROCESS_MARKER="${PREPROCESSED_DIR}/'
            f'{PREPROCESS_MARKER_NAME}"'
        ) in text
        assert '[[ -s "$PREPROCESS_MARKER" ]]' in text
        assert '--plans "$PLANS_NAME"' in text
        assert 'marker.get("target_plan_sha256")' in text
        assert "hashlib.sha256(plan_path.read_bytes()).hexdigest()" in text
        assert "--seed" not in text
        assert "readonly SEED" not in text


def test_exp47c_preparation_selects_existing_frozen_cases_and_runs_no_generation() -> None:
    text = _read(
        NNUNET_SLURM_DIR / "prepare_synthmask_common105_exp47c_d656.slurm"
    )

    assert "#SBATCH --partition=CPUQ" in text
    assert "#SBATCH --time=0-12:00:00" in text
    assert 'readonly LABEL="exp47c"' in text
    assert "readonly DATASET_ID=656" in text
    assert "readonly EXPECTED_CASES=105" in text
    assert (
        'readonly CANDIDATE_MASKS="${EVAL_ROOT}/'
        'ordered_common_seg_masks_seed42/candidates525"'
    ) in text
    assert 'readonly COMMON_IDS_FILE="${MATCHED_ROOT}/common_candidate_ids.txt"' in text
    assert 'readonly MATCHED_VIEW="${MATCHED_ROOT}/${LABEL}"' in text
    _assert_strict_mask_gate(text, "PANEL_MASK_MARKER")
    for key, value in (
        ("threshold", "0.05"),
        ("fill_holes", "true"),
        ("dilate_pixels", "2"),
    ):
        assert (
            f'require_marker_value "$DATASET_MASK_MARKER" {key} {value}'
            in text
        )

    assert '"$CANDIDATE_MASKS/$candidate_id/seg.nii.gz"' in text
    assert '"$CANONICAL_DIR/$candidate_id/bravo.nii.gz"' in text
    assert 'mv -- "$STAGED_VIEW" "$MATCHED_VIEW"' in text
    assert 'export EXP16_LABEL_OVERRIDE="$LABEL"' in text
    assert 'export EXP16_DATASET_ID_OVERRIDE="$DATASET_ID"' in text
    assert 'bash "$CONVERT_SCRIPT"' in text
    assert 'bash "$PREPROCESS_SCRIPT"' in text
    assert "python -m medgen.scripts.generate" not in text


def test_exp47a_preparation_requires_the_recovered_frozen_view() -> None:
    text = _read(
        NNUNET_SLURM_DIR / "prepare_synthmask_common105_exp47a_d654.slurm"
    )

    assert "#SBATCH --partition=CPUQ" in text
    assert "#SBATCH --time=0-12:00:00" in text
    assert 'readonly LABEL="exp47a"' in text
    assert "readonly DATASET_ID=654" in text
    assert "readonly EXPECTED_CASES=105" in text
    assert 'readonly RECOVERED_CANDIDATE="00400"' in text
    assert "readonly EXPECTED_CANONICAL_CASES=155" in text
    assert "readonly EXPECTED_PANEL_VOLUMES=2125" in text
    assert (
        'readonly CANDIDATE_MASKS="${EVAL_ROOT}/'
        'ordered_common_seg_masks_seed42/candidates525"'
    ) in text
    assert 'readonly COMMON_IDS_FILE="${MATCHED_ROOT}/common_candidate_ids.txt"' in text
    assert 'readonly MATCHED_VIEW="${MATCHED_ROOT}/${LABEL}"' in text
    assert (
        'readonly CASE_RECOVERY_MARKER="${CANONICAL_DIR}/'
        '${RECOVERED_CANDIDATE}/.recovery_provenance"'
    ) in text
    assert (
        'readonly VIEW_RECOVERY_MARKER="${MATCHED_VIEW}/.recovery_provenance"'
        in text
    )
    _assert_strict_mask_gate(text, "PANEL_MASK_MARKER")
    for key, value in (
        ("threshold", "0.05"),
        ("fill_holes", "true"),
        ("dilate_pixels", "2"),
        ("accepted_cases", '"$EXPECTED_CANONICAL_CASES"'),
        ("extension_candidate", '"$RECOVERED_CANDIDATE"'),
    ):
        assert (
            f'require_marker_value "$DATASET_MASK_MARKER" {key} {value}'
            in text
        )

    assert 'require_marker_value "$CASE_RECOVERY_MARKER" job_id "$recovery_job"' in text
    assert 'require_marker_value "$VIEW_RECOVERY_MARKER" job_id "$recovery_job"' in text
    assert (
        'require_marker_value "$CASE_RECOVERY_MARKER" candidate_id '
        '"$RECOVERED_CANDIDATE"'
    ) in text
    assert (
        'require_marker_value "$VIEW_RECOVERY_MARKER" candidate_id '
        '"$RECOVERED_CANDIDATE"'
    ) in text
    assert 'require_marker_value "$DATASET_MASK_MARKER" extension_job_id "$recovery_job"' in text
    assert 'require_marker_value "$PANEL_MASK_MARKER" extension_job_id "$recovery_job"' in text
    assert 'require_marker_value "$PANEL_MASK_MARKER" extension_label "$LABEL"' in text
    assert (
        'require_marker_value "$PANEL_MASK_MARKER" extension_candidate '
        '"$RECOVERED_CANDIDATE"'
    ) in text
    assert (
        'require_marker_value "$PANEL_MASK_MARKER" accepted_volumes '
        '"$EXPECTED_PANEL_VOLUMES"'
    ) in text
    assert 'cmp -s "$CASE_RECOVERY_MARKER" "$VIEW_RECOVERY_MARKER"' in text
    assert (
        'require_marker_value "$CASE_RECOVERY_MARKER" selection_rule '
        'first_passing_draw_in_4_to_50'
    ) in text
    assert (
        '"$CANONICAL_DIR/$RECOVERED_CANDIDATE/seg.nii.gz"'
        in text
    )
    assert 'require_marker_value "$CASE_RECOVERY_MARKER" mask_sha256 "$mask_sha256"' in text
    assert (
        'require_marker_value "$CASE_RECOVERY_MARKER" '
        'masked_bravo_sha256 "$bravo_sha256"'
    ) in text
    assert '"$CANDIDATE_MASKS/$candidate_id/seg.nii.gz"' in text
    assert '"$CANONICAL_DIR/$candidate_id/bravo.nii.gz"' in text
    assert 'validate_view "$MATCHED_VIEW"' in text
    assert "flock -u 9" in text
    assert (
        '[[ "$RAW_DIR/source_selection.txt" -nt "$VIEW_RECOVERY_MARKER" ]]'
        in text
    )
    assert (
        '[[ "$PREPROCESS_MARKER" -nt "$RAW_DIR/source_selection.txt" ]]'
        in text
    )
    assert '[[ "$PREPROCESS_MARKER" -nt "$VIEW_RECOVERY_MARKER" ]]' in text
    assert 'mv -- "$STAGED_VIEW" "$MATCHED_VIEW"' not in text
    assert 'ln -s --' not in text
    assert 'export EXP16_LABEL_OVERRIDE="$LABEL"' in text
    assert 'export EXP16_DATASET_ID_OVERRIDE="$DATASET_ID"' in text
    assert 'bash "$CONVERT_SCRIPT"' in text
    assert 'bash "$PREPROCESS_SCRIPT"' in text
    assert "python -m medgen.scripts.generate" not in text


def test_common105_preparation_scripts_support_a_fail_closed_single_source() -> None:
    for filename in (
        "convert_synthmask_common105_panel.slurm",
        "preprocess_synthmask_common105_panel.slurm",
    ):
        text = _read(NNUNET_SLURM_DIR / filename)
        assert '"${EXP16_LABEL_OVERRIDE:-}"' in text
        assert '"${EXP16_DATASET_ID_OVERRIDE:-}"' in text
        assert (
            "EXP16_LABEL_OVERRIDE and EXP16_DATASET_ID_OVERRIDE must be set together"
            in text
        )
        assert 'readonly LABEL DATASET_ID' in text


def test_exp16_2_preprocessing_transfers_and_validates_the_exp3_plan() -> None:
    text = _read(NNUNET_SLURM_DIR / "preprocess_synthmask_common105_panel.slurm")
    flat = _flatten_shell(text)

    assert "#SBATCH --array=0-4%2" in text
    assert 'readonly SOURCE_DATASET_ID=600' in text
    assert 'readonly SOURCE_PLANS="nnUNetResEncUNetLPlans"' in text
    assert f'readonly TARGET_PLANS="{CONTROLLED_PLANS}"' in text
    _assert_strict_mask_gate(text, "MASK_COMPLETE")

    assert (
        'readonly HISTORICAL_PLAN_FILE="${nnUNet_results}/'
        "exp3_baseline_v2_d600/Dataset600_BrainMet/"
        "nnUNetTrainerBrainMets__nnUNetResEncUNetLPlans__3d_fullres/"
        'plans.json"'
    ) in text
    assert "if live != historical:" in text
    assert "if target != expected:" in text

    assert (
        'nnUNetv2_extract_fingerprint -d "$DATASET_ID" -np 4 '
        "--verify_dataset_integrity --clean --no_pbar"
    ) in flat
    assert (
        'nnUNetv2_move_plans_between_datasets -s "$SOURCE_DATASET_ID" '
        '-t "$DATASET_ID" -sp "$SOURCE_PLANS" -tp "$TARGET_PLANS"'
    ) in flat
    assert (
        'nnUNetv2_preprocess -d "$DATASET_ID" -plans_name "$TARGET_PLANS" '
        "-c 3d_fullres -np 4 --no_pbar"
    ) in flat
    assert (
        'cp -- "$RAW_DIR/dataset.json" "$PREPROCESSED_DIR/dataset.json"'
    ) in flat

    assert (
        'readonly LOCK_FILE="${nnUNet_preprocessed}/'
        '.exp16_2_d600_preprocess_Dataset${DATASET_ID}.lock"'
    ) in text
    assert 'exec 9>"$LOCK_FILE"' in text
    assert "flock -n 9 || fatal" in text

    for signature in (
        '"patch_size": [160, 192, 160]',
        '"batch_size": 3',
        '"spacing": [1.0, 0.9375, 0.9375]',
        '"normalization_schemes": ["ZScoreNormalization"]',
        '"use_mask_for_norm": [True]',
        '"preprocessor_name": "DefaultPreprocessor"',
        '"batch_dice": False',
        "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet",
        "expected_features = [32, 64, 128, 256, 320, 320]",
    ):
        assert signature in text

    assert (
        'readonly COMPLETE_MARKER="${PREPROCESSED_DIR}/'
        f'{PREPROCESS_MARKER_NAME}"'
    ) in text
    assert 'rm -f -- "$COMPLETE_MARKER"' in text
    assert 'echo "source_dataset_id=${SOURCE_DATASET_ID}"' in text
    assert 'echo "source_plan_sha256=${source_plan_sha256}"' in text
    assert 'echo "target_plans=${TARGET_PLANS}"' in text
    assert 'echo "target_plan_sha256=${target_plan_sha256}"' in text
    assert 'echo "patch_size=160,192,160"' in text
    assert 'echo "use_mask_for_norm=true"' in text
    assert 'mv -f -- "$marker_tmp" "$COMPLETE_MARKER"' in text


def test_conversion_requires_completed_canonical_brain_masking() -> None:
    text = _read(NNUNET_SLURM_DIR / "convert_synthmask_common105_panel.slurm")

    assert (
        'readonly BRAIN_MASK_COMPLETE_MARKER="${EVAL_ROOT}/'
        f'{MASK_MARKER_NAME}"'
    ) in text
    _assert_strict_mask_gate(text, "BRAIN_MASK_COMPLETE_MARKER")


def test_common96_metrics_require_masking_and_use_a_distinct_output() -> None:
    text = _read(METRICS_SLURM)

    assert (
        'readonly BRAIN_MASK_COMPLETE_MARKER="${EVAL_ROOT}/'
        f'{MASK_MARKER_NAME}"'
    ) in text
    _assert_strict_mask_gate(text, "BRAIN_MASK_COMPLETE_MARKER")
    assert (
        'readonly FINAL_DIR="${RESULT_ROOT}/'
        '${EVAL_ID}_all14_common96_brainmasked"'
    ) in text
    assert 'readonly REPORT_NAME="all_metrics_train105_all14_common96_brainmasked.json"' in text
    assert "panel=all14_common96_brainmasked" in text


def test_core5_common105_metrics_use_the_controlled_masked_panel() -> None:
    text = _read(CORE5_METRICS_SLURM)

    assert (
        'readonly BRAIN_MASK_COMPLETE_MARKER="${EVAL_ROOT}/'
        f'{MASK_MARKER_NAME}"'
    ) in text
    _assert_strict_mask_gate(text, "BRAIN_MASK_COMPLETE_MARKER")
    assert 'readonly NUM_COMMON=105' in text
    assert 'readonly MATCHED_ROOT="${EVAL_ROOT}/matched_common105"' in text
    assert (
        'readonly FINAL_DIR="${RESULT_ROOT}/'
        '${EVAL_ID}_core5_common105_brainmasked"'
    ) in text
    assert (
        'readonly REPORT_NAME="all_metrics_train105_'
        'core5_common105_brainmasked.json"'
    ) in text
    assert "panel=core5_common105_brainmasked" in text
    assert "--diversity-cap 105" in text

    for label, _ in CORE5_TRAINING_JOBS.values():
        assert f"    {label}\n" in text
