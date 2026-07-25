"""Static contracts for the exp17 nested-dose hybrid study."""

import re
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SLURM_DIR = ROOT / "IDUN" / "train" / "downstream" / "nnunet"

ARMS = {
    "exp17_1_hybrid_25syn_exp1_to_exp48c_t025_d663.slurm": (25, 109),
    "exp17_2_hybrid_50syn_exp1_to_exp48c_t025_d663.slurm": (50, 134),
    "exp17_3_hybrid_105syn_exp1_to_exp48c_t025_d663.slurm": (105, 189),
    "exp17_4_hybrid_210syn_exp1_to_exp48c_t025_d663.slurm": (210, 294),
}


def test_exp17_training_arms_use_one_controlled_nested_dataset() -> None:
    for filename, (n_synthetic, n_training) in ARMS.items():
        content = (SLURM_DIR / filename).read_text()

        assert "#SBATCH --time=0-12:00:00" in content
        assert '#SBATCH --constraint="h100|gpu80g"' in content
        assert "#SBATCH --array=0-4%5" in content
        assert "readonly DATASET_ID=663" in content
        assert f"readonly N_SYNTHETIC={n_synthetic}" in content
        assert f'readonly EXPERIMENT_NAME="{filename.removesuffix(".slurm")}"' in content
        assert 'readonly PLANS_NAME="nnUNetResEncUNetLPlansD600"' in content
        assert 'readonly SYNTHETIC_MANIFEST="${RAW_DIR}/synthetic_candidate_order.txt"' in content
        assert 'readonly PREPROCESS_MARKER="${PREPROCESSED_DIR}/.exp17_d600_preprocess_complete"' in content
        assert "nnUNet_preprocessed_exp3_baseline_v2_d600" in content
        assert "synthetic_candidate_order_sha256" in content
        assert "conversion_marker_sha256" in content
        assert 'split["val"] == historical_splits[fold]["val"]' in content
        assert 'sorted(train_real) == historical_splits[fold]["train"]' in content
        assert "--experiment mixed" in content
        assert '--n-synthetic "$N_SYNTHETIC"' in content
        assert '--synthetic-manifest "$SYNTHETIC_MANIFEST"' in content
        assert "--trainer nnUNetTrainerBrainMets" in content
        assert '--plans "$PLANS_NAME"' in content
        assert 'echo "Per-fold training:   $((84 + N_SYNTHETIC)) cases"' in content
        assert 84 + n_synthetic == n_training
        assert "--continue-training" in content
        assert "sbatch --array=\"$FOLD\"" in content
        assert "flock -n 9" in content
        assert 'another allocation is already training ${EXPERIMENT_NAME} fold ${FOLD}' in content
        assert '--dependency="afterany:${SLURM_JOB_ID}"' in content
        assert "--seed" not in content
        assert "Evaluation is intentionally separate" in content


def test_exp17_has_one_conversion_and_one_controlled_preprocess_job() -> None:
    conversion = (
        SLURM_DIR / "convert_exp17_weighted_huber_handoff_210_d663.slurm"
    ).read_text()
    preprocessing = (
        SLURM_DIR / "preprocess_exp17_weighted_huber_handoff_210_d663.slurm"
    ).read_text()

    assert "readonly DATASET_ID=663" in conversion
    assert "readonly EXPECTED_REAL_CASES=105" in conversion
    assert "readonly EXPECTED_SYNTHETIC_CASES=210" in conversion
    assert "readonly EXPECTED_ORIGINAL_PAIRS=151" in conversion
    assert "readonly EXPECTED_EXTENSION_PAIRS=59" in conversion
    assert "hybrid_pairs_210" in conversion
    assert "hybrid_candidate_ids_210.txt" in conversion
    assert "synthetic_candidate_order.txt" in conversion
    assert ".exp17_d663_conversion_complete" in conversion
    assert "--dataset-id \"$DATASET_ID\"" in conversion
    assert "--modality bravo" in conversion
    assert "all real images and labels numerically match Dataset600" in conversion
    assert "np.array_equal(current_data, baseline_data" in conversion
    assert "#SBATCH --array" not in conversion

    assert "readonly DATASET_ID=663" in preprocessing
    assert "readonly EXPECTED_TRAINING_CASES=315" in preprocessing
    assert 'readonly SOURCE_PLANS="nnUNetResEncUNetLPlans"' in preprocessing
    assert 'readonly TARGET_PLANS="nnUNetResEncUNetLPlansD600"' in preprocessing
    assert "nnUNetv2_extract_fingerprint" in preprocessing
    assert "nnUNetv2_move_plans_between_datasets" in preprocessing
    assert "nnUNetv2_preprocess" in preprocessing
    assert ".exp17_d600_preprocess_complete" in preprocessing
    assert "conversion marker synthetic order SHA-256" in preprocessing
    assert "conversion marker source-selection SHA-256" in preprocessing
    assert "#SBATCH --array" not in preprocessing


def test_exp17_arm_sizes_are_strictly_nested() -> None:
    sizes = [value[0] for value in ARMS.values()]
    assert sizes == [25, 50, 105, 210]
    assert all(left < right for left, right in pairwise(sizes))


def test_exp17_scripts_do_not_reference_previous_panel_datasets() -> None:
    stale = re.compile(r"Dataset(?:650|651|652|654|656|661|662)|common105_panel")
    paths = [
        SLURM_DIR / "convert_exp17_weighted_huber_handoff_210_d663.slurm",
        SLURM_DIR / "preprocess_exp17_weighted_huber_handoff_210_d663.slurm",
        *(SLURM_DIR / filename for filename in ARMS),
    ]
    for path in paths:
        assert stale.search(path.read_text()) is None, path
