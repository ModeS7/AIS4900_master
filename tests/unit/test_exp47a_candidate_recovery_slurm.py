"""Static contract for the one-candidate exp47a synthetic-mask recovery."""

import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[2]
RECOVERY_SLURM = (
    PROJECT_ROOT / "IDUN/generate/recover_synthmask_exp47a_00400.slurm"
)


def test_exp47a_00400_recovery_preserves_the_fixed_protocol() -> None:
    text = RECOVERY_SLURM.read_text()

    assert "#SBATCH --time=0-12:00:00" in text
    assert '#SBATCH --constraint="a100|h100|gpu80g"' in text
    assert 'readonly LABEL="exp47a"' in text
    assert 'readonly LOW_RUN="exp47a_lpips_strong_20260425-055252"' in text
    assert 'readonly CANDIDATE_ID="00400"' in text
    assert "readonly CANDIDATE_INDEX=400" in text
    assert "readonly ORIGINAL_BASE_SEED=42" in text
    assert "readonly ATTEMPT_STRIDE=1000000" in text
    assert "readonly FIRST_RETRY_DRAW=4" in text
    assert "readonly LAST_RETRY_DRAW=50" in text
    assert 'seed="$RETRY_BASE_SEED"' in text
    assert 'num_images="$NUM_IMAGES"' in text
    assert 'current_image="$CANDIDATE_INDEX"' in text
    assert 'max_image_attempts_per_mask="$RETRY_DRAWS"' in text
    assert "skip_failed_fixed_masks=false" in text
    assert "mask_outside_brain=false" in text
    assert "brain_containment_margin_mm=\"$BRAIN_MARGIN_MM\"" in text
    assert "brain_support_pca_path=\"$BRAIN_SUPPORT_PCA\"" in text

    constants = {
        name: int(value)
        for name, value in re.findall(
            r"^readonly (ORIGINAL_BASE_SEED|ATTEMPT_STRIDE|FIRST_RETRY_DRAW|"
            r"LAST_RETRY_DRAW|CANDIDATE_INDEX)=([0-9]+)$",
            text,
            flags=re.MULTILINE,
        )
    }
    retry_seeds = [
        constants["ORIGINAL_BASE_SEED"]
        + constants["CANDIDATE_INDEX"]
        + (draw - 1) * constants["ATTEMPT_STRIDE"]
        for draw in range(
            constants["FIRST_RETRY_DRAW"], constants["LAST_RETRY_DRAW"] + 1
        )
    ]
    assert retry_seeds == list(range(3_000_442, 49_000_443, 1_000_000))
    assert len(retry_seeds) == 47


def test_exp47a_00400_recovery_masks_validates_and_is_restart_safe() -> None:
    text = RECOVERY_SLURM.read_text()

    assert "python -m medgen.scripts.brain_mask_existing" in text
    assert "--threshold 0.05" in text
    assert "--dilate-pixels 2" in text
    assert "--expected-shape 256 256 150" in text
    assert 'grep -Fxq "$CANDIDATE_ID" "$COMMON105_IDS"' in text
    assert "generated segmentation differs from fixed candidate 00400" in text
    assert 'cp -- "$CANONICAL_BINS" "$TRANSACTION_PREP/bins.before"' in text
    assert 'cp -- "$DATASET_MASK_MARKER" "$TRANSACTION_PREP/dataset_marker.before"' in text
    assert 'cp -- "$PANEL_MASK_MARKER" "$TRANSACTION_PREP/panel_marker.before"' in text
    assert 'ln -- "$CANDIDATE_MASK" "$TRANSACTION_PREP/case.after/seg.nii.gz"' in text
    assert 'ln -- "$SHARD_CASE/bravo.nii.gz" "$TRANSACTION_PREP/case.after/bravo.nii.gz"' in text
    assert 'mkdir "$TRANSACTION_PREP/matched105.after"' in text
    assert 'mv -- "$TRANSACTION_PREP" "$TRANSACTION_DIR"' in text
    assert 'mv -- "$TRANSACTION_DIR/case.after" "$CANONICAL_CASE"' in text
    assert 'mv -- "$TRANSACTION_DIR/matched105.after" "$MATCHED105_VIEW"' in text
    assert "rollback_incomplete_transaction" in text
    assert "trap cleanup_recovery EXIT" in text
    assert "trap 'exit 129' HUP" in text
    assert "trap 'exit 130' INT" in text
    assert "trap 'exit 143' TERM" in text
    assert 'touch "$TRANSACTION_DIR/COMMITTED"' in text
    assert "accepted_cases=\" replacement" in text
    assert "accepted_volumes=\" replacement" in text
    assert 'extension_candidate=$CANDIDATE_ID' in text
    assert 'require_marker_value "$CASE_RECOVERY_MARKER" mask_sha256' in text
    assert 'require_marker_value "$CASE_RECOVERY_MARKER" masked_bravo_sha256' in text
    assert "Recovery is already complete for" in text
    assert "canonical_cases_before" in text
    assert "canonical_cases_after" in text


def test_exp47a_00400_recovery_locks_and_checks_parity_before_generation() -> None:
    text = RECOVERY_SLURM.read_text()

    lock_position = text.index('flock -n 9')
    parity_position = text.index(
        'validate_case_bin_parity "$EXPECTED_CANONICAL_BEFORE"'
    )
    generation_position = text.index("time python -m medgen.scripts.generate")
    assert lock_position < parity_position < generation_position
