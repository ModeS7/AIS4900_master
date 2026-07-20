"""Static contracts for masking and rescoring the real-seg all-14 panel."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[2]
MASK_SLURM = PROJECT_ROOT / "IDUN/eval/brain_mask_realseg_all14_inplace.slurm"
METRIC_SLURM = (
    PROJECT_ROOT / "IDUN/generate/eval_generator_panel_metrics.slurm"
)
EXPECTED_LABELS = (
    "exp1_1_1000",
    "exp1_1_1000plus",
    "exp32_2_1000",
    "exp32_3_1000",
    "exp47a",
    "exp47b",
    "exp47c",
    "exp47d",
    "exp47e",
    "exp1_to_exp48a_t025",
    "exp1_to_exp48b_t025",
    "exp1_to_exp48c_t025",
    "exp1_to_exp48d_t025",
    "exp1_to_exp48e_t025",
)


def _read(path: Path) -> str:
    assert path.is_file(), f"missing Slurm file: {path}"
    return path.read_text()


def test_realseg_masking_is_exact_in_place_stage3_for_all_14() -> None:
    text = _read(MASK_SLURM)

    assert 'EVAL_ID="${EVAL_ID:-source_selection_train105_seed42_euler100}"' in text
    assert "readonly NUM_CASES=105" in text
    assert "readonly TOTAL_BRAVOS=1470" in text
    assert '[[ ! -e "$COMPLETE_MARKER" ]]' in text
    assert 'exec 9>"$LOCK_FILE"' in text
    assert "flock -n 9 || fatal" in text
    assert "--in-place" in text
    assert "--threshold 0.05" in text
    assert "--dilate-pixels 2" in text
    assert "--expected-shape 256 256 150" in text
    assert 'echo "conditioning=real_train_segmentation"' in text
    assert "seg_tree_sha256" in text
    assert "bins_sha256" in text
    assert "conditioning_map_sha256" in text
    assert "unmasked_report_sha256" in text
    assert 'mv -f -- "$marker_tmp" "$COMPLETE_MARKER"' in text
    assert "--dst" not in text

    for label in EXPECTED_LABELS:
        assert f"    {label}\n" in text


def test_realseg_metric_rerun_is_mask_gated_and_non_overwriting() -> None:
    text = _read(METRIC_SLURM)

    assert 'PANEL_ID="${PANEL_ID:-source_selection_train105_seed42_euler100}"' in text
    assert (
        'BRAIN_MASK_COMPLETE_MARKER="${PANEL_ROOT}/'
        '.brain_mask_stage3_all14_complete"'
    ) in text
    for key, value in (
        ("eval_id", '"$PANEL_ID"'),
        ("mode", "in_place_atomic"),
        ("threshold", "0.05"),
        ("fill_holes", "true"),
        ("dilate_pixels", "2"),
        ("conditioning", "real_train_segmentation"),
        ("generators", "14"),
        ("generated_volumes", "1470"),
    ):
        assert (
            f'require_marker_value "$BRAIN_MASK_COMPLETE_MARKER" {key} {value}'
            in text
        )

    assert 'RESULT_DIR="${CLUSTER_BASE}/AIS4900_master/runs/eval/${PANEL_ID}"' in text
    assert 'REPORT="${RESULT_DIR}/all_metrics_train105_brainmasked.json"' in text
    assert '[[ ! -e "$REPORT" ]]' in text
    assert '--diversity-cap 105' in text

    for label in EXPECTED_LABELS:
        assert f"    {label}\n" in text
