"""Tests for prediction-only paper metric recomputation."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from medgen.metrics.paper_segmentation import (
    axial_feret_diameter_mm,
    classify_legacy_size,
    hd95_mm,
    matched_component_metrics,
    slicewise_dice,
    volumetric_dice,
    volumetric_iou,
)
from medgen.scripts.recompute_nnunet_metrics import (
    _paired_statistics,
    _strict_pairs,
    _validate_geometry,
)

IDENTITY = np.eye(4, dtype=np.float64)


def _mask(shape: tuple[int, int, int] = (20, 20, 20)) -> np.ndarray:
    return np.zeros(shape, dtype=bool)


def _save_mask(path: Path, mask: np.ndarray, affine: np.ndarray = IDENTITY) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), str(path))


def _save_provenance(stdout_path: Path, prediction_dir: Path, cases: int) -> None:
    stdout_path.write_text(
        "\n".join(
            [
                "=== 5-fold ensemble eval ===",
                "Folds: (0, 1, 2, 3, 4)",
                f"Output: {prediction_dir}",
                f"Inference complete: {cases} predictions",
            ]
        ),
        encoding="utf-8",
    )
    stdout_path.with_suffix(".err").write_text(
        f"INFO: Evaluating {cases} cases\n",
        encoding="utf-8",
    )


def test_complete_volume_overlap_metrics_cover_empty_and_partial_masks() -> None:
    empty = _mask((2, 2, 2))
    target = empty.copy()
    prediction = empty.copy()
    target[0, 0, 0] = True
    target[1, 1, 1] = True
    prediction[0, 0, 0] = True

    assert volumetric_dice(empty, empty) == 1.0
    assert volumetric_iou(empty, empty) == 1.0
    assert volumetric_dice(prediction, target) == pytest.approx(2 / 3)
    assert volumetric_iou(prediction, target) == pytest.approx(1 / 2)


def test_slicewise_dice_exposes_all_axes_and_foreground_only_policy() -> None:
    target = _mask((2, 2, 2))
    prediction = _mask((2, 2, 2))
    target[0, 0, 0] = True
    prediction[0, 0, 0] = True
    prediction[0, 1, 0] = True

    # Axis 0 has one non-empty slice with Dice 2/3 and one true-negative slice.
    assert slicewise_dice(prediction, target, axis=0) == pytest.approx(5 / 6)
    assert slicewise_dice(
        prediction,
        target,
        axis=0,
        include_empty_slices=False,
    ) == pytest.approx(2 / 3)
    # The same masks exercise the coronal and axial paths explicitly.
    assert slicewise_dice(prediction, target, axis=1) == pytest.approx(1 / 2)
    assert slicewise_dice(prediction, target, axis=2) == pytest.approx(5 / 6)
    assert (
        slicewise_dice(
            _mask((2, 2, 2)),
            _mask((2, 2, 2)),
            axis=2,
            include_empty_slices=False,
        )
        is None
    )


def test_hd95_uses_physical_anisotropic_spacing() -> None:
    affine = np.diag([0.8, 0.9, 2.0, 1.0])
    target = _mask((5, 5, 5))
    prediction = _mask((5, 5, 5))
    target[1, 1, 1] = True
    prediction[1, 1, 2] = True

    result = hd95_mm(prediction, target, (0.8, 0.9, 2.0), affine)

    assert result["status"] == "nonempty_pair"
    assert result["conditional_mm"] == pytest.approx(2.0)
    assert result["failure_aware_mm"] == pytest.approx(2.0)


def test_hd95_empty_mask_policy_is_explicit() -> None:
    target = _mask((5, 5, 5))
    prediction = _mask((5, 5, 5))
    target[1, 1, 1] = True

    one_empty = hd95_mm(prediction, target, (1.0, 1.0, 1.0), IDENTITY)
    both_empty = hd95_mm(prediction, prediction, (1.0, 1.0, 1.0), IDENTITY)

    assert one_empty["status"] == "empty_prediction"
    assert one_empty["conditional_mm"] is None
    assert one_empty["failure_aware_mm"] == pytest.approx(np.sqrt(48))
    assert both_empty["status"] == "both_empty"
    assert both_empty["conditional_mm"] == 0.0


def test_physical_hd95_rejects_a_sheared_grid() -> None:
    mask = _mask((3, 3, 3))
    affine = np.asarray(
        [
            [1.0, 0.2, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    spacing = tuple(float(value) for value in nib.affines.voxel_sizes(affine))

    with pytest.raises(ValueError, match="sheared/non-orthogonal"):
        _validate_geometry("case", mask, mask, affine, affine, spacing, spacing)


def test_matching_uses_full_components_without_smoothing() -> None:
    target = _mask()
    prediction = _mask()
    target[1:3, 1:3, 1:3] = True  # 8 voxels
    prediction[1:5, 1:3, 1:3] = True  # 16 voxels; connected over-segmentation

    result = matched_component_metrics(prediction, target, (1.0, 1.0, 1.0))

    assert result["counts"]["tp"] == 1
    assert result["counts"]["fn"] == 0
    assert result["counts"]["fp"] == 0
    assert result["gt_lesions"][0]["matched_dice"] == pytest.approx(2 * 8 / 24)


def test_one_to_one_matching_penalizes_merged_lesions() -> None:
    target = _mask()
    prediction = _mask()
    target[1:3, 1:3, 1:3] = True
    target[5:7, 1:3, 1:3] = True
    prediction[:] = target
    prediction[3:5, 1, 1] = True  # join both targets into one prediction component

    result = matched_component_metrics(prediction, target, (1.0, 1.0, 1.0))

    assert result["counts"]["tp"] == 1
    assert result["counts"]["fn"] == 1
    assert result["counts"]["fp"] == 0


def test_one_to_one_matching_penalizes_split_predictions() -> None:
    target = _mask()
    prediction = _mask()
    target[1:7, 1:3, 1:3] = True
    prediction[1:3, 1:3, 1:3] = True
    prediction[5:7, 1:3, 1:3] = True

    result = matched_component_metrics(prediction, target, (1.0, 1.0, 1.0))

    assert result["counts"]["tp"] == 1
    assert result["counts"]["fn"] == 0
    assert result["counts"]["fp"] == 1


def test_detection_threshold_is_strictly_greater_than_point_one() -> None:
    target = _mask((20, 20, 3))
    prediction = _mask((20, 20, 3))
    target[1:6, 1, 1] = True  # five voxels
    prediction[1, 1:16, 1] = True  # fifteen voxels, one-voxel intersection

    result = matched_component_metrics(
        prediction,
        target,
        (1.0, 1.0, 1.0),
        detection_threshold=0.1,
    )

    assert result["gt_lesions"][0]["best_candidate_dice"] == pytest.approx(0.1)
    assert result["counts"]["tp"] == 0
    assert result["counts"]["fn"] == 1
    assert result["counts"]["fp"] == 1


def test_26_connectivity_and_minimum_component_policy() -> None:
    diagonal = _mask((8, 8, 8))
    for index in range(5):
        diagonal[index + 1, index + 1, index + 1] = True

    result = matched_component_metrics(diagonal, diagonal, (1.0, 1.0, 1.0))

    assert result["counts"]["n_gt"] == 1
    assert result["counts"]["tp"] == 1
    assert result["gt_lesions"][0]["matched_dice"] == 1.0


def test_size_measure_preserves_legacy_four_connected_axial_region() -> None:
    component = _mask((10, 10, 1))
    for index in range(5):
        component[index, index, 0] = True

    # The 3D component is 26-connected, but the historical maximum-slice
    # measurement selects one of five 4-connected single-pixel regions.
    assert axial_feret_diameter_mm(component, (1.0, 1.0, 1.0)) == pytest.approx(1.0)


def test_false_positive_is_counted_when_ground_truth_is_empty() -> None:
    target = _mask()
    prediction = _mask()
    prediction[1:3, 1:3, 1:3] = True

    result = matched_component_metrics(prediction, target, (1.0, 1.0, 1.0))

    assert result["counts"]["n_gt"] == 0
    assert result["counts"]["fp"] == 1
    assert result["f1"] == 0.0


def test_prediction_touching_only_excluded_tiny_gt_is_false_positive() -> None:
    target = _mask()
    prediction = _mask()
    target[1:5, 1, 1] = True  # four voxels: outside the >=5 analysis
    prediction[1:6, 1, 1] = True

    result = matched_component_metrics(prediction, target, (1.0, 1.0, 1.0))

    assert result["counts"]["n_gt"] == 0
    assert result["counts"]["fp"] == 1
    assert result["counts"]["ignored_gt_below_min_voxels"] == 1
    assert result["counts"]["unmatched_pred_touching_excluded_gt"] == 1
    assert result["false_positive_lesions"][0]["touches_excluded_gt_component"] is True


@pytest.mark.parametrize(
    ("diameter", "expected"),
    [(9.999, "tiny"), (10.0, "small"), (20.0, "medium"), (30.0, "large")],
)
def test_study_defined_size_boundaries(diameter: float, expected: str) -> None:
    assert classify_legacy_size(diameter) == expected


def test_strict_pairing_ignores_metadata_and_rejects_case_mismatch(tmp_path: Path) -> None:
    pred_dir = tmp_path / "predictions"
    gt_dir = tmp_path / "labelsTs"
    _save_mask(pred_dir / "BrainMet_001.nii.gz", _mask((3, 3, 3)))
    _save_mask(gt_dir / "BrainMet_001.nii.gz", _mask((3, 3, 3)))
    (pred_dir / "dataset.json").write_text("{}", encoding="utf-8")

    pairs = _strict_pairs(pred_dir, gt_dir, expected_cases=1)
    assert [pair[0] for pair in pairs] == ["BrainMet_001"]

    _save_mask(gt_dir / "BrainMet_002.nii.gz", _mask((3, 3, 3)))
    with pytest.raises(ValueError, match="Expected 1 NIfTIs"):
        _strict_pairs(pred_dir, gt_dir, expected_cases=1)


def test_paired_statistics_join_by_case_id_not_dictionary_order() -> None:
    def condition(values: list[tuple[str, float]]) -> dict[str, object]:
        return {
            "per_case": {
                case_id: {
                    "volumetric_dice": value,
                    "target_mask_sha256": f"target-{case_id}",
                }
                for case_id, value in values
            }
        }

    results = {
        "real": condition([("case_b", 0.4), ("case_a", 0.2), ("case_c", 0.6)]),
        "hybrid": condition([("case_c", 0.7), ("case_a", 0.3), ("case_b", 0.5)]),
    }

    paired = _paired_statistics(
        results,
        baseline_label="real",
        compare_labels=["hybrid"],
        bootstrap_draws=100,
        seed=0,
    )

    comparison = paired["comparisons"][0]
    assert comparison["mean_difference"] == pytest.approx(0.1)
    assert comparison["n_patients"] == 3
    assert comparison["n_zero_differences"] == 0


def test_cli_smoke_reads_saved_masks_without_predictor(tmp_path: Path) -> None:
    results_root = tmp_path / "results"
    raw_root = tmp_path / "raw"
    prediction_path = results_root / "toy_exp" / "eval_toy_exp" / "predictions" / "case.nii.gz"
    target_path = raw_root / "Dataset600_Toy" / "labelsTs" / "case.nii.gz"
    mask = _mask((8, 8, 8))
    mask[2:4, 2:4, 2:4] = True
    _save_mask(prediction_path, mask)
    _save_mask(target_path, mask)
    output_dir = tmp_path / "output"
    provenance = tmp_path / "toy_eval.out"
    _save_provenance(provenance, prediction_path.parent, cases=1)
    environment = dict(os.environ)
    environment["MPLCONFIGDIR"] = str(tmp_path / "mpl")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "medgen.scripts.recompute_nnunet_metrics",
            "--condition",
            f"toy|toy_exp|600|{provenance}",
            "--nnunet-results",
            str(results_root),
            "--nnunet-raw",
            str(raw_root),
            "--output-dir",
            str(output_dir),
            "--baseline",
            "toy",
            "--expected-cases",
            "1",
            "--bootstrap-draws",
            "10",
        ],
        check=True,
        cwd=Path(__file__).parents[2],
        env=environment,
        capture_output=True,
        text=True,
    )

    with (output_dir / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    toy = summary["conditions"]["toy"]["summary"]
    assert toy["volumetric_dice"]["mean"] == 1.0
    assert toy["volumetric_dice"]["bootstrap_mean_95ci"] == [1.0, 1.0]
    assert toy["volumetric_iou"]["mean"] == 1.0
    assert toy["voxel_micro"]["dice"] == 1.0
    assert toy["voxel_micro"]["iou"] == 1.0
    for plane in ("sagittal", "coronal", "axial"):
        assert toy[f"{plane}_slicewise_dice"]["mean"] == 1.0
        assert toy[f"{plane}_foreground_slicewise_dice"]["mean"] == 1.0
    assert toy["matched_lesions"]["per_patient_penalized_lesion_dice"]["mean"] == 1.0
    assert summary["metadata"]["created_by"].endswith("recompute_nnunet_metrics")
    assert summary["metadata"]["conditions"][0]["prediction_provenance"]["folds"] == [
        0,
        1,
        2,
        3,
        4,
    ]


def test_slurm_is_cpu_only_and_has_no_inference_command() -> None:
    repository = Path(__file__).parents[2]
    slurm_path = repository / "IDUN/train/downstream/nnunet/recompute_paper_metrics.slurm"
    script = slurm_path.read_text(encoding="utf-8")
    executable = "\n".join(
        line for line in script.splitlines() if not line.lstrip().startswith("#")
    )

    assert "#SBATCH --partition=CPUQ" in script
    assert "#SBATCH --gres" not in script
    assert "medgen.scripts.recompute_nnunet_metrics" in executable
    for forbidden in (
        "nvidia-smi",
        "eval_nnunet",
        "nnUNetv2_predict",
        "checkpoint_latest",
        "medgen.scripts.generate",
    ):
        assert forbidden not in executable
