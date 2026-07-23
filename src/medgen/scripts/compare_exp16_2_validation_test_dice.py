"""Show validation and test Dice, including tiny-lesion Dice, for exp16_2."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from medgen.metrics.paper_segmentation import matched_component_metrics, sample_summary
from medgen.scripts.recompute_nnunet_metrics import (
    _load_binary,
    _validate_geometry,
)
from medgen.scripts.select_exp16_2_validation_source import (
    COMPLETE_MARKER,
    PANEL_CONDITIONS,
    SELECTION_SCHEMA,
    _read_json,
    _sha256,
)

EXPECTED_VALIDATION_CASES = 105
EXPECTED_TEST_CASES = 51


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _verify_selection(selection_dir: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    complete = _read_json(selection_dir / COMPLETE_MARKER)
    if complete.get("schema") != SELECTION_SCHEMA:
        raise ValueError("Wrong validation-selection completion marker")
    if complete.get("n_conditions") != len(PANEL_CONDITIONS):
        raise ValueError("Validation selection does not contain all seven conditions")
    if complete.get("n_cases_per_condition") != EXPECTED_VALIDATION_CASES:
        raise ValueError("Validation selection does not contain 105 cases per condition")

    for name in ("selection.json", "validation_per_case.csv"):
        path = selection_dir / name
        expected = complete.get("artifacts_sha256", {}).get(name)
        if not expected or _sha256(path) != expected:
            raise ValueError(f"Validation-selection artifact changed: {path}")

    selection = _read_json(selection_dir / "selection.json")
    if selection.get("schema") != SELECTION_SCHEMA:
        raise ValueError("Wrong selection.json schema")
    if selection.get("selection_rule", {}).get("checkpoint") != "checkpoint_best.pth":
        raise ValueError("Validation results are not from checkpoint_best.pth")
    if selection.get("selection_rule", {}).get("official_test_used") is not False:
        raise ValueError("Official test unexpectedly informed source selection")

    rows = _read_rows(selection_dir / "validation_per_case.csv")
    expected_rows = len(PANEL_CONDITIONS) * EXPECTED_VALIDATION_CASES
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} validation rows, found {len(rows)}")
    return selection, rows


def _validation_metrics(
    selection: dict[str, Any],
    rows: list[dict[str, str]],
) -> dict[str, dict[str, float | int]]:
    panel = {condition.label: condition for condition in PANEL_CONDITIONS}
    selected = selection["winner"]["label"]
    grouped = {label: [] for label in panel}
    case_ids = {label: set() for label in panel}

    for index, row in enumerate(rows, start=1):
        label = row["condition"]
        if label not in panel:
            raise ValueError(f"Unknown validation condition: {label}")
        condition = panel[label]
        if row["experiment"] != condition.experiment:
            raise ValueError(f"Wrong experiment for {label}")
        if int(row["dataset_id"]) != condition.dataset_id:
            raise ValueError(f"Wrong dataset for {label}")
        case_id = row["case_id"]
        if case_id in case_ids[label]:
            raise ValueError(f"Duplicate validation case: {label}/{case_id}")
        case_ids[label].add(case_id)

        prediction_path = Path(row["prediction_path"])
        target_path = Path(row["target_path"])
        if _sha256(prediction_path) != row["prediction_sha256"]:
            raise ValueError(f"Prediction changed: {label}/{case_id}")
        if _sha256(target_path) != row["target_sha256"]:
            raise ValueError(f"Target changed: {label}/{case_id}")

        prediction, prediction_affine, prediction_spacing = _load_binary(prediction_path)
        target, target_affine, target_spacing = _load_binary(target_path)
        _validate_geometry(
            case_id,
            prediction,
            target,
            prediction_affine,
            target_affine,
            prediction_spacing,
            target_spacing,
        )
        lesions = matched_component_metrics(
            prediction,
            target,
            target_spacing,
            min_voxels=5,
            detection_threshold=0.1,
        )
        grouped[label].extend(
            float(record["matched_dice"])
            for record in lesions["gt_lesions"]
            if record["size_category"] == "tiny"
        )
        if index == 1 or index % 25 == 0 or index == len(rows):
            print(f"Validation masks: {index}/{len(rows)}", flush=True)

    locked = {item["label"]: item for item in selection["conditions"]}
    metrics = {}
    reference_cases: set[str] | None = None
    for condition in PANEL_CONDITIONS:
        label = condition.label
        if len(case_ids[label]) != EXPECTED_VALIDATION_CASES:
            raise ValueError(f"{label} does not contain 105 unique validation cases")
        if reference_cases is None:
            reference_cases = case_ids[label]
        elif case_ids[label] != reference_cases:
            raise ValueError(f"Validation case IDs differ for {label}")
        tiny = sample_summary(grouped[label])
        metrics[label] = {
            "selected": label == selected,
            "n_cases": EXPECTED_VALIDATION_CASES,
            "n_tiny_lesions": int(tiny["n"]),
            "dice": float(locked[label]["volumetric_dice"]["mean"]),
            "tiny_dice": float(tiny["mean"]),
        }
    return metrics


def _test_metrics(report: dict[str, Any]) -> dict[str, dict[str, float | int]]:
    parameters = report.get("metadata", {}).get("parameters", {})
    if parameters.get("expected_cases") != EXPECTED_TEST_CASES:
        raise ValueError("Test report does not contain 51 cases")
    if parameters.get("min_component_voxels") != 5:
        raise ValueError("Test report used a different lesion-size filter")
    if not math.isclose(
        float(parameters.get("detection_threshold_strictly_greater_than", -1)),
        0.1,
        rel_tol=0,
        abs_tol=1e-15,
    ):
        raise ValueError("Test report used a different lesion-detection threshold")

    metrics = {}
    for condition in PANEL_CONDITIONS:
        result = report["conditions"][condition.label]
        if result["condition"]["experiment"] != condition.experiment:
            raise ValueError(f"Wrong test experiment for {condition.label}")
        if result["condition"]["dataset_id"] != condition.dataset_id:
            raise ValueError(f"Wrong test dataset for {condition.label}")
        summary = result["summary"]
        tiny = summary["matched_lesions"]["by_gt_size"]["tiny"]
        metrics[condition.label] = {
            "n_cases": int(summary["n_cases"]),
            "n_tiny_lesions": int(tiny["n_gt"]),
            "dice": float(summary["volumetric_dice"]["mean"]),
            "tiny_dice": float(tiny["all_gt_dice"]["mean"]),
        }
    return metrics


def _comparison_rows(
    validation: dict[str, dict[str, float | int]],
    test: dict[str, dict[str, float | int]],
) -> list[dict[str, Any]]:
    return [
        {
            "condition": condition.label,
            "selected": validation[condition.label]["selected"],
            "validation_n_cases": validation[condition.label]["n_cases"],
            "validation_n_tiny_lesions": validation[condition.label]["n_tiny_lesions"],
            "validation_dice": validation[condition.label]["dice"],
            "validation_tiny_dice": validation[condition.label]["tiny_dice"],
            "test_n_cases": test[condition.label]["n_cases"],
            "test_n_tiny_lesions": test[condition.label]["n_tiny_lesions"],
            "test_dice": test[condition.label]["dice"],
            "test_tiny_dice": test[condition.label]["tiny_dice"],
        }
        for condition in PANEL_CONDITIONS
    ]


def _print_table(rows: list[dict[str, Any]]) -> None:
    print("\n=== Validation and test Dice ===")
    print(
        f"{'condition':38s} {'sel':>3s} "
        f"{'val Dice':>9s} {'val tiny':>9s} {'test Dice':>10s} {'test tiny':>10s}"
    )
    for row in rows:
        print(
            f"{row['condition']:38s} {('*' if row['selected'] else ''):>3s} "
            f"{row['validation_dice']:9.4f} {row['validation_tiny_dice']:9.4f} "
            f"{row['test_dice']:10.4f} {row['test_tiny_dice']:10.4f}"
        )
    print("Tiny Dice includes every retained tiny GT lesion; misses score zero.")
    print("* source already selected using validation Dice")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--test-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    selection, validation_rows = _verify_selection(args.selection_dir)
    validation = _validation_metrics(selection, validation_rows)
    test = _test_metrics(_read_json(args.test_summary))
    rows = _comparison_rows(validation, test)

    args.output_dir.mkdir(parents=True)
    csv_path = args.output_dir / "validation_test_dice.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _write_json(
        args.output_dir / "validation_test_dice.json",
        {
            "tiny_dice_definition": (
                "pooled all-GT Dice for lesions with axial Feret diameter below 10 mm; "
                "components below 5 voxels excluded; misses score zero"
            ),
            "validation_inference": "one held-out checkpoint_best.pth model per case",
            "test_inference": "five checkpoint_best.pth models ensembled",
            "selection_unchanged": selection["winner"],
            "selection_json_sha256": _sha256(args.selection_dir / "selection.json"),
            "test_summary_sha256": _sha256(args.test_summary),
            "rows": rows,
        },
    )
    _print_table(rows)
    print(f"Output: {csv_path}")


if __name__ == "__main__":
    main()
