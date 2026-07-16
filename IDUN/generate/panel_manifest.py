#!/usr/bin/env python3
"""Write and validate provenance for the fixed generator source-selection panel.

This helper intentionally uses only the Python standard library so it can run
before any model or metric imports.  It is orchestration metadata, separate
from any provenance emitted by ``medgen.scripts.generate`` itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPECTED_CASES = 105
EXPECTED_SEED = 42
PANEL_SOURCE_SCOPE = ["pyproject.toml", "configs", "src/medgen", "IDUN/generate"]
COMMON_PROTOCOL = {
    "expected_strategy": "rflow",
    "expected_real_cases": EXPECTED_CASES,
    "expected_real_depth": 150,
    "seed": EXPECTED_SEED,
    "num_images": EXPECTED_CASES,
    "current_image": 0,
    "num_steps_bravo": 100,
    "trim_slices": 10,
    "fov_mm": 240.0,
    "ode_solver": "euler",
    "shift_ratio_bravo": 1.0,
    "cfg_scale_bravo": 1.0,
    "validate_size_bins": False,
    "validate_brain_mask": False,
    "brain_atlas_path": None,
    "brain_pca_path": None,
    "diffrs_checkpoint": None,
    "mask_outside_brain": False,
    "mask_outside_brain_dilate_pixels": 0,
    "require_real_bravo_pairs": True,
    "validate_real_seg_masks": True,
    "provenance_hash_checkpoints": True,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_ids(root: Path) -> list[str]:
    _require(root.is_dir(), f"Input root is not a directory: {root}")
    seg_ids = sorted(path.parent.name for path in root.glob("*/seg.nii.gz"))
    bravo_ids = sorted(path.parent.name for path in root.glob("*/bravo.nii.gz"))
    _require(
        len(seg_ids) == EXPECTED_CASES,
        f"Expected {EXPECTED_CASES} input masks, found {len(seg_ids)}",
    )
    _require(
        len(bravo_ids) == EXPECTED_CASES,
        f"Expected {EXPECTED_CASES} input BRAVO volumes, found {len(bravo_ids)}",
    )
    _require(len(set(seg_ids)) == EXPECTED_CASES, "Input mask patient IDs are not unique")
    _require(len(set(bravo_ids)) == EXPECTED_CASES, "Input BRAVO patient IDs are not unique")
    _require(seg_ids == bravo_ids, "Input mask and BRAVO patient IDs differ")
    return seg_ids


def _output_ids(root: Path) -> list[str]:
    _require(root.is_dir(), f"Dataset root is not a directory: {root}")
    ids = sorted(path.name for path in root.iterdir() if path.is_dir() and path.name.isdigit())
    expected = [f"{index:05d}" for index in range(EXPECTED_CASES)]
    _require(
        ids == expected,
        f"Expected output directories 00000..00104, found {ids[:3]}...{ids[-3:] if ids else []}",
    )
    for case_id in ids:
        case_dir = root / case_id
        for filename in ("seg.nii.gz", "bravo.nii.gz"):
            path = case_dir / filename
            _require(path.is_file() and path.stat().st_size > 0, f"Missing or empty output: {path}")
    return ids


def _checkpoint(path_text: str | None) -> dict[str, str] | None:
    if not path_text:
        return None
    path = Path(path_text).resolve()
    _require(path.is_file(), f"Checkpoint does not exist: {path}")
    _require(path.name == "checkpoint_latest.pt", f"Only checkpoint_latest.pt is allowed: {path}")
    return {"path": str(path), "sha256": _sha256(path)}


def _validate_generation_manifest(
    path: Path,
    *,
    input_root: Path,
    input_ids: list[str],
    low_checkpoint: dict[str, str],
    high_checkpoint: dict[str, str] | None,
    handoff_t: float | None,
) -> dict[str, Any]:
    _require(path.is_file() and path.stat().st_size > 0, f"Missing generation manifest: {path}")
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    _require(manifest.get("schema_version") == 1, "Unexpected generation manifest schema")
    _require(manifest.get("status") == "complete", "Generation manifest is not complete")
    _require(manifest.get("mode") == "bravo", "Generation manifest mode is not bravo")
    _require(manifest.get("spatial_dims") == 3, "Generation manifest is not 3D")
    _require(manifest.get("seed") == EXPECTED_SEED, "Generation manifest seed is not 42")
    _require(
        manifest.get("num_images") == EXPECTED_CASES, "Generation manifest sample count is not 105"
    )
    _require(
        Path(manifest.get("real_seg_dir", "")).resolve() == input_root,
        "Generation manifest input root differs",
    )
    _require(
        manifest.get("expected_real_cases") == EXPECTED_CASES,
        "Generation manifest expected case count differs",
    )
    _require(
        manifest.get("expected_real_depth") == 150,
        "Generation manifest expected input depth differs",
    )
    _require(
        manifest.get("require_real_bravo_pairs") is True,
        "Generation manifest did not require real BRAVO pairs",
    )
    _require(
        manifest.get("validate_real_seg_masks") is True,
        "Generation manifest did not validate real masks",
    )

    models = manifest.get("models", {})
    expected_models = {"image_low_t": low_checkpoint, "image_high_t": high_checkpoint}
    for key, expected in expected_models.items():
        observed = models.get(key)
        if expected is None:
            _require(observed is None, f"Generation manifest unexpectedly has {key}")
        else:
            _require(
                Path(observed.get("path", "")).resolve() == Path(expected["path"]),
                f"Generation manifest {key} path differs",
            )
            _require(
                observed.get("sha256") == expected["sha256"],
                f"Generation manifest {key} hash differs",
            )

    sampling = manifest.get("sampling", {})
    expected_sampling = {
        "strategy": "rflow",
        "ode_solver": "euler",
        "num_steps_bravo": 100,
        "shift_ratio_bravo": 1.0,
        "cfg_scale_bravo": 1.0,
        "handoff_t": handoff_t,
    }
    for key, expected in expected_sampling.items():
        _require(sampling.get(key) == expected, f"Generation manifest sampling.{key} differs")

    geometry = manifest.get("geometry", {})
    expected_geometry = {
        "image_size": 256,
        "generation_depth": 160,
        "trim_slices": 10,
        "fov_mm": 240.0,
    }
    for key, expected in expected_geometry.items():
        _require(geometry.get(key) == expected, f"Generation manifest geometry.{key} differs")

    quality = manifest.get("quality_control", {})
    expected_quality = {
        "brain_atlas_path": None,
        "brain_pca_path": None,
        "validate_brain_mask": False,
        "mask_outside_brain": False,
        "mask_outside_brain_dilate_pixels": 0,
        "diffrs_checkpoint": None,
    }
    for key, expected in expected_quality.items():
        _require(quality.get(key) == expected, f"Generation manifest quality_control.{key} differs")

    samples = manifest.get("samples", [])
    _require(len(samples) == EXPECTED_CASES, "Generation manifest does not contain 105 samples")
    for index, (sample, patient_id) in enumerate(zip(samples, input_ids, strict=True)):
        _require(
            sample.get("index") == index, f"Generation manifest sample index differs at {index}"
        )
        _require(
            sample.get("patient_id") == patient_id,
            f"Generation manifest patient ID differs at {index}",
        )
        expected_mask = (input_root / patient_id / "seg.nii.gz").resolve()
        _require(
            Path(sample.get("conditioning_mask", "")).resolve() == expected_mask,
            f"Conditioning mask differs at {index}",
        )
        _require(sample.get("seg_noise_seed") is None, f"Unexpected generated-mask seed at {index}")
        _require(
            sample.get("image_noise_seed") == EXPECTED_SEED + index,
            f"Image noise seed differs at {index}",
        )
        _require(sample.get("image_attempt") == 0, f"Unexpected image retry at {index}")
    return manifest


def write_manifest(args: argparse.Namespace) -> None:
    input_root = Path(args.input_root).resolve()
    staging_root = Path(args.dataset_root).resolve()
    final_root = Path(args.final_dataset_root).resolve()
    output = Path(args.output).resolve()
    _require(not final_root.exists(), f"Final dataset already exists: {final_root}")

    input_ids = _input_ids(input_root)
    output_ids = _output_ids(staging_root)
    low_checkpoint = _checkpoint(args.low_checkpoint)
    assert low_checkpoint is not None
    high_checkpoint = _checkpoint(args.high_checkpoint)
    handoff_t = args.handoff_t if high_checkpoint is not None else None
    _require(
        (high_checkpoint is None) == (handoff_t is None),
        "High-t checkpoint and handoff_t must be specified together",
    )
    if handoff_t is not None:
        _require(handoff_t == 0.25, f"Panel handoff_t must be 0.25, got {handoff_t}")
    generation_manifest_path = staging_root / "generation_manifest.json"
    generation_manifest = _validate_generation_manifest(
        generation_manifest_path,
        input_root=input_root,
        input_ids=input_ids,
        low_checkpoint=low_checkpoint,
        high_checkpoint=high_checkpoint,
        handoff_t=handoff_t,
    )
    _require(
        generation_manifest.get("git_commit") == args.git_commit,
        "Core generation and panel manifests record different Git commits",
    )

    protocol = dict(COMMON_PROTOCOL)
    protocol.update(
        {
            "spatial_dims": 3,
            "gen_mode": "bravo",
            "image_size": 256,
            "depth": 160,
            "handoff_t": handoff_t,
        }
    )
    manifest: dict[str, Any] = {
        "schema": "medgen.fixed_generator_panel_job.v1",
        "label": args.label,
        "dataset_root": str(final_root),
        "reference_root": str(input_root),
        "ordered_input_ids": input_ids,
        "ordered_output_ids": output_ids,
        "cases": [
            {"patient_id": patient_id, "output_id": output_id, "noise_seed": EXPECTED_SEED + index}
            for index, (patient_id, output_id) in enumerate(zip(input_ids, output_ids, strict=True))
        ],
        "checkpoints": {
            "low_t": low_checkpoint,
            "high_t": high_checkpoint,
        },
        "generation_manifest": {
            "path": str(final_root / "generation_manifest.json"),
            "sha256": _sha256(generation_manifest_path),
        },
        "protocol": protocol,
        "runtime": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": args.git_commit,
            "git_dirty": args.git_dirty == "true",
            "source_scope": PANEL_SOURCE_SCOPE,
            "source_scope_clean": True,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
            "slurm_node": os.environ.get("SLURMD_NODENAME"),
        },
    }

    _require(not output.exists(), f"Manifest already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output)
    print(f"Wrote panel job manifest: {output}")


def validate_manifests(args: argparse.Namespace) -> None:
    manifest_paths = [Path(path).resolve() for path in args.manifest]
    expected_labels = args.expected_label
    _require(
        len(manifest_paths) == len(expected_labels), "Manifest and expected-label counts differ"
    )
    _require(len(manifest_paths) == 14, f"Expected 14 panel manifests, got {len(manifest_paths)}")
    _require(len(set(expected_labels)) == 14, "Expected labels are not unique")

    panel_root = Path(args.panel_root).resolve()
    common_input_ids: list[str] | None = None
    common_reference_root: str | None = None
    common_git_commit: str | None = None
    observed_labels: list[str] = []
    for manifest_path, expected_label in zip(manifest_paths, expected_labels, strict=True):
        _require(manifest_path.is_file(), f"Missing manifest: {manifest_path}")
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)

        _require(
            manifest.get("schema") == "medgen.fixed_generator_panel_job.v1",
            f"Unexpected schema in {manifest_path}",
        )
        label = manifest.get("label")
        _require(label == expected_label, f"Expected label {expected_label}, found {label}")
        observed_labels.append(label)
        expected_dataset_root = (panel_root / label).resolve()
        _require(
            Path(manifest["dataset_root"]).resolve() == expected_dataset_root,
            f"Dataset path mismatch for {label}",
        )
        _require(
            manifest.get("protocol", {}).items() >= COMMON_PROTOCOL.items(),
            f"Protocol mismatch for {label}",
        )

        is_composite = label.startswith("exp1_to_exp48")
        expected_handoff = 0.25 if is_composite else None
        _require(
            manifest["protocol"].get("handoff_t") == expected_handoff,
            f"Handoff mismatch for {label}",
        )
        _require(
            (manifest["checkpoints"].get("high_t") is not None) == is_composite,
            f"High-t checkpoint mismatch for {label}",
        )
        for checkpoint in manifest["checkpoints"].values():
            if checkpoint is not None:
                _require(
                    checkpoint.get("path", "").endswith("/checkpoint_latest.pt"),
                    f"Non-latest checkpoint in {label}",
                )
                _require(
                    len(checkpoint.get("sha256", "")) == 64, f"Missing checkpoint hash in {label}"
                )

        input_ids = manifest.get("ordered_input_ids")
        _require(
            input_ids == _input_ids(Path(manifest["reference_root"])),
            f"Input IDs no longer match for {label}",
        )
        output_ids = _output_ids(expected_dataset_root)
        _require(
            manifest.get("ordered_output_ids") == output_ids,
            f"Output IDs no longer match for {label}",
        )
        generation_manifest = manifest.get("generation_manifest", {})
        expected_generation_manifest = expected_dataset_root / "generation_manifest.json"
        _require(
            Path(generation_manifest.get("path", "")).resolve() == expected_generation_manifest,
            f"Generation manifest path differs for {label}",
        )
        _require(
            _sha256(expected_generation_manifest) == generation_manifest.get("sha256"),
            f"Generation manifest hash differs for {label}",
        )
        expected_cases = [
            {"patient_id": patient_id, "output_id": output_id, "noise_seed": EXPECTED_SEED + index}
            for index, (patient_id, output_id) in enumerate(zip(input_ids, output_ids, strict=True))
        ]
        _require(
            manifest.get("cases") == expected_cases, f"Case mapping or seeds differ for {label}"
        )

        runtime = manifest.get("runtime", {})
        git_commit = runtime.get("git_commit")
        git_dirty = runtime.get("git_dirty")
        _require(isinstance(git_commit, str) and bool(git_commit), f"Git commit missing for {label}")
        _require(isinstance(git_dirty, bool), f"Git dirty state missing for {label}")
        _require(
            runtime.get("source_scope") == PANEL_SOURCE_SCOPE,
            f"Panel source scope differs for {label}",
        )
        _require(runtime.get("source_scope_clean") is True, f"Panel source was dirty for {label}")
        _require(
            git_commit == args.expected_git_commit,
            f"Generator Git commit differs from metric job for {label}",
        )

        if common_input_ids is None:
            common_input_ids = input_ids
            common_reference_root = str(Path(manifest["reference_root"]).resolve())
            common_git_commit = git_commit
        else:
            _require(input_ids == common_input_ids, f"Input ordering differs for {label}")
            _require(
                str(Path(manifest["reference_root"]).resolve()) == common_reference_root,
                f"Reference root differs for {label}",
            )
            _require(git_commit == common_git_commit, f"Git commit differs for {label}")

    _require(
        observed_labels == expected_labels, "Manifest labels are not in the declared panel order"
    )
    print(f"Validated {len(observed_labels)} panel manifests with one common 105-case protocol")


def validate_report(args: argparse.Namespace) -> None:
    report_path = Path(args.report).resolve()
    _require(report_path.is_file(), f"Metric report does not exist: {report_path}")
    with report_path.open(encoding="utf-8") as handle:
        report = json.load(handle)

    expected_labels = args.expected_label
    _require(
        len(expected_labels) == 14 and len(set(expected_labels)) == 14,
        "Expected exactly 14 unique report labels",
    )
    config = report.get("config", {})
    _require(
        config.get("references") == {"train": EXPECTED_CASES},
        "Report must contain only the 105-case train reference",
    )
    _require(config.get("pca_model") == "none", "PCA must be disabled in the panel report")
    _require(config.get("seed") == EXPECTED_SEED, "Metric report seed must be 42")
    _require(config.get("pool_cap") == 0, "Synthetic pool cap must be disabled")
    _require(
        config.get("source_git_commit") == args.expected_git_commit,
        "Metric report Git commit differs",
    )

    datasets = report.get("datasets", {})
    _require(
        list(datasets) == expected_labels,
        "Metric report dataset labels or order differ from the declared panel",
    )
    for label, result in datasets.items():
        _require(result.get("total_found") == EXPECTED_CASES, f"{label}: total_found is not 105")
        _require(result.get("pool_size") == EXPECTED_CASES, f"{label}: pool_size is not 105")
        generation_manifest = result.get("generation_manifest") or {}
        expected_manifest = Path(result.get("source_path", "")) / "generation_manifest.json"
        _require(
            Path(generation_manifest.get("path", "")).resolve() == expected_manifest.resolve(),
            f"{label}: generation manifest path differs",
        )
        _require(
            expected_manifest.is_file(),
            f"{label}: generation manifest file is missing",
        )
        _require(
            generation_manifest.get("sha256") == _sha256(expected_manifest),
            f"{label}: generation manifest hash differs",
        )
        _require(
            list(result.get("per_reference", {})) == ["train"],
            f"{label}: report contains a non-train reference",
        )
        _require(
            not any(key.startswith("pca_") for key in result),
            f"{label}: PCA result found despite disabled PCA",
        )
    print(
        "Validated combined metric report: 14 datasets x 105 volumes, train-only reference, no PCA"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    write_parser = subparsers.add_parser("write", help="Write one completed generator-job manifest")
    write_parser.add_argument("--output", required=True)
    write_parser.add_argument("--label", required=True)
    write_parser.add_argument("--input-root", required=True)
    write_parser.add_argument("--dataset-root", required=True, help="Staging dataset to validate")
    write_parser.add_argument(
        "--final-dataset-root", required=True, help="Final path recorded in the manifest"
    )
    write_parser.add_argument("--low-checkpoint", required=True)
    write_parser.add_argument("--high-checkpoint")
    write_parser.add_argument("--handoff-t", type=float)
    write_parser.add_argument("--git-commit", required=True)
    write_parser.add_argument("--git-dirty", choices=("true", "false"), required=True)
    write_parser.set_defaults(handler=write_manifest)

    validate_parser = subparsers.add_parser(
        "validate", help="Validate all manifests before combined metrics"
    )
    validate_parser.add_argument("--panel-root", required=True)
    validate_parser.add_argument("--expected-git-commit", required=True)
    validate_parser.add_argument("--manifest", action="append", required=True)
    validate_parser.add_argument("--expected-label", action="append", required=True)
    validate_parser.set_defaults(handler=validate_manifests)

    report_parser = subparsers.add_parser(
        "validate-report", help="Validate the combined report before publication"
    )
    report_parser.add_argument("--report", required=True)
    report_parser.add_argument("--expected-git-commit", required=True)
    report_parser.add_argument("--expected-label", action="append", required=True)
    report_parser.set_defaults(handler=validate_report)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
