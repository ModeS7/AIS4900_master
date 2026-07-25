"""Read-only integrity and lineage audit for nnU-Net training checkpoints.

The audit is intended for interrupted or accidentally overlapping training
allocations. It never changes a checkpoint. In addition to deserializing each
file, it compares the logger history embedded in ``checkpoint_best.pth`` with
the corresponding prefix in ``checkpoint_latest.pth`` or
``checkpoint_final.pth``. A matching prefix establishes that the files belong
to one training trajectory.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import zipfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

CHECKPOINT_NAMES = (
    "checkpoint_best.pth",
    "checkpoint_latest.pth",
    "checkpoint_final.pth",
)
REQUIRED_KEYS = {
    "network_weights",
    "optimizer_state",
    "grad_scaler_state",
    "logging",
    "_best_ema",
    "current_epoch",
    "init_args",
    "trainer_name",
    "inference_allowed_mirroring_axes",
}
EXPECTED_LOG_KEYS = {
    "mean_fg_dice",
    "ema_fg_dice",
    "dice_per_class_or_region",
    "train_losses",
    "val_losses",
    "lrs",
    "epoch_start_timestamps",
    "epoch_end_timestamps",
}


@dataclass
class CheckpointAudit:
    path: str
    exists: bool = False
    size_bytes: int | None = None
    mtime_ns: int | None = None
    valid: bool = False
    epoch: int | None = None
    best_ema: float | None = None
    network_tensor_count: int | None = None
    network_parameter_count: int | None = None
    network_signature: str | None = None
    optimizer_state_entries: int | None = None
    optimizer_param_groups: int | None = None
    logging_lengths: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class FoldAudit:
    experiment: str
    fold: int
    synthetic_cases: int
    fold_dir: str
    split_valid: bool
    split_errors: list[str]
    checkpoints: dict[str, CheckpointAudit]
    anchor: str | None
    best_matches_anchor: bool | None
    action: str
    reasons: list[str]


def _iter_tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _iter_tensors(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _iter_tensors(nested)


def _all_float_tensors_finite(value: Any) -> bool:
    for tensor in _iter_tensors(value):
        if tensor.is_floating_point() or tensor.is_complex():
            if not bool(torch.isfinite(tensor).all()):
                return False
    return True


def _audit_network_weights(value: Any) -> tuple[list[str], int, int, str | None]:
    errors: list[str] = []
    if not isinstance(value, Mapping):
        return ["network_weights is not a mapping"], 0, 0, None
    if not value:
        return ["network_weights is empty"], 0, 0, None

    metadata: list[tuple[str, tuple[int, ...], str]] = []
    parameter_count = 0
    for key, tensor in value.items():
        if not isinstance(key, str):
            errors.append(f"network_weights contains a non-string key: {key!r}")
            continue
        if not isinstance(tensor, torch.Tensor):
            errors.append(f"network_weights[{key!r}] is not a tensor")
            continue
        shape = tuple(int(size) for size in tensor.shape)
        metadata.append((key, shape, str(tensor.dtype)))
        parameter_count += tensor.numel()

    if not metadata:
        errors.append("network_weights contains no tensors")
        return errors, 0, 0, None
    if len(metadata) != len(value):
        return errors, len(metadata), parameter_count, None

    digest = hashlib.sha256()
    for key, shape, dtype in sorted(metadata):
        digest.update(key.encode())
        digest.update(b"\0")
        digest.update(repr(shape).encode())
        digest.update(b"\0")
        digest.update(dtype.encode())
        digest.update(b"\n")
    return errors, len(metadata), parameter_count, digest.hexdigest()


def _audit_optimizer_state(value: Any) -> tuple[list[str], int, int]:
    errors: list[str] = []
    if not isinstance(value, Mapping):
        return ["optimizer_state is not a mapping"], 0, 0
    state = value.get("state")
    param_groups = value.get("param_groups")
    if not isinstance(state, Mapping):
        errors.append("optimizer_state.state is not a mapping")
        state_count = 0
    else:
        state_count = len(state)
        if state_count == 0:
            errors.append("optimizer_state.state is empty")
    if not isinstance(param_groups, list):
        errors.append("optimizer_state.param_groups is not a list")
        group_count = 0
    else:
        group_count = len(param_groups)
        if group_count == 0:
            errors.append("optimizer_state.param_groups is empty")
    return errors, state_count, group_count


def _nested_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        try:
            left_tensor = torch.as_tensor(left)
            right_tensor = torch.as_tensor(right)
        except (TypeError, ValueError):
            return False
        if left_tensor.shape != right_tensor.shape:
            return False
        if left_tensor.is_floating_point() or right_tensor.is_floating_point():
            return bool(torch.allclose(left_tensor, right_tensor, rtol=0, atol=0, equal_nan=True))
        return bool(torch.equal(left_tensor, right_tensor))
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return bool(np.array_equal(np.asarray(left), np.asarray(right), equal_nan=True))
        except TypeError:
            return bool(np.array_equal(np.asarray(left), np.asarray(right)))
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        try:
            left_float = float(left)
            right_float = float(right)
        except (TypeError, ValueError):
            return False
        if math.isnan(left_float) and math.isnan(right_float):
            return True
        return left_float == right_float
    return left == right


def _logging_is_prefix(best_logging: dict[str, Any], anchor_logging: dict[str, Any]) -> bool:
    if best_logging.keys() != anchor_logging.keys():
        return False
    for key, best_values in best_logging.items():
        anchor_values = anchor_logging[key]
        if not isinstance(best_values, list) or not isinstance(anchor_values, list):
            return False
        if len(best_values) > len(anchor_values):
            return False
        if not _nested_equal(best_values, anchor_values[: len(best_values)]):
            return False
    return True


def _max_finite_ema(logging: dict[str, Any]) -> float | None:
    values = logging.get("ema_fg_dice")
    if not isinstance(values, list):
        return None
    finite: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            finite.append(number)
    return max(finite) if finite else None


def _validate_init_args(init_args: Any, *, fold: int) -> list[str]:
    errors: list[str] = []
    if not isinstance(init_args, dict):
        return ["init_args is not a dictionary"]
    if init_args.get("fold") != fold:
        errors.append(f"init_args.fold={init_args.get('fold')!r}, expected {fold}")
    if init_args.get("configuration") != "3d_fullres":
        errors.append(
            f"init_args.configuration={init_args.get('configuration')!r}, expected '3d_fullres'"
        )
    plans = init_args.get("plans")
    if not isinstance(plans, dict):
        errors.append("init_args.plans is not a dictionary")
    else:
        if plans.get("dataset_name") != "Dataset663_BrainMet":
            errors.append(
                f"plans.dataset_name={plans.get('dataset_name')!r}, expected 'Dataset663_BrainMet'"
            )
        if plans.get("plans_name") != "nnUNetResEncUNetLPlansD600":
            errors.append(
                "plans.plans_name="
                f"{plans.get('plans_name')!r}, expected 'nnUNetResEncUNetLPlansD600'"
            )
    return errors


def audit_checkpoint(path: Path, *, fold: int, kind: str) -> tuple[CheckpointAudit, Any | None]:
    result = CheckpointAudit(path=str(path))
    if not path.is_file():
        return result, None
    result.exists = True
    stat = path.stat()
    result.size_bytes = stat.st_size
    result.mtime_ns = stat.st_mtime_ns
    if stat.st_size == 0:
        result.errors.append("file is empty")
        return result, None

    try:
        if not zipfile.is_zipfile(path):
            raise RuntimeError("file is not a ZIP-format torch checkpoint")
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP CRC failed for member {bad_member!r}")
    except Exception as exc:
        result.errors.append(f"archive integrity: {type(exc).__name__}: {exc}")
        return result, None

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    except Exception as exc:
        result.errors.append(f"torch.load: {type(exc).__name__}: {exc}")
        return result, None
    if not isinstance(checkpoint, dict):
        result.errors.append("checkpoint root is not a dictionary")
        return result, checkpoint

    missing = sorted(REQUIRED_KEYS - checkpoint.keys())
    if missing:
        result.errors.append(f"missing keys: {missing}")
    if checkpoint.get("trainer_name") != "nnUNetTrainerBrainMets":
        result.errors.append(
            f"trainer_name={checkpoint.get('trainer_name')!r}, expected 'nnUNetTrainerBrainMets'"
        )

    epoch = checkpoint.get("current_epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int):
        result.errors.append(f"current_epoch is not an integer: {epoch!r}")
    else:
        result.epoch = epoch
        if not 1 <= epoch <= 1000:
            result.errors.append(f"current_epoch is outside 1..1000: {epoch}")
        if kind == "final" and epoch != 1000:
            result.errors.append(f"final checkpoint epoch is {epoch}, expected 1000")
        if kind == "latest" and epoch % 50 != 0:
            result.errors.append(f"latest checkpoint epoch is {epoch}, expected a multiple of 50")

    result.errors.extend(_validate_init_args(checkpoint.get("init_args"), fold=fold))

    logging = checkpoint.get("logging")
    if not isinstance(logging, dict):
        result.errors.append("logging is not a dictionary")
    else:
        missing_logs = sorted(EXPECTED_LOG_KEYS - logging.keys())
        if missing_logs:
            result.errors.append(f"missing logging keys: {missing_logs}")
        for key, values in logging.items():
            if isinstance(values, list):
                result.logging_lengths[key] = len(values)
        if result.epoch is not None:
            wrong_lengths = {
                key: len(logging[key])
                for key in EXPECTED_LOG_KEYS & logging.keys()
                if not isinstance(logging[key], list) or len(logging[key]) != result.epoch
            }
            if wrong_lengths:
                result.errors.append(
                    f"logging lengths do not equal current_epoch {result.epoch}: {wrong_lengths}"
                )

    best_ema = checkpoint.get("_best_ema")
    if best_ema is not None:
        try:
            result.best_ema = float(best_ema)
        except (TypeError, ValueError):
            result.errors.append(f"_best_ema is not numeric: {best_ema!r}")
    if isinstance(logging, dict) and result.best_ema is not None:
        maximum = _max_finite_ema(logging)
        if maximum is None:
            result.errors.append("no finite ema_fg_dice value exists")
        elif not math.isclose(result.best_ema, maximum, rel_tol=1e-6, abs_tol=1e-8):
            result.errors.append(
                f"_best_ema={result.best_ema} does not match maximum EMA {maximum}"
            )

    if not _all_float_tensors_finite(checkpoint.get("network_weights")):
        result.errors.append("network_weights contains non-finite tensors")
    if not _all_float_tensors_finite(checkpoint.get("optimizer_state")):
        result.errors.append("optimizer_state contains non-finite tensors")
    if not _all_float_tensors_finite(checkpoint.get("grad_scaler_state")):
        result.errors.append("grad_scaler_state contains non-finite tensors")

    network_errors, tensor_count, parameter_count, signature = _audit_network_weights(
        checkpoint.get("network_weights")
    )
    result.errors.extend(network_errors)
    result.network_tensor_count = tensor_count
    result.network_parameter_count = parameter_count
    result.network_signature = signature

    optimizer_errors, state_count, group_count = _audit_optimizer_state(
        checkpoint.get("optimizer_state")
    )
    result.errors.extend(optimizer_errors)
    result.optimizer_state_entries = state_count
    result.optimizer_param_groups = group_count

    result.valid = not result.errors
    return result, checkpoint


def _validate_split(path: Path, *, fold: int, synthetic_cases: int) -> tuple[bool, list[str]]:
    errors: list[str] = []
    try:
        splits = json.loads(path.read_text())
        if not isinstance(splits, list) or len(splits) != 5:
            raise ValueError("splits_final.json must contain five folds")
        split = splits[fold]
        train = split["train"]
        val = split["val"]
        if len(train) != 84 + synthetic_cases:
            errors.append(f"train count={len(train)}, expected {84 + synthetic_cases}")
        if len(val) != 21:
            errors.append(f"validation count={len(val)}, expected 21")
        if len(train) != len(set(train)):
            errors.append("training split contains duplicates")
        if len(val) != len(set(val)):
            errors.append("validation split contains duplicates")
        if set(train) & set(val):
            errors.append("training and validation splits overlap")
        selected_synthetic = [case for case in train if case.startswith("BrainMetSyn_")]
        if len(selected_synthetic) != synthetic_cases:
            errors.append(
                f"synthetic training count={len(selected_synthetic)}, expected {synthetic_cases}"
            )
        if any(case.startswith("BrainMetSyn_") for case in val):
            errors.append("validation split contains synthetic cases")
    except Exception as exc:
        errors.append(f"split validation: {type(exc).__name__}: {exc}")
    return not errors, errors


def _lineage_matches(best: Any, anchor: Any) -> bool:
    if not isinstance(best, dict) or not isinstance(anchor, dict):
        return False
    best_epoch = best.get("current_epoch")
    anchor_epoch = anchor.get("current_epoch")
    if not isinstance(best_epoch, int) or not isinstance(anchor_epoch, int):
        return False
    if best_epoch > anchor_epoch:
        return False
    _, _, _, best_signature = _audit_network_weights(best.get("network_weights"))
    _, _, _, anchor_signature = _audit_network_weights(anchor.get("network_weights"))
    if best_signature is None or best_signature != anchor_signature:
        return False
    if not _logging_is_prefix(best.get("logging", {}), anchor.get("logging", {})):
        return False
    try:
        best_ema = float(best["_best_ema"])
        anchor_ema = float(anchor["_best_ema"])
    except (KeyError, TypeError, ValueError):
        return False
    return math.isclose(best_ema, anchor_ema, rel_tol=1e-6, abs_tol=1e-8)


def audit_fold(
    *,
    experiment: str,
    synthetic_cases: int,
    fold: int,
    results_root: Path,
    shadows_root: Path,
    model_name: str,
) -> FoldAudit:
    fold_dir = results_root / experiment / "Dataset663_BrainMet" / model_name / f"fold_{fold}"
    split_path = (
        shadows_root
        / f"nnUNet_preprocessed_{experiment}"
        / "Dataset663_BrainMet"
        / "splits_final.json"
    )
    split_valid, split_errors = _validate_split(
        split_path,
        fold=fold,
        synthetic_cases=synthetic_cases,
    )

    audits: dict[str, CheckpointAudit] = {}
    loaded: dict[str, Any | None] = {}
    for filename in CHECKPOINT_NAMES:
        kind = filename.removeprefix("checkpoint_").removesuffix(".pth")
        audit, checkpoint = audit_checkpoint(fold_dir / filename, fold=fold, kind=kind)
        audits[kind] = audit
        loaded[kind] = checkpoint

    anchor_name: str | None = None
    if audits["final"].valid:
        anchor_name = "final"
    elif audits["latest"].valid:
        anchor_name = "latest"

    lineage: bool | None = None
    if audits["best"].valid and anchor_name is not None:
        lineage = _lineage_matches(loaded["best"], loaded[anchor_name])

    reasons: list[str] = []
    if not split_valid:
        action = "BLOCKED_SPLIT"
        reasons.extend(split_errors)
    elif audits["final"].valid and audits["best"].valid and lineage:
        action = "KEEP_COMPLETE"
        reasons.append("valid final and best checkpoints share one embedded history")
    elif audits["latest"].valid and audits["best"].valid and lineage:
        action = "RESUME_LATEST"
        reasons.append("valid latest and best checkpoints share one embedded history")
    elif audits["best"].valid:
        action = "RESUME_BEST"
        if anchor_name is None:
            reasons.append("best is valid and no valid final/latest anchor exists")
        else:
            reasons.append(f"best does not share the embedded history of {anchor_name}")
    elif audits["latest"].valid:
        action = "REVIEW_LATEST_WITHOUT_BEST"
        reasons.append("latest is valid but no valid best checkpoint exists")
    elif audits["final"].valid:
        action = "REVIEW_FINAL_WITHOUT_BEST"
        reasons.append("final is valid but no valid best checkpoint exists")
    else:
        action = "RESTART"
        reasons.append("no valid best, latest, or final checkpoint exists")

    result = FoldAudit(
        experiment=experiment,
        fold=fold,
        synthetic_cases=synthetic_cases,
        fold_dir=str(fold_dir),
        split_valid=split_valid,
        split_errors=split_errors,
        checkpoints=audits,
        anchor=anchor_name,
        best_matches_anchor=lineage,
        action=action,
        reasons=reasons,
    )
    del loaded
    gc.collect()
    return result


def _parse_arm(value: str) -> tuple[str, int]:
    try:
        name, count_text = value.rsplit("=", 1)
        count = int(count_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("arm must be EXPERIMENT=COUNT") from exc
    if not name or count < 0:
        raise argparse.ArgumentTypeError("arm must have a name and non-negative count")
    return name, count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--shadows-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", action="append", type=_parse_arm, required=True)
    parser.add_argument(
        "--model-name",
        default=("nnUNetTrainerBrainMets__nnUNetResEncUNetLPlansD600__3d_fullres"),
    )
    args = parser.parse_args()

    results: list[FoldAudit] = []
    for experiment, synthetic_cases in args.arm:
        for fold in range(5):
            print(f"Auditing {experiment} fold {fold}...", flush=True)
            results.append(
                audit_fold(
                    experiment=experiment,
                    synthetic_cases=synthetic_cases,
                    fold=fold,
                    results_root=args.results_root,
                    shadows_root=args.shadows_root,
                    model_name=args.model_name,
                )
            )

    payload = {
        "results_root": str(args.results_root),
        "shadows_root": str(args.shadows_root),
        "folds": [asdict(result) for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(temporary, args.output)

    print("\n=== exp17 checkpoint audit ===")
    print(f"{'experiment':64s} {'fold':>4s} {'best':>6s} {'latest':>6s} {'final':>6s} action")
    for result in results:
        epochs = {
            key: (
                str(value.epoch)
                if value.valid and value.epoch is not None
                else ("BAD" if value.exists else "-")
            )
            for key, value in result.checkpoints.items()
        }
        print(
            f"{result.experiment:64s} {result.fold:4d} "
            f"{epochs['best']:>6s} {epochs['latest']:>6s} {epochs['final']:>6s} "
            f"{result.action}"
        )
        for key, checkpoint in result.checkpoints.items():
            for error in checkpoint.errors:
                print(f"  {key}: {error}")
        for reason in result.reasons:
            print(f"  decision: {reason}")
    print(f"\nJSON report: {args.output}")

    if any(result.action == "BLOCKED_SPLIT" for result in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
