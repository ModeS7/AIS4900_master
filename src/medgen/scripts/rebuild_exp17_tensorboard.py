"""Rebuild clean exp17 TensorBoard histories from completed nnU-Net checkpoints.

The exp17 recovery intentionally restored audited checkpoints without restoring
TensorBoard event streams from the quarantined, overlapping allocations.  A
completed nnU-Net checkpoint embeds the complete per-epoch logger history, so
the canonical curves can be reconstructed without selecting or combining any
of the quarantined event files.

This command is fail-closed. It validates all twenty ``checkpoint_final.pth``
files and builds every replacement before changing a normal TensorBoard path.
Existing partial event directories are moved intact to a separate archive.
Installation is rolled back if any replacement or verification fails.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import gc
import hashlib
import json
import math
import os
import shutil
import signal
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from medgen.scripts.audit_nnunet_checkpoints import audit_checkpoint
from medgen.scripts.recover_exp17_checkpoints import EXPECTED_RECOVERY

MODEL_NAME = "nnUNetTrainerBrainMets__nnUNetResEncUNetLPlansD600__3d_fullres"
EXPECTED_EPOCHS = 1000
EXPECTED_EXPERIMENTS = (
    "exp17_1_hybrid_25syn_exp1_to_exp48c_t025_d663",
    "exp17_2_hybrid_50syn_exp1_to_exp48c_t025_d663",
    "exp17_3_hybrid_105syn_exp1_to_exp48c_t025_d663",
    "exp17_4_hybrid_210syn_exp1_to_exp48c_t025_d663",
)
SCALAR_KEYS = {
    "train/loss": "train_losses",
    "train/learning_rate": "lrs",
    "val/loss": "val_losses",
    "val/mean_fg_dice": "mean_fg_dice",
    "val/ema_fg_dice": "ema_fg_dice",
}


class RebuildError(RuntimeError):
    """Raised when a complete and unambiguous reconstruction is impossible."""


@dataclass(frozen=True)
class FoldHistory:
    experiment: str
    fold: int
    checkpoint: str
    checkpoint_size_bytes: int
    checkpoint_mtime_ns: int
    checkpoint_device: int
    checkpoint_inode: int
    checkpoint_sha256: str
    network_signature: str
    walltimes: tuple[float, ...]
    scalars: dict[str, tuple[float, ...]]


@dataclass(frozen=True)
class RecoveryProvenance:
    path: str
    sha256: str


@dataclass(frozen=True)
class TensorboardState:
    path: str
    files: dict[str, dict[str, Any]] | None


@dataclass
class InstallState:
    installed: list[tuple[Path, Path]]
    archived: list[tuple[Path, Path]]


@contextmanager
def _defer_install_signals():
    """Defer process-wide termination signals until commit or rollback completes."""
    if threading.current_thread() is not threading.main_thread():
        raise RebuildError("TensorBoard installation must run on the main thread")
    watched = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)
    previous: dict[signal.Signals, Any] = {}
    pending = {signal_number: False for signal_number in watched}

    def defer(signal_number: int, _frame: Any) -> None:
        pending[signal.Signals(signal_number)] = True

    try:
        for signal_number in watched:
            previous[signal_number] = signal.getsignal(signal_number)
            signal.signal(signal_number, defer)
        yield
    finally:
        for signal_number, handler in previous.items():
            signal.signal(signal_number, handler)
        for signal_number in watched:
            if pending[signal_number]:
                os.kill(os.getpid(), signal_number)


def _number(value: Any, *, context: str, require_finite: bool = False) -> float:
    if isinstance(value, bool):
        raise RebuildError(f"{context} is boolean, expected a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RebuildError(f"{context} is not numeric: {value!r}") from exc
    if require_finite and not math.isfinite(number):
        raise RebuildError(f"{context} is not finite: {number!r}")
    return number


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _publish_no_replace(staging: Path, destination: Path) -> None:
    """Atomically rename a directory while refusing an existing destination."""
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RebuildError("renameat2 is unavailable; refusing non-atomic publication")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(staging),
        at_fdcwd,
        os.fsencode(destination),
        rename_noreplace,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise RebuildError(f"refusing to overwrite existing output: {destination}")
    raise RebuildError(
        f"failed to publish {staging} as {destination}: "
        f"[errno {error_number}] {os.strerror(error_number)}"
    )


def _verify_noreplace_support(parent: Path) -> None:
    """Exercise no-replace directory renames on the target filesystem."""
    source = parent / f".exp17_tb_rename_probe_source_{os.getpid()}"
    destination = parent / f".exp17_tb_rename_probe_destination_{os.getpid()}"
    if source.exists() or destination.exists():
        raise RebuildError(f"rename probe path already exists under {parent}")
    source.mkdir()
    destination.mkdir()
    try:
        try:
            _publish_no_replace(source, destination)
        except RebuildError as exc:
            if "refusing to overwrite existing output" not in str(exc):
                raise
        else:
            raise RebuildError("no-replace publication unexpectedly replaced a directory")
        if not source.is_dir() or not destination.is_dir():
            raise RebuildError("no-replace probe changed an existing directory")
    finally:
        shutil.rmtree(source, ignore_errors=True)
        shutil.rmtree(destination, ignore_errors=True)

    source.mkdir()
    try:
        _publish_no_replace(source, destination)
        if source.exists() or not destination.is_dir():
            raise RebuildError("no-replace publication probe produced an invalid state")
    finally:
        shutil.rmtree(source, ignore_errors=True)
        shutil.rmtree(destination, ignore_errors=True)


def _numeric_series(
    value: Any, *, context: str, require_finite: bool = False
) -> tuple[float, ...]:
    if not isinstance(value, list):
        raise RebuildError(f"{context} is not a list")
    if len(value) != EXPECTED_EPOCHS:
        raise RebuildError(
            f"{context} has {len(value)} values, expected {EXPECTED_EPOCHS}"
        )
    return tuple(
        _number(
            item,
            context=f"{context}[{index}]",
            require_finite=require_finite,
        )
        for index, item in enumerate(value)
    )


def _dice_series(value: Any, *, context: str) -> dict[str, tuple[float, ...]]:
    if not isinstance(value, list):
        raise RebuildError(f"{context} is not a list")
    if len(value) != EXPECTED_EPOCHS:
        raise RebuildError(
            f"{context} has {len(value)} values, expected {EXPECTED_EPOCHS}"
        )

    rows: list[tuple[float, ...]] = []
    width: int | None = None
    for epoch, row in enumerate(value):
        array = np.asarray(row)
        if array.ndim == 0:
            array = array.reshape(1)
        if array.ndim != 1:
            raise RebuildError(f"{context}[{epoch}] is not one-dimensional")
        numbers = tuple(
            _number(
                item,
                context=f"{context}[{epoch}][{index}]",
                require_finite=True,
            )
            for index, item in enumerate(array.tolist())
        )
        if not numbers:
            raise RebuildError(f"{context}[{epoch}] is empty")
        if width is None:
            width = len(numbers)
        elif len(numbers) != width:
            raise RebuildError(
                f"{context}[{epoch}] has {len(numbers)} classes, expected {width}"
            )
        rows.append(numbers)

    assert width is not None
    return {
        f"val/dice_class_{class_index}": tuple(
            row[class_index] for row in rows
        )
        for class_index in range(width)
    }


def _checkpoint_path(results_root: Path, experiment: str, fold: int) -> Path:
    return (
        results_root
        / experiment
        / "Dataset663_BrainMet"
        / MODEL_NAME
        / f"fold_{fold}"
        / "checkpoint_final.pth"
    )


def _tensorboard_path(results_root: Path, experiment: str, fold: int) -> Path:
    return _checkpoint_path(results_root, experiment, fold).parent / "tensorboard"


def _capture_tensorboard_state(path: Path) -> TensorboardState:
    if path.is_symlink():
        raise RebuildError(f"refusing symlinked TensorBoard directory: {path}")
    if not path.exists():
        return TensorboardState(path=str(path), files=None)
    if not path.is_dir():
        raise RebuildError(f"TensorBoard path is not a directory: {path}")
    files: dict[str, dict[str, Any]] = {}
    for item in sorted(path.rglob("*")):
        relative = str(item.relative_to(path))
        if item.is_symlink():
            raise RebuildError(f"refusing symlink inside TensorBoard directory: {item}")
        if item.is_dir():
            files[relative] = {"type": "directory"}
        elif item.is_file():
            stat_result = item.stat()
            files[relative] = {
                "type": "file",
                "size_bytes": stat_result.st_size,
                "mtime_ns": stat_result.st_mtime_ns,
                "sha256": _sha256(item),
            }
        else:
            raise RebuildError(f"unsupported entry inside TensorBoard directory: {item}")
    return TensorboardState(path=str(path), files=files)


def validate_recovery_manifest(
    manifest_path: Path, results_root: Path
) -> RecoveryProvenance:
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RebuildError(
            f"recovery manifest must be a regular, non-symlinked file: {manifest_path}"
        )
    try:
        payload = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RebuildError(f"cannot read recovery manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RebuildError(f"recovery manifest root is not an object: {manifest_path}")
    if payload.get("source_quarantine_preserved") is not True:
        raise RebuildError(f"recovery manifest does not preserve quarantine: {manifest_path}")
    active_results = payload.get("active_results")
    if not isinstance(active_results, str) or Path(active_results).resolve() != results_root:
        raise RebuildError(
            f"recovery manifest active_results does not match {results_root}: "
            f"{active_results!r}"
        )
    fold_records = payload.get("folds")
    if not isinstance(fold_records, list):
        raise RebuildError(f"recovery manifest folds is not a list: {manifest_path}")
    actual: dict[tuple[str, int], str] = {}
    for index, record in enumerate(fold_records):
        if not isinstance(record, dict):
            raise RebuildError(f"recovery manifest folds[{index}] is not an object")
        experiment = record.get("experiment")
        fold = record.get("fold")
        action = record.get("action")
        if not isinstance(experiment, str) or isinstance(fold, bool) or not isinstance(fold, int):
            raise RebuildError(f"invalid recovery identity at folds[{index}]")
        if not isinstance(action, str):
            raise RebuildError(f"invalid recovery action at folds[{index}]")
        key = (experiment, fold)
        if key in actual:
            raise RebuildError(f"duplicate recovery record for {experiment} fold {fold}")
        actual[key] = action
    expected = {key: action_epoch[0] for key, action_epoch in EXPECTED_RECOVERY.items()}
    if actual != expected:
        missing = sorted(expected.keys() - actual.keys())
        extra = sorted(actual.keys() - expected.keys())
        wrong = sorted(
            (key, expected[key], actual[key])
            for key in expected.keys() & actual.keys()
            if expected[key] != actual[key]
        )
        raise RebuildError(
            "recovery manifest does not match the audited exp17 plan: "
            f"missing={missing}, extra={extra}, wrong_actions={wrong}"
        )
    return RecoveryProvenance(path=str(manifest_path), sha256=_sha256(manifest_path))


def load_history(results_root: Path, experiment: str, fold: int) -> FoldHistory:
    checkpoint_path = _checkpoint_path(results_root, experiment, fold)
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise RebuildError(
            f"final checkpoint must be a regular, non-symlinked file: {checkpoint_path}"
        )
    source_stat = checkpoint_path.stat()
    checkpoint_sha256 = _sha256(checkpoint_path)
    audit, checkpoint = audit_checkpoint(checkpoint_path, fold=fold, kind="final")
    if not audit.valid or checkpoint is None:
        details = "; ".join(audit.errors) if audit.errors else "checkpoint is missing"
        raise RebuildError(f"invalid final checkpoint {checkpoint_path}: {details}")
    if audit.epoch != EXPECTED_EPOCHS:
        raise RebuildError(
            f"{checkpoint_path} ends at epoch {audit.epoch}, expected {EXPECTED_EPOCHS}"
        )
    if audit.network_signature is None:
        raise RebuildError(f"{checkpoint_path} has no audited network signature")
    if (
        audit.size_bytes != source_stat.st_size
        or audit.mtime_ns != source_stat.st_mtime_ns
    ):
        raise RebuildError(f"source checkpoint changed while loading: {checkpoint_path}")

    logging = checkpoint.get("logging")
    if not isinstance(logging, dict):
        raise RebuildError(f"{checkpoint_path}: logging is not a dictionary")

    scalars = {
        tag: _numeric_series(
            logging.get(key),
            context=f"{checkpoint_path}: logging.{key}",
            require_finite=True,
        )
        for tag, key in SCALAR_KEYS.items()
    }
    scalars.update(
        _dice_series(
            logging.get("dice_per_class_or_region"),
            context=f"{checkpoint_path}: logging.dice_per_class_or_region",
        )
    )
    walltimes = _numeric_series(
        logging.get("epoch_end_timestamps"),
        context=f"{checkpoint_path}: logging.epoch_end_timestamps",
        require_finite=True,
    )
    if any(right <= left for left, right in pairwise(walltimes)):
        raise RebuildError(f"{checkpoint_path}: epoch end timestamps are not increasing")
    if any(value < 0 for value in scalars["train/learning_rate"]):
        raise RebuildError(f"{checkpoint_path}: learning-rate history contains negatives")
    for tag in ("val/mean_fg_dice", "val/ema_fg_dice", "val/dice_class_0"):
        if any(value < 0 or value > 1 for value in scalars[tag]):
            raise RebuildError(f"{checkpoint_path}: {tag} contains values outside [0, 1]")
    dice_tags = sorted(tag for tag in scalars if tag.startswith("val/dice_class_"))
    if dice_tags != ["val/dice_class_0"]:
        raise RebuildError(
            f"{checkpoint_path}: expected one foreground Dice stream, got {dice_tags}"
        )
    if not np.allclose(
        scalars["val/mean_fg_dice"],
        scalars["val/dice_class_0"],
        rtol=0,
        atol=1e-7,
    ):
        raise RebuildError(f"{checkpoint_path}: mean Dice differs from class-0 Dice")
    expected_ema = [scalars["val/mean_fg_dice"][0]]
    for mean_dice in scalars["val/mean_fg_dice"][1:]:
        expected_ema.append(0.9 * expected_ema[-1] + 0.1 * mean_dice)
    if not np.allclose(
        scalars["val/ema_fg_dice"], expected_ema, rtol=0, atol=1e-6
    ):
        raise RebuildError(f"{checkpoint_path}: EMA Dice recurrence is inconsistent")

    history = FoldHistory(
        experiment=experiment,
        fold=fold,
        checkpoint=str(checkpoint_path),
        checkpoint_size_bytes=int(audit.size_bytes or 0),
        checkpoint_mtime_ns=int(audit.mtime_ns or 0),
        checkpoint_device=source_stat.st_dev,
        checkpoint_inode=source_stat.st_ino,
        checkpoint_sha256=checkpoint_sha256,
        network_signature=audit.network_signature,
        walltimes=walltimes,
        scalars=scalars,
    )
    del checkpoint
    gc.collect()
    return history


def load_all_histories(results_root: Path) -> list[FoldHistory]:
    if not results_root.is_dir():
        raise RebuildError(f"results root is missing: {results_root}")
    expected_checkpoints = {
        _checkpoint_path(results_root, experiment, fold)
        for experiment in EXPECTED_EXPERIMENTS
        for fold in range(5)
    }
    for experiment in EXPECTED_EXPERIMENTS:
        experiment_root = results_root / experiment
        discovered = set(experiment_root.rglob("checkpoint_final.pth"))
        expected = {path for path in expected_checkpoints if experiment in path.parts}
        if discovered != expected:
            missing = sorted(str(path) for path in expected - discovered)
            extra = sorted(str(path) for path in discovered - expected)
            raise RebuildError(
                f"unexpected final-checkpoint layout for {experiment}: "
                f"missing={missing}, extra={extra}"
            )

    histories: list[FoldHistory] = []
    for experiment in EXPECTED_EXPERIMENTS:
        for fold in range(5):
            print(f"Validating {experiment} fold {fold}...", flush=True)
            histories.append(load_history(results_root, experiment, fold))
    if len(histories) != 20:
        raise RebuildError(f"validated {len(histories)} folds, expected 20")
    signatures = {history.network_signature for history in histories}
    if len(signatures) != 1:
        raise RebuildError(f"final checkpoints have different network signatures: {signatures}")
    return histories


def _write_history(output: Path, history: FoldHistory) -> None:
    from torch.utils.tensorboard import SummaryWriter

    output.mkdir(parents=True, exist_ok=False)
    writer = SummaryWriter(log_dir=str(output))
    try:
        for tag, values in history.scalars.items():
            for step, (value, walltime) in enumerate(
                zip(values, history.walltimes, strict=True)
            ):
                writer.add_scalar(tag, value, step, walltime=walltime)
        writer.flush()
    finally:
        writer.close()


def _verify_history(output: Path, history: FoldHistory) -> dict[str, int]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_files = sorted(output.glob("events.out.tfevents.*"))
    if len(event_files) != 1:
        raise RebuildError(
            f"{output} contains {len(event_files)} event files, expected exactly one"
        )
    accumulator = EventAccumulator(str(output), size_guidance={"scalars": 0})
    accumulator.Reload()
    actual_tags = set(accumulator.Tags().get("scalars", []))
    expected_tags = set(history.scalars)
    if actual_tags != expected_tags:
        raise RebuildError(
            f"{output} scalar tags differ: actual={sorted(actual_tags)}, "
            f"expected={sorted(expected_tags)}"
        )

    counts: dict[str, int] = {}
    for tag, expected_values in history.scalars.items():
        events = accumulator.Scalars(tag)
        expected_steps = list(range(EXPECTED_EPOCHS))
        actual_steps = [event.step for event in events]
        if actual_steps != expected_steps:
            raise RebuildError(f"{output} tag {tag} does not contain steps 0..999")
        actual_values = np.asarray([event.value for event in events], dtype=np.float32)
        expected_array = np.asarray(expected_values, dtype=np.float32)
        if not np.array_equal(actual_values, expected_array, equal_nan=True):
            raise RebuildError(f"{output} tag {tag} values differ after serialization")
        actual_walltimes = [event.wall_time for event in events]
        if actual_walltimes != list(history.walltimes):
            raise RebuildError(f"{output} tag {tag} wall times differ after serialization")
        counts[tag] = len(events)
    return counts


def _rollback_install(state: InstallState) -> None:
    errors: list[str] = []
    for target, reconstructed in reversed(state.installed):
        try:
            if not target.is_dir():
                raise RebuildError(f"installed TensorBoard directory is missing: {target}")
            reconstructed.parent.mkdir(parents=True, exist_ok=True)
            if reconstructed.exists() or reconstructed.is_symlink():
                raise RebuildError(f"rollback destination already exists: {reconstructed}")
            os.rename(target, reconstructed)
        except Exception as exc:
            errors.append(str(exc))
    if not errors:
        state.installed.clear()

    for target, archived in reversed(state.archived):
        try:
            if target.exists() or target.is_symlink():
                raise RebuildError(f"cannot restore over existing target: {target}")
            if not archived.is_dir():
                raise RebuildError(f"archived TensorBoard directory is missing: {archived}")
            target.parent.mkdir(parents=True, exist_ok=True)
            os.rename(archived, target)
        except Exception as exc:
            errors.append(str(exc))
    if not errors:
        state.archived.clear()
    if errors:
        raise RebuildError("TensorBoard rollback failed: " + "; ".join(errors))


def _install_reconstructed(
    *,
    results_root: Path,
    staging: Path,
    histories: list[FoldHistory],
    state: InstallState,
) -> None:
    for history in histories:
        relative = Path(history.experiment) / f"fold_{history.fold}" / "tensorboard"
        target = _tensorboard_path(results_root, history.experiment, history.fold)
        reconstructed = staging / "reconstructed" / relative
        archived = staging / "previous" / relative
        if target.is_symlink():
            raise RebuildError(f"refusing symlinked TensorBoard target: {target}")
        if target.exists():
            if not target.is_dir():
                raise RebuildError(f"TensorBoard target is not a directory: {target}")
            archived.parent.mkdir(parents=True, exist_ok=True)
            os.rename(target, archived)
            state.archived.append((target, archived))
        _publish_no_replace(reconstructed, target)
        state.installed.append((target, reconstructed))


def rebuild(results_root: Path, archive_root: Path, recovery_manifest: Path) -> Path:
    results_root = results_root.resolve()
    archive_root = archive_root.absolute()
    recovery_manifest = recovery_manifest.absolute()
    if archive_root.is_symlink():
        raise RebuildError(f"refusing symlinked archive: {archive_root}")
    archive_root = archive_root.resolve()
    if archive_root.is_relative_to(results_root) or results_root.is_relative_to(archive_root):
        raise RebuildError(
            f"archive and source trees must be disjoint: source={results_root}, "
            f"archive={archive_root}"
        )
    if archive_root.exists() or archive_root.is_symlink():
        raise RebuildError(f"refusing to overwrite existing archive: {archive_root}")
    recovery = validate_recovery_manifest(recovery_manifest, results_root)
    histories = load_all_histories(results_root)
    previous_states = {
        (history.experiment, history.fold): _capture_tensorboard_state(
            _tensorboard_path(results_root, history.experiment, history.fold)
        )
        for history in histories
    }

    archive_root.parent.mkdir(parents=True, exist_ok=True)
    if results_root.stat().st_dev != archive_root.parent.stat().st_dev:
        raise RebuildError(
            f"results and archive must share one filesystem: {results_root}, "
            f"{archive_root.parent}"
        )
    _verify_noreplace_support(archive_root.parent)
    staging = archive_root.with_name(f".{archive_root.name}.tmp_{os.getpid()}")
    if staging.exists() or staging.is_symlink():
        raise RebuildError(f"staging path already exists: {staging}")
    staging.mkdir()
    (staging / "reconstructed").mkdir()
    (staging / "previous").mkdir()

    records: list[dict[str, Any]] = []
    install_state = InstallState(installed=[], archived=[])
    published = False
    try:
        for history in histories:
            relative = Path(history.experiment) / f"fold_{history.fold}" / "tensorboard"
            destination = staging / "reconstructed" / relative
            print(f"Writing {relative}...", flush=True)
            _write_history(destination, history)
            counts = _verify_history(destination, history)
            event_file = next(destination.glob("events.out.tfevents.*"))
            previous = previous_states[(history.experiment, history.fold)]
            archived_previous = (
                str(archive_root / "previous" / relative)
                if previous.files is not None
                else None
            )
            records.append(
                {
                    "experiment": history.experiment,
                    "fold": history.fold,
                    "source_checkpoint": history.checkpoint,
                    "source_checkpoint_size_bytes": history.checkpoint_size_bytes,
                    "source_checkpoint_mtime_ns": history.checkpoint_mtime_ns,
                    "source_checkpoint_device": history.checkpoint_device,
                    "source_checkpoint_inode": history.checkpoint_inode,
                    "source_checkpoint_sha256": history.checkpoint_sha256,
                    "network_signature": history.network_signature,
                    "event_directory": str(
                        _tensorboard_path(
                            results_root, history.experiment, history.fold
                        )
                    ),
                    "event_file_sha256": _sha256(event_file),
                    "previous_tensorboard": previous.files,
                    "archived_previous_directory": archived_previous,
                    "scalar_counts": counts,
                    "steps": [0, EXPECTED_EPOCHS - 1],
                }
            )

        for history in histories:
            source = Path(history.checkpoint)
            if source.is_symlink() or not source.is_file():
                raise RebuildError(f"source checkpoint changed during rebuild: {source}")
            source_stat = source.stat()
            if (
                source_stat.st_size != history.checkpoint_size_bytes
                or source_stat.st_mtime_ns != history.checkpoint_mtime_ns
                or source_stat.st_dev != history.checkpoint_device
                or source_stat.st_ino != history.checkpoint_inode
                or _sha256(source) != history.checkpoint_sha256
            ):
                raise RebuildError(f"source checkpoint changed during rebuild: {source}")

        for key, previous in previous_states.items():
            current = _capture_tensorboard_state(Path(previous.path))
            if current != previous:
                raise RebuildError(
                    f"TensorBoard source changed during rebuild for {key}: {previous.path}"
                )

        manifest = {
            "method": "reconstructed from logging embedded in checkpoint_final.pth",
            "source_results_root": str(results_root),
            "recovery_manifest": recovery.path,
            "recovery_manifest_sha256": recovery.sha256,
            "archive_root": str(archive_root),
            "installation": "one reconstructed event file at each normal fold/tensorboard path",
            "checkpoints_modified": False,
            "original_recovery_quarantine_modified": False,
            "expected_epochs": EXPECTED_EPOCHS,
            "walltime_policy": "checkpoint logging.epoch_end_timestamps",
            "folds": records,
        }
        (staging / "reconstruction_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    # All expensive loading, serialization, and event parsing has completed.
    # Defer normal Slurm cancellation signals only for the short sequence of
    # same-filesystem directory renames that changes the canonical paths. Keep
    # rollback inside the deferred-signal region so a pending signal cannot
    # interrupt it.
    try:
        with _defer_install_signals():
            try:
                _install_reconstructed(
                    results_root=results_root,
                    staging=staging,
                    histories=histories,
                    state=install_state,
                )
                shutil.rmtree(staging / "reconstructed")
                _publish_no_replace(staging, archive_root)
                published = True
                install_state.installed.clear()
                install_state.archived.clear()
            except BaseException as original_error:
                rollback_error: Exception | None = None
                if install_state.installed or install_state.archived:
                    try:
                        _rollback_install(install_state)
                    except Exception as exc:
                        rollback_error = exc
                if rollback_error is None:
                    shutil.rmtree(staging, ignore_errors=True)
                if rollback_error is not None:
                    raise RebuildError(
                        f"rebuild failed ({original_error}); {rollback_error}; "
                        f"staging was preserved at {staging}"
                    ) from original_error
                raise
    except BaseException:
        # Covers a failure installing the temporary handlers. A failed rollback
        # keeps non-empty state and its staging tree for explicit recovery.
        if not install_state.installed and not install_state.archived:
            shutil.rmtree(staging, ignore_errors=True)
        raise

    if not published:
        raise RebuildError("TensorBoard archive was not published")
    print("Installed and verified 20 complete histories in the normal fold directories")
    print(f"Previous partial histories and provenance: {archive_root}")
    return archive_root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path, required=True)
    parser.add_argument("--recovery-manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        rebuild(
            args.results_root,
            args.archive_root,
            args.recovery_manifest,
        )
    except RebuildError as exc:
        raise SystemExit(f"FATAL: {exc}") from exc


if __name__ == "__main__":
    main()
