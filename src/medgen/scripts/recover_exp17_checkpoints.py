"""Restore audited exp17 nnU-Net checkpoints without copying mixed run artifacts.

The source quarantine is treated as immutable. A clean active results tree is
built from the checkpoints selected by the audit report and the static nnU-Net
model metadata. Training logs, progress plots, validation outputs, and stale
branches remain in quarantine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

MODEL_NAME = "nnUNetTrainerBrainMets__nnUNetResEncUNetLPlansD600__3d_fullres"
EXPECTED_EXPERIMENTS = {
    "exp17_1_hybrid_25syn_exp1_to_exp48c_t025_d663",
    "exp17_2_hybrid_50syn_exp1_to_exp48c_t025_d663",
    "exp17_3_hybrid_105syn_exp1_to_exp48c_t025_d663",
    "exp17_4_hybrid_210syn_exp1_to_exp48c_t025_d663",
}
EXPECTED_RECOVERY = {
    ("exp17_1_hybrid_25syn_exp1_to_exp48c_t025_d663", 0): ("KEEP_COMPLETE", 1000),
    ("exp17_1_hybrid_25syn_exp1_to_exp48c_t025_d663", 1): ("KEEP_COMPLETE", 1000),
    ("exp17_1_hybrid_25syn_exp1_to_exp48c_t025_d663", 2): ("KEEP_COMPLETE", 1000),
    ("exp17_1_hybrid_25syn_exp1_to_exp48c_t025_d663", 3): ("RESUME_BEST", 25),
    ("exp17_1_hybrid_25syn_exp1_to_exp48c_t025_d663", 4): ("KEEP_COMPLETE", 1000),
    ("exp17_2_hybrid_50syn_exp1_to_exp48c_t025_d663", 0): ("RESUME_BEST", 977),
    ("exp17_2_hybrid_50syn_exp1_to_exp48c_t025_d663", 1): ("RESUME_LATEST", 900),
    ("exp17_2_hybrid_50syn_exp1_to_exp48c_t025_d663", 2): ("RESUME_LATEST", 750),
    ("exp17_2_hybrid_50syn_exp1_to_exp48c_t025_d663", 3): ("RESUME_LATEST", 700),
    ("exp17_2_hybrid_50syn_exp1_to_exp48c_t025_d663", 4): ("KEEP_COMPLETE", 1000),
    ("exp17_3_hybrid_105syn_exp1_to_exp48c_t025_d663", 0): ("RESUME_LATEST", 400),
    ("exp17_3_hybrid_105syn_exp1_to_exp48c_t025_d663", 1): ("RESUME_LATEST", 650),
    ("exp17_3_hybrid_105syn_exp1_to_exp48c_t025_d663", 2): ("RESUME_LATEST", 850),
    ("exp17_3_hybrid_105syn_exp1_to_exp48c_t025_d663", 3): ("RESUME_BEST", 594),
    ("exp17_3_hybrid_105syn_exp1_to_exp48c_t025_d663", 4): ("RESUME_BEST", 18),
    ("exp17_4_hybrid_210syn_exp1_to_exp48c_t025_d663", 0): ("RESUME_BEST", 715),
    ("exp17_4_hybrid_210syn_exp1_to_exp48c_t025_d663", 1): ("RESUME_LATEST", 700),
    ("exp17_4_hybrid_210syn_exp1_to_exp48c_t025_d663", 2): ("RESUME_BEST", 663),
    ("exp17_4_hybrid_210syn_exp1_to_exp48c_t025_d663", 3): ("RESUME_LATEST", 450),
    ("exp17_4_hybrid_210syn_exp1_to_exp48c_t025_d663", 4): ("RESTART", 0),
}
ALLOWED_ACTIONS = {
    "KEEP_COMPLETE",
    "RESUME_LATEST",
    "RESUME_BEST",
    "RESTART",
}
STATIC_MODEL_FILES = (
    "dataset.json",
    "dataset_fingerprint.json",
    "plans.json",
)
TRAIN_JOB_NAMES = {
    "exp17_1_h25",
    "exp17_2_h50",
    "exp17_3_h105",
    "exp17_4_h210",
}


class RecoveryError(RuntimeError):
    """Raised when recovery cannot proceed without risking existing data."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_independent(source: Path, destination: Path) -> None:
    """Copy a file using copy-on-write when available, never a hard link."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise RecoveryError(f"refusing to overwrite recovery target: {destination}")
    command = [
        "cp",
        "--reflink=auto",
        "--preserve=mode,timestamps",
        "--",
        str(source),
        str(destination),
    ]
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RecoveryError(f"failed to copy {source} to {destination}: {exc}") from exc
    source_stat = source.stat()
    destination_stat = destination.stat()
    if (source_stat.st_dev, source_stat.st_ino) == (
        destination_stat.st_dev,
        destination_stat.st_ino,
    ):
        raise RecoveryError(f"recovery copy is a hard link: {destination}")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _checkpoint_path(fold: dict[str, Any], kind: str) -> Path:
    return Path(fold["checkpoints"][kind]["path"])


def _checkpoint_valid(fold: dict[str, Any], kind: str) -> bool:
    return bool(fold["checkpoints"][kind]["valid"])


def _validate_action(fold: dict[str, Any]) -> None:
    action = fold["action"]
    best = _checkpoint_valid(fold, "best")
    latest = _checkpoint_valid(fold, "latest")
    final = _checkpoint_valid(fold, "final")
    lineage = fold["best_matches_anchor"]

    valid = {
        "KEEP_COMPLETE": final and best and lineage is True,
        "RESUME_LATEST": latest and best and not final and lineage is True,
        "RESUME_BEST": best and not ((final or latest) and lineage is True),
        "RESTART": not (best or latest or final),
    }
    if action not in valid or not valid[action]:
        raise RecoveryError(
            f"audit action is inconsistent for {fold['experiment']} fold {fold['fold']}: {action}"
        )


def _validate_checkpoint_snapshot(checkpoint: dict[str, Any]) -> None:
    path = Path(checkpoint["path"])
    if path.is_symlink():
        raise RecoveryError(f"checkpoint source must not be a symlink: {path}")
    expected_exists = bool(checkpoint["exists"])
    if path.is_file() != expected_exists:
        raise RecoveryError(f"checkpoint existence changed since audit: {path}")
    if not expected_exists:
        return
    stat = path.stat()
    if stat.st_size != checkpoint["size_bytes"]:
        raise RecoveryError(f"checkpoint size changed since audit: {path}")
    if stat.st_mtime_ns != checkpoint["mtime_ns"]:
        raise RecoveryError(f"checkpoint mtime changed since audit: {path}")


def load_and_validate_audit(audit_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        payload = json.loads(audit_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RecoveryError(f"cannot read audit report {audit_path}: {exc}") from exc

    results_root = Path(payload.get("results_root", ""))
    shadows_root = Path(payload.get("shadows_root", ""))
    if not results_root.is_dir():
        raise RecoveryError(f"quarantined results root is missing: {results_root}")
    if not shadows_root.is_dir():
        raise RecoveryError(f"quarantined shadows root is missing: {shadows_root}")
    if results_root != audit_path.parent / "results":
        raise RecoveryError(f"unexpected quarantined results root: {results_root}")
    if shadows_root != audit_path.parent / "preprocessed_shadows":
        raise RecoveryError(f"unexpected quarantined shadows root: {shadows_root}")

    folds = payload.get("folds")
    if not isinstance(folds, list):
        raise RecoveryError("audit report has no fold list")
    keys = [(fold.get("experiment"), fold.get("fold")) for fold in folds]
    expected_keys = {(experiment, fold) for experiment in EXPECTED_EXPERIMENTS for fold in range(5)}
    if len(keys) != 20 or set(keys) != expected_keys or len(keys) != len(set(keys)):
        raise RecoveryError("audit report does not contain exactly the expected 20 exp17 folds")

    selected_structures: set[tuple[str, int, int, int, int]] = set()
    for fold in folds:
        experiment = fold["experiment"]
        fold_number = fold["fold"]
        if fold.get("action") not in ALLOWED_ACTIONS:
            raise RecoveryError(
                f"unresolved audit action for {experiment} fold {fold_number}: {fold.get('action')}"
            )
        expected_action, expected_epoch = EXPECTED_RECOVERY[(experiment, fold_number)]
        if fold["action"] != expected_action:
            raise RecoveryError(
                f"audit action changed for {experiment} fold {fold_number}: "
                f"expected {expected_action}, got {fold['action']}"
            )
        if fold.get("split_valid") is not True:
            raise RecoveryError(f"invalid split for {experiment} fold {fold_number}")
        expected_fold_dir = (
            results_root / experiment / "Dataset663_BrainMet" / MODEL_NAME / f"fold_{fold_number}"
        )
        if Path(fold["fold_dir"]) != expected_fold_dir:
            raise RecoveryError(f"unexpected fold path in audit: {fold['fold_dir']}")
        _validate_action(fold)
        if fold["action"] == "KEEP_COMPLETE":
            recovery_epoch = fold["checkpoints"]["final"]["epoch"]
        elif fold["action"] == "RESUME_LATEST":
            recovery_epoch = fold["checkpoints"]["latest"]["epoch"]
        elif fold["action"] == "RESUME_BEST":
            recovery_epoch = fold["checkpoints"]["best"]["epoch"]
        else:
            recovery_epoch = 0
        if recovery_epoch != expected_epoch:
            raise RecoveryError(
                f"audit epoch changed for {experiment} fold {fold_number}: "
                f"expected {expected_epoch}, got {recovery_epoch}"
            )
        for kind, checkpoint in fold["checkpoints"].items():
            expected_path = expected_fold_dir / f"checkpoint_{kind}.pth"
            if Path(checkpoint["path"]) != expected_path:
                raise RecoveryError(f"unexpected checkpoint path in audit: {checkpoint['path']}")
            _validate_checkpoint_snapshot(checkpoint)
        for kind in _selected_kinds(fold["action"]):
            checkpoint = fold["checkpoints"][kind]
            structure = (
                checkpoint.get("network_signature"),
                checkpoint.get("network_tensor_count"),
                checkpoint.get("network_parameter_count"),
                checkpoint.get("optimizer_state_entries"),
                checkpoint.get("optimizer_param_groups"),
            )
            if (
                not isinstance(structure[0], str)
                or not structure[0]
                or any(not isinstance(value, int) or value <= 0 for value in structure[1:])
            ):
                raise RecoveryError(
                    f"missing selected-checkpoint structure for {experiment} "
                    f"fold {fold_number} {kind}"
                )
            selected_structures.add(structure)

    if len(selected_structures) != 1:
        raise RecoveryError("selected checkpoints do not share one network and optimizer structure")

    return payload, sorted(folds, key=lambda item: (item["experiment"], item["fold"]))


def _check_no_training_jobs() -> None:
    user = os.environ.get("USER")
    if not user:
        raise RecoveryError("USER is not set")
    try:
        result = subprocess.run(
            ["squeue", "-h", "-u", user, "-o", "%j"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RecoveryError(f"could not verify the Slurm queue: {exc}") from exc
    active = sorted(set(result.stdout.splitlines()) & TRAIN_JOB_NAMES)
    if active:
        raise RecoveryError(f"exp17 training jobs are still present: {', '.join(active)}")


def _tree_snapshot(root: Path) -> list[tuple[str, str, int, int, str | None]]:
    """Capture content-relevant metadata without following source symlinks."""
    snapshot: list[tuple[str, str, int, int, str | None]] = []
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        stat = path.lstat()
        if path.is_symlink():
            kind = "symlink"
            target = os.readlink(path)
        elif path.is_dir():
            kind = "directory"
            target = None
        elif path.is_file():
            kind = "file"
            target = None
        else:
            kind = "other"
            target = None
        snapshot.append((relative, kind, stat.st_size, stat.st_mtime_ns, target))
    return snapshot


def _selected_kinds(action: str) -> tuple[str, ...]:
    if action == "KEEP_COMPLETE":
        return ("best", "final")
    if action == "RESUME_LATEST":
        return ("best", "latest")
    if action == "RESUME_BEST":
        return ("best",)
    if action == "RESTART":
        return ()
    raise RecoveryError(f"unsupported recovery action: {action}")


def _print_plan(folds: list[dict[str, Any]]) -> None:
    counts = Counter(fold["action"] for fold in folds)
    print("=== exp17 recovery plan ===")
    print(f"{'experiment':64s} {'fold':>4s} {'epoch':>6s} action")
    for fold in folds:
        action = fold["action"]
        if action == "KEEP_COMPLETE":
            epoch = fold["checkpoints"]["final"]["epoch"]
        elif action == "RESUME_LATEST":
            epoch = fold["checkpoints"]["latest"]["epoch"]
        elif action == "RESUME_BEST":
            epoch = fold["checkpoints"]["best"]["epoch"]
        else:
            epoch = 0
        print(f"{fold['experiment']:64s} {fold['fold']:4d} {epoch:6d} {action}")
    print("Summary: " + ", ".join(f"{action}={counts[action]}" for action in sorted(counts)))


def recover(
    *,
    audit_path: Path,
    active_results: Path,
    active_logs_root: Path | None = None,
    apply: bool,
    check_jobs: bool = True,
    copier: Callable[[Path, Path], None] = _copy_independent,
) -> Path | None:
    audit_sha256 = _sha256(audit_path)
    payload, folds = load_and_validate_audit(audit_path)
    quarantine_results = Path(payload["results_root"])
    source_snapshot = _tree_snapshot(quarantine_results)
    if check_jobs:
        _check_no_training_jobs()
    _print_plan(folds)

    for experiment in EXPECTED_EXPERIMENTS:
        target = active_results / experiment
        if target.exists() or target.is_symlink():
            raise RecoveryError(f"active result target already exists: {target}")
        if active_logs_root is not None:
            log_directory = active_logs_root / experiment
            if log_directory.is_dir() and any(log_directory.iterdir()):
                raise RecoveryError(f"active log directory is not empty: {log_directory}")
            if log_directory.exists() and not log_directory.is_dir():
                raise RecoveryError(f"active log path is not a directory: {log_directory}")

    source_roots: dict[str, Path] = {}
    for fold in folds:
        source_model = Path(fold["fold_dir"]).parent
        existing = source_roots.setdefault(fold["experiment"], source_model)
        if existing != source_model:
            raise RecoveryError(f"multiple model roots for {fold['experiment']}")
    for source_model in source_roots.values():
        for required in ("dataset.json", "plans.json"):
            if not (source_model / required).is_file():
                raise RecoveryError(f"missing static model metadata: {source_model / required}")

    if not apply:
        print("Dry run complete. No files were changed.")
        return None

    if _sha256(audit_path) != audit_sha256:
        raise RecoveryError("audit report changed while the recovery plan was validated")
    if active_logs_root is not None:
        for experiment in EXPECTED_EXPERIMENTS:
            (active_logs_root / experiment).mkdir(parents=True, exist_ok=True)

    active_results.mkdir(parents=True, exist_ok=True)
    stale_staging = sorted(active_results.glob(f".exp17_recovery_{audit_path.stem}_*"))
    if stale_staging:
        raise RecoveryError(
            "stale recovery staging exists: " + ", ".join(str(path) for path in stale_staging)
        )
    staging = active_results / f".exp17_recovery_{audit_path.stem}_{os.getpid()}"
    if staging.exists() or staging.is_symlink():
        raise RecoveryError(f"staging target already exists: {staging}")
    staging.mkdir()

    source_hashes: dict[Path, str] = {}
    manifest_folds: list[dict[str, Any]] = []
    manifest_static: dict[str, dict[str, Any]] = {}
    committed: list[tuple[Path, Path]] = []
    manifest_path: Path | None = None
    manifest_written = False
    try:
        for experiment, source_model in source_roots.items():
            destination_model = staging / experiment / "Dataset663_BrainMet" / MODEL_NAME
            destination_model.mkdir(parents=True)
            manifest_static[experiment] = {}
            for filename in STATIC_MODEL_FILES:
                source = source_model / filename
                if source.is_file():
                    destination = destination_model / filename
                    copier(source, destination)
                    source_hash = source_hashes.setdefault(source, _sha256(source))
                    if _sha256(destination) != source_hash:
                        raise RecoveryError(
                            f"static metadata hash mismatch after copy: {destination}"
                        )
                    manifest_static[experiment][filename] = {
                        "source": str(source),
                        "destination": str(
                            active_results
                            / experiment
                            / "Dataset663_BrainMet"
                            / MODEL_NAME
                            / filename
                        ),
                        "sha256": source_hash,
                    }

        for fold in folds:
            action = fold["action"]
            fold_number = fold["fold"]
            destination_fold = (
                staging
                / fold["experiment"]
                / "Dataset663_BrainMet"
                / MODEL_NAME
                / f"fold_{fold_number}"
            )
            copied: dict[str, dict[str, Any]] = {}
            if action != "RESTART":
                destination_fold.mkdir()
            for kind in _selected_kinds(action):
                source = _checkpoint_path(fold, kind)
                destination = destination_fold / f"checkpoint_{kind}.pth"
                final_destination = (
                    active_results
                    / fold["experiment"]
                    / "Dataset663_BrainMet"
                    / MODEL_NAME
                    / f"fold_{fold_number}"
                    / f"checkpoint_{kind}.pth"
                )
                copier(source, destination)
                source_hash = source_hashes.setdefault(source, _sha256(source))
                destination_hash = _sha256(destination)
                if destination_hash != source_hash:
                    raise RecoveryError(f"checkpoint hash mismatch after copy: {destination}")
                copied[kind] = {
                    "source": str(source),
                    "destination": str(final_destination),
                    "epoch": fold["checkpoints"][kind]["epoch"],
                    "sha256": source_hash,
                }

            if action == "RESUME_BEST":
                source = _checkpoint_path(fold, "best")
                destination = destination_fold / "checkpoint_latest.pth"
                final_destination = (
                    active_results
                    / fold["experiment"]
                    / "Dataset663_BrainMet"
                    / MODEL_NAME
                    / f"fold_{fold_number}"
                    / "checkpoint_latest.pth"
                )
                copier(source, destination)
                source_hash = source_hashes.setdefault(source, _sha256(source))
                if _sha256(destination) != source_hash:
                    raise RecoveryError(f"promoted best hash mismatch: {destination}")
                best_destination = destination_fold / "checkpoint_best.pth"
                if (
                    best_destination.stat().st_dev,
                    best_destination.stat().st_ino,
                ) == (destination.stat().st_dev, destination.stat().st_ino):
                    raise RecoveryError(f"promoted best was hard-linked: {destination}")
                copied["promoted_latest"] = {
                    "source": str(source),
                    "destination": str(final_destination),
                    "epoch": fold["checkpoints"]["best"]["epoch"],
                    "sha256": source_hash,
                }

            manifest_folds.append(
                {
                    "experiment": fold["experiment"],
                    "fold": fold_number,
                    "action": action,
                    "copied_checkpoints": copied,
                }
            )

        manifest = {
            "audit": str(audit_path),
            "audit_sha256": audit_sha256,
            "quarantine_results": payload["results_root"],
            "active_results": str(active_results),
            "source_quarantine_preserved": True,
            "static_model_files": manifest_static,
            "folds": manifest_folds,
        }
        manifest_path = (
            active_results / ".exp17_recovery_manifests" / f"recovery_{audit_path.stem}.json"
        )
        if manifest_path.exists() or manifest_path.is_symlink():
            raise RecoveryError(f"recovery manifest already exists: {manifest_path}")
        for experiment in sorted(EXPECTED_EXPERIMENTS):
            source = staging / experiment
            destination = active_results / experiment
            if destination.exists() or destination.is_symlink():
                raise RecoveryError(f"active result target appeared during recovery: {destination}")
            os.rename(source, destination)
            committed.append((destination, source))
        if _tree_snapshot(quarantine_results) != source_snapshot:
            raise RecoveryError("quarantined results changed during recovery")
        _atomic_json(manifest_path, manifest)
        manifest_written = True
        staging.rmdir()
    except Exception:
        if manifest_written and manifest_path is not None:
            manifest_path.unlink(missing_ok=True)
        for destination, source in reversed(committed):
            if destination.exists() and not source.exists():
                os.rename(destination, source)
        shutil.rmtree(staging, ignore_errors=True)
        raise

    assert manifest_path is not None
    print(f"Recovery applied. Manifest: {manifest_path}")
    print("The quarantined source tree was not modified.")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--active-results", type=Path, required=True)
    parser.add_argument("--active-logs-root", type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    try:
        recover(
            audit_path=args.audit,
            active_results=args.active_results,
            active_logs_root=args.active_logs_root,
            apply=args.apply,
        )
    except RecoveryError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
