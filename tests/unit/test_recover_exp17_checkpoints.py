import json
import shutil
from pathlib import Path

import pytest

from medgen.scripts.recover_exp17_checkpoints import (
    EXPECTED_EXPERIMENTS,
    EXPECTED_RECOVERY,
    MODEL_NAME,
    RecoveryError,
    recover,
)

ROOT = Path(__file__).resolve().parents[2]

ACTION_MAP = {
    "exp17_1_hybrid_25syn_exp1_to_exp48c_t025_d663": {
        0: "KEEP_COMPLETE",
        1: "KEEP_COMPLETE",
        2: "KEEP_COMPLETE",
        3: "RESUME_BEST",
        4: "KEEP_COMPLETE",
    },
    "exp17_2_hybrid_50syn_exp1_to_exp48c_t025_d663": {
        0: "RESUME_BEST",
        1: "RESUME_LATEST",
        2: "RESUME_LATEST",
        3: "RESUME_LATEST",
        4: "KEEP_COMPLETE",
    },
    "exp17_3_hybrid_105syn_exp1_to_exp48c_t025_d663": {
        0: "RESUME_LATEST",
        1: "RESUME_LATEST",
        2: "RESUME_LATEST",
        3: "RESUME_BEST",
        4: "RESUME_BEST",
    },
    "exp17_4_hybrid_210syn_exp1_to_exp48c_t025_d663": {
        0: "RESUME_BEST",
        1: "RESUME_LATEST",
        2: "RESUME_BEST",
        3: "RESUME_LATEST",
        4: "RESTART",
    },
}


def _checkpoint_record(path: Path, *, valid: bool, epoch: int | None) -> dict:
    if valid:
        path.write_bytes(f"checkpoint:{path.parent.name}:{path.name}:{epoch}".encode())
        stat = path.stat()
        return {
            "path": str(path),
            "exists": True,
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "valid": True,
            "epoch": epoch,
            "network_signature": "shared-network-signature",
            "network_tensor_count": 100,
            "network_parameter_count": 1000,
            "optimizer_state_entries": 100,
            "optimizer_param_groups": 1,
        }
    return {
        "path": str(path),
        "exists": False,
        "size_bytes": None,
        "mtime_ns": None,
        "valid": False,
        "epoch": None,
        "network_signature": None,
        "network_tensor_count": None,
        "network_parameter_count": None,
        "optimizer_state_entries": None,
        "optimizer_param_groups": None,
    }


def _make_audit(tmp_path: Path) -> tuple[Path, Path, Path]:
    quarantine = tmp_path / "quarantine"
    results = quarantine / "results"
    shadows = quarantine / "preprocessed_shadows"
    shadows.mkdir(parents=True)
    folds = []

    for experiment in sorted(EXPECTED_EXPERIMENTS):
        model = results / experiment / "Dataset663_BrainMet" / MODEL_NAME
        model.mkdir(parents=True)
        (model / "dataset.json").write_text("{}")
        (model / "plans.json").write_text("{}")
        (model / "dataset_fingerprint.json").write_text("{}")
        for fold_number in range(5):
            action = ACTION_MAP[experiment][fold_number]
            fold_dir = model / f"fold_{fold_number}"
            fold_dir.mkdir()
            (fold_dir / "progress.png").write_bytes(b"mixed progress")
            (fold_dir / "training_log_mixed.txt").write_text("mixed log")

            best_valid = action != "RESTART"
            latest_valid = action == "RESUME_LATEST" or (
                action == "RESUME_BEST" and fold_number % 2 == 0
            )
            final_valid = action == "KEEP_COMPLETE"
            expected_action, expected_epoch = EXPECTED_RECOVERY[(experiment, fold_number)]
            assert action == expected_action
            best_epoch = expected_epoch if action == "RESUME_BEST" else 25 + fold_number
            if not best_valid:
                best_epoch = None
            latest_epoch = expected_epoch if action == "RESUME_LATEST" else 50
            checkpoints = {
                "best": _checkpoint_record(
                    fold_dir / "checkpoint_best.pth",
                    valid=best_valid,
                    epoch=best_epoch,
                ),
                "latest": _checkpoint_record(
                    fold_dir / "checkpoint_latest.pth",
                    valid=latest_valid,
                    epoch=latest_epoch if latest_valid else None,
                ),
                "final": _checkpoint_record(
                    fold_dir / "checkpoint_final.pth",
                    valid=final_valid,
                    epoch=1000 if final_valid else None,
                ),
            }
            folds.append(
                {
                    "experiment": experiment,
                    "fold": fold_number,
                    "fold_dir": str(fold_dir),
                    "split_valid": True,
                    "best_matches_anchor": (
                        True
                        if action in {"KEEP_COMPLETE", "RESUME_LATEST"}
                        else (False if latest_valid else None)
                    ),
                    "action": action,
                    "checkpoints": checkpoints,
                }
            )

    audit = quarantine / "checkpoint_audit_24940260.json"
    audit.write_text(
        json.dumps(
            {
                "results_root": str(results),
                "shadows_root": str(shadows),
                "folds": folds,
            }
        )
    )
    return audit, results, tmp_path / "active_results"


def _source_snapshot(root: Path) -> dict[str, bytes | None]:
    return {
        str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
        for path in sorted(root.rglob("*"))
    }


def test_recovery_preserves_quarantine_and_builds_clean_active_tree(tmp_path):
    audit, quarantine_results, active = _make_audit(tmp_path)
    source_before = _source_snapshot(quarantine_results)

    assert (
        recover(
            audit_path=audit,
            active_results=active,
            apply=False,
            check_jobs=False,
        )
        is None
    )
    assert not active.exists()

    manifest_path = recover(
        audit_path=audit,
        active_results=active,
        apply=True,
        check_jobs=False,
    )
    assert manifest_path is not None and manifest_path.is_file()
    assert manifest_path.parent.parent == active
    assert _source_snapshot(quarantine_results) == source_before

    for experiment in sorted(EXPECTED_EXPERIMENTS):
        source_model = quarantine_results / experiment / "Dataset663_BrainMet" / MODEL_NAME
        active_model = active / experiment / "Dataset663_BrainMet" / MODEL_NAME
        assert (active_model / "dataset.json").is_file()
        assert (active_model / "plans.json").is_file()
        for fold_number, action in ACTION_MAP[experiment].items():
            source_fold = source_model / f"fold_{fold_number}"
            active_fold = active_model / f"fold_{fold_number}"
            assert (source_fold / "progress.png").is_file()
            assert (source_fold / "training_log_mixed.txt").is_file()
            if action == "RESTART":
                assert not active_fold.exists()
                continue
            assert active_fold.is_dir()
            assert not (active_fold / "progress.png").exists()
            assert not (active_fold / "training_log_mixed.txt").exists()
            assert (active_fold / "checkpoint_best.pth").is_file()
            if action == "KEEP_COMPLETE":
                assert (active_fold / "checkpoint_final.pth").is_file()
                assert not (active_fold / "checkpoint_latest.pth").exists()
            elif action == "RESUME_LATEST":
                assert (active_fold / "checkpoint_latest.pth").read_bytes() == (
                    source_fold / "checkpoint_latest.pth"
                ).read_bytes()
                assert not (active_fold / "checkpoint_final.pth").exists()
            else:
                best = active_fold / "checkpoint_best.pth"
                latest = active_fold / "checkpoint_latest.pth"
                assert latest.read_bytes() == best.read_bytes()
                assert (latest.stat().st_dev, latest.stat().st_ino) != (
                    best.stat().st_dev,
                    best.stat().st_ino,
                )
                assert not (active_fold / "checkpoint_final.pth").exists()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["source_quarantine_preserved"] is True
    assert len(manifest["folds"]) == 20


def test_recovery_rolls_back_an_injected_copy_failure(tmp_path):
    audit, quarantine_results, active = _make_audit(tmp_path)
    source_before = _source_snapshot(quarantine_results)
    copy_count = 0

    def fail_during_copy(source: Path, destination: Path) -> None:
        nonlocal copy_count
        copy_count += 1
        if copy_count == 7:
            raise RuntimeError("injected copy failure")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    with pytest.raises(RuntimeError, match="injected copy failure"):
        recover(
            audit_path=audit,
            active_results=active,
            apply=True,
            check_jobs=False,
            copier=fail_during_copy,
        )

    assert _source_snapshot(quarantine_results) == source_before
    assert not any(active.glob("exp17_*"))
    assert not any(active.glob(".exp17_recovery_*"))


def test_recovery_refuses_existing_results_and_nonempty_logs(tmp_path):
    audit, _, active = _make_audit(tmp_path)
    experiment = sorted(EXPECTED_EXPERIMENTS)[0]
    (active / experiment).mkdir(parents=True)
    with pytest.raises(RecoveryError, match="active result target already exists"):
        recover(
            audit_path=audit,
            active_results=active,
            apply=False,
            check_jobs=False,
        )

    shutil.rmtree(active)
    logs = tmp_path / "logs"
    log_directory = logs / experiment
    log_directory.mkdir(parents=True)
    (log_directory / "stale.out").write_text("old job")
    with pytest.raises(RecoveryError, match="active log directory is not empty"):
        recover(
            audit_path=audit,
            active_results=active,
            active_logs_root=logs,
            apply=False,
            check_jobs=False,
        )


def test_recovery_slurm_defaults_to_dry_run_for_exact_audit():
    content = (
        ROOT / "IDUN/train/downstream/nnunet/recover_exp17_quarantined_checkpoints.slurm"
    ).read_text()
    assert "checkpoint_audit_24940260.json" in content
    assert 'readonly APPLY="${APPLY:-0}"' in content
    assert 'if [[ "$APPLY" == 1 ]]' in content
    assert "--active-logs-root" in content
