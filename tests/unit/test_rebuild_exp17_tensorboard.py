import json
import os
import signal
from pathlib import Path

import numpy as np
import pytest
import torch

import medgen.scripts.rebuild_exp17_tensorboard as rebuild_module
from medgen.scripts.rebuild_exp17_tensorboard import (
    EXPECTED_EPOCHS,
    EXPECTED_EXPERIMENTS,
    MODEL_NAME,
    RebuildError,
    rebuild,
)
from medgen.scripts.recover_exp17_checkpoints import EXPECTED_RECOVERY


def _logging(offset: float) -> dict:
    values = [offset + index / 10_000 for index in range(EXPECTED_EPOCHS)]
    mean_dice = [0.2 + index / 10_000 for index in range(EXPECTED_EPOCHS)]
    ema_dice = [mean_dice[0]]
    for value in mean_dice[1:]:
        ema_dice.append(0.9 * ema_dice[-1] + 0.1 * value)
    return {
        "train_losses": values,
        "lrs": [0.01 - index / 200_000 for index in range(EXPECTED_EPOCHS)],
        "val_losses": [value + 0.1 for value in values],
        "mean_fg_dice": mean_dice,
        "ema_fg_dice": ema_dice,
        "dice_per_class_or_region": [
            np.asarray([0.2 + index / 10_000]) for index in range(EXPECTED_EPOCHS)
        ],
        "epoch_start_timestamps": [1_700_000_000.0 + index for index in range(EXPECTED_EPOCHS)],
        "epoch_end_timestamps": [1_700_000_000.5 + index for index in range(EXPECTED_EPOCHS)],
    }


def _write_final(path: Path, *, fold: int, offset: float) -> None:
    logging = _logging(offset)
    checkpoint = {
        "network_weights": {"weight": torch.ones(1)},
        "optimizer_state": {
            "state": {0: {"step": torch.tensor(1.0)}},
            "param_groups": [{"params": [0]}],
        },
        "grad_scaler_state": {"scale": torch.tensor(1.0)},
        "logging": logging,
        "_best_ema": max(logging["ema_fg_dice"]),
        "current_epoch": EXPECTED_EPOCHS,
        "init_args": {
            "fold": fold,
            "configuration": "3d_fullres",
            "plans": {
                "dataset_name": "Dataset663_BrainMet",
                "plans_name": "nnUNetResEncUNetLPlansD600",
            },
        },
        "trainer_name": "nnUNetTrainerBrainMets",
        "inference_allowed_mirroring_axes": (0, 1, 2),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def _write_recovery_manifest(path: Path, results_root: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "active_results": str(results_root),
                "source_quarantine_preserved": True,
                "folds": [
                    {
                        "experiment": experiment,
                        "fold": fold,
                        "action": action_epoch[0],
                    }
                    for (experiment, fold), action_epoch in sorted(
                        EXPECTED_RECOVERY.items()
                    )
                ],
            }
        )
    )
    return path


@pytest.fixture
def complete_results(tmp_path: Path) -> Path:
    root = tmp_path / "results"
    for experiment_index, experiment in enumerate(EXPECTED_EXPERIMENTS):
        for fold in range(5):
            _write_final(
                root
                / experiment
                / "Dataset663_BrainMet"
                / MODEL_NAME
                / f"fold_{fold}"
                / "checkpoint_final.pth",
                fold=fold,
                offset=float(experiment_index + fold),
            )
            tensorboard = (
                root
                / experiment
                / "Dataset663_BrainMet"
                / MODEL_NAME
                / f"fold_{fold}"
                / "tensorboard"
            )
            tensorboard.mkdir()
            (tensorboard / f"events.out.tfevents.partial_{experiment_index}_{fold}").write_bytes(
                f"partial:{experiment_index}:{fold}".encode()
            )
    return root


def _checkpoint_snapshot(root: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("checkpoint_final.pth")
    }


def _partial_tensorboard_snapshot(root: Path) -> dict[Path, bytes]:
    return {
        path.relative_to(root): path.read_bytes()
        for path in root.glob("*/Dataset663_BrainMet/*/fold_*/tensorboard/*")
        if path.is_file()
    }


def test_rebuild_creates_twenty_verified_histories_without_touching_sources(
    complete_results: Path, tmp_path: Path
) -> None:
    checkpoint_snapshot = _checkpoint_snapshot(complete_results)
    partial_snapshot = _partial_tensorboard_snapshot(complete_results)
    archive = tmp_path / "tensorboard_archive"
    recovery = _write_recovery_manifest(tmp_path / "recovery.json", complete_results)

    rebuilt = rebuild(complete_results, archive, recovery)

    assert rebuilt == archive
    event_files = sorted(
        complete_results.glob("*/Dataset663_BrainMet/*/fold_*/tensorboard/events.out.tfevents.*")
    )
    assert len(event_files) == 20
    assert all(len(list(path.parent.iterdir())) == 1 for path in event_files)
    archived_partial = {
        path.relative_to(archive / "previous"): path.read_bytes()
        for path in (archive / "previous").glob(
            "*/fold_*/tensorboard/events.out.tfevents.partial_*"
        )
    }
    expected_partial = {
        Path(path.parts[0], path.parts[-3], "tensorboard", path.name): value
        for path, value in partial_snapshot.items()
    }
    assert archived_partial == expected_partial
    manifest = json.loads((archive / "reconstruction_manifest.json").read_text())
    assert len(manifest["folds"]) == 20
    assert manifest["checkpoints_modified"] is False
    assert all(
        set(record["scalar_counts"])
        == {
            "train/loss",
            "train/learning_rate",
            "val/loss",
            "val/mean_fg_dice",
            "val/ema_fg_dice",
            "val/dice_class_0",
        }
        for record in manifest["folds"]
    )
    assert _checkpoint_snapshot(complete_results) == checkpoint_snapshot


def test_rebuild_fails_before_writing_when_a_final_checkpoint_is_missing(
    complete_results: Path, tmp_path: Path
) -> None:
    missing = (
        complete_results
        / EXPECTED_EXPERIMENTS[-1]
        / "Dataset663_BrainMet"
        / MODEL_NAME
        / "fold_4"
        / "checkpoint_final.pth"
    )
    missing.unlink()
    archive = tmp_path / "tensorboard_archive"
    recovery = _write_recovery_manifest(tmp_path / "recovery.json", complete_results)

    with pytest.raises(RebuildError, match="unexpected final-checkpoint layout"):
        rebuild(complete_results, archive, recovery)

    assert not archive.exists()


def test_rebuild_refuses_to_overwrite_existing_output(
    complete_results: Path, tmp_path: Path
) -> None:
    archive = tmp_path / "tensorboard_archive"
    archive.mkdir()
    recovery = _write_recovery_manifest(tmp_path / "recovery.json", complete_results)

    with pytest.raises(RebuildError, match="refusing to overwrite existing archive"):
        rebuild(complete_results, archive, recovery)


def test_rebuild_rejects_output_inside_source_tree(
    complete_results: Path, tmp_path: Path
) -> None:
    archive = complete_results / "tensorboard_archive"
    recovery = _write_recovery_manifest(tmp_path / "recovery.json", complete_results)

    with pytest.raises(RebuildError, match="archive and source trees must be disjoint"):
        rebuild(complete_results, archive, recovery)

    assert not archive.exists()


def test_rebuild_detects_source_mutation_and_cleans_staging(
    complete_results: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "tensorboard_archive"
    recovery = _write_recovery_manifest(tmp_path / "recovery.json", complete_results)
    source = (
        complete_results
        / EXPECTED_EXPERIMENTS[0]
        / "Dataset663_BrainMet"
        / MODEL_NAME
        / "fold_0"
        / "checkpoint_final.pth"
    )
    original_write = rebuild_module._write_history
    mutated = False

    def write_then_mutate(destination: Path, history) -> None:
        nonlocal mutated
        original_write(destination, history)
        if not mutated:
            source.write_bytes(source.read_bytes() + b"changed")
            mutated = True

    monkeypatch.setattr(rebuild_module, "_write_history", write_then_mutate)

    with pytest.raises(RebuildError, match="source checkpoint changed during rebuild"):
        rebuild(complete_results, archive, recovery)

    assert not archive.exists()
    assert not list(tmp_path.glob(".tensorboard_archive.tmp_*"))


def test_rebuild_restores_all_partial_directories_after_install_failure(
    complete_results: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_snapshot = _checkpoint_snapshot(complete_results)
    partial_snapshot = _partial_tensorboard_snapshot(complete_results)
    archive = tmp_path / "tensorboard_archive"
    recovery = _write_recovery_manifest(tmp_path / "recovery.json", complete_results)
    original_publish = rebuild_module._publish_no_replace
    normal_install_count = 0

    def fail_during_second_normal_install(source: Path, destination: Path) -> None:
        nonlocal normal_install_count
        if destination.name == "tensorboard" and destination.is_relative_to(
            complete_results
        ):
            normal_install_count += 1
            if normal_install_count == 2:
                raise RebuildError("injected installation failure")
        original_publish(source, destination)

    monkeypatch.setattr(
        rebuild_module, "_publish_no_replace", fail_during_second_normal_install
    )

    with pytest.raises(RebuildError, match="injected installation failure"):
        rebuild(complete_results, archive, recovery)

    assert _checkpoint_snapshot(complete_results) == checkpoint_snapshot
    assert _partial_tensorboard_snapshot(complete_results) == partial_snapshot
    assert not archive.exists()
    assert not list(tmp_path.glob(".tensorboard_archive.tmp_*"))


def test_install_signal_is_deferred_and_replayed_after_transaction() -> None:
    received: list[int] = []
    original_handler = signal.getsignal(signal.SIGTERM)

    def record(signal_number: int, _frame) -> None:
        received.append(signal_number)

    signal.signal(signal.SIGTERM, record)
    try:
        with rebuild_module._defer_install_signals():
            os.kill(os.getpid(), signal.SIGTERM)
            assert received == []
        assert received == [signal.SIGTERM]
    finally:
        signal.signal(signal.SIGTERM, original_handler)
