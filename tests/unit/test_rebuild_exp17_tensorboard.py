import json
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
    return root


def test_rebuild_creates_twenty_verified_histories_without_touching_sources(
    complete_results: Path, tmp_path: Path
) -> None:
    source_snapshot = {
        path.relative_to(complete_results): path.read_bytes()
        for path in complete_results.rglob("*")
        if path.is_file()
    }
    output = tmp_path / "tensorboard_reconstructed"
    recovery = _write_recovery_manifest(tmp_path / "recovery.json", complete_results)

    rebuilt = rebuild(complete_results, output, recovery)

    assert rebuilt == output
    event_files = sorted(output.glob("*/fold_*/events.out.tfevents.*"))
    assert len(event_files) == 20
    manifest = json.loads((output / "reconstruction_manifest.json").read_text())
    assert len(manifest["folds"]) == 20
    assert manifest["source_trees_modified"] is False
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
    assert {
        path.relative_to(complete_results): path.read_bytes()
        for path in complete_results.rglob("*")
        if path.is_file()
    } == source_snapshot


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
    output = tmp_path / "tensorboard_reconstructed"
    recovery = _write_recovery_manifest(tmp_path / "recovery.json", complete_results)

    with pytest.raises(RebuildError, match="unexpected final-checkpoint layout"):
        rebuild(complete_results, output, recovery)

    assert not output.exists()


def test_rebuild_refuses_to_overwrite_existing_output(
    complete_results: Path, tmp_path: Path
) -> None:
    output = tmp_path / "tensorboard_reconstructed"
    output.mkdir()
    recovery = _write_recovery_manifest(tmp_path / "recovery.json", complete_results)

    with pytest.raises(RebuildError, match="refusing to overwrite"):
        rebuild(complete_results, output, recovery)


def test_rebuild_rejects_output_inside_source_tree(
    complete_results: Path, tmp_path: Path
) -> None:
    output = complete_results / "tensorboard_reconstructed"
    recovery = _write_recovery_manifest(tmp_path / "recovery.json", complete_results)

    with pytest.raises(RebuildError, match="source trees must be disjoint"):
        rebuild(complete_results, output, recovery)

    assert not output.exists()


def test_rebuild_detects_source_mutation_and_cleans_staging(
    complete_results: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "tensorboard_reconstructed"
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
        rebuild(complete_results, output, recovery)

    assert not output.exists()
    assert not list(tmp_path.glob(".tensorboard_reconstructed.tmp_*"))
