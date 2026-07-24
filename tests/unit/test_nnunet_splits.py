"""Focused tests for deterministic nnU-Net synthetic subset selection."""

from pathlib import Path

import numpy as np
import pytest

from medgen.downstream.nnunet.splits import (
    _load_synthetic_manifest,
    create_isolated_preprocessed_dir,
    generate_experiment_splits,
)


def _real_cases() -> list[str]:
    return [f"BrainMet_{index:03d}" for index in range(10)]


def _synthetic_cases(count: int = 210) -> list[str]:
    return [f"BrainMetSyn_{index:05d}" for index in range(count)]


def _selected_synthetic(split: dict[str, list[str]]) -> set[str]:
    return {case for case in split["train"] if case.startswith("BrainMetSyn_")}


def test_default_subset_preserves_legacy_seeded_choice() -> None:
    synthetic = _synthetic_cases(20)
    splits = generate_experiment_splits(
        "mixed",
        _real_cases(),
        synthetic,
        n_synthetic=7,
        synthetic_seed=42,
    )

    expected = set(
        np.random.default_rng(42).choice(synthetic, size=7, replace=False)
    )
    assert _selected_synthetic(splits[0]) == expected


def test_explicit_order_selects_prefix_before_sorting() -> None:
    synthetic = _synthetic_cases(8)
    order = [synthetic[index] for index in (7, 2, 5, 0, 6, 1, 4, 3)]
    splits = generate_experiment_splits(
        "mixed",
        _real_cases(),
        synthetic,
        n_synthetic=3,
        synthetic_order=order,
    )

    assert _selected_synthetic(splits[0]) == set(order[:3])


def test_exp17_counts_are_nested_manifest_prefixes() -> None:
    synthetic = _synthetic_cases()
    order = list(reversed(synthetic))
    selected = []
    for count in (25, 50, 105, 210):
        splits = generate_experiment_splits(
            "mixed",
            _real_cases(),
            synthetic,
            n_synthetic=count,
            synthetic_order=order,
        )
        chosen = _selected_synthetic(splits[0])
        assert chosen == set(order[:count])
        selected.append(chosen)

    assert selected[0] < selected[1] < selected[2] < selected[3]


def test_explicit_order_rejects_duplicates_and_unknown_cases() -> None:
    synthetic = _synthetic_cases(4)
    with pytest.raises(ValueError, match="duplicate"):
        generate_experiment_splits(
            "mixed",
            _real_cases(),
            synthetic,
            n_synthetic=2,
            synthetic_order=[synthetic[0], synthetic[0], *synthetic[2:]],
        )
    with pytest.raises(ValueError, match="complete permutation"):
        generate_experiment_splits(
            "mixed",
            _real_cases(),
            synthetic,
            n_synthetic=2,
            synthetic_order=[*synthetic[:-1], "BrainMetSyn_99999"],
        )


def test_explicit_order_rejects_oversized_request() -> None:
    synthetic = _synthetic_cases(4)
    with pytest.raises(ValueError, match="outside 0--4"):
        generate_experiment_splits(
            "mixed",
            _real_cases(),
            synthetic,
            n_synthetic=5,
            synthetic_order=synthetic,
        )


def test_manifest_loader_normalizes_raw_ids_and_preserves_order(
    tmp_path: Path,
) -> None:
    available = [
        "BrainMetSyn_00003",
        "BrainMetSyn_00001",
        "BrainMetSyn_00002",
    ]
    manifest = tmp_path / "synthetic_ids.txt"
    manifest.write_text("00002\nBrainMetSyn_00003\n00001\n")

    assert _load_synthetic_manifest(str(manifest), available) == [
        "BrainMetSyn_00002",
        "BrainMetSyn_00003",
        "BrainMetSyn_00001",
    ]


def test_manifest_loader_rejects_duplicate_and_unknown_rows(tmp_path: Path) -> None:
    available = _synthetic_cases(2)
    duplicate = tmp_path / "duplicate.txt"
    duplicate.write_text("00000\n00000\n")
    with pytest.raises(ValueError, match="duplicate"):
        _load_synthetic_manifest(str(duplicate), available)

    unknown = tmp_path / "unknown.txt"
    unknown.write_text("00000\n99999\n")
    with pytest.raises(ValueError, match="absent"):
        _load_synthetic_manifest(str(unknown), available)


def test_isolated_preprocessed_dir_rejects_stale_symlink(tmp_path: Path) -> None:
    shared_root = tmp_path / "nnUNet_preprocessed"
    shared_dataset = shared_root / "Dataset663_BrainMet"
    shared_dataset.mkdir(parents=True)
    source_plan = shared_dataset / "nnUNetPlans.json"
    source_plan.write_text("current")

    wrong_plan = tmp_path / "wrong_plan.json"
    wrong_plan.write_text("stale")
    isolated_dataset = Path(
        f"{shared_root}_exp17_1_hybrid_25syn_exp1_to_exp48c_t025_d663"
    ) / shared_dataset.name
    isolated_dataset.mkdir(parents=True)
    (isolated_dataset / source_plan.name).symlink_to(wrong_plan)

    with pytest.raises(RuntimeError, match="points to stale data"):
        create_isolated_preprocessed_dir(
            experiment_name=(
                "exp17_1_hybrid_25syn_exp1_to_exp48c_t025_d663"
            ),
            splits=[{"train": ["BrainMet_001"], "val": ["BrainMet_002"]}],
            nnunet_preprocessed=str(shared_root),
            dataset_id=663,
        )


def test_isolated_preprocessed_dir_reuses_correct_symlink(tmp_path: Path) -> None:
    shared_root = tmp_path / "nnUNet_preprocessed"
    shared_dataset = shared_root / "Dataset663_BrainMet"
    shared_dataset.mkdir(parents=True)
    source_plan = shared_dataset / "nnUNetPlans.json"
    source_plan.write_text("current")
    splits = [{"train": ["BrainMet_001"], "val": ["BrainMet_002"]}]

    isolated_root = Path(
        create_isolated_preprocessed_dir(
            experiment_name="exp17_test",
            splits=splits,
            nnunet_preprocessed=str(shared_root),
            dataset_id=663,
        )
    )
    isolated_plan = isolated_root / shared_dataset.name / source_plan.name
    assert isolated_plan.is_symlink()
    assert isolated_plan.resolve() == source_plan.resolve()

    repeated_root = create_isolated_preprocessed_dir(
        experiment_name="exp17_test",
        splits=splits,
        nnunet_preprocessed=str(shared_root),
        dataset_id=663,
    )
    assert repeated_root == str(isolated_root)
    assert isolated_plan.resolve() == source_plan.resolve()
