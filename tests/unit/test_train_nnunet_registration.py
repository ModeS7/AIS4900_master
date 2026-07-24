"""Concurrency contract for nnU-Net trainer registration."""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType

from medgen.scripts.train_nnunet import _register_trainer


def test_register_trainer_is_safe_when_array_tasks_start_together(
    tmp_path: Path,
    monkeypatch,
) -> None:
    package = ModuleType("nnunetv2")
    package.__path__ = []
    training = ModuleType("nnunetv2.training")
    training.__path__ = []
    trainers = ModuleType("nnunetv2.training.nnUNetTrainer")
    trainers.__file__ = str(tmp_path / "__init__.py")

    monkeypatch.setitem(sys.modules, "nnunetv2", package)
    monkeypatch.setitem(sys.modules, "nnunetv2.training", training)
    monkeypatch.setitem(
        sys.modules,
        "nnunetv2.training.nnUNetTrainer",
        trainers,
    )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: _register_trainer(), range(16)))

    target = tmp_path / "nnUNetTrainerTensorBoard.py"
    expected = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "medgen"
        / "downstream"
        / "nnunet"
        / "trainer.py"
    )
    assert target.is_symlink()
    assert target.resolve(strict=True) == expected.resolve(strict=True)
