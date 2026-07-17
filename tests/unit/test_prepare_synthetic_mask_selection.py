"""Tests for the deterministic synthetic-mask split."""

import json
from pathlib import Path

import numpy as np
import pytest

from medgen.scripts.prepare_synthetic_mask_selection import prepare_mask_selection


def _make_pool(root: Path) -> Path:
    for index in range(525):
        mask = root / f"{index:05d}" / "seg.nii.gz"
        mask.parent.mkdir(parents=True)
        mask.write_bytes(str(index).encode())
    return root


def test_seed42_split_uses_one_shared_random_order(tmp_path: Path):
    pool = _make_pool(tmp_path / "pool")
    output = tmp_path / "ordered"

    selection = prepare_mask_selection(pool, output)

    expected = np.random.default_rng(42).permutation(525).astype(int).tolist()
    assert selection["screening_source_indices"] == expected[:105]
    assert selection["extension_source_indices"] == expected[105:]
    assert len(list((output / "screening105").glob("*/seg.nii.gz"))) == 105
    assert len(list((output / "extension420").glob("*/seg.nii.gz"))) == 420
    assert (output / "screening105/00000/seg.nii.gz").resolve() == (
        pool / f"{expected[0]:05d}/seg.nii.gz"
    ).resolve()
    assert json.loads((output / "selection.json").read_text()) == selection


def test_existing_output_is_not_replaced(tmp_path: Path):
    pool = _make_pool(tmp_path / "pool")
    output = tmp_path / "ordered"
    output.mkdir()

    with pytest.raises(ValueError, match="already exists"):
        prepare_mask_selection(pool, output)
