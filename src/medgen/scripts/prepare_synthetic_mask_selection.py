"""Create one deterministic 105/420 split of a 525-mask pool."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

POOL_SIZE = 525
SCREENING_SIZE = 105
SEED = 42


def _require_mask_pool(pool_root: Path) -> list[Path]:
    expected_names = [f"{index:05d}" for index in range(POOL_SIZE)]
    observed_names = sorted(
        path.name for path in pool_root.iterdir() if path.is_dir() and path.name.isdigit()
    )
    if observed_names != expected_names:
        raise ValueError("Mask pool must contain exactly 00000 through 00524")

    masks = [pool_root / name / "seg.nii.gz" for name in expected_names]
    missing = [str(path) for path in masks if not path.is_file()]
    if missing:
        raise ValueError(f"Missing segmentation masks: {missing[:3]}")
    return masks


def prepare_mask_selection(
    pool_root: Path,
    output_root: Path,
    *,
    seed: int = SEED,
) -> dict[str, int | list[int]]:
    """Randomly order the pool once and expose the first 105 and remaining 420 masks."""
    pool_root = pool_root.resolve()
    if output_root.exists():
        raise ValueError(f"Output already exists: {output_root}")

    masks = _require_mask_pool(pool_root)
    order = np.random.default_rng(seed).permutation(POOL_SIZE).astype(int).tolist()
    selection: dict[str, int | list[int]] = {
        "seed": seed,
        "screening_source_indices": order[:SCREENING_SIZE],
        "extension_source_indices": order[SCREENING_SIZE:],
    }

    output_root.mkdir(parents=True)
    try:
        for rank, source_index in enumerate(order):
            if rank < SCREENING_SIZE:
                destination = output_root / "screening105" / f"{rank:05d}"
            else:
                destination = output_root / "extension420" / f"{rank - SCREENING_SIZE:05d}"
            destination.mkdir(parents=True)
            (destination / "seg.nii.gz").symlink_to(masks[source_index])

        with (output_root / "selection.json").open("w", encoding="utf-8") as handle:
            json.dump(selection, handle, indent=2)
            handle.write("\n")
        return selection
    except Exception:
        shutil.rmtree(output_root, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    selection = prepare_mask_selection(args.pool_root, args.output_root)
    print(
        f"Prepared {len(selection['screening_source_indices'])} screening masks and "
        f"{len(selection['extension_source_indices'])} extension masks in {args.output_root}"
    )


if __name__ == "__main__":
    main()
