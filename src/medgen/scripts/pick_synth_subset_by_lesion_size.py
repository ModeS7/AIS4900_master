"""Pick N synth volumes by size-bucket priority and symlink them into a subset dir.

Reads a manifest produced by `classify_synth_by_lesion_size.py` and selects
N synthetic volumes, drawing from the buckets in this priority order:

    tiny → small → medium → large

(Volumes whose largest lesion is tiny are filled first; if the tiny bucket
runs out, top up from small; then medium; then large.)

Within a single bucket, volumes are sampled randomly with a per-experiment
seed so different N values produce **independent** subsets (i.e. exp8_1's
25 vols are not nested in exp8_2's 50 vols — each picks fresh).

Output: a directory of symlinks `<output_dir>/<sample_id>` → `<pool>/<sample_id>`,
ready to feed into `medgen.downstream.nnunet.convert_dataset --synthetic-dir`.

Usage:

    python -m medgen.scripts.pick_synth_subset_by_lesion_size \\
        --manifest manifest.json \\
        --n 25 \\
        --seed 1 \\
        --output-dir /tmp/exp8_1_subset_25
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import random

PRIORITY_ORDER: tuple[str, ...] = ('tiny', 'small', 'medium', 'large')

logger = logging.getLogger(__name__)


def pick_subset(
    manifest: dict,
    n: int,
    seed: int,
    priority_order: tuple[str, ...] = PRIORITY_ORDER,
) -> list[str]:
    """Return a list of n sample IDs picked by bucket priority.

    Sampling within a bucket is deterministic given the seed. Empty-seg
    volumes (max_bucket='empty') are excluded entirely.
    """
    by_bucket: dict[str, list[str]] = {b: [] for b in priority_order}
    for sid, info in manifest['volumes'].items():
        bucket = info['max_bucket']
        if bucket in by_bucket:
            by_bucket[bucket].append(sid)

    rng = random.Random(seed)
    picked: list[str] = []
    used_per_bucket: dict[str, int] = {}

    for bucket in priority_order:
        if len(picked) >= n:
            break
        remaining = n - len(picked)
        ids = sorted(by_bucket[bucket])  # deterministic input order
        rng.shuffle(ids)
        take = ids[:remaining]
        picked.extend(take)
        used_per_bucket[bucket] = len(take)

    if len(picked) < n:
        raise RuntimeError(
            f"Synth pool too small: only {len(picked)}/{n} samples available "
            f"across buckets {priority_order} "
            f"(pool sizes: { {b: len(v) for b, v in by_bucket.items()} })"
        )

    logger.info(
        f"Picked {len(picked)} vols (seed={seed}) by priority "
        + ', '.join(f'{b}={used_per_bucket.get(b, 0)}' for b in priority_order)
    )
    return picked


def materialise_subset(
    pool_dir: str,
    sample_ids: list[str],
    output_dir: str,
    overwrite: bool = False,
) -> None:
    """Create a directory of symlinks `output_dir/<sid>` -> `pool_dir/<sid>`."""
    if os.path.exists(output_dir):
        if not overwrite:
            raise FileExistsError(
                f"Output dir already exists: {output_dir} (use --overwrite)"
            )
        for entry in os.listdir(output_dir):
            full = os.path.join(output_dir, entry)
            if os.path.islink(full):
                os.unlink(full)
    os.makedirs(output_dir, exist_ok=True)
    for sid in sample_ids:
        src = os.path.abspath(os.path.join(pool_dir, sid))
        dst = os.path.join(output_dir, sid)
        if not os.path.isdir(src):
            raise FileNotFoundError(f"Source missing: {src}")
        with contextlib.suppress(FileExistsError):
            os.symlink(src, dst)
    logger.info(f"Materialised {len(sample_ids)} symlinks in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True,
                        help='Manifest JSON from classify_synth_by_lesion_size')
    parser.add_argument('--n', type=int, required=True,
                        help='Number of synth volumes to select')
    parser.add_argument('--seed', type=int, required=True,
                        help='Random seed for within-bucket sampling')
    parser.add_argument('--output-dir', required=True,
                        help='Directory to populate with symlinks to picked vols')
    parser.add_argument('--overwrite', action='store_true',
                        help='Remove existing symlinks in output-dir before writing')
    parser.add_argument('--save-selection',
                        help='Optional: also write the picked IDs to this JSON file')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    with open(args.manifest) as f:
        manifest = json.load(f)

    picked = pick_subset(manifest, n=args.n, seed=args.seed)
    materialise_subset(
        pool_dir=manifest['pool_dir'],
        sample_ids=picked,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )

    if args.save_selection:
        os.makedirs(
            os.path.dirname(os.path.abspath(args.save_selection)) or '.',
            exist_ok=True,
        )
        with open(args.save_selection, 'w') as f:
            json.dump({
                'n': args.n,
                'seed': args.seed,
                'pool_dir': manifest['pool_dir'],
                'picked': picked,
            }, f, indent=2)
        logger.info(f"Selection record saved to {args.save_selection}")


if __name__ == '__main__':
    main()
