#!/usr/bin/env python3
"""Recompute latent normalization stats (metadata.json) from existing cache.

Use this after fixing the _welford_stats bug to update metadata.json
without re-encoding latents. Only recomputes latent_shift/latent_scale
(and latent_seg_shift/latent_seg_scale if present).

Usage:
    # Recompute for a specific cache directory
    python -m medgen.scripts.recompute_latent_stats \
        /path/to/brainmetshare-3-latents-vqvae-3d-abc123/train

    # Auto-discover all caches under a data root
    python -m medgen.scripts.recompute_latent_stats \
        --data-root /path/to/MedicalDataSets/brainmetshare-3

    # Dry run (show stats without writing)
    python -m medgen.scripts.recompute_latent_stats \
        /path/to/cache/train --dry-run
"""
import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(name)s][%(levelname)s] - %(message)s',
)
logger = logging.getLogger(__name__)


def recompute_stats(cache_dir: str, dry_run: bool = False) -> dict | None:
    """Recompute normalization stats for a single cache directory.

    Args:
        cache_dir: Path to cache directory containing .pt files and metadata.json.
        dry_run: If True, only print stats without writing.

    Returns:
        Updated stats dict, or None if no .pt files found.
    """
    from medgen.data.loaders.latent import LatentCacheBuilder

    metadata_path = Path(cache_dir) / 'metadata.json'
    if not metadata_path.exists():
        logger.warning(f"No metadata.json in {cache_dir}, skipping")
        return None

    with open(metadata_path) as f:
        metadata = json.load(f)

    # Show old stats
    old_shift = metadata.get('latent_shift')
    old_scale = metadata.get('latent_scale')
    if old_shift is not None:
        logger.info(f"  OLD shift: {old_shift}")
        logger.info(f"  OLD scale: {old_scale}")

    # Recompute
    stats = LatentCacheBuilder.compute_channel_stats(cache_dir)
    if not stats:
        logger.warning(f"  No .pt files found in {cache_dir}")
        return None

    logger.info(f"  NEW shift: {stats['latent_shift']}")
    logger.info(f"  NEW scale: {stats['latent_scale']}")

    if 'latent_seg_shift' in stats:
        old_seg_shift = metadata.get('latent_seg_shift')
        old_seg_scale = metadata.get('latent_seg_scale')
        if old_seg_shift is not None:
            logger.info(f"  OLD seg_shift: {old_seg_shift}")
            logger.info(f"  OLD seg_scale: {old_seg_scale}")
        logger.info(f"  NEW seg_shift: {stats['latent_seg_shift']}")
        logger.info(f"  NEW seg_scale: {stats['latent_seg_scale']}")

    if dry_run:
        logger.info("  [DRY RUN] Not writing to metadata.json")
        return stats

    # Update metadata (preserve all other fields)
    metadata.update(stats)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"  Updated {metadata_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Recompute latent normalization stats from existing cache",
    )
    parser.add_argument('cache_dirs', nargs='*',
                        help='Cache directories containing .pt files and metadata.json')
    parser.add_argument('--data-root', default=None,
                        help='Auto-discover latent caches under this data root')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show stats without writing')
    args = parser.parse_args()

    cache_dirs = list(args.cache_dirs)

    # Auto-discover caches
    if args.data_root:
        root = Path(args.data_root)
        # Look for */train/metadata.json patterns
        for meta in sorted(root.parent.glob(f"{root.name}-latents-*/train/metadata.json")):
            cache_dirs.append(str(meta.parent))

    if not cache_dirs:
        parser.error("No cache directories specified. Use positional args or --data-root.")

    logger.info(f"Found {len(cache_dirs)} cache(s) to process")

    for cache_dir in cache_dirs:
        logger.info(f"\nProcessing: {cache_dir}")
        recompute_stats(cache_dir, dry_run=args.dry_run)

    logger.info("\nDone.")


if __name__ == '__main__':
    main()
