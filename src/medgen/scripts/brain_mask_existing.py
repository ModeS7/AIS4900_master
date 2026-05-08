"""Apply brain-mask zeroing to existing generated bravo volumes.

Replays the same final post-processing step that newer generation runs do
inline (`generate.py` Stage 3, `mask_outside_brain=true`), but on volumes
that were generated *before* that step existed. This lets us bring older
synth datasets into the same nnU-Net plan family (Group A) as the brain-
masked candidates without re-running diffusion sampling.

Operations per case (`<src>/<id>/bravo.nii.gz`):
  1. Load bravo NIfTI.
  2. Compute brain mask via `create_brain_mask` (same threshold/dilate as
     `generate.py`).
  3. Zero outside the mask.
  4. Save to `<dst>/<id>/bravo.nii.gz`.
  5. Copy `<src>/<id>/seg.nii.gz` to `<dst>/<id>/seg.nii.gz` unchanged.

Usage:
    python -m medgen.scripts.brain_mask_existing \\
        --src /cluster/work/$USER/MedicalDataSets/generated/exp1_1_bravo_radimagenet_525 \\
        --dst /cluster/work/$USER/MedicalDataSets/generated/exp1_1_bravo_radimagenet_525_brain_masked \\
        --threshold 0.05 --dilate-pixels 2
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from medgen.metrics.brain_mask import create_brain_mask

logger = logging.getLogger(__name__)


def mask_one_case(
    src_dir: Path,
    dst_dir: Path,
    threshold: float,
    dilate_pixels: int,
    overwrite: bool,
) -> bool:
    """Process one case directory. Returns True on success (or skip), False on error."""
    bravo_src = src_dir / "bravo.nii.gz"
    seg_src = src_dir / "seg.nii.gz"
    if not bravo_src.is_file() or not seg_src.is_file():
        logger.warning("skip %s — missing bravo or seg", src_dir.name)
        return True  # not an error, just an incomplete case

    dst_dir.mkdir(parents=True, exist_ok=True)
    bravo_dst = dst_dir / "bravo.nii.gz"
    seg_dst = dst_dir / "seg.nii.gz"

    if bravo_dst.is_file() and seg_dst.is_file() and not overwrite:
        return True  # already done

    img = nib.load(str(bravo_src))
    vol = img.get_fdata().astype(np.float32)
    mask = create_brain_mask(
        vol, threshold=threshold, fill_holes=True, dilate_pixels=dilate_pixels
    )
    masked = vol * mask.astype(vol.dtype)
    nib.save(nib.Nifti1Image(masked, img.affine, img.header), str(bravo_dst))

    if not seg_dst.is_file() or overwrite:
        shutil.copy2(str(seg_src), str(seg_dst))
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--src", required=True,
                        help="Source dir containing <id>/bravo.nii.gz subdirs.")
    parser.add_argument("--dst", required=True,
                        help="Destination dir; created if missing.")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="Brain-mask intensity threshold (default 0.05).")
    parser.add_argument("--dilate-pixels", type=int, default=2,
                        help="Brain-mask dilation iterations (default 2).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing files in --dst.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N cases (debug).")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.is_dir():
        logger.error("source not found: %s", src)
        return 2

    case_dirs = sorted(p for p in src.iterdir() if p.is_dir() and p.name[0].isdigit())
    if args.limit:
        case_dirs = case_dirs[: args.limit]

    logger.info("Brain-masking %d cases", len(case_dirs))
    logger.info("  src: %s", src)
    logger.info("  dst: %s", dst)
    logger.info("  threshold=%.3f  dilate_pixels=%d  overwrite=%s",
                args.threshold, args.dilate_pixels, args.overwrite)

    n_done = 0
    n_failed = 0
    for cd in case_dirs:
        try:
            ok = mask_one_case(cd, dst / cd.name, args.threshold,
                               args.dilate_pixels, args.overwrite)
            n_done += 1 if ok else 0
            n_failed += 0 if ok else 1
        except Exception as e:
            logger.exception("FAIL %s: %s", cd.name, e)
            n_failed += 1
        if n_done % 25 == 0 and n_done > 0:
            logger.info("  processed %d/%d", n_done, len(case_dirs))

    logger.info("Done. %d cases processed, %d failed.", n_done, n_failed)

    # Sanity check: copy bins.csv if present (preserves size-bin metadata).
    bins_src = src / "bins.csv"
    if bins_src.is_file():
        shutil.copy2(str(bins_src), str(dst / "bins.csv"))
        logger.info("Copied bins.csv")

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
