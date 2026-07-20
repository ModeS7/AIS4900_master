"""Apply brain-mask zeroing to existing generated bravo volumes.

Applies the same mask construction and zeroing used by newer generation runs
inline (`generate.py` Stage 3, `mask_outside_brain=true`) to already-saved
volumes. Posthoc inputs have already been trimmed to their saved depth, so
this is Stage-3-equivalent processing rather than a claim of byte-identical
replay of the pre-trim generation tensor.

Operations per case (`<src>/<id>/bravo.nii.gz`):
  1. Load bravo NIfTI.
  2. Compute brain mask via `create_brain_mask` (same threshold/dilate as
     `generate.py`).
  3. Zero outside the mask.
  4. Either save to `<dst>/<id>/bravo.nii.gz`, or atomically replace the
     source BRAVO when `--in-place` is selected.
  5. In copy mode only, copy `<src>/<id>/seg.nii.gz` to the destination
     unchanged. In-place mode never writes the segmentation or `bins.csv`.

Usage:
    python -m medgen.scripts.brain_mask_existing \\
        --src /cluster/work/$USER/MedicalDataSets/generated/exp1_1_bravo_radimagenet_525 \\
        --dst /cluster/work/$USER/MedicalDataSets/generated/exp1_1_bravo_radimagenet_525_brain_masked \\
        --threshold 0.05 --dilate-pixels 2

    python -m medgen.scripts.brain_mask_existing \\
        --src /cluster/work/$USER/MedicalDataSets/evalModels/my_panel/exp47a \\
        --in-place --threshold 0.05 --dilate-pixels 2
"""

from __future__ import annotations

import argparse
import errno
import logging
import os
import shutil
import stat
import sys
import uuid
from pathlib import Path

import nibabel as nib
import numpy as np

from medgen.metrics.brain_mask import create_brain_mask

logger = logging.getLogger(__name__)


def _fsync_directory(path: Path) -> None:
    """Persist a completed rename when the filesystem supports directory fsync."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        try:
            os.fsync(descriptor)
        except OSError as exc:
            if exc.errno not in {errno.EINVAL, errno.ENOTSUP}:
                raise
            logger.warning("directory fsync is not supported for %s", path)
    finally:
        os.close(descriptor)


def _atomic_replace_bravo(
    bravo_path: Path,
    source_image: nib.spatialimages.SpatialImage,
    masked: np.ndarray,
) -> None:
    """Save, verify, and atomically replace one regular BRAVO file."""
    if bravo_path.is_symlink():
        raise RuntimeError(f"refusing to replace symlinked BRAVO: {bravo_path}")
    source_stat = bravo_path.stat()
    temporary = bravo_path.with_name(
        f".{bravo_path.name}.masking-{os.getpid()}-{uuid.uuid4().hex}.nii.gz"
    )
    try:
        output_image = nib.Nifti1Image(
            masked,
            source_image.affine,
            source_image.header.copy(),
        )
        nib.save(output_image, str(temporary))
        os.chmod(temporary, stat.S_IMODE(source_stat.st_mode))
        if temporary.stat().st_gid != source_stat.st_gid:
            os.chown(temporary, -1, source_stat.st_gid)

        verified_image = nib.load(str(temporary))
        verified = np.asarray(verified_image.dataobj, dtype=np.float32)
        if verified.shape != masked.shape:
            raise RuntimeError(f"temporary BRAVO shape changed: {verified.shape} != {masked.shape}")
        if not np.isfinite(verified).all():
            raise RuntimeError(f"temporary BRAVO contains non-finite values: {temporary}")
        if not np.allclose(verified_image.affine, source_image.affine, rtol=0.0, atol=1e-6):
            raise RuntimeError(f"temporary BRAVO affine changed: {temporary}")
        if not np.array_equal(verified, masked):
            raise RuntimeError(f"temporary BRAVO data failed verification: {temporary}")

        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, bravo_path)
        _fsync_directory(bravo_path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def mask_one_case(
    src_dir: Path,
    dst_dir: Path,
    threshold: float,
    dilate_pixels: int,
    overwrite: bool,
    in_place: bool = False,
    expected_shape: tuple[int, int, int] | None = None,
) -> bool:
    """Process one case directory. Returns True on success (or skip), False on error."""
    bravo_src = src_dir / "bravo.nii.gz"
    seg_src = src_dir / "seg.nii.gz"
    if not bravo_src.is_file() or not seg_src.is_file():
        if in_place:
            raise FileNotFoundError(f"missing bravo or seg in {src_dir}")
        logger.warning("skip %s — missing bravo or seg", src_dir.name)
        return True  # not an error, just an incomplete case

    if in_place and bravo_src.is_symlink():
        raise RuntimeError(f"refusing to replace symlinked BRAVO: {bravo_src}")

    if not in_place:
        dst_dir.mkdir(parents=True, exist_ok=True)
    bravo_dst = dst_dir / "bravo.nii.gz"
    seg_dst = dst_dir / "seg.nii.gz"

    if not in_place and bravo_dst.is_file() and seg_dst.is_file() and not overwrite:
        return True  # already done

    img = nib.load(str(bravo_src))
    vol = img.get_fdata().astype(np.float32)
    if vol.ndim != 3 or not np.isfinite(vol).all():
        raise RuntimeError(f"invalid source BRAVO: {bravo_src}")
    if expected_shape is not None and vol.shape != expected_shape:
        raise RuntimeError(
            f"unexpected source BRAVO shape {vol.shape}, expected {expected_shape}: {bravo_src}"
        )
    mask = create_brain_mask(vol, threshold=threshold, fill_holes=True, dilate_pixels=dilate_pixels)
    if not np.any(mask):
        raise RuntimeError(f"empty brain support: {bravo_src}")
    masked = vol * mask.astype(vol.dtype)

    if in_place:
        # Reruns after interruption are safe: an already masked volume is a
        # fixed point, and os.replace breaks any source hard link atomically.
        if not np.array_equal(masked, vol):
            _atomic_replace_bravo(bravo_src, img, masked)
        return True

    nib.save(nib.Nifti1Image(masked, img.affine, img.header), str(bravo_dst))

    if not seg_dst.is_file() or overwrite:
        shutil.copy2(str(seg_src), str(seg_dst))
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--src", required=True, help="Source dir containing <id>/bravo.nii.gz subdirs."
    )
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--dst", help="Destination dir; created if missing.")
    destination.add_argument(
        "--in-place",
        action="store_true",
        help="Atomically replace source BRAVOs; never modifies seg or bins.csv.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Brain-mask intensity threshold (default 0.05).",
    )
    parser.add_argument(
        "--dilate-pixels", type=int, default=2, help="Brain-mask dilation iterations (default 2)."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing files in --dst."
    )
    parser.add_argument("--limit", type=int, default=None, help="Process at most N cases (debug).")
    parser.add_argument(
        "--expected-shape",
        type=int,
        nargs=3,
        metavar=("X", "Y", "Z"),
        default=None,
        help="Require every source BRAVO to have this exact 3D shape.",
    )
    args = parser.parse_args(argv)

    if args.in_place and args.overwrite:
        parser.error("--overwrite cannot be combined with --in-place")
    if args.expected_shape is not None and any(value <= 0 for value in args.expected_shape):
        parser.error("--expected-shape values must be positive")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    src = Path(args.src)
    dst = src if args.in_place else Path(args.dst)
    if not src.is_dir():
        logger.error("source not found: %s", src)
        return 2

    case_dirs = sorted(p for p in src.iterdir() if p.is_dir() and p.name[0].isdigit())
    if args.limit:
        case_dirs = case_dirs[: args.limit]

    logger.info("Brain-masking %d cases", len(case_dirs))
    logger.info("  src: %s", src)
    logger.info("  dst: %s", dst)
    logger.info(
        "  threshold=%.3f  dilate_pixels=%d  overwrite=%s",
        args.threshold,
        args.dilate_pixels,
        args.overwrite,
    )

    n_done = 0
    n_failed = 0
    for cd in case_dirs:
        try:
            ok = mask_one_case(
                cd,
                dst / cd.name,
                args.threshold,
                args.dilate_pixels,
                args.overwrite,
                args.in_place,
                tuple(args.expected_shape) if args.expected_shape else None,
            )
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
    if not args.in_place and bins_src.is_file():
        shutil.copy2(str(bins_src), str(dst / "bins.csv"))
        logger.info("Copied bins.csv")

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
