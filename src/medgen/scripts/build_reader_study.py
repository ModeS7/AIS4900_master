"""Build a blind reader-study set (real vs synthetic) for radiologist evaluation.

Randomly samples N real volumes and N synthetic volumes (bravo + seg only),
normalizes them consistently, gives each a random 1..2N case ID, and writes:

  <output-dir>/volumes/<id>/bravo.nii.gz   (normalized to [0, 1])
  <output-dir>/volumes/<id>/seg.nii.gz     (binary segmentation, unchanged)
  <output-dir>/answer_key.csv              (case_id, true_label, source)   ← DO NOT SHARE
  <output-dir>/radiologist_form.csv        (blank form for the reader)     ← SHARE THIS

By default both sides are brain-mask zeroed so background appearance can't
be used as a tell. Disable with --no-brain-mask if the source dirs are
already consistent.

Usage:
    python -m medgen.scripts.build_reader_study \\
        --real-root /path/to/brainmetshare-3 --real-split train \\
        --synth-dir /path/to/generated/seg_candidates_525/exp48c_handoff_exp32 \\
        --output-dir runs/reader_study/exp48c_handoff_exp32 \\
        --n-real 50 --n-synth 50 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import logging
import shutil
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)


def _find_subjects(root: Path) -> list[Path]:
    """Return all subdirs under root that have both bravo.nii.gz and seg.nii.gz."""
    if not root.is_dir():
        raise FileNotFoundError(root)
    out = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        if (sub / "bravo.nii.gz").is_file() and (sub / "seg.nii.gz").is_file():
            out.append(sub)
    return out


def _find_real_subjects_across_splits(
    root: Path, splits: list[str]
) -> list[Path]:
    """Collect subjects across multiple splits, dedup by subject directory name.

    Stanford BrainMetShare has 156 unique patients spread across splits with
    overlap (e.g. test_new ⊂ test1). Walking all splits and deduping by the
    leaf directory name (subject ID) gives the full 156-patient pool.
    """
    seen: dict[str, Path] = {}
    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            logger.warning("split dir missing: %s — skipping", split_dir)
            continue
        for subj in _find_subjects(split_dir):
            # First-come wins on duplicates; doesn't matter which path —
            # files for the same subject are bit-identical across splits.
            seen.setdefault(subj.name, subj)
    return sorted(seen.values(), key=lambda p: p.name)


def _normalize_bravo(vol: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]."""
    vmin, vmax = float(vol.min()), float(vol.max())
    if vmax > vmin:
        return ((vol - vmin) / (vmax - vmin)).astype(np.float32)
    return vol.astype(np.float32)


def _apply_brain_mask(
    vol: np.ndarray, threshold: float = 0.05, dilate_pixels: int = 2
) -> np.ndarray:
    """Zero outside brain — same logic as generate.py's mask_outside_brain step.

    Uses the project's create_brain_mask so behaviour matches what was
    applied to synth volumes at generation time.
    """
    from medgen.metrics.brain_mask import create_brain_mask
    mask = create_brain_mask(
        vol, threshold=threshold, fill_holes=True, dilate_pixels=dilate_pixels
    )
    return (vol * mask.astype(vol.dtype)).astype(np.float32)


def _save_nifti_like(
    data: np.ndarray, reference_img: nib.Nifti1Image, dst: Path
) -> None:
    """Save `data` as a NIfTI at `dst`, preserving affine + header from ref."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    out = nib.Nifti1Image(data, reference_img.affine, reference_img.header)
    nib.save(out, str(dst))


def _process_case(
    src_dir: Path,
    dst_dir: Path,
    brain_mask: bool,
    threshold: float,
    dilate_pixels: int,
) -> None:
    """Load bravo + seg from src_dir, normalize+mask bravo, copy seg, save to dst_dir."""
    bravo_img = nib.load(str(src_dir / "bravo.nii.gz"))
    bravo = bravo_img.get_fdata().astype(np.float32)
    bravo = _normalize_bravo(bravo)
    if brain_mask:
        bravo = _apply_brain_mask(bravo, threshold=threshold, dilate_pixels=dilate_pixels)
    _save_nifti_like(bravo, bravo_img, dst_dir / "bravo.nii.gz")

    # Seg: copy unchanged to preserve label semantics.
    shutil.copy2(str(src_dir / "seg.nii.gz"), str(dst_dir / "seg.nii.gz"))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--real-root", required=True, type=Path,
                        help="Real dataset root (e.g. .../brainmetshare-3).")
    parser.add_argument("--real-splits", nargs="+",
                        default=["train", "val", "test1", "test_new"],
                        help="Splits to union when building the real pool. "
                             "Default = all 4 splits (deduped → 156 unique patients).")
    parser.add_argument("--synth-dir", required=True, type=Path,
                        help="Synthetic root with <id>/{bravo,seg}.nii.gz cases.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--n-real", type=int, default=50)
    parser.add_argument("--n-synth", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--brain-mask", action=argparse.BooleanOptionalAction, default=True,
                        help="Apply brain-mask zeroing to BOTH real and synth bravo "
                             "for fair comparison (default: yes).")
    parser.add_argument("--brain-mask-threshold", type=float, default=0.05)
    parser.add_argument("--brain-mask-dilate", type=int, default=2)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)

    real_subjects = _find_real_subjects_across_splits(args.real_root, args.real_splits)
    synth_subjects = _find_subjects(args.synth_dir)
    logger.info("Real pool: %d unique subjects across splits %s under %s",
                len(real_subjects), args.real_splits, args.real_root)
    logger.info("Synth pool: %d subjects in %s", len(synth_subjects), args.synth_dir)

    if args.n_real > len(real_subjects):
        raise ValueError(
            f"Asked for {args.n_real} real but only {len(real_subjects)} available"
        )
    if args.n_synth > len(synth_subjects):
        raise ValueError(
            f"Asked for {args.n_synth} synth but only {len(synth_subjects)} available"
        )

    # Sample (without replacement) from each pool deterministically via seed.
    real_picks = [real_subjects[i] for i in rng.choice(
        len(real_subjects), size=args.n_real, replace=False)]
    synth_picks = [synth_subjects[i] for i in rng.choice(
        len(synth_subjects), size=args.n_synth, replace=False)]

    # Build the combined pool of (source_path, label) and shuffle to assign IDs.
    pool = [(p, "real") for p in real_picks] + [(p, "synthetic") for p in synth_picks]
    order = rng.permutation(len(pool))

    out_dir = args.output_dir
    volumes_dir = out_dir / "volumes"
    volumes_dir.mkdir(parents=True, exist_ok=True)
    id_width = len(str(len(pool)))  # zero-pad case IDs

    answer_rows: list[tuple[str, str, str]] = []
    form_rows: list[tuple[str, str, str, str]] = []

    for new_idx, src_idx in enumerate(order, start=1):
        src_path, label = pool[src_idx]
        case_id = str(new_idx).zfill(id_width)
        dst_dir = volumes_dir / case_id

        logger.info("[%s] %s ← %s", case_id, label, src_path)
        _process_case(
            src_dir=src_path,
            dst_dir=dst_dir,
            brain_mask=args.brain_mask,
            threshold=args.brain_mask_threshold,
            dilate_pixels=args.brain_mask_dilate,
        )

        answer_rows.append((case_id, label, str(src_path)))
        form_rows.append((case_id, "", "", ""))

    # ── Write answer key (KEEP HIDDEN) ──────────────────────────────────────
    answer_key_path = out_dir / "answer_key.csv"
    with open(answer_key_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "true_label", "source_path"])
        w.writerows(answer_rows)
    logger.info("Wrote answer key: %s (DO NOT SHARE)", answer_key_path)

    # ── Write blank radiologist form (SHARE THIS) ───────────────────────────
    form_path = out_dir / "radiologist_form.csv"
    with open(form_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "prediction", "confidence", "comments"])
        # Header notes go in a separate README so they don't break the CSV
        w.writerows(form_rows)
    logger.info("Wrote radiologist form: %s", form_path)

    # ── README explaining the form ──────────────────────────────────────────
    readme = out_dir / "README.txt"
    readme.write_text(
        "Reader study — real vs synthetic brain-MRI evaluation\n"
        "=====================================================\n"
        f"  {len(pool)} cases ({args.n_real} real + {args.n_synth} synthetic), randomly shuffled.\n"
        f"  Each case has bravo.nii.gz (T1-weighted post-contrast, normalized to [0,1])\n"
        "  and seg.nii.gz (binary brain-metastasis segmentation).\n\n"
        "Files in this directory:\n"
        f"  volumes/{1:0{id_width}d}..{len(pool):0{id_width}d}/  — case folders\n"
        "  radiologist_form.csv — please fill this in (one row per case_id)\n"
        "  answer_key.csv       — TRUTH (do not open before reader scoring)\n\n"
        "Form columns:\n"
        "  case_id     — pre-filled, do not change\n"
        "  prediction  — 'real' or 'synthetic'\n"
        "  confidence  — low / medium / high  (optional)\n"
        "  comments    — free-text reasoning (optional)\n\n"
        f"Brain-mask zeroing applied to bravo: {args.brain_mask}\n"
    )
    logger.info("Wrote %s", readme)

    n_real_assigned = sum(1 for _, lbl, _ in answer_rows if lbl == "real")
    n_synth_assigned = sum(1 for _, lbl, _ in answer_rows if lbl == "synthetic")
    print(f"\nDone. {len(pool)} cases written to {volumes_dir}")
    print(f"  real: {n_real_assigned}   synthetic: {n_synth_assigned}")
    print(f"  answer key:        {answer_key_path}  (keep hidden)")
    print(f"  radiologist form:  {form_path}        (share)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
