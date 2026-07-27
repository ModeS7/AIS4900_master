"""Rebuild nnU-Net TensorBoard curves from the logging history in each checkpoint.

nnU-Net stores the COMPLETE per-epoch logger history inside every checkpoint
(``checkpoint['logging']`` — train/val loss, mean/ema foreground Dice, LR,
per-class Dice, epoch timestamps). If the live event files were lost or
corrupted (e.g. overlapping SLURM allocations writing the same dir), the real
curves are still fully recoverable from ``checkpoint_final.pth``.

This writes ONE clean event file per fold, directly in that fold's
``tensorboard/`` dir. It is NON-DESTRUCTIVE: any pre-existing ``events.out.*``
files are moved into a local ``tensorboard/superseded/`` subdir first (not an
external quarantine). Checkpoints are only read, never modified.

Usage (one line per fold discovered):
    python -m medgen.scripts.rebuild_tensorboard_from_checkpoints \
        --results-root /cluster/work/$USER/AIS4900_master/runs/downstream/nnunet \
        --glob 'exp17_*_d663' [--dry-run]
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import torch

# nnU-Net logger key -> TensorBoard tag. These are the standard nnUNetLogger
# fields; each is a list with one value per epoch.
SCALAR_TAGS = {
    "train/loss": "train_losses",
    "train/learning_rate": "lrs",
    "val/loss": "val_losses",
    "val/mean_fg_dice": "mean_fg_dice",
    "val/ema_fg_dice": "ema_fg_dice",
}


def _load_logging(ckpt_path: Path) -> dict:
    """Read only the logging history from a checkpoint (weights loaded then dropped)."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    log = ckpt.get("logging")
    del ckpt  # free the (large) weights immediately
    if not isinstance(log, dict):
        raise SystemExit(f"{ckpt_path}: no 'logging' dict in checkpoint")
    if not log.get("mean_fg_dice"):
        raise SystemExit(f"{ckpt_path}: logging has no per-epoch history")
    return log


def _supersede_existing(tb_dir: Path, dry_run: bool) -> int:
    """Move any existing event files to tb_dir/superseded/ (kept, not deleted)."""
    existing = sorted(tb_dir.glob("events.out.tfevents.*"))
    if not existing:
        return 0
    if dry_run:
        return len(existing)
    dest = tb_dir / "superseded"
    dest.mkdir(exist_ok=True)
    for ev in existing:
        ev.rename(dest / ev.name)
    return len(existing)


def rebuild_fold(ckpt_path: Path, dry_run: bool) -> int:
    """Rebuild one fold's tensorboard/ from its checkpoint. Returns #epochs."""
    log = _load_logging(ckpt_path)
    series = {tag: log[key] for tag, key in SCALAR_TAGS.items() if log.get(key)}
    n_epochs = len(log["mean_fg_dice"])
    timestamps = log.get("epoch_end_timestamps") or []
    dice_per_class = log.get("dice_per_class_or_region")  # list[epoch] of list[class]
    tb_dir = ckpt_path.parent / "tensorboard"

    moved = _supersede_existing(tb_dir, dry_run)
    if dry_run:
        print(f"[dry-run] {tb_dir}  -> {n_epochs} epochs, {len(series)} scalars"
              f"{f', would supersede {moved} old event file(s)' if moved else ''}")
        return n_epochs

    from torch.utils.tensorboard import SummaryWriter
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))
    for tag, values in series.items():
        for step, value in enumerate(values):
            wt = timestamps[step] if step < len(timestamps) and timestamps[step] else None
            writer.add_scalar(tag, float(value), step, walltime=wt)
    if dice_per_class:
        for step, per_class in enumerate(dice_per_class):
            for ci, value in enumerate(per_class):
                writer.add_scalar(f"val/dice_class_{ci}", float(value), step)
    writer.close()
    print(f"rebuilt {tb_dir}  ({n_epochs} epochs"
          f"{f', {moved} old event file(s) -> superseded/' if moved else ''})")
    return n_epochs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-root", required=True,
                    help="nnUNet_results root (contains the experiment dirs).")
    ap.add_argument("--glob", default="*",
                    help="Experiment-dir glob under results-root (e.g. 'exp17_*_d663').")
    ap.add_argument("--dataset", default="Dataset663_BrainMet")
    ap.add_argument("--model", default="nnUNetTrainerBrainMets__nnUNetResEncUNetLPlansD600__3d_fullres")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would be rebuilt without writing anything.")
    args = ap.parse_args()

    pattern = os.path.join(args.results_root, args.glob, args.dataset, args.model,
                           "fold_*", "checkpoint_final.pth")
    checkpoints = sorted(glob.glob(pattern))
    if not checkpoints:
        raise SystemExit(f"No checkpoints matched:\n  {pattern}")

    print(f"Found {len(checkpoints)} checkpoint(s){' (dry-run)' if args.dry_run else ''}.")
    total_epochs = 0
    for ckpt in checkpoints:
        total_epochs += rebuild_fold(Path(ckpt), args.dry_run)
    verb = "would rebuild" if args.dry_run else "rebuilt"
    print(f"\nDone: {verb} {len(checkpoints)} fold(s), {total_epochs} epoch-points total.")


if __name__ == "__main__":
    main()
