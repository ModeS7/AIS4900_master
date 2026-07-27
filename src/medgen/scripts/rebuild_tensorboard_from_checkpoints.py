"""Rebuild nnU-Net TensorBoard curves from the logging history in each checkpoint.

nnU-Net stores the COMPLETE per-epoch logger history inside every checkpoint
(``checkpoint['logging']`` — train/val loss, mean/ema foreground Dice, LR,
per-class Dice, epoch timestamps). If the live event files were lost or
corrupted (e.g. overlapping SLURM allocations writing the same dir), the real
curves are still fully recoverable from ``checkpoint_final.pth``.

This writes exactly ONE event file per fold, directly in that fold's
``tensorboard/`` dir, so TensorBoard shows a single run per fold. Any pre-existing
event files (including ones a validation re-run left behind) are removed first —
safe, because the curve is reconstructed losslessly from the checkpoint and the
pre-recovery originals live in the external quarantine. Checkpoints are only read.
Run this LAST, after any validation re-run (which writes its own event file).

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


def _clear_existing_events(tb_dir: Path, dry_run: bool) -> int:
    """Remove existing event files (and a prior run's superseded/ subdir) so the fold
    ends with exactly ONE event file = one TensorBoard run.

    Safe to delete: the curve is losslessly reconstructed from the checkpoint (re-runnable
    any time), and the pre-recovery originals are preserved in the external quarantine.
    TensorBoard scans subdirs recursively, so a kept superseded/ would show as an extra run.
    """
    import shutil
    existing = sorted(tb_dir.glob("events.out.tfevents.*"))
    old_superseded = tb_dir / "superseded"
    count = len(existing) + (1 if old_superseded.exists() else 0)
    if count == 0:
        return 0
    if dry_run:
        return len(existing)
    for ev in existing:
        ev.unlink()
    if old_superseded.is_dir():
        shutil.rmtree(old_superseded, ignore_errors=True)
    return len(existing)


def _write_progress_png(log: dict, fold_dir: Path) -> None:
    """Reconstruct nnU-Net's progress.png (loss/dice, epoch time, LR) from the logging dict."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = list(range(len(log["mean_fg_dice"])))
    ts = log.get("epoch_end_timestamps") or []
    ep_time = [ts[i] - ts[i - 1] for i in range(1, len(ts))] if len(ts) > 1 else []

    fig, ax = plt.subplots(3, 1, figsize=(18, 24))
    # panel 0: losses (left axis) + pseudo dice (right axis)
    ax0, ax0b = ax[0], ax[0].twinx()
    if log.get("train_losses"):
        ax0.plot(epochs, log["train_losses"], color="b", label="train loss")
    if log.get("val_losses"):
        ax0.plot(epochs, log["val_losses"], color="r", label="val loss")
    ax0b.plot(epochs, log["mean_fg_dice"], color="g", ls="dotted", label="pseudo dice")
    if log.get("ema_fg_dice"):
        ax0b.plot(epochs, log["ema_fg_dice"], color="g", label="ema pseudo dice")
    ax0.set_xlabel("epoch")
    ax0.set_ylabel("loss")
    ax0b.set_ylabel("pseudo dice")
    ax0.legend(loc="upper left")
    ax0b.legend(loc="lower right")
    ax0.set_title("loss & pseudo dice")
    # panel 1: epoch duration
    if ep_time:
        ax[1].plot(epochs[1:], ep_time, color="b")
        ax[1].set_xlabel("epoch")
        ax[1].set_ylabel("time (s)")
        ax[1].set_title("epoch duration")
    # panel 2: learning rate
    if log.get("lrs"):
        ax[2].plot(epochs, log["lrs"], color="b")
        ax[2].set_xlabel("epoch")
        ax[2].set_ylabel("lr")
        ax[2].set_title("learning rate")
    fig.tight_layout()
    fig.savefig(fold_dir / "progress.png")
    plt.close(fig)


def _write_training_log(log: dict, fold_dir: Path) -> None:
    """Reconstruct a per-epoch training_log text file from the logging dict."""
    from datetime import datetime
    ts = log.get("epoch_end_timestamps") or []
    n = len(log["mean_fg_dice"])
    def stamp(i):
        if i < len(ts) and ts[i]:
            return datetime.fromtimestamp(ts[i]).strftime("%Y-%m-%d %H:%M:%S")
        return "unknown-time"
    out = fold_dir / "training_log_reconstructed_from_checkpoint.txt"
    with open(out, "w") as f:
        f.write("# Reconstructed from checkpoint_final.pth['logging']; the original per-epoch\n")
        f.write("# text log was lost. Metric values and epoch-end timestamps are exact.\n\n")
        for i in range(n):
            f.write(f"{stamp(i)}: Epoch {i}\n")
            if log.get("lrs"):
                f.write(f"{stamp(i)}: Current learning rate: {round(float(log['lrs'][i]), 5)}\n")
            if log.get("train_losses"):
                f.write(f"{stamp(i)}: train_loss {round(float(log['train_losses'][i]), 4)}\n")
            if log.get("val_losses"):
                f.write(f"{stamp(i)}: val_loss {round(float(log['val_losses'][i]), 4)}\n")
            dpc = log.get("dice_per_class_or_region")
            if dpc:
                pseudo = [round(float(x), 4) for x in dpc[i]]
            else:
                pseudo = [round(float(log["mean_fg_dice"][i]), 4)]
            f.write(f"{stamp(i)}: Pseudo dice {pseudo}\n")
            if 0 < i < len(ts) and ts[i] and ts[i - 1]:
                f.write(f"{stamp(i)}: Epoch time: {round(ts[i] - ts[i - 1], 2)} s\n")


def rebuild_fold(ckpt_path: Path, dry_run: bool, aux: bool = True) -> int:
    """Rebuild one fold's tensorboard/ (+progress.png/training_log) from its checkpoint."""
    log = _load_logging(ckpt_path)
    series = {tag: log[key] for tag, key in SCALAR_TAGS.items() if log.get(key)}
    n_epochs = len(log["mean_fg_dice"])
    timestamps = log.get("epoch_end_timestamps") or []
    dice_per_class = log.get("dice_per_class_or_region")  # list[epoch] of list[class]
    fold_dir = ckpt_path.parent
    tb_dir = fold_dir / "tensorboard"

    removed = _clear_existing_events(tb_dir, dry_run)
    extras = " + progress.png + training_log" if aux else ""
    if dry_run:
        print(f"[dry-run] {tb_dir}  -> {n_epochs} epochs, {len(series)} scalars{extras}"
              f"{f', would remove {removed} old event file(s)' if removed else ''}")
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
    if aux:
        _write_progress_png(log, fold_dir)
        _write_training_log(log, fold_dir)
    print(f"rebuilt {tb_dir}{extras}  ({n_epochs} epochs"
          f"{f', removed {removed} old event file(s)' if removed else ''})")
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
    ap.add_argument("--no-aux", action="store_true",
                    help="Only rebuild tensorboard/; skip progress.png + training_log.")
    args = ap.parse_args()

    pattern = os.path.join(args.results_root, args.glob, args.dataset, args.model,
                           "fold_*", "checkpoint_final.pth")
    checkpoints = sorted(glob.glob(pattern))
    if not checkpoints:
        raise SystemExit(f"No checkpoints matched:\n  {pattern}")

    print(f"Found {len(checkpoints)} checkpoint(s){' (dry-run)' if args.dry_run else ''}.")
    total_epochs = 0
    for ckpt in checkpoints:
        total_epochs += rebuild_fold(Path(ckpt), args.dry_run, aux=not args.no_aux)
    verb = "would rebuild" if args.dry_run else "rebuilt"
    print(f"\nDone: {verb} {len(checkpoints)} fold(s), {total_epochs} epoch-points total.")


if __name__ == "__main__":
    main()
