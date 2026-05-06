"""Visual reconstruction comparison for compression models (VAE / VQ-VAE / DC-AE, 2D and 3D).

Loads N real volumes, runs each through one or more compression models,
saves per-model NIfTI reconstructions plus a thesis-ready figure showing
original vs. reconstructions side-by-side with PSNR/SSIM annotations.

Usage:
    python -m medgen.scripts.eval_compression_visual \\
        --data-root /path/to/brainmetshare-3 \\
        --split test \\
        --num-volumes 3 \\
        --output-dir runs/eval/compression_visual \\
        --model "DC-AE 2D f32:/path/to/dcae_2d_f32/checkpoint_best.pt" \\
        --model "DC-AE 2D f128:/path/to/dcae_2d_f128/checkpoint_best.pt" \\
        --model "VQ-VAE 3D 4x:/path/to/vqvae_3d/checkpoint_best.pt" \\
        --model "VAE 3D 4x:/path/to/vae_3d/checkpoint_best.pt"

Each --model arg is "label:checkpoint_path". Spatial dims (2 vs 3) and
compression type are auto-detected from the checkpoint.

Outputs (under --output-dir):
    metrics.csv                         per-volume PSNR/MS-SSIM per model
    reconstruction_grid.png             N rows × (1+M) cols mid-slice grid
    reconstruction_grid.pdf             same, vector for thesis
    nifti/<label>/<NN>.nii.gz           reconstructed volumes (sanitized labels)
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from torch.amp import autocast

logger = logging.getLogger(__name__)


def _sanitize(label: str) -> str:
    """Make a label safe for use as a filesystem path."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label.strip()).strip("_") or "model"


def _find_split_dir(data_root: Path, split: str, modality: str) -> Path:
    """Resolve the split directory under data_root.

    Tries ``data_root/split`` first, then any subdir matching ``split*``.
    """
    direct = data_root / split
    if direct.is_dir() and any(direct.glob(f"*/{modality}.nii.gz")):
        return direct
    for sub in sorted(data_root.iterdir()):
        if sub.is_dir() and sub.name.startswith(split):
            if any(sub.glob(f"*/{modality}.nii.gz")):
                return sub
    raise FileNotFoundError(
        f"No split directory under {data_root} matching '{split}' contains {modality}.nii.gz"
    )


def load_volume_nifti(
    path: Path, depth: int, image_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI, normalize to [0,1], crop/pad depth, return (volume, affine).

    Returns:
        vol: [D, H, W] float32 in [0, 1].
        affine: 4x4 affine for saving reconstructions back as NIfTI.
    """
    img = nib.load(str(path))
    vol = img.get_fdata().astype(np.float32)
    affine = img.affine

    vmin, vmax = float(vol.min()), float(vol.max())
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)

    # NIfTI is [H, W, D] -> internal [D, H, W]
    vol = np.transpose(vol, (2, 0, 1))

    d = vol.shape[0]
    if d < depth:
        pad = np.zeros((depth - d, vol.shape[1], vol.shape[2]), dtype=np.float32)
        vol = np.concatenate([vol, pad], axis=0)
    elif d > depth:
        vol = vol[:depth]

    # Center-crop H, W to image_size if larger
    h, w = vol.shape[1], vol.shape[2]
    if h > image_size:
        s = (h - image_size) // 2
        vol = vol[:, s : s + image_size]
    if w > image_size:
        s = (w - image_size) // 2
        vol = vol[:, :, s : s + image_size]

    return vol, affine


@torch.no_grad()
def reconstruct_3d(
    model: torch.nn.Module, vol: torch.Tensor, ctype: str
) -> torch.Tensor:
    """Encode-decode a [1, 1, D, H, W] tensor with a 3D compression model."""
    with autocast("cuda", dtype=torch.bfloat16):
        if ctype == "vqvae":
            z = model.encode(vol)
            recon = model.decode_stage_2_outputs(z)
        elif ctype == "vae":
            z_mu, _ = model.encode(vol)
            recon = model.decode(z_mu)
        elif ctype == "dcae":
            z = model.encode(vol)
            recon = model.decode(z)
        else:
            raise ValueError(f"Unknown compression type: {ctype}")
    return recon.float().clamp(0, 1)


@torch.no_grad()
def reconstruct_2d_per_slice(
    model: torch.nn.Module,
    vol: torch.Tensor,
    ctype: str,
    chunk: int = 16,
) -> torch.Tensor:
    """Encode-decode a 3D volume slice-by-slice through a 2D compression model.

    Args:
        vol: [1, 1, D, H, W] tensor in [0, 1].
        chunk: number of slices per forward pass.
    Returns:
        [1, 1, D, H, W] reconstructed tensor in [0, 1].
    """
    _, _, d, h, w = vol.shape
    slices = vol.squeeze(0).squeeze(0)  # [D, H, W]
    out_slices = []
    for start in range(0, d, chunk):
        end = min(start + chunk, d)
        batch = slices[start:end].unsqueeze(1)  # [chunk, 1, H, W]
        with autocast("cuda", dtype=torch.bfloat16):
            if ctype == "vqvae":
                z = model.encode(batch)
                recon = model.decode_stage_2_outputs(z)
            elif ctype == "vae":
                z_mu, _ = model.encode(batch)
                recon = model.decode(z_mu)
            elif ctype == "dcae":
                z = model.encode(batch)
                recon = model.decode(z)
            else:
                raise ValueError(f"Unknown compression type: {ctype}")
        out_slices.append(recon.float().clamp(0, 1).squeeze(1))  # [chunk, H, W]
    full = torch.cat(out_slices, dim=0)  # [D, H, W]
    return full.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]


def save_nifti_volume(vol: np.ndarray, affine: np.ndarray, path: Path) -> None:
    """Save a [D, H, W] volume as NIfTI (transposed back to [H, W, D])."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.transpose(vol, (1, 2, 0))  # [H, W, D]
    nib.save(nib.Nifti1Image(arr.astype(np.float32), affine), str(path))


def plot_grid(
    originals: list[np.ndarray],
    recons_per_model: list[list[np.ndarray]],
    model_labels: list[str],
    volume_ids: list[str],
    metrics: list[list[dict]],
    output_path_png: Path,
    output_path_pdf: Path,
) -> None:
    """Build a (N rows × (1 + M) cols) figure of axial mid-slices.

    Row i = volume i; col 0 = original, cols 1..M = each model's reconstruction.
    Each reconstruction cell is annotated with PSNR / MS-SSIM.
    """
    n_vols = len(originals)
    n_models = len(model_labels)
    n_cols = 1 + n_models

    fig, axes = plt.subplots(
        n_vols, n_cols,
        figsize=(2.5 * n_cols, 2.5 * n_vols),
        squeeze=False,
    )

    for i, orig in enumerate(originals):
        mid = orig.shape[0] // 2  # axial mid
        axes[i, 0].imshow(orig[mid], cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_ylabel(volume_ids[i], fontsize=9)
        if i == 0:
            axes[i, 0].set_title("Original", fontsize=10)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        for j, label in enumerate(model_labels):
            rec = recons_per_model[j][i]
            ax = axes[i, 1 + j]
            ax.imshow(rec[mid], cmap="gray", vmin=0, vmax=1)
            if i == 0:
                ax.set_title(label, fontsize=10)
            psnr = metrics[j][i]["psnr"]
            ssim = metrics[j][i]["msssim"]
            ax.set_xlabel(f"PSNR {psnr:.2f}\nMS-SSIM {ssim:.3f}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    output_path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path_png, dpi=200, bbox_inches="tight")
    fig.savefig(output_path_pdf, bbox_inches="tight")
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, help="Path to brainmetshare-3 root.")
    parser.add_argument("--split", default="test", help="Split subdir to sample from.")
    parser.add_argument("--modality", default="bravo", help="Modality file stem (default: bravo).")
    parser.add_argument("--num-volumes", type=int, default=3, help="Number of volumes to reconstruct.")
    parser.add_argument("--volume-ids", nargs="*", default=None,
                        help="Explicit subject IDs; overrides --num-volumes if given.")
    parser.add_argument("--depth", type=int, default=160)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--output-dir", required=True, help="Where to write outputs.")
    parser.add_argument("--model", action="append", default=[], required=True,
                        metavar="LABEL:CKPT",
                        help='Compression model spec, repeatable. Format: "label:/abs/path/to/ckpt".')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    parsed_models: list[tuple[str, Path]] = []
    for spec in args.model:
        if ":" not in spec:
            parser.error(f"--model must be 'label:path', got {spec!r}")
        label, raw_path = spec.split(":", 1)
        ckpt = Path(raw_path).expanduser()
        if not ckpt.is_file():
            parser.error(f"Checkpoint not found: {ckpt}")
        parsed_models.append((label.strip(), ckpt))
    args.model_specs = parsed_models
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)

    from medgen.data.loaders.compression_detection import load_compression_model
    from medgen.metrics.quality import compute_msssim, compute_psnr

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # ── Pick volumes ──────────────────────────────────────────────────────────
    data_root = Path(args.data_root)
    split_dir = _find_split_dir(data_root, args.split, args.modality)
    all_subjects = sorted(p.parent.name for p in split_dir.glob(f"*/{args.modality}.nii.gz"))
    if not all_subjects:
        logger.error("No subjects found in %s", split_dir)
        return 2

    if args.volume_ids:
        missing = [v for v in args.volume_ids if v not in all_subjects]
        if missing:
            logger.error("Subjects not found in split: %s", missing)
            return 2
        chosen = args.volume_ids
    else:
        idx = rng.choice(len(all_subjects), size=min(args.num_volumes, len(all_subjects)), replace=False)
        chosen = [all_subjects[i] for i in sorted(idx)]
    logger.info("Selected %d volumes from %s: %s", len(chosen), split_dir, chosen)

    # ── Load originals (CPU) ──────────────────────────────────────────────────
    originals: list[np.ndarray] = []
    affines: list[np.ndarray] = []
    for subj in chosen:
        vol, aff = load_volume_nifti(split_dir / subj / f"{args.modality}.nii.gz",
                                     depth=args.depth, image_size=args.image_size)
        originals.append(vol)
        affines.append(aff)
        logger.info("  loaded %s shape=%s", subj, vol.shape)

    # ── Run each model ────────────────────────────────────────────────────────
    model_labels: list[str] = []
    recons_per_model: list[list[np.ndarray]] = []
    metrics_per_model: list[list[dict]] = []

    for label, ckpt in args.model_specs:
        logger.info("=" * 70)
        logger.info("Loading model: %s  (%s)", label, ckpt)
        model, ctype, sdims, scale, latent_ch = load_compression_model(
            str(ckpt), None, device, spatial_dims="auto"
        )
        model.eval()
        logger.info("  detected: type=%s spatial_dims=%dD scale=%dx latent_ch=%d",
                    ctype, sdims, scale, latent_ch)

        recons: list[np.ndarray] = []
        per_vol_metrics: list[dict] = []
        for i, vol_np in enumerate(originals):
            vol_t = torch.from_numpy(vol_np).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,D,H,W]
            if sdims == 3:
                rec_t = reconstruct_3d(model, vol_t, ctype)
            else:
                rec_t = reconstruct_2d_per_slice(model, vol_t, ctype)
            rec_np = rec_t.squeeze(0).squeeze(0).cpu().numpy()  # [D, H, W]
            recons.append(rec_np)

            # Metrics — both 3D MS-SSIM (volume-level) and PSNR
            psnr = compute_psnr(rec_t, vol_t)
            try:
                msssim = compute_msssim(rec_t, vol_t, spatial_dims=3)
            except (ValueError, RuntimeError) as exc:
                logger.warning("MS-SSIM failed for %s vol %s, falling back to 2D mean: %s",
                               label, chosen[i], exc)
                # Fallback: mean MS-SSIM across axial slices
                rec_slices = rec_t.squeeze(0).permute(1, 0, 2, 3)  # [D, 1, H, W]
                ref_slices = vol_t.squeeze(0).permute(1, 0, 2, 3)
                msssim = compute_msssim(rec_slices, ref_slices, spatial_dims=2)

            per_vol_metrics.append({"psnr": psnr, "msssim": msssim})
            logger.info("  %s vs %s: PSNR=%.3f MS-SSIM=%.4f", label, chosen[i], psnr, msssim)

            # Save reconstructed NIfTI
            nifti_path = out_root / "nifti" / _sanitize(label) / f"{chosen[i]}.nii.gz"
            save_nifti_volume(rec_np, affines[i], nifti_path)

        model_labels.append(label)
        recons_per_model.append(recons)
        metrics_per_model.append(per_vol_metrics)

        # Free GPU memory before next model
        del model
        torch.cuda.empty_cache()

    # ── CSV ────────────────────────────────────────────────────────────────────
    csv_path = out_root / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "subject", "psnr", "msssim"])
        for label, per_vol in zip(model_labels, metrics_per_model, strict=True):
            for subj, m in zip(chosen, per_vol, strict=True):
                w.writerow([label, subj, f"{m['psnr']:.4f}", f"{m['msssim']:.4f}"])
        # Aggregate row per model (mean across volumes)
        w.writerow([])
        w.writerow(["model", "subject", "psnr_mean", "msssim_mean"])
        for label, per_vol in zip(model_labels, metrics_per_model, strict=True):
            psnr_mean = float(np.mean([m["psnr"] for m in per_vol]))
            msssim_mean = float(np.mean([m["msssim"] for m in per_vol]))
            w.writerow([label, "MEAN", f"{psnr_mean:.4f}", f"{msssim_mean:.4f}"])
    logger.info("Wrote metrics to %s", csv_path)

    # ── Figure ────────────────────────────────────────────────────────────────
    plot_grid(
        originals, recons_per_model, model_labels, chosen, metrics_per_model,
        out_root / "reconstruction_grid.png",
        out_root / "reconstruction_grid.pdf",
    )
    logger.info("Wrote figure to %s and .pdf", out_root / "reconstruction_grid.png")

    # ── Print compact summary ────────────────────────────────────────────────
    print("\n=== Summary (mean across volumes) ===")
    print(f"{'model':<30} {'PSNR':>8} {'MS-SSIM':>10}")
    for label, per_vol in zip(model_labels, metrics_per_model, strict=True):
        psnr_mean = float(np.mean([m["psnr"] for m in per_vol]))
        msssim_mean = float(np.mean([m["msssim"] for m in per_vol]))
        print(f"{label:<30} {psnr_mean:8.3f} {msssim_mean:10.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
