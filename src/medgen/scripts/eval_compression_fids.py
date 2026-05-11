"""Compute ImageNet + RadImageNet FID across multiple compression models.

Companion to `eval_compression_visual.py` (visualization) and the existing
single-model `eval_compression_fid.py`. This script orchestrates FID
computation for a zoo of compression models at once, using both ImageNet
and RadImageNet ResNet50 backbones, plus a real-vs-real baseline.

For each `--model "label:path"`:
  1. Reconstruct N real test volumes through encode→decode (reusing the
     dispatch logic from eval_compression_visual).
  2. Extract slice-level features with each ResNet50 backbone.
  3. Compute FID against the cached real-volume features.

The "REAL baseline" row is computed by splitting the real volumes into
two disjoint VOLUME halves and computing FID between their slice
features. This is the irreducible FID floor from finite sample size and
within-distribution variance — every model's FID should be interpreted
relative to it.

Usage:
    python -m medgen.scripts.eval_compression_fids \\
        --data-root /path/to/brainmetshare-3 \\
        --split test_new \\
        --output-dir runs/eval/compression_fid_zoo \\
        --model "DC-AE 2D f128:.../checkpoint_latest.pt" \\
        --model "VQ-VAE 3D:.../checkpoint_best.pt" \\
        --model "MAISI VAE 3D:bundles/maisi_ct_generative"
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _real_split_features(
    real_features: torch.Tensor, slices_per_volume: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split slice-level features into two disjoint VOLUME halves.

    Splitting at the slice level would put adjacent (highly-correlated)
    slices of the same volume on both sides, inflating similarity and
    making the baseline FID artificially low. Volume-level split gives
    an honest two-sample-of-the-same-distribution baseline.
    """
    total_slices = real_features.shape[0]
    num_volumes = total_slices // slices_per_volume
    h1_vols = num_volumes // 2
    cutoff = h1_vols * slices_per_volume
    return real_features[:cutoff], real_features[cutoff:]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="test_new")
    parser.add_argument("--modality", default="bravo")
    parser.add_argument("--num-volumes", type=int, default=None,
                        help="Max volumes to use; default = all in split.")
    parser.add_argument("--depth", type=int, default=160)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", action="append", default=[], required=True,
                        metavar="LABEL:CKPT",
                        help='Compression model spec, repeatable. "label:/path/to/ckpt".')
    parser.add_argument("--feature-chunk-size", type=int, default=64,
                        help="Slices per ResNet forward pass.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    parsed: list[tuple[str, Path]] = []
    for spec in args.model:
        if ":" not in spec:
            parser.error(f"--model must be 'label:path', got {spec!r}")
        label, raw_path = spec.split(":", 1)
        p = Path(raw_path).expanduser()
        if not p.exists():
            parser.error(f"Model path not found: {p}")
        parsed.append((label.strip(), p))
    args.model_specs = parsed
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    args = parse_args(argv)

    # Reuse: load/build/reconstruct helpers from the visualization script
    from medgen.metrics.feature_extractors import ResNet50Features
    from medgen.metrics.generation import compute_fid
    from medgen.metrics.generation_3d import extract_features_3d
    from medgen.scripts.eval_compression_visual import (
        _build_model_for_label,
        _find_split_dir,
        load_volume_nifti,
        reconstruct_2d_per_slice,
        reconstruct_3d,
    )

    device = torch.device(args.device)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # ── Load real volumes ────────────────────────────────────────────────────
    data_root = Path(args.data_root)
    split_dir = _find_split_dir(data_root, args.split, args.modality)
    all_subjects = sorted(
        p.parent.name for p in split_dir.glob(f"*/{args.modality}.nii.gz")
    )
    if args.num_volumes:
        all_subjects = all_subjects[: args.num_volumes]
    if not all_subjects:
        logger.error("No subjects found in %s", split_dir)
        return 2
    logger.info("Loading %d real volumes from %s", len(all_subjects), split_dir)

    real_vols_list: list[torch.Tensor] = []
    for subj in all_subjects:
        vol_np, _ = load_volume_nifti(
            split_dir / subj / f"{args.modality}.nii.gz",
            depth=args.depth, image_size=args.image_size,
        )
        real_vols_list.append(torch.from_numpy(vol_np).float())
    real_vols = torch.stack(real_vols_list).unsqueeze(1)  # [N, 1, D, H, W]
    n_volumes = real_vols.shape[0]
    slices_per_volume = real_vols.shape[2]
    logger.info(
        "Real tensor shape: %s (%d volumes × %d slices = %d slices/backbone)",
        list(real_vols.shape), n_volumes, slices_per_volume,
        n_volumes * slices_per_volume,
    )

    # ── Set up feature extractors (both backbones) ───────────────────────────
    extractors = {
        "imagenet":    ResNet50Features(device, network_type="imagenet"),
        "radimagenet": ResNet50Features(device, network_type="radimagenet"),
    }

    # ── Cache real-data features per backbone ────────────────────────────────
    logger.info("Extracting real features under both backbones...")
    real_features: dict[str, torch.Tensor] = {}
    for name, ext in extractors.items():
        ext.to(device).eval()
        feats = extract_features_3d(
            real_vols.to(device), ext, chunk_size=args.feature_chunk_size,
        )
        real_features[name] = feats
        logger.info("  %-12s real features: %s", name, list(feats.shape))
        torch.cuda.empty_cache()

    # ── Real-baseline FID (volume-level half split) ──────────────────────────
    logger.info("Computing real-baseline FID (volume-level half split)...")
    baseline_fids: dict[str, float] = {}
    for name, feats in real_features.items():
        f1, f2 = _real_split_features(feats, slices_per_volume)
        baseline_fids[name] = compute_fid(f1, f2)
        logger.info("  %-12s baseline FID: %.4f  (half1=%d slices vs half2=%d)",
                    name, baseline_fids[name], f1.shape[0], f2.shape[0])

    # ── Per-model: reconstruct → features → FID ──────────────────────────────
    results: list[tuple[str, str, float]] = []
    for label, ckpt in args.model_specs:
        logger.info("=" * 70)
        logger.info("Model: %s  (%s)", label, ckpt)
        model, ctype, sdims, scale, lat = _build_model_for_label(
            label, ckpt, device
        )
        model.eval()
        logger.info("  type=%s sdims=%dD scale=%dx lat=%d",
                    ctype, sdims, scale, lat)

        # Reconstruct all volumes; keep on CPU between models to free GPU
        recons_list: list[torch.Tensor] = []
        for i, vol_t_cpu in enumerate(real_vols_list):
            vol_t = vol_t_cpu.unsqueeze(0).unsqueeze(0).to(device)
            if sdims == 3:
                rec = reconstruct_3d(model, vol_t, ctype)
            else:
                rec = reconstruct_2d_per_slice(model, vol_t, ctype)
            recons_list.append(rec.squeeze(0).cpu())  # [1, D, H, W]
            if (i + 1) % 10 == 0 or (i + 1) == n_volumes:
                logger.info("  reconstructed %d/%d", i + 1, n_volumes)

        recon_vols = torch.stack(recons_list)  # [N, 1, D, H, W]
        del model
        torch.cuda.empty_cache()

        # Features + FID per backbone
        for name, ext in extractors.items():
            recon_features = extract_features_3d(
                recon_vols.to(device), ext, chunk_size=args.feature_chunk_size,
            )
            fid = compute_fid(real_features[name], recon_features)
            results.append((label, name, fid))
            logger.info("  %-12s FID: %.4f  (Δ vs baseline = %.4f)",
                        name, fid, fid - baseline_fids[name])
            del recon_features
            torch.cuda.empty_cache()
        del recon_vols, recons_list

    # ── Write CSV ────────────────────────────────────────────────────────────
    csv_path = out_root / "fid_metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "backbone", "fid", "delta_vs_baseline"])
        for name in extractors:
            w.writerow([
                "REAL baseline (vol-half vs vol-half)",
                name, f"{baseline_fids[name]:.4f}", "0.0000",
            ])
        for label, name, fid in results:
            delta = fid - baseline_fids[name]
            w.writerow([label, name, f"{fid:.4f}", f"{delta:.4f}"])
    logger.info("Wrote %s", csv_path)

    # ── Pretty-printed grouped summary ───────────────────────────────────────
    summary_path = out_root / "fid_summary.txt"
    lines: list[str] = []
    lines.append("=== Compression-model FID summary ===")
    lines.append(
        f"N volumes: {n_volumes} × {slices_per_volume} slices = "
        f"{n_volumes * slices_per_volume} per backbone"
    )
    lines.append("")
    for name in extractors:
        lines.append(f"--- {name} ResNet50 backbone ---")
        lines.append(
            f"  {'REAL baseline (vol-half vs vol-half)':<45}"
            f"  FID = {baseline_fids[name]:8.4f}    Δ = baseline"
        )
        for label, bname, fid in results:
            if bname == name:
                delta = fid - baseline_fids[name]
                lines.append(
                    f"  {label:<45}  FID = {fid:8.4f}    Δ = {delta:+8.4f}"
                )
        lines.append("")
    summary_text = "\n".join(lines)
    summary_path.write_text(summary_text)
    print("\n" + summary_text)
    logger.info("Wrote %s", summary_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
