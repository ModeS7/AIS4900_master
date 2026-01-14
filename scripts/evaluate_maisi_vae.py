#!/usr/bin/env python3
"""
Evaluate pretrained MAISI VAE on test/val/train datasets.

MAISI (Medical AI for Synthetic Imaging) VAE from NVIDIA:
- 4x spatial compression per dimension
- 4 latent channels
- Total compression: 16x

Usage:
    python scripts/evaluate_maisi_vae.py --split test_new
    python scripts/evaluate_maisi_vae.py --split val
    python scripts/evaluate_maisi_vae.py --split train --max_samples 100
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from monai.inferers import SlidingWindowInferer

# Use existing project utilities
from medgen.pipeline.metrics.quality import compute_psnr, compute_msssim, compute_lpips_3d
from medgen.pipeline.tracking.worst_batch import create_worst_batch_figure_3d


def load_maisi_vae(bundle_path: str, device: torch.device) -> AutoencoderKlMaisi:
    """Load pretrained MAISI VAE from bundle."""
    # CRITICAL: norm_float16=False to avoid dtype mismatch bug in MaisiGroupNorm3D
    # The bug: norm_float16=True converts output to float16 but keeps weights in float32
    model = AutoencoderKlMaisi(
        spatial_dims=3,
        in_channels=1,  # MAISI trained on single-channel CT
        out_channels=1,
        latent_channels=4,
        num_channels=[64, 128, 256],
        num_res_blocks=[2, 2, 2],
        norm_num_groups=32,
        norm_eps=1e-6,
        attention_levels=[False, False, False],
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
        use_checkpointing=False,  # Disable for inference
        use_convtranspose=False,
        norm_float16=False,  # CRITICAL: Must be False to avoid dtype mismatch
        num_splits=1,  # Simpler inference without tensor splitting
        dim_split=1,
    )

    checkpoint_path = Path(bundle_path) / "models" / "autoencoder.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model = model.to(device).float()  # Ensure float32
    model.eval()

    print(f"Loaded MAISI VAE from {checkpoint_path}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate MAISI VAE")
    parser.add_argument("--bundle_path", type=str, default="bundles/maisi_ct_generative",
                        help="Path to MAISI bundle")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory (default: from config)")
    parser.add_argument("--split", type=str, default="test_new", choices=["train", "val", "test_new"],
                        help="Dataset split to evaluate")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum samples to evaluate (default: all)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file for results")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = load_maisi_vae(args.bundle_path, device)

    # Load data
    if args.data_dir is None:
        # Use cluster paths as default
        args.data_dir = "/cluster/work/modestas/MedicalDataSets/brainmetshare-3"

    print(f"Evaluating on {args.split} split from {args.data_dir}")

    split_dir = Path(args.data_dir) / args.split

    # Find all patient directories
    patient_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    print(f"Found {len(patient_dirs)} patients")

    if args.max_samples:
        patient_dirs = patient_dirs[:args.max_samples]

    # Evaluation metrics
    all_metrics = {"psnr": [], "msssim": [], "lpips": [], "l1": []}

    # Track worst sample (by LPIPS - higher is worse)
    worst_lpips = -1.0
    worst_sample_data = None

    # Sliding window inferer for decode only (encode is cheap, decode is expensive)
    decode_inferer = SlidingWindowInferer(
        roi_size=(64, 64, 64),  # Smaller ROI for latent space (4x compressed)
        sw_batch_size=1,
        overlap=0.25,
        mode="gaussian",
    )

    import nibabel as nib

    for patient_dir in tqdm(patient_dirs, desc=f"Evaluating {args.split}"):
        # Load first available modality (t1_pre, t1_gd, flair)
        nifti_files = list(patient_dir.glob("*.nii.gz"))
        if not nifti_files:
            continue

        # Prefer t1_pre or t1_gd (skip seg)
        target_file = None
        for name in ["t1_pre", "t1_gd", "flair"]:
            matches = [f for f in nifti_files if name in f.name]
            if matches:
                target_file = matches[0]
                break

        if target_file is None:
            # Skip if only seg available
            non_seg = [f for f in nifti_files if "seg" not in f.name]
            if non_seg:
                target_file = non_seg[0]
            else:
                continue

        # Load and preprocess
        nifti = nib.load(target_file)
        volume = nifti.get_fdata().astype(np.float32)

        # Normalize to [0, 1] using MAISI's MRI preprocessing:
        # "For MR images, intensities were normalized such that the 0th to 99.5th
        # percentile values were scaled to the range [0,1]."
        p0 = np.percentile(volume, 0)
        p99_5 = np.percentile(volume, 99.5)
        volume = np.clip(volume, p0, p99_5)
        volume = (volume - p0) / (p99_5 - p0 + 1e-8)

        # Convert to tensor [1, 1, H, W, D]
        volume_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).to(device).float()

        # Pad all dimensions to multiple of 4 (MAISI 4x compression requirement)
        H, W, D = volume_tensor.shape[2], volume_tensor.shape[3], volume_tensor.shape[4]
        pad_h = (4 - H % 4) % 4
        pad_w = (4 - W % 4) % 4
        pad_d = (4 - D % 4) % 4
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            # F.pad format: (left, right, top, bottom, front, back) for 3D
            volume_tensor = F.pad(volume_tensor, (0, pad_d, 0, pad_w, 0, pad_h))

        with torch.no_grad():
            # Encode (simple forward pass, no sliding window needed)
            z = model.encode(volume_tensor)
            # z is a tuple (z_mu, z_sigma) for VAE, take the mean
            if isinstance(z, (list, tuple)):
                z = z[0]

            # Decode with sliding window for memory efficiency
            # Use the latent directly (no scale_factor for reconstruction eval)
            def decode_fn(latent):
                return model.decode(latent)

            # For smaller volumes, direct decode; for larger, use sliding window
            latent_size = z.shape[2] * z.shape[3] * z.shape[4]
            if latent_size > 64 * 64 * 64:  # Large volume, use SWI
                reconstructed = decode_inferer(z, decode_fn)
            else:
                reconstructed = model.decode(z)

        reconstructed = torch.clamp(reconstructed, 0, 1).float()

        # Remove padding
        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            reconstructed = reconstructed[:, :, :H, :W, :D]
            volume_tensor = volume_tensor[:, :, :H, :W, :D]

        # Compute metrics using project's existing functions
        psnr_val = compute_psnr(reconstructed, volume_tensor)
        msssim_val = compute_msssim(reconstructed, volume_tensor, spatial_dims=3)
        l1_val = F.l1_loss(reconstructed, volume_tensor).item()

        # LPIPS (2D metric applied slice-by-slice)
        # compute_lpips_3d expects [B, C, D, H, W] but we have [B, C, H, W, D]
        try:
            recon_for_lpips = reconstructed.permute(0, 1, 4, 2, 3)  # [B,C,H,W,D] -> [B,C,D,H,W]
            vol_for_lpips = volume_tensor.permute(0, 1, 4, 2, 3)
            lpips_val = compute_lpips_3d(recon_for_lpips, vol_for_lpips, device=device)
        except Exception:
            lpips_val = 0.0

        all_metrics["psnr"].append(psnr_val)
        all_metrics["msssim"].append(msssim_val)
        all_metrics["lpips"].append(lpips_val)
        all_metrics["l1"].append(l1_val)

        # Track worst sample (highest LPIPS = worst perceptual quality)
        if lpips_val > worst_lpips:
            worst_lpips = lpips_val
            # Store data for visualization - permute to [1, C, D, H, W] for create_worst_batch_figure_3d
            worst_sample_data = {
                "original": volume_tensor.permute(0, 1, 4, 2, 3).cpu(),  # [1,1,H,W,D] -> [1,1,D,H,W]
                "generated": reconstructed.permute(0, 1, 4, 2, 3).cpu(),
                "patient": patient_dir.name,
                "metrics": {"psnr": psnr_val, "msssim": msssim_val, "lpips": lpips_val, "l1": l1_val},
            }

    # Print results
    print("\n" + "=" * 60)
    print(f"MAISI VAE Evaluation Results ({args.split} split)")
    print("=" * 60)
    print(f"Samples evaluated: {len(all_metrics['psnr'])}")
    print(f"PSNR:    {np.mean(all_metrics['psnr']):.4f} +/- {np.std(all_metrics['psnr']):.4f}")
    print(f"MS-SSIM: {np.mean(all_metrics['msssim']):.4f} +/- {np.std(all_metrics['msssim']):.4f}")
    print(f"LPIPS:   {np.mean(all_metrics['lpips']):.4f} +/- {np.std(all_metrics['lpips']):.4f}")
    print(f"L1:      {np.mean(all_metrics['l1']):.6f} +/- {np.std(all_metrics['l1']):.6f}")
    print("=" * 60)

    # Save results
    if args.output:
        results = {
            "split": args.split,
            "n_samples": len(all_metrics["psnr"]),
            "metrics": {
                k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                for k, v in all_metrics.items()
            }
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")

    # Save worst sample visualization
    if worst_sample_data is not None:
        output_dir = Path(args.output).parent if args.output else Path("runs/compression_3d/maisi_eval")
        output_dir.mkdir(parents=True, exist_ok=True)

        m = worst_sample_data["metrics"]
        fig = create_worst_batch_figure_3d(
            original=worst_sample_data["original"],
            generated=worst_sample_data["generated"],
            loss=m["lpips"],
            loss_breakdown={"PSNR": m["psnr"], "MS-SSIM": m["msssim"], "LPIPS": m["lpips"], "L1": m["l1"]},
            num_slices=8,
        )
        fig.suptitle(
            f"MAISI VAE Worst Sample - {args.split}/{worst_sample_data['patient']}\n"
            f"PSNR: {m['psnr']:.2f} | MS-SSIM: {m['msssim']:.4f} | LPIPS: {m['lpips']:.4f} | L1: {m['l1']:.6f}",
            fontsize=10, y=1.02
        )

        fig_path = output_dir / f"maisi_worst_{args.split}_{worst_sample_data['patient']}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Worst sample visualization saved to {fig_path}")


if __name__ == "__main__":
    main()
