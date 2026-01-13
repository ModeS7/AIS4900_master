#!/usr/bin/env python3
"""
Evaluate pretrained MAISI VAE on test/val/train datasets.

MAISI (Medical AI for Synthetic Imaging) VAE from NVIDIA:
- 4x spatial compression per dimension
- 4 latent channels
- Total compression: 16x

Usage:
    python scripts/evaluate_maisi_vae.py --split test
    python scripts/evaluate_maisi_vae.py --split val
    python scripts/evaluate_maisi_vae.py --split train --max_samples 100
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
from monai.inferers import SlidingWindowInferer

# Use existing project metrics (no new dependencies)
from medgen.pipeline.metrics.quality import compute_psnr, compute_msssim, compute_lpips_3d


def load_maisi_vae(bundle_path: str, device: torch.device) -> AutoencoderKlMaisi:
    """Load pretrained MAISI VAE from bundle."""
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
        use_checkpointing=True,  # Memory efficiency
        use_convtranspose=False,
        norm_float16=True,
        num_splits=2,
        dim_split=1,
    )

    checkpoint_path = Path(bundle_path) / "models" / "autoencoder.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model = model.to(device)
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
    parser.add_argument("--no_sliding_window", action="store_true",
                        help="Disable sliding window inference")
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

    # Sliding window inferer for memory efficiency
    inferer = SlidingWindowInferer(
        roi_size=(80, 80, 80),
        sw_batch_size=1,
        overlap=0.4,
        mode="gaussian",
    ) if not args.no_sliding_window else None

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

        # Normalize to [0, 1]
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)

        # Convert to tensor [1, 1, H, W, D]
        volume_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).to(device)

        # Pad depth to multiple of 4 (MAISI requirement)
        D = volume_tensor.shape[-1]
        pad_d = (4 - D % 4) % 4
        if pad_d > 0:
            volume_tensor = F.pad(volume_tensor, (0, pad_d))

        with torch.no_grad(), torch.amp.autocast('cuda', enabled=False):
            # MAISI uses norm_float16=True which can cause dtype mismatches
            # Ensure float32 for inference
            volume_tensor = volume_tensor.float()

            if inferer:
                def encode_fn(x):
                    return model.encode(x.float())[0]
                def decode_fn(z):
                    return model.decode(z.float())

                z = inferer(volume_tensor, encode_fn)
                reconstructed = inferer(z, decode_fn)
            else:
                z = model.encode(volume_tensor)[0]
                reconstructed = model.decode(z)

        reconstructed = torch.clamp(reconstructed, 0, 1)

        # Remove padding
        if pad_d > 0:
            reconstructed = reconstructed[..., :-pad_d]
            volume_tensor = volume_tensor[..., :-pad_d]

        # Compute metrics using project's existing functions
        psnr_val = compute_psnr(reconstructed, volume_tensor)
        msssim_val = compute_msssim(reconstructed, volume_tensor, spatial_dims=3)
        l1_val = F.l1_loss(reconstructed, volume_tensor).item()

        # LPIPS (2D metric applied slice-by-slice)
        # compute_lpips_3d expects [B, C, D, H, W] but we have [B, C, H, W, D]
        # Permute to match expected format
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

    # Print results
    print("\n" + "=" * 60)
    print(f"MAISI VAE Evaluation Results ({args.split} split)")
    print("=" * 60)
    print(f"Samples evaluated: {len(all_metrics['psnr'])}")
    print(f"PSNR:    {np.mean(all_metrics['psnr']):.4f} ± {np.std(all_metrics['psnr']):.4f}")
    print(f"MS-SSIM: {np.mean(all_metrics['msssim']):.4f} ± {np.std(all_metrics['msssim']):.4f}")
    print(f"LPIPS:   {np.mean(all_metrics['lpips']):.4f} ± {np.std(all_metrics['lpips']):.4f}")
    print(f"L1:      {np.mean(all_metrics['l1']):.6f} ± {np.std(all_metrics['l1']):.6f}")
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


if __name__ == "__main__":
    main()
