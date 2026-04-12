#!/usr/bin/env python3
"""Generate paired (degraded, clean) volumes for restoration training.

Creates SDEdit-degraded versions of real volumes at various t₀ strengths.
Each real volume gets multiple degraded variants for data diversity.

Output directory structure:
    {output_dir}/
        train/
            patient_001/
                clean_bravo.nii.gz
                seg.nii.gz
                degraded_001.nii.gz   # SDEdit at random t₀
                degraded_002.nii.gz
                degraded_003.nii.gz
            patient_002/
                ...
        val/
            ...
        test_new/
            ...

Usage:
    python -m medgen.scripts.generate_degradation_pairs \
        --bravo-model /path/to/checkpoint.pt \
        --data-root /path/to/brainmetshare-3 \
        --output-dir /path/to/restoration_pairs \
        --t0-min 0.25 --t0-max 0.55 \
        --num-degradations 4 --num-steps 32
"""
import argparse
import json
import logging
import random
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.amp import autocast

from medgen.data.utils import save_nifti
from medgen.diffusion import RFlowStrategy, load_diffusion_model

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


def sdedit_denoise(
    model: torch.nn.Module,
    strategy: RFlowStrategy,
    clean_volume: torch.Tensor,
    seg_mask: torch.Tensor,
    t0: float,
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Apply SDEdit: add noise at level t₀, then denoise from t₀ to 0.

    Args:
        model: Trained bravo diffusion model.
        strategy: RFlowStrategy instance.
        clean_volume: [1, 1, D, H, W] clean volume.
        seg_mask: [1, 1, D, H, W] binary segmentation mask.
        t0: Noise level in [0, 1]. Higher = more degradation.
        num_steps: Total Euler steps (only runs from t₀ to 0).
        device: CUDA device.

    Returns:
        [1, 1, D, H, W] denoised volume, clamped to [0, 1].
    """
    T = strategy.scheduler.num_train_timesteps

    # 1. Add noise at level t₀
    noise = torch.randn_like(clean_volume)
    t_scaled = torch.tensor([int(t0 * T)], device=device)
    noisy = strategy.scheduler.add_noise(clean_volume, noise, t_scaled)

    # 2. Setup scheduler
    d, h, w = clean_volume.shape[2], clean_volume.shape[3], clean_volume.shape[4]
    numel = d * h * w
    strategy.scheduler.set_timesteps(
        num_inference_steps=num_steps, device=device,
        input_img_size_numel=numel,
    )

    # 3. Filter timesteps to only those <= t₀ * T
    t0_threshold = t0 * T
    all_timesteps = strategy.scheduler.timesteps
    all_next = torch.cat((
        all_timesteps[1:],
        torch.tensor([0], dtype=all_timesteps.dtype, device=device),
    ))

    # 4. Euler from t₀ down to 0
    x = noisy
    for t, next_t in zip(all_timesteps, all_next):
        if t.item() > t0_threshold:
            continue

        timesteps_batch = t.unsqueeze(0).to(device)
        next_timestep = next_t.to(device) if isinstance(next_t, torch.Tensor) else torch.tensor(
            next_t, device=device,
        )

        model_input = torch.cat([x, seg_mask], dim=1)

        with autocast('cuda', dtype=torch.bfloat16):
            velocity = model(model_input, timesteps_batch)

        x, _ = strategy.scheduler.step(velocity.float(), t, x, next_timestep)

    return x.clamp(0, 1)


def load_volume_with_meta(
    path: Path, depth: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load NIfTI, normalize, transpose to [D,H,W], pad depth.

    Returns:
        (volume [D,H,W] float32 in [0,1], affine matrix)
    """
    img = nib.load(str(path))
    vol = img.get_fdata().astype(np.float32)
    affine = img.affine

    vmax = vol.max()
    if vmax > 0:
        vol = vol / vmax

    # [H, W, D] -> [D, H, W]
    vol = np.transpose(vol, (2, 0, 1))

    if vol.shape[0] < depth:
        vol = np.pad(vol, ((0, depth - vol.shape[0]), (0, 0), (0, 0)))
    elif vol.shape[0] > depth:
        vol = vol[:depth]

    return vol, affine


def discover_patients(
    split_dir: Path,
) -> list[tuple[str, Path, Path]]:
    """Find all patients with both bravo.nii.gz and seg.nii.gz.

    Returns:
        List of (patient_id, bravo_path, seg_path).
    """
    patients = []
    for patient_dir in sorted(split_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        bravo = patient_dir / "bravo.nii.gz"
        seg = patient_dir / "seg.nii.gz"
        if bravo.exists() and seg.exists():
            patients.append((patient_dir.name, bravo, seg))
    return patients


def main():
    parser = argparse.ArgumentParser(description="Generate SDEdit degradation pairs")
    parser.add_argument("--bravo-model", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True,
                        help="Dataset root (brainmetshare-3)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for restoration pairs")
    parser.add_argument("--t0-min", type=float, default=0.25,
                        help="Min SDEdit strength (default: 0.25)")
    parser.add_argument("--t0-max", type=float, default=0.55,
                        help="Max SDEdit strength (default: 0.55)")
    parser.add_argument("--num-degradations", type=int, default=4,
                        help="Number of degraded variants per volume (default: 4)")
    parser.add_argument("--num-steps", type=int, default=32,
                        help="Euler steps for SDEdit (default: 32)")
    parser.add_argument("--splits", type=str, default="train,val,test_new",
                        help="Comma-separated splits to process")
    parser.add_argument("--depth", type=int, default=160)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="Skip patients that already have all degraded files")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    splits = [s.strip() for s in args.splits.split(",")]

    # ── Load model ───────────────────────────────────────────────────
    logger.info(f"Loading bravo model: {args.bravo_model}")
    model = load_diffusion_model(
        args.bravo_model, device=device,
        in_channels=2, out_channels=1,
        compile_model=False, spatial_dims=3,
    )

    strategy = RFlowStrategy()
    strategy.setup_scheduler(
        num_timesteps=1000,
        image_size=args.image_size,
        depth_size=args.depth,
        spatial_dims=3,
    )

    # ── Process each split ───────────────────────────────────────────
    total_generated = 0
    total_skipped = 0

    for split in splits:
        split_dir = data_root / split
        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}, skipping")
            continue

        patients = discover_patients(split_dir)
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Split: {split} — {len(patients)} patients")
        logger.info(f"{'=' * 50}")

        out_split = output_dir / split
        out_split.mkdir(parents=True, exist_ok=True)

        for p_idx, (patient_id, bravo_path, seg_path) in enumerate(patients):
            out_patient = out_split / patient_id
            out_patient.mkdir(parents=True, exist_ok=True)

            # Check if already done (resume mode)
            if args.resume:
                existing = list(out_patient.glob("degraded_*.nii.gz"))
                if len(existing) >= args.num_degradations:
                    total_skipped += 1
                    continue

            logger.info(f"  [{p_idx + 1}/{len(patients)}] {patient_id}")

            # Load clean bravo volume
            bravo_vol, affine = load_volume_with_meta(bravo_path, args.depth)
            # Load seg mask
            seg_vol, _ = load_volume_with_meta(seg_path, args.depth)
            seg_vol = (seg_vol > 0.5).astype(np.float32)

            # Save clean + seg (copy or create normalized version)
            clean_out = out_patient / "clean_bravo.nii.gz"
            seg_out = out_patient / "seg.nii.gz"

            if not clean_out.exists():
                # Save as [D, H, W] -> transpose back to [H, W, D] for NIfTI
                save_nifti(np.transpose(bravo_vol, (1, 2, 0)), str(clean_out))
            if not seg_out.exists():
                save_nifti(np.transpose(seg_vol, (1, 2, 0)), str(seg_out))

            # Convert to tensors
            clean_t = torch.from_numpy(bravo_vol).unsqueeze(0).unsqueeze(0).to(device)
            seg_t = torch.from_numpy(seg_vol).unsqueeze(0).unsqueeze(0).to(device)

            # Generate degraded variants
            for deg_idx in range(args.num_degradations):
                deg_path = out_patient / f"degraded_{deg_idx + 1:03d}.nii.gz"

                if args.resume and deg_path.exists():
                    continue

                # Random t₀ in configured range
                t0 = random.uniform(args.t0_min, args.t0_max)

                with torch.no_grad():
                    degraded = sdedit_denoise(
                        model, strategy, clean_t, seg_t,
                        t0=t0, num_steps=args.num_steps, device=device,
                    )

                # Save degraded volume
                deg_vol = degraded.squeeze().cpu().numpy()  # [D, H, W]
                save_nifti(np.transpose(deg_vol, (1, 2, 0)), str(deg_path))

                total_generated += 1
                logger.info(f"    degraded_{deg_idx + 1:03d} (t₀={t0:.3f})")

            # Save metadata
            meta = {
                "patient_id": patient_id,
                "t0_range": [args.t0_min, args.t0_max],
                "num_degradations": args.num_degradations,
                "num_steps": args.num_steps,
                "bravo_model": args.bravo_model,
            }
            with open(out_patient / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────
    logger.info(f"\n{'=' * 50}")
    logger.info("GENERATION COMPLETE")
    logger.info(f"{'=' * 50}")
    logger.info(f"Total degraded volumes generated: {total_generated}")
    logger.info(f"Total skipped (resume): {total_skipped}")
    logger.info(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
