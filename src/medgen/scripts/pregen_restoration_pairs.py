#!/usr/bin/env python3
"""Pre-generate exp1_1_1000 outputs for IR-SDE restoration pair training (exp43b).

Iterates over subjects in a real-data split (e.g. train/), and for each subject:
  1. Copies clean real volume → clean_bravo.nii.gz
  2. Copies real seg mask     → seg.nii.gz
  3. Generates N variants using the provided bravo model conditioned on this
     subject's seg → degraded_000.nii.gz, degraded_001.nii.gz, ...

Output structure matches `Restoration3DDataset` precomputed expectations:

    <output_root>/
        patient_001/
            clean_bravo.nii.gz
            seg.nii.gz
            degraded_000.nii.gz
            degraded_001.nii.gz
            ...

The script is resumable: subjects with all N variants already present are
skipped; partial ones fill in the missing variants only.

Usage:
    python -m medgen.scripts.pregen_restoration_pairs \\
        --checkpoint /cluster/work/.../exp1_1_1000.../checkpoint_latest.pt \\
        --data-root /cluster/work/.../brainmetshare-3/train \\
        --output-root /cluster/work/.../restoration_pairs/train \\
        --num-steps 32 --num-variants 3 --seed-base 42

Cost estimate: 105 subjects × 3 variants × ~45s/gen ≈ 2.5h on A100.
"""
import argparse
import logging
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from torch.amp import autocast

from medgen.diffusion import RFlowStrategy, load_diffusion_model

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_seg(path: Path, depth: int = 160) -> np.ndarray:
    """Load seg as [D, H, W] binary."""
    vol = nib.load(str(path)).get_fdata().astype(np.float32)
    vol = np.transpose(vol, (2, 0, 1))
    d = vol.shape[0]
    if d < depth:
        vol = np.pad(vol, ((0, depth - d), (0, 0), (0, 0)))
    elif d > depth:
        vol = vol[:depth]
    return (vol > 0.5).astype(np.float32)


def save_nifti_bravo_layout(vol: np.ndarray, path: Path) -> None:
    """Save [D, H, W] as [H, W, D] NIfTI matching brainmetshare convention."""
    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.transpose(vol, (1, 2, 0)).astype(np.float32)
    nib.save(nib.Nifti1Image(out, affine=np.eye(4)), str(path))


@torch.no_grad()
def generate_one(
    model,
    strategy: RFlowStrategy,
    seg: torch.Tensor,
    num_steps: int,
    T: int,
    seed: int,
    device: torch.device,
) -> np.ndarray:
    """Deterministic Euler generation conditioned on seg."""
    torch.manual_seed(seed)
    d, h, w = seg.shape[2], seg.shape[3], seg.shape[4]
    x_t = torch.randn(1, 1, d, h, w, device=device)

    steps = torch.linspace(T, 0.0, num_steps + 1, device=device)
    strategy.scheduler.set_timesteps(
        num_inference_steps=num_steps, device=device,
        input_img_size_numel=d * h * w,
    )
    for i in range(num_steps):
        t = steps[i]
        next_t = steps[i + 1]
        t_batch = t.unsqueeze(0).to(device)
        with autocast('cuda', dtype=torch.bfloat16):
            velocity = model(torch.cat([x_t, seg], dim=1), t_batch)
        x_t, _ = strategy.scheduler.step(velocity.float(), t, x_t, next_t)

    return x_t.clamp(0, 1).squeeze().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-generate restoration pair training data")
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', required=True,
                        help='Root of real split (contains subject/bravo.nii.gz + seg.nii.gz)')
    parser.add_argument('--output-root', required=True,
                        help='Where to write restoration_pairs/split/ layout')
    parser.add_argument('--num-steps', type=int, default=32)
    parser.add_argument('--num-variants', type=int, default=3,
                        help='Number of degraded variants per subject')
    parser.add_argument('--depth', type=int, default=160)
    parser.add_argument('--seed-base', type=int, default=42)
    parser.add_argument('--max-subjects', type=int, default=0,
                        help='0 = all subjects; otherwise cap (for quick tests)')
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    logger.info(f"Loading exp1_1_1000 from: {args.checkpoint}")
    model = load_diffusion_model(
        args.checkpoint, device=device,
        in_channels=2, out_channels=1, spatial_dims=3,
    ).eval()
    strategy = RFlowStrategy()
    strategy.setup_scheduler(
        num_timesteps=1000, image_size=256,
        depth_size=args.depth, spatial_dims=3,
    )
    T = strategy.scheduler.num_train_timesteps

    subjects = sorted([d for d in data_root.iterdir() if d.is_dir()])
    if args.max_subjects:
        subjects = subjects[:args.max_subjects]
    logger.info(f"Processing {len(subjects)} subjects × {args.num_variants} variants")

    skipped = 0
    generated = 0
    for si, subj_dir in enumerate(subjects):
        bravo_src = subj_dir / 'bravo.nii.gz'
        seg_src = subj_dir / 'seg.nii.gz'
        if not bravo_src.exists() or not seg_src.exists():
            logger.warning(f"[{si + 1}/{len(subjects)}] SKIP {subj_dir.name}: missing bravo or seg")
            continue

        out_dir = output_root / subj_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Copy clean + seg if not already there
        clean_dst = out_dir / 'clean_bravo.nii.gz'
        seg_dst = out_dir / 'seg.nii.gz'
        if not clean_dst.exists():
            shutil.copy(bravo_src, clean_dst)
        if not seg_dst.exists():
            shutil.copy(seg_src, seg_dst)

        # Load seg once for all variants (same conditioning)
        seg_np = load_seg(seg_src, args.depth)
        seg_t = torch.from_numpy(seg_np).unsqueeze(0).unsqueeze(0).to(device)

        for v in range(args.num_variants):
            out_path = out_dir / f'degraded_{v:03d}.nii.gz'
            if out_path.exists():
                skipped += 1
                continue

            # Deterministic per-(subject, variant) seed
            seed = args.seed_base + v * 100003 + (abs(hash(subj_dir.name)) % 100003)
            gen_np = generate_one(model, strategy, seg_t, args.num_steps, T, seed, device)
            save_nifti_bravo_layout(gen_np, out_path)
            generated += 1

            if generated % 5 == 0 or si == 0:
                logger.info(
                    f"[{si + 1}/{len(subjects)}] {subj_dir.name} variant={v}  "
                    f"(generated so far: {generated}, skipped: {skipped})"
                )

    logger.info(f"✓ Done — generated={generated}, already-present-skipped={skipped}")


if __name__ == '__main__':
    main()
