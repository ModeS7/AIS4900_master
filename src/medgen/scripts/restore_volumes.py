#!/usr/bin/env python3
"""Apply trained restoration model to generated (or degraded) volumes.

Loads a trained restoration model and applies it to a directory of
generated volumes, producing restored (deblurred) outputs.

Supports all restoration strategies:
    - rflow: RFlow Bridge (Euler integration from degraded to clean)
    - irsde: IR-SDE (reverse mean-reverting SDE)
    - resfusion: Resfusion (truncated DDPM with residual)

Usage:
    # Restore generated volumes
    python -m medgen.scripts.restore_volumes \
        --restoration-model runs/restoration/checkpoint.pt \
        --strategy rflow \
        --input-dir /path/to/generated/bravo_volumes \
        --output-dir /path/to/restored/bravo_volumes \
        --num-steps 25

    # Restore and evaluate against real data
    python -m medgen.scripts.restore_volumes \
        --restoration-model runs/restoration/checkpoint.pt \
        --strategy rflow \
        --input-dir /path/to/generated/bravo_volumes \
        --output-dir /path/to/restored/bravo_volumes \
        --num-steps 25 \
        --eval --data-root /path/to/real/data
"""
import argparse
import json
import logging
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from medgen.data.utils import save_nifti
from medgen.diffusion import RFlowStrategy, load_diffusion_model

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


def load_strategy(strategy_name: str, num_timesteps: int, image_size: int, depth: int):
    """Create and setup a strategy instance by name."""
    if strategy_name == 'rflow':
        strategy = RFlowStrategy()
        strategy.setup_scheduler(
            num_timesteps=1000, image_size=image_size,
            depth_size=depth, spatial_dims=3,
        )
        return strategy
    elif strategy_name == 'irsde':
        from medgen.diffusion.strategy_irsde import IRSDEStrategy
        strategy = IRSDEStrategy()
        strategy.setup_scheduler(
            num_timesteps=num_timesteps, image_size=image_size,
            depth_size=depth, spatial_dims=3,
        )
        return strategy
    elif strategy_name == 'resfusion':
        from medgen.diffusion.strategy_resfusion import ResfusionStrategy
        strategy = ResfusionStrategy()
        strategy.setup_scheduler(
            num_timesteps=1000, image_size=image_size,
            depth_size=depth, spatial_dims=3,
        )
        return strategy
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def restore_volume(
    model: torch.nn.Module,
    strategy,
    degraded: torch.Tensor,
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Restore one degraded volume using the trained restoration model.

    Args:
        model: Trained restoration model (2in, 1out).
        strategy: Configured strategy instance.
        degraded: [1, 1, D, H, W] degraded/generated volume.
        num_steps: Number of reverse steps.
        device: CUDA device.

    Returns:
        [1, 1, D, H, W] restored volume.
    """
    # Model input: [degraded_as_start, degraded_as_conditioning]
    model_input = torch.cat([degraded, degraded], dim=1)

    with torch.no_grad():
        restored = strategy.generate(
            model, model_input, num_steps, device,
        )

    return restored.clamp(0, 1)


def load_volume(path: Path, depth: int) -> np.ndarray:
    """Load NIfTI, normalize to [0,1], transpose to [D,H,W], pad depth."""
    vol = nib.load(str(path)).get_fdata().astype(np.float32)
    vmax = vol.max()
    if vmax > 0:
        vol = vol / vmax
    vol = np.transpose(vol, (2, 0, 1))  # [H,W,D] -> [D,H,W]
    if vol.shape[0] < depth:
        vol = np.pad(vol, ((0, depth - vol.shape[0]), (0, 0), (0, 0)))
    elif vol.shape[0] > depth:
        vol = vol[:depth]
    return vol


def main():
    parser = argparse.ArgumentParser(description="Apply restoration model to volumes")
    parser.add_argument("--restoration-model", type=str, required=True,
                        help="Path to trained restoration checkpoint")
    parser.add_argument("--strategy", type=str, default="rflow",
                        choices=["rflow", "irsde", "resfusion", "bridge"],
                        help="Restoration strategy (default: rflow)")
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory with generated volumes (*/bravo.nii.gz or flat *.nii.gz)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for restored volumes")
    parser.add_argument("--num-steps", type=int, default=25,
                        help="Number of reverse steps (default: 25)")
    parser.add_argument("--num-timesteps", type=int, default=100,
                        help="IR-SDE num_timesteps (default: 100)")
    parser.add_argument("--depth", type=int, default=160)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-volumes", type=int, default=None,
                        help="Max volumes to process (default: all)")
    parser.add_argument("--eval", action="store_true",
                        help="Compute metrics after restoration")
    parser.add_argument("--data-root", type=str, default=None,
                        help="Real data root for evaluation (required if --eval)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Find input volumes ───────────────────────────────────────────
    # Support both flat directory (*.nii.gz) and nested (*/bravo.nii.gz)
    files = sorted(input_dir.glob("*/bravo.nii.gz"))
    if not files:
        files = sorted(input_dir.glob("*.nii.gz"))
    if not files:
        raise ValueError(f"No NIfTI files found in {input_dir}")
    if args.max_volumes:
        files = files[:args.max_volumes]
    logger.info(f"Found {len(files)} volumes to restore")

    # ── Load model ───────────────────────────────────────────────────
    logger.info(f"Loading restoration model: {args.restoration_model}")
    model = load_diffusion_model(
        args.restoration_model, device=device,
        in_channels=2, out_channels=1,
        compile_model=False, spatial_dims=3,
    )

    # ── Setup strategy ───────────────────────────────────────────────
    logger.info(f"Setting up {args.strategy} strategy...")
    strategy = load_strategy(
        args.strategy, args.num_timesteps, args.image_size, args.depth,
    )

    # ── Restore volumes ──────────────────────────────────────────────
    t_start = time.time()
    restored_volumes = []

    for idx, fpath in enumerate(files):
        logger.info(f"[{idx + 1}/{len(files)}] Processing {fpath.name}...")

        vol = load_volume(fpath, args.depth)
        vol_t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).to(device)

        restored_t = restore_volume(model, strategy, vol_t, args.num_steps, device)

        restored_np = restored_t.squeeze().cpu().numpy()  # [D, H, W]
        restored_volumes.append(restored_np)

        # Save restored volume
        # Determine output path: mirror input structure
        if fpath.parent != input_dir:
            # Nested: input_dir/patient_id/bravo.nii.gz
            patient_out = output_dir / fpath.parent.name
            patient_out.mkdir(parents=True, exist_ok=True)
            out_path = patient_out / fpath.name
        else:
            # Flat
            out_path = output_dir / fpath.name

        save_nifti(np.transpose(restored_np, (1, 2, 0)), str(out_path))

        if (idx + 1) % 10 == 0:
            logger.info(f"  {idx + 1}/{len(files)} volumes done")

    elapsed = time.time() - t_start
    logger.info(f"\nRestoration complete: {len(files)} volumes in {elapsed:.0f}s "
                f"({elapsed / len(files):.1f}s per volume)")

    # ── Optional evaluation ──────────────────────────────────────────
    if args.eval and args.data_root:
        logger.info("\nComputing evaluation metrics...")

        try:
            from medgen.metrics.fwd import compute_fwd_3d

            # Load real volumes for comparison
            data_root = Path(args.data_root)
            real_volumes = []
            for split in ['test_new', 'test', 'val']:
                split_dir = data_root / split
                if not split_dir.exists():
                    continue
                for fp in sorted(split_dir.glob("*/bravo.nii.gz")):
                    real_volumes.append(load_volume(fp, args.depth))
                if real_volumes:
                    break

            if real_volumes:
                logger.info(f"Loaded {len(real_volumes)} real volumes for FWD")
                fwd, fwd_bands = compute_fwd_3d(
                    real_volumes, restored_volumes,
                    trim_slices=10, max_level=4,
                )

                n_bands = len(fwd_bands)
                quarter = n_bands // 4
                vals = list(fwd_bands.values())
                fwd_low = float(np.mean(vals[:quarter]))
                fwd_mid = float(np.mean(vals[quarter:3 * quarter]))
                fwd_high = float(np.mean(vals[3 * quarter:]))

                eval_results = {
                    'fwd': fwd,
                    'fwd_low': fwd_low,
                    'fwd_mid': fwd_mid,
                    'fwd_high': fwd_high,
                    'num_restored': len(restored_volumes),
                    'num_real': len(real_volumes),
                    'strategy': args.strategy,
                    'num_steps': args.num_steps,
                }
                logger.info(f"FWD={fwd:.4f} (low={fwd_low:.4f} mid={fwd_mid:.4f} high={fwd_high:.4f})")

                with open(output_dir / "eval_results.json", "w") as f:
                    json.dump(eval_results, f, indent=2)
            else:
                logger.warning("No real volumes found for evaluation")

        except ImportError as e:
            logger.warning(f"Could not compute metrics: {e}")

    logger.info(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
