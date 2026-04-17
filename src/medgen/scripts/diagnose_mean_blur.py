#!/usr/bin/env python3
"""Diagnose MSE-induced mean-blur via stochastic prediction diversity.

Three complementary tests:

1) **Multi-noise single-step diversity**: For a fixed real x₀, add N different
   noise realizations at each t. Predict x₀ from each x_t. Measure:
   - Pairwise diversity (pixel std across predictions)
   - Individual sharpness (high-freq energy of each prediction)
   - Mean vs individual sharpness gap (signature of mean-blur)

2) **Multi-step vs single-step denoising**: For a fixed real x₀ noised at t,
   compare direct single-Euler prediction `x₀ = x_t + t·v` against a chain
   of small Euler steps from t to 0. Do chained steps add blur?

3) **Generation trajectory**: Compare full-trajectory generation from pure
   noise vs a shortcut single-step "one-shot denoise" prediction from noise.
   How much does the full trajectory diverge from the posterior mean?

Outputs:
    - diversity_plot.png     (metric curves across t values)
    - visual_grid_t{T}.png   (N predictions side-by-side for visual check)
    - chain_vs_direct.png    (single vs multi-step comparison)
    - diagnose_blur.json     (numerical results)

Usage:
    python -m medgen.scripts.diagnose_mean_blur \
        --bravo-model runs/checkpoint_latest.pt \
        --data-root ~/MedicalDataSets/brainmetshare-3 \
        --output-dir runs/diagnose_blur \
        --n-noise 8 --n-volumes 5
"""
import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from torch.amp import autocast

from medgen.diffusion import RFlowStrategy, load_diffusion_model

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
)
logger = logging.getLogger(__name__)


DEFAULT_T_GRID = [0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.80]
VISUAL_T_VALUES = [0.05, 0.20, 0.50]  # Detailed visual grids at these t


def load_volume(path: Path, depth: int) -> np.ndarray:
    img = nib.load(str(path))
    vol = img.get_fdata().astype(np.float32)
    vmax = vol.max()
    if vmax > 0:
        vol /= vmax
    vol = np.transpose(vol, (2, 0, 1))
    if vol.shape[0] < depth:
        vol = np.pad(vol, ((0, depth - vol.shape[0]), (0, 0), (0, 0)))
    elif vol.shape[0] > depth:
        vol = vol[:depth]
    return vol


def discover_test_patients(
    data_root: Path, max_n: int,
) -> list[tuple[str, Path, Path]]:
    test_dir = data_root / 'test'
    if not test_dir.exists():
        test_dir = data_root / 'test_new'
    out = []
    for d in sorted(test_dir.iterdir()):
        if not d.is_dir():
            continue
        bravo = d / "bravo.nii.gz"
        seg = d / "seg.nii.gz"
        if bravo.exists() and seg.exists():
            out.append((d.name, bravo, seg))
            if len(out) >= max_n:
                break
    return out


def compute_hf_energy_ratio(volume: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Fraction of FFT energy above radial freq 0.25 (Nyquist/4)."""
    if mask is not None:
        volume = volume * mask
    fft = np.fft.fftn(volume)
    fft_shift = np.fft.fftshift(fft)
    power = np.abs(fft_shift) ** 2
    d, h, w = volume.shape
    cd, ch, cw = d // 2, h // 2, w // 2
    dz, dy, dx = np.ogrid[-cd:d - cd, -ch:h - ch, -cw:w - cw]
    radius = np.sqrt((dz / d) ** 2 + (dy / h) ** 2 + (dx / w) ** 2)
    high = radius > 0.25
    total = power.sum()
    if total <= 0:
        return 0.0
    return float(power[high].sum() / total)


def predict_single_step(
    model, strategy, x_0, seg, noise, t_norm, T, device,
) -> torch.Tensor:
    """Add noise at t and predict x_0 in one Euler step."""
    t_int = max(1, min(T - 1, round(t_norm * T)))
    t_tensor = torch.tensor([t_int], device=device, dtype=torch.long)
    x_t = strategy.scheduler.add_noise(x_0, noise, t_tensor)
    model_input = torch.cat([x_t, seg], dim=1)
    with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
        v_pred = model(model_input, t_tensor)
    v_pred = v_pred.float()
    x0_pred = strategy.compute_predicted_clean(x_t, v_pred, t_tensor)
    return x0_pred.clamp(0, 1)


def predict_chained(
    model, strategy, x_0, seg, noise, t_start_norm, n_steps, T, device,
) -> torch.Tensor:
    """Add noise at t_start, denoise via n_steps uniform Euler from t_start to 0."""
    t_start_int = max(1, min(T - 1, round(t_start_norm * T)))
    t_tensor = torch.tensor([t_start_int], device=device, dtype=torch.long)
    x_t = strategy.scheduler.add_noise(x_0, noise, t_tensor)

    # Uniform timestep sequence from t_start down to 0
    start_t = t_start_norm * T
    steps = torch.linspace(start_t, 0.0, n_steps + 1, device=device)
    all_t, all_next = steps[:-1], steps[1:]

    x = x_t
    # Need scheduler state set up
    d, h, w = x_0.shape[2], x_0.shape[3], x_0.shape[4]
    strategy.scheduler.set_timesteps(
        num_inference_steps=T, device=device,
        input_img_size_numel=d * h * w,
    )
    for t, next_t in zip(all_t, all_next):
        t_batch = t.unsqueeze(0).to(device)
        model_input = torch.cat([x, seg], dim=1)
        with torch.no_grad(), autocast('cuda', dtype=torch.bfloat16):
            velocity = model(model_input, t_batch)
        x, _ = strategy.scheduler.step(velocity.float(), t, x, next_t)
    return x.clamp(0, 1)


def save_visual_grid(
    real: torch.Tensor, preds: list[torch.Tensor],
    output_path: Path, title: str,
) -> None:
    """Save side-by-side grid: real + N predictions, with zoomed inset."""
    n_preds = len(preds)
    n_cols = n_preds + 1
    fig, axes = plt.subplots(2, n_cols, figsize=(3.2 * n_cols, 6.5))
    slice_idx = real.shape[2] // 2

    def _slice(x, i):
        # x: [1,1,D,H,W] → numpy 2D axial
        return x[0, 0, i].cpu().numpy()

    h, w = real.shape[3], real.shape[4]
    ch, cw = h // 2, w // 2
    crop = h // 4

    # Column 0: Real
    r = _slice(real, slice_idx)
    axes[0, 0].imshow(r, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title("Real x₀", fontsize=10, fontweight='bold')
    axes[0, 0].axis('off')
    axes[1, 0].imshow(r[ch-crop:ch+crop, cw-crop:cw+crop], cmap='gray', vmin=0, vmax=1)
    axes[1, 0].set_title("Real (zoom)", fontsize=9)
    axes[1, 0].axis('off')

    for i, p in enumerate(preds):
        ps = _slice(p, slice_idx)
        col = i + 1
        l1 = float(torch.abs(p - real).mean().item())
        axes[0, col].imshow(ps, cmap='gray', vmin=0, vmax=1)
        axes[0, col].set_title(f"Pred {i} (L1={l1:.4f})", fontsize=10)
        axes[0, col].axis('off')
        axes[1, col].imshow(ps[ch-crop:ch+crop, cw-crop:cw+crop], cmap='gray', vmin=0, vmax=1)
        axes[1, col].set_title(f"Pred {i} zoom", fontsize=9)
        axes[1, col].axis('off')

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=140, bbox_inches='tight')
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Mean-blur diagnostic for diffusion models")
    parser.add_argument("--bravo-model", type=str, required=True)
    parser.add_argument("--data-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--n-noise", type=int, default=8,
                        help="Noise realizations per (volume, t) for diversity test")
    parser.add_argument("--n-volumes", type=int, default=5,
                        help="Number of volumes to test (averages across them)")
    parser.add_argument("--depth", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--t-grid", nargs='+', type=float, default=DEFAULT_T_GRID)
    parser.add_argument("--n-chain-steps", type=int, default=32,
                        help="Euler steps for chained-denoising comparison")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda')

    # ── Load ──
    logger.info(f"Loading bravo model from {args.bravo_model}")
    model = load_diffusion_model(
        args.bravo_model, device=device, compile_model=False, spatial_dims=3,
    )
    model.eval()
    strategy = RFlowStrategy()
    strategy.setup_scheduler(num_timesteps=1000, image_size=256, depth_size=args.depth, spatial_dims=3)
    T = strategy.scheduler.num_train_timesteps

    data_root = Path(args.data_root)
    patients = discover_test_patients(data_root, args.n_volumes)
    logger.info(f"Found {len(patients)} test patients")

    # Cache volumes on CPU
    cached = []
    for pid, bp, sp in patients:
        clean = load_volume(bp, args.depth)
        seg = (load_volume(sp, args.depth) > 0.5).astype(np.float32)
        cached.append((pid, clean, seg))

    # ====================================================================
    # DIAGNOSTIC 1: Stochastic diversity at fixed x₀
    # ====================================================================
    logger.info("=" * 60)
    logger.info("Diagnostic 1: Multi-noise single-step diversity")
    logger.info("=" * 60)

    diversity_results = []
    for t_norm in args.t_grid:
        t_int = max(1, min(T - 1, round(t_norm * T)))
        per_vol = {
            'pred_std': [], 'pred_hf_mean': [], 'pred_hf_of_mean': [],
            'pred_l1_mean': [],
        }
        for vol_idx, (_pid, clean_np, seg_np) in enumerate(cached):
            clean = torch.from_numpy(clean_np).unsqueeze(0).unsqueeze(0).to(device)
            seg = torch.from_numpy(seg_np).unsqueeze(0).unsqueeze(0).to(device)
            mask_np = (clean_np > 0.02).astype(np.float32)

            preds_stack = []
            for i in range(args.n_noise):
                torch.manual_seed(args.seed + 10000 * vol_idx + 100 * t_int + i)
                noise = torch.randn_like(clean)
                pred = predict_single_step(model, strategy, clean, seg, noise, t_norm, T, device)
                preds_stack.append(pred.cpu())
                del noise, pred
                torch.cuda.empty_cache()

            # Stack: [N, 1, 1, D, H, W]
            stacked = torch.stack(preds_stack, dim=0)
            # Per-pixel std across predictions
            pred_std = stacked.std(dim=0).mean().item()
            # Individual high-freq energy (mean across the N predictions)
            hfs = [compute_hf_energy_ratio(p.squeeze().numpy(), mask_np) for p in preds_stack]
            pred_hf_mean = float(np.mean(hfs))
            # High-freq of the MEAN prediction (this would be low if mean-blur dominates)
            mean_pred = stacked.mean(dim=0).squeeze().numpy()
            pred_hf_of_mean = compute_hf_energy_ratio(mean_pred, mask_np)
            # L1 of each pred vs real
            l1s = [float(torch.abs(p - clean.cpu()).mean().item()) for p in preds_stack]
            pred_l1_mean = float(np.mean(l1s))

            per_vol['pred_std'].append(pred_std)
            per_vol['pred_hf_mean'].append(pred_hf_mean)
            per_vol['pred_hf_of_mean'].append(pred_hf_of_mean)
            per_vol['pred_l1_mean'].append(pred_l1_mean)

            # Save visual grid at selected t, first volume
            if t_norm in VISUAL_T_VALUES and vol_idx == 0:
                save_visual_grid(
                    clean, preds_stack[:5],
                    output_dir / f'visual_grid_t{t_norm:.2f}_{_pid}.png',
                    f'{_pid} — t={t_norm:.2f}: 5 predictions from different noise seeds',
                )

            del clean, seg, stacked, preds_stack
            torch.cuda.empty_cache()

        agg = {'t_norm': t_norm}
        for k, vals in per_vol.items():
            agg[f'{k}_mean'] = float(np.mean(vals))
            agg[f'{k}_std'] = float(np.std(vals))
        diversity_results.append(agg)
        logger.info(
            f"  t={t_norm:.3f}: pred_std={agg['pred_std_mean']:.5f}  "
            f"HF_individual={agg['pred_hf_mean_mean']:.4f}  "
            f"HF_of_mean={agg['pred_hf_of_mean_mean']:.4f}  "
            f"L1={agg['pred_l1_mean_mean']:.4f}"
        )

    # ====================================================================
    # DIAGNOSTIC 2: Single-step vs chained-Euler at fixed start t
    # ====================================================================
    logger.info("=" * 60)
    logger.info("Diagnostic 2: Single-step vs chained denoising")
    logger.info("=" * 60)

    chain_results = []
    for t_norm in args.t_grid:
        per_vol = {
            'single_hf': [], 'chain_hf': [], 'single_l1': [], 'chain_l1': [],
            'single_vs_chain_l1': [],
        }
        for vol_idx, (_pid, clean_np, seg_np) in enumerate(cached):
            clean = torch.from_numpy(clean_np).unsqueeze(0).unsqueeze(0).to(device)
            seg = torch.from_numpy(seg_np).unsqueeze(0).unsqueeze(0).to(device)
            mask_np = (clean_np > 0.02).astype(np.float32)

            torch.manual_seed(args.seed + vol_idx)
            noise = torch.randn_like(clean)

            # Single-step
            single = predict_single_step(model, strategy, clean, seg, noise, t_norm, T, device)

            # Chained
            chain = predict_chained(
                model, strategy, clean, seg, noise, t_norm,
                args.n_chain_steps, T, device,
            )

            s_hf = compute_hf_energy_ratio(single.squeeze().cpu().numpy(), mask_np)
            c_hf = compute_hf_energy_ratio(chain.squeeze().cpu().numpy(), mask_np)
            s_l1 = float(torch.abs(single - clean).mean().item())
            c_l1 = float(torch.abs(chain - clean).mean().item())
            sc_l1 = float(torch.abs(single - chain).mean().item())

            per_vol['single_hf'].append(s_hf)
            per_vol['chain_hf'].append(c_hf)
            per_vol['single_l1'].append(s_l1)
            per_vol['chain_l1'].append(c_l1)
            per_vol['single_vs_chain_l1'].append(sc_l1)

            del clean, seg, noise, single, chain
            torch.cuda.empty_cache()

        agg = {'t_norm': t_norm}
        for k, vals in per_vol.items():
            agg[f'{k}_mean'] = float(np.mean(vals))
            agg[f'{k}_std'] = float(np.std(vals))
        chain_results.append(agg)
        logger.info(
            f"  t={t_norm:.3f}: single_HF={agg['single_hf_mean']:.4f}  "
            f"chain_HF={agg['chain_hf_mean']:.4f}  "
            f"single_L1={agg['single_l1_mean']:.4f}  "
            f"chain_L1={agg['chain_l1_mean']:.4f}  "
            f"Δ_s-c={agg['single_vs_chain_l1_mean']:.4f}"
        )

    # ── Save numerical results ──
    with open(output_dir / 'diagnose_blur.json', 'w') as f:
        json.dump({
            'checkpoint': args.bravo_model,
            'n_noise': args.n_noise,
            'n_volumes': args.n_volumes,
            'n_chain_steps': args.n_chain_steps,
            't_grid': args.t_grid,
            'diversity': diversity_results,
            'chain_vs_single': chain_results,
        }, f, indent=2)

    # ── Plot: Diagnostic 1 ──
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ts = np.array([r['t_norm'] for r in diversity_results])

    # Panel A: Pixel-wise std across predictions (= diversity)
    means = np.array([r['pred_std_mean'] for r in diversity_results])
    stds = np.array([r['pred_std_std'] for r in diversity_results])
    axes[0, 0].errorbar(ts, means, yerr=stds, marker='o', linewidth=2, color='steelblue', capsize=3)
    axes[0, 0].set_xlabel('t (normalized noise level)')
    axes[0, 0].set_ylabel('Pixel-std across N predictions')
    axes[0, 0].set_title('Diversity: how much do predictions differ?\nLow = regression-to-mean')
    axes[0, 0].set_xscale('log')
    axes[0, 0].grid(True, alpha=0.3)

    # Panel B: HF of individual vs HF of mean-prediction
    hf_ind = np.array([r['pred_hf_mean_mean'] for r in diversity_results])
    hf_mean = np.array([r['pred_hf_of_mean_mean'] for r in diversity_results])
    axes[0, 1].plot(ts, hf_ind, 'o-', color='green', linewidth=2, label='Individual predictions (mean)')
    axes[0, 1].plot(ts, hf_mean, 'o-', color='red', linewidth=2, label='HF of mean-prediction')
    axes[0, 1].set_xlabel('t')
    axes[0, 1].set_ylabel('High-freq energy ratio')
    axes[0, 1].set_title('Sharpness: individual vs their mean\nGap = mean-blur amount')
    axes[0, 1].set_xscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Panel C: Diagnostic 2 — single vs chain HF
    ts_c = np.array([r['t_norm'] for r in chain_results])
    single_hf = np.array([r['single_hf_mean'] for r in chain_results])
    chain_hf = np.array([r['chain_hf_mean'] for r in chain_results])
    axes[1, 0].plot(ts_c, single_hf, 'o-', color='steelblue', linewidth=2, label='Single-step')
    axes[1, 0].plot(ts_c, chain_hf, 's-', color='darkorange', linewidth=2, label=f'Chained ({args.n_chain_steps} Euler steps)')
    axes[1, 0].set_xlabel('t')
    axes[1, 0].set_ylabel('High-freq energy ratio')
    axes[1, 0].set_title('Single-step vs chained: does multi-step add blur?')
    axes[1, 0].set_xscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Panel D: L1 gap between single and chained outputs
    sc_l1 = np.array([r['single_vs_chain_l1_mean'] for r in chain_results])
    axes[1, 1].plot(ts_c, sc_l1, 'o-', color='purple', linewidth=2)
    axes[1, 1].set_xlabel('t')
    axes[1, 1].set_ylabel('L1(single − chain)')
    axes[1, 1].set_title('How much does chaining change the output from single-step?')
    axes[1, 1].set_xscale('log')
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(
        f'Mean-Blur Diagnostics — exp1_1_1000 ({args.n_volumes} volumes × {args.n_noise} noise samples)',
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(str(output_dir / 'diagnose_blur_plot.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Summary ──
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    logger.info(f"Output: {output_dir}")
    logger.info("")
    logger.info("Interpretation guide:")
    logger.info("  - High pred_std + low HF_of_mean relative to HF_individual = mean-blur CONFIRMED")
    logger.info("  - Low pred_std = model is near-deterministic (ignores noise) — no mean-blur to 'fix'")
    logger.info("  - chain_HF << single_HF = compounding blur over trajectory — specialist loss would help")
    logger.info("  - chain_HF ≈ single_HF = single-step and chain are similar — per-step issue is the whole story")


if __name__ == '__main__':
    main()
