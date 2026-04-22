#!/usr/bin/env python3
"""SR feasibility check — does real_ds_us spectrum match generated-output spectrum?

Before building a 3D super-resolution cascade trained on (real, real_downsampled_upsampled)
pairs, verify the core assumption: that the degradation produced by bicubic
downsample→upsample is *spectrally similar* to what the generator (exp1_1_1000 /
exp37_3) produces at inference. If yes, a SR model trained on the synthetic
degradation will transfer. If no, we're back in exp33 territory.

Process:
  1. Load N real volumes from test1.
  2. For each real volume, produce two degraded variants:
       - real_ds2: downsample by 2× (trilinear), upsample back to 256³×160
       - real_ds4: downsample by 4×, upsample back
  3. Load N generated volumes from compare_imagenet_optima (exp1_1_1000 + exp37_3).
  4. Compute, for every variant, per-band spectral energy (ratio to real).
  5. Compute Frangi vessel mean.
  6. Visualize axial mid-slice of each variant side-by-side.
  7. Report: numeric comparison table, visual figure, spectral overlay.

Decision rule (rough):
  - If real_ds2 band ratios match exp1_1_1000 ± ~0.05 across mid/high/very_high → viable
  - If real_ds2 overshoots (close to 1.0) while generators undershoot (~0.6) → SR would
    train on degradations the network never sees at inference
  - If real_ds4 matches better than real_ds2 → need more aggressive training degradation

Usage:
    python -m medgen.scripts.analyze_sr_feasibility \\
        --compare-dir /home/mode/NTNU/MedicalDataSets/generated/compare_imagenet_optima_20260421-165759 \\
        --real-dir /home/mode/NTNU/MedicalDataSets/brainmetshare-3/test1 \\
        --output-dir runs/eval/sr_feasibility \\
        --num-volumes 5
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
import torch.nn.functional as F
from scipy.ndimage import binary_fill_holes
from skimage.filters import frangi

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


BRAIN_THRESHOLD = 0.05
FRANGI_SIGMAS = (0.5, 1.0, 1.5, 2.0, 3.0)

BANDS: dict[str, tuple[float, float]] = {
    'very_low':  (0.00, 0.05),
    'low':       (0.05, 0.10),
    'low_mid':   (0.10, 0.20),
    'mid':       (0.20, 0.30),
    'high':      (0.30, 0.40),
    'very_high': (0.40, 0.50),
}

# Variants we compare. Tuple of (key, human-readable label, color).
VARIANTS = [
    ('real',          'real',                              'black'),
    ('trilinear3d_f2', 'real → trilinear3d (f2)',         'tab:green'),
    ('axial2d_f2',    'real → axial2d (f2, D preserved)', 'tab:olive'),
    ('pseudo3d_f2',   'real → pseudo3d (seq 2D, f2)',     'tab:cyan'),
    ('exp1_1_1000',   'exp1_1_1000 (baseline gen)',        'gray'),
    ('exp37_3',       'exp37_3 (best gen)',                'tab:brown'),
]


def load_volume(path: Path, depth: int = 160) -> np.ndarray:
    vol = nib.load(str(path)).get_fdata().astype(np.float32)
    vmin, vmax = vol.min(), vol.max()
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)
    vol = np.transpose(vol, (2, 0, 1))
    d = vol.shape[0]
    if d < depth:
        vol = np.pad(vol, ((0, depth - d), (0, 0), (0, 0)))
    elif d > depth:
        vol = vol[:depth]
    return vol


def downsample_upsample_trilinear3d(vol: np.ndarray, factor: float) -> np.ndarray:
    """True 3D trilinear downsample + upsample on 5D tensor."""
    t = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).float()  # [1, 1, D, H, W]
    down = F.interpolate(t, scale_factor=1.0 / factor, mode='trilinear', align_corners=False)
    up = F.interpolate(down, size=t.shape[2:], mode='trilinear', align_corners=False)
    return up.squeeze(0).squeeze(0).numpy()


def downsample_upsample_axial2d(vol: np.ndarray, factor: float) -> np.ndarray:
    """2D bilinear on each axial (H,W) slice. D dimension untouched."""
    D, H, W = vol.shape
    t = torch.from_numpy(vol).unsqueeze(1).float()  # [D, 1, H, W]
    down = F.interpolate(t, scale_factor=1.0 / factor, mode='bilinear', align_corners=False)
    up = F.interpolate(down, size=(H, W), mode='bilinear', align_corners=False)
    return up.squeeze(1).numpy()


def downsample_upsample_pseudo3d(vol: np.ndarray, factor: float) -> np.ndarray:
    """Sequential 2D bilinear ds/us in axial, coronal, sagittal orientations.

    HF loss in all three directions via 2D operations only — approximates 3D
    downsampling without true 3D mixing at the numerical interpolation step.
    """
    D, H, W = vol.shape
    x = torch.from_numpy(vol).float()
    # axial planes: downsample H, W
    t1 = x.unsqueeze(1)
    x = F.interpolate(
        F.interpolate(t1, scale_factor=1.0 / factor, mode='bilinear', align_corners=False),
        size=(H, W), mode='bilinear', align_corners=False,
    ).squeeze(1)
    # coronal planes: downsample D, W
    t2 = x.permute(1, 0, 2).unsqueeze(1)
    x = F.interpolate(
        F.interpolate(t2, scale_factor=1.0 / factor, mode='bilinear', align_corners=False),
        size=(D, W), mode='bilinear', align_corners=False,
    ).squeeze(1).permute(1, 0, 2)
    # sagittal planes: downsample D, H
    t3 = x.permute(2, 0, 1).unsqueeze(1)
    x = F.interpolate(
        F.interpolate(t3, scale_factor=1.0 / factor, mode='bilinear', align_corners=False),
        size=(D, H), mode='bilinear', align_corners=False,
    ).squeeze(1).permute(1, 2, 0)
    return x.numpy()


def band_energies(vol: np.ndarray) -> dict[str, float]:
    fft = np.fft.fftshift(np.fft.fftn(vol))
    power = np.abs(fft) ** 2
    d, h, w = vol.shape
    cd, ch, cw = d // 2, h // 2, w // 2
    dz, dy, dx = np.ogrid[-cd:d - cd, -ch:h - ch, -cw:w - cw]
    radius = np.sqrt((dz / d) ** 2 + (dy / h) ** 2 + (dx / w) ** 2)
    return {
        name: float(power[(radius >= lo) & (radius < hi)].sum())
        for name, (lo, hi) in BANDS.items()
    }


def brain_mask(vol: np.ndarray, thresh: float = BRAIN_THRESHOLD) -> np.ndarray:
    m = (vol > thresh).astype(np.uint8)
    out = np.zeros_like(m)
    for i in range(m.shape[0]):
        out[i] = binary_fill_holes(m[i])
    return out.astype(bool)


def vessel_mean(vol: np.ndarray) -> float:
    mask = brain_mask(vol)
    if not mask.any():
        return 0.0
    v = frangi(vol, sigmas=FRANGI_SIGMAS, alpha=0.5, beta=0.5,
               gamma=None, black_ridges=False).astype(np.float32)
    return float(v[mask].mean())


def find_bravo_files(root: Path, max_volumes: int) -> list[Path]:
    files = sorted(root.glob("*/bravo.nii.gz"))
    if not files:
        files = sorted(root.glob("*.nii.gz"))
    return files[:max_volumes]


def analyze_one(vol: np.ndarray) -> dict:
    be = band_energies(vol)
    return {'band_energies': be, 'vessel_mean': vessel_mean(vol)}


def aggregate_ratios(variants: dict[str, list[dict]],
                     real_bands_mean: dict[str, float]) -> dict[str, dict[str, float]]:
    """For each variant, report mean band-energy RATIO to real mean."""
    out: dict[str, dict[str, float]] = {}
    for key, stats in variants.items():
        ratios: dict[str, list[float]] = {b: [] for b in BANDS}
        for s in stats:
            for b in BANDS:
                if real_bands_mean[b] > 0:
                    ratios[b].append(s['band_energies'][b] / real_bands_mean[b])
        out[key] = {
            b: {
                'mean': float(np.mean(ratios[b])),
                'std': float(np.std(ratios[b])),
            } for b in BANDS
        }
        ves = [s['vessel_mean'] for s in stats]
        out[key]['vessel_mean'] = {'mean': float(np.mean(ves)), 'std': float(np.std(ves))}
    return out


def plot_band_bars(agg: dict, output_base: Path) -> None:
    band_names = list(BANDS.keys())
    variant_keys = [v[0] for v in VARIANTS]
    variant_labels = [v[1] for v in VARIANTS]
    colors = [v[2] for v in VARIANTS]

    means = np.zeros((len(variant_keys), len(band_names)))
    stds = np.zeros((len(variant_keys), len(band_names)))
    for i, k in enumerate(variant_keys):
        if k not in agg:
            continue
        for j, b in enumerate(band_names):
            means[i, j] = agg[k][b]['mean']
            stds[i, j] = agg[k][b]['std']

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(band_names))
    width = 0.9 / max(len(variant_keys), 1)
    for i, k in enumerate(variant_keys):
        if k not in agg:
            continue
        ax.bar(x + (i - len(variant_keys) / 2 + 0.5) * width, means[i],
               yerr=stds[i], capsize=3, width=width,
               label=variant_labels[i], color=colors[i], alpha=0.85,
               edgecolor='black', linewidth=0.4)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.6,
               label='real = 1.0')
    ax.set_xticks(x)
    ax.set_xticklabels(band_names)
    ax.set_ylabel('per-band energy ratio to real (mean ± std)')
    ax.set_title('SR feasibility — does real_ds_us spectrum match generator output?')
    ax.grid(True, axis='y', linestyle='-', linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc='best', fontsize=9, ncol=2)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'), dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


def plot_vessel_bar(agg: dict, output_base: Path) -> None:
    variant_keys = [v[0] for v in VARIANTS if v[0] in agg]
    variant_labels = [v[1] for v in VARIANTS if v[0] in agg]
    colors = [v[2] for v in VARIANTS if v[0] in agg]

    means = [agg[k]['vessel_mean']['mean'] for k in variant_keys]
    stds = [agg[k]['vessel_mean']['std'] for k in variant_keys]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(variant_keys))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85,
           edgecolor='black', linewidth=0.4)
    if 'real' in agg:
        ax.axhline(agg['real']['vessel_mean']['mean'], color='black',
                   linestyle='--', linewidth=0.8, alpha=0.6, label='real')
        ax.legend(loc='best', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(variant_labels, rotation=25, ha='right')
    ax.set_ylabel('Mean Frangi (inside brain)')
    ax.set_title('SR feasibility — vessel mean comparison')
    ax.grid(True, axis='y', linestyle='-', linewidth=0.3, alpha=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'), dpi=200, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


def render_visual_grid(
    examples: dict[str, np.ndarray],
    output_base: Path,
) -> None:
    """One axial mid-slice per variant, side-by-side."""
    keys = [v[0] for v in VARIANTS if v[0] in examples]
    labels = [v[1] for v in VARIANTS if v[0] in examples]
    fig, axes = plt.subplots(1, len(keys), figsize=(3.2 * len(keys), 3.5), squeeze=False)
    for ax, k, lab in zip(axes[0], keys, labels):
        vol = examples[k]
        zmid = vol.shape[0] // 2
        ax.imshow(vol[zmid], cmap='gray', vmin=0, vmax=1)
        ax.set_title(lab, fontsize=9)
        ax.axis('off')
    plt.suptitle('Axial mid-slice comparison', fontsize=11)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'), dpi=180, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


def render_difference_maps(
    examples: dict[str, np.ndarray],
    output_base: Path,
) -> None:
    """Residual = real - variant, for each non-real variant. Highlights WHERE HF is lost."""
    if 'real' not in examples:
        return
    real = examples['real']
    zmid = real.shape[0] // 2
    others = [(k, lab) for k, lab, _ in VARIANTS if k != 'real' and k in examples]
    fig, axes = plt.subplots(2, len(others), figsize=(3.0 * len(others), 6), squeeze=False)
    for col, (k, lab) in enumerate(others):
        variant = examples[k]
        # Top row: variant slice
        axes[0, col].imshow(variant[zmid], cmap='gray', vmin=0, vmax=1)
        axes[0, col].set_title(lab, fontsize=9)
        axes[0, col].axis('off')
        # Bottom row: |real - variant| amplified
        diff = np.abs(real[zmid] - variant[zmid])
        vmax = np.percentile(diff, 99)
        axes[1, col].imshow(diff, cmap='hot', vmin=0, vmax=max(vmax, 0.01))
        axes[1, col].set_title(f'|real - variant|  (p99={vmax:.3f})', fontsize=8)
        axes[1, col].axis('off')
    plt.suptitle('Difference maps (hot = where HF content was lost)', fontsize=11)
    plt.tight_layout()
    for ext in ('png', 'pdf'):
        plt.savefig(output_base.with_suffix(f'.{ext}'), dpi=180, bbox_inches='tight')
    plt.close()
    logger.info(f"  saved {output_base.name}.png + .pdf")


def print_summary_table(agg: dict) -> None:
    print()
    print("=" * 120)
    print("SR feasibility — band-energy ratio to real (mean ± std)")
    print("=" * 120)
    header = f"{'variant':<30}" + "".join(f"{b:>14}" for b in BANDS) + f"{'vessel_mean':>14}"
    print(header)
    print("-" * len(header))
    for key, label, _ in VARIANTS:
        if key not in agg:
            continue
        row = f"{label:<30}"
        for b in BANDS:
            m = agg[key][b]['mean']
            s = agg[key][b]['std']
            row += f" {m:>6.3f}±{s:.3f}"
        vm = agg[key]['vessel_mean']['mean']
        vs = agg[key]['vessel_mean']['std']
        row += f"  {vm:>6.4f}±{vs:.4f}"
        print(row)
    print("=" * 120)
    # Decision: which f2 method best matches exp1_1_1000 at very_high band?
    print()
    if 'exp1_1_1000' in agg:
        hf_gen = agg['exp1_1_1000']['very_high']['mean']
        print("DECISION METRIC (very_high band): which f2 method best matches exp1_1_1000?")
        print(f"  exp1_1_1000:      {hf_gen:.3f}  (target degradation we want to reproduce)")
        candidates = []
        for key, label in [('trilinear3d_f2', 'trilinear3d_f2'),
                           ('axial2d_f2', 'axial2d_f2'),
                           ('pseudo3d_f2', 'pseudo3d_f2')]:
            if key in agg:
                m = agg[key]['very_high']['mean']
                dist = abs(m - hf_gen)
                candidates.append((label, m, dist))
                print(f"  {label:<18}  {m:.3f}  |dist|={dist:.3f}")
        if candidates:
            best = min(candidates, key=lambda c: c[2])
            print(f"  ▶ Closest match: {best[0]} (|dist|={best[2]:.3f})")
            if best[2] < 0.15:
                print("  VERDICT: ✓ Viable — build SR training pairs with this method.")
            else:
                print("  VERDICT: ✗ Gap too large — this degradation family doesn't reproduce"
                      " the generator's phenotype.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="SR feasibility: real_ds_us vs generated spectrum")
    parser.add_argument('--compare-dir', required=True,
                        help='Dir with {exp1_1_1000, exp37_3}/XXXXX/bravo.nii.gz')
    parser.add_argument('--real-dir', required=True)
    parser.add_argument('--output-dir', default='runs/eval/sr_feasibility')
    parser.add_argument('--num-volumes', type=int, default=5)
    parser.add_argument('--depth', type=int, default=160)
    parser.add_argument('--thesis-dir',
                        default='/home/mode/NTNU/AIS4900_doc/AIS4900-master-thesis/Images/sr_feasibility')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    compare_root = Path(args.compare_dir)
    real_root = Path(args.real_dir)

    # Load real
    real_files = find_bravo_files(real_root, args.num_volumes)
    if not real_files:
        raise SystemExit(f"No real volumes in {real_root}")
    logger.info(f"Loading {len(real_files)} real volumes")
    real_vols = [load_volume(f, args.depth) for f in real_files]

    # Compute real band means (ground-truth reference)
    real_bands = [band_energies(v) for v in real_vols]
    real_bands_mean = {b: float(np.mean([rb[b] for rb in real_bands])) for b in BANDS}
    logger.info(f"Real band means: {real_bands_mean}")

    variants: dict[str, list[dict]] = {'real': [analyze_one(v) for v in real_vols]}

    # Compute three f2 degradation variants on-the-fly
    logger.info("Computing trilinear3d_f2, axial2d_f2, pseudo3d_f2 degradations")
    tri_vols = [downsample_upsample_trilinear3d(v, 2.0) for v in real_vols]
    ax_vols = [downsample_upsample_axial2d(v, 2.0) for v in real_vols]
    ps_vols = [downsample_upsample_pseudo3d(v, 2.0) for v in real_vols]
    variants['trilinear3d_f2'] = [analyze_one(v) for v in tri_vols]
    variants['axial2d_f2'] = [analyze_one(v) for v in ax_vols]
    variants['pseudo3d_f2'] = [analyze_one(v) for v in ps_vols]

    # Generated variants
    for model_key in ('exp1_1_1000', 'exp37_3'):
        gen_dir = compare_root / model_key
        if not gen_dir.is_dir():
            logger.warning(f"Skipping {model_key}: {gen_dir} not a dir")
            continue
        gen_files = find_bravo_files(gen_dir, args.num_volumes)
        if not gen_files:
            logger.warning(f"No volumes in {gen_dir}")
            continue
        logger.info(f"Loading {len(gen_files)} generated volumes from {model_key}")
        gen_vols = [load_volume(f, args.depth) for f in gen_files]
        variants[model_key] = [analyze_one(v) for v in gen_vols]

    # Aggregate
    agg = aggregate_ratios(variants, real_bands_mean)

    # Plots
    logger.info("Plotting")
    plot_band_bars(agg, output_dir / 'band_bars')
    plot_vessel_bar(agg, output_dir / 'vessel_bar')

    # Visuals — use the first volume of each variant
    examples = {
        'real': real_vols[0],
        'trilinear3d_f2': tri_vols[0],
        'axial2d_f2': ax_vols[0],
        'pseudo3d_f2': ps_vols[0],
    }
    for model_key in ('exp1_1_1000', 'exp37_3'):
        gen_dir = compare_root / model_key
        gen_files = find_bravo_files(gen_dir, 1)
        if gen_files:
            examples[model_key] = load_volume(gen_files[0], args.depth)
    render_visual_grid(examples, output_dir / 'visual_grid')
    render_difference_maps(examples, output_dir / 'difference_maps')

    print_summary_table(agg)

    with open(output_dir / 'sr_feasibility_results.json', 'w') as f:
        json.dump({
            'compare_dir': str(compare_root),
            'real_dir': str(real_root),
            'num_volumes': args.num_volumes,
            'real_band_energies_mean': real_bands_mean,
            'aggregate_band_ratios': agg,
        }, f, indent=2)
    logger.info(f"Saved: {output_dir / 'sr_feasibility_results.json'}")

    if args.thesis_dir:
        try:
            thesis_dir = Path(args.thesis_dir)
            thesis_dir.mkdir(parents=True, exist_ok=True)
            for name in ('band_bars', 'vessel_bar', 'visual_grid', 'difference_maps'):
                for ext in ('png', 'pdf'):
                    src = output_dir / f'{name}.{ext}'
                    if src.exists():
                        (thesis_dir / f'{name}.{ext}').write_bytes(src.read_bytes())
            logger.info(f"Copied figures to thesis dir: {thesis_dir}")
        except OSError as e:
            logger.warning(f"Could not copy to thesis dir {args.thesis_dir}: {e}")


if __name__ == '__main__':
    main()
