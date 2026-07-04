"""Sample-to-sample diversity (MS-SSIM-3D / perceptual-3D) over one or more volume sets.

Measures intra-set generation diversity: how different the volumes within a set
are from each other. Higher = more diverse (less mode collapse). This is the
same-slice pairwise formulation from ``metrics.quality.compute_msssim_diversity_3d``
(compare slice d across volumes, average over d and over all pairs), but:

  1. It runs over a whole DIRECTORY of NIfTI volumes, not a validation batch.
  2. Pairs are batched into a single ``compute_msssim`` call per slice instead of
     looped one-at-a-time, so 100s of volumes are tractable.
  3. Because raw mean-pairwise-MS-SSIM is an UNBIASED estimator of a distributional
     quantity but its VARIANCE grows as sets shrink, sets of unequal size are NOT
     directly comparable by their point estimate. So we subsample every set to a
     common N and bootstrap, reporting mean ± 95% CI at equal N. The raw full-pool
     number (N-confounded) is also reported, clearly labelled.

Each dataset is a directory searched recursively for ``--pattern`` (default
``bravo.nii.gz``); this matches both the generated layout (``<NNNNN>/bravo.nii.gz``)
and the real layout (``Mets_XXX/bravo.nii.gz``). Volumes are per-volume min-max
normalized to [0, 1] (same as training ``ScaleIntensity``) and resized to a common
shape so cross-set comparison is at one resolution.

Example:
    python -m medgen.scripts.eval_diversity \
        --dataset real_train:$HOME/MedicalDataSets/brainmetshare-3/train \
        --dataset real_test:$HOME/MedicalDataSets/brainmetshare-3/test1 \
        --dataset exp1_1_imagenet:$HOME/MedicalDataSets/generated/exp1_1_bravo_imagenet_525 \
        --output diversity_report.json
"""

import argparse
import glob
import json
import logging
import os

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from medgen.metrics.quality import compute_msssim, compute_perceptual_distance

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_diversity")


# =============================================================================
# Loading
# =============================================================================

def find_volumes(dataset_dir: str, pattern: str) -> list[str]:
    """Recursively find volume files under ``dataset_dir`` matching ``pattern``."""
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    hits = sorted(glob.glob(os.path.join(dataset_dir, "**", pattern), recursive=True))
    # Also allow the pattern to match files directly in the top level.
    hits += sorted(glob.glob(os.path.join(dataset_dir, pattern)))
    return sorted(set(hits))


def load_volume(path: str, target_shape: tuple[int, int, int]) -> torch.Tensor:
    """Load one NIfTI volume as [1, D, H, W] float32, [0, 1]-normalized, resized.

    NIfTI data is (X, Y, Z); Z is treated as depth D. Per-volume min-max to [0, 1]
    (matching training ScaleIntensity). Resized to ``target_shape`` (D, H, W) only
    if it does not already match.
    """
    vol = nib.load(path).get_fdata().astype(np.float32)  # (X, Y, Z)
    vmin, vmax = float(vol.min()), float(vol.max())
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)
    else:
        vol = np.zeros_like(vol)
    # (X, Y, Z) -> (Z, X, Y) = (D, H, W)
    vol = np.transpose(vol, (2, 0, 1))
    t = torch.from_numpy(np.ascontiguousarray(vol))[None, None]  # [1, 1, D, H, W]
    if tuple(t.shape[2:]) != tuple(target_shape):
        t = F.interpolate(t, size=target_shape, mode="trilinear", align_corners=False)
    return t[0]  # [1, D, H, W]


def load_dataset(
    dataset_dir: str,
    pattern: str,
    target_shape: tuple[int, int, int],
    pool_cap: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, int]:
    """Load up to ``pool_cap`` volumes from a dataset into a CPU tensor.

    Returns (volumes [K, 1, D, H, W] on CPU, total_found). If the set has more
    than ``pool_cap`` volumes, a random ``pool_cap`` subset is loaded and the cap
    is logged (never silently truncated).
    """
    paths = find_volumes(dataset_dir, pattern)
    total_found = len(paths)
    if total_found < 2:
        raise ValueError(
            f"Need >=2 volumes for diversity, found {total_found} matching "
            f"'{pattern}' under {dataset_dir}"
        )
    if total_found > pool_cap:
        keep = rng.choice(total_found, size=pool_cap, replace=False)
        paths = [paths[i] for i in sorted(keep)]
        logger.warning(
            "%s: %d volumes found, loading a random %d into memory (--pool-cap). "
            "Diversity is estimated from this pool.",
            dataset_dir, total_found, pool_cap,
        )
    vols = [load_volume(p, target_shape) for p in tqdm(paths, desc=f"load {os.path.basename(dataset_dir.rstrip('/'))}")]
    return torch.stack(vols), total_found  # [K, 1, D, H, W]


# =============================================================================
# Diversity computation
# =============================================================================

def _pair_indices(n: int) -> tuple[np.ndarray, np.ndarray]:
    """All unique unordered pairs (i, j), i < j, for n items."""
    i, j = np.triu_indices(n, k=1)
    return i, j


@torch.no_grad()
def pairwise_diversity_3d(
    volumes: torch.Tensor,
    device: torch.device,
    scorer,
    is_similarity: bool,
    depth_stride: int = 1,
    pair_chunk: int = 512,
) -> float:
    """Mean pairwise diversity over same-slice pairs. Batched, GPU-friendly.

    Same-slice pairwise formulation (compare slice d across volumes, average over d
    and over all pairs), but all pairs of a slice are batched into ONE ``scorer``
    call instead of the per-pair Python loop in metrics.quality — so 100s of volumes
    are tractable for both MS-SSIM and a perceptual network.

    Args:
        volumes: [B, 1, D, H, W] in [0, 1], on CPU (slices moved to device per depth).
        device: compute device.
        scorer: ``scorer(gen[P,C,H,W], ref[P,C,H,W]) -> float`` returning the MEAN
            statistic over the batch of P pairs.
        is_similarity: True if the statistic is a SIMILARITY in [0, 1] (MS-SSIM) —
            diversity = 1 - mean. False if it is a DISTANCE >= 0 (LPIPS/perceptual) —
            diversity = mean (the distance itself; higher = more diverse). Do NOT
            apply `1 -` to a distance.
        depth_stride: evaluate every k-th depth slice (>=1). Larger = faster, coarser.
        pair_chunk: number of pairs per scorer call (bounds VRAM).

    Returns:
        Mean diversity across all evaluated (pair, slice) combinations.
    """
    B, C, D, H, W = volumes.shape
    ii, jj = _pair_indices(B)
    num_pairs = len(ii)
    if num_pairs == 0:
        return 0.0

    stat_sum = 0.0
    stat_count = 0
    for d in range(0, D, depth_stride):
        sl = volumes[:, :, d].to(device)  # [B, C, H, W]
        for start in range(0, num_pairs, pair_chunk):
            pi = ii[start:start + pair_chunk]
            pj = jj[start:start + pair_chunk]
            mean_stat = scorer(sl[pi], sl[pj])  # mean over the P pairs
            n = len(pi)
            stat_sum += mean_stat * n
            stat_count += n
    mean_stat = stat_sum / stat_count if stat_count else 0.0
    return (1.0 - mean_stat) if is_similarity else mean_stat


@torch.no_grad()
def diversity_with_ci(
    volumes: torch.Tensor,
    device: torch.device,
    subsample_n: int,
    bootstrap: int,
    depth_stride: int,
    pair_chunk: int,
    rng: np.random.Generator,
    metric: str = "msssim",
    network_type: str = "alex",
    compute_full: bool = True,
) -> dict:
    """Full-pool diversity plus subsample+bootstrap mean and 95% CI at common N.

    The full-pool number uses every loaded volume (its N is dataset-dependent, so
    it is NOT comparable across sets of different size — reported for transparency).
    The bootstrap number draws ``subsample_n`` volumes ``bootstrap`` times and
    averages, giving an equal-N, comparable estimate with a confidence interval.

    metric='perceptual' scores mean pairwise perceptual DISTANCE (higher = more
    diverse) using ``network_type`` (alex/vgg/squeeze = true LPIPS; radimagenet_* =
    RadImageNet feature distance, NOT LPIPS). compute_full is skipped by default for
    perceptual because the full pool (all pairs at N up to pool_cap) is far too
    expensive for a network scorer — only the equal-N bootstrap is reported.
    """
    B = volumes.shape[0]

    if metric == "perceptual":
        def scorer(gen, ref):
            return compute_perceptual_distance(
                gen, ref, device=device, network_type=network_type, use_compile=False
            )
        is_similarity = False
    else:
        def scorer(gen, ref):
            return compute_msssim(gen, ref, data_range=1.0, spatial_dims=2)
        is_similarity = True

    def _score(vols: torch.Tensor) -> float:
        return pairwise_diversity_3d(vols, device, scorer, is_similarity, depth_stride, pair_chunk)

    full = _score(volumes) if compute_full else None

    n = min(subsample_n, B)
    draws = []
    if n >= 2 and bootstrap > 0:
        for _ in tqdm(range(bootstrap), desc=f"bootstrap n={n}", leave=False):
            idx = rng.choice(B, size=n, replace=False)
            draws.append(_score(volumes[idx]))
    draws = np.asarray(draws, dtype=np.float64)

    result = {
        "metric": metric,
        "network_type": network_type if metric == "perceptual" else None,
        "pool_size": int(B),
        "full_pool_diversity": float(full) if full is not None else None,
        "subsample_n": int(n),
        "bootstrap_draws": len(draws),
    }
    if len(draws):
        result.update(
            {
                "diversity_mean": float(draws.mean()),
                "diversity_std": float(draws.std(ddof=1)) if len(draws) > 1 else 0.0,
                "diversity_ci95_low": float(np.percentile(draws, 2.5)),
                "diversity_ci95_high": float(np.percentile(draws, 97.5)),
            }
        )
    return result


# =============================================================================
# CLI
# =============================================================================

def parse_dataset_arg(spec: str) -> tuple[str, str]:
    """Parse ``LABEL:PATH`` (label may be omitted -> basename of path).

    Split on the first colon. On Linux an absolute path has no colon, so
    ``LABEL:/abs/path`` splits cleanly and a bare ``/abs/path`` gets a
    basename label.
    """
    if ":" in spec:
        label, path = spec.split(":", 1)
    else:
        path = spec
        label = os.path.basename(path.rstrip("/"))
    return label, os.path.expanduser(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--dataset", action="append", required=True, metavar="LABEL:PATH",
        help="Dataset to evaluate as LABEL:PATH (repeatable). Label optional.",
    )
    ap.add_argument("--pattern", default="bravo.nii.gz", help="Filename glob searched recursively per dataset (default bravo.nii.gz).")
    ap.add_argument("--target-shape", nargs=3, type=int, default=[150, 256, 256], metavar=("D", "H", "W"), help="Common (D H W) all volumes are resized to.")
    ap.add_argument("--pool-cap", type=int, default=256, help="Max volumes loaded into memory per set (random subset if more).")
    ap.add_argument("--subsample-n", type=int, default=0, help="Common N for the comparable estimate. 0 = min set size across all datasets.")
    ap.add_argument("--bootstrap", type=int, default=20, help="Bootstrap resamples for the CI (0 = skip, full-pool only).")
    ap.add_argument("--depth-stride", type=int, default=1, help="Evaluate every k-th depth slice (>=1). Larger = faster, coarser.")
    ap.add_argument("--pair-chunk", type=int, default=512, help="Pairs per scorer call (bounds VRAM). Use a smaller value (e.g. 128) for perceptual.")
    ap.add_argument("--metric", choices=["msssim", "perceptual"], default="msssim", help="Diversity metric (default msssim). 'perceptual' = mean pairwise perceptual distance (see --network-type).")
    ap.add_argument("--network-type", default="alex", help="Perceptual backbone (metric=perceptual only): alex/vgg/squeeze = true LPIPS (default alex); radimagenet_resnet50 = RadImageNet distance (NOT LPIPS).")
    ap.add_argument("--full-pool", dest="full_pool", action="store_true", default=None, help="Force computing the N-confounded full-pool number (default: on for msssim, off for perceptual — too expensive).")
    ap.add_argument("--no-full-pool", dest="full_pool", action="store_false", help="Skip the full-pool number.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for pool/bootstrap sampling.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", default="diversity_report.json", help="Where to write the JSON report.")
    args = ap.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    target_shape = tuple(args.target_shape)
    datasets = [parse_dataset_arg(s) for s in args.dataset]

    # Full-pool default: on for msssim (cheap), off for perceptual (network all-pairs
    # at N up to pool_cap is intractable). Explicit --full-pool/--no-full-pool wins.
    compute_full = (args.metric == "msssim") if args.full_pool is None else args.full_pool

    metric_label = args.metric if args.metric != "perceptual" else f"perceptual:{args.network_type}"
    logger.info("Device=%s  target_shape=%s  metric=%s  full_pool=%s", device, target_shape, metric_label, compute_full)

    # Pass 1: count files only (cheap glob) so the common subsample N can default to
    # the smallest set WITHOUT holding every dataset in RAM at once. Pass 2 loads,
    # scores, and frees one dataset at a time (8 sets x 256 vols would be ~80 GB).
    counts = {}
    for label, path in datasets:
        counts[label] = len(find_volumes(path, args.pattern))
        if counts[label] < 2:
            raise ValueError(f"{label}: need >=2 volumes, found {counts[label]} ({path})")
    effective_pools = {label: min(c, args.pool_cap) for label, c in counts.items()}
    subsample_n = args.subsample_n or min(effective_pools.values())
    logger.info("Found counts: %s | common subsample N = %d", counts, subsample_n)

    report = {
        "config": {
            "metric": args.metric,
            "network_type": args.network_type if args.metric == "perceptual" else None,
            "target_shape": list(target_shape),
            "pool_cap": args.pool_cap,
            "subsample_n": subsample_n,
            "bootstrap": args.bootstrap,
            "depth_stride": args.depth_stride,
            "pattern": args.pattern,
            "seed": args.seed,
        },
        "datasets": {},
    }

    # Pass 2: load + score + free one dataset at a time.
    for label, path in datasets:
        logger.info("Loading %-28s <- %s", label, path)
        vols, total_found = load_dataset(path, args.pattern, target_shape, args.pool_cap, rng)
        logger.info("Scoring %s (pool=%d, found=%d)", label, vols.shape[0], total_found)
        res = diversity_with_ci(
            vols, device, subsample_n, args.bootstrap,
            args.depth_stride, args.pair_chunk, rng, metric=args.metric,
            network_type=args.network_type, compute_full=compute_full,
        )
        res["total_found"] = total_found
        report["datasets"][label] = res
        del vols
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Console table.
    print(f"\n=== Diversity report ({metric_label}), higher = more diverse ===")
    print(f"{'dataset':30s} {'found':>6s} {'pool':>5s} {'fullN':>8s}  {'mean@N':>8s} {'95% CI':>17s}")
    for label, res in report["datasets"].items():
        ci = (
            f"[{res['diversity_ci95_low']:.4f},{res['diversity_ci95_high']:.4f}]"
            if "diversity_mean" in res else "n/a"
        )
        mean = f"{res['diversity_mean']:.4f}" if "diversity_mean" in res else "n/a"
        full = f"{res['full_pool_diversity']:.4f}" if res["full_pool_diversity"] is not None else "n/a"
        print(
            f"{label:30s} {res['total_found']:6d} {res['pool_size']:5d} "
            f"{full:>8s}  {mean:>8s} {ci:>17s}"
        )
    print(f"\n(mean@N and CI are at common N={subsample_n}, {args.bootstrap} bootstrap draws — the comparable numbers)")

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
