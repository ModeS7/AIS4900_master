"""Med3D whole-volume Gaussian-MMD of each set vs a real reference distribution.

Two-sample MMD in MedicalNet 3D ResNet-50 feature space (2048-d whole-volume): for
each dataset, MMD(real_reference, dataset). Lower = closer to the real distribution.
Uses the SAME primitives as training's ``Generation_3d/MED3D_MMD``:
``Med3DFeatures`` (input_size=None, native res) + ``compute_cmmd`` (RBF kernel), so the
numbers are consistent with the TensorBoard metric.

Companion to eval_diversity.py — same dataset layout (recursive ``bravo.nii.gz``),
[0,1] per-volume normalization, and the same equal-N bootstrap methodology so unequal
set sizes (51 / 105 / 525) do not confound the ranking. Two differences from diversity:
  * MMD is a TWO-sample distance, so a --reference (real) set is required.
  * The bootstrap resamples WITH replacement (proper bootstrap; MMD is robust to
    duplicate samples, unlike the pairwise-diversity statistic). This gives a valid CI
    even for a set whose size equals N. Matches FlowLet's "bootstrap resamples of the
    generated sets" protocol.

For comparability, the RBF bandwidth is fixed ONCE from the reference (median heuristic)
and reused for every dataset, rather than re-estimated per call.

Example:
    python -m medgen.scripts.eval_med3d_mmd \
        --reference real_train:$HOME/MedicalDataSets/brainmetshare-3/train \
        --dataset real_test:$HOME/MedicalDataSets/brainmetshare-3/test1 \
        --dataset exp1_1_imagenet:$HOME/MedicalDataSets/generated/exp1_1_bravo_imagenet_525 \
        --output med3d_mmd_report.json
"""

import argparse
import json
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from medgen.metrics.feature_extractors import Med3DFeatures
from medgen.metrics.generation import compute_cmmd
from medgen.scripts.eval_diversity import find_volumes, load_volume, parse_dataset_arg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_med3d_mmd")


# =============================================================================
# Feature extraction
# =============================================================================

@torch.no_grad()
def extract_med3d_features(
    paths: list[str],
    target_shape: tuple[int, int, int],
    extractor: Med3DFeatures,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    """Extract [N, 2048] Med3D features, streaming batches (only features retained).

    Volumes are loaded per-file, resized to ``target_shape``, [0,1]-normalized (via
    eval_diversity.load_volume), stacked into batches of ``batch_size``, and run
    through Med3DFeatures. Only the small feature tensors are kept in memory.
    """
    feats = []
    for start in tqdm(range(0, len(paths), batch_size), desc="med3d", leave=False):
        chunk = paths[start:start + batch_size]
        vols = torch.stack([load_volume(p, target_shape) for p in chunk])  # [B, 1, D, H, W]
        f = extractor.extract_features(vols.to(device))  # [B, 2048]
        feats.append(f.cpu())
    return torch.cat(feats, dim=0)


def load_features(
    dataset_dir: str,
    pattern: str,
    target_shape: tuple[int, int, int],
    extractor: Med3DFeatures,
    device: torch.device,
    batch_size: int,
    cap: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, int]:
    """Find volumes, extract Med3D features (up to ``cap``). Returns (feats, total_found)."""
    paths = find_volumes(dataset_dir, pattern)
    total_found = len(paths)
    if total_found < 2:
        raise ValueError(f"Need >=2 volumes, found {total_found} matching '{pattern}' under {dataset_dir}")
    if cap and total_found > cap:
        keep = sorted(rng.choice(total_found, size=cap, replace=False))
        paths = [paths[i] for i in keep]
        logger.warning("%s: %d found, extracting a random %d (--pool-cap).", dataset_dir, total_found, cap)
    return extract_med3d_features(paths, target_shape, extractor, device, batch_size), total_found


# =============================================================================
# MMD
# =============================================================================

def median_bandwidth(ref_feats: torch.Tensor, n_sample: int = 500) -> float:
    """Median-heuristic RBF bandwidth from L2-normalized reference features.

    Fixed once from the reference so every dataset's MMD uses the SAME kernel and the
    numbers are directly comparable (compute_cmmd would otherwise re-estimate it per
    call from a min(500, ...) subset of each pair).
    """
    x = F.normalize(ref_feats.float(), p=2, dim=1)
    k = min(n_sample, x.shape[0])
    x = x[:k]
    d = torch.cdist(x, x, p=2)
    med = torch.median(d[d > 0])
    return max(float(med.item()), 0.1)


@torch.no_grad()
def mmd_with_ci(
    ref_feats: torch.Tensor,
    feats: torch.Tensor,
    device: torch.device,
    subsample_n: int,
    bootstrap: int,
    bandwidth: float,
    rng: np.random.Generator,
    compute_full: bool = True,
) -> dict:
    """Full MMD(ref, all) plus subsample mean/95% CI at common N.

    Resamples the dataset features to size N WITHOUT replacement. Duplicate samples
    would create identical-pair terms k(v,v)=1 in the within-set MMD sum and inflate
    MMD upward (worst for small pools, where the true MMD is ~0) — so replacement is
    NOT used, matching eval_diversity.py. When pool == N there is only one subset, so
    the CI collapses to the full value (honest: no subsampling freedom at that size).
    The reference is fixed (all reference features).
    """
    ref = ref_feats.to(device)
    B = feats.shape[0]

    def _mmd(sub: torch.Tensor) -> float:
        return compute_cmmd(ref, sub.to(device), kernel_bandwidth=bandwidth)

    full = _mmd(feats) if compute_full else None

    n = min(subsample_n, B)
    draws = []
    if n >= 2 and bootstrap > 0:
        for _ in range(bootstrap):
            idx = rng.choice(B, size=n, replace=False)  # WITHOUT replacement (no duplicate inflation)
            draws.append(_mmd(feats[idx]))
    draws = np.asarray(draws, dtype=np.float64)

    result = {
        "pool_size": int(B),
        "full_mmd": float(full) if full is not None else None,
        "subsample_n": int(n),
        "bootstrap_draws": len(draws),
    }
    if len(draws):
        result.update({
            "mmd_mean": float(draws.mean()),
            "mmd_std": float(draws.std(ddof=1)) if len(draws) > 1 else 0.0,
            "mmd_ci95_low": float(np.percentile(draws, 2.5)),
            "mmd_ci95_high": float(np.percentile(draws, 97.5)),
        })
    return result


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, metavar="LABEL:PATH", help="Real reference set (the distribution to measure distance FROM).")
    ap.add_argument("--dataset", action="append", required=True, metavar="LABEL:PATH", help="Dataset to score as LABEL:PATH (repeatable).")
    ap.add_argument("--pattern", default="bravo.nii.gz", help="Filename glob searched recursively per set (default bravo.nii.gz).")
    ap.add_argument("--target-shape", nargs=3, type=int, default=[150, 256, 256], metavar=("D", "H", "W"), help="Common (D H W) all volumes are resized to before Med3D.")
    ap.add_argument("--med3d-input-size", nargs=3, type=int, default=None, metavar=("D", "H", "W"), help="Optional Med3D internal resize grid (default None = native/target-shape, matching training).")
    ap.add_argument("--pool-cap", type=int, default=256, help="Max volumes per dataset (random subset if more).")
    ap.add_argument("--ref-cap", type=int, default=0, help="Cap reference volumes (0 = use all; real sets are small).")
    ap.add_argument("--subsample-n", type=int, default=51, help="Common N for the comparable estimate. 0 = min dataset size.")
    ap.add_argument("--bootstrap", type=int, default=30, help="Bootstrap resamples (with replacement) for the CI.")
    ap.add_argument("--batch-size", type=int, default=2, help="Med3D extraction batch size (3D ResNet-50 on full volumes is heavy).")
    ap.add_argument("--bandwidth", type=float, default=0.0, help="Fixed RBF bandwidth. 0 = median heuristic from the reference.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", default="med3d_mmd_report.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    target_shape = tuple(args.target_shape)
    input_size = tuple(args.med3d_input_size) if args.med3d_input_size else None
    ref_label, ref_path = parse_dataset_arg(args.reference)
    datasets = [parse_dataset_arg(s) for s in args.dataset]

    logger.info("Device=%s  target_shape=%s  med3d_input_size=%s", device, target_shape, input_size)
    extractor = Med3DFeatures(device, input_size=input_size, compile_model=False)

    # Reference features (extracted once).
    logger.info("Reference %-20s <- %s", ref_label, ref_path)
    ref_feats, ref_found = load_features(
        ref_path, args.pattern, target_shape, extractor, device, args.batch_size, args.ref_cap, rng,
    )
    logger.info("Reference features: %s (from %d volumes)", tuple(ref_feats.shape), ref_found)

    bandwidth = args.bandwidth or median_bandwidth(ref_feats)
    logger.info("RBF bandwidth (fixed) = %.4f", bandwidth)

    # Pass 1: count each dataset so common N can default to the smallest.
    counts = {}
    for label, path in datasets:
        counts[label] = min(len(find_volumes(path, args.pattern)), args.pool_cap)
    subsample_n = args.subsample_n or min(counts.values())
    logger.info("Effective pools: %s | common subsample N = %d", counts, subsample_n)

    report = {
        "config": {
            "reference": f"{ref_label}:{ref_path}", "reference_volumes": ref_found,
            "target_shape": list(target_shape), "med3d_input_size": list(input_size) if input_size else None,
            "bandwidth": bandwidth, "pool_cap": args.pool_cap, "subsample_n": subsample_n,
            "bootstrap": args.bootstrap, "pattern": args.pattern, "seed": args.seed,
        },
        "datasets": {},
    }

    # Pass 2: extract + score + free one dataset at a time.
    for label, path in datasets:
        logger.info("Scoring %-24s <- %s", label, path)
        feats, total_found = load_features(
            path, args.pattern, target_shape, extractor, device, args.batch_size, args.pool_cap, rng,
        )
        res = mmd_with_ci(ref_feats, feats, device, subsample_n, args.bootstrap, bandwidth, rng)
        res["total_found"] = total_found
        report["datasets"][label] = res
        del feats
        if device.type == "cuda":
            torch.cuda.empty_cache()

    extractor.unload()

    # Console table (lower MMD = closer to real).
    print(f"\n=== Med3D-MMD vs {ref_label} ({ref_found} vols), LOWER = closer to real ===")
    print(f"{'dataset':30s} {'found':>6s} {'pool':>5s} {'fullMMD':>9s}  {'mean@N':>9s} {'95% CI':>21s}")
    for label, res in report["datasets"].items():
        ci = (f"[{res['mmd_ci95_low']:.5f},{res['mmd_ci95_high']:.5f}]" if "mmd_mean" in res else "n/a")
        mean = f"{res['mmd_mean']:.5f}" if "mmd_mean" in res else "n/a"
        full = f"{res['full_mmd']:.5f}" if res["full_mmd"] is not None else "n/a"
        print(f"{label:30s} {res['total_found']:6d} {res['pool_size']:5d} {full:>9s}  {mean:>9s} {ci:>21s}")
    print(f"\n(mean@N and CI at common N={subsample_n}, {args.bootstrap} bootstrap draws; bandwidth={bandwidth:.4f})")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
