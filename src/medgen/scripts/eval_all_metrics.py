"""Full generation-metric panel for each set vs a real reference, in one run.

Per dataset, computed against --reference (real): FID/KID (ImageNet + RadImageNet, 2D
axial slices), CMMD (BiomedCLIP), Med3D-MMD (true-3D whole-volume), PCA brain-shape
pass-rate + mean error, and intra-set MS-SSIM diversity. One row per dataset → the full
metrics-vs-downstream table.

Reference and generated volumes go through the SAME extraction functions
(extract_generated_features for the 2D backbones; the Med3D extractor for whole-volume),
so FID/KID/CMMD/MMD are on matched preprocessing. Reuses the exact primitives as training
+ eval_ode_solvers + eval_diversity + eval_med3d_mmd, so numbers are consistent with them.

Each dataset's volumes are loaded ONCE (stacked CPU tensor) and reused for every metric;
all four extractors are instantiated once. Point estimates on the loaded pool (N shown);
for equal-N bootstrap CIs on Med3D-MMD / diversity use eval_med3d_mmd.py / eval_diversity.py.

Example:
    python -m medgen.scripts.eval_all_metrics \
        --reference real_train:$HOME/MedicalDataSets/brainmetshare-3/train \
        --dataset real_test:$HOME/MedicalDataSets/brainmetshare-3/test1 \
        --dataset exp1_1_imagenet:$HOME/MedicalDataSets/generated/exp1_1_bravo_imagenet_525 \
        --pca-model data/brain_pca_256x256x160.npz --output all_metrics.json
"""

import argparse
import json
import logging
import os

import numpy as np
import torch
from tqdm import tqdm

from medgen.metrics.brain_mask import BrainPCAModel, create_brain_mask
from medgen.metrics.feature_extractors import (
    BiomedCLIPFeatures,
    Med3DFeatures,
    ResNet50Features,
)
from medgen.metrics.generation import compute_cmmd, compute_fid, compute_kid
from medgen.metrics.quality import compute_msssim
from medgen.scripts.eval_diversity import (
    find_volumes,
    load_volume,
    pairwise_diversity_3d,
    parse_dataset_arg,
)
from medgen.scripts.eval_med3d_mmd import median_bandwidth

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("eval_all_metrics")


# =============================================================================
# Loading + extraction
# =============================================================================

def load_stacked(paths, target_shape):
    """Load volumes into a [B, 1, D, H, W] CPU tensor (all [0,1]-normalized)."""
    return torch.stack([load_volume(p, target_shape) for p in tqdm(paths, desc="load", leave=False)])


def to_numpy_list(vols):
    """[B, 1, D, H, W] tensor -> list of [D, H, W] numpy views (no copy)."""
    return [vols[i, 0].numpy() for i in range(vols.shape[0])]


@torch.no_grad()
def med3d_from_tensor(vols, extractor, device, batch_size):
    """Extract [N, 2048] Med3D features from a stacked tensor, in batches."""
    feats = []
    for s in range(0, vols.shape[0], batch_size):
        feats.append(extractor.extract_features(vols[s:s + batch_size].to(device)).cpu())
    return torch.cat(feats, dim=0)


def extract_all_features(vols, paths, extractors, trim_slices, device, med3d_batch):
    """Extract IN/RIN/CLIP slice features + Med3D whole-volume features for one set."""
    from medgen.scripts.eval_ode_solvers import extract_generated_features
    np_list = to_numpy_list(vols)
    return {
        "in": extract_generated_features(np_list, extractors["in"], trim_slices),
        "rin": extract_generated_features(np_list, extractors["rin"], trim_slices),
        "clip": extract_generated_features(np_list, extractors["clip"], trim_slices),
        "med3d": med3d_from_tensor(vols, extractors["med3d"], device, med3d_batch),
    }


# =============================================================================
# Metrics
# =============================================================================

def pca_stats(vols, pca_model, threshold):
    """Brain-shape PCA pass-rate + mean reconstruction error over a set's volumes."""
    errors, passes = [], 0
    for i in range(vols.shape[0]):
        vol_np = vols[i, 0].numpy()
        mask = create_brain_mask(vol_np, threshold=threshold, fill_holes=True, dilate_pixels=0)
        ok, err = pca_model.is_valid(mask.astype(np.float32))
        errors.append(err)
        passes += int(ok)
    return passes / vols.shape[0], float(np.mean(errors))


def compute_panel(gen_feats, ref_feats, vols, device, bandwidth, pca_model, pca_threshold):
    """All metrics for one dataset vs the (precomputed) reference features."""
    kid_sub = min(100, gen_feats["in"].shape[0], ref_feats["in"].shape[0])
    kid_in_m, kid_in_s = compute_kid(ref_feats["in"], gen_feats["in"], subset_size=kid_sub)
    kid_rin_m, kid_rin_s = compute_kid(ref_feats["rin"], gen_feats["rin"], subset_size=kid_sub)

    res = {
        "fid_imagenet": compute_fid(ref_feats["in"], gen_feats["in"]),
        "kid_imagenet": kid_in_m, "kid_imagenet_std": kid_in_s,
        "fid_radimagenet": compute_fid(ref_feats["rin"], gen_feats["rin"]),
        "kid_radimagenet": kid_rin_m, "kid_radimagenet_std": kid_rin_s,
        "cmmd": compute_cmmd(ref_feats["clip"], gen_feats["clip"]),
        "med3d_mmd": compute_cmmd(ref_feats["med3d"].to(device), gen_feats["med3d"].to(device), kernel_bandwidth=bandwidth),
        "msssim_diversity": pairwise_diversity_3d(
            vols, device,
            lambda g, r: compute_msssim(g, r, data_range=1.0, spatial_dims=2),
            is_similarity=True,
        ),
    }
    if pca_model is not None:
        pass_rate, mean_err = pca_stats(vols, pca_model, pca_threshold)
        res["pca_pass_rate"] = pass_rate
        res["pca_mean_error"] = mean_err
    return res


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True, metavar="LABEL:PATH", help="Real reference set.")
    ap.add_argument("--dataset", action="append", required=True, metavar="LABEL:PATH", help="Dataset to score (repeatable).")
    ap.add_argument("--pattern", default="bravo.nii.gz")
    ap.add_argument("--target-shape", nargs=3, type=int, default=[150, 256, 256], metavar=("D", "H", "W"))
    ap.add_argument("--pool-cap", type=int, default=256, help="Max volumes per set (random subset if more).")
    ap.add_argument("--ref-cap", type=int, default=0, help="Cap reference volumes (0 = all).")
    ap.add_argument("--trim-slices", type=int, default=0, help="End slices to drop before 2D feature extraction.")
    ap.add_argument("--med3d-batch", type=int, default=2, help="Med3D extraction batch (heavy 3D net).")
    ap.add_argument("--pca-model", default="data/brain_pca_256x256x160.npz", help="Brain-shape PCA npz ('none' to skip).")
    ap.add_argument("--pca-threshold", type=float, default=0.05, help="Brain-mask intensity threshold for PCA.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output", default="all_metrics_report.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)
    target_shape = tuple(args.target_shape)
    ref_label, ref_path = parse_dataset_arg(args.reference)
    datasets = [parse_dataset_arg(s) for s in args.dataset]

    pca_model = None
    if args.pca_model and args.pca_model.lower() != "none":
        pca_model = BrainPCAModel(args.pca_model)
        logger.info("Brain PCA loaded (threshold=%.6f)", pca_model.error_threshold)

    logger.info("Device=%s  target_shape=%s  loading extractors...", device, target_shape)
    extractors = {
        "in": ResNet50Features(device, network_type="imagenet", compile_model=False),
        "rin": ResNet50Features(device, network_type="radimagenet", compile_model=False),
        "clip": BiomedCLIPFeatures(device, compile_model=False),
        "med3d": Med3DFeatures(device, input_size=None, compile_model=False),
    }

    def _capped_paths(path, cap):
        paths = find_volumes(path, args.pattern)
        total = len(paths)
        if total < 2:
            raise ValueError(f"Need >=2 volumes, found {total} under {path}")
        if cap and total > cap:
            keep = sorted(rng.choice(total, size=cap, replace=False))
            paths = [paths[i] for i in keep]
            logger.warning("%s: %d found, using a random %d (cap).", path, total, cap)
        return paths, total

    # Reference features (once).
    logger.info("Reference %-20s <- %s", ref_label, ref_path)
    ref_paths, ref_found = _capped_paths(ref_path, args.ref_cap)
    ref_vols = load_stacked(ref_paths, target_shape)
    ref_feats = extract_all_features(ref_vols, ref_paths, extractors, args.trim_slices, device, args.med3d_batch)
    bandwidth = median_bandwidth(ref_feats["med3d"])
    del ref_vols
    logger.info("Reference: %d vols, Med3D bandwidth=%.4f", ref_found, bandwidth)

    report = {
        "config": {
            "reference": f"{ref_label}:{ref_path}", "reference_volumes": ref_found,
            "target_shape": list(target_shape), "pool_cap": args.pool_cap,
            "trim_slices": args.trim_slices, "med3d_bandwidth": bandwidth,
            "pca_model": args.pca_model, "pattern": args.pattern, "seed": args.seed,
        },
        "datasets": {},
    }

    for label, path in datasets:
        logger.info("Scoring %-24s <- %s", label, path)
        paths, total_found = _capped_paths(path, args.pool_cap)
        vols = load_stacked(paths, target_shape)
        gen_feats = extract_all_features(vols, paths, extractors, args.trim_slices, device, args.med3d_batch)
        res = compute_panel(gen_feats, ref_feats, vols, device, bandwidth, pca_model, args.pca_threshold)
        res["total_found"] = total_found
        res["pool_size"] = int(vols.shape[0])
        report["datasets"][label] = res
        del vols, gen_feats
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for e in extractors.values():
        if hasattr(e, "unload"):
            e.unload()

    # Console table (key columns; full panel in JSON).
    hdr = f"{'dataset':28s} {'N':>4s} {'FID_IN':>8s} {'FID_RIN':>8s} {'CMMD':>8s} {'MED3D':>8s} {'PCA%':>6s} {'DIV':>7s}"
    print(f"\n=== All-metrics panel vs {ref_label} ({ref_found} vols) ===")
    print(hdr)
    for label, r in report["datasets"].items():
        pca = f"{r['pca_pass_rate']*100:.0f}" if "pca_pass_rate" in r else "n/a"
        print(f"{label:28s} {r['pool_size']:>4d} {r['fid_imagenet']:>8.2f} {r['fid_radimagenet']:>8.2f} "
              f"{r['cmmd']:>8.4f} {r['med3d_mmd']:>8.4f} {pca:>6s} {r['msssim_diversity']:>7.4f}")
    print("\n(FID/KID lower=better, CMMD/MED3D lower=better, PCA%=shape-pass higher=better, DIV higher=better; full panel + KID in JSON)")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
