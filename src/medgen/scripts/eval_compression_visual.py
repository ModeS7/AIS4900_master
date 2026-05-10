"""Visual reconstruction comparison for compression models (VAE / VQ-VAE / DC-AE, 2D and 3D).

Loads N real volumes, runs each through one or more compression models,
saves per-model NIfTI reconstructions plus a thesis-ready figure showing
original vs. reconstructions side-by-side with PSNR/SSIM annotations.

Usage:
    python -m medgen.scripts.eval_compression_visual \\
        --data-root /path/to/brainmetshare-3 \\
        --split test \\
        --num-volumes 3 \\
        --output-dir runs/eval/compression_visual \\
        --model "DC-AE 2D f32:/path/to/dcae_2d_f32/checkpoint_best.pt" \\
        --model "DC-AE 2D f128:/path/to/dcae_2d_f128/checkpoint_best.pt" \\
        --model "VQ-VAE 3D 4x:/path/to/vqvae_3d/checkpoint_best.pt" \\
        --model "VAE 3D 4x:/path/to/vae_3d/checkpoint_best.pt"

Each --model arg is "label:checkpoint_path". Spatial dims (2 vs 3) and
compression type are auto-detected from the checkpoint.

Outputs (under --output-dir):
    metrics.csv                         per-volume PSNR/MS-SSIM per model
    reconstruction_grid.png             N rows × (1+M) cols mid-slice grid
    reconstruction_grid.pdf             same, vector for thesis
    nifti/<label>/<NN>.nii.gz           reconstructed volumes (sanitized labels)
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from torch.amp import autocast

logger = logging.getLogger(__name__)


def _sanitize(label: str) -> str:
    """Make a label safe for use as a filesystem path."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label.strip()).strip("_") or "model"


def _find_split_dir(data_root: Path, split: str, modality: str) -> Path:
    """Resolve the split directory under data_root.

    Tries ``data_root/split`` first, then any subdir matching ``split*``.
    """
    direct = data_root / split
    if direct.is_dir() and any(direct.glob(f"*/{modality}.nii.gz")):
        return direct
    for sub in sorted(data_root.iterdir()):
        if sub.is_dir() and sub.name.startswith(split):
            if any(sub.glob(f"*/{modality}.nii.gz")):
                return sub
    raise FileNotFoundError(
        f"No split directory under {data_root} matching '{split}' contains {modality}.nii.gz"
    )


def load_volume_nifti(
    path: Path, depth: int, image_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI, normalize to [0,1], crop/pad depth, return (volume, affine).

    Returns:
        vol: [D, H, W] float32 in [0, 1].
        affine: 4x4 affine for saving reconstructions back as NIfTI.
    """
    img = nib.load(str(path))
    vol = img.get_fdata().astype(np.float32)
    affine = img.affine

    vmin, vmax = float(vol.min()), float(vol.max())
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)

    # NIfTI is [H, W, D] -> internal [D, H, W]
    vol = np.transpose(vol, (2, 0, 1))

    d = vol.shape[0]
    if d < depth:
        pad = np.zeros((depth - d, vol.shape[1], vol.shape[2]), dtype=np.float32)
        vol = np.concatenate([vol, pad], axis=0)
    elif d > depth:
        vol = vol[:depth]

    # Center-crop H, W to image_size if larger
    h, w = vol.shape[1], vol.shape[2]
    if h > image_size:
        s = (h - image_size) // 2
        vol = vol[:, s : s + image_size]
    if w > image_size:
        s = (w - image_size) // 2
        vol = vol[:, :, s : s + image_size]

    return vol, affine


@torch.no_grad()
def reconstruct_3d(
    model: torch.nn.Module, vol: torch.Tensor, ctype: str
) -> torch.Tensor:
    """Encode-decode a [1, 1, D, H, W] tensor with a 3D compression model."""
    if ctype == "maisi":
        return _reconstruct_maisi(model, vol)
    with autocast("cuda", dtype=torch.bfloat16):
        if ctype == "vqvae":
            z = model.encode(vol)
            recon = model.decode_stage_2_outputs(z)
        elif ctype == "vae":
            z_mu, _ = model.encode(vol)
            recon = model.decode(z_mu)
        elif ctype == "dcae":
            z = model.encode(vol)
            recon = model.decode(z)
        else:
            raise ValueError(f"Unknown compression type: {ctype}")
    return recon.float().clamp(0, 1)


@torch.no_grad()
def _reconstruct_maisi(model: torch.nn.Module, vol: torch.Tensor) -> torch.Tensor:
    """Encode-decode for MAISI VAE.

    MAISI uses [B, C, H, W, D] convention (depth-last) and requires spatial
    dims be multiples of 4 (4× compression per axis). For full-resolution
    volumes the decoder is memory-heavy, so we use sliding-window inference
    over the latent. Forces float32 because MAISI's GroupNorm has a bf16
    dtype-mismatch bug.

    Critically, MAISI was trained with **0th-99.5th percentile clip-and-scale
    to [0,1]**, not the min-max normalization our load_volume_nifti uses.
    Without applying MAISI's native normalization, MAISI sees out-of-distribution
    inputs and produces ~0.5 MS-SSIM reconstructions. We renormalize internally
    so MAISI's metrics reflect its true reconstruction quality on its training
    convention.
    """
    import torch.nn.functional as F
    from monai.inferers import SlidingWindowInferer

    # Apply MAISI's training-time normalization (0-99.5 percentile clip to [0,1]).
    # vol is currently min-max normalized to [0,1]. We re-derive the 99.5
    # percentile in the current space and re-stretch.
    p995 = torch.quantile(vol.flatten(), 0.995)
    if p995 > 0:
        vol_norm = (vol / p995).clamp(0, 1)
    else:
        vol_norm = vol

    # Our convention: [B, C, D, H, W]. MAISI: [B, C, H, W, D]. Permute.
    vol_maisi = vol_norm.permute(0, 1, 3, 4, 2).contiguous().float()

    h, w, d = vol_maisi.shape[2], vol_maisi.shape[3], vol_maisi.shape[4]
    pad_h = (4 - h % 4) % 4
    pad_w = (4 - w % 4) % 4
    pad_d = (4 - d % 4) % 4
    if pad_h or pad_w or pad_d:
        vol_maisi = F.pad(vol_maisi, (0, pad_d, 0, pad_w, 0, pad_h))

    z = model.encode(vol_maisi)
    if isinstance(z, (list, tuple)):
        z = z[0]  # z_mu

    latent_size = z.shape[2] * z.shape[3] * z.shape[4]
    if latent_size > 64 ** 3:
        decode_inferer = SlidingWindowInferer(
            roi_size=(64, 64, 64), sw_batch_size=1, overlap=0.25, mode="gaussian",
        )
        recon = decode_inferer(z, lambda lat: model.decode(lat))
    else:
        recon = model.decode(z)

    if pad_h or pad_w or pad_d:
        recon = recon[:, :, :h, :w, :d]

    # Permute back to [B, C, D, H, W]
    recon = recon.permute(0, 1, 4, 2, 3).contiguous().clamp(0, 1)

    # Un-normalize: reverse the p99.5 stretch so output is in same range as
    # the original min-max input. This makes metrics directly comparable to
    # the other models (which all use min-max-normalized ground truth).
    if p995 > 0:
        recon = (recon * p995).clamp(0, 1)

    return recon.float()


def _unwrap_encoder_output(out):
    """Extract latent tensor from possibly-wrapped diffusers EncoderOutput / tuple."""
    if hasattr(out, "latent"):
        return out.latent
    if isinstance(out, tuple):
        return out[0]
    return out


def _unwrap_decoder_output(out):
    """Extract sample tensor from possibly-wrapped diffusers DecoderOutput / tuple."""
    if hasattr(out, "sample"):
        return out.sample
    if isinstance(out, tuple):
        return out[0]
    return out


@torch.no_grad()
def reconstruct_2d_per_slice(
    model: torch.nn.Module,
    vol: torch.Tensor,
    ctype: str,
    chunk: int = 16,
) -> torch.Tensor:
    """Encode-decode a 3D volume slice-by-slice through a 2D compression model.

    Args:
        vol: [1, 1, D, H, W] tensor in [0, 1].
        chunk: number of slices per forward pass.
    Returns:
        [1, 1, D, H, W] reconstructed tensor in [0, 1].
    """
    _, _, d, _h, _w = vol.shape
    slices = vol.squeeze(0).squeeze(0)  # [D, H, W]
    out_slices = []
    for start in range(0, d, chunk):
        end = min(start + chunk, d)
        batch = slices[start:end].unsqueeze(1)  # [chunk, 1, H, W]
        with autocast("cuda", dtype=torch.bfloat16):
            if ctype == "vqvae":
                z = model.encode(batch)
                recon = model.decode_stage_2_outputs(z)
            elif ctype == "vae":
                z_mu, _ = model.encode(batch)
                recon = model.decode(z_mu)
            elif ctype == "dcae":
                # diffusers AutoencoderDC returns EncoderOutput / DecoderOutput,
                # StructuredAutoencoderDC delegates to the same. Unwrap both.
                z = _unwrap_encoder_output(model.encode(batch))
                recon = _unwrap_decoder_output(model.decode(z))
            else:
                raise ValueError(f"Unknown compression type: {ctype}")
        out_slices.append(recon.float().clamp(0, 1).squeeze(1))  # [chunk, H, W]
    full = torch.cat(out_slices, dim=0)  # [D, H, W]
    return full.unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]


def save_nifti_volume(vol: np.ndarray, affine: np.ndarray, path: Path) -> None:
    """Save a [D, H, W] volume as NIfTI (transposed back to [H, W, D])."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.transpose(vol, (1, 2, 0))  # [H, W, D]
    nib.save(nib.Nifti1Image(arr.astype(np.float32), affine), str(path))


def plot_grid(
    originals: list[np.ndarray],
    recons_per_model: list[list[np.ndarray]],
    model_labels: list[str],
    volume_ids: list[str],
    metrics: list[list[dict]],
    output_path_png: Path,
    output_path_pdf: Path,
) -> None:
    """Build a (N rows × (1 + M) cols) figure of axial mid-slices.

    Row i = volume i; col 0 = original, cols 1..M = each model's reconstruction.
    Each reconstruction cell is annotated with PSNR / MS-SSIM.
    """
    n_vols = len(originals)
    n_models = len(model_labels)
    n_cols = 1 + n_models

    fig, axes = plt.subplots(
        n_vols, n_cols,
        figsize=(2.5 * n_cols, 2.5 * n_vols),
        squeeze=False,
    )

    for i, orig in enumerate(originals):
        mid = orig.shape[0] // 2  # axial mid
        axes[i, 0].imshow(orig[mid], cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_ylabel(volume_ids[i], fontsize=9)
        if i == 0:
            axes[i, 0].set_title("Original", fontsize=10)
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])

        for j, label in enumerate(model_labels):
            rec = recons_per_model[j][i]
            ax = axes[i, 1 + j]
            ax.imshow(rec[mid], cmap="gray", vmin=0, vmax=1)
            if i == 0:
                ax.set_title(label, fontsize=10)
            psnr = metrics[j][i]["psnr"]
            ssim = metrics[j][i]["msssim"]
            ax.set_xlabel(f"PSNR {psnr:.2f}\nMS-SSIM {ssim:.3f}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    output_path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path_png, dpi=200, bbox_inches="tight")
    fig.savefig(output_path_pdf, bbox_inches="tight")
    plt.close(fig)


def _strip_state_dict_prefix(state_dict: dict, prefix: str = "model.") -> dict:
    """Strip a leading prefix from all keys in a state_dict, idempotently."""
    if not any(k.startswith(prefix) for k in state_dict):
        return state_dict
    return {(k[len(prefix):] if k.startswith(prefix) else k): v
            for k, v in state_dict.items()}


def _build_model_for_label(
    label: str, ckpt_path: Path, device: torch.device
) -> tuple[torch.nn.Module, str, int, int, int]:
    """Build & load a compression model based on label and checkpoint structure.

    Bypasses the generic loader for DC-AE 2D where production checkpoints are
    saved with structured-latent wrapping (encoder.conv_out.conv.* and
    decoder.conv_in.conv.*); falls back to the generic loader for VAE/VQ-VAE.

    Returns:
        (model, compression_type, spatial_dims, scale_factor, latent_channels)
    """
    label_lower = label.lower()

    # MAISI checked first — separate architecture, separate loader.
    if "maisi" in label_lower:
        return _build_maisi_vae(label, ckpt_path, device)

    # Determine type from label (avoid the broken auto-detect for AutoencoderDC).
    # Check DC-AE and VQ-VAE substrings before VAE since "VQ-VAE" contains "VAE".
    if "dc-ae" in label_lower or "dcae" in label_lower:
        ctype = "dcae"
    elif "vq-vae" in label_lower or "vqvae" in label_lower:
        ctype = "vqvae"
    elif "vae" in label_lower:
        ctype = "vae"
    else:
        raise ValueError(f"Cannot infer compression type from label: {label!r}")

    sdims = 3 if " 3d" in label_lower or "_3d" in label_lower else 2

    if ctype == "dcae" and sdims == 2:
        return _build_dcae_2d(label, ckpt_path, device)

    # VAE/VQ-VAE 3D — generic loader works (no version-skew on these architectures).
    from medgen.data.loaders.compression_detection import load_compression_model
    return load_compression_model(
        str(ckpt_path), ctype, device, spatial_dims=sdims,
    )


def _build_maisi_vae(
    label: str, ckpt_path: Path, device: torch.device
) -> tuple[torch.nn.Module, str, int, int, int]:
    """Load NVIDIA's MAISI VAE from a MONAI bundle.

    The path can point to either:
      - The bundle directory (`bundles/maisi_ct_generative/`), in which case
        `models/autoencoder.pt` is loaded.
      - The autoencoder.pt checkpoint file directly.

    MAISI VAE: 3D, 4× spatial compression per axis, 4 latent channels (16× total).
    `norm_float16=False` to avoid a known dtype-mismatch bug in MaisiGroupNorm3D.
    """
    from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi

    p = Path(ckpt_path)
    if p.is_dir():
        ckpt_file = p / "models" / "autoencoder.pt"
    else:
        ckpt_file = p
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"MAISI checkpoint not found: {ckpt_file}")

    model = AutoencoderKlMaisi(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        latent_channels=4,
        num_channels=[64, 128, 256],
        num_res_blocks=[2, 2, 2],
        norm_num_groups=32,
        norm_eps=1e-6,
        attention_levels=[False, False, False],
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
        use_checkpointing=False,
        use_convtranspose=False,
        norm_float16=False,
        num_splits=1,
        dim_split=1,
    )
    state = torch.load(str(ckpt_file), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model = model.to(device).float()
    model.eval()

    logger.info(
        "  loaded MAISI VAE: latent_ch=4, scale=4× per axis (16× total), checkpoint=%s",
        ckpt_file,
    )
    return model, "maisi", 3, 4, 4


def _build_dcae_2d(
    label: str, ckpt_path: Path, device: torch.device
) -> tuple[torch.nn.Module, str, int, int, int]:
    """Build & load a 2D DC-AE checkpoint, auto-detecting structured-latent wrapping.

    Structured DC-AE (DC-AE 1.5) wraps `encoder.conv_out` and `decoder.conv_in`
    with `AdaptiveOutputConv2d` / `AdaptiveInputConv2d`, which contain a `.conv`
    sublayer. Detection: presence of `encoder.conv_out.conv.weight` in state_dict.
    """
    from diffusers import AutoencoderDC

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    raw_state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    state_dict = _strip_state_dict_prefix(raw_state_dict)

    # Detect structured wrapping vs. vanilla and the latent channel count.
    if "encoder.conv_out.conv.weight" in state_dict:
        latent_ch = int(state_dict["encoder.conv_out.conv.weight"].shape[0])
        is_structured = True
    elif "encoder.conv_out.weight" in state_dict:
        latent_ch = int(state_dict["encoder.conv_out.weight"].shape[0])
        is_structured = False
    else:
        raise RuntimeError(
            "Could not locate encoder.conv_out weight in DC-AE 2D checkpoint "
            f"({ckpt_path}); state_dict has {len(state_dict)} keys."
        )

    # Compression ratio = 32x for c=32, 64x for c=64, 128x for c=128 in DC-AE
    # (this codebase uses fXcX naming where the f-number IS the spatial ratio).
    scale = latent_ch

    base_model = AutoencoderDC(
        in_channels=1,
        latent_channels=latent_ch,
        encoder_block_out_channels=(128, 256, 512, 512, 1024, 1024),
        decoder_block_out_channels=(128, 256, 512, 512, 1024, 1024),
        encoder_layers_per_block=(2, 2, 2, 3, 3, 3),
        decoder_layers_per_block=(3, 3, 3, 3, 3, 3),
        encoder_qkv_multiscales=((), (), (), (5,), (5,), (5,)),
        decoder_qkv_multiscales=((), (), (), (5,), (5,), (5,)),
        encoder_block_types="ResBlock",
        decoder_block_types="ResBlock",
        downsample_block_type="pixel_unshuffle",
        upsample_block_type="pixel_shuffle",
        encoder_out_shortcut=True,
        decoder_in_shortcut=True,
    ).to(device)

    if is_structured:
        from medgen.models.dcae_structured import StructuredAutoencoderDC
        # Default channel_steps used during training: range(min, latent+1, step)
        # with min=16, step=4 per configs/dcae/default.yaml structured_latent block.
        channel_steps = list(range(16, latent_ch + 1, 4))
        model = StructuredAutoencoderDC(base_model, channel_steps).to(device)
        logger.info(
            "  DC-AE 2D structured-latent wrapper applied (channel_steps=%s)",
            channel_steps,
        )
    else:
        model = base_model
        logger.info("  DC-AE 2D vanilla (no structured-latent wrapping)")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logger.warning(
            "DC-AE 2D state_dict load: %d missing, %d unexpected keys",
            len(missing), len(unexpected),
        )
        if missing:
            logger.warning("  first 5 missing: %s", missing[:5])
        if unexpected:
            logger.warning("  first 5 unexpected: %s", unexpected[:5])

    logger.info(
        "  loaded DC-AE 2D: latent_ch=%d, scale=%dx, structured=%s",
        latent_ch, scale, is_structured,
    )
    return model, "dcae", 2, scale, latent_ch


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, help="Path to brainmetshare-3 root.")
    parser.add_argument("--split", default="test", help="Split subdir to sample from.")
    parser.add_argument("--modality", default="bravo", help="Modality file stem (default: bravo).")
    parser.add_argument("--num-volumes", type=int, default=3, help="Number of volumes to reconstruct.")
    parser.add_argument("--volume-ids", nargs="*", default=None,
                        help="Explicit subject IDs; overrides --num-volumes if given.")
    parser.add_argument("--depth", type=int, default=160)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--output-dir", required=True, help="Where to write outputs.")
    parser.add_argument("--model", action="append", default=[], required=True,
                        metavar="LABEL:CKPT",
                        help='Compression model spec, repeatable. Format: "label:/abs/path/to/ckpt".')
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    parsed_models: list[tuple[str, Path]] = []
    for spec in args.model:
        if ":" not in spec:
            parser.error(f"--model must be 'label:path', got {spec!r}")
        label, raw_path = spec.split(":", 1)
        ckpt = Path(raw_path).expanduser()
        if not ckpt.is_file():
            parser.error(f"Checkpoint not found: {ckpt}")
        parsed_models.append((label.strip(), ckpt))
    args.model_specs = parsed_models
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args(argv)

    from medgen.metrics.quality import compute_msssim, compute_psnr

    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # ── Pick volumes ──────────────────────────────────────────────────────────
    data_root = Path(args.data_root)
    split_dir = _find_split_dir(data_root, args.split, args.modality)
    all_subjects = sorted(p.parent.name for p in split_dir.glob(f"*/{args.modality}.nii.gz"))
    if not all_subjects:
        logger.error("No subjects found in %s", split_dir)
        return 2

    if args.volume_ids:
        missing = [v for v in args.volume_ids if v not in all_subjects]
        if missing:
            logger.error("Subjects not found in split: %s", missing)
            return 2
        chosen = args.volume_ids
    else:
        idx = rng.choice(len(all_subjects), size=min(args.num_volumes, len(all_subjects)), replace=False)
        chosen = [all_subjects[i] for i in sorted(idx)]
    logger.info("Selected %d volumes from %s: %s", len(chosen), split_dir, chosen)

    # ── Load originals (CPU) ──────────────────────────────────────────────────
    originals: list[np.ndarray] = []
    affines: list[np.ndarray] = []
    for subj in chosen:
        vol, aff = load_volume_nifti(split_dir / subj / f"{args.modality}.nii.gz",
                                     depth=args.depth, image_size=args.image_size)
        originals.append(vol)
        affines.append(aff)
        logger.info("  loaded %s shape=%s", subj, vol.shape)

    # ── Run each model ────────────────────────────────────────────────────────
    model_labels: list[str] = []
    recons_per_model: list[list[np.ndarray]] = []
    metrics_per_model: list[list[dict]] = []

    for label, ckpt in args.model_specs:
        logger.info("=" * 70)
        logger.info("Loading model: %s  (%s)", label, ckpt)
        model, ctype, sdims, scale, latent_ch = _build_model_for_label(
            label, ckpt, device
        )
        model.eval()
        logger.info("  detected: type=%s spatial_dims=%dD scale=%dx latent_ch=%d",
                    ctype, sdims, scale, latent_ch)

        recons: list[np.ndarray] = []
        per_vol_metrics: list[dict] = []
        for i, vol_np in enumerate(originals):
            vol_t = torch.from_numpy(vol_np).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,D,H,W]
            if sdims == 3:
                rec_t = reconstruct_3d(model, vol_t, ctype)
            else:
                rec_t = reconstruct_2d_per_slice(model, vol_t, ctype)
            rec_np = rec_t.squeeze(0).squeeze(0).cpu().numpy()  # [D, H, W]
            recons.append(rec_np)

            # Metrics — both 3D MS-SSIM (volume-level) and PSNR
            psnr = compute_psnr(rec_t, vol_t)
            try:
                msssim = compute_msssim(rec_t, vol_t, spatial_dims=3)
            except (ValueError, RuntimeError) as exc:
                logger.warning("MS-SSIM failed for %s vol %s, falling back to 2D mean: %s",
                               label, chosen[i], exc)
                # Fallback: mean MS-SSIM across axial slices
                rec_slices = rec_t.squeeze(0).permute(1, 0, 2, 3)  # [D, 1, H, W]
                ref_slices = vol_t.squeeze(0).permute(1, 0, 2, 3)
                msssim = compute_msssim(rec_slices, ref_slices, spatial_dims=2)

            per_vol_metrics.append({"psnr": psnr, "msssim": msssim})
            logger.info("  %s vs %s: PSNR=%.3f MS-SSIM=%.4f", label, chosen[i], psnr, msssim)

            # Save reconstructed NIfTI
            nifti_path = out_root / "nifti" / _sanitize(label) / f"{chosen[i]}.nii.gz"
            save_nifti_volume(rec_np, affines[i], nifti_path)

        model_labels.append(label)
        recons_per_model.append(recons)
        metrics_per_model.append(per_vol_metrics)

        # Free GPU memory before next model
        del model
        torch.cuda.empty_cache()

    # ── CSV ────────────────────────────────────────────────────────────────────
    csv_path = out_root / "metrics.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "subject", "psnr", "msssim"])
        for label, per_vol in zip(model_labels, metrics_per_model, strict=True):
            for subj, m in zip(chosen, per_vol, strict=True):
                w.writerow([label, subj, f"{m['psnr']:.4f}", f"{m['msssim']:.4f}"])
        # Aggregate row per model (mean across volumes)
        w.writerow([])
        w.writerow(["model", "subject", "psnr_mean", "msssim_mean"])
        for label, per_vol in zip(model_labels, metrics_per_model, strict=True):
            psnr_mean = float(np.mean([m["psnr"] for m in per_vol]))
            msssim_mean = float(np.mean([m["msssim"] for m in per_vol]))
            w.writerow([label, "MEAN", f"{psnr_mean:.4f}", f"{msssim_mean:.4f}"])
    logger.info("Wrote metrics to %s", csv_path)

    # ── Figure ────────────────────────────────────────────────────────────────
    plot_grid(
        originals, recons_per_model, model_labels, chosen, metrics_per_model,
        out_root / "reconstruction_grid.png",
        out_root / "reconstruction_grid.pdf",
    )
    logger.info("Wrote figure to %s and .pdf", out_root / "reconstruction_grid.png")

    # ── Print compact summary ────────────────────────────────────────────────
    print("\n=== Summary (mean across volumes) ===")
    print(f"{'model':<30} {'PSNR':>8} {'MS-SSIM':>10}")
    for label, per_vol in zip(model_labels, metrics_per_model, strict=True):
        psnr_mean = float(np.mean([m["psnr"] for m in per_vol]))
        msssim_mean = float(np.mean([m["msssim"] for m in per_vol]))
        print(f"{label:<30} {psnr_mean:8.3f} {msssim_mean:10.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
