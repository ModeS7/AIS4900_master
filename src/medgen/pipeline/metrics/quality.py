"""
Quality metrics for image comparison.

Provides SSIM, PSNR, and LPIPS metrics used by both DiffusionTrainer and VAETrainer.
"""
import logging
from typing import Any, Optional

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_skimage

logger = logging.getLogger(__name__)

# Global LPIPS model (lazy-loaded, shared across calls)
_lpips_model: Optional[Any] = None
_lpips_device: Optional[torch.device] = None


def compute_ssim(generated: torch.Tensor, reference: torch.Tensor) -> float:
    """Compute SSIM between generated and reference images.

    Args:
        generated: Generated images [B, C, H, W] in [0, 1] range.
        reference: Reference images [B, C, H, W] in [0, 1] range.

    Returns:
        Average SSIM across batch and channels.
    """
    # Convert to float32 for numpy compatibility (handles BFloat16)
    gen_np = np.clip(generated.float().cpu().numpy(), 0, 1)
    ref_np = np.clip(reference.float().cpu().numpy(), 0, 1)

    ssim_values = []
    batch_size, num_channels = gen_np.shape[:2]

    for b in range(batch_size):
        for c in range(num_channels):
            gen_img = gen_np[b, c]
            ref_img = ref_np[b, c]
            ssim_val = ssim_skimage(gen_img, ref_img, data_range=1.0)
            ssim_values.append(ssim_val)

    return float(np.mean(ssim_values))


def compute_psnr(generated: torch.Tensor, reference: torch.Tensor) -> float:
    """Compute PSNR between generated and reference images.

    Args:
        generated: Generated images [B, C, H, W] in [0, 1] range.
        reference: Reference images [B, C, H, W] in [0, 1] range.

    Returns:
        Average PSNR across batch.
    """
    # Convert to float32 for numpy compatibility (handles BFloat16)
    gen_np = np.clip(generated.float().cpu().numpy(), 0, 1)
    ref_np = np.clip(reference.float().cpu().numpy(), 0, 1)

    mse = np.mean((gen_np - ref_np) ** 2)
    if mse < 1e-10:
        return 100.0

    psnr = 10 * np.log10(1.0 / mse)
    return float(psnr)


def _get_lpips_model(device: torch.device) -> Optional[Any]:
    """Get or initialize the global LPIPS model.

    Args:
        device: Device to load model on.

    Returns:
        LPIPS model or None if not available.
    """
    global _lpips_model, _lpips_device

    # Return existing model if on same device
    if _lpips_model is not None and _lpips_device == device:
        return _lpips_model

    # Initialize LPIPS model
    try:
        import lpips
        _lpips_model = lpips.LPIPS(net='alex', verbose=False).to(device)
        _lpips_model.eval()
        for param in _lpips_model.parameters():
            param.requires_grad = False
        _lpips_device = device
        logger.info("LPIPS model loaded (AlexNet)")
        return _lpips_model
    except ImportError:
        logger.warning("lpips package not installed - LPIPS metric disabled")
        return None


def compute_lpips(
    generated: torch.Tensor,
    reference: torch.Tensor,
    device: Optional[torch.device] = None
) -> float:
    """Compute LPIPS (perceptual similarity) between generated and reference images.

    Args:
        generated: Generated images [B, C, H, W] in [0, 1] range.
        reference: Reference images [B, C, H, W] in [0, 1] range.
        device: Device for LPIPS model. Defaults to generated.device.

    Returns:
        Average LPIPS across batch (lower is better, 0 = identical).
    """
    if device is None:
        device = generated.device

    lpips_model = _get_lpips_model(device)
    if lpips_model is None:
        return 0.0

    # LPIPS expects images in [-1, 1] range
    gen = generated.float() * 2.0 - 1.0
    ref = reference.float() * 2.0 - 1.0

    # Handle different channel counts (LPIPS expects 3-channel RGB)
    num_channels = gen.shape[1]

    with torch.no_grad():
        if num_channels == 1:
            # Single channel: replicate to 3
            gen = gen.repeat(1, 3, 1, 1)
            ref = ref.repeat(1, 3, 1, 1)
            lpips_values = lpips_model(gen, ref)
        elif num_channels <= 3:
            # 2-3 channels: compute per-channel LPIPS and average
            lpips_per_channel = []
            for ch in range(num_channels):
                ch_gen = gen[:, ch:ch+1].repeat(1, 3, 1, 1)
                ch_ref = ref[:, ch:ch+1].repeat(1, 3, 1, 1)
                lpips_per_channel.append(lpips_model(ch_gen, ch_ref))
            lpips_values = torch.stack(lpips_per_channel).mean(dim=0)
        else:
            # >3 channels: compute per-channel for first 3 and average
            lpips_per_channel = []
            for ch in range(3):
                ch_gen = gen[:, ch:ch+1].repeat(1, 3, 1, 1)
                ch_ref = ref[:, ch:ch+1].repeat(1, 3, 1, 1)
                lpips_per_channel.append(lpips_model(ch_gen, ch_ref))
            lpips_values = torch.stack(lpips_per_channel).mean(dim=0)

    return float(lpips_values.mean().item())
