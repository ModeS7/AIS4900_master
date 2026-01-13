"""
Generation quality metrics for diffusion models.

Tracks distributional distance between generated and real images using:
- KID: Kernel Inception Distance (unbiased, small sample friendly)
- CMMD: CLIP Maximum Mean Discrepancy (domain-aware via BiomedCLIP)
- FID: Fréchet Inception Distance (for final test evaluation)

These metrics compare feature distributions rather than individual samples,
enabling detection of mode collapse and overfitting during training.

Usage:
    config = GenerationMetricsConfig(enabled=True)
    gen_metrics = GenerationMetrics(config, device, run_dir)

    # At training start
    gen_metrics.set_fixed_conditioning(train_dataset, num_masks=500)
    gen_metrics.cache_reference_features(train_loader, val_loader, experiment_id)

    # Every epoch
    results = gen_metrics.compute_epoch_metrics(model, strategy, mode)

    # Every N epochs
    extended = gen_metrics.compute_extended_metrics(model, strategy, mode)

    # Final test
    test_results = gen_metrics.compute_test_metrics(model, strategy, mode, test_loader)

Reference:
- KID: https://arxiv.org/abs/1801.01401 (MMD-based, unbiased)
- CMMD: CLIP-based MMD for domain-specific comparison
- FID: https://arxiv.org/abs/1706.08500 (Fréchet distance on Inception features)
"""
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GenerationMetricsConfig:
    """Configuration for generation quality metrics.

    Attributes:
        enabled: Whether generation metrics are enabled.
        samples_per_epoch: Number of samples for quick metrics (every epoch).
        samples_extended: Number of samples for extended metrics (every N epochs).
        samples_test: Number of samples for final test evaluation.
        steps_per_epoch: Denoising steps for quick metrics.
        steps_extended: Denoising steps for extended metrics.
        steps_test: Denoising steps for test evaluation.
        cache_dir: Directory for caching reference features.
        feature_batch_size: Batch size for feature extraction.
    """
    enabled: bool = False
    samples_per_epoch: int = 100
    samples_extended: int = 500
    samples_test: int = 1000
    steps_per_epoch: int = 10
    steps_extended: int = 25
    steps_test: int = 50
    cache_dir: str = ".cache/generation_features"
    feature_batch_size: int = 16  # Set by trainer to match training.batch_size


# =============================================================================
# Feature Extractors
# =============================================================================

class ResNet50Features(nn.Module):
    """ResNet50 feature extractor for KID/FID.

    Uses torchvision ResNet50 with fc layer removed for 2048-dim features.
    Supports both ImageNet and RadImageNet (medical-domain) pretrained weights.

    Args:
        device: PyTorch device for computation.
        network_type: 'imagenet' or 'radimagenet' pretrained weights.
        cache_dir: Optional directory for RadImageNet checkpoint.
    """

    def __init__(
        self,
        device: torch.device,
        network_type: str = "imagenet",
        cache_dir: Optional[Path] = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.network_type = network_type
        self.cache_dir = cache_dir
        self._model: Optional[nn.Module] = None

    def _ensure_model(self) -> None:
        """Lazy-load the ResNet50 model."""
        if self._model is not None:
            return

        import torchvision.models as models

        if self.network_type == "radimagenet":
            # Try to load RadImageNet checkpoint
            checkpoint_path = None
            if self.cache_dir:
                checkpoint_path = self.cache_dir / "RadImageNet-ResNet50_notop.pth"

            if checkpoint_path and checkpoint_path.exists():
                model = models.resnet50(weights=None)
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded RadImageNet weights from {checkpoint_path}")
            else:
                # Fallback to torch.hub
                try:
                    model = torch.hub.load(
                        "Warvito/radimagenet-models",
                        model="radimagenet_resnet50",
                        verbose=False,
                    )
                    logger.info("Loaded RadImageNet weights from torch.hub")
                except Exception as e:
                    logger.warning(f"RadImageNet not available ({e}), falling back to ImageNet")
                    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            # ImageNet pretrained
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            logger.info("Loaded ImageNet pretrained ResNet50")

        # Remove final FC layer to get 2048-dim features
        model.fc = nn.Identity()
        model = model.to(self.device)
        model.eval()

        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        # Compile for faster inference
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("ResNet50 compiled with torch.compile")
        except Exception as e:
            logger.warning(f"torch.compile failed for ResNet50: {e}")

        self._model = model

    @torch.no_grad()
    def extract_features(
        self,
        images: torch.Tensor,
        use_amp: bool = True,
    ) -> torch.Tensor:
        """Extract ResNet50 features with AMP support.

        Args:
            images: Images [B, C, H, W] in [0, 1] range, 1-3 channels.
            use_amp: Whether to use automatic mixed precision.

        Returns:
            Feature tensor [B, 2048].
        """
        self._ensure_model()

        # Handle grayscale by repeating to 3 channels
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        elif images.shape[1] == 2:
            # Dual mode: average channels then repeat
            images = images.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

        # Resize to 224x224 (ResNet input size)
        if images.shape[2] != 224 or images.shape[3] != 224:
            images = F.interpolate(
                images, size=(224, 224), mode='bilinear', align_corners=False
            )

        # Ensure [0, 1] range
        images = torch.clamp(images, 0, 1)

        # Move to device with non_blocking
        images = images.to(self.device, non_blocking=True)

        # Apply preprocessing based on network type
        if self.network_type == "radimagenet":
            # RadImageNet: BGR + mean subtraction
            images = images[:, [2, 1, 0], ...]  # RGB to BGR
            mean = torch.tensor([0.406, 0.456, 0.485], device=self.device).view(1, 3, 1, 1)
            images = images - mean
        else:
            # ImageNet: standard normalization
            mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            images = (images - mean) / std

        # Extract features with AMP
        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            features = self._model(images)

        # Global average pooling if needed (shouldn't be with fc=Identity)
        if len(features.shape) > 2:
            features = features.mean([2, 3])

        return features.float()  # Return fp32 for metric computation


class BiomedCLIPFeatures(nn.Module):
    """BiomedCLIP feature extractor for CMMD.

    Uses Microsoft's BiomedCLIP model trained on 15M biomedical
    image-text pairs for domain-specific embeddings.

    Args:
        device: PyTorch device for computation.
        cache_dir: Optional cache directory for model weights.
    """

    def __init__(
        self,
        device: torch.device,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.device = device
        self.cache_dir = cache_dir
        self._model: Optional[nn.Module] = None
        self._processor = None

    def _ensure_model(self) -> None:
        """Lazy-load the BiomedCLIP model."""
        if self._model is not None:
            return

        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "open_clip_torch is required for CMMD metrics. "
                "Install with: pip install open_clip_torch"
            )

        model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

        logger.info(f"Loading BiomedCLIP from {model_name}...")

        # Load using open_clip which properly handles BiomedCLIP
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name,
            device=self.device,
        )

        self._model.eval()

        # Freeze all parameters
        for param in self._model.parameters():
            param.requires_grad = False

        # Compile for faster inference
        try:
            self._model = torch.compile(self._model, mode="reduce-overhead")
            logger.info("BiomedCLIP compiled with torch.compile")
        except Exception as e:
            logger.warning(f"torch.compile failed for BiomedCLIP: {e}")

        logger.info("Loaded BiomedCLIP for feature extraction")

    @torch.no_grad()
    def extract_features(
        self,
        images: torch.Tensor,
        use_amp: bool = True,
    ) -> torch.Tensor:
        """Extract BiomedCLIP image features with AMP support.

        Args:
            images: Images [B, C, H, W] in [0, 1] range, 1-3 channels.
            use_amp: Whether to use automatic mixed precision.

        Returns:
            Feature tensor [B, 512] (CLIP embedding dimension).
        """
        self._ensure_model()

        # Handle grayscale by repeating to 3 channels
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        elif images.shape[1] == 2:
            images = images.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)

        # Resize to 224x224 (CLIP input size)
        if images.shape[2] != 224 or images.shape[3] != 224:
            images = F.interpolate(
                images, size=(224, 224), mode='bilinear', align_corners=False
            )

        # Ensure [0, 1] range
        images = torch.clamp(images, 0, 1)

        # Move to device with non_blocking
        images = images.to(self.device, non_blocking=True)

        # BiomedCLIP/OpenCLIP expects normalized images
        # Use OpenAI CLIP normalization (what open_clip uses)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)
        images = (images - mean) / std

        # Extract image features with AMP
        with autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            features = self._model.encode_image(images)

        return features.float()  # Return fp32 for metric computation


# =============================================================================
# Metric Computation Functions
# =============================================================================

def compute_kid(
    real_features: torch.Tensor,
    generated_features: torch.Tensor,
    subset_size: int = 100,
    num_subsets: int = 50,
) -> Tuple[float, float]:
    """Compute Kernel Inception Distance between real and generated features.

    KID uses a polynomial kernel k(x,y) = (x^T y / d + 1)^3 and computes
    an unbiased MMD estimate. Unlike FID, KID has unbiased estimator and
    works well with small sample sizes.

    Args:
        real_features: Real image features [N, D].
        generated_features: Generated image features [M, D].
        subset_size: Size of random subsets for MMD estimation.
        num_subsets: Number of random subsets to average over.

    Returns:
        Tuple of (kid_mean, kid_std).
    """
    real_features = real_features.float()
    generated_features = generated_features.float()

    n_real = real_features.shape[0]
    n_gen = generated_features.shape[0]
    d = real_features.shape[1]

    # Adjust subset size if we don't have enough samples
    subset_size = min(subset_size, n_real, n_gen)

    if subset_size < 2:
        logger.warning(f"Not enough samples for KID (real={n_real}, gen={n_gen})")
        return 0.0, 0.0

    def polynomial_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Polynomial kernel k(x,y) = (x^T y / d + 1)^3."""
        return ((x @ y.T) / d + 1) ** 3

    kid_values = []

    for _ in range(num_subsets):
        # Random subset indices
        real_idx = torch.randperm(n_real)[:subset_size]
        gen_idx = torch.randperm(n_gen)[:subset_size]

        real_subset = real_features[real_idx]
        gen_subset = generated_features[gen_idx]

        # Compute kernel matrices
        k_rr = polynomial_kernel(real_subset, real_subset)
        k_gg = polynomial_kernel(gen_subset, gen_subset)
        k_rg = polynomial_kernel(real_subset, gen_subset)

        # Unbiased MMD estimate
        # MMD^2 = E[k(r,r')] + E[k(g,g')] - 2*E[k(r,g)]
        # Unbiased: exclude diagonal for same-set terms
        n = subset_size

        # Sum excluding diagonal
        k_rr_sum = k_rr.sum() - k_rr.trace()
        k_gg_sum = k_gg.sum() - k_gg.trace()

        mmd_squared = (
            k_rr_sum / (n * (n - 1)) +
            k_gg_sum / (n * (n - 1)) -
            2 * k_rg.mean()
        )

        kid_values.append(float(mmd_squared.item()))

    kid_mean = np.mean(kid_values)
    kid_std = np.std(kid_values)

    return kid_mean, kid_std


def compute_cmmd(
    real_features: torch.Tensor,
    generated_features: torch.Tensor,
    kernel_bandwidth: Optional[float] = None,
) -> float:
    """Compute CLIP Maximum Mean Discrepancy with RBF kernel.

    CMMD uses an RBF (Gaussian) kernel on CLIP embeddings. The bandwidth
    is set using the median heuristic if not provided.

    Args:
        real_features: Real image CLIP features [N, D].
        generated_features: Generated image CLIP features [M, D].
        kernel_bandwidth: RBF kernel bandwidth (sigma). If None, uses median heuristic.

    Returns:
        CMMD value (lower is better).
    """
    real_features = real_features.float()
    generated_features = generated_features.float()

    # L2 normalize features (CLIP embeddings are typically normalized)
    real_features = F.normalize(real_features, p=2, dim=1)
    generated_features = F.normalize(generated_features, p=2, dim=1)

    # Median heuristic for bandwidth if not provided
    if kernel_bandwidth is None:
        # Compute pairwise distances for a random subset
        n_sample = min(500, real_features.shape[0], generated_features.shape[0])
        real_sample = real_features[:n_sample]
        gen_sample = generated_features[:n_sample]

        # Pairwise squared distances
        all_features = torch.cat([real_sample, gen_sample], dim=0)
        dists = torch.cdist(all_features, all_features, p=2)
        median_dist = torch.median(dists[dists > 0])
        kernel_bandwidth = float(median_dist.item())

        # Avoid very small bandwidth
        kernel_bandwidth = max(kernel_bandwidth, 0.1)

    def rbf_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float) -> torch.Tensor:
        """Compute RBF kernel matrix."""
        # Pairwise squared distances
        xx = (x ** 2).sum(dim=1, keepdim=True)
        yy = (y ** 2).sum(dim=1, keepdim=True)
        xy = x @ y.T
        distances = xx + yy.T - 2 * xy
        return torch.exp(-distances / (2 * sigma ** 2))

    # Compute kernel matrices
    k_rr = rbf_kernel(real_features, real_features, kernel_bandwidth)
    k_gg = rbf_kernel(generated_features, generated_features, kernel_bandwidth)
    k_rg = rbf_kernel(real_features, generated_features, kernel_bandwidth)

    # Unbiased MMD^2 estimate
    n, m = real_features.shape[0], generated_features.shape[0]

    # Zero diagonal for unbiased estimate
    k_rr_no_diag = k_rr.clone()
    k_rr_no_diag.fill_diagonal_(0)
    k_gg_no_diag = k_gg.clone()
    k_gg_no_diag.fill_diagonal_(0)

    mmd_squared = (
        k_rr_no_diag.sum() / (n * (n - 1)) +
        k_gg_no_diag.sum() / (m * (m - 1)) -
        2 * k_rg.sum() / (n * m)
    )

    # Return sqrt for CMMD (like standard deviation)
    return float(torch.sqrt(torch.clamp(mmd_squared, min=0)).item())


def compute_fid(
    real_features: torch.Tensor,
    generated_features: torch.Tensor,
) -> float:
    """Compute Fréchet Inception Distance between real and generated features.

    FID measures the Fréchet distance between two multivariate Gaussians
    fitted to the feature distributions.

    FID = ||mu_r - mu_g||^2 + Tr(Sigma_r + Sigma_g - 2*sqrt(Sigma_r @ Sigma_g))

    Args:
        real_features: Real image features [N, D].
        generated_features: Generated image features [M, D].

    Returns:
        FID value (lower is better).
    """
    real_features = real_features.float().cpu().numpy()
    generated_features = generated_features.float().cpu().numpy()

    # Compute mean and covariance
    mu_r = np.mean(real_features, axis=0)
    mu_g = np.mean(generated_features, axis=0)

    sigma_r = np.cov(real_features, rowvar=False)
    sigma_g = np.cov(generated_features, rowvar=False)

    # Handle edge case of single sample
    if real_features.shape[0] == 1:
        sigma_r = np.zeros_like(sigma_r)
    if generated_features.shape[0] == 1:
        sigma_g = np.zeros_like(sigma_g)

    # Compute FID
    diff = mu_r - mu_g
    diff_squared = np.sum(diff ** 2)

    # Matrix square root using scipy
    try:
        from scipy import linalg
        covmean, _ = linalg.sqrtm(sigma_r @ sigma_g, disp=False)

        # Handle numerical issues
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        trace_term = np.trace(sigma_r + sigma_g - 2 * covmean)

        fid = diff_squared + trace_term
        return float(fid)

    except Exception as e:
        logger.warning(f"FID computation failed: {e}")
        return float('inf')


# =============================================================================
# Reference Feature Cache
# =============================================================================

class ReferenceFeatureCache:
    """Caches reference features from train/val datasets.

    Features are extracted once at training start and reused for all epochs.
    Cached to disk for fast loading on training resumption.

    Args:
        resnet_extractor: ResNet50 feature extractor.
        biomed_extractor: BiomedCLIP feature extractor.
        cache_dir: Directory for caching features.
        device: PyTorch device.
        batch_size: Batch size for feature extraction.
    """

    def __init__(
        self,
        resnet_extractor: ResNet50Features,
        biomed_extractor: BiomedCLIPFeatures,
        cache_dir: Path,
        device: torch.device,
        batch_size: int = 32,
    ) -> None:
        self.resnet = resnet_extractor
        self.biomed = biomed_extractor
        self.cache_dir = Path(cache_dir)
        self.device = device
        self.batch_size = batch_size

        # Feature storage
        self.train_resnet: Optional[torch.Tensor] = None
        self.train_biomed: Optional[torch.Tensor] = None
        self.val_resnet: Optional[torch.Tensor] = None
        self.val_biomed: Optional[torch.Tensor] = None

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _extract_features_from_loader(
        self,
        dataloader: DataLoader,
        extractor: nn.Module,
        name: str,
        max_samples: Optional[int] = None,
    ) -> torch.Tensor:
        """Extract features from a dataloader, filtering for positive masks.

        Args:
            dataloader: DataLoader providing batches.
            extractor: Feature extractor module.
            name: Name for logging.
            max_samples: Maximum samples to process.

        Returns:
            Feature tensor [N, D].
        """
        all_features = []
        sample_count = 0

        for batch in dataloader:
            # Handle different batch formats
            if isinstance(batch, dict):
                images = batch.get('images', batch.get('image'))
                masks = batch.get('seg', batch.get('mask', batch.get('labels')))
            elif isinstance(batch, (tuple, list)):
                if len(batch) == 2:
                    images, masks = batch
                else:
                    images = batch[0]
                    masks = None
            else:
                images = batch
                masks = None

            images = images.to(self.device)

            # Filter for positive masks (has tumor)
            if masks is not None:
                masks = masks.to(self.device)
                # Check if mask has any positive pixels per sample
                positive_mask = masks.sum(dim=(1, 2, 3)) > 0
                if positive_mask.sum() == 0:
                    continue
                images = images[positive_mask]

            # Extract in sub-batches
            for i in range(0, images.shape[0], self.batch_size):
                sub_batch = images[i:i + self.batch_size]
                features = extractor.extract_features(sub_batch)
                all_features.append(features.cpu())
                sample_count += sub_batch.shape[0]

                if max_samples and sample_count >= max_samples:
                    break

            if max_samples and sample_count >= max_samples:
                break

        if not all_features:
            logger.warning(f"No features extracted for {name}")
            return torch.empty(0)

        features = torch.cat(all_features, dim=0)
        logger.info(f"Extracted {features.shape[0]} {name} features")
        return features

    def extract_and_cache(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        experiment_id: str,
    ) -> None:
        """Extract and cache reference features.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            experiment_id: Unique experiment identifier for cache file.
        """
        cache_file = self.cache_dir / f"{experiment_id}_reference_features.pt"

        if cache_file.exists():
            logger.info(f"Loading cached reference features from {cache_file}")
            cached = torch.load(cache_file, map_location='cpu', weights_only=True)
            # Support both old (inception) and new (resnet) cache format
            self.train_resnet = cached.get('train_resnet', cached.get('train_inception'))
            self.train_biomed = cached['train_biomed']
            self.val_resnet = cached.get('val_resnet', cached.get('val_inception'))
            self.val_biomed = cached['val_biomed']
            logger.info(
                f"Loaded: train={self.train_resnet.shape[0]}, "
                f"val={self.val_resnet.shape[0]} samples"
            )
            return

        logger.info("Extracting reference features (this happens once per experiment)...")

        # Extract train features
        logger.info("  Extracting train ResNet50 features...")
        self.train_resnet = self._extract_features_from_loader(
            train_loader, self.resnet, "train_resnet"
        )

        logger.info("  Extracting train BiomedCLIP features...")
        self.train_biomed = self._extract_features_from_loader(
            train_loader, self.biomed, "train_biomed"
        )

        # Extract val features
        logger.info("  Extracting val ResNet50 features...")
        self.val_resnet = self._extract_features_from_loader(
            val_loader, self.resnet, "val_resnet"
        )

        logger.info("  Extracting val BiomedCLIP features...")
        self.val_biomed = self._extract_features_from_loader(
            val_loader, self.biomed, "val_biomed"
        )

        # Cache to disk
        torch.save({
            'train_resnet': self.train_resnet,
            'train_biomed': self.train_biomed,
            'val_resnet': self.val_resnet,
            'val_biomed': self.val_biomed,
        }, cache_file)
        logger.info(f"Cached features to {cache_file}")


# =============================================================================
# Main Generation Metrics Class
# =============================================================================

class GenerationMetrics:
    """Tracks generation quality during diffusion training.

    Computes KID and CMMD metrics comparing generated samples against
    train and validation distributions to detect overfitting.

    Args:
        config: GenerationMetricsConfig instance.
        device: PyTorch device.
        run_dir: Run directory for caching.

    Example:
        config = GenerationMetricsConfig(enabled=True)
        metrics = GenerationMetrics(config, device, run_dir)

        # At training start
        metrics.set_fixed_conditioning(train_dataset, num_masks=500)
        metrics.cache_reference_features(train_loader, val_loader, "exp1")

        # Every epoch
        results = metrics.compute_epoch_metrics(model, strategy, mode)
        # Returns: {'KID_train_mean': 0.05, 'KID_val_mean': 0.06, ...}
    """

    def __init__(
        self,
        config: GenerationMetricsConfig,
        device: torch.device,
        run_dir: Path,
    ) -> None:
        self.config = config
        self.device = device
        self.run_dir = Path(run_dir)

        # Initialize feature extractors (lazy-loaded)
        self.resnet = ResNet50Features(device, cache_dir=Path(config.cache_dir))
        self.biomed = BiomedCLIPFeatures(device, cache_dir=config.cache_dir)

        # Reference feature cache
        self.cache = ReferenceFeatureCache(
            self.resnet,
            self.biomed,
            Path(config.cache_dir),
            device,
            config.feature_batch_size,
        )

        # Fixed conditioning masks (loaded once, used every epoch)
        self.fixed_conditioning_masks: Optional[torch.Tensor] = None
        self.fixed_gt_images: Optional[torch.Tensor] = None

    def set_fixed_conditioning(
        self,
        train_dataset: Dataset,
        num_masks: int = 500,
        seg_channel_idx: int = 1,
    ) -> None:
        """Load fixed conditioning masks from training dataset.

        Uses the same masks every epoch for reproducible comparisons.

        Args:
            train_dataset: Training dataset to sample from.
            num_masks: Number of masks to sample.
            seg_channel_idx: Channel index for segmentation mask.
        """
        logger.info(f"Sampling {num_masks} fixed conditioning masks...")

        masks = []
        gt_images = []
        attempts = 0
        max_attempts = len(train_dataset)

        # Use fixed seed for reproducibility
        rng = torch.Generator()
        rng.manual_seed(42)

        while len(masks) < num_masks and attempts < max_attempts:
            idx = int(torch.randint(0, len(train_dataset), (1,), generator=rng).item())
            data = train_dataset[idx]

            # Handle tuple (images, seg) format
            if isinstance(data, tuple):
                images, seg_arr = data
                if hasattr(images, '__array__'):
                    images = torch.from_numpy(images).float()
                else:
                    images = images.float() if isinstance(images, torch.Tensor) else torch.tensor(images).float()
                if hasattr(seg_arr, '__array__'):
                    seg_arr = torch.from_numpy(seg_arr).float()
                else:
                    seg_arr = seg_arr.float() if isinstance(seg_arr, torch.Tensor) else torch.tensor(seg_arr).float()
                tensor = torch.cat([images, seg_arr], dim=0)
            else:
                if hasattr(data, '__array__'):
                    tensor = torch.from_numpy(data).float()
                else:
                    tensor = data.float() if isinstance(data, torch.Tensor) else torch.tensor(data).float()

            seg = tensor[seg_channel_idx:seg_channel_idx + 1, :, :]

            if seg.sum() > 0:  # Has positive mask
                masks.append(seg)
                if seg_channel_idx > 0:
                    gt_images.append(tensor[0:seg_channel_idx, :, :])
            attempts += 1

        if len(masks) < num_masks:
            logger.warning(f"Only found {len(masks)} positive masks (requested {num_masks})")

        self.fixed_conditioning_masks = torch.stack(masks).to(self.device)
        if gt_images:
            self.fixed_gt_images = torch.stack(gt_images).to(self.device)

        logger.info(f"Loaded {len(masks)} fixed conditioning masks")

    def cache_reference_features(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        experiment_id: str,
    ) -> None:
        """Cache reference features from train and val datasets.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            experiment_id: Unique experiment identifier.
        """
        self.cache.extract_and_cache(train_loader, val_loader, experiment_id)

    @torch.no_grad()
    def _generate_samples(
        self,
        model: nn.Module,
        strategy: Any,
        mode: Any,
        num_samples: int,
        num_steps: int,
        batch_size: int = 16,
    ) -> torch.Tensor:
        """Generate samples using fixed conditioning in batches.

        Args:
            model: Diffusion model.
            strategy: Diffusion strategy (DDPM/RFlow).
            mode: Training mode (bravo/dual/seg).
            num_samples: Number of samples to generate.
            num_steps: Number of denoising steps.
            batch_size: Batch size for generation (to avoid OOM).

        Returns:
            Generated samples [N, C, H, W].
        """
        if self.fixed_conditioning_masks is None:
            raise RuntimeError("Must call set_fixed_conditioning() first")

        # Round up to full batches for torch.compile consistency
        # e.g., 100 samples with batch_size=16 → 112 samples (7 full batches)
        num_batches = math.ceil(num_samples / batch_size)
        num_to_use = num_batches * batch_size

        # Cap at available masks
        num_available = self.fixed_conditioning_masks.shape[0]
        if num_to_use > num_available:
            # Round down to full batches if we don't have enough masks
            num_batches = num_available // batch_size
            num_to_use = num_batches * batch_size
            if num_to_use == 0:
                raise RuntimeError(
                    f"Not enough conditioning masks ({num_available}) for batch_size={batch_size}"
                )

        # Get model config for output channels
        model_config = mode.get_model_config()
        out_channels = model_config['out_channels']

        model.eval()
        all_samples = []

        # Generate in batches to avoid OOM with CUDA graphs
        for start_idx in range(0, num_to_use, batch_size):
            end_idx = min(start_idx + batch_size, num_to_use)
            masks = self.fixed_conditioning_masks[start_idx:end_idx]

            # Generate noise
            noise = torch.randn_like(masks)

            if out_channels == 2:  # Dual mode
                noise_pre = torch.randn_like(masks)
                noise_gd = torch.randn_like(masks)
                model_input = torch.cat([noise_pre, noise_gd, masks], dim=1)
            else:  # Single channel modes (bravo, seg)
                model_input = torch.cat([noise, masks], dim=1)

            # Generate samples
            with autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
                samples = strategy.generate(
                    model, model_input, num_steps=num_steps, device=self.device
                )

            # Move to CPU immediately to free GPU memory
            all_samples.append(torch.clamp(samples.float(), 0, 1).cpu())

            # Clear intermediate tensors
            del samples, model_input, noise, masks
            if out_channels == 2:
                del noise_pre, noise_gd

        # Concatenate all samples
        result = torch.cat(all_samples, dim=0)

        # Free list memory
        del all_samples
        torch.cuda.empty_cache()

        return result.to(self.device)

    def _extract_features_batched(
        self,
        samples: torch.Tensor,
        extractor: nn.Module,
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """Extract features from samples in batches.

        Args:
            samples: Input samples [N, C, H, W].
            extractor: Feature extractor (ResNet50 or BiomedCLIP).
            batch_size: Batch size for extraction (default: config.feature_batch_size).

        Returns:
            Feature tensor [N, D].
        """
        if batch_size is None:
            batch_size = self.config.feature_batch_size
        all_features = []
        for start_idx in range(0, samples.shape[0], batch_size):
            end_idx = min(start_idx + batch_size, samples.shape[0])
            batch = samples[start_idx:end_idx]
            features = extractor.extract_features(batch)
            all_features.append(features.cpu())
            del features, batch

        result = torch.cat(all_features, dim=0)
        del all_features
        return result

    def _compute_metrics_against_reference(
        self,
        gen_resnet: torch.Tensor,
        gen_biomed: torch.Tensor,
        ref_resnet: torch.Tensor,
        ref_biomed: torch.Tensor,
        prefix: str = "",
    ) -> Dict[str, float]:
        """Compute KID and CMMD against reference features.

        Args:
            gen_resnet: Generated ResNet50 features.
            gen_biomed: Generated BiomedCLIP features.
            ref_resnet: Reference ResNet50 features.
            ref_biomed: Reference BiomedCLIP features.
            prefix: Prefix for metric names (e.g., "extended_").

        Returns:
            Dictionary of metric values.
        """
        results = {}

        # KID
        kid_mean, kid_std = compute_kid(
            ref_resnet.to(self.device),
            gen_resnet.to(self.device),
        )
        results[f'{prefix}KID_mean'] = kid_mean
        results[f'{prefix}KID_std'] = kid_std

        # CMMD
        cmmd = compute_cmmd(
            ref_biomed.to(self.device),
            gen_biomed.to(self.device),
        )
        results[f'{prefix}CMMD'] = cmmd

        return results

    def compute_epoch_metrics(
        self,
        model: nn.Module,
        strategy: Any,
        mode: Any,
    ) -> Dict[str, float]:
        """Compute quick generation metrics (every epoch).

        Uses fewer samples and steps for fast feedback.

        Args:
            model: Diffusion model.
            strategy: Diffusion strategy.
            mode: Training mode.

        Returns:
            Dictionary with KID/CMMD vs train and val.
        """
        # Preserve RNG state (per pitfall #42)
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state(self.device)

        try:
            # Generate samples
            samples = self._generate_samples(
                model, strategy, mode,
                self.config.samples_per_epoch,
                self.config.steps_per_epoch,
            )

            # Extract features in batches
            gen_resnet = self._extract_features_batched(samples, self.resnet)
            gen_biomed = self._extract_features_batched(samples, self.biomed)

            # Free samples
            del samples
            torch.cuda.empty_cache()

            results = {}

            # Metrics vs train
            train_metrics = self._compute_metrics_against_reference(
                gen_resnet, gen_biomed,
                self.cache.train_resnet, self.cache.train_biomed,
                prefix="",
            )
            for key, value in train_metrics.items():
                results[f'{key}_train'] = value

            # Metrics vs val
            val_metrics = self._compute_metrics_against_reference(
                gen_resnet, gen_biomed,
                self.cache.val_resnet, self.cache.val_biomed,
                prefix="",
            )
            for key, value in val_metrics.items():
                results[f'{key}_val'] = value

            return results

        finally:
            # Restore RNG state
            torch.set_rng_state(rng_state)
            torch.cuda.set_rng_state(cuda_rng_state, self.device)

    def compute_extended_metrics(
        self,
        model: nn.Module,
        strategy: Any,
        mode: Any,
    ) -> Dict[str, float]:
        """Compute extended generation metrics (every N epochs).

        Uses more samples and steps for detailed analysis.

        Args:
            model: Diffusion model.
            strategy: Diffusion strategy.
            mode: Training mode.

        Returns:
            Dictionary with extended_ prefixed KID/CMMD metrics.
        """
        # Preserve RNG state
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state(self.device)

        try:
            # Generate samples
            samples = self._generate_samples(
                model, strategy, mode,
                self.config.samples_extended,
                self.config.steps_extended,
            )

            # Extract features in batches
            gen_resnet = self._extract_features_batched(samples, self.resnet)
            gen_biomed = self._extract_features_batched(samples, self.biomed)

            # Free samples
            del samples
            torch.cuda.empty_cache()

            results = {}

            # Metrics vs train
            train_metrics = self._compute_metrics_against_reference(
                gen_resnet, gen_biomed,
                self.cache.train_resnet, self.cache.train_biomed,
                prefix="extended_",
            )
            for key, value in train_metrics.items():
                results[f'{key}_train'] = value

            # Metrics vs val
            val_metrics = self._compute_metrics_against_reference(
                gen_resnet, gen_biomed,
                self.cache.val_resnet, self.cache.val_biomed,
                prefix="extended_",
            )
            for key, value in val_metrics.items():
                results[f'{key}_val'] = value

            return results

        finally:
            # Restore RNG state
            torch.set_rng_state(rng_state)
            torch.cuda.set_rng_state(cuda_rng_state, self.device)

    def compute_test_metrics(
        self,
        model: nn.Module,
        strategy: Any,
        mode: Any,
        test_loader: Optional[DataLoader] = None,
    ) -> Dict[str, float]:
        """Compute full test generation metrics with FID.

        Uses the most samples and steps for final evaluation.
        Optionally compares against test set if loader provided.

        Args:
            model: Diffusion model.
            strategy: Diffusion strategy.
            mode: Training mode.
            test_loader: Optional test data loader for FID vs test.

        Returns:
            Dictionary with FID, KID, CMMD test metrics.
        """
        # Generate samples
        samples = self._generate_samples(
            model, strategy, mode,
            self.config.samples_test,
            self.config.steps_test,
        )

        # Extract features in batches
        gen_resnet = self._extract_features_batched(samples, self.resnet)
        gen_biomed = self._extract_features_batched(samples, self.biomed)

        # Free samples
        del samples
        torch.cuda.empty_cache()

        results = {}

        # If test loader provided, compute metrics vs test
        if test_loader is not None:
            # Extract test features
            test_resnet = self.cache._extract_features_from_loader(
                test_loader, self.resnet, "test_resnet",
                max_samples=self.config.samples_test,
            )
            test_biomed = self.cache._extract_features_from_loader(
                test_loader, self.biomed, "test_biomed",
                max_samples=self.config.samples_test,
            )

            # FID (only for test)
            fid = compute_fid(test_resnet, gen_resnet)
            results['FID'] = fid

            # KID and CMMD vs test
            test_metrics = self._compute_metrics_against_reference(
                gen_resnet, gen_biomed,
                test_resnet, test_biomed,
                prefix="",
            )
            results.update(test_metrics)
        else:
            # Use val as reference if no test loader
            val_metrics = self._compute_metrics_against_reference(
                gen_resnet, gen_biomed,
                self.cache.val_resnet, self.cache.val_biomed,
                prefix="",
            )
            results.update(val_metrics)

            # FID vs val
            fid = compute_fid(
                self.cache.val_resnet,
                gen_resnet,
            )
            results['FID'] = fid

        return results
