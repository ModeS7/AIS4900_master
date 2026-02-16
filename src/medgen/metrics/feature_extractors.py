"""
Feature extractors for generation quality metrics.

Provides pre-trained neural network feature extractors for computing
distributional metrics (KID, CMMD, FID) between real and generated images.

Classes:
    ResNet50Features: ResNet50 feature extractor (ImageNet or RadImageNet)
    BiomedCLIPFeatures: BiomedCLIP feature extractor (medical domain)

Usage:
    from medgen.metrics.feature_extractors import ResNet50Features, BiomedCLIPFeatures

    # ResNet50 for KID/FID
    resnet = ResNet50Features(device, network_type='imagenet')
    features = resnet.extract_features(images)  # [B, 2048]

    # BiomedCLIP for CMMD (medical domain-aware)
    biomed = BiomedCLIPFeatures(device)
    features = biomed.extract_features(images)  # [B, 512]
"""
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

logger = logging.getLogger(__name__)


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
        cache_dir: Path | None = None,
        compile_model: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        self.network_type = network_type
        self.cache_dir = cache_dir
        self.compile_model = compile_model
        self._model: nn.Module | None = None

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
                state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
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
                except (FileNotFoundError, RuntimeError, ImportError) as e:
                    logger.error(
                        f"RadImageNet not available ({e}), falling back to ImageNet. "
                        f"Metrics will NOT be comparable with RadImageNet-based runs."
                    )
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

        # Compile for faster inference (skip if compile_model=False for load/unload pattern)
        if self.compile_model:
            try:
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("ResNet50 compiled with torch.compile")
            except RuntimeError as e:
                logger.warning(f"torch.compile failed for ResNet50: {e}")
        else:
            logger.info("ResNet50 loaded without torch.compile (load/unload mode)")

        self._model = model

    @torch.no_grad()
    def extract_features(
        self,
        images: torch.Tensor,
        use_amp: bool = True,
    ) -> torch.Tensor:
        """Extract ResNet50 features with AMP support.

        Args:
            images: Images [B, C, H, W] or [B, C, D, H, W] in [0, 1] range, 1-3 channels.
                For 3D volumes, middle slice is extracted for feature computation.
            use_amp: Whether to use automatic mixed precision.

        Returns:
            Feature tensor [B, 2048].
        """
        self._ensure_model()

        # Handle 3D volumes [B, C, D, H, W] by taking middle slice -> [B, C, H, W]
        if images.dim() == 5:
            mid_slice = images.shape[2] // 2
            images = images[:, :, mid_slice]  # [B, C, H, W]

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
        with autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=use_amp):
            features = self._model(images)

        # Global average pooling if needed (shouldn't be with fc=Identity)
        if len(features.shape) > 2:
            features = features.mean([2, 3])

        return features.float()  # Return fp32 for metric computation

    def unload(self) -> None:
        """Unload the model to free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            torch.cuda.empty_cache()
            logger.info("ResNet50 unloaded from GPU")


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
        cache_dir: str | None = None,
        compile_model: bool = True,
    ) -> None:
        super().__init__()
        self.device = device
        self.cache_dir = cache_dir
        self.compile_model = compile_model
        self._model: nn.Module | None = None
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
            ) from None

        model_name = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

        logger.info(f"Loading BiomedCLIP from {model_name}...")

        # Suppress verbose open_clip/HF Hub logging during model creation
        _open_clip_loggers = ['root', 'open_clip', 'huggingface_hub']
        _prev_levels = {}
        for name in _open_clip_loggers:
            _lg = logging.getLogger(name if name != 'root' else None)
            _prev_levels[name] = _lg.level
            _lg.setLevel(logging.WARNING)

        try:
            self._model, _, self._preprocess = open_clip.create_model_and_transforms(
                model_name,
                device=self.device,
            )
        finally:
            for name, level in _prev_levels.items():
                logging.getLogger(name if name != 'root' else None).setLevel(level)

        self._model.eval()

        # Freeze all parameters
        for param in self._model.parameters():
            param.requires_grad = False

        # Compile for faster inference (skip if compile_model=False for load/unload pattern)
        if self.compile_model:
            try:
                self._model = torch.compile(self._model, mode="reduce-overhead")
                logger.info("BiomedCLIP compiled with torch.compile")
            except RuntimeError as e:
                logger.warning(f"torch.compile failed for BiomedCLIP: {e}")
        else:
            logger.info("BiomedCLIP loaded without torch.compile (load/unload mode)")

        logger.info("Loaded BiomedCLIP for feature extraction")

    @torch.no_grad()
    def extract_features(
        self,
        images: torch.Tensor,
        use_amp: bool = True,
    ) -> torch.Tensor:
        """Extract BiomedCLIP image features with AMP support.

        Args:
            images: Images [B, C, H, W] or [B, C, D, H, W] in [0, 1] range, 1-3 channels.
                For 3D volumes, middle slice is extracted for feature computation.
            use_amp: Whether to use automatic mixed precision.

        Returns:
            Feature tensor [B, 512] (CLIP embedding dimension).
        """
        self._ensure_model()

        # Handle 3D volumes [B, C, D, H, W] by taking middle slice -> [B, C, H, W]
        if images.dim() == 5:
            mid_slice = images.shape[2] // 2
            images = images[:, :, mid_slice]  # [B, C, H, W]

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
        with autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=use_amp):
            features = self._model.encode_image(images)

        return features.float()  # Return fp32 for metric computation

    def unload(self) -> None:
        """Unload the model to free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self._processor = None
            torch.cuda.empty_cache()
            logger.info("BiomedCLIP unloaded from GPU")
