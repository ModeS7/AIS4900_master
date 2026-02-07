"""
Training mode implementations for diffusion models.

This module defines different training modes for the diffusion model,
including unconditional segmentation generation and conditional image
generation modes.
"""
import logging
from abc import ABC, abstractmethod
from typing import Any

import torch

from medgen.core.constants import DEFAULT_DUAL_IMAGE_KEYS

logger = logging.getLogger(__name__)


def _validate_tensor(value: Any, name: str) -> torch.Tensor:
    """Validate that value is a torch.Tensor.

    Args:
        value: Value to validate.
        name: Name for error messages.

    Returns:
        The tensor if valid.

    Raises:
        ValueError: If value is None.
        TypeError: If value is not a Tensor.
    """
    if value is None:
        raise ValueError(f"{name} is None, expected torch.Tensor")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} is {type(value).__name__}, expected torch.Tensor")
    return value


def _validate_dict_keys(batch: dict[str, Any], required_keys: list[str], context: str) -> None:
    """Validate that dict contains all required keys.

    Args:
        batch: Dictionary to validate.
        required_keys: Keys that must be present.
        context: Context string for error messages.

    Raises:
        KeyError: If a required key is missing.
    """
    for key in required_keys:
        if key not in batch:
            raise KeyError(f"{context}: missing required key '{key}'")


def _move_to_device(v: Any, device: torch.device) -> Any:
    """Move value to device, handling various types.

    Args:
        v: Value to move (may be None, Tensor, MetaTensor, dict, or scalar).
        device: Target device.

    Returns:
        Value on target device, or unchanged for scalars/None.
    """
    if v is None:
        return None
    if isinstance(v, dict):
        return {k: _move_to_device(val, device) for k, val in v.items()}
    if hasattr(v, 'as_tensor'):
        return v.as_tensor().to(device, non_blocking=True)
    if isinstance(v, torch.Tensor):
        return v.to(device, non_blocking=True)
    return v  # Scalars, strings unchanged


def _to_device(batch: torch.Tensor | dict[str, Any] | tuple, device: torch.device) -> torch.Tensor | dict[str, Any] | tuple:
    """Convert batch tensor, dict, or tuple to device, handling MONAI MetaTensors.

    MetaTensors from MONAI need to be converted to regular tensors
    before being moved to device to avoid metadata issues.

    Args:
        batch: Input tensor (may be MetaTensor), dict of tensors, or tuple of tensors.
        device: Target device.

    Returns:
        Tensor, dict of tensors, or tuple of tensors on target device.

    Raises:
        TypeError: If batch is not Tensor, Dict, or tuple.
    """
    if isinstance(batch, dict):
        # Handle dict batch format (e.g., from 3D dataloaders)
        # Also handles nested dicts
        return {k: _move_to_device(v, device) for k, v in batch.items()}
    elif isinstance(batch, tuple):
        # Handle tuple batch format (e.g., 3D seg mode: (seg_tensor, size_bins))
        # Also handles dicts within tuples
        return tuple(_move_to_device(v, device) for v in batch)
    elif isinstance(batch, torch.Tensor):
        return _move_to_device(batch, device)
    else:
        raise TypeError(f"_to_device: expected Tensor, Dict, or tuple, got {type(batch).__name__}")


def _is_latent_batch(batch: Any) -> bool:
    """Check if batch is from a latent dataloader.

    Latent batches are dicts with 'latent' key containing pre-encoded data.

    Args:
        batch: Batch from dataloader.

    Returns:
        True if this is a latent batch format.
    """
    return isinstance(batch, dict) and 'latent' in batch


def _prepare_latent_batch(
    batch: dict[str, Any],
    device: torch.device,
    is_conditional: bool = True
) -> dict[str, Any]:
    """Prepare latent batch for training.

    Latent batches have format:
    - 'latent': Tensor [B, C_latent, H_latent, W_latent]
    - 'seg_mask': Optional Tensor [1, H, W] (pixel-space for conditioning)

    Args:
        batch: Dict from LatentDataset.
        device: Target device.
        is_conditional: Whether mode uses conditioning.

    Returns:
        Prepared batch dictionary.

    Raises:
        KeyError: If batch is missing required 'latent' key.
    """
    # Validate required key
    if 'latent' not in batch:
        raise KeyError(
            "_prepare_latent_batch requires 'latent' key in batch. "
            "Ensure latent dataloader is configured correctly."
        )

    latent = batch['latent'].to(device, non_blocking=True)
    seg_mask = batch.get('seg_mask')

    if seg_mask is not None and is_conditional:
        seg_mask = seg_mask.to(device, non_blocking=True)
    else:
        seg_mask = None

    return {
        'images': latent,
        'labels': seg_mask,
        'is_latent': True,  # Flag for downstream processing
    }


class TrainingMode(ABC):
    """Abstract base class for different training modes.

    Defines the interface for training modes that determine how batches
    are prepared and how the model input/output channels are configured.
    """

    @property
    @abstractmethod
    def is_conditional(self) -> bool:
        """Whether this mode uses conditioning.

        Returns:
            True if mode uses conditioning signal, False otherwise.
        """
        pass

    @abstractmethod
    def prepare_batch(
        self, batch: torch.Tensor, device: torch.device
    ) -> dict[str, Any]:
        """Prepare batch data for training.

        Args:
            batch: Raw batch tensor from dataloader.
            device: Target device for tensors.

        Returns:
            Dictionary with 'images' and optionally 'labels' keys.
        """
        pass

    @abstractmethod
    def get_model_config(self) -> dict[str, int]:
        """Return model configuration.

        Returns:
            Dictionary with 'in_channels' and 'out_channels' keys.
        """
        pass

    @abstractmethod
    def format_model_input(
        self,
        noisy_images: torch.Tensor | dict[str, torch.Tensor],
        labels_dict: dict[str, torch.Tensor | None]
    ) -> torch.Tensor:
        """Format input for model forward pass.

        Args:
            noisy_images: Noisy images (tensor or dict for dual mode).
            labels_dict: Dictionary containing optional labels/conditioning.

        Returns:
            Formatted model input tensor.
        """
        pass


class SegmentationMode(TrainingMode):
    """Unconditional segmentation mask generation mode.

    Mode 1: Train to generate segmentation masks only (unconditional).

    Input: [B, 1, H, W] - segmentation masks
    Output: [B, 1, H, W] - denoised segmentation masks

    This is a pure generative model for masks without conditioning.
    """

    @property
    def is_conditional(self) -> bool:
        """Segmentation mode is unconditional."""
        return False

    def prepare_batch(
        self, batch: torch.Tensor | dict[str, Any], device: torch.device
    ) -> dict[str, Any]:
        """Prepare segmentation batch.

        Args:
            batch: One of:
                - Tensor [B, 1, H, W]: 2D format
                - Dict with 'image': 3D format from Volume3DDataset
                - Dict with 'latent': latent space format
            device: Target device.

        Returns:
            Dictionary with images and None labels.
        """
        if _is_latent_batch(batch):
            return _prepare_latent_batch(batch, device, is_conditional=False)

        batch = _to_device(batch, device)

        # Handle dict batch format (from unified or 3D loaders)
        if isinstance(batch, dict):
            images = batch.get('image')
            if images is None:
                raise ValueError("Dict batch missing 'image' key")
            images = _validate_tensor(images, "SegmentationMode batch['image']")

            # Validate: 2D=[B,C,H,W], 3D=[B,C,D,H,W] (loaders must provide channel dim)
            if images.dim() not in (4, 5):
                raise ValueError(
                    f"SegmentationMode: expected 4D or 5D image tensor, "
                    f"got {images.dim()}D with shape {list(images.shape)}"
                )

            return {
                'images': images,
                'labels': None,
            }

        # Tensor input
        if isinstance(batch, torch.Tensor):
            return {
                'images': batch,
                'labels': None
            }

        raise TypeError(
            f"SegmentationMode.prepare_batch: expected Tensor or Dict, "
            f"got {type(batch).__name__}"
        )

    def get_model_config(self) -> dict[str, int]:
        """Get model channel configuration.

        Model takes: [noisy_seg] = 1 input channel
        Model outputs: [noise_pred] or [velocity_pred] = 1 output channel

        Returns:
            Channel configuration dictionary.
        """
        return {
            'in_channels': 1,
            'out_channels': 1
        }

    def format_model_input(
        self,
        noisy_images: torch.Tensor,
        labels_dict: dict[str, torch.Tensor | None]
    ) -> torch.Tensor:
        """Format model input (no conditioning).

        Args:
            noisy_images: [B, 1, H, W] noisy segmentation masks.
            labels_dict: Not used in unconditional mode.

        Returns:
            [B, 1, H, W] - same as input.
        """
        return noisy_images


class ConditionalSingleMode(TrainingMode):
    """Conditional single image generation mode.

    Mode 2: Train single image (BRAVO) conditioned on segmentation mask.

    Input: [B, 2, H, W] - [BRAVO, seg_mask]
    Output: [B, 1, H, W] - denoised BRAVO image

    Uses segmentation mask as conditioning signal.
    """

    @property
    def is_conditional(self) -> bool:
        """Conditional single mode uses conditioning."""
        return True

    def prepare_batch(
        self, batch: torch.Tensor | dict[str, Any], device: torch.device
    ) -> dict[str, Any]:
        """Prepare conditional single batch.

        Args:
            batch: One of:
                - Tensor [B, 2, H, W]: 2D format (channel 0: BRAVO, channel 1: seg)
                - Dict with 'image' and optional 'seg': 3D format from VAE dataloaders
                - Dict with 'latent': latent space format
            device: Target device.

        Returns:
            Dictionary with separated images and labels.
        """
        if _is_latent_batch(batch):
            return _prepare_latent_batch(batch, device, is_conditional=True)

        batch = _to_device(batch, device)

        # Handle dict batch format (from unified or 3D loaders)
        if isinstance(batch, dict):
            # Dict format: {'image': [B, C, H, W] or [B, D, H, W], 'seg': ...}
            images = batch.get('image')
            if images is None:
                raise ValueError("Dict batch missing 'image' key")

            # Validate: 2D=[B,C,H,W], 3D=[B,C,D,H,W] (loaders must provide channel dim)
            if images.dim() not in (4, 5):
                raise ValueError(
                    f"ConditionalSingleMode: expected 4D or 5D image tensor, "
                    f"got {images.dim()}D with shape {list(images.shape)}"
                )

            seg_masks = batch.get('seg')
            if seg_masks is not None and seg_masks.dim() not in (4, 5):
                raise ValueError(
                    f"ConditionalSingleMode: expected 4D or 5D seg tensor, "
                    f"got {seg_masks.dim()}D with shape {list(seg_masks.shape)}"
                )

            return {
                'images': images,
                'labels': seg_masks,
            }

        # Handle 2D tensor format [B, 2, H, W]
        bravo_images = batch[:, 0:1]
        seg_masks = batch[:, 1:2]

        return {
            'images': bravo_images,
            'labels': seg_masks
        }

    def get_model_config(self) -> dict[str, int]:
        """Get model channel configuration.

        Model takes: [noisy_bravo, seg_mask] = 2 input channels
        Model outputs: [noise_pred] or [velocity_pred] = 1 output channel

        Returns:
            Channel configuration dictionary.
        """
        return {
            'in_channels': 2,
            'out_channels': 1
        }

    def format_model_input(
        self,
        noisy_images: torch.Tensor,
        labels_dict: dict[str, torch.Tensor | None]
    ) -> torch.Tensor:
        """Concatenate noisy BRAVO with segmentation mask.

        Args:
            noisy_images: [B, 1, H, W] noisy BRAVO images.
            labels_dict: Dictionary with 'labels' key containing seg masks.

        Returns:
            [B, 2, H, W] - [noisy_bravo, seg_mask] concatenated.
        """
        return torch.cat([noisy_images, labels_dict['labels']], dim=1)


class ConditionalDualMode(TrainingMode):
    """Conditional dual image generation mode.

    Mode 3: Train two images (T1 pre + T1 gd) conditioned on segmentation mask.

    Input: [B, 3, H, W] - [T1_pre, T1_gd, seg_mask]
    Output: [B, 2, H, W] - [denoised_T1_pre, denoised_T1_gd]

    Uses segmentation mask as conditioning. Model learns to keep both
    images anatomically consistent.
    """

    def __init__(self, image_keys: list[str] = None) -> None:
        """Initialize conditional dual mode.

        Args:
            image_keys: Names of the two image types to train.
                Default: DEFAULT_DUAL_IMAGE_KEYS ('t1_pre', 't1_gd')
        """
        if image_keys is None:
            image_keys = DEFAULT_DUAL_IMAGE_KEYS.copy()
        if len(image_keys) != 2:
            raise ValueError(f"ConditionalDualMode requires exactly 2 image types, got {len(image_keys)}: {image_keys}")
        self.image_keys: list[str] = image_keys

    @property
    def is_conditional(self) -> bool:
        """Conditional dual mode uses conditioning."""
        return True

    def prepare_batch(
        self, batch: torch.Tensor | dict[str, Any], device: torch.device
    ) -> dict[str, Any]:
        """Prepare conditional dual batch.

        Args:
            batch: One of:
                - Tensor [B, 3, H, W]: 2D format (ch0: T1 pre, ch1: T1 gd, ch2: seg)
                - Dict with image keys and optional 'seg': 3D/unified format
                - Dict with 'latent': latent space format
            device: Target device.

        Returns:
            Dictionary with image dict and labels.
        """
        if _is_latent_batch(batch):
            return _prepare_latent_batch(batch, device, is_conditional=True)

        batch = _to_device(batch, device)

        # Handle dict batch format (from unified or 3D loaders)
        if isinstance(batch, dict):
            images_dict: dict[str, torch.Tensor] = {}
            for i, key in enumerate(self.image_keys):
                img = batch.get(key)
                if img is None:
                    raise ValueError(f"Dict batch missing '{key}' key")
                if img.dim() not in (4, 5):
                    raise ValueError(
                        f"ConditionalDualMode: expected 4D or 5D tensor for '{key}', "
                        f"got {img.dim()}D with shape {list(img.shape)}"
                    )
                images_dict[key] = img

            seg_mask = batch.get('seg')
            if seg_mask is not None and seg_mask.dim() not in (4, 5):
                raise ValueError(
                    f"ConditionalDualMode: expected 4D or 5D seg tensor, "
                    f"got {seg_mask.dim()}D with shape {list(seg_mask.shape)}"
                )

            return {'images': images_dict, 'labels': seg_mask}

        # Handle 2D tensor format [B, 3, H, W] (dimension-agnostic slicing)
        images: dict[str, torch.Tensor] = {
            self.image_keys[0]: batch[:, 0:1],
            self.image_keys[1]: batch[:, 1:2],
        }
        seg_mask = batch[:, 2:3]

        return {
            'images': images,
            'labels': seg_mask
        }

    def get_model_config(self) -> dict[str, int]:
        """Get model channel configuration.

        Model takes: [noisy_t1_pre, noisy_t1_gd, seg_mask] = 3 input channels
        Model outputs: [noise/velocity_pred_pre, noise/velocity_pred_gd] = 2 channels

        Returns:
            Channel configuration dictionary.
        """
        return {
            'in_channels': 3,
            'out_channels': 2
        }

    def format_model_input(
        self,
        noisy_images: dict[str, torch.Tensor],
        labels_dict: dict[str, torch.Tensor | None]
    ) -> torch.Tensor:
        """Concatenate both noisy images with segmentation mask.

        Args:
            noisy_images: Dictionary with noisy images for each key.
            labels_dict: Dictionary with 'labels' key containing seg mask.

        Returns:
            [B, 3, H, W] - [noisy_t1_pre, noisy_t1_gd, seg_mask] concatenated.

        Raises:
            TypeError: If noisy_images is not a dict.
            KeyError: If noisy_images is missing required keys.
            ValueError: If labels are None.
        """
        # Type guard - this method requires dict input
        if not isinstance(noisy_images, dict):
            raise TypeError(
                f"ConditionalDualMode.format_model_input requires Dict, "
                f"got {type(noisy_images).__name__}. "
                f"Ensure prepare_batch returned dict format."
            )

        # Validate required keys exist
        for key in self.image_keys:
            if key not in noisy_images:
                raise KeyError(f"noisy_images missing required key: '{key}'")
            _validate_tensor(noisy_images[key], f"noisy_images['{key}']")

        labels = labels_dict.get('labels')
        if labels is None:
            raise ValueError("labels_dict['labels'] is None, cannot concatenate")

        channels = [noisy_images[key] for key in self.image_keys]
        channels.append(labels)
        return torch.cat(channels, dim=1)


class MultiModalityMode(TrainingMode):
    """Multi-modality diffusion training mode with mode embedding.

    Trains on all modalities (bravo, flair, t1_pre, t1_gd) conditioned on
    segmentation mask, with mode embedding to identify the modality.

    Input: tuple (image [B,1,H,W], seg [B,1,H,W], mode_id [B])
    Model: [B, 2, H, W] -> [B, 1, H, W] (same as bravo mode)
    Extra: mode_id for embedding (0=bravo, 1=flair, 2=t1_pre, 3=t1_gd)

    This is a conditional model like bravo, but trained on pooled modalities
    with mode embedding to identify which modality is being generated.
    """

    def __init__(self, image_keys: list[str] = None) -> None:
        """Initialize multi-modality mode.

        Args:
            image_keys: Names of modalities to train on.
                Default: ['bravo', 'flair', 't1_pre', 't1_gd']
        """
        if image_keys is None:
            image_keys = ['bravo', 'flair', 't1_pre', 't1_gd']
        self.image_keys: list[str] = image_keys

    @property
    def is_conditional(self) -> bool:
        """Multi-modality mode uses conditioning (seg + mode_id)."""
        return True

    def prepare_batch(
        self, batch: torch.Tensor | dict[str, Any], device: torch.device
    ) -> dict[str, Any]:
        """Prepare multi-modality batch with mode_id.

        Args:
            batch: Dict with 'image', 'seg', 'mode_id' or latent dict where:
                - image: [B, 1, H, W] single modality image
                - seg: [B, 1, H, W] segmentation mask
                - mode_id: [B] integer tensor (0=bravo, 1=flair, 2=t1_pre, 3=t1_gd)
            device: Target device.

        Returns:
            Dictionary with images, labels (seg), and mode_id.

        Raises:
            TypeError: If batch is not a dict.
            KeyError: If batch is missing required keys.
        """
        if _is_latent_batch(batch):
            result = _prepare_latent_batch(batch, device, is_conditional=True)
            # For multi-modality, we might have mode_id in the batch
            result['mode_id'] = batch.get('mode_id')
            if result['mode_id'] is not None:
                result['mode_id'] = result['mode_id'].to(device)
            return result

        # Validate batch is dict format
        if not isinstance(batch, dict):
            raise TypeError(
                f"MultiModalityMode.prepare_batch requires dict batch, "
                f"got {type(batch).__name__}. Update dataloader to return dict format."
            )

        # Validate required keys
        _validate_dict_keys(batch, ['image', 'seg', 'mode_id'], 'MultiModalityMode.prepare_batch')

        batch = _to_device(batch, device)
        return {
            'images': batch['image'],
            'labels': batch['seg'],
            'mode_id': batch['mode_id'],
        }

    def get_model_config(self) -> dict[str, int]:
        """Get model channel configuration.

        Model takes: [noisy_image, seg_mask] = 2 input channels (same as bravo)
        Model outputs: [noise_pred] or [velocity_pred] = 1 output channel

        Returns:
            Channel configuration dictionary.
        """
        return {
            'in_channels': 2,
            'out_channels': 1
        }

    def format_model_input(
        self,
        noisy_images: torch.Tensor,
        labels_dict: dict[str, torch.Tensor | None]
    ) -> torch.Tensor:
        """Concatenate noisy image with segmentation mask.

        Args:
            noisy_images: [B, 1, H, W] noisy modality image.
            labels_dict: Dictionary with 'labels' key containing seg masks.

        Returns:
            [B, 2, H, W] - [noisy_image, seg_mask] concatenated.
        """
        return torch.cat([noisy_images, labels_dict['labels']], dim=1)


class SegmentationConditionedMode(TrainingMode):
    """Segmentation mask generation conditioned on tumor size distribution.

    Mode for generating segmentation masks with size bin conditioning.
    Uses a 6-dimensional vector where each element is the count of tumors
    in that size bin (Feret diameter, RANO-BM aligned).

    Size bins (mm): [0-3, 3-6, 6-10, 10-15, 15-20, 20-30, 30+]

    Input: tuple (seg [B,1,H,W], size_bins [B,6])
    Model: [B, 1, H, W] -> [B, 1, H, W]
    Extra: size_bins embedding for conditioning

    Conditioning is done via embedding (added to timestep), NOT concatenation.
    """

    def __init__(self, size_bin_config: dict[str, Any] | None = None) -> None:
        """Initialize segmentation conditioned mode.

        Args:
            size_bin_config: Configuration for size bins. Defaults:
                - edges: [0, 3, 6, 10, 15, 20, 30]
                - num_bins: 7
                - max_count: 10
                - embedding_dim: 32
        """
        self.size_bin_config = size_bin_config or {
            'edges': [0, 3, 6, 10, 15, 20, 30],
            'num_bins': 7,
            'max_count': 10,
            'embedding_dim': 32,
        }

    @property
    def is_conditional(self) -> bool:
        """Segmentation conditioned mode uses size bin conditioning."""
        return True

    def prepare_batch(
        self, batch: torch.Tensor | dict[str, Any], device: torch.device
    ) -> dict[str, Any]:
        """Prepare segmentation conditioned batch with size_bins.

        Args:
            batch: Dict with 'image' and 'size_bins' where:
                - image: [B, 1, H, W] segmentation mask
                - size_bins: [B, num_bins] tumor count per size bin
            device: Target device.

        Returns:
            Dictionary with images (seg), labels (None), and size_bins.
        """
        if _is_latent_batch(batch):
            result = _prepare_latent_batch(batch, device, is_conditional=False)
            result['size_bins'] = batch.get('size_bins')
            if result['size_bins'] is not None:
                result['size_bins'] = result['size_bins'].to(device)
            return result

        # All batches are now dict format
        batch = _to_device(batch, device)
        _validate_dict_keys(batch, ['image', 'size_bins'], 'SegmentationConditionedMode')
        return {
            'images': batch['image'],
            'labels': None,  # No label concatenation, conditioning via embedding
            'size_bins': batch['size_bins'],
        }

    def get_model_config(self) -> dict[str, int]:
        """Get model channel configuration.

        Model takes: [noisy_seg] = 1 input channel (conditioning via embedding)
        Model outputs: [noise_pred] or [velocity_pred] = 1 output channel

        Returns:
            Channel configuration dictionary.
        """
        return {
            'in_channels': 1,
            'out_channels': 1
        }

    def format_model_input(
        self,
        noisy_images: torch.Tensor,
        labels_dict: dict[str, torch.Tensor | None]
    ) -> torch.Tensor:
        """Format model input (no concatenation, conditioning via embedding).

        Args:
            noisy_images: [B, 1, H, W] noisy segmentation masks.
            labels_dict: Not used for concatenation (size_bins used via embedding).

        Returns:
            [B, 1, H, W] - same as input.
        """
        return noisy_images


class SegmentationConditionedInputMode(TrainingMode):
    """Segmentation mask generation with input channel conditioning.

    Instead of FiLM conditioning on time embedding, size bins are converted
    to spatial maps and concatenated with the noise input. This provides
    stronger conditioning as the information is present at every layer.

    Input: tuple (seg [B,1,H,W], size_bins [B,7], bin_maps [B,7,H,W])
    Model: [B, 1+7, H, W] -> [B, 1, H, W]

    The bin_maps are normalized to [0, 1] (count / max_count).
    """

    def __init__(self, size_bin_config: dict[str, Any] | None = None) -> None:
        """Initialize segmentation conditioned input mode.

        Args:
            size_bin_config: Configuration for size bins. Keys:
                - num_bins: Number of size bins (default: 7)
                - max_count: Max count per bin for normalization (default: 10)
        """
        self.size_bin_config = size_bin_config or {
            'num_bins': 7,
            'max_count': 10,
        }
        self.num_bins = self.size_bin_config.get('num_bins', 7)

    @property
    def is_conditional(self) -> bool:
        """Input conditioning mode is conditional."""
        return True

    def prepare_batch(
        self, batch: torch.Tensor | dict[str, Any], device: torch.device
    ) -> dict[str, Any]:
        """Prepare batch with bin_maps for input conditioning.

        Args:
            batch: Dict with 'image', 'size_bins', and optionally 'bin_maps' where:
                - image: [B, 1, H, W] segmentation mask
                - size_bins: [B, num_bins] tumor count per size bin
                - bin_maps: [B, num_bins, H, W] spatial conditioning maps (optional)
            device: Target device.

        Returns:
            Dictionary with images (seg), bin_maps, and size_bins.
        """
        # All batches are now dict format
        batch = _to_device(batch, device)
        _validate_dict_keys(batch, ['image'], 'SegmentationConditionedInputMode')
        return {
            'images': batch['image'],
            'labels': None,
            'size_bins': batch.get('size_bins'),
            'bin_maps': batch.get('bin_maps'),
        }

    def get_model_config(self) -> dict[str, int]:
        """Get model channel configuration.

        Model takes: [noisy_seg, bin_maps] = 1 + num_bins input channels
        Model outputs: [noise_pred] or [velocity_pred] = 1 output channel

        Returns:
            Channel configuration dictionary.
        """
        return {
            'in_channels': 1 + self.num_bins,
            'out_channels': 1
        }

    def format_model_input(
        self,
        noisy_images: torch.Tensor,
        labels_dict: dict[str, torch.Tensor | None]
    ) -> torch.Tensor:
        """Format model input by concatenating bin_maps.

        Args:
            noisy_images: [B, 1, H, W] or [B, 1, D, H, W] noisy segmentation.
            labels_dict: Dict containing 'bin_maps' [B, num_bins, H, W] or [B, num_bins, D, H, W].

        Returns:
            [B, 1+num_bins, H, W] or [B, 1+num_bins, D, H, W] concatenated input.
        """
        bin_maps = labels_dict.get('bin_maps')
        if bin_maps is None:
            # No conditioning - use zeros
            if noisy_images.dim() == 4:
                # 2D: [B, 1, H, W]
                B, _, H, W = noisy_images.shape
                bin_maps = torch.zeros(B, self.num_bins, H, W,
                                       device=noisy_images.device, dtype=noisy_images.dtype)
            else:
                # 3D: [B, 1, D, H, W]
                B, _, D, H, W = noisy_images.shape
                bin_maps = torch.zeros(B, self.num_bins, D, H, W,
                                       device=noisy_images.device, dtype=noisy_images.dtype)

        return torch.cat([noisy_images, bin_maps], dim=1)


class LatentSegConditionedMode(TrainingMode):
    """Latent bravo generation conditioned on latent segmentation.

    Mode for generating BRAVO latents conditioned on latent seg masks.
    Both bravo and seg are encoded to latent space using the same VQ-VAE,
    then concatenated as UNet input.

    Input: [noisy_bravo_latent (4ch), seg_latent (4ch)] = 8 channels
    Output: [noise/velocity_pred (4ch)] = 4 channels

    This is for latent diffusion only. Both modalities must be pre-encoded.
    The seg_latent is never decoded - it just provides spatial structure.

    Note: VQ-VAE wasn't trained on seg masks, but the spatial structure
    is still useful for conditioning.
    """

    def __init__(self, latent_channels: int = 4) -> None:
        """Initialize latent seg conditioned mode.

        Args:
            latent_channels: Number of channels per modality in latent space.
                Default 4 for VQ-VAE.
        """
        self.latent_channels = latent_channels

    @property
    def is_conditional(self) -> bool:
        """Latent seg conditioned mode uses conditioning."""
        return True

    def prepare_batch(
        self, batch: torch.Tensor | dict[str, Any], device: torch.device
    ) -> dict[str, Any]:
        """Prepare batch with latent seg conditioning.

        Expects batch from LatentDataset with:
        - 'latent': Pre-encoded bravo latent [B, C_lat, ...]
        - 'latent_seg': Pre-encoded seg latent [B, C_lat, ...]
        - 'seg_mask': Optional pixel-space seg for regional metrics

        Args:
            batch: Dict from LatentDataset.
            device: Target device.

        Returns:
            Dictionary with images (bravo latent), labels (seg latent),
            and optional seg_mask for regional metrics.

        Raises:
            ValueError: If batch is not from latent dataloader or missing latent_seg.
        """
        if not _is_latent_batch(batch):
            raise ValueError(
                "bravo_seg_cond mode requires latent dataloader. "
                "Ensure latent.enabled=true and cache includes latent_seg."
            )

        latent = batch['latent'].to(device, non_blocking=True)
        latent_seg = batch.get('latent_seg')

        if latent_seg is None:
            raise ValueError(
                "bravo_seg_cond mode requires 'latent_seg' in batch. "
                "Rebuild latent cache with seg encoding enabled."
            )

        latent_seg = latent_seg.to(device, non_blocking=True)

        # Keep pixel-space seg for regional metrics (optional)
        seg_mask = batch.get('seg_mask')
        if seg_mask is not None:
            seg_mask = seg_mask.to(device, non_blocking=True)

        return {
            'images': latent,       # Bravo latent [B, C_lat, ...]
            'labels': latent_seg,   # Seg latent [B, C_lat, ...] - already encoded
            'seg_mask': seg_mask,   # Pixel seg for regional metrics
            'is_latent': True,      # Flag: skip encoding in trainer
            'labels_is_latent': True,  # Flag: labels are already in latent space
        }

    def get_model_config(self) -> dict[str, int]:
        """Get model channel configuration.

        Returns pixel-equivalent channels so that space.get_latent_channels()
        produces the correct values:
        - 2 -> 8 (bravo_latent + seg_latent)
        - 1 -> 4 (noise prediction for bravo_latent)

        Returns:
            Channel configuration dictionary.
        """
        return {
            'in_channels': 2,  # Expands to 8 in latent space
            'out_channels': 1  # Expands to 4 in latent space
        }

    def format_model_input(
        self,
        noisy_images: torch.Tensor,
        labels_dict: dict[str, torch.Tensor | None]
    ) -> torch.Tensor:
        """Concatenate noisy bravo latent with seg latent.

        Args:
            noisy_images: [B, C_lat, ...] noisy bravo latent.
            labels_dict: Dictionary with 'labels' key containing seg latent.

        Returns:
            [B, 2*C_lat, ...] - [noisy_bravo_latent, seg_latent] concatenated.
        """
        seg_latent = labels_dict['labels']
        if seg_latent is None:
            raise ValueError(
                "bravo_seg_cond mode requires seg_latent for conditioning. "
                "Ensure latent cache includes 'latent_seg'."
            )
        return torch.cat([noisy_images, seg_latent], dim=1)
