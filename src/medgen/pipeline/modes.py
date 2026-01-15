"""
Training mode implementations for diffusion models.

This module defines different training modes for the diffusion model,
including unconditional segmentation generation and conditional image
generation modes.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import torch

from medgen.core.constants import DEFAULT_DUAL_IMAGE_KEYS


def _to_device(batch: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Convert batch tensor to device, handling MONAI MetaTensors.

    MetaTensors from MONAI need to be converted to regular tensors
    before being moved to device to avoid metadata issues.

    Args:
        batch: Input tensor (may be MetaTensor).
        device: Target device.

    Returns:
        Tensor on target device.
    """
    if hasattr(batch, 'as_tensor'):
        return batch.as_tensor().to(device)
    return batch.to(device)


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
    batch: Dict[str, Any],
    device: torch.device,
    is_conditional: bool = True
) -> Dict[str, Any]:
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
    """
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
    ) -> Dict[str, Any]:
        """Prepare batch data for training.

        Args:
            batch: Raw batch tensor from dataloader.
            device: Target device for tensors.

        Returns:
            Dictionary with 'images' and optionally 'labels' keys.
        """
        pass

    @abstractmethod
    def get_model_config(self) -> Dict[str, int]:
        """Return model configuration.

        Returns:
            Dictionary with 'in_channels' and 'out_channels' keys.
        """
        pass

    @abstractmethod
    def format_model_input(
        self,
        noisy_images: Union[torch.Tensor, Dict[str, torch.Tensor]],
        labels_dict: Dict[str, Optional[torch.Tensor]]
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
        self, batch: Union[torch.Tensor, Dict[str, Any]], device: torch.device
    ) -> Dict[str, Any]:
        """Prepare segmentation batch.

        Args:
            batch: Tensor [B, 1, H, W] or latent dict with 'latent' key.
            device: Target device.

        Returns:
            Dictionary with images and None labels.
        """
        if _is_latent_batch(batch):
            return _prepare_latent_batch(batch, device, is_conditional=False)

        batch = _to_device(batch, device)
        return {
            'images': batch,
            'labels': None
        }

    def get_model_config(self) -> Dict[str, int]:
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
        labels_dict: Dict[str, Optional[torch.Tensor]]
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
        self, batch: Union[torch.Tensor, Dict[str, Any]], device: torch.device
    ) -> Dict[str, Any]:
        """Prepare conditional single batch.

        Args:
            batch: Tensor [B, 2, H, W] or latent dict where:
                - Channel 0: BRAVO image
                - Channel 1: Segmentation mask
            device: Target device.

        Returns:
            Dictionary with separated images and labels.
        """
        if _is_latent_batch(batch):
            return _prepare_latent_batch(batch, device, is_conditional=True)

        batch = _to_device(batch, device)
        bravo_images = batch[:, 0:1, :, :]
        seg_masks = batch[:, 1:2, :, :]

        return {
            'images': bravo_images,
            'labels': seg_masks
        }

    def get_model_config(self) -> Dict[str, int]:
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
        labels_dict: Dict[str, Optional[torch.Tensor]]
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

    def __init__(self, image_keys: List[str] = None) -> None:
        """Initialize conditional dual mode.

        Args:
            image_keys: Names of the two image types to train.
                Default: DEFAULT_DUAL_IMAGE_KEYS ('t1_pre', 't1_gd')
        """
        if image_keys is None:
            image_keys = DEFAULT_DUAL_IMAGE_KEYS.copy()
        if len(image_keys) != 2:
            raise ValueError(f"ConditionalDualMode requires exactly 2 image types, got {len(image_keys)}: {image_keys}")
        self.image_keys: List[str] = image_keys

    @property
    def is_conditional(self) -> bool:
        """Conditional dual mode uses conditioning."""
        return True

    def prepare_batch(
        self, batch: Union[torch.Tensor, Dict[str, Any]], device: torch.device
    ) -> Dict[str, Any]:
        """Prepare conditional dual batch.

        Args:
            batch: Tensor [B, 3, H, W] or latent dict where:
                - Channel 0: T1 pre-contrast
                - Channel 1: T1 gadolinium (post-contrast)
                - Channel 2: Segmentation mask
            device: Target device.

        Returns:
            Dictionary with image dict and labels.
        """
        if _is_latent_batch(batch):
            return _prepare_latent_batch(batch, device, is_conditional=True)

        batch = _to_device(batch, device)
        images: Dict[str, torch.Tensor] = {
            self.image_keys[0]: batch[:, 0:1, :, :],
            self.image_keys[1]: batch[:, 1:2, :, :],
        }
        seg_mask = batch[:, 2:3, :, :]

        return {
            'images': images,
            'labels': seg_mask
        }

    def get_model_config(self) -> Dict[str, int]:
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
        noisy_images: Dict[str, torch.Tensor],
        labels_dict: Dict[str, Optional[torch.Tensor]]
    ) -> torch.Tensor:
        """Concatenate both noisy images with segmentation mask.

        Args:
            noisy_images: Dictionary with noisy images for each key.
            labels_dict: Dictionary with 'labels' key containing seg mask.

        Returns:
            [B, 3, H, W] - [noisy_t1_pre, noisy_t1_gd, seg_mask] concatenated.
        """
        channels = [noisy_images[key] for key in self.image_keys]
        channels.append(labels_dict['labels'])
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

    def __init__(self, image_keys: List[str] = None) -> None:
        """Initialize multi-modality mode.

        Args:
            image_keys: Names of modalities to train on.
                Default: ['bravo', 'flair', 't1_pre', 't1_gd']
        """
        if image_keys is None:
            image_keys = ['bravo', 'flair', 't1_pre', 't1_gd']
        self.image_keys: List[str] = image_keys

    @property
    def is_conditional(self) -> bool:
        """Multi-modality mode uses conditioning (seg + mode_id)."""
        return True

    def prepare_batch(
        self, batch: Union[torch.Tensor, tuple, Dict[str, Any]], device: torch.device
    ) -> Dict[str, Any]:
        """Prepare multi-modality batch with mode_id.

        Args:
            batch: Tuple (image, seg, mode_id) or latent dict where:
                - image: [B, 1, H, W] single modality image
                - seg: [B, 1, H, W] segmentation mask
                - mode_id: [B] integer tensor (0=bravo, 1=flair, 2=t1_pre, 3=t1_gd)
            device: Target device.

        Returns:
            Dictionary with images, labels (seg), and mode_id.
        """
        if _is_latent_batch(batch):
            result = _prepare_latent_batch(batch, device, is_conditional=True)
            # For multi-modality, we might have mode_id in the batch
            result['mode_id'] = batch.get('mode_id')
            if result['mode_id'] is not None:
                result['mode_id'] = result['mode_id'].to(device)
            return result

        if isinstance(batch, (tuple, list)):
            image, seg, mode_id = batch
            image = _to_device(image, device)
            seg = _to_device(seg, device)
            mode_id = mode_id.to(device)
        else:
            # Fallback for tensor format (shouldn't happen with multi dataloader)
            batch = _to_device(batch, device)
            image = batch[:, 0:1, :, :]
            seg = batch[:, 1:2, :, :]
            mode_id = None

        return {
            'images': image,
            'labels': seg,
            'mode_id': mode_id,
        }

    def get_model_config(self) -> Dict[str, int]:
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
        labels_dict: Dict[str, Optional[torch.Tensor]]
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

    def __init__(self, size_bin_config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize segmentation conditioned mode.

        Args:
            size_bin_config: Configuration for size bins. Defaults:
                - edges: [0, 2, 4, 6, 9, 13, 18, 25, 35, 50]
                - num_bins: 9
                - max_count_per_bin: 10
                - embedding_dim: 32
        """
        self.size_bin_config = size_bin_config or {
            'edges': [0, 3, 6, 10, 15, 20, 30],
            'num_bins': 6,
            'max_count_per_bin': 10,
            'embedding_dim': 32,
        }

    @property
    def is_conditional(self) -> bool:
        """Segmentation conditioned mode uses size bin conditioning."""
        return True

    def prepare_batch(
        self, batch: Union[torch.Tensor, tuple, Dict[str, Any]], device: torch.device
    ) -> Dict[str, Any]:
        """Prepare segmentation conditioned batch with size_bins.

        Args:
            batch: Tuple (seg, size_bins) where:
                - seg: [B, 1, H, W] segmentation mask
                - size_bins: [B, 9] tumor count per size bin
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

        if isinstance(batch, (tuple, list)):
            seg, size_bins = batch
            seg = _to_device(seg, device)
            size_bins = size_bins.to(device)
        else:
            # Fallback for tensor format
            batch = _to_device(batch, device)
            seg = batch
            size_bins = None

        return {
            'images': seg,
            'labels': None,  # No label concatenation, conditioning via embedding
            'size_bins': size_bins,
        }

    def get_model_config(self) -> Dict[str, int]:
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
        labels_dict: Dict[str, Optional[torch.Tensor]]
    ) -> torch.Tensor:
        """Format model input (no concatenation, conditioning via embedding).

        Args:
            noisy_images: [B, 1, H, W] noisy segmentation masks.
            labels_dict: Not used for concatenation (size_bins used via embedding).

        Returns:
            [B, 1, H, W] - same as input.
        """
        return noisy_images
