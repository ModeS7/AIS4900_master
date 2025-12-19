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

    def __init__(self) -> None:
        """Initialize segmentation mode."""
        pass

    @property
    def is_conditional(self) -> bool:
        """Segmentation mode is unconditional."""
        return False

    def prepare_batch(
        self, batch: torch.Tensor, device: torch.device
    ) -> Dict[str, Any]:
        """Prepare segmentation batch.

        Args:
            batch: Tensor [B, 1, H, W] - segmentation masks.
            device: Target device.

        Returns:
            Dictionary with images and None labels.
        """
        if hasattr(batch, 'as_tensor'):
            batch = batch.as_tensor().to(device)
        else:
            batch = batch.to(device)

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

    def __init__(self) -> None:
        """Initialize conditional single mode."""
        pass

    @property
    def is_conditional(self) -> bool:
        """Conditional single mode uses conditioning."""
        return True

    def prepare_batch(
        self, batch: torch.Tensor, device: torch.device
    ) -> Dict[str, Any]:
        """Prepare conditional single batch.

        Args:
            batch: Tensor [B, 2, H, W] where:
                - Channel 0: BRAVO image
                - Channel 1: Segmentation mask
            device: Target device.

        Returns:
            Dictionary with separated images and labels.
        """
        if hasattr(batch, 'as_tensor'):
            batch = batch.as_tensor().to(device)
        else:
            batch = batch.to(device)

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
        self, batch: torch.Tensor, device: torch.device
    ) -> Dict[str, Any]:
        """Prepare conditional dual batch.

        Args:
            batch: Tensor [B, 3, H, W] where:
                - Channel 0: T1 pre-contrast
                - Channel 1: T1 gadolinium (post-contrast)
                - Channel 2: Segmentation mask
            device: Target device.

        Returns:
            Dictionary with image dict and labels.
        """
        if hasattr(batch, 'as_tensor'):
            batch = batch.as_tensor().to(device)
        else:
            batch = batch.to(device)

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
