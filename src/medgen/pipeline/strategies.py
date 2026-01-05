"""
Diffusion strategy implementations for image generation.

This module provides abstract and concrete implementations of diffusion
strategies including DDPM and Rectified Flow algorithms.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from monai.networks.schedulers import DDPMScheduler, RFlowScheduler
from torch import nn
from tqdm import tqdm

ImageTensor = torch.Tensor
ImageDict = Dict[str, torch.Tensor]
ImageOrDict = Union[ImageTensor, ImageDict]


class ParsedModelInput:
    """Container for parsed model input components."""

    def __init__(
        self,
        noisy_images: Optional[torch.Tensor],
        noisy_pre: Optional[torch.Tensor],
        noisy_gd: Optional[torch.Tensor],
        conditioning: Optional[torch.Tensor],
        is_dual: bool,
    ):
        self.noisy_images = noisy_images
        self.noisy_pre = noisy_pre
        self.noisy_gd = noisy_gd
        self.conditioning = conditioning
        self.is_dual = is_dual


class DiffusionStrategy(ABC):
    """Abstract base class for diffusion algorithms.

    Defines the interface for diffusion strategies including DDPM and
    Rectified Flow. Subclasses must implement all abstract methods.

    Attributes:
        scheduler: The noise scheduler for the diffusion process.
    """

    scheduler: Any

    def _parse_model_input(self, model_input: torch.Tensor) -> ParsedModelInput:
        """Parse model input into components based on channel count.

        Args:
            model_input: Input tensor with noise and optional conditioning.

        Returns:
            ParsedModelInput with extracted components.

        Raises:
            ValueError: If channel count is unexpected.
        """
        num_channels = model_input.shape[1]

        if num_channels == 1:
            # Unconditional: just noise
            return ParsedModelInput(
                noisy_images=model_input,
                noisy_pre=None,
                noisy_gd=None,
                conditioning=None,
                is_dual=False,
            )
        elif num_channels == 2:
            # Conditional single: [noise, conditioning]
            return ParsedModelInput(
                noisy_images=model_input[:, 0:1, :, :],
                noisy_pre=None,
                noisy_gd=None,
                conditioning=model_input[:, 1:2, :, :],
                is_dual=False,
            )
        elif num_channels == 3:
            # Conditional dual: [noise_pre, noise_gd, conditioning]
            return ParsedModelInput(
                noisy_images=None,
                noisy_pre=model_input[:, 0:1, :, :],
                noisy_gd=model_input[:, 1:2, :, :],
                conditioning=model_input[:, 2:3, :, :],
                is_dual=True,
            )
        else:
            raise ValueError(f"Unexpected number of channels: {num_channels}")

    def _call_model(
        self,
        model: nn.Module,
        model_input: torch.Tensor,
        timesteps: torch.Tensor,
        omega: Optional[torch.Tensor],
        mode_id: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Call model with appropriate arguments based on conditioning.

        Args:
            model: Diffusion model (may be wrapped with conditioning).
            model_input: Formatted input tensor.
            timesteps: Current timesteps.
            omega: Optional ScoreAug omega conditioning.
            mode_id: Optional mode ID for multi-modality.

        Returns:
            Model prediction (noise or velocity).
        """
        with torch.no_grad():
            if omega is not None or mode_id is not None:
                return model(model_input, timesteps=timesteps, omega=omega, mode_id=mode_id)
            else:
                return model(x=model_input, timesteps=timesteps)

    @abstractmethod
    def setup_scheduler(self, num_timesteps: int, image_size: int) -> Any:
        """Setup the noise scheduler.

        Args:
            num_timesteps: Number of diffusion timesteps.
            image_size: Size of input images (assumes square).

        Returns:
            Configured scheduler instance.
        """
        pass

    def add_noise(
        self, clean_images: ImageOrDict, noise: ImageOrDict, timesteps: torch.Tensor
    ) -> ImageOrDict:
        """Add noise to clean images at specified timesteps.

        Uses the scheduler's add_noise method. Handles both single-image
        (tensor) and dual-image (dict) formats.

        Args:
            clean_images: Clean images (tensor or dict for dual-image mode).
            noise: Noise to add (same format as clean_images).
            timesteps: Diffusion timesteps tensor.

        Returns:
            Noisy images in same format as input.
        """
        if isinstance(clean_images, dict):
            return {
                key: self.scheduler.add_noise(clean_images[key], noise[key], timesteps)
                for key in clean_images.keys()
            }
        else:
            return self.scheduler.add_noise(clean_images, noise, timesteps)

    @abstractmethod
    def predict_noise_or_velocity(
        self, model: nn.Module, model_input: torch.Tensor, timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Get model prediction (noise for DDPM, velocity for RFlow).

        Args:
            model: Diffusion model.
            model_input: Formatted model input tensor.
            timesteps: Current timesteps.

        Returns:
            Model prediction tensor.
        """
        pass

    @abstractmethod
    def compute_loss(
        self,
        prediction: torch.Tensor,
        target_images: ImageOrDict,
        noise: ImageOrDict,
        noisy_images: ImageOrDict,
        timesteps: torch.Tensor
    ) -> Tuple[torch.Tensor, ImageOrDict]:
        """Compute loss and predicted clean images.

        Args:
            prediction: Model output (can be multi-channel).
            target_images: Clean target images (tensor or dict for dual-image).
            noise: Noise that was added (same format as target_images).
            noisy_images: Noisy images at timestep t (same format as target_images).
            timesteps: Diffusion timesteps.

        Returns:
            Tuple of (mse_loss, predicted_clean_images) where
            predicted_clean_images matches the format of target_images.
        """
        pass

    @abstractmethod
    def sample_timesteps(self, images: ImageOrDict) -> torch.Tensor:
        """Sample timesteps for training.

        Args:
            images: Image tensor or dict of tensors (for batch size/device).

        Returns:
            Sampled timesteps tensor.
        """
        pass

    @abstractmethod
    def generate(
        self,
        model: nn.Module,
        noise: torch.Tensor,
        num_steps: int,
        device: torch.device,
        use_progress_bars: bool = False,
        omega: Optional[torch.Tensor] = None,
        mode_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate samples using the diffusion process.

        Args:
            model: Trained diffusion model.
            noise: Initial noise tensor.
            num_steps: Number of sampling steps.
            device: Computation device.
            use_progress_bars: Whether to show progress bars.
            omega: Optional ScoreAug omega conditioning tensor.
            mode_id: Optional mode ID tensor for multi-modality conditioning.

        Returns:
            Generated image tensor.
        """
        pass


class DDPMStrategy(DiffusionStrategy):
    """Denoising Diffusion Probabilistic Model strategy.

    Implements the DDPM algorithm for noise prediction-based diffusion.
    """

    def setup_scheduler(
        self, num_timesteps: int = 1000, image_size: int = 128
    ) -> DDPMScheduler:
        """Setup DDPM scheduler with cosine schedule.

        Args:
            num_timesteps: Number of diffusion timesteps.
            image_size: Size of input images (unused but kept for interface).

        Returns:
            Configured DDPMScheduler instance.
        """
        self.scheduler = DDPMScheduler(
            num_train_timesteps=num_timesteps, schedule='cosine'
        )
        return self.scheduler

    def predict_noise_or_velocity(self, model, model_input, timesteps):
        """DDPM predicts noise"""
        return model(x=model_input, timesteps=timesteps)

    def compute_loss(self, prediction, target_images, noise, noisy_images, timesteps):
        """
        Compute DDPM loss

        Args:
            prediction: Model output [B, C, H, W] where C=1 (single) or C=2 (dual)
            target_images: Clean images (tensor or dict)
            noise: Noise added (tensor or dict)
            noisy_images: Noisy images at timestep t (tensor or dict)
            timesteps: Timestep tensor

        Returns:
            (mse_loss, predicted_clean_images)
        """
        device = prediction.device

        # Handle dual-image case
        if isinstance(target_images, dict):
            # prediction has 2 channels: [noise_pred_pre, noise_pred_post]
            keys = list(target_images.keys())
            noise_pred_pre = prediction[:, 0:1, :, :]
            noise_pred_post = prediction[:, 1:2, :, :]

            # Compute loss for each image
            mse_loss_pre = F.mse_loss(noise_pred_pre.float(), noise[keys[0]].float())
            mse_loss_post = F.mse_loss(noise_pred_post.float(), noise[keys[1]].float())
            mse_loss = (mse_loss_pre + mse_loss_post) / 2

            # Reconstruct clean images using passed noisy_images
            alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
            alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

            predicted_clean = {
                keys[0]: torch.clamp(
                    (noisy_images[keys[0]] - sqrt_one_minus_alpha_t * noise_pred_pre) / sqrt_alpha_t,
                    0, 1
                ),
                keys[1]: torch.clamp(
                    (noisy_images[keys[1]] - sqrt_one_minus_alpha_t * noise_pred_post) / sqrt_alpha_t,
                    0, 1
                )
            }

        else:
            # Single-image case
            mse_loss = F.mse_loss(prediction.float(), noise.float())

            alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
            alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)

            predicted_clean = torch.clamp(
                (noisy_images - sqrt_one_minus_alpha_t * prediction) / sqrt_alpha_t,
                0, 1
            )

        return mse_loss, predicted_clean

    def sample_timesteps(self, images):
        # Extract batch size from images
        if isinstance(images, dict):
            batch_size = list(images.values())[0].shape[0]
            device = list(images.values())[0].device
        else:
            batch_size = images.shape[0]
            device = images.device

        return torch.randint(
            0, self.scheduler.num_train_timesteps,
            (batch_size,), device=device
        ).long()

    def generate(
        self,
        model,
        model_input,
        num_steps,
        device,
        use_progress_bars=False,
        omega: Optional[torch.Tensor] = None,
        mode_id: Optional[torch.Tensor] = None,
    ):
        """
        Generate using DDPM sampling

        Handles both unconditional and conditional generation:
        - Unconditional: model_input is [B, 1, H, W] (just noise)
        - Conditional single: model_input is [B, 2, H, W] (noise + conditioning)
        - Conditional dual: model_input is [B, 3, H, W] (noise_pre + noise_gd + conditioning)

        Args:
            model: Trained diffusion model (may be wrapped with ScoreAug/ModeEmbed).
            model_input: Input tensor with noise and optional conditioning.
            num_steps: Number of sampling steps.
            device: Computation device.
            use_progress_bars: Whether to show progress bars.
            omega: Optional ScoreAug omega conditioning tensor [B, 5].
            mode_id: Optional mode ID tensor [B] for multi-modality conditioning.
        """
        batch_size = model_input.shape[0]

        # Set timesteps for inference
        self.scheduler.set_timesteps(num_inference_steps=num_steps)

        # Parse model input into components
        parsed = self._parse_model_input(model_input)
        noisy_images = parsed.noisy_images
        noisy_pre = parsed.noisy_pre
        noisy_gd = parsed.noisy_gd
        conditioning = parsed.conditioning
        is_dual = parsed.is_dual

        # Sampling loop
        timesteps = self.scheduler.timesteps
        if use_progress_bars:
            timesteps_iter = tqdm(timesteps, desc="DDPM sampling")
        else:
            timesteps_iter = timesteps

        for t in timesteps_iter:
            timesteps_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)

            if is_dual:
                # Dual-image: process each channel through model together
                current_model_input = torch.cat([noisy_pre, noisy_gd, conditioning], dim=1)
                noise_pred = self._call_model(model, current_model_input, timesteps_batch, omega, mode_id)

                # Split predictions for each channel
                noise_pred_pre = noise_pred[:, 0:1, :, :]
                noise_pred_gd = noise_pred[:, 1:2, :, :]

                # Denoise each channel SEPARATELY using scheduler
                noisy_pre, _ = self.scheduler.step(noise_pred_pre, t, noisy_pre)
                noisy_gd, _ = self.scheduler.step(noise_pred_gd, t, noisy_gd)

            else:
                # Single image or unconditional
                if conditioning is not None:
                    current_model_input = torch.cat([noisy_images, conditioning], dim=1)
                else:
                    current_model_input = noisy_images

                noise_pred = self._call_model(model, current_model_input, timesteps_batch, omega, mode_id)

                noisy_images, _ = self.scheduler.step(noise_pred, t, noisy_images)

        # Return final denoised images
        if is_dual:
            return torch.cat([noisy_pre, noisy_gd], dim=1)  # [B, 2, H, W]
        else:
            return noisy_images


class RFlowStrategy(DiffusionStrategy):
    """Rectified Flow algorithm"""

    def setup_scheduler(self, num_timesteps=1000, image_size=128):
        self.scheduler = RFlowScheduler(
            num_train_timesteps=num_timesteps,
            use_discrete_timesteps=True,
            sample_method='logit-normal',
            use_timestep_transform=True,
            base_img_size_numel=image_size * image_size,
            spatial_dim=2
        )
        return self.scheduler

    def predict_noise_or_velocity(self, model, model_input, timesteps):
        """RFlow predicts velocity"""
        return model(x=model_input, timesteps=timesteps)

    def compute_loss(self, prediction, target_images, noise, noisy_images, timesteps):
        """
        Compute RFlow loss (velocity prediction)

        Args:
            prediction: Model velocity prediction
            target_images: Clean images (x_0)
            noise: Pure Gaussian noise (x_1)
            noisy_images: Interpolated images at timestep t (x_t)
            timesteps: Timestep tensor

        Returns:
            (mse_loss, predicted_clean_images)
        """
        # Get normalized timestep t in [0, 1]
        t = timesteps.float() / self.scheduler.num_train_timesteps
        t = t.view(-1, 1, 1, 1)

        # Handle dual-image case
        if isinstance(target_images, dict):
            keys = list(target_images.keys())
            velocity_pred_pre = prediction[:, 0:1, :, :]
            velocity_pred_post = prediction[:, 1:2, :, :]

            # Target velocity is (clean - noise) = (x_0 - x_1)
            # This matches MONAI RFlowScheduler convention
            velocity_target_pre = target_images[keys[0]] - noise[keys[0]]
            velocity_target_post = target_images[keys[1]] - noise[keys[1]]

            mse_loss_pre = F.mse_loss(velocity_pred_pre.float(), velocity_target_pre.float())
            mse_loss_post = F.mse_loss(velocity_pred_post.float(), velocity_target_post.float())
            mse_loss = (mse_loss_pre + mse_loss_post) / 2

            # Reconstruct clean from velocity: x_0 = x_t + t * v (since v = x_0 - x_1)
            predicted_clean = {
                keys[0]: torch.clamp(noisy_images[keys[0]] + t * velocity_pred_pre, 0, 1),
                keys[1]: torch.clamp(noisy_images[keys[1]] + t * velocity_pred_post, 0, 1)
            }

        else:
            # Single-image case
            # Target velocity is (clean - noise) = (x_0 - x_1)
            velocity_target = target_images - noise
            mse_loss = F.mse_loss(prediction.float(), velocity_target.float())

            # Reconstruct clean from velocity: x_0 = x_t + t * v
            predicted_clean = torch.clamp(noisy_images + t * prediction, 0, 1)

        return mse_loss, predicted_clean

    def sample_timesteps(self, images):
        # RFlow needs actual tensor for logit-normal sampling
        if isinstance(images, dict):
            # Use first image in dict
            sample_tensor = list(images.values())[0]
        else:
            sample_tensor = images
        return self.scheduler.sample_timesteps(sample_tensor)

    def generate(
        self,
        model,
        model_input,
        num_steps,
        device,
        use_progress_bars=False,
        omega: Optional[torch.Tensor] = None,
        mode_id: Optional[torch.Tensor] = None,
    ):
        """
        Generate using RFlow sampling

        Handles both unconditional and conditional generation:
        - Unconditional: model_input is [B, 1, H, W]
        - Conditional single: model_input is [B, 2, H, W]
        - Conditional dual: model_input is [B, 3, H, W]

        Args:
            model: Trained diffusion model (may be wrapped with ScoreAug/ModeEmbed).
            model_input: Input tensor with noise and optional conditioning.
            num_steps: Number of sampling steps.
            device: Computation device.
            use_progress_bars: Whether to show progress bars.
            omega: Optional ScoreAug omega conditioning tensor [B, 5].
            mode_id: Optional mode ID tensor [B] for multi-modality conditioning.
        """
        batch_size = model_input.shape[0]
        image_size = model_input.shape[-1]

        # Parse model input into components
        parsed = self._parse_model_input(model_input)
        noisy_images = parsed.noisy_images
        noisy_pre = parsed.noisy_pre
        noisy_gd = parsed.noisy_gd
        conditioning = parsed.conditioning
        is_dual = parsed.is_dual

        # Setup scheduler
        input_img_size_numel = image_size * image_size
        self.scheduler.set_timesteps(
            num_inference_steps=num_steps,
            device=device,
            input_img_size_numel=input_img_size_numel
        )

        all_next_timesteps = torch.cat((
            self.scheduler.timesteps[1:],
            torch.tensor([0], dtype=self.scheduler.timesteps.dtype, device=device)
        ))

        # Sampling loop
        timestep_pairs = zip(self.scheduler.timesteps, all_next_timesteps)
        if use_progress_bars:
            timestep_pairs = tqdm(list(timestep_pairs), desc="RFlow sampling")

        for t, next_t in timestep_pairs:
            timesteps_batch = t.unsqueeze(0).repeat(batch_size).to(device)
            next_timestep = next_t.to(device) if isinstance(next_t, torch.Tensor) else torch.tensor(next_t,
                                                                                                    device=device)

            if is_dual:
                # Dual-image: process each channel through model together
                current_model_input = torch.cat([noisy_pre, noisy_gd, conditioning], dim=1)
                velocity_pred = self._call_model(model, current_model_input, timesteps_batch, omega, mode_id)

                # Split predictions for each channel
                velocity_pred_pre = velocity_pred[:, 0:1, :, :]
                velocity_pred_gd = velocity_pred[:, 1:2, :, :]

                # Update each channel SEPARATELY using scheduler
                noisy_pre, _ = self.scheduler.step(velocity_pred_pre, t, noisy_pre, next_timestep)
                noisy_gd, _ = self.scheduler.step(velocity_pred_gd, t, noisy_gd, next_timestep)

            else:
                # Single image or unconditional
                if conditioning is not None:
                    current_model_input = torch.cat([noisy_images, conditioning], dim=1)
                else:
                    current_model_input = noisy_images

                velocity_pred = self._call_model(model, current_model_input, timesteps_batch, omega, mode_id)
                noisy_images, _ = self.scheduler.step(velocity_pred, t, noisy_images, next_timestep)

        # Return final denoised images
        if is_dual:
            return torch.cat([noisy_pre, noisy_gd], dim=1)  # [B, 2, H, W]
        else:
            return noisy_images
