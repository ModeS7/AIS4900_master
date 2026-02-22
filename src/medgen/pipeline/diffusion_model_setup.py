"""Helper functions for DiffusionTrainer model setup.

These functions are extracted from DiffusionTrainer methods to reduce file size.
Each function takes a `trainer` (DiffusionTrainer instance) as its first argument
and accesses/sets trainer attributes directly.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from .trainer import DiffusionTrainer

logger = logging.getLogger(__name__)


def create_fallback_save_dir(trainer: DiffusionTrainer) -> str:
    """Create fallback save directory for diffusion trainer."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = trainer._training_config.name
    strategy_name = trainer.cfg.strategy.name
    mode_name = trainer.cfg.mode.name
    image_size = getattr(trainer.cfg.model, 'image_size', 256)
    run_name = f"{exp_name}{strategy_name}_{image_size}_{timestamp}"
    return os.path.join(trainer._paths_config.model_dir, 'diffusion_2d', mode_name, run_name)


def setup_model(trainer: DiffusionTrainer, train_dataset: Dataset) -> None:
    """Initialize model, optimizer, and loss functions.

    Args:
        trainer: DiffusionTrainer instance.
        train_dataset: Training dataset for model config extraction.
    """
    from monai.networks.nets import DiffusionModelUNet
    from torch.optim import AdamW

    from medgen.core import (
        ModeType,
        create_plateau_scheduler,
        create_warmup_constant_scheduler,
        create_warmup_cosine_scheduler,
        wrap_model_for_training,
    )
    from medgen.evaluation import ValidationVisualizer
    from medgen.losses import PerceptualLoss
    from medgen.models import (
        ControlNetConditionedUNet,
        create_controlnet_for_unet,
        create_diffusion_model,
        freeze_unet_for_controlnet,
        get_model_type,
        is_transformer_model,
    )

    model_cfg = trainer.mode.get_model_config()

    # Adjust channels for latent space
    in_channels = trainer.space.get_latent_channels(model_cfg['in_channels'])
    out_channels = trainer.space.get_latent_channels(model_cfg['out_channels'])

    # For ControlNet Stage 1 or Stage 2: conditioning goes through ControlNet, not concatenation
    # UNet in_channels = out_channels (no +1 for conditioning)
    if trainer.use_controlnet or trainer.controlnet_stage1:
        in_channels = out_channels
        if trainer.is_main_process:
            stage = "Stage 1 (prep)" if trainer.controlnet_stage1 else "Stage 2"
            logger.info(f"ControlNet {stage}: UNet in_channels={in_channels} (no conditioning concatenation)")

    if trainer.is_main_process and trainer.space.scale_factor > 1:
        space_name = type(trainer.space).__name__
        logger.info(f"{space_name}: {model_cfg['in_channels']} -> {in_channels} channels, "
                   f"scale factor {trainer.space.scale_factor}x")

    # Get model type and check if transformer-based
    trainer.model_type = get_model_type(trainer.cfg)
    trainer.is_transformer = is_transformer_model(trainer.cfg)

    # Create raw model via factory
    if trainer.is_transformer:
        raw_model = create_diffusion_model(trainer.cfg, trainer.device, in_channels, out_channels)

        if trainer.use_omega_conditioning or trainer.use_mode_embedding:
            raise ValueError(
                "Omega/mode conditioning wrappers are not yet supported for transformer models. "
                "Either use model=default (UNet) or disable omega_conditioning/mode_embedding."
            )
    else:
        from .base_config import ModelConfig
        mc = ModelConfig.from_hydra(trainer.cfg)
        channels = tuple(mc.channels)
        attention_levels = tuple(mc.attention_levels)
        num_res_blocks = mc.num_res_blocks
        num_head_channels = mc.num_head_channels

        raw_model = DiffusionModelUNet(
            spatial_dims=mc.spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            attention_levels=attention_levels,
            num_res_blocks=num_res_blocks,
            num_head_channels=num_head_channels,
            norm_num_groups=getattr(trainer.cfg.model, 'norm_num_groups', 32),
        ).to(trainer.device)

    # Enable gradient checkpointing (required to fit large models in GPU memory)
    # Must be applied BEFORE torch.compile or DDP wrapping
    if trainer.use_gradient_checkpointing:
        if trainer.is_transformer:
            if hasattr(raw_model, 'enable_gradient_checkpointing'):
                raw_model.enable_gradient_checkpointing()
                if trainer.is_main_process:
                    logger.info("Transformer gradient checkpointing enabled")
        else:
            from .checkpointing import enable_unet_gradient_checkpointing
            enable_unet_gradient_checkpointing(raw_model)

    # Determine if DDPOptimizer should be disabled for large models
    disable_ddp_opt = getattr(trainer.cfg.training, 'disable_ddp_optimizer', False)
    if trainer.mode_name == ModeType.DUAL and trainer.image_size >= 256:
        disable_ddp_opt = True

    use_compile = getattr(trainer.cfg.training, 'use_compile', True)

    # Handle embedding wrappers (UNet only)
    if not trainer.is_transformer:
        channels = tuple(mc.channels)
        time_embed_dim = 4 * channels[0]
    else:
        time_embed_dim = None

    # Handle embedding wrappers: omega, mode, or both
    if not trainer.is_transformer and (trainer.use_omega_conditioning or trainer.use_mode_embedding):
        from medgen.data import create_conditioning_wrapper
        wrapper, wrapper_name = create_conditioning_wrapper(
            model=raw_model,
            use_omega=trainer.use_omega_conditioning,
            use_mode=trainer.use_mode_embedding,
            embed_dim=time_embed_dim,
            mode_strategy=trainer.mode_embedding_strategy,
            mode_dropout_prob=trainer.mode_embedding_dropout,
            late_mode_start_level=trainer.late_mode_start_level,
        )
        wrapper = wrapper.to(trainer.device)

        if trainer.is_main_process:
            logger.info(f"Conditioning: {wrapper_name} wrapper applied (embed_dim={time_embed_dim})")

        if use_compile:
            wrapper.model = torch.compile(wrapper.model, mode="default")
            if trainer.is_main_process:
                logger.info(f"Single-GPU: Compiled inner UNet ({wrapper_name} wrapper uncompiled)")

        trainer.model = wrapper
        trainer.model_raw = wrapper

    else:
        trainer.model, trainer.model_raw = wrap_model_for_training(
            raw_model,
            use_multi_gpu=trainer.use_multi_gpu,
            local_rank=trainer.local_rank if trainer.use_multi_gpu else 0,
            use_compile=use_compile,
            compile_mode="default",
            disable_ddp_optimizer=disable_ddp_opt,
            is_main_process=trainer.is_main_process,
        )

    # Block DDP with embedding wrappers - embeddings won't sync across GPUs
    if trainer.use_multi_gpu and (trainer.use_omega_conditioning or trainer.use_mode_embedding):
        raise ValueError(
            "DDP is not compatible with embedding wrappers (ScoreAug, ModeEmbed). "
            "Embeddings would NOT be synchronized across GPUs, causing silent training corruption. "
            "Either disable DDP (use single GPU) or disable omega_conditioning/mode_embedding."
        )

    # Handle size bin embedding for seg_conditioned mode
    if trainer.use_size_bin_embedding:
        from medgen.data import SizeBinModelWrapper

        if trainer.is_transformer:
            raise ValueError(
                "Size bin embedding is not supported with transformer models. "
                "Use model=default (UNet) for seg_conditioned mode."
            )

        # Get time_embed_dim from model
        if time_embed_dim is None:
            time_embed_dim = 4 * mc.channels[0]

        # Wrap with size bin embedding
        size_bin_wrapper = SizeBinModelWrapper(
            model=trainer.model_raw,
            embed_dim=time_embed_dim,
            num_bins=trainer.size_bin_num_bins,
            max_count=trainer.size_bin_max_count,
            per_bin_embed_dim=trainer.size_bin_embed_dim,
            projection_hidden_dim=trainer.size_bin_projection_hidden_dim,
            projection_num_layers=trainer.size_bin_projection_num_layers,
        ).to(trainer.device)

        if trainer.is_main_process:
            logger.info(
                f"Size bin embedding: wrapper applied (num_bins={trainer.size_bin_num_bins}, "
                f"embed_dim={time_embed_dim})"
            )

        trainer.model = size_bin_wrapper
        trainer.model_raw = size_bin_wrapper

        # Block DDP with size bin embedding
        if trainer.use_multi_gpu:
            raise ValueError(
                "DDP is not compatible with size bin embedding. "
                "Embeddings would NOT be synchronized across GPUs. "
                "Use single GPU for seg_conditioned mode."
            )

        # Setup auxiliary bin prediction head (forces bottleneck to encode bin info)
        if getattr(trainer, 'size_bin_aux_loss_weight', 0) > 0:
            from medgen.models.wrappers.size_bin_embed import BinPredictionHead

            bottleneck_channels = mc.channels[-1]
            trainer._bin_prediction_head = BinPredictionHead(
                bottleneck_channels=bottleneck_channels,
                num_bins=trainer.size_bin_num_bins,
            ).to(trainer.device)
            # Hook on inner UNet's mid_block (wrapper.model is the raw UNet)
            inner_model = trainer.model_raw.model
            trainer._bin_pred_hook = inner_model.mid_block.register_forward_hook(
                trainer._bin_prediction_head.hook_fn
            )
            if trainer.is_main_process:
                logger.info(
                    f"Auxiliary bin prediction head: bottleneck_channels={bottleneck_channels}, "
                    f"weight={trainer.size_bin_aux_loss_weight}"
                )

    # Setup ControlNet for pixel-resolution conditioning in latent diffusion
    if trainer.use_controlnet:
        if trainer.is_transformer:
            raise ValueError("ControlNet is not supported with transformer models. Use model=default (UNet).")

        # Determine latent channels for ControlNet
        latent_channels = out_channels  # Same as UNet in_channels after ControlNet adjustment

        # Create ControlNet matching UNet architecture
        trainer.controlnet = create_controlnet_for_unet(
            unet=trainer.model_raw,
            cfg=trainer.cfg,
            device=trainer.device,
            spatial_dims=trainer.spatial_dims,
            latent_channels=latent_channels,
        )

        # Enable gradient checkpointing if requested
        controlnet_cfg = trainer._diffusion_config.controlnet
        if controlnet_cfg.gradient_checkpointing:
            if hasattr(trainer.controlnet, 'enable_gradient_checkpointing'):
                trainer.controlnet.enable_gradient_checkpointing()
                if trainer.is_main_process:
                    logger.info("ControlNet gradient checkpointing enabled")

        # Freeze UNet if configured (Stage 2 training)
        if trainer.controlnet_freeze_unet:
            freeze_unet_for_controlnet(trainer.model_raw)

        # Create combined model wrapper
        trainer.model = ControlNetConditionedUNet(
            unet=trainer.model_raw,
            controlnet=trainer.controlnet,
            conditioning_scale=trainer.controlnet_scale,
        )

        if trainer.is_main_process:
            num_params = sum(p.numel() for p in trainer.controlnet.parameters())
            logger.info(f"ControlNet created: {num_params:,} parameters")

    # Setup perceptual loss (skip for seg modes where perceptual_weight=0)
    # This saves ~200MB GPU memory from loading ResNet50
    cache_dir = trainer._paths_config.cache_dir
    if trainer.perceptual_weight > 0:
        trainer.perceptual_loss_fn = PerceptualLoss(
            spatial_dims=2,
            network_type="radimagenet_resnet50",
            cache_dir=cache_dir,
            pretrained=True,
            device=trainer.device,
            use_compile=use_compile,
        )
    else:
        trainer.perceptual_loss_fn = None
        if trainer.is_main_process:
            logger.info("Perceptual loss disabled (perceptual_weight=0), skipping ResNet50 loading")

    # Compile fused forward pass setup
    # Compiled forward doesn't support mode_id parameter, so disable when using
    # mode embedding or omega conditioning (wrappers need extra kwargs)
    compile_fused = getattr(trainer.cfg.training, 'compile_fused_forward', True)
    if trainer.use_multi_gpu or trainer.spatial_dims == 3 or trainer.space.scale_factor > 1 or trainer.use_min_snr or trainer.regional_weight_computer is not None or trainer.score_aug is not None or trainer.sda is not None or trainer.use_mode_embedding or trainer.use_omega_conditioning or trainer.augmented_diffusion_enabled or trainer.use_controlnet or trainer.use_size_bin_embedding or trainer.mode_name not in (ModeType.SEG, ModeType.BRAVO, ModeType.DUAL):
        compile_fused = False

    trainer._setup_compiled_forward(compile_fused)

    # Setup optimizer
    # Determine which parameters to train
    if trainer.use_controlnet:
        if trainer.controlnet_freeze_unet:
            # Stage 2: Only train ControlNet
            train_params = list(trainer.controlnet.parameters())
            if trainer.is_main_process:
                trainable = sum(p.numel() for p in train_params if p.requires_grad)
                logger.info(f"Training only ControlNet ({trainable:,} trainable params, UNet frozen)")
        else:
            # Joint training: Both UNet and ControlNet
            train_params = list(trainer.model_raw.parameters()) + list(trainer.controlnet.parameters())
            if trainer.is_main_process:
                trainable = sum(p.numel() for p in train_params if p.requires_grad)
                logger.info(f"Joint training: UNet + ControlNet ({trainable:,} trainable params)")
    else:
        train_params = list(trainer.model_raw.parameters())
        # Add auxiliary bin prediction head parameters (separate module, not in model_raw)
        bin_pred_head = getattr(trainer, '_bin_prediction_head', None)
        if bin_pred_head is not None:
            train_params += list(bin_pred_head.parameters())

    trainer.optimizer = AdamW(
        train_params,
        lr=trainer.learning_rate,
        weight_decay=trainer.weight_decay,
    )

    if trainer.is_main_process and trainer.weight_decay > 0:
        logger.info(f"Using weight decay: {trainer.weight_decay}")

    # Learning rate scheduler (cosine, constant, or plateau)
    trainer.scheduler_type = trainer._diffusion_config.scheduler_type
    if trainer.scheduler_type == 'constant':
        trainer.lr_scheduler = create_warmup_constant_scheduler(
            trainer.optimizer,
            warmup_epochs=trainer.warmup_epochs,
            total_epochs=trainer.n_epochs,
        )
        if trainer.is_main_process:
            logger.info("Using constant LR scheduler (warmup then constant)")
    elif trainer.scheduler_type == 'plateau':
        plateau_cfg = trainer.cfg.training.get('plateau', {})  # Not yet in typed config
        trainer.lr_scheduler = create_plateau_scheduler(
            trainer.optimizer,
            mode='min',
            factor=plateau_cfg.get('factor', 0.5),
            patience=plateau_cfg.get('patience', 10),
            min_lr=plateau_cfg.get('min_lr', 1e-6),
        )
        if trainer.is_main_process:
            logger.info(
                f"Using ReduceLROnPlateau scheduler "
                f"(factor={plateau_cfg.get('factor', 0.5)}, patience={plateau_cfg.get('patience', 10)})"
            )
    else:
        trainer.lr_scheduler = create_warmup_cosine_scheduler(
            trainer.optimizer,
            warmup_epochs=trainer.warmup_epochs,
            total_epochs=trainer.n_epochs,
            eta_min=trainer.eta_min,
        )

    # Create EMA wrapper if enabled
    trainer._setup_ema()

    # Initialize visualization helper
    trainer.visualizer = ValidationVisualizer(
        cfg=trainer.cfg,
        strategy=trainer.strategy,
        mode=trainer.mode,
        writer=trainer.writer,
        save_dir=trainer.save_dir,
        device=trainer.device,
        is_main_process=trainer.is_main_process,
        space=trainer.space,
        use_controlnet=trainer.use_controlnet,
        controlnet=trainer.controlnet,
    )

    # Initialize generation metrics if enabled
    if trainer._gen_metrics_config is not None and trainer._gen_metrics_config.enabled:
        from medgen.metrics.generation import GenerationMetrics
        trainer._gen_metrics = GenerationMetrics(
            trainer._gen_metrics_config,
            trainer.device,
            trainer.save_dir,
            space=trainer.space,
            mode_name=trainer.mode_name,
        )
        if trainer.is_main_process:
            logger.info("Generation metrics initialized (caching happens at training start)")

    # Save metadata
    if trainer.is_main_process:
        trainer._save_metadata()

    # Setup feature perturbation hooks if enabled
    trainer._setup_feature_perturbation()


def setup_compiled_forward(trainer: DiffusionTrainer, enabled: bool) -> None:
    """Setup compiled forward functions for fused model + loss computation."""
    trainer._use_compiled_forward = enabled

    if not enabled:
        trainer._compiled_forward_single = None
        trainer._compiled_forward_dual = None
        return

    # Capture use_fp32_loss for closure
    use_fp32 = trainer.use_fp32_loss

    # Define and compile forward functions
    # When use_fp32=False, reproduces pre-Jan-7-2026 BF16 behavior
    def _forward_single(
        model: nn.Module,
        perceptual_fn: nn.Module,
        model_input: torch.Tensor,
        timesteps: torch.Tensor,
        images: torch.Tensor,
        noise: torch.Tensor,
        noisy_images: torch.Tensor,
        perceptual_weight: float,
        strategy_name: str,
        num_train_timesteps: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prediction = model(model_input, timesteps)

        if strategy_name == 'rflow':
            t_normalized = timesteps.float() / float(num_train_timesteps)
            t_expanded = t_normalized.view(-1, 1, 1, 1)
            predicted_clean = noisy_images + t_expanded * prediction
        else:
            predicted_clean = noisy_images - prediction

        if strategy_name == 'rflow':
            target = images - noise
            if use_fp32:
                # FP32: accurate gradients (recommended)
                mse_loss = ((prediction.float() - target.float()) ** 2).mean()
            else:
                # BF16: reproduces old behavior (suboptimal gradients)
                mse_loss = ((prediction - target) ** 2).mean()
        else:
            if use_fp32:
                mse_loss = ((prediction.float() - noise.float()) ** 2).mean()
            else:
                mse_loss = ((prediction - noise) ** 2).mean()

        # Perceptual loss always uses FP32 (pretrained networks need it)
        p_loss = perceptual_fn(predicted_clean.float(), images.float()) if perceptual_weight > 0 else torch.tensor(0.0, device=images.device)
        total_loss = mse_loss + perceptual_weight * p_loss
        return total_loss, mse_loss, p_loss, predicted_clean

    def _forward_dual(
        model: nn.Module,
        perceptual_fn: nn.Module,
        model_input: torch.Tensor,
        timesteps: torch.Tensor,
        images_0: torch.Tensor,
        images_1: torch.Tensor,
        noise_0: torch.Tensor,
        noise_1: torch.Tensor,
        noisy_0: torch.Tensor,
        noisy_1: torch.Tensor,
        perceptual_weight: float,
        strategy_name: str,
        num_train_timesteps: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prediction = model(model_input, timesteps)
        pred_0 = prediction[:, 0:1, :, :]
        pred_1 = prediction[:, 1:2, :, :]

        if strategy_name == 'rflow':
            t_normalized = timesteps.float() / float(num_train_timesteps)
            t_expanded = t_normalized.view(-1, 1, 1, 1)
            clean_0 = noisy_0 + t_expanded * pred_0
            clean_1 = noisy_1 + t_expanded * pred_1
            target_0 = images_0 - noise_0
            target_1 = images_1 - noise_1
            if use_fp32:
                # FP32: accurate gradients (recommended)
                mse_loss = (((pred_0.float() - target_0.float()) ** 2).mean() + ((pred_1.float() - target_1.float()) ** 2).mean()) / 2
            else:
                # BF16: reproduces old behavior (suboptimal gradients)
                mse_loss = (((pred_0 - target_0) ** 2).mean() + ((pred_1 - target_1) ** 2).mean()) / 2
        else:
            clean_0 = noisy_0 - pred_0
            clean_1 = noisy_1 - pred_1
            if use_fp32:
                mse_loss = (((pred_0.float() - noise_0.float()) ** 2).mean() + ((pred_1.float() - noise_1.float()) ** 2).mean()) / 2
            else:
                mse_loss = (((pred_0 - noise_0) ** 2).mean() + ((pred_1 - noise_1) ** 2).mean()) / 2

        if perceptual_weight > 0:
            # Perceptual loss always uses FP32 (pretrained networks need it)
            p_loss = (perceptual_fn(clean_0.float(), images_0.float()) + perceptual_fn(clean_1.float(), images_1.float())) / 2
        else:
            p_loss = torch.tensor(0.0, device=images_0.device)

        total_loss = mse_loss + perceptual_weight * p_loss
        return total_loss, mse_loss, p_loss, clean_0, clean_1

    trainer._compiled_forward_single = torch.compile(
        _forward_single, mode="reduce-overhead", fullgraph=True
    )
    trainer._compiled_forward_dual = torch.compile(
        _forward_dual, mode="reduce-overhead", fullgraph=True
    )

    if trainer.is_main_process:
        precision = "FP32" if use_fp32 else "BF16 (legacy)"
        logger.info(f"Compiled fused forward passes (CUDA graphs enabled, MSE precision: {precision})")
