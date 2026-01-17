"""
MedGen Models Package.

Provides diffusion model architectures including UNet and SiT.
"""

from .sit import SiT, create_sit, SiT_S, SiT_B, SiT_L, SiT_XL, SIT_VARIANTS
from .factory import create_diffusion_model, get_model_type, is_transformer_model
from .embeddings import (
    PatchEmbed2D,
    PatchEmbed3D,
    TimestepEmbedder,
    get_2d_sincos_pos_embed,
    get_3d_sincos_pos_embed,
)
from .sit_blocks import SiTBlock, Attention, CrossAttention, Mlp, FinalLayer, DropPath
from .autoencoder_dc_3d import AutoencoderDC3D, CheckpointedAutoencoderDC3D

# ControlNet utilities
from .controlnet import (
    create_controlnet_for_unet,
    freeze_unet_for_controlnet,
    unfreeze_unet,
    ControlNetConditionedUNet,
    load_controlnet_checkpoint,
    save_controlnet_checkpoint,
    ControlNetGenerationWrapper,
)

__all__ = [
    # Main model classes
    "SiT",
    "create_sit",
    "SiT_S",
    "SiT_B",
    "SiT_L",
    "SiT_XL",
    "SIT_VARIANTS",
    # Factory
    "create_diffusion_model",
    "get_model_type",
    "is_transformer_model",
    # Embeddings
    "PatchEmbed2D",
    "PatchEmbed3D",
    "TimestepEmbedder",
    "get_2d_sincos_pos_embed",
    "get_3d_sincos_pos_embed",
    # Blocks
    "SiTBlock",
    "Attention",
    "CrossAttention",
    "Mlp",
    "FinalLayer",
    "DropPath",
    # 3D Autoencoders
    "AutoencoderDC3D",
    "CheckpointedAutoencoderDC3D",
    # ControlNet
    "create_controlnet_for_unet",
    "freeze_unet_for_controlnet",
    "unfreeze_unet",
    "ControlNetConditionedUNet",
    "load_controlnet_checkpoint",
    "save_controlnet_checkpoint",
    "ControlNetGenerationWrapper",
]
