"""
MedGen Models Package.

Provides diffusion model architectures including UNet and DiT.
"""

from .dit import DiT, create_dit, DiT_S, DiT_B, DiT_L, DiT_XL, DIT_VARIANTS
from .factory import create_diffusion_model, get_model_type, is_transformer_model
from .embeddings import (
    PatchEmbed2D,
    PatchEmbed3D,
    TimestepEmbedder,
    get_2d_sincos_pos_embed,
    get_3d_sincos_pos_embed,
)
from .dit_blocks import DiTBlock, Attention, CrossAttention, Mlp, FinalLayer, DropPath
from .autoencoder_dc_3d import AutoencoderDC3D, CheckpointedAutoencoderDC3D
from .dcae_adaptive_layers import AdaptiveOutputConv2d, AdaptiveInputConv2d
from .dcae_structured import StructuredAutoencoderDC

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

# Backward compatibility aliases
SiT = DiT
SiTBlock = DiTBlock
SIT_VARIANTS = DIT_VARIANTS
create_sit = create_dit
SiT_S = DiT_S
SiT_B = DiT_B
SiT_L = DiT_L
SiT_XL = DiT_XL

__all__ = [
    # Main model classes
    "DiT",
    "create_dit",
    "DiT_S",
    "DiT_B",
    "DiT_L",
    "DiT_XL",
    "DIT_VARIANTS",
    # Backward compat
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
    "DiTBlock",
    "SiTBlock",
    "Attention",
    "CrossAttention",
    "Mlp",
    "FinalLayer",
    "DropPath",
    # 3D Autoencoders
    "AutoencoderDC3D",
    "CheckpointedAutoencoderDC3D",
    # DC-AE 1.5 Structured Latent Space
    "AdaptiveOutputConv2d",
    "AdaptiveInputConv2d",
    "StructuredAutoencoderDC",
    # ControlNet
    "create_controlnet_for_unet",
    "freeze_unet_for_controlnet",
    "unfreeze_unet",
    "ControlNetConditionedUNet",
    "load_controlnet_checkpoint",
    "save_controlnet_checkpoint",
    "ControlNetGenerationWrapper",
]
