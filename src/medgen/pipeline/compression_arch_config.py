"""Architecture configuration for compression trainers (VAE, VQ-VAE, DC-AE).

Each dataclass captures the model-architecture-specific cfg.get() calls from the
trainer __init__, providing typed defaults and a from_hydra() factory method.
"""
from dataclasses import dataclass, field

from omegaconf import DictConfig


@dataclass
class VAEArchConfig:
    """VAE model architecture config (2D/3D)."""
    kl_weight: float = 1e-6
    latent_channels: int = 3
    channels: tuple[int, ...] = (128, 256, 512)
    attention_levels: tuple[bool, ...] = (False, False, True)
    num_res_blocks: int = 2

    @classmethod
    def from_hydra(cls, cfg: DictConfig, spatial_dims: int) -> 'VAEArchConfig':
        """Extract VAE architecture config from Hydra DictConfig."""
        vae_cfg = cfg.vae_3d if spatial_dims == 3 else cfg.vae
        return cls(
            kl_weight=vae_cfg.get('kl_weight', 1e-6),
            latent_channels=vae_cfg.latent_channels,
            channels=tuple(vae_cfg.channels),
            attention_levels=tuple(vae_cfg.attention_levels),
            num_res_blocks=vae_cfg.get('num_res_blocks', 2),
        )


@dataclass
class VQVAEArchConfig:
    """VQ-VAE model architecture config (2D/3D)."""
    # Quantization
    num_embeddings: int = 512
    embedding_dim: int = 64
    commitment_cost: float = 0.25
    decay: float = 0.99
    epsilon: float = 1e-5
    # Architecture
    channels: tuple[int, ...] = (96, 96, 192)
    num_res_layers: int = 3
    num_res_channels: tuple[int, ...] = (96, 96, 192)
    downsample_parameters: tuple[tuple[int, ...], ...] = ((2, 4, 1, 1),) * 3
    upsample_parameters: tuple[tuple[int, ...], ...] = ((2, 4, 1, 1, 0),) * 3
    # Segmentation mode (3D only)
    seg_mode: bool = False
    seg_loss_weights: dict = field(default_factory=lambda: {'bce': 1.0, 'dice': 1.0, 'boundary': 0.5})

    @classmethod
    def from_hydra(cls, cfg: DictConfig, spatial_dims: int) -> 'VQVAEArchConfig':
        """Extract VQ-VAE architecture config from Hydra DictConfig."""
        vqvae_cfg = cfg.vqvae_3d if spatial_dims == 3 else cfg.vqvae

        default_channels = [96, 96, 192] if spatial_dims == 2 else [64, 128]
        default_res_layers = 3 if spatial_dims == 2 else 2
        default_res_channels = [96, 96, 192] if spatial_dims == 2 else [64, 128]
        default_downsample = [[2, 4, 1, 1]] * (3 if spatial_dims == 2 else 2)
        default_upsample = [[2, 4, 1, 1, 0]] * (3 if spatial_dims == 2 else 2)

        seg_mode = vqvae_cfg.get('seg_mode', False) if spatial_dims == 3 else False
        seg_weights = dict(vqvae_cfg.get('seg_loss_weights', {})) if seg_mode else {'bce': 1.0, 'dice': 1.0, 'boundary': 0.5}

        return cls(
            num_embeddings=vqvae_cfg.get('num_embeddings', 512),
            embedding_dim=vqvae_cfg.get('embedding_dim', 64 if spatial_dims == 2 else 3),
            commitment_cost=vqvae_cfg.get('commitment_cost', 0.25),
            decay=vqvae_cfg.get('decay', 0.99),
            epsilon=vqvae_cfg.get('epsilon', 1e-5),
            channels=tuple(vqvae_cfg.get('channels', default_channels)),
            num_res_layers=vqvae_cfg.get('num_res_layers', default_res_layers),
            num_res_channels=tuple(vqvae_cfg.get('num_res_channels', default_res_channels)),
            downsample_parameters=tuple(
                tuple(p) for p in vqvae_cfg.get('downsample_parameters', default_downsample)
            ),
            upsample_parameters=tuple(
                tuple(p) for p in vqvae_cfg.get('upsample_parameters', default_upsample)
            ),
            seg_mode=seg_mode,
            seg_loss_weights=seg_weights,
        )


@dataclass
class DCAEArchConfig:
    """DC-AE model architecture config (2D/3D)."""
    l1_weight: float = 1.0
    latent_channels: int = 32
    scaling_factor: float = 1.0
    # 2D-specific
    compression_ratio: int = 32
    pretrained: str | None = None
    training_phase: int = 1
    seg_mode: bool = False
    seg_loss_weights: dict = field(default_factory=lambda: {'bce': 1.0, 'dice': 1.0, 'boundary': 0.5})
    # Structured latent (DC-AE 1.5, 2D only)
    structured_latent_enabled: bool = False
    structured_latent_min: int = 16
    structured_latent_step: int = 4
    # 3D-specific
    encoder_block_out_channels: tuple[int, ...] = ()
    decoder_block_out_channels: tuple[int, ...] = ()
    encoder_layers_per_block: tuple[int, ...] = ()
    decoder_layers_per_block: tuple[int, ...] = ()
    depth_factors: tuple[int, ...] = ()
    encoder_out_shortcut: bool = True
    decoder_in_shortcut: bool = True

    @classmethod
    def from_hydra(cls, cfg: DictConfig, spatial_dims: int) -> 'DCAEArchConfig':
        """Extract DC-AE architecture config from Hydra DictConfig."""
        dcae_cfg = cfg.dcae_3d if spatial_dims == 3 else cfg.dcae

        # Common fields
        l1_weight = dcae_cfg.get('l1_weight', 1.0)
        latent_channels = dcae_cfg.latent_channels
        scaling_factor = dcae_cfg.get('scaling_factor', 1.0)

        if spatial_dims == 2:
            seg_mode = dcae_cfg.get('seg_mode', False)
            seg_weights = dict(dcae_cfg.get('seg_loss_weights', {})) if seg_mode else {'bce': 1.0, 'dice': 1.0, 'boundary': 0.5}
            structured_cfg = dcae_cfg.get('structured_latent', {})

            return cls(
                l1_weight=l1_weight,
                latent_channels=latent_channels,
                scaling_factor=scaling_factor,
                compression_ratio=dcae_cfg.compression_ratio,
                pretrained=dcae_cfg.get('pretrained', None),
                training_phase=cfg.training.get('phase', 1),
                seg_mode=seg_mode,
                seg_loss_weights=seg_weights,
                structured_latent_enabled=structured_cfg.get('enabled', False),
                structured_latent_min=structured_cfg.get('min_channels', 16),
                structured_latent_step=structured_cfg.get('channel_step', 4),
            )
        else:
            return cls(
                l1_weight=l1_weight,
                latent_channels=latent_channels,
                scaling_factor=scaling_factor,
                encoder_block_out_channels=tuple(dcae_cfg.encoder_block_out_channels),
                decoder_block_out_channels=tuple(dcae_cfg.decoder_block_out_channels),
                encoder_layers_per_block=tuple(dcae_cfg.encoder_layers_per_block),
                decoder_layers_per_block=tuple(dcae_cfg.decoder_layers_per_block),
                depth_factors=tuple(dcae_cfg.depth_factors),
                encoder_out_shortcut=dcae_cfg.get('encoder_out_shortcut', True),
                decoder_in_shortcut=dcae_cfg.get('decoder_in_shortcut', True),
                seg_mode=False,
                training_phase=1,
                structured_latent_enabled=False,
            )
