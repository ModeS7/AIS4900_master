"""Configuration extraction for compression trainers.

This module provides:
- CompressionConfig: Dataclass holding all extracted configuration values
- CompressionConfigExtractor: Extracts config values from Hydra DictConfig

These classes consolidate the configuration extraction logic from BaseCompressionTrainer,
making it reusable and testable.
"""
from dataclasses import dataclass
from typing import Any

from omegaconf import DictConfig


@dataclass
class CompressionConfig:
    """Extracted configuration for compression training.

    All configuration values needed by compression trainers (VAE, VQ-VAE, DC-AE)
    in both 2D and 3D modes.
    """
    # Discriminator config
    disc_lr: float
    disc_num_layers: int
    disc_num_channels: int
    disable_gan: bool

    # Loss weights
    perceptual_weight: float
    perceptual_loss_type: str
    adv_weight: float
    reconstruction_weight: float

    # 3D-specific perceptual loss
    use_2_5d_perceptual: bool
    perceptual_slice_fraction: float

    # EMA config
    use_ema: bool
    ema_decay: float

    # Precision config
    pure_weights: bool
    use_compile: bool
    weight_dtype_str: str

    # Progressive training
    use_constant_lr: bool

    # Optimizer
    optimizer_betas: tuple[float, float]


class CompressionConfigExtractor:
    """Extracts configuration values from Hydra DictConfig.

    Handles the complexity of looking up values across multiple config sections
    (vae, vqvae, dcae, vae_3d, vqvae_3d, dcae_3d) with dimension-specific defaults.
    """

    # Config sections to check for trainer-specific settings
    CONFIG_SECTIONS = ('vae', 'vqvae', 'dcae', 'vae_3d', 'vqvae_3d', 'dcae_3d')

    # Default config values - can be overridden via class attributes in subclasses
    DEFAULT_DISC_LR_2D: float = 5e-4
    DEFAULT_DISC_LR_3D: float = 5e-4
    DEFAULT_PERCEPTUAL_WEIGHT_2D: float = 0.001
    DEFAULT_PERCEPTUAL_WEIGHT_3D: float = 0.001
    DEFAULT_ADV_WEIGHT_2D: float = 0.01
    DEFAULT_ADV_WEIGHT_3D: float = 0.01

    def __init__(
        self,
        cfg: DictConfig,
        spatial_dims: int,
        config_section_2d: str = 'vae',
        config_section_3d: str = 'vae_3d',
    ) -> None:
        """Initialize the extractor.

        Args:
            cfg: Hydra configuration object.
            spatial_dims: Spatial dimensions (2 or 3).
            config_section_2d: Config section name for 2D (e.g., 'vae', 'vqvae', 'dcae').
            config_section_3d: Config section name for 3D (e.g., 'vae_3d', 'vqvae_3d', 'dcae_3d').
        """
        self.cfg = cfg
        self.spatial_dims = spatial_dims
        self._config_section_2d = config_section_2d
        self._config_section_3d = config_section_3d

    def _get_config_section(self) -> str:
        """Get the config section name for this trainer's spatial_dims."""
        return self._config_section_3d if self.spatial_dims == 3 else self._config_section_2d

    def _get_config_value(self, key: str, default: Any) -> Any:
        """Get a value from any trainer config section.

        Searches through vae, vqvae, dcae, vae_3d, vqvae_3d sections
        for the specified key.

        Args:
            key: Config key to look for.
            default: Default value if key not found.

        Returns:
            Value from config or default.
        """
        for section in self.CONFIG_SECTIONS:
            if section in self.cfg:
                return self.cfg[section].get(key, default)
        return default

    def _get_config_value_dimensional(
        self, key: str, default_2d: Any, default_3d: Any
    ) -> Any:
        """Get config value with dimension-specific defaults.

        Searches the trainer's config section first (e.g., 'vae' or 'vae_3d'),
        then falls back to dimension-specific default.

        Args:
            key: Config key to look for.
            default_2d: Default for 2D (spatial_dims=2).
            default_3d: Default for 3D (spatial_dims=3).

        Returns:
            Config value or appropriate default.
        """
        section = self._get_config_section()
        default = default_3d if self.spatial_dims == 3 else default_2d
        if section in self.cfg:
            return self.cfg[section].get(key, default)
        return default

    def get_disc_lr(self) -> float:
        """Get discriminator learning rate from config."""
        return self._get_config_value_dimensional(
            'disc_lr', self.DEFAULT_DISC_LR_2D, self.DEFAULT_DISC_LR_3D
        )

    def get_perceptual_weight(self) -> float:
        """Get perceptual loss weight from config."""
        return self._get_config_value_dimensional(
            'perceptual_weight', self.DEFAULT_PERCEPTUAL_WEIGHT_2D, self.DEFAULT_PERCEPTUAL_WEIGHT_3D
        )

    def get_adv_weight(self) -> float:
        """Get adversarial loss weight from config."""
        return self._get_config_value_dimensional(
            'adv_weight', self.DEFAULT_ADV_WEIGHT_2D, self.DEFAULT_ADV_WEIGHT_3D
        )

    def get_disable_gan(self) -> bool:
        """Get disable_gan flag from config.

        Checks progressive config first (for staged training),
        then falls back to trainer-specific config.
        """
        # Check progressive config first (for staged training)
        progressive_cfg = self.cfg.get('progressive', {})
        if progressive_cfg.get('disable_gan', False):
            return True
        return self._get_config_value_dimensional('disable_gan', False, False)

    def get_disc_num_layers(self) -> int:
        """Get discriminator number of layers from config."""
        return self._get_config_value('disc_num_layers', 3)

    def get_disc_num_channels(self) -> int:
        """Get discriminator number of channels from config."""
        return self._get_config_value('disc_num_channels', 64)

    def get_perceptual_loss_type(self) -> str:
        """Get perceptual loss type from config.

        Options:
            - 'radimagenet': MONAI's RadImageNet ResNet50 (default)
            - 'lpips': LPIPS library with VGG backbone (DC-AE paper uses this)

        Returns:
            Loss type string.
        """
        return self._get_config_value('perceptual_loss_type', 'radimagenet')

    def get_2_5d_perceptual(self) -> bool:
        """Get 2.5D perceptual loss flag (3D only).

        When enabled, perceptual loss is computed on sampled 2D slices
        rather than full 3D volumes.

        Returns:
            True if 2.5D perceptual loss is enabled.
        """
        for section in ['vae_3d', 'vqvae_3d', 'dcae_3d']:
            if section in self.cfg:
                return self.cfg[section].get('use_2_5d_perceptual', True)
        return True

    def get_perceptual_slice_fraction(self) -> float:
        """Get fraction of slices to sample for 2.5D perceptual loss.

        Returns:
            Fraction of depth slices to sample (0.0-1.0).
        """
        for section in ['vae_3d', 'vqvae_3d', 'dcae_3d']:
            if section in self.cfg:
                return self.cfg[section].get('perceptual_slice_fraction', 0.25)
        return 0.25

    def get_reconstruction_loss_weight(self) -> float:
        """Get weight for L1 reconstruction loss.

        Default is 1.0, but DC-AE may use different values.

        Returns:
            Weight for L1 loss.
        """
        return self._get_config_value('reconstruction_loss_weight', 1.0)

    def extract(self) -> CompressionConfig:
        """Extract all configuration into a dataclass.

        Returns:
            CompressionConfig with all extracted values.
        """
        # Extract precision config
        precision_cfg = self.cfg.training.get('precision', {})
        dtype_str = precision_cfg.get('dtype', 'bf16')

        # Extract optimizer betas
        optimizer_cfg = self.cfg.training.get('optimizer', {})
        betas_list = optimizer_cfg.get('betas', [0.9, 0.999])
        optimizer_betas = tuple(betas_list)

        # Extract progressive training config
        progressive_cfg = self.cfg.get('progressive', {})

        return CompressionConfig(
            # Discriminator
            disc_lr=self.get_disc_lr(),
            disc_num_layers=self.get_disc_num_layers(),
            disc_num_channels=self.get_disc_num_channels(),
            disable_gan=self.get_disable_gan(),
            # Loss weights
            perceptual_weight=self.get_perceptual_weight(),
            perceptual_loss_type=self.get_perceptual_loss_type(),
            adv_weight=self.get_adv_weight(),
            reconstruction_weight=self.get_reconstruction_loss_weight(),
            # 3D-specific
            use_2_5d_perceptual=self.get_2_5d_perceptual(),
            perceptual_slice_fraction=self.get_perceptual_slice_fraction(),
            # EMA
            use_ema=self.cfg.training.get('use_ema', True),
            ema_decay=self.cfg.training.get('ema', {}).get('decay', 0.9999),
            # Precision
            pure_weights=precision_cfg.get('pure_weights', False),
            use_compile=self.cfg.training.get('use_compile', True),
            weight_dtype_str=dtype_str,
            # Progressive training
            use_constant_lr=progressive_cfg.get('use_constant_lr', False),
            # Optimizer
            optimizer_betas=optimizer_betas,
        )
