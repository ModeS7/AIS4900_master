"""Mode configuration dataclass.

Provides type-safe access to training mode settings.
"""
from dataclasses import dataclass, field

from omegaconf import DictConfig


@dataclass
class SizeBinConfig:
    """Size bin configuration for seg_conditioned mode.

    Attributes:
        enabled: Whether size bin embedding is enabled.
        edges: Bin edges in mm (e.g., [0, 3, 6, 10, 15, 20, 30]).
        num_bins: Number of bins.
        max_count: Maximum count per bin.
        embed_dim: Embedding dimension per bin.
        fov_mm: Field of view in mm for size calculations.
    """
    enabled: bool = False
    edges: list[float] = field(default_factory=lambda: [0, 3, 6, 10, 15, 20, 30])
    num_bins: int = 7
    max_count: int = 10
    embed_dim: int = 32
    fov_mm: float = 240.0

    @classmethod
    def from_hydra(cls, cfg: DictConfig, mode_name: str) -> 'SizeBinConfig':
        """Extract size bin config from Hydra DictConfig.

        Args:
            cfg: Hydra configuration object.
            mode_name: Training mode name.

        Returns:
            SizeBinConfig instance.
        """
        enabled = (mode_name == 'seg_conditioned')
        if not enabled:
            return cls(enabled=False)

        size_bin_cfg = cfg.mode.get('size_bins', {})
        edges = list(size_bin_cfg.get('edges', [0, 3, 6, 10, 15, 20, 30]))
        return cls(
            enabled=True,
            edges=edges,
            num_bins=size_bin_cfg.get('num_bins', len(edges)),
            max_count=size_bin_cfg.get('max_count_per_bin', 10),
            embed_dim=size_bin_cfg.get('embedding_dim', 32),
            fov_mm=float(size_bin_cfg.get('fov_mm', 240.0)),
        )


@dataclass
class ModeEmbeddingConfig:
    """Mode embedding configuration for multi-modality training.

    Attributes:
        enabled: Whether mode embedding is enabled.
        strategy: Embedding strategy ('full', 'late', 'attention_only').
        dropout: Dropout probability for embedding.
        late_start_level: UNet level to start late mode embedding.
    """
    enabled: bool = False
    strategy: str = 'full'
    dropout: float = 0.2
    late_start_level: int = 2

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> 'ModeEmbeddingConfig':
        """Extract mode embedding config from Hydra DictConfig.

        Args:
            cfg: Hydra configuration object.

        Returns:
            ModeEmbeddingConfig instance.
        """
        mode = cfg.mode
        return cls(
            enabled=mode.get('use_mode_embedding', False),
            strategy=mode.get('mode_embedding_strategy', 'full'),
            dropout=mode.get('mode_embedding_dropout', 0.2),
            late_start_level=mode.get('late_mode_start_level', 2),
        )


@dataclass
class ModeConfig:
    """Training mode configuration.

    Attributes:
        name: Mode name ('seg', 'bravo', 'dual', 'multi', 'seg_conditioned').
        in_channels: Number of input channels for this mode.
        image_keys: List of image keys for dual/multi modes.
        mode_embedding: Mode embedding configuration.
        size_bin: Size bin embedding configuration.
    """
    name: str
    in_channels: int = 1
    image_keys: list[str] | None = None
    mode_embedding: ModeEmbeddingConfig = field(default_factory=ModeEmbeddingConfig)
    size_bin: SizeBinConfig = field(default_factory=SizeBinConfig)

    @property
    def is_seg_mode(self) -> bool:
        """Check if this is a segmentation mode."""
        return self.name in ('seg', 'seg_conditioned')

    @property
    def is_multi_modality(self) -> bool:
        """Check if this is a multi-modality mode."""
        return self.name == 'multi' or self.name == 'multi_modality'

    @property
    def is_dual(self) -> bool:
        """Check if this is dual mode."""
        return self.name == 'dual'

    @classmethod
    def from_hydra(cls, cfg: DictConfig) -> 'ModeConfig':
        """Extract mode config from Hydra DictConfig.

        Args:
            cfg: Hydra configuration object.

        Returns:
            ModeConfig instance.
        """
        mode = cfg.mode
        mode_name = mode.name

        # Extract image keys for dual/multi modes
        image_keys = None
        if 'image_keys' in mode:
            image_keys = list(mode.image_keys)

        return cls(
            name=mode_name,
            in_channels=mode.get('in_channels', 1),
            image_keys=image_keys,
            mode_embedding=ModeEmbeddingConfig.from_hydra(cfg),
            size_bin=SizeBinConfig.from_hydra(cfg, mode_name),
        )
