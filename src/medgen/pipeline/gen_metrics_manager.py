"""Generation metrics management for diffusion training.

This module provides:
- GenerationMetricsManager: Manages optional generation metrics (FID, KID, CMMD)

Consolidates the generation metrics logic from DiffusionTrainer into a reusable class.
"""
import hashlib
import logging
from typing import TYPE_CHECKING, Optional

import torch
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from medgen.diffusion import DiffusionSpace
    from medgen.metrics.generation import GenerationMetrics, GenerationMetricsConfig

logger = logging.getLogger(__name__)


class GenerationMetricsManager:
    """Manages optional generation metrics (FID, KID, CMMD) for diffusion training.

    Handles:
    - Lazy initialization of GenerationMetrics
    - Fixed conditioning setup for seg_conditioned mode
    - Reference feature caching
    - Epoch, extended, and test metrics computation
    """

    def __init__(
        self,
        enabled: bool,
        spatial_dims: int,
        mode_name: str,
        image_size: int,
        device: torch.device,
        save_dir: str,
        space: Optional['DiffusionSpace'] = None,
        samples_per_epoch: int = 100,
        samples_extended: int = 500,
        samples_test: int = 1000,
        steps_per_epoch: int = 10,
        steps_extended: int = 25,
        steps_test: int = 50,
        cache_dir: str | None = None,
        feature_batch_size: int = 16,
        original_depth: int | None = None,
        size_bin_edges: list[float] | None = None,
        size_bin_fov_mm: float = 240.0,
        is_main_process: bool = True,
    ) -> None:
        """Initialize generation metrics manager.

        Args:
            enabled: Whether generation metrics are enabled.
            spatial_dims: Spatial dimensions (2 or 3).
            mode_name: Training mode name.
            image_size: Image size.
            device: Compute device.
            save_dir: Directory to save generated samples.
            space: Optional DiffusionSpace for encoding/decoding.
            samples_per_epoch: Number of samples for epoch metrics.
            samples_extended: Number of samples for extended metrics.
            samples_test: Number of samples for test metrics.
            steps_per_epoch: Diffusion steps for epoch generation.
            steps_extended: Diffusion steps for extended generation.
            steps_test: Diffusion steps for test generation.
            cache_dir: Cache directory for features.
            feature_batch_size: Batch size for feature extraction.
            original_depth: Original depth for 3D (excludes padded slices).
            size_bin_edges: Size bin edges for seg_conditioned mode.
            size_bin_fov_mm: FOV in mm for size bins.
            is_main_process: Whether this is the main process.
        """
        self.enabled = enabled
        self.spatial_dims = spatial_dims
        self.mode_name = mode_name
        self.image_size = image_size
        self.device = device
        self.save_dir = save_dir
        self.space = space
        self.is_main_process = is_main_process

        # Config
        self.samples_per_epoch = samples_per_epoch
        self.samples_extended = samples_extended
        self.samples_test = samples_test
        self.steps_per_epoch = steps_per_epoch
        self.steps_extended = steps_extended
        self.steps_test = steps_test
        self.cache_dir = cache_dir
        self.feature_batch_size = feature_batch_size
        self.original_depth = original_depth
        self.size_bin_edges = size_bin_edges
        self.size_bin_fov_mm = size_bin_fov_mm

        # State
        self._metrics: GenerationMetrics | None = None
        self._config: GenerationMetricsConfig | None = None
        self._initialized = False

    def _create_config(self) -> 'GenerationMetricsConfig':
        """Create GenerationMetricsConfig."""
        from medgen.metrics.generation import GenerationMetricsConfig
        return GenerationMetricsConfig(
            enabled=True,
            samples_per_epoch=self.samples_per_epoch,
            samples_extended=self.samples_extended,
            samples_test=self.samples_test,
            steps_per_epoch=self.steps_per_epoch,
            steps_extended=self.steps_extended,
            steps_test=self.steps_test,
            cache_dir=self.cache_dir,
            feature_batch_size=self.feature_batch_size,
            original_depth=self.original_depth,
            size_bin_edges=self.size_bin_edges,
            size_bin_fov_mm=self.size_bin_fov_mm,
        )

    def create_metrics(self) -> Optional['GenerationMetrics']:
        """Create GenerationMetrics instance.

        Returns:
            GenerationMetrics instance if enabled, None otherwise.
        """
        if not self.enabled:
            return None

        from medgen.metrics.generation import GenerationMetrics

        self._config = self._create_config()
        self._metrics = GenerationMetrics(
            self._config,
            self.device,
            self.save_dir,
            space=self.space,
            mode_name=self.mode_name,
        )

        if self.is_main_process:
            logger.info("Generation metrics initialized (caching happens at training start)")

        return self._metrics

    def initialize(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset | None,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
    ) -> None:
        """Initialize metrics with reference data.

        Sets up fixed conditioning for seg_conditioned mode and caches
        reference features for FID/KID/CMMD computation.

        Args:
            train_dataset: Training dataset.
            val_dataset: Validation dataset (optional).
            train_loader: Training data loader.
            val_loader: Validation data loader (optional).
        """
        if not self.enabled or self._metrics is None or self._initialized:
            return

        # Set fixed conditioning for seg_conditioned mode
        if self.mode_name in ('seg_conditioned', 'seg_conditioned_input') or self.mode_name == 'seg':
            seg_channel_idx = 0
        else:
            # For conditional modes, seg is typically channel 2 (after t1_pre, t1_gd)
            seg_channel_idx = 2

        self._metrics.set_fixed_conditioning(
            train_dataset,
            num_masks=self.samples_extended,
            seg_channel_idx=seg_channel_idx,
        )

        # Cache reference features
        cache_key = f"{self.mode_name}_{self.image_size}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        cache_id = f"{self.mode_name}_{self.image_size}_{cache_hash}"
        self._metrics.cache_reference_features(train_loader, val_loader, cache_id=cache_id)

        self._initialized = True

        if self.is_main_process:
            logger.info(f"Generation metrics reference features cached (cache_id={cache_id})")

    @property
    def is_initialized(self) -> bool:
        """Check if metrics are initialized."""
        return self._initialized

    @property
    def metrics(self) -> Optional['GenerationMetrics']:
        """Get underlying GenerationMetrics instance."""
        return self._metrics

    @property
    def config(self) -> Optional['GenerationMetricsConfig']:
        """Get configuration."""
        return self._config

    def compute_epoch_metrics(
        self,
        model,
        strategy,
        mode,
        scheduler=None,
    ) -> dict[str, float]:
        """Compute generation metrics for epoch.

        Args:
            model: Diffusion model.
            strategy: Diffusion strategy.
            mode: Training mode.
            scheduler: Optional scheduler.

        Returns:
            Dictionary of metrics (kid, cmmd, etc.).
        """
        if not self._initialized or self._metrics is None:
            return {}
        return self._metrics.compute_epoch_metrics(model, strategy, mode, scheduler)

    def compute_extended_metrics(
        self,
        model,
        strategy,
        mode,
        scheduler=None,
    ) -> dict[str, float]:
        """Compute extended metrics (more samples).

        Args:
            model: Diffusion model.
            strategy: Diffusion strategy.
            mode: Training mode.
            scheduler: Optional scheduler.

        Returns:
            Dictionary of metrics.
        """
        if not self._initialized or self._metrics is None:
            return {}
        return self._metrics.compute_extended_metrics(model, strategy, mode, scheduler)

    def compute_test_metrics(
        self,
        model,
        strategy,
        mode,
        scheduler=None,
    ) -> dict[str, float]:
        """Compute test metrics.

        Args:
            model: Diffusion model.
            strategy: Diffusion strategy.
            mode: Training mode.
            scheduler: Optional scheduler.

        Returns:
            Dictionary of metrics.
        """
        if not self._initialized or self._metrics is None:
            return {}
        return self._metrics.compute_test_metrics(model, strategy, mode, scheduler)


def create_gen_metrics_manager_from_config(
    cfg,
    spatial_dims: int,
    device: torch.device,
    save_dir: str,
    space: Optional['DiffusionSpace'] = None,
    is_main_process: bool = True,
) -> GenerationMetricsManager:
    """Create GenerationMetricsManager from Hydra config.

    Args:
        cfg: Hydra configuration object.
        spatial_dims: Spatial dimensions.
        device: Compute device.
        save_dir: Save directory.
        space: Optional DiffusionSpace.
        is_main_process: Whether this is main process.

    Returns:
        Configured GenerationMetricsManager.
    """
    gen_cfg = cfg.training.get('generation_metrics', {})
    mode_name = cfg.mode.name

    if not gen_cfg.get('enabled', False):
        return GenerationMetricsManager(
            enabled=False,
            spatial_dims=spatial_dims,
            mode_name=mode_name,
            image_size=cfg.model.image_size,
            device=device,
            save_dir=save_dir,
            is_main_process=is_main_process,
        )

    # Use training batch_size by default for torch.compile consistency
    feature_batch_size = gen_cfg.get('feature_batch_size', None)
    if feature_batch_size is None:
        if spatial_dims == 3:
            feature_batch_size = max(32, cfg.training.get('batch_size', 16) * 16)
        else:
            feature_batch_size = cfg.training.get('batch_size', 16)

    # Use absolute cache_dir from paths config
    gen_cache_dir = gen_cfg.get('cache_dir', None)
    if gen_cache_dir is None:
        base_cache = getattr(cfg.paths, 'cache_dir', '.cache')
        gen_cache_dir = f"{base_cache}/generation_features"

    # 3D volumes are much larger - cap sample counts
    if spatial_dims == 3:
        samples_per_epoch = min(gen_cfg.get('samples_per_epoch', 1), 2)
        samples_extended = min(gen_cfg.get('samples_extended', 4), 4)
        samples_test = min(gen_cfg.get('samples_test', 10), 10)
    else:
        samples_per_epoch = gen_cfg.get('samples_per_epoch', 100)
        samples_extended = gen_cfg.get('samples_extended', 500)
        samples_test = gen_cfg.get('samples_test', 1000)

    # Get original_depth for 3D (used to exclude padded slices)
    original_depth = None
    if spatial_dims == 3:
        original_depth = cfg.volume.get('original_depth', None)

    # Get size bin config for seg_conditioned mode
    size_bin_edges = None
    size_bin_fov_mm = 240.0
    if mode_name == 'seg_conditioned':
        size_bin_cfg = cfg.mode.get('size_bins', {})
        size_bin_edges = list(size_bin_cfg.get('edges', [0, 3, 6, 10, 15, 20, 30]))
        size_bin_fov_mm = float(size_bin_cfg.get('fov_mm', 240.0))

    return GenerationMetricsManager(
        enabled=True,
        spatial_dims=spatial_dims,
        mode_name=mode_name,
        image_size=cfg.model.image_size,
        device=device,
        save_dir=save_dir,
        space=space,
        samples_per_epoch=samples_per_epoch,
        samples_extended=samples_extended,
        samples_test=samples_test,
        steps_per_epoch=gen_cfg.get('steps_per_epoch', 10),
        steps_extended=gen_cfg.get('steps_extended', 25),
        steps_test=gen_cfg.get('steps_test', 50),
        cache_dir=gen_cache_dir,
        feature_batch_size=feature_batch_size,
        original_depth=original_depth,
        size_bin_edges=size_bin_edges,
        size_bin_fov_mm=size_bin_fov_mm,
        is_main_process=is_main_process,
    )
