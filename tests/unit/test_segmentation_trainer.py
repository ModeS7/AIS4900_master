"""Smoke tests for SegmentationTrainer.

These tests catch runtime errors like incorrect parameter names,
missing imports, and basic functionality issues.
"""

import pytest
import torch
import tempfile
from pathlib import Path
from omegaconf import OmegaConf


@pytest.fixture
def minimal_seg_config():
    """Minimal config for SegmentationTrainer instantiation."""
    return OmegaConf.create({
        'paths': {
            'name': 'local',
            'model_dir': '/tmp/test_seg',
            'data_dir': '/tmp/data',
            'fov_mm': 240.0,
        },
        'model': {
            'spatial_dims': 2,
            'image_size': 64,
            'in_channels': 1,
            'out_channels': 1,
            'init_filters': 8,  # Small for fast tests
            'blocks_down': [1, 1, 1],
            'blocks_up': [1, 1],
            'dropout_prob': 0.0,
        },
        'training': {
            'name': 'test_',
            'epochs': 2,
            'batch_size': 2,
            'learning_rate': 1e-4,
            'weight_decay': 1e-5,
            'warmup_epochs': 1,
            'gradient_clip_norm': 1.0,
            'patience': 9999,
            'val_every': 1,
            'save_best': False,
            'save_last': False,
        },
        'data': {
            'modality': 'bravo',
            'real_dir': '/tmp/data/train',
        },
        'evaluation': {
            'fov_mm': 240.0,
            'size_bins': {
                'tiny': [0, 10],
                'small': [10, 20],
                'medium': [20, 30],
                'large': [30, 1000],
            },
        },
    })


@pytest.fixture
def minimal_seg_config_3d(minimal_seg_config):
    """3D variant of minimal config."""
    cfg = minimal_seg_config.copy()
    cfg.model.spatial_dims = 3
    cfg.model.image_size = 32
    cfg.volume = {
        'height': 32,
        'width': 32,
        'pad_depth_to': 16,
    }
    return cfg


class TestSegmentationTrainerInit:
    """Test SegmentationTrainer initialization."""

    def test_import(self):
        """SMOKE: SegmentationTrainer can be imported."""
        from medgen.downstream import SegmentationTrainer
        assert SegmentationTrainer is not None

    def test_instantiation_2d(self, minimal_seg_config):
        """SMOKE: 2D trainer can be instantiated."""
        from medgen.downstream import SegmentationTrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_seg_config.paths.model_dir = tmpdir
            trainer = SegmentationTrainer(minimal_seg_config, spatial_dims=2)
            assert trainer.spatial_dims == 2

    def test_instantiation_3d(self, minimal_seg_config_3d):
        """SMOKE: 3D trainer can be instantiated."""
        from medgen.downstream import SegmentationTrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_seg_config_3d.paths.model_dir = tmpdir
            trainer = SegmentationTrainer(minimal_seg_config_3d, spatial_dims=3)
            assert trainer.spatial_dims == 3

    def test_create_2d_factory(self, minimal_seg_config):
        """SMOKE: create_2d factory method works."""
        from medgen.downstream import SegmentationTrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_seg_config.paths.model_dir = tmpdir
            trainer = SegmentationTrainer.create_2d(minimal_seg_config)
            assert trainer.spatial_dims == 2

    def test_create_3d_factory(self, minimal_seg_config_3d):
        """SMOKE: create_3d factory method works."""
        from medgen.downstream import SegmentationTrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_seg_config_3d.paths.model_dir = tmpdir
            trainer = SegmentationTrainer.create_3d(minimal_seg_config_3d)
            assert trainer.spatial_dims == 3


class TestSegmentationTrainerSetup:
    """Test SegmentationTrainer.setup_model() - catches parameter errors."""

    def test_setup_model_2d(self, minimal_seg_config):
        """SMOKE: setup_model() runs without errors for 2D.

        This catches issues like wrong parameter names (e.g., min_lr vs eta_min).
        """
        from medgen.downstream import SegmentationTrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_seg_config.paths.model_dir = tmpdir
            trainer = SegmentationTrainer(minimal_seg_config, spatial_dims=2)
            trainer.setup_model()

            # Verify components are initialized
            assert trainer.model is not None
            assert trainer.optimizer is not None
            assert trainer.lr_scheduler is not None
            assert trainer.loss_fn is not None

    def test_setup_model_3d(self, minimal_seg_config_3d):
        """SMOKE: setup_model() runs without errors for 3D."""
        from medgen.downstream import SegmentationTrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_seg_config_3d.paths.model_dir = tmpdir
            trainer = SegmentationTrainer(minimal_seg_config_3d, spatial_dims=3)
            trainer.setup_model()

            assert trainer.model is not None
            assert trainer.optimizer is not None
            assert trainer.lr_scheduler is not None


class TestSegmentationTrainerForward:
    """Test forward pass through the model."""

    def test_train_step_2d(self, minimal_seg_config):
        """SMOKE: Single training step executes without error."""
        from medgen.downstream import SegmentationTrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_seg_config.paths.model_dir = tmpdir
            trainer = SegmentationTrainer(minimal_seg_config, spatial_dims=2)
            trainer.setup_model()

            # Create dummy batch
            batch = {
                'image': torch.rand(2, 1, 64, 64),
                'seg': (torch.rand(2, 1, 64, 64) > 0.5).float(),
            }

            # Execute training step
            result = trainer.train_step(batch)

            assert result.total_loss > 0
            assert not torch.isnan(torch.tensor(result.total_loss))

    def test_train_step_3d(self, minimal_seg_config_3d):
        """SMOKE: Single 3D training step executes without error."""
        from medgen.downstream import SegmentationTrainer

        with tempfile.TemporaryDirectory() as tmpdir:
            minimal_seg_config_3d.paths.model_dir = tmpdir
            trainer = SegmentationTrainer(minimal_seg_config_3d, spatial_dims=3)
            trainer.setup_model()

            # Create dummy batch
            batch = {
                'image': torch.rand(1, 1, 16, 32, 32),
                'seg': (torch.rand(1, 1, 16, 32, 32) > 0.5).float(),
            }

            # Execute training step
            result = trainer.train_step(batch)

            assert result.total_loss > 0
            assert not torch.isnan(torch.tensor(result.total_loss))


class TestSegmentationLoss:
    """Test SegmentationLoss function directly."""

    def test_loss_instantiation(self):
        """SMOKE: SegmentationLoss can be instantiated."""
        from medgen.losses import SegmentationLoss

        loss_fn = SegmentationLoss(
            bce_weight=1.0,
            dice_weight=1.0,
            boundary_weight=0.5,
        )
        assert loss_fn is not None

    def test_loss_forward_2d(self):
        """SMOKE: Loss forward pass works for 2D."""
        from medgen.losses import SegmentationLoss

        loss_fn = SegmentationLoss(spatial_dims=2)

        logits = torch.randn(2, 1, 64, 64)
        target = (torch.rand(2, 1, 64, 64) > 0.5).float()

        total_loss, breakdown = loss_fn(logits, target)

        assert total_loss.item() > 0
        assert 'bce' in breakdown
        assert 'dice' in breakdown
        assert 'boundary' in breakdown

    def test_loss_forward_3d(self):
        """SMOKE: Loss forward pass works for 3D."""
        from medgen.losses import SegmentationLoss

        loss_fn = SegmentationLoss(spatial_dims=3)

        logits = torch.randn(1, 1, 8, 16, 16)
        target = (torch.rand(1, 1, 8, 16, 16) > 0.5).float()

        total_loss, breakdown = loss_fn(logits, target)

        assert total_loss.item() > 0
        assert 'bce' in breakdown
