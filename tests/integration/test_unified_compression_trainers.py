"""Unit tests for unified compression trainers (VAE, VQVAE, DCAE).

Tests verify that the spatial_dims unification works correctly for both 2D and 3D modes.
Uses synthetic data to avoid slow dataset loading.
"""
import os
import pytest
import torch
from hydra import compose, initialize_config_dir


# =============================================================================
# Fixtures
# =============================================================================

# Path to configs directory
CONFIGS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'configs')
CONFIGS_DIR = os.path.abspath(CONFIGS_DIR)


@pytest.fixture(scope="module")
def vae_cfg():
    """Load VAE config from actual config files."""
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="vae", overrides=[
            "mode=multi_modality",
            "training.epochs=1",
            "training.batch_size=2",
            "training.warmup_epochs=0",
            "training.use_compile=false",
            "training.use_ema=false",
            "training.logging.flops=false",
            "vae.disable_gan=true",
            "vae.perceptual_weight=0.0",
            "+save_dir_override=/tmp/test/vae_test",
        ])
        return cfg


@pytest.fixture(scope="module")
def vae_3d_cfg():
    """Load 3D VAE config from actual config files."""
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="vae_3d", overrides=[
            "mode=multi_modality",
            "training.epochs=1",
            "training.batch_size=1",
            "training.warmup_epochs=0",
            "training.use_compile=false",
            "training.use_ema=false",
            "training.logging.flops=false",
            "++vae_3d.disable_gan=true",
            "++vae_3d.perceptual_weight=0.0",
            "++vae_3d.gradient_checkpointing=false",
            "+save_dir_override=/tmp/test/vae_3d_test",
        ])
        return cfg


@pytest.fixture(scope="module")
def vqvae_cfg():
    """Load VQVAE config from actual config files."""
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="vqvae", overrides=[
            "mode=multi_modality",
            "training.epochs=1",
            "training.batch_size=2",
            "training.warmup_epochs=0",
            "training.use_compile=false",
            "training.use_ema=false",
            "training.logging.flops=false",
            "vqvae.disable_gan=true",
            "vqvae.perceptual_weight=0.0",
            "+save_dir_override=/tmp/test/vqvae_test",
        ])
        return cfg


@pytest.fixture(scope="module")
def vqvae_3d_cfg():
    """Load 3D VQVAE config from actual config files."""
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="vqvae_3d", overrides=[
            "mode=multi_modality",
            "training.epochs=1",
            "training.batch_size=1",
            "training.warmup_epochs=0",
            "training.use_compile=false",
            "training.use_ema=false",
            "training.logging.flops=false",
            "++vqvae_3d.disable_gan=true",
            "++vqvae_3d.perceptual_weight=0.0",
            "++vqvae_3d.gradient_checkpointing=false",
            "+save_dir_override=/tmp/test/vqvae_3d_test",
        ])
        return cfg


@pytest.fixture(scope="module")
def dcae_cfg():
    """Load DCAE config from actual config files."""
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="dcae", overrides=[
            "mode=multi_modality",
            "training.epochs=1",
            "training.batch_size=2",
            "training.warmup_epochs=0",
            "training.use_compile=false",
            "training.use_ema=false",
            "training.logging.flops=false",
            "++dcae.disable_gan=true",
            "++dcae.perceptual_weight=0.0",
            "+save_dir_override=/tmp/test/dcae_test",
        ])
        return cfg


@pytest.fixture(scope="module")
def dcae_3d_cfg():
    """Load 3D DCAE config from actual config files."""
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base=None):
        cfg = compose(config_name="dcae_3d", overrides=[
            "mode=multi_modality",
            "training.epochs=1",
            "training.batch_size=1",
            "training.warmup_epochs=0",
            "training.use_compile=false",
            "training.use_ema=false",
            "training.logging.flops=false",
            "++dcae_3d.disable_gan=true",
            "++dcae_3d.perceptual_weight=0.0",
            "++dcae_3d.gradient_checkpointing=false",
            "+save_dir_override=/tmp/test/dcae_3d_test",
        ])
        return cfg


# =============================================================================
# Import Tests
# =============================================================================

class TestImports:
    """Test that all trainers can be imported and have correct interface."""

    def test_import_unified_trainers(self):
        """Verify unified trainers are importable."""
        from medgen.pipeline import VAETrainer, VQVAETrainer, DCAETrainer
        assert VAETrainer is not None
        assert VQVAETrainer is not None
        assert DCAETrainer is not None

    def test_spatial_dims_parameter_exists(self):
        """Verify all unified trainers have spatial_dims parameter."""
        import inspect
        from medgen.pipeline import VAETrainer, VQVAETrainer, DCAETrainer

        for cls in [VAETrainer, VQVAETrainer, DCAETrainer]:
            sig = inspect.signature(cls.__init__)
            assert 'spatial_dims' in sig.parameters, f"{cls.__name__} missing spatial_dims"
            assert sig.parameters['spatial_dims'].default == 2, f"{cls.__name__} default should be 2"

    def test_factory_methods_exist(self):
        """Verify all trainers have create_2d and create_3d factory methods."""
        from medgen.pipeline import VAETrainer, VQVAETrainer, DCAETrainer

        for cls in [VAETrainer, VQVAETrainer, DCAETrainer]:
            assert hasattr(cls, 'create_2d'), f"{cls.__name__} missing create_2d"
            assert hasattr(cls, 'create_3d'), f"{cls.__name__} missing create_3d"
            assert callable(cls.create_2d), f"{cls.__name__}.create_2d not callable"
            assert callable(cls.create_3d), f"{cls.__name__}.create_3d not callable"


# =============================================================================
# VAETrainer Tests
# =============================================================================

class TestVAETrainer:
    """Tests for unified VAETrainer."""

    def test_init_2d(self, vae_cfg):
        """Test 2D VAE trainer initialization."""
        from medgen.pipeline import VAETrainer
        trainer = VAETrainer(vae_cfg, spatial_dims=2)
        assert trainer.spatial_dims == 2

    def test_init_3d(self, vae_3d_cfg):
        """Test 3D VAE trainer initialization."""
        from medgen.pipeline import VAETrainer
        trainer = VAETrainer(vae_3d_cfg, spatial_dims=3)
        assert trainer.spatial_dims == 3

    def test_create_2d_factory(self, vae_cfg):
        """Test create_2d factory method."""
        from medgen.pipeline import VAETrainer
        trainer = VAETrainer.create_2d(vae_cfg)
        assert trainer.spatial_dims == 2

    def test_create_3d_factory(self, vae_3d_cfg):
        """Test create_3d factory method."""
        from medgen.pipeline import VAETrainer
        trainer = VAETrainer.create_3d(vae_3d_cfg)
        assert trainer.spatial_dims == 3

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_setup_model_2d(self, vae_cfg):
        """Test 2D VAE model setup."""
        from medgen.pipeline import VAETrainer
        trainer = VAETrainer(vae_cfg, spatial_dims=2)
        trainer.setup_model()
        assert trainer.model is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_setup_model_3d(self, vae_3d_cfg):
        """Test 3D VAE model setup."""
        from medgen.pipeline import VAETrainer
        trainer = VAETrainer(vae_3d_cfg, spatial_dims=3)
        trainer.setup_model()
        assert trainer.model is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_train_step_2d(self, vae_cfg):
        """Test 2D VAE training step with synthetic data."""
        from medgen.pipeline import VAETrainer
        trainer = VAETrainer(vae_cfg, spatial_dims=2)
        trainer.setup_model()

        # Synthetic batch: (images, seg_masks)
        image_size = vae_cfg.model.image_size
        batch = (
            torch.randn(2, 1, image_size, image_size, device=trainer.device),
            torch.zeros(2, 1, image_size, image_size, device=trainer.device),
        )
        result = trainer.train_step(batch)

        assert result.total_loss > 0
        assert result.reconstruction_loss >= 0

    @pytest.mark.timeout(120)
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.slow
    def test_train_step_3d(self, vae_3d_cfg):
        """Test 3D VAE training step with synthetic data.

        Note: Requires >24GB VRAM for full-size volumes. Skip with -m "not slow".
        """
        from medgen.pipeline import VAETrainer
        trainer = VAETrainer(vae_3d_cfg, spatial_dims=3)
        trainer.setup_model()

        # Use smaller volume to avoid OOM
        d, h, w = 32, 64, 64  # Smaller than config default
        batch = (
            torch.randn(1, 1, d, h, w, device=trainer.device),
            torch.zeros(1, 1, d, h, w, device=trainer.device),
        )
        result = trainer.train_step(batch)

        assert result.total_loss > 0
        assert result.reconstruction_loss >= 0


# =============================================================================
# VQVAETrainer Tests
# =============================================================================

class TestVQVAETrainer:
    """Tests for unified VQVAETrainer."""

    def test_init_2d(self, vqvae_cfg):
        """Test 2D VQVAE trainer initialization."""
        from medgen.pipeline import VQVAETrainer
        trainer = VQVAETrainer(vqvae_cfg, spatial_dims=2)
        assert trainer.spatial_dims == 2

    def test_init_3d(self, vqvae_3d_cfg):
        """Test 3D VQVAE trainer initialization."""
        from medgen.pipeline import VQVAETrainer
        trainer = VQVAETrainer(vqvae_3d_cfg, spatial_dims=3)
        assert trainer.spatial_dims == 3

    def test_create_2d_factory(self, vqvae_cfg):
        """Test create_2d factory method."""
        from medgen.pipeline import VQVAETrainer
        trainer = VQVAETrainer.create_2d(vqvae_cfg)
        assert trainer.spatial_dims == 2

    def test_create_3d_factory(self, vqvae_3d_cfg):
        """Test create_3d factory method."""
        from medgen.pipeline import VQVAETrainer
        trainer = VQVAETrainer.create_3d(vqvae_3d_cfg)
        assert trainer.spatial_dims == 3

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_setup_model_2d(self, vqvae_cfg):
        """Test 2D VQVAE model setup."""
        from medgen.pipeline import VQVAETrainer
        trainer = VQVAETrainer(vqvae_cfg, spatial_dims=2)
        trainer.setup_model()
        assert trainer.model is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_setup_model_3d(self, vqvae_3d_cfg):
        """Test 3D VQVAE model setup."""
        from medgen.pipeline import VQVAETrainer
        trainer = VQVAETrainer(vqvae_3d_cfg, spatial_dims=3)
        trainer.setup_model()
        assert trainer.model is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_train_step_2d(self, vqvae_cfg):
        """Test 2D VQVAE training step with synthetic data."""
        from medgen.pipeline import VQVAETrainer
        trainer = VQVAETrainer(vqvae_cfg, spatial_dims=2)
        trainer.setup_model()

        image_size = vqvae_cfg.model.image_size
        batch = (
            torch.randn(2, 1, image_size, image_size, device=trainer.device),
            torch.zeros(2, 1, image_size, image_size, device=trainer.device),
        )
        result = trainer.train_step(batch)

        assert result.total_loss > 0
        assert result.reconstruction_loss >= 0

    @pytest.mark.timeout(120)
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.slow
    def test_train_step_3d(self, vqvae_3d_cfg):
        """Test 3D VQVAE training step with synthetic data.

        Note: Requires >24GB VRAM for full-size volumes. Skip with -m "not slow".
        """
        from medgen.pipeline import VQVAETrainer
        trainer = VQVAETrainer(vqvae_3d_cfg, spatial_dims=3)
        trainer.setup_model()

        # Use smaller volume to avoid OOM
        d, h, w = 32, 64, 64
        batch = (
            torch.randn(1, 1, d, h, w, device=trainer.device),
            torch.zeros(1, 1, d, h, w, device=trainer.device),
        )
        result = trainer.train_step(batch)

        assert result.total_loss > 0
        assert result.reconstruction_loss >= 0


# =============================================================================
# DCAETrainer Tests
# =============================================================================

class TestDCAETrainer:
    """Tests for unified DCAETrainer."""

    def test_init_2d(self, dcae_cfg):
        """Test 2D DCAE trainer initialization."""
        from medgen.pipeline import DCAETrainer
        trainer = DCAETrainer(dcae_cfg, spatial_dims=2)
        assert trainer.spatial_dims == 2

    def test_init_3d(self, dcae_3d_cfg):
        """Test 3D DCAE trainer initialization."""
        from medgen.pipeline import DCAETrainer
        trainer = DCAETrainer(dcae_3d_cfg, spatial_dims=3)
        assert trainer.spatial_dims == 3

    def test_create_2d_factory(self, dcae_cfg):
        """Test create_2d factory method."""
        from medgen.pipeline import DCAETrainer
        trainer = DCAETrainer.create_2d(dcae_cfg)
        assert trainer.spatial_dims == 2

    def test_create_3d_factory(self, dcae_3d_cfg):
        """Test create_3d factory method."""
        from medgen.pipeline import DCAETrainer
        trainer = DCAETrainer.create_3d(dcae_3d_cfg)
        assert trainer.spatial_dims == 3

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_setup_model_2d(self, dcae_cfg):
        """Test 2D DCAE model setup."""
        from medgen.pipeline import DCAETrainer
        trainer = DCAETrainer(dcae_cfg, spatial_dims=2)
        trainer.setup_model()
        assert trainer.model is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_setup_model_3d(self, dcae_3d_cfg):
        """Test 3D DCAE model setup."""
        from medgen.pipeline import DCAETrainer
        trainer = DCAETrainer(dcae_3d_cfg, spatial_dims=3)
        trainer.setup_model()
        assert trainer.model is not None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_train_step_2d(self, dcae_cfg):
        """Test 2D DCAE training step with synthetic data."""
        from medgen.pipeline import DCAETrainer
        trainer = DCAETrainer(dcae_cfg, spatial_dims=2)
        trainer.setup_model()

        image_size = dcae_cfg.dcae.image_size
        batch = (
            torch.randn(2, 1, image_size, image_size, device=trainer.device),
            torch.zeros(2, 1, image_size, image_size, device=trainer.device),
        )
        result = trainer.train_step(batch)

        assert result.total_loss > 0
        assert result.reconstruction_loss >= 0

    @pytest.mark.timeout(120)
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    @pytest.mark.slow
    def test_train_step_3d(self, dcae_3d_cfg):
        """Test 3D DCAE training step with synthetic data.

        Note: Requires >24GB VRAM for full-size volumes. Skip with -m "not slow".
        """
        from medgen.pipeline import DCAETrainer
        trainer = DCAETrainer(dcae_3d_cfg, spatial_dims=3)
        trainer.setup_model()

        # Use smaller volume to avoid OOM
        d, h, w = 32, 64, 64
        batch = (
            torch.randn(1, 1, d, h, w, device=trainer.device),
            torch.zeros(1, 1, d, h, w, device=trainer.device),
        )
        result = trainer.train_step(batch)

        assert result.total_loss > 0
        assert result.reconstruction_loss >= 0


# =============================================================================
# Invalid Input Tests
# =============================================================================

class TestInvalidInputs:
    """Test error handling for invalid inputs."""

    def test_invalid_spatial_dims(self, vae_cfg):
        """Test that invalid spatial_dims raises error."""
        from medgen.pipeline import VAETrainer

        with pytest.raises(ValueError, match="spatial_dims must be 2 or 3"):
            VAETrainer(vae_cfg, spatial_dims=4)

        with pytest.raises(ValueError, match="spatial_dims must be 2 or 3"):
            VAETrainer(vae_cfg, spatial_dims=1)
