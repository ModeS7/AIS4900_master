"""
Unit tests for configuration validation.

Tests verify that configuration validators correctly catch invalid
configurations before training starts.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch
from omegaconf import DictConfig, OmegaConf


# =============================================================================
# Helper: Create minimal valid configs
# =============================================================================


def make_minimal_config(**overrides) -> DictConfig:
    """Create minimal valid DictConfig with optional overrides."""
    base = {
        'training': {
            'epochs': 10,
            'batch_size': 4,
            'learning_rate': 1e-4,
            'use_compile': False,
            'gradient_checkpointing': False,
        },
        'model': {
            'image_size': 64,
            'spatial_dims': 2,
        },
        'paths': {
            'data_dir': '/tmp/test_data',
            'output_dir': '/tmp/test_output',
        },
        'strategy': {
            'name': 'rflow',
        },
        'mode': {
            'name': 'bravo',
        },
    }

    # Apply overrides
    cfg = OmegaConf.create(base)
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value)

    return cfg


def make_vae_config(**overrides) -> DictConfig:
    """Create VAE config with optional overrides."""
    cfg = make_minimal_config()
    cfg.vae = OmegaConf.create({
        'latent_channels': 4,
        'channels': [32, 64, 128],
    })
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value)
    return cfg


def make_vqvae_config(**overrides) -> DictConfig:
    """Create VQ-VAE config with optional overrides."""
    cfg = make_minimal_config()
    cfg.vqvae = OmegaConf.create({
        'num_embeddings': 512,
        'embedding_dim': 64,
        'channels': [32, 64, 128],
    })
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value)
    return cfg


# =============================================================================
# TestRequiredFields - Required configuration fields
# =============================================================================


class TestRequiredFields:
    """Test that required fields are validated."""

    def test_missing_mode_raises(self):
        """Config without 'mode.name' should raise validation error."""
        from medgen.core.validation import validate_diffusion_config

        cfg = make_minimal_config()
        cfg.mode.name = None  # Remove mode

        errors = validate_diffusion_config(cfg)
        # Should have an error about mode
        assert any('mode' in e.lower() for e in errors), \
            f"Should detect missing/invalid mode, got errors: {errors}"

    def test_missing_strategy_raises(self):
        """Config without valid 'strategy.name' should raise validation error."""
        from medgen.core.validation import validate_diffusion_config

        cfg = make_minimal_config()
        cfg.strategy.name = 'invalid_strategy'

        errors = validate_diffusion_config(cfg)
        assert any('strategy' in e.lower() for e in errors), \
            f"Should detect invalid strategy, got errors: {errors}"


# =============================================================================
# TestValueRanges - Numeric value range validation
# =============================================================================


class TestValueRanges:
    """Test that numeric values are in valid ranges."""

    def test_negative_epochs_invalid(self):
        """training.epochs must be > 0."""
        from medgen.core.validation import validate_common_config

        cfg = make_minimal_config(**{'training.epochs': -5})

        with patch('os.path.exists', return_value=True), \
             patch('torch.cuda.is_available', return_value=True):
            errors = validate_common_config(cfg)

        assert any('epochs' in e.lower() for e in errors), \
            f"Should detect negative epochs, got errors: {errors}"

    def test_batch_size_must_be_positive(self):
        """training.batch_size must be > 0."""
        from medgen.core.validation import validate_common_config

        cfg = make_minimal_config(**{'training.batch_size': 0})

        with patch('os.path.exists', return_value=True), \
             patch('torch.cuda.is_available', return_value=True):
            errors = validate_common_config(cfg)

        assert any('batch_size' in e.lower() for e in errors), \
            f"Should detect zero batch_size, got errors: {errors}"

    def test_learning_rate_must_be_positive(self):
        """training.learning_rate must be > 0."""
        from medgen.core.validation import validate_common_config

        cfg = make_minimal_config(**{'training.learning_rate': -0.001})

        with patch('os.path.exists', return_value=True), \
             patch('torch.cuda.is_available', return_value=True):
            errors = validate_common_config(cfg)

        assert any('learning_rate' in e.lower() for e in errors), \
            f"Should detect negative learning_rate, got errors: {errors}"

    def test_valid_values_pass(self):
        """Valid configuration values should produce no errors."""
        from medgen.core.validation import validate_common_config

        cfg = make_minimal_config()

        with patch('os.path.exists', return_value=True), \
             patch('torch.cuda.is_available', return_value=True):
            errors = validate_common_config(cfg)

        assert len(errors) == 0, f"Valid config should have no errors, got: {errors}"


# =============================================================================
# TestModeStrategyCompatibility - Mode/Strategy combinations
# =============================================================================


class TestModeStrategyCompatibility:
    """Test valid mode/strategy combinations."""

    @pytest.mark.parametrize("mode,strategy,valid", [
        ("bravo", "ddpm", True),
        ("bravo", "rflow", True),
        ("seg", "ddpm", True),
        ("seg", "rflow", True),
        ("seg_conditioned", "rflow", True),
        ("seg_conditioned", "ddpm", True),
        ("dual", "rflow", True),
        ("dual", "ddpm", True),
        ("multi", "rflow", True),
        ("multi", "ddpm", True),
        # Invalid combinations
        ("invalid_mode", "rflow", False),
        ("bravo", "invalid_strategy", False),
    ])
    def test_mode_strategy_combinations(self, mode, strategy, valid):
        """Test various mode/strategy combinations."""
        from medgen.core.validation import validate_diffusion_config

        cfg = make_minimal_config(**{
            'mode.name': mode,
            'strategy.name': strategy,
        })

        errors = validate_diffusion_config(cfg)

        if valid:
            assert len(errors) == 0, \
                f"({mode}, {strategy}) should be valid, got errors: {errors}"
        else:
            assert len(errors) > 0, \
                f"({mode}, {strategy}) should be invalid but no errors found"


# =============================================================================
# TestSpatialDimsValidation - 2D/3D configuration
# =============================================================================


class TestSpatialDimsValidation:
    """Test spatial_dims validation for 2D/3D modes."""

    @pytest.mark.parametrize("spatial_dims,valid", [
        (2, True),
        (3, True),
        (1, False),
        (4, False),
        (0, False),
        (-1, False),
    ])
    def test_spatial_dims_values(self, spatial_dims, valid):
        """Test spatial_dims value validation."""
        cfg = make_minimal_config(**{'model.spatial_dims': spatial_dims})

        # Custom validation for spatial_dims
        errors = validate_spatial_dims(cfg)

        if valid:
            assert len(errors) == 0, \
                f"spatial_dims={spatial_dims} should be valid, got errors: {errors}"
        else:
            assert len(errors) > 0, \
                f"spatial_dims={spatial_dims} should be invalid but no errors found"


def validate_spatial_dims(cfg: DictConfig):
    """Validate spatial_dims is 2 or 3."""
    errors = []
    spatial_dims = cfg.model.get('spatial_dims', 2)
    if spatial_dims not in [2, 3]:
        errors.append(f"spatial_dims must be 2 or 3, got {spatial_dims}")
    return errors


# =============================================================================
# TestVAEConfigValidation - VAE-specific validation
# =============================================================================


class TestVAEConfigValidation:
    """Test VAE configuration validation."""

    def test_missing_vae_section_detected(self):
        """Missing vae section should be detected."""
        from medgen.core.validation import validate_vae_config

        cfg = make_minimal_config()  # No vae section

        errors = validate_vae_config(cfg)
        assert any('vae' in e.lower() for e in errors), \
            f"Should detect missing vae section, got errors: {errors}"

    def test_invalid_latent_channels(self):
        """vae.latent_channels <= 0 should be invalid."""
        from medgen.core.validation import validate_vae_config

        cfg = make_vae_config(**{'vae.latent_channels': 0})

        errors = validate_vae_config(cfg)
        assert any('latent_channels' in e.lower() for e in errors), \
            f"Should detect invalid latent_channels, got errors: {errors}"

    def test_empty_channels_list(self):
        """vae.channels cannot be empty."""
        from medgen.core.validation import validate_vae_config

        cfg = make_vae_config(**{'vae.channels': []})

        errors = validate_vae_config(cfg)
        assert any('channels' in e.lower() for e in errors), \
            f"Should detect empty channels, got errors: {errors}"

    def test_valid_vae_config(self):
        """Valid VAE config should pass validation."""
        from medgen.core.validation import validate_vae_config

        cfg = make_vae_config()

        errors = validate_vae_config(cfg)
        assert len(errors) == 0, f"Valid VAE config should have no errors, got: {errors}"


# =============================================================================
# TestVQVAEConfigValidation - VQ-VAE specific validation
# =============================================================================


class TestVQVAEConfigValidation:
    """Test VQ-VAE configuration validation."""

    def test_missing_vqvae_section_detected(self):
        """Missing vqvae section should be detected."""
        from medgen.core.validation import validate_vqvae_config

        cfg = make_minimal_config()  # No vqvae section

        errors = validate_vqvae_config(cfg)
        assert any('vq-vae' in e.lower() or 'vqvae' in e.lower() for e in errors), \
            f"Should detect missing vqvae section, got errors: {errors}"

    def test_invalid_num_embeddings(self):
        """vqvae.num_embeddings <= 0 should be invalid."""
        from medgen.core.validation import validate_vqvae_config

        cfg = make_vqvae_config(**{'vqvae.num_embeddings': 0})

        errors = validate_vqvae_config(cfg)
        assert any('num_embeddings' in e.lower() for e in errors), \
            f"Should detect invalid num_embeddings, got errors: {errors}"

    def test_invalid_embedding_dim(self):
        """vqvae.embedding_dim <= 0 should be invalid."""
        from medgen.core.validation import validate_vqvae_config

        cfg = make_vqvae_config(**{'vqvae.embedding_dim': -1})

        errors = validate_vqvae_config(cfg)
        assert any('embedding_dim' in e.lower() for e in errors), \
            f"Should detect invalid embedding_dim, got errors: {errors}"


# =============================================================================
# TestTrainingConfigValidation - Training-specific conflicts
# =============================================================================


class TestTrainingConfigValidation:
    """Test training configuration conflict detection."""

    def test_compile_with_gradient_checkpointing_conflict(self):
        """use_compile + gradient_checkpointing should raise error."""
        from medgen.core.validation import validate_training_config

        cfg = make_minimal_config(**{
            'training.use_compile': True,
            'training.gradient_checkpointing': True,
        })

        errors = validate_training_config(cfg)
        assert len(errors) > 0, \
            "Should detect compile + gradient_checkpointing conflict"
        assert any('compile' in e.lower() and 'checkpoint' in e.lower() for e in errors), \
            f"Error should mention both compile and checkpointing, got: {errors}"

    def test_compile_alone_valid(self):
        """use_compile=True alone should be valid."""
        from medgen.core.validation import validate_training_config

        cfg = make_minimal_config(**{
            'training.use_compile': True,
            'training.gradient_checkpointing': False,
        })

        errors = validate_training_config(cfg)
        assert len(errors) == 0, f"compile alone should be valid, got errors: {errors}"

    def test_gradient_checkpointing_alone_valid(self):
        """gradient_checkpointing=True alone should be valid."""
        from medgen.core.validation import validate_training_config

        cfg = make_minimal_config(**{
            'training.use_compile': False,
            'training.gradient_checkpointing': True,
        })

        errors = validate_training_config(cfg)
        assert len(errors) == 0, \
            f"gradient_checkpointing alone should be valid, got errors: {errors}"


# =============================================================================
# TestRunValidation - Aggregate validation
# =============================================================================


class TestRunValidation:
    """Test the run_validation aggregation function."""

    def test_run_validation_raises_on_errors(self):
        """run_validation raises ValueError when any validator fails."""
        from medgen.core.validation import run_validation, validate_common_config

        cfg = make_minimal_config(**{'training.epochs': -1})

        with patch('os.path.exists', return_value=True), \
             patch('torch.cuda.is_available', return_value=True):
            with pytest.raises(ValueError) as excinfo:
                run_validation(cfg, [validate_common_config])

        assert 'epochs' in str(excinfo.value).lower()

    def test_run_validation_passes_on_valid(self):
        """run_validation does not raise for valid config."""
        from medgen.core.validation import run_validation, validate_training_config

        cfg = make_minimal_config()

        # Should not raise
        run_validation(cfg, [validate_training_config])

    def test_run_validation_aggregates_multiple_validators(self):
        """run_validation runs all validators and aggregates errors."""
        from medgen.core.validation import (
            run_validation,
            validate_common_config,
            validate_diffusion_config,
        )

        # Config with multiple issues
        cfg = make_minimal_config(**{
            'training.epochs': -1,
            'strategy.name': 'invalid',
        })

        with patch('os.path.exists', return_value=True), \
             patch('torch.cuda.is_available', return_value=True):
            with pytest.raises(ValueError) as excinfo:
                run_validation(cfg, [validate_common_config, validate_diffusion_config])

        error_msg = str(excinfo.value).lower()
        assert 'epochs' in error_msg, "Should mention epochs error"
        assert 'strategy' in error_msg, "Should mention strategy error"


# =============================================================================
# TestModeConfigSync - Mode configs match ModeType enum
# =============================================================================


class TestModeConfigSync:
    """Verify mode configs and ModeType enum are synchronized.

    REGRESSION: Catches errors where a mode config exists but the mode name
    hasn't been added to ModeType enum (e.g., seg_conditioned_input).
    """

    def test_all_mode_configs_in_enum(self):
        """Every configs/mode/*.yaml must have its name in ModeType.

        REGRESSION: Catches the error where seg_conditioned_input.yaml was
        created but seg_conditioned_input wasn't added to ModeType enum.
        """
        import yaml
        from medgen.core.constants import ModeType

        configs_dir = Path(__file__).parent.parent.parent / "configs" / "mode"
        if not configs_dir.exists():
            pytest.skip("configs/mode directory not found")

        valid_modes = {m.value for m in ModeType}
        errors = []

        for config_file in configs_dir.glob("*.yaml"):
            # Skip compression-only modes (they don't use ModeType)
            if "compression" in config_file.name:
                continue

            with open(config_file) as f:
                cfg = yaml.safe_load(f)

            mode_name = cfg.get("name")
            if mode_name and mode_name not in valid_modes:
                # Generate the suggested enum entry
                enum_name = mode_name.upper().replace("-", "_")
                errors.append(
                    f"{config_file.name}: mode '{mode_name}' not in ModeType enum.\n"
                    f"    Add to src/medgen/core/constants.py:\n"
                    f"    {enum_name} = \"{mode_name}\""
                )

        if errors:
            pytest.fail(
                f"Found {len(errors)} mode configs missing from ModeType enum:\n\n"
                + "\n\n".join(errors)
            )

    def test_all_enum_modes_have_configs(self):
        """Every ModeType value should have a corresponding config file.

        This test is informational - not all enum values need config files
        (some may be dynamically configured), but it helps catch orphaned enums.
        """
        from medgen.core.constants import ModeType

        configs_dir = Path(__file__).parent.parent.parent / "configs" / "mode"
        if not configs_dir.exists():
            pytest.skip("configs/mode directory not found")

        existing_configs = {f.stem for f in configs_dir.glob("*.yaml")}

        missing = []
        for mode in ModeType:
            mode_name = mode.value
            # Check for exact match or with _3d/_compression suffix
            has_config = (
                mode_name in existing_configs
                or f"{mode_name}_3d" in existing_configs
                or f"{mode_name}_compression" in existing_configs
                or any(mode_name in c for c in existing_configs)
            )
            if not has_config:
                missing.append(mode_name)

        # This is informational, not a hard failure
        if missing:
            pytest.skip(
                f"ModeType values without config files (may be intentional): {missing}"
            )

    def test_mode_config_names_match_filenames(self):
        """Config 'name' field should match the filename (without .yaml).

        REGRESSION: Catches copy-paste errors where a config file is copied
        but the internal 'name' field isn't updated.
        """
        import yaml

        configs_dir = Path(__file__).parent.parent.parent / "configs" / "mode"
        if not configs_dir.exists():
            pytest.skip("configs/mode directory not found")

        errors = []

        for config_file in configs_dir.glob("*.yaml"):
            with open(config_file) as f:
                cfg = yaml.safe_load(f)

            mode_name = cfg.get("name")
            expected_name = config_file.stem  # filename without .yaml

            # Allow _3d suffix in filename but not in name
            if expected_name.endswith("_3d"):
                expected_base = expected_name[:-3]  # Remove _3d
                if mode_name and mode_name not in (expected_name, expected_base):
                    errors.append(
                        f"{config_file.name}: name='{mode_name}' doesn't match "
                        f"filename (expected '{expected_name}' or '{expected_base}')"
                    )
            elif mode_name and mode_name != expected_name:
                errors.append(
                    f"{config_file.name}: name='{mode_name}' doesn't match "
                    f"filename (expected '{expected_name}')"
                )

        if errors:
            pytest.fail(
                f"Found {len(errors)} mode configs with mismatched names:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )
