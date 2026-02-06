"""Tests for ModeFactory."""
import pytest
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf

from medgen.core.mode_factory import ModeFactory, ModeConfig, ModeCategory
from medgen.core.constants import ModeType


class TestNormalizeMode:
    """Tests for ModeFactory.normalize_mode()."""

    def test_enum_passthrough(self):
        """ModeType enum should pass through unchanged."""
        assert ModeFactory.normalize_mode(ModeType.BRAVO) == ModeType.BRAVO
        assert ModeFactory.normalize_mode(ModeType.SEG) == ModeType.SEG
        assert ModeFactory.normalize_mode(ModeType.DUAL) == ModeType.DUAL

    def test_string_to_enum_lowercase(self):
        """Lowercase string should convert to ModeType."""
        assert ModeFactory.normalize_mode('bravo') == ModeType.BRAVO
        assert ModeFactory.normalize_mode('seg') == ModeType.SEG
        assert ModeFactory.normalize_mode('dual') == ModeType.DUAL
        assert ModeFactory.normalize_mode('multi') == ModeType.MULTI
        assert ModeFactory.normalize_mode('seg_conditioned') == ModeType.SEG_CONDITIONED

    def test_string_to_enum_uppercase(self):
        """Uppercase string should also convert (case-insensitive)."""
        assert ModeFactory.normalize_mode('BRAVO') == ModeType.BRAVO
        assert ModeFactory.normalize_mode('SEG') == ModeType.SEG
        assert ModeFactory.normalize_mode('Dual') == ModeType.DUAL

    def test_invalid_string_raises_value_error(self):
        """Unknown mode string should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown mode"):
            ModeFactory.normalize_mode('invalid_mode')
        with pytest.raises(ValueError, match="Unknown mode"):
            ModeFactory.normalize_mode('foo')

    def test_invalid_type_raises_type_error(self):
        """Non-string, non-ModeType input should raise TypeError."""
        with pytest.raises(TypeError):
            ModeFactory.normalize_mode(123)
        with pytest.raises(TypeError):
            ModeFactory.normalize_mode(None)
        with pytest.raises(TypeError):
            ModeFactory.normalize_mode(['bravo'])


class TestGetModeConfig:
    """Tests for ModeFactory.get_mode_config()."""

    def test_bravo_config(self):
        """BRAVO mode should be SINGLE category with no image_keys."""
        cfg = OmegaConf.create({
            'mode': {'name': 'bravo'},
            'model': {'spatial_dims': 2},
        })
        config = ModeFactory.get_mode_config(cfg)
        assert config.mode == ModeType.BRAVO
        assert config.category == ModeCategory.SINGLE
        assert config.spatial_dims == 2
        assert config.image_keys == []
        assert config.conditioning is None
        assert config.use_latent is False

    def test_seg_config(self):
        """SEG mode should be SINGLE category."""
        cfg = OmegaConf.create({
            'mode': {'name': 'seg'},
            'model': {'spatial_dims': 2},
        })
        config = ModeFactory.get_mode_config(cfg)
        assert config.mode == ModeType.SEG
        assert config.category == ModeCategory.SINGLE
        assert config.conditioning is None

    def test_dual_config_with_defaults(self):
        """DUAL mode should use default image_keys and conditioning."""
        cfg = OmegaConf.create({
            'mode': {'name': 'dual'},
            'model': {'spatial_dims': 2},
        })
        config = ModeFactory.get_mode_config(cfg)
        assert config.mode == ModeType.DUAL
        assert config.category == ModeCategory.DUAL
        assert config.image_keys == ['t1_pre', 't1_gd']
        assert config.conditioning == 'seg'

    def test_dual_config_custom_keys(self):
        """DUAL mode should use custom image_keys if specified."""
        cfg = OmegaConf.create({
            'mode': {'name': 'dual', 'image_keys': ['flair', 'bravo']},
            'model': {'spatial_dims': 2},
        })
        config = ModeFactory.get_mode_config(cfg)
        assert config.image_keys == ['flair', 'bravo']

    def test_multi_config_with_defaults(self):
        """MULTI mode should use default image_keys."""
        cfg = OmegaConf.create({
            'mode': {'name': 'multi'},
            'model': {'spatial_dims': 2},
        })
        config = ModeFactory.get_mode_config(cfg)
        assert config.mode == ModeType.MULTI
        assert config.category == ModeCategory.MULTI
        assert config.image_keys == ['bravo', 'flair', 't1_pre', 't1_gd']
        assert config.conditioning == 'seg'

    def test_multi_config_custom_keys(self):
        """MULTI mode should use custom image_keys if specified."""
        cfg = OmegaConf.create({
            'mode': {'name': 'multi', 'image_keys': ['bravo', 'flair']},
            'model': {'spatial_dims': 2},
        })
        config = ModeFactory.get_mode_config(cfg)
        assert config.image_keys == ['bravo', 'flair']

    def test_seg_conditioned_config(self):
        """SEG_CONDITIONED mode should be SEG_CONDITIONED category."""
        cfg = OmegaConf.create({
            'mode': {'name': 'seg_conditioned'},
            'model': {'spatial_dims': 2},
        })
        config = ModeFactory.get_mode_config(cfg)
        assert config.mode == ModeType.SEG_CONDITIONED
        assert config.category == ModeCategory.SEG_CONDITIONED

    def test_seg_conditioned_input_config(self):
        """SEG_CONDITIONED_INPUT mode should be SEG_CONDITIONED category."""
        cfg = OmegaConf.create({
            'mode': {'name': 'seg_conditioned_input'},
            'model': {'spatial_dims': 2},
        })
        config = ModeFactory.get_mode_config(cfg)
        assert config.mode == ModeType.SEG_CONDITIONED_INPUT
        assert config.category == ModeCategory.SEG_CONDITIONED

    def test_bravo_seg_cond_config(self):
        """BRAVO_SEG_COND mode should be SINGLE category."""
        cfg = OmegaConf.create({
            'mode': {'name': 'bravo_seg_cond'},
            'model': {'spatial_dims': 2},
        })
        config = ModeFactory.get_mode_config(cfg)
        assert config.mode == ModeType.BRAVO_SEG_COND
        assert config.category == ModeCategory.SINGLE

    def test_3d_spatial_dims(self):
        """Config with spatial_dims=3 should be extracted correctly."""
        cfg = OmegaConf.create({
            'mode': {'name': 'seg'},
            'model': {'spatial_dims': 3},
        })
        config = ModeFactory.get_mode_config(cfg)
        assert config.spatial_dims == 3

    def test_latent_enabled(self):
        """Latent enabled config should set use_latent=True."""
        cfg = OmegaConf.create({
            'mode': {'name': 'bravo'},
            'model': {'spatial_dims': 2},
            'latent': {'enabled': True},
        })
        config = ModeFactory.get_mode_config(cfg)
        assert config.use_latent is True

    def test_default_spatial_dims(self):
        """Missing spatial_dims should default to 2."""
        cfg = OmegaConf.create({
            'mode': {'name': 'bravo'},
            'model': {},
        })
        config = ModeFactory.get_mode_config(cfg)
        assert config.spatial_dims == 2


class TestCreateDataloaders:
    """Tests for ModeFactory.create_*_dataloader() methods."""

    @patch('medgen.data.loaders.unified.create_dataloader')
    def test_create_train_dataloader_calls_unified(self, mock_create):
        """create_train_dataloader should delegate to unified.create_dataloader."""
        mock_create.return_value = (MagicMock(), MagicMock())
        cfg = OmegaConf.create({
            'mode': {'name': 'bravo'},
            'model': {'spatial_dims': 2},
        })

        ModeFactory.create_train_dataloader(cfg)

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs['task'] == 'diffusion'
        assert call_kwargs['mode'] == 'bravo'
        assert call_kwargs['split'] == 'train'
        assert call_kwargs['spatial_dims'] == 2

    @patch('medgen.data.loaders.unified.create_dataloader')
    def test_create_train_dataloader_passes_distributed_params(self, mock_create):
        """Distributed training parameters should be passed through."""
        mock_create.return_value = (MagicMock(), MagicMock())
        cfg = OmegaConf.create({
            'mode': {'name': 'bravo'},
            'model': {'spatial_dims': 2},
        })

        ModeFactory.create_train_dataloader(
            cfg, use_distributed=True, rank=2, world_size=4
        )

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs['use_distributed'] is True
        assert call_kwargs['rank'] == 2
        assert call_kwargs['world_size'] == 4

    @patch('medgen.data.loaders.unified.create_dataloader')
    def test_create_val_dataloader_calls_unified(self, mock_create):
        """create_val_dataloader should delegate to unified.create_dataloader."""
        mock_create.return_value = (MagicMock(), MagicMock())
        cfg = OmegaConf.create({
            'mode': {'name': 'dual'},
            'model': {'spatial_dims': 2},
        })

        ModeFactory.create_val_dataloader(cfg)

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs['task'] == 'diffusion'
        assert call_kwargs['mode'] == 'dual'
        assert call_kwargs['split'] == 'val'

    @patch('medgen.data.loaders.unified.create_dataloader')
    def test_create_val_dataloader_returns_none_on_error(self, mock_create):
        """create_val_dataloader should return None if no val data."""
        mock_create.side_effect = ValueError("No validation data found")
        cfg = OmegaConf.create({
            'mode': {'name': 'bravo'},
            'model': {'spatial_dims': 2},
        })

        result = ModeFactory.create_val_dataloader(cfg)

        assert result is None

    @patch('medgen.data.loaders.unified.create_dataloader')
    def test_create_test_dataloader_calls_unified(self, mock_create):
        """create_test_dataloader should delegate to unified.create_dataloader."""
        mock_create.return_value = (MagicMock(), MagicMock())
        cfg = OmegaConf.create({
            'mode': {'name': 'seg'},
            'model': {'spatial_dims': 2},
        })

        ModeFactory.create_test_dataloader(cfg)

        mock_create.assert_called_once()
        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs['task'] == 'diffusion'
        assert call_kwargs['mode'] == 'seg'
        assert call_kwargs['split'] == 'test'

    @patch('medgen.data.loaders.unified.create_dataloader')
    def test_create_pixel_loader_for_latent_cache(self, mock_create):
        """create_pixel_loader_for_latent_cache should disable augmentation."""
        mock_create.return_value = (MagicMock(), MagicMock())
        cfg = OmegaConf.create({
            'mode': {'name': 'bravo'},
            'model': {'spatial_dims': 2},
        })
        mode_config = ModeFactory.get_mode_config(cfg)

        ModeFactory.create_pixel_loader_for_latent_cache(cfg, mode_config)

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs['augment'] is False


class TestPerModalityValLoaders:
    """Tests for ModeFactory.create_per_modality_val_loaders()."""

    def test_returns_empty_dict_for_non_multi_mode(self):
        """Non-MULTI modes should return empty dict."""
        cfg = OmegaConf.create({
            'mode': {'name': 'bravo'},
            'model': {'spatial_dims': 2},
        })
        mode_config = ModeFactory.get_mode_config(cfg)
        loaders = ModeFactory.create_per_modality_val_loaders(cfg, mode_config)
        assert loaders == {}

    def test_returns_empty_dict_for_dual_mode(self):
        """DUAL mode should return empty dict (not MULTI)."""
        cfg = OmegaConf.create({
            'mode': {'name': 'dual'},
            'model': {'spatial_dims': 2},
        })
        mode_config = ModeFactory.get_mode_config(cfg)
        loaders = ModeFactory.create_per_modality_val_loaders(cfg, mode_config)
        assert loaders == {}

    @patch('medgen.data.loaders.builder_2d.create_single_modality_diffusion_val_loader')
    def test_creates_loaders_for_multi_mode(self, mock_create):
        """MULTI mode should create per-modality loaders."""
        mock_loader = MagicMock()
        mock_create.return_value = mock_loader

        cfg = OmegaConf.create({
            'mode': {'name': 'multi', 'image_keys': ['bravo', 'flair']},
            'model': {'spatial_dims': 2},
        })
        mode_config = ModeFactory.get_mode_config(cfg)
        loaders = ModeFactory.create_per_modality_val_loaders(cfg, mode_config)

        assert 'bravo' in loaders
        assert 'flair' in loaders
        assert loaders['bravo'] == mock_loader
        assert mock_create.call_count == 2


class TestGetImageTypeForMode:
    """Tests for ModeFactory.get_image_type_for_mode()."""

    def test_seg_mode(self):
        """SEG mode should return 'seg'."""
        assert ModeFactory.get_image_type_for_mode(ModeType.SEG) == 'seg'

    def test_seg_conditioned_mode(self):
        """SEG_CONDITIONED modes should return 'seg'."""
        assert ModeFactory.get_image_type_for_mode(ModeType.SEG_CONDITIONED) == 'seg'
        assert ModeFactory.get_image_type_for_mode(ModeType.SEG_CONDITIONED_INPUT) == 'seg'

    def test_bravo_mode(self):
        """BRAVO mode should return 'bravo'."""
        assert ModeFactory.get_image_type_for_mode(ModeType.BRAVO) == 'bravo'

    def test_bravo_seg_cond_mode(self):
        """BRAVO_SEG_COND mode should return 'bravo'."""
        assert ModeFactory.get_image_type_for_mode(ModeType.BRAVO_SEG_COND) == 'bravo'


class TestModeCategory:
    """Tests for ModeCategory enum."""

    def test_all_modes_have_category(self):
        """Every ModeType should have a category mapping."""
        for mode in [
            ModeType.SEG,
            ModeType.BRAVO,
            ModeType.BRAVO_SEG_COND,
            ModeType.DUAL,
            ModeType.MULTI,
            ModeType.SEG_CONDITIONED,
            ModeType.SEG_CONDITIONED_INPUT,
        ]:
            assert mode in ModeFactory.MODE_CATEGORIES, f"{mode} missing category"
