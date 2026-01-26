"""Tests for feature extractors: ResNet50Features, BiomedCLIPFeatures."""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock

from medgen.metrics.feature_extractors import ResNet50Features, BiomedCLIPFeatures
from medgen.metrics.generation import extract_features_3d


class TestResNet50Features:
    """Test ResNet50Features class."""

    def test_resnet50_init_no_model_loaded(self):
        """Model not loaded on init (lazy loading)."""
        extractor = ResNet50Features(device=torch.device('cpu'), compile_model=False)
        # Model should be None until first use
        assert extractor._model is None

    def test_resnet50_compile_model_parameter(self):
        """compile_model parameter stored correctly."""
        extractor = ResNet50Features(device=torch.device('cpu'), compile_model=False)
        assert extractor.compile_model is False

        extractor2 = ResNet50Features(device=torch.device('cpu'), compile_model=True)
        assert extractor2.compile_model is True

    @pytest.mark.timeout(60)
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_grayscale_input(self):
        """1-channel repeated to 3 channels."""
        extractor = ResNet50Features(device=torch.device('cpu'), compile_model=False)
        images = torch.rand(2, 1, 224, 224)
        features = extractor.extract_features(images)
        assert features.shape == (2, 2048)

    @pytest.mark.timeout(60)
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_resize_to_224(self):
        """Input resized to 224x224."""
        extractor = ResNet50Features(device=torch.device('cpu'), compile_model=False)
        images = torch.rand(2, 3, 64, 64)
        features = extractor.extract_features(images)
        assert features.shape == (2, 2048)

    @pytest.mark.timeout(60)
    @pytest.mark.slow
    @pytest.mark.gpu
    def test_3d_input_takes_middle_slice(self):
        """5D input: middle slice extracted."""
        extractor = ResNet50Features(device=torch.device('cpu'), compile_model=False)
        volumes = torch.rand(2, 1, 16, 64, 64)
        features = extractor.extract_features(volumes)
        assert features.shape == (2, 2048)


class TestBiomedCLIPFeatures:
    """Test BiomedCLIPFeatures class."""

    def test_biomedclip_init_no_model_loaded(self):
        """Model not loaded on init (lazy loading)."""
        extractor = BiomedCLIPFeatures(device=torch.device('cpu'), compile_model=False)
        assert extractor._model is None

    def test_biomedclip_compile_model_parameter(self):
        """compile_model parameter stored correctly."""
        extractor = BiomedCLIPFeatures(device=torch.device('cpu'), compile_model=False)
        assert extractor.compile_model is False


class TestExtractFeatures3D:
    """Test extract_features_3d function."""

    def test_output_shape(self):
        """Returns [B*D, feat_dim]."""
        # Mock extractor with extract_features method
        def mock_extract(x, use_amp=True):
            return torch.randn(x.shape[0], 2048)

        mock_extractor = Mock()
        mock_extractor.extract_features = Mock(side_effect=mock_extract)
        mock_extractor.eval = Mock(return_value=mock_extractor)
        mock_extractor.to = Mock(return_value=mock_extractor)
        mock_extractor.train = Mock(return_value=mock_extractor)

        volumes = torch.rand(2, 1, 16, 64, 64)  # B=2, D=16

        features = extract_features_3d(volumes, mock_extractor, chunk_size=8)

        # Should return [B*D, feat_dim] = [32, 2048]
        assert features.shape == (32, 2048)

    def test_slicewise_extraction(self):
        """Features extracted per slice."""
        call_count = [0]

        def mock_extract(x, use_amp=True):
            call_count[0] += 1
            return torch.randn(x.shape[0], 2048)

        mock_extractor = Mock()
        mock_extractor.extract_features = Mock(side_effect=mock_extract)
        mock_extractor.eval = Mock(return_value=mock_extractor)
        mock_extractor.to = Mock(return_value=mock_extractor)
        mock_extractor.train = Mock(return_value=mock_extractor)

        volumes = torch.rand(2, 1, 16, 64, 64)

        features = extract_features_3d(volumes, mock_extractor, chunk_size=16)

        # With chunk_size=16 and B*D=32, should call twice
        assert call_count[0] == 2

    def test_chunk_size_parameter(self):
        """Chunked processing for memory."""
        call_count = [0]

        def mock_extract(x, use_amp=True):
            call_count[0] += 1
            return torch.randn(x.shape[0], 2048)

        mock_extractor = Mock()
        mock_extractor.extract_features = Mock(side_effect=mock_extract)
        mock_extractor.eval = Mock(return_value=mock_extractor)
        mock_extractor.to = Mock(return_value=mock_extractor)
        mock_extractor.train = Mock(return_value=mock_extractor)

        volumes = torch.rand(2, 1, 16, 64, 64)  # 32 slices total

        # With chunk_size=8, should make 4 calls
        features = extract_features_3d(volumes, mock_extractor, chunk_size=8)

        assert call_count[0] == 4
