"""Benchmark-specific fixtures for pytest-benchmark tests."""
import pytest
import torch


@pytest.fixture
def benchmark_images_2d():
    """Standard 2D images for benchmarking.

    Returns 8 grayscale images of 256x256 pixels in [0, 1] range.
    """
    return torch.rand(8, 1, 256, 256)


@pytest.fixture
def benchmark_volumes_3d():
    """Standard 3D volumes for benchmarking.

    Returns 2 volumes with shape [C=1, D=32, H=256, W=256] in [0, 1] range.
    """
    return torch.rand(2, 1, 32, 256, 256)


@pytest.fixture
def benchmark_features():
    """Feature vectors for generation metrics benchmarking.

    Returns 100 vectors of dimension 2048 (ResNet50 feature size).
    """
    return torch.randn(100, 2048)


@pytest.fixture
def benchmark_clip_features():
    """CLIP feature vectors for CMMD benchmarking.

    Returns 100 vectors of dimension 512 (BiomedCLIP feature size).
    """
    return torch.randn(100, 512)


@pytest.fixture
def benchmark_masks_2d():
    """Binary segmentation masks for benchmarking.

    Returns 8 binary masks of 256x256 pixels.
    """
    return (torch.rand(8, 1, 256, 256) > 0.5).float()
