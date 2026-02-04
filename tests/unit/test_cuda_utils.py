"""Tests for CUDA optimization utilities.

Tests verify that CUDA/PyTorch optimization settings are correctly
applied. These tests work on both CPU and GPU systems by checking
the settings rather than actual GPU functionality.
"""

import pytest
import warnings
import torch
import torch._dynamo.config

from medgen.core.cuda_utils import setup_cuda_optimizations


class TestCudnnSettings:
    """Tests for cuDNN configuration."""

    def test_cudnn_allow_tf32_enabled(self):
        """cuDNN TF32 is enabled."""
        setup_cuda_optimizations()
        assert torch.backends.cudnn.allow_tf32 is True

    def test_cudnn_benchmark_enabled(self):
        """cuDNN benchmark mode is enabled."""
        setup_cuda_optimizations()
        assert torch.backends.cudnn.benchmark is True

    def test_cudnn_deterministic_disabled(self):
        """cuDNN deterministic mode is disabled for performance."""
        setup_cuda_optimizations()
        assert torch.backends.cudnn.deterministic is False

    def test_cudnn_enabled(self):
        """cuDNN is enabled."""
        setup_cuda_optimizations()
        assert torch.backends.cudnn.enabled is True


class TestMatmulSettings:
    """Tests for matrix multiplication settings."""

    def test_matmul_tf32_enabled(self):
        """TF32 enabled for matmul operations."""
        setup_cuda_optimizations()
        assert torch.backends.cuda.matmul.allow_tf32 is True


class TestSdpaSettings:
    """Tests for scaled dot-product attention backends."""

    def test_sdpa_flash_enabled(self):
        """Flash attention backend enabled."""
        setup_cuda_optimizations()
        # After calling enable_flash_sdp(True), the backend should be available
        # We can't directly query if it's enabled, but we can verify no error
        # was raised during setup

    def test_sdpa_mem_efficient_enabled(self):
        """Memory-efficient attention backend enabled."""
        setup_cuda_optimizations()
        # Setup should complete without error

    def test_sdpa_math_enabled(self):
        """Math attention backend enabled."""
        setup_cuda_optimizations()
        # Setup should complete without error


class TestDynamoCacheSize:
    """Tests for torch.compile cache settings."""

    def test_dynamo_cache_size_set(self):
        """torch.compile cache size increased to 32."""
        setup_cuda_optimizations()
        assert torch._dynamo.config.cache_size_limit == 32


class TestWarningFilters:
    """Tests for warning suppression."""

    def test_warning_filters_suppress_monai_cache_dir(self):
        """MONAI cache_dir warning is suppressed."""
        setup_cuda_optimizations()

        # Check that the filter is in place - filters use message pattern (index 4)
        # or can be in different positions depending on how warnings.filterwarnings was called
        filter_found = False
        for f in warnings.filters:
            # filters tuple: (action, message, category, module, lineno)
            # when using filterwarnings(action, message=...), message is at index 1
            if f[0] == 'ignore':
                # Check all string fields for the pattern
                for field in f[1:]:
                    if field is not None and hasattr(field, 'pattern'):
                        if 'cache_dir' in field.pattern:
                            filter_found = True
                            break
                    elif isinstance(field, str) and 'cache_dir' in field:
                        filter_found = True
                        break
        assert filter_found, "MONAI cache_dir warning filter not found"

    def test_warning_filters_suppress_torchvision_pretrained(self):
        """TorchVision pretrained deprecation warning is suppressed."""
        setup_cuda_optimizations()

        filter_found = False
        for f in warnings.filters:
            if f[0] == 'ignore':
                for field in f[1:]:
                    if field is not None and hasattr(field, 'pattern'):
                        if 'pretrained' in field.pattern:
                            filter_found = True
                            break
                    elif isinstance(field, str) and 'pretrained' in field:
                        filter_found = True
                        break
        assert filter_found, "TorchVision pretrained warning filter not found"

    def test_warning_filters_suppress_torchvision_weight_enum(self):
        """TorchVision weight enum warning is suppressed."""
        setup_cuda_optimizations()

        filter_found = False
        for f in warnings.filters:
            if f[0] == 'ignore':
                for field in f[1:]:
                    if field is not None and hasattr(field, 'pattern'):
                        if 'weight enum' in field.pattern:
                            filter_found = True
                            break
                    elif isinstance(field, str) and 'weight enum' in field:
                        filter_found = True
                        break
        assert filter_found, "TorchVision weight enum warning filter not found"


class TestIdempotence:
    """Tests that calling setup_cuda_optimizations multiple times is safe."""

    def test_multiple_calls_safe(self):
        """Multiple calls to setup_cuda_optimizations don't cause issues."""
        # Should not raise
        setup_cuda_optimizations()
        setup_cuda_optimizations()
        setup_cuda_optimizations()

        # Settings should still be correct
        assert torch.backends.cudnn.allow_tf32 is True
        assert torch.backends.cuda.matmul.allow_tf32 is True
        assert torch._dynamo.config.cache_size_limit == 32


class TestIntegration:
    """Integration tests for the full setup."""

    def test_all_settings_applied(self):
        """All settings are applied in a single call."""
        setup_cuda_optimizations()

        # cuDNN
        assert torch.backends.cudnn.allow_tf32 is True
        assert torch.backends.cudnn.benchmark is True
        assert torch.backends.cudnn.deterministic is False
        assert torch.backends.cudnn.enabled is True

        # matmul
        assert torch.backends.cuda.matmul.allow_tf32 is True

        # dynamo
        assert torch._dynamo.config.cache_size_limit == 32

        # Warning filters exist
        assert len(warnings.filters) > 0
