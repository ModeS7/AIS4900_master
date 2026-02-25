"""Tests for compression model encode() return type contracts.

Verifies:
- VAE (AutoencoderKL): encode() returns (z_mu, z_logvar) tuple
- VQ-VAE (VQVAE): encode() returns a quantized tensor
- _infer_vae_config_from_state_dict: roundtrip from state_dict → config → model
- _detect_scale_factor_from_dict: correct scale for VAE vs VQ-VAE

Catches silent API breakage if MONAI changes encode() signatures.
"""

import pytest
import torch
from monai.networks.nets import AutoencoderKL, VQVAE

from medgen.data.loaders.compression_detection import (
    _detect_scale_factor_from_dict,
    _infer_vae_config_from_state_dict,
)


# ---------------------------------------------------------------------------
# VAE encode contract
# ---------------------------------------------------------------------------


class TestVAEEncodeContract:
    """AutoencoderKL.encode() must return (z_mu, z_logvar) tuple."""

    @staticmethod
    def _make_tiny_vae(spatial_dims: int = 2) -> AutoencoderKL:
        return AutoencoderKL(
            spatial_dims=spatial_dims,
            in_channels=1,
            out_channels=1,
            channels=(8, 16),
            latent_channels=4,
            num_res_blocks=1,
            attention_levels=(False, False),
            norm_num_groups=8,
            with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False,
        )

    def test_encode_returns_tuple_2d(self):
        vae = self._make_tiny_vae(spatial_dims=2)
        x = torch.randn(1, 1, 32, 32)
        result = vae.encode(x)

        assert isinstance(result, tuple), (
            f"VAE.encode() should return tuple, got {type(result)}"
        )
        assert len(result) == 2, f"Expected (z_mu, z_logvar), got len={len(result)}"
        z_mu, z_logvar = result
        assert z_mu.shape == z_logvar.shape
        assert z_mu.shape[1] == 4  # latent_channels

    def test_encode_returns_tuple_3d(self):
        vae = self._make_tiny_vae(spatial_dims=3)
        x = torch.randn(1, 1, 16, 16, 16)
        result = vae.encode(x)

        assert isinstance(result, tuple)
        assert len(result) == 2
        z_mu, z_logvar = result
        assert z_mu.ndim == 5  # [B, C, D, H, W]
        assert z_mu.shape[1] == 4

    def test_decode_accepts_mu(self):
        """decode(z_mu) should produce pixel-space output."""
        vae = self._make_tiny_vae()
        x = torch.randn(1, 1, 32, 32)
        z_mu, _ = vae.encode(x)
        recon = vae.decode(z_mu)
        assert recon.shape == x.shape


# ---------------------------------------------------------------------------
# VQ-VAE encode contract
# ---------------------------------------------------------------------------


class TestVQVAEEncodeContract:
    """VQVAE.encode() must return a quantized tensor."""

    @staticmethod
    def _make_tiny_vqvae() -> VQVAE:
        return VQVAE(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(8, 16),
            num_res_layers=1,
            num_res_channels=(8, 16),
            # downsample: (kernel, stride, padding, dilation)
            downsample_parameters=((4, 2, 1, 1), (4, 2, 1, 1)),
            # upsample: (stride, kernel, dilation, padding, output_padding)
            upsample_parameters=((2, 4, 1, 1, 0), (2, 4, 1, 1, 0)),
            num_embeddings=32,
            embedding_dim=4,
        )

    def test_encode_returns_tensor(self):
        vqvae = self._make_tiny_vqvae()
        x = torch.randn(1, 1, 32, 32)
        result = vqvae.encode(x)

        assert isinstance(result, torch.Tensor), (
            f"VQVAE.encode() should return Tensor, got {type(result)}"
        )
        assert result.shape[1] == 4  # embedding_dim

    def test_decode_stage_2_outputs(self):
        """decode_stage_2_outputs(quantized) should produce pixel output."""
        vqvae = self._make_tiny_vqvae()
        x = torch.randn(1, 1, 32, 32)
        quantized = vqvae.encode(x)
        recon = vqvae.decode_stage_2_outputs(quantized)
        assert recon.ndim == 4  # [B, C, H, W]
        assert recon.shape[0] == 1  # batch
        assert recon.shape[1] == 1  # out_channels


# ---------------------------------------------------------------------------
# _infer_vae_config_from_state_dict roundtrip
# ---------------------------------------------------------------------------


class TestInferVAEConfig:
    """Create VAE → get state_dict → infer config → verify matches."""

    @staticmethod
    def _make_vae(
        channels: tuple,
        latent_channels: int,
        attention: tuple,
        spatial_dims: int = 2,
        with_mid: bool = True,
    ) -> AutoencoderKL:
        return AutoencoderKL(
            spatial_dims=spatial_dims,
            in_channels=1,
            out_channels=1,
            channels=channels,
            latent_channels=latent_channels,
            num_res_blocks=2,
            attention_levels=attention,
            norm_num_groups=8,
            with_encoder_nonlocal_attn=with_mid,
            with_decoder_nonlocal_attn=with_mid,
        )

    def test_roundtrip_2level_no_attn(self):
        """2-level VAE, no attention, no mid block."""
        vae = self._make_vae(
            channels=(8, 16), latent_channels=4,
            attention=(False, False), with_mid=False,
        )
        sd = vae.state_dict()
        cfg = _infer_vae_config_from_state_dict(sd)

        assert cfg["channels"] == [8, 16]
        assert cfg["latent_channels"] == 4
        assert cfg["attention_levels"] == [False, False]
        assert cfg["in_channels"] == 1
        assert cfg["with_encoder_nonlocal_attn"] is False

    def test_roundtrip_3level_with_attn(self):
        """3-level VAE with attention at last level + mid block."""
        vae = self._make_vae(
            channels=(8, 16, 32), latent_channels=4,
            attention=(False, False, True), with_mid=True,
        )
        sd = vae.state_dict()
        cfg = _infer_vae_config_from_state_dict(sd)

        assert cfg["channels"] == [8, 16, 32]
        assert cfg["latent_channels"] == 4
        assert cfg["attention_levels"] == [False, False, True]
        assert cfg["with_encoder_nonlocal_attn"] is True

    def test_roundtrip_3d(self):
        """3D VAE infers spatial_dims=3."""
        vae = self._make_vae(
            channels=(8, 16), latent_channels=4,
            attention=(False, False), spatial_dims=3, with_mid=False,
        )
        sd = vae.state_dict()
        cfg = _infer_vae_config_from_state_dict(sd)

        assert cfg.get("spatial_dims") == 3

    def test_inferred_config_can_create_model(self):
        """Inferred config should produce a model that loads the same state_dict."""
        # Use channels divisible by 32 since _infer_vae_config returns norm_num_groups=32
        vae = AutoencoderKL(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            channels=(32, 64),
            latent_channels=4,
            num_res_blocks=2,
            attention_levels=(False, False),
            norm_num_groups=32,
            with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False,
        )
        sd = vae.state_dict()
        cfg = _infer_vae_config_from_state_dict(sd)

        # Create a new model from inferred config
        new_vae = AutoencoderKL(
            spatial_dims=cfg.get("spatial_dims", 2),
            in_channels=cfg["in_channels"],
            out_channels=cfg["out_channels"],
            channels=tuple(cfg["channels"]),
            latent_channels=cfg["latent_channels"],
            num_res_blocks=cfg["num_res_blocks"],
            attention_levels=tuple(cfg["attention_levels"]),
            norm_num_groups=cfg["norm_num_groups"],
            with_encoder_nonlocal_attn=cfg["with_encoder_nonlocal_attn"],
            with_decoder_nonlocal_attn=cfg["with_decoder_nonlocal_attn"],
        )
        # Must load without errors (strict=True is default)
        new_vae.load_state_dict(sd)


# ---------------------------------------------------------------------------
# _detect_scale_factor_from_dict
# ---------------------------------------------------------------------------


class TestDetectScaleFactor:
    """Regression: VAE has len(channels)-1 downsamples, VQ-VAE has len(channels)."""

    def test_vae_3_levels_is_4x(self):
        """3-level VAE: 2 downsamples → 4x, NOT 8x."""
        ckpt = {"config": {"channels": [32, 64, 128]}}
        assert _detect_scale_factor_from_dict(ckpt, "vae") == 4

    def test_vae_4_levels_is_8x(self):
        """4-level VAE: 3 downsamples → 8x."""
        ckpt = {"config": {"channels": [32, 64, 128, 128]}}
        assert _detect_scale_factor_from_dict(ckpt, "vae") == 8

    def test_vae_2_levels_is_2x(self):
        """2-level VAE: 1 downsample → 2x."""
        ckpt = {"config": {"channels": [32, 64]}}
        assert _detect_scale_factor_from_dict(ckpt, "vae") == 2

    def test_vqvae_3_levels_is_8x(self):
        """3-level VQ-VAE: 3 strided convs → 8x."""
        ckpt = {"config": {"channels": [64, 128, 256]}}
        assert _detect_scale_factor_from_dict(ckpt, "vqvae") == 8

    def test_vqvae_2_levels_is_4x(self):
        """2-level VQ-VAE: 2 strided convs → 4x."""
        ckpt = {"config": {"channels": [64, 128]}}
        assert _detect_scale_factor_from_dict(ckpt, "vqvae") == 4
