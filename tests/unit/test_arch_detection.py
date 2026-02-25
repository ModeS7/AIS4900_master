"""Unit tests for architecture detection helpers in diffusion/loading.py.

Pure unit tests using synthetic state dicts with fake tensors — no real
model creation needed. Catches wrong-key and wrong-shape regressions that
have historically caused silent loading failures on the cluster.
"""

import pytest
import torch

from medgen.diffusion.loading import (
    _detect_model_arch_from_state_dict,
    _infer_channels_from_state_dict,
    _infer_hdit_level_depths,
    _infer_spatial_from_pos_embeds,
    _infer_variant_from_state_dict,
)


# ---------------------------------------------------------------------------
# _detect_model_arch_from_state_dict
# ---------------------------------------------------------------------------


class TestDetectModelArch:
    """Detect architecture type from state dict key patterns."""

    def test_hdit_detected(self):
        sd = {
            "encoder_levels.0.0.attn.qkv.weight": torch.empty(1),
            "decoder_levels.0.0.attn.qkv.weight": torch.empty(1),
            "mid_blocks.0.attn.qkv.weight": torch.empty(1),
        }
        assert _detect_model_arch_from_state_dict(sd) == "hdit"

    def test_uvit_detected(self):
        sd = {
            "in_blocks.0.mlp.fc1.weight": torch.empty(1),
            "out_blocks.0.mlp.fc1.weight": torch.empty(1),
            "mid_block.mlp.fc1.weight": torch.empty(1),
        }
        assert _detect_model_arch_from_state_dict(sd) == "uvit"

    def test_dit_detected(self):
        sd = {
            "blocks.0.attn.qkv.weight": torch.empty(1),
            "blocks.1.attn.qkv.weight": torch.empty(1),
            "final_layer.adaLN_modulation.1.weight": torch.empty(1),
        }
        assert _detect_model_arch_from_state_dict(sd) == "dit"

    def test_unet_fallback(self):
        sd = {
            "time_embed.0.weight": torch.empty(1),
            "down_blocks.0.resnets.0.conv1.weight": torch.empty(1),
        }
        assert _detect_model_arch_from_state_dict(sd) == "unet"

    def test_hdit_has_priority_over_dit_keys(self):
        """HDiT also contains blocks-like keys; encoder_levels must win."""
        sd = {
            "encoder_levels.0.0.attn.qkv.weight": torch.empty(1),
            "blocks.0.attn.qkv.weight": torch.empty(1),
            "final_layer.adaLN_modulation.1.weight": torch.empty(1),
        }
        assert _detect_model_arch_from_state_dict(sd) == "hdit"

    def test_empty_state_dict_returns_unet(self):
        assert _detect_model_arch_from_state_dict({}) == "unet"


# ---------------------------------------------------------------------------
# _infer_variant_from_state_dict
# ---------------------------------------------------------------------------


class TestInferVariant:
    """Infer S/B/L/XL from hidden_size in weight shapes."""

    @pytest.mark.parametrize(
        "hidden_size, expected",
        [(384, "S"), (768, "B"), (1024, "L"), (1152, "XL")],
    )
    def test_dit_variants(self, hidden_size, expected):
        # blocks.0.mlp.fc1.weight shape: [4*hidden, hidden]
        sd = {"blocks.0.mlp.fc1.weight": torch.empty(4 * hidden_size, hidden_size)}
        assert _infer_variant_from_state_dict(sd, "dit") == expected

    @pytest.mark.parametrize(
        "hidden_size, expected",
        [(384, "S"), (768, "B"), (1024, "L"), (1152, "XL")],
    )
    def test_hdit_variants(self, hidden_size, expected):
        sd = {
            "encoder_levels.0.0.mlp.fc1.weight": torch.empty(
                4 * hidden_size, hidden_size
            )
        }
        assert _infer_variant_from_state_dict(sd, "hdit") == expected

    @pytest.mark.parametrize(
        "hidden_size, expected",
        [(512, "S"), (768, "M"), (1024, "L")],
    )
    def test_uvit_variants(self, hidden_size, expected):
        sd = {"in_blocks.0.mlp.fc1.weight": torch.empty(4 * hidden_size, hidden_size)}
        assert _infer_variant_from_state_dict(sd, "uvit") == expected

    def test_unknown_hidden_size_defaults_to_s(self):
        sd = {"blocks.0.mlp.fc1.weight": torch.empty(999, 123)}
        assert _infer_variant_from_state_dict(sd, "dit") == "S"

    def test_missing_key_defaults_to_s(self):
        assert _infer_variant_from_state_dict({}, "dit") == "S"


# ---------------------------------------------------------------------------
# _infer_hdit_level_depths
# ---------------------------------------------------------------------------


class TestInferHditLevelDepths:
    """Infer [enc..., mid, dec...] from synthetic encoder/mid/decoder keys."""

    def test_standard_3level(self):
        """Standard [2, 4, 2] HDiT: 1 enc level, 1 mid, 1 dec level."""
        sd = {}
        # Encoder level 0: 2 blocks
        for b in range(2):
            sd[f"encoder_levels.0.{b}.attn.qkv.weight"] = torch.empty(1)
        # Mid blocks: 4 blocks
        for b in range(4):
            sd[f"mid_blocks.{b}.attn.qkv.weight"] = torch.empty(1)
        # Decoder level 0: 2 blocks
        for b in range(2):
            sd[f"decoder_levels.0.{b}.attn.qkv.weight"] = torch.empty(1)

        assert _infer_hdit_level_depths(sd) == [2, 4, 2]

    def test_5level(self):
        """[1, 2, 6, 2, 1] HDiT: 2 enc, 1 mid, 2 dec."""
        sd = {}
        # Encoder level 0: 1 block
        sd["encoder_levels.0.0.x"] = torch.empty(1)
        # Encoder level 1: 2 blocks
        for b in range(2):
            sd[f"encoder_levels.1.{b}.x"] = torch.empty(1)
        # Mid: 6 blocks
        for b in range(6):
            sd[f"mid_blocks.{b}.x"] = torch.empty(1)
        # Decoder level 0: 2 blocks
        for b in range(2):
            sd[f"decoder_levels.0.{b}.x"] = torch.empty(1)
        # Decoder level 1: 1 block
        sd["decoder_levels.1.0.x"] = torch.empty(1)

        assert _infer_hdit_level_depths(sd) == [1, 2, 6, 2, 1]

    def test_empty_returns_zero_mid(self):
        # With no keys, mid_blocks is empty → depths = [0] (mid block count = 0)
        # The default [2,4,6,4,2] is only returned when depths list itself is empty,
        # but mid_blocks.append always runs, so depths is never truly empty.
        assert _infer_hdit_level_depths({}) == [0]


# ---------------------------------------------------------------------------
# _infer_spatial_from_pos_embeds
# ---------------------------------------------------------------------------


class TestInferSpatialFromPosEmbeds:
    """Infer (input_size, depth_size) from positional embedding shape."""

    def test_dit_2d_square(self):
        """DiT 2D: 16 tokens at patch_size=2 → input_size=8."""
        sd = {"pos_embed": torch.empty(1, 16, 384)}  # 4x4 tokens
        size, depth = _infer_spatial_from_pos_embeds(
            sd, spatial_dims=2, patch_size=2,
            default_input_size=999, default_depth_size=None, model_type="dit",
        )
        assert size == 8
        assert depth is None

    def test_dit_2d_larger(self):
        """DiT 2D: 256 tokens at patch_size=2 → input_size=32."""
        sd = {"pos_embed": torch.empty(1, 256, 384)}  # 16x16 tokens
        size, depth = _infer_spatial_from_pos_embeds(
            sd, spatial_dims=2, patch_size=2,
            default_input_size=999, default_depth_size=None, model_type="dit",
        )
        assert size == 32

    def test_hdit_2d_uses_level0(self):
        """HDiT uses pos_embeds.0 key."""
        sd = {"pos_embeds.0": torch.empty(1, 64, 384)}  # 8x8 tokens
        size, depth = _infer_spatial_from_pos_embeds(
            sd, spatial_dims=2, patch_size=4,
            default_input_size=999, default_depth_size=None, model_type="hdit",
        )
        assert size == 32  # 8 * 4

    def test_uvit_2d_subtracts_time_token(self):
        """UViT pos_embed includes 1 time token: total = 1 + spatial tokens."""
        # 4x4 = 16 spatial tokens + 1 time token = 17
        sd = {"pos_embed": torch.empty(1, 17, 512)}
        size, depth = _infer_spatial_from_pos_embeds(
            sd, spatial_dims=2, patch_size=2,
            default_input_size=999, default_depth_size=None, model_type="uvit",
        )
        assert size == 8

    def test_dit_3d_depth_greater_than_spatial(self):
        """Regression: 128x128x160 volume must not reject depth > spatial."""
        # (128/4)^2 * (160/4) = 32*32*40 = 40960 tokens
        sd = {"pos_embed": torch.empty(1, 40960, 384)}
        size, depth = _infer_spatial_from_pos_embeds(
            sd, spatial_dims=3, patch_size=4,
            default_input_size=999, default_depth_size=999, model_type="dit",
        )
        assert size == 128
        assert depth == 160

    def test_hdit_3d_depth_greater_than_spatial(self):
        """Regression: HDiT on 128x128x160 volume, patch_size=4."""
        # Same token count, uses pos_embeds.0 key
        sd = {"pos_embeds.0": torch.empty(1, 40960, 384)}
        size, depth = _infer_spatial_from_pos_embeds(
            sd, spatial_dims=3, patch_size=4,
            default_input_size=999, default_depth_size=999, model_type="hdit",
        )
        assert size == 128
        assert depth == 160

    def test_dit_3d_cube_volume(self):
        """128x128x128 cube: depth == spatial, ratio = 1.0 (perfect)."""
        # (128/4)^2 * (128/4) = 32*32*32 = 32768 tokens
        sd = {"pos_embed": torch.empty(1, 32768, 384)}
        size, depth = _infer_spatial_from_pos_embeds(
            sd, spatial_dims=3, patch_size=4,
            default_input_size=999, default_depth_size=999, model_type="dit",
        )
        assert size == 128
        assert depth == 128

    def test_dit_3d_depth_less_than_spatial(self):
        """64x64x32 volume: depth < spatial still works."""
        # (64/2)^2 * (32/2) = 32*32*16 = 16384 tokens
        sd = {"pos_embed": torch.empty(1, 16384, 384)}
        size, depth = _infer_spatial_from_pos_embeds(
            sd, spatial_dims=3, patch_size=2,
            default_input_size=999, default_depth_size=999, model_type="dit",
        )
        assert size == 64
        assert depth == 32

    def test_missing_key_returns_defaults(self):
        size, depth = _infer_spatial_from_pos_embeds(
            {}, spatial_dims=2, patch_size=2,
            default_input_size=42, default_depth_size=7, model_type="dit",
        )
        assert size == 42
        assert depth == 7


# ---------------------------------------------------------------------------
# _infer_channels_from_state_dict
# ---------------------------------------------------------------------------


class TestInferChannels:
    """Infer in/out channels from weight shapes."""

    def test_unet_keys(self):
        """UNet: conv_in.conv.weight and out.2.conv.weight."""
        sd = {
            "conv_in.conv.weight": torch.empty(64, 2, 3, 3),   # in=2
            "out.2.conv.weight": torch.empty(1, 64, 3, 3),     # out=1
        }
        in_ch, out_ch = _infer_channels_from_state_dict(sd)
        assert in_ch == 2
        assert out_ch == 1

    def test_unet_wrapped_keys(self):
        """Wrapped UNet uses model. prefix."""
        sd = {
            "model.conv_in.conv.weight": torch.empty(64, 3, 3, 3),
            "model.out.2.conv.weight": torch.empty(2, 64, 3, 3),
        }
        in_ch, out_ch = _infer_channels_from_state_dict(sd)
        assert in_ch == 3
        assert out_ch == 2

    def test_transformer_keys(self):
        """DiT/HDiT: x_embedder.proj.weight and final_conv.weight."""
        sd = {
            "x_embedder.proj.weight": torch.empty(384, 4, 2, 2),   # in=4
            "final_conv.weight": torch.empty(1, 1, 3, 3),          # out=1
        }
        in_ch, out_ch = _infer_channels_from_state_dict(sd)
        assert in_ch == 4
        assert out_ch == 1

    def test_empty_returns_none(self):
        in_ch, out_ch = _infer_channels_from_state_dict({})
        assert in_ch is None
        assert out_ch is None

    def test_non_tensor_values_skipped(self):
        sd = {
            "conv_in.conv.weight": "not a tensor",
            "out.2.conv.weight": 42,
        }
        in_ch, out_ch = _infer_channels_from_state_dict(sd)
        assert in_ch is None
        assert out_ch is None
