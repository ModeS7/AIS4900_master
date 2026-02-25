"""Roundtrip tests: create model → forward → save → load → verify.

Creates REAL (tiny) models, saves a checkpoint, reloads via
load_diffusion_model_with_metadata, and verifies wrapper_type, epoch,
and output equality. Catches wrong key patterns, missing config fields,
and shape mismatches that mocked tests cannot.
"""

import tempfile
from pathlib import Path

import pytest
import torch
from monai.networks.nets import DiffusionModelUNet

from medgen.diffusion.loading import load_diffusion_model_with_metadata
from medgen.models.dit import create_dit
from medgen.models.hdit import create_hdit
from medgen.models.uvit import create_uvit


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


def _save_checkpoint(
    model: torch.nn.Module,
    path: Path,
    *,
    epoch: int = 5,
    model_config: dict | None = None,
    config: dict | None = None,
) -> None:
    """Save a minimal checkpoint matching the format load_diffusion_model expects."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "model_config": model_config or {},
            "config": config or {},
        },
        path,
    )


# ---------------------------------------------------------------------------
# UNet roundtrip
# ---------------------------------------------------------------------------


class TestUNetRoundtrip:
    """Tiny UNet: create → save → load → verify."""

    @staticmethod
    def _make_unet() -> DiffusionModelUNet:
        return DiffusionModelUNet(
            spatial_dims=2,
            in_channels=2,
            out_channels=1,
            channels=(8, 16),
            attention_levels=(False, False),
            num_res_blocks=1,
            num_head_channels=8,
            norm_num_groups=8,
        )

    def test_roundtrip(self, tmp_dir: Path):
        model = self._make_unet()
        x = torch.randn(1, 2, 32, 32)
        t = torch.zeros(1)
        with torch.no_grad():
            expected = model(x, t)

        ckpt = tmp_dir / "unet.pt"
        _save_checkpoint(
            model,
            ckpt,
            epoch=42,
            model_config={
                "in_channels": 2,
                "out_channels": 1,
                "channels": [8, 16],
                "attention_levels": [False, False],
                "num_res_blocks": 1,
                "num_head_channels": 8,
                "norm_num_groups": 8,
            },
        )

        result = load_diffusion_model_with_metadata(
            str(ckpt), device=torch.device("cpu"),
        )
        assert result.wrapper_type == "raw"
        assert result.epoch == 42

        with torch.no_grad():
            actual = result.model(x, t)
        torch.testing.assert_close(actual, expected)


# ---------------------------------------------------------------------------
# DiT roundtrip
# ---------------------------------------------------------------------------


class TestDiTRoundtrip:
    """Tiny DiT-S: create → save → load → verify."""

    @staticmethod
    def _make_dit() -> torch.nn.Module:
        return create_dit(
            variant="S",
            spatial_dims=2,
            input_size=8,
            patch_size=2,
            in_channels=2,
            out_channels=1,
        )

    def test_roundtrip(self, tmp_dir: Path):
        model = self._make_dit()
        x = torch.randn(1, 2, 8, 8)
        t = torch.zeros(1)
        with torch.no_grad():
            expected = model(x, t)

        ckpt = tmp_dir / "dit.pt"
        _save_checkpoint(
            model,
            ckpt,
            epoch=10,
            model_config={
                "model_type": "dit",
                "variant": "S",
                "in_channels": 2,
                "out_channels": 1,
                "patch_size": 2,
                "image_size": 8,
            },
        )

        result = load_diffusion_model_with_metadata(
            str(ckpt), device=torch.device("cpu"),
        )
        assert result.wrapper_type == "raw"
        assert result.epoch == 10

        with torch.no_grad():
            actual = result.model(x, t)
        torch.testing.assert_close(actual, expected)


# ---------------------------------------------------------------------------
# HDiT roundtrip
# ---------------------------------------------------------------------------


class TestHDiTRoundtrip:
    """Tiny HDiT-S: create → save → load → verify."""

    @staticmethod
    def _make_hdit() -> torch.nn.Module:
        return create_hdit(
            variant="S",
            spatial_dims=2,
            input_size=16,
            patch_size=4,
            in_channels=2,
            out_channels=1,
            level_depths=[1, 2, 1],
        )

    def test_roundtrip(self, tmp_dir: Path):
        model = self._make_hdit()
        x = torch.randn(1, 2, 16, 16)
        t = torch.zeros(1)
        with torch.no_grad():
            expected = model(x, t)

        ckpt = tmp_dir / "hdit.pt"
        _save_checkpoint(
            model,
            ckpt,
            epoch=7,
            model_config={
                "model_type": "hdit",
                "variant": "S",
                "in_channels": 2,
                "out_channels": 1,
                "patch_size": 4,
                "image_size": 16,
            },
        )

        result = load_diffusion_model_with_metadata(
            str(ckpt), device=torch.device("cpu"),
        )
        assert result.wrapper_type == "raw"
        assert result.epoch == 7

        with torch.no_grad():
            actual = result.model(x, t)
        torch.testing.assert_close(actual, expected)


# ---------------------------------------------------------------------------
# UViT roundtrip
# ---------------------------------------------------------------------------


class TestUViTRoundtrip:
    """Tiny UViT-S: create → save → load → verify."""

    @staticmethod
    def _make_uvit() -> torch.nn.Module:
        return create_uvit(
            variant="S",
            spatial_dims=2,
            input_size=8,
            patch_size=2,
            in_channels=2,
            out_channels=1,
        )

    def test_roundtrip(self, tmp_dir: Path):
        model = self._make_uvit()
        x = torch.randn(1, 2, 8, 8)
        t = torch.zeros(1)
        with torch.no_grad():
            expected = model(x, t)

        ckpt = tmp_dir / "uvit.pt"
        _save_checkpoint(
            model,
            ckpt,
            epoch=3,
            model_config={
                "model_type": "uvit",
                "variant": "S",
                "in_channels": 2,
                "out_channels": 1,
                "patch_size": 2,
                "image_size": 8,
            },
        )

        result = load_diffusion_model_with_metadata(
            str(ckpt), device=torch.device("cpu"),
        )
        assert result.wrapper_type == "raw"
        assert result.epoch == 3

        with torch.no_grad():
            actual = result.model(x, t)
        torch.testing.assert_close(actual, expected)


# ---------------------------------------------------------------------------
# Auto-detect architecture (no model_type in config)
# ---------------------------------------------------------------------------


class TestAutoDetectArch:
    """load_diffusion_model_with_metadata should auto-detect arch from state dict."""

    def test_dit_auto_detect(self, tmp_dir: Path):
        """DiT detected from blocks.* + final_layer keys."""
        model = create_dit(
            variant="S", spatial_dims=2, input_size=8,
            patch_size=2, in_channels=1, out_channels=1,
        )
        ckpt = tmp_dir / "dit_auto.pt"
        # Save WITHOUT model_type — forces auto-detection
        _save_checkpoint(
            model, ckpt,
            model_config={"in_channels": 1, "out_channels": 1, "variant": "S",
                          "patch_size": 2, "image_size": 8},
        )
        result = load_diffusion_model_with_metadata(
            str(ckpt), device=torch.device("cpu"),
        )
        assert result.wrapper_type == "raw"

    def test_hdit_auto_detect(self, tmp_dir: Path):
        """HDiT detected from encoder_levels.* keys."""
        model = create_hdit(
            variant="S", spatial_dims=2, input_size=16,
            patch_size=4, in_channels=1, out_channels=1,
            level_depths=[1, 2, 1],
        )
        ckpt = tmp_dir / "hdit_auto.pt"
        _save_checkpoint(
            model, ckpt,
            model_config={"in_channels": 1, "out_channels": 1, "variant": "S",
                          "patch_size": 4, "image_size": 16},
        )
        result = load_diffusion_model_with_metadata(
            str(ckpt), device=torch.device("cpu"),
        )
        assert result.wrapper_type == "raw"
