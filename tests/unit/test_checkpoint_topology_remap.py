"""Tests for bare→wrapped checkpoint key remapping in DiffusionTrainer.load_checkpoint.

Scenario: a checkpoint is saved from a bare UNet (no ScoreAug / omega wrapper);
a new run applies ScoreAugModelWrapper, producing a wrapped model whose
state_dict has `model.` prefixed keys plus `omega_time_embed.*` top-level keys
(aliased to the inner OmegaTimeEmbed). The trainer must auto-remap instead of
raising on strict load.
"""
from __future__ import annotations

import torch
from torch import nn


class _BareUNetLike(nn.Module):
    """Minimal stand-in for DiffusionModelUNet with a time_embed Sequential."""

    def __init__(self) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(1, 4, kernel_size=3, padding=1)
        self.time_embed = nn.Sequential(nn.Linear(16, 8), nn.SiLU(), nn.Linear(8, 8))
        self.out = nn.Conv2d(4, 1, kernel_size=1)


class _OmegaTimeEmbed(nn.Module):
    def __init__(self, original: nn.Module, embed_dim: int) -> None:
        super().__init__()
        self.original = original
        self.omega_mlp = nn.Sequential(nn.Linear(4, embed_dim), nn.SiLU(), nn.Linear(embed_dim, embed_dim))
        nn.init.zeros_(self.omega_mlp[0].weight)
        nn.init.zeros_(self.omega_mlp[0].bias)
        nn.init.zeros_(self.omega_mlp[2].weight)
        nn.init.zeros_(self.omega_mlp[2].bias)
        self.register_buffer('_omega_encoding', torch.zeros(1, 4))


class _ScoreAugLikeWrapper(nn.Module):
    """Mirrors ScoreAugModelWrapper's state_dict topology."""

    def __init__(self, inner: nn.Module, embed_dim: int = 8) -> None:
        super().__init__()
        self.model = inner
        wrapped = _OmegaTimeEmbed(inner.time_embed, embed_dim)
        self.model.time_embed = wrapped
        self.omega_time_embed = wrapped  # aliased — same tensors as model.time_embed


def _make_trainer_stub(model: nn.Module):
    """Build a stub that exposes just the remap helpers we want to test."""
    from medgen.pipeline.trainer import DiffusionTrainer

    class _Stub:
        model_raw = model
        _detect_bare_to_wrapped_mismatch = DiffusionTrainer._detect_bare_to_wrapped_mismatch
        _remap_bare_to_wrapped = DiffusionTrainer._remap_bare_to_wrapped

    return _Stub()


def test_detects_bare_to_wrapper_mismatch():
    bare = _BareUNetLike()
    wrapped = _ScoreAugLikeWrapper(_BareUNetLike())
    trainer = _make_trainer_stub(wrapped)

    bare_sd = bare.state_dict()
    assert trainer._detect_bare_to_wrapped_mismatch(bare_sd) is True


def test_matches_topology_returns_false():
    wrapped = _ScoreAugLikeWrapper(_BareUNetLike())
    trainer = _make_trainer_stub(wrapped)

    wrapped_sd = wrapped.state_dict()
    # Wrapped→wrapped: model.* keys are present in ckpt → no mismatch
    assert trainer._detect_bare_to_wrapped_mismatch(wrapped_sd) is False


def test_remap_populates_all_wrapped_paths():
    bare = _BareUNetLike()
    wrapped = _ScoreAugLikeWrapper(_BareUNetLike())
    trainer = _make_trainer_stub(wrapped)

    bare_sd = bare.state_dict()
    remapped = trainer._remap_bare_to_wrapped(bare_sd)

    # Every conv/output key gains `model.` prefix
    assert 'model.conv_in.weight' in remapped
    assert 'model.out.weight' in remapped
    # time_embed.N.* → model.time_embed.original.N.*
    assert 'model.time_embed.original.0.weight' in remapped
    assert 'model.time_embed.original.2.weight' in remapped
    # Alias to omega_time_embed.original.* is also populated
    assert 'omega_time_embed.original.0.weight' in remapped
    assert 'omega_time_embed.original.2.weight' in remapped


def test_remap_load_succeeds_on_wrapper():
    """End-to-end: load a bare state_dict into a wrapper via remap + strict=False,
    verify only wrapper-added params (omega_mlp, _omega_encoding) are missing."""
    bare = _BareUNetLike()
    wrapped = _ScoreAugLikeWrapper(_BareUNetLike())
    trainer = _make_trainer_stub(wrapped)

    bare_sd = bare.state_dict()
    remapped = trainer._remap_bare_to_wrapped(bare_sd)

    missing, unexpected = wrapped.load_state_dict(remapped, strict=False)
    assert unexpected == [], f"Expected no unexpected keys, got {unexpected}"
    # Only new wrapper params should be missing
    for k in missing:
        assert 'omega_mlp' in k or '_omega_encoding' in k, f"Unexpected missing key: {k}"


def test_remap_preserves_pretrained_values():
    """After remap + load, the wrapped model's inner UNet weights must equal the
    bare-checkpoint values (not the wrapper's random init)."""
    bare = _BareUNetLike()
    # Set distinct values so we can distinguish loaded vs re-initialized.
    with torch.no_grad():
        bare.conv_in.weight.fill_(0.1234)
        bare.time_embed[0].weight.fill_(0.5678)

    wrapped = _ScoreAugLikeWrapper(_BareUNetLike())
    trainer = _make_trainer_stub(wrapped)

    remapped = trainer._remap_bare_to_wrapped(bare.state_dict())
    wrapped.load_state_dict(remapped, strict=False)

    assert torch.allclose(wrapped.model.conv_in.weight, torch.full_like(wrapped.model.conv_in.weight, 0.1234))
    assert torch.allclose(
        wrapped.model.time_embed.original[0].weight,
        torch.full_like(wrapped.model.time_embed.original[0].weight, 0.5678),
    )
    # Via aliasing, omega_time_embed.original.* sees the same tensor
    assert wrapped.omega_time_embed.original[0].weight.data_ptr() == wrapped.model.time_embed.original[0].weight.data_ptr()
    # Wrapper-added omega_mlp stays at zero-init (identity behavior)
    assert torch.all(wrapped.model.time_embed.omega_mlp[0].weight == 0)
    assert torch.all(wrapped.model.time_embed.omega_mlp[2].weight == 0)
