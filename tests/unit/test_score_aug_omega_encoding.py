"""Tests for ScoreAug omega encoding (D1, D2 paper-conformance fixes).

Paper: Hou et al. 2025, arXiv:2508.07926.
- D2: identity must encode to all zeros (paper §Sampling: "set the condition
      to zeros ... to generate an untransformed image").
- D1: cutout must encode size (paper §Augmentation and Condition: "only the
      cutout size ω_c = (h, w) is kept").
"""
from __future__ import annotations

import pytest
import torch

from medgen.augmentation.score_aug_omega import OMEGA_ENCODING_DIM, encode_omega


def _cpu():
    return torch.device('cpu')


# --- D2: identity → all zeros ---------------------------------------------

def test_identity_omega_encodes_to_all_zeros_2d():
    enc = encode_omega(None, _cpu(), spatial_dims=2)
    assert enc.shape == (1, OMEGA_ENCODING_DIM)
    assert torch.all(enc == 0), f"Identity (2D) must be all zeros, got nonzero at {torch.nonzero(enc)}"


def test_identity_omega_encodes_to_all_zeros_3d():
    enc = encode_omega(None, _cpu(), spatial_dims=3)
    assert torch.all(enc == 0), f"Identity (3D) must be all zeros, got nonzero at {torch.nonzero(enc)}"


def test_identity_preserves_mode_bits():
    """Mode one-hot is unrelated to the transform — it must still fire for identity."""
    mode_id = torch.tensor(2)  # t1_pre
    enc = encode_omega(None, _cpu(), mode_id=mode_id, spatial_dims=2)
    assert enc[0, 32 + 2] == 1.0
    # All non-mode dims must be zero
    enc_no_mode = enc.clone()
    enc_no_mode[0, 32:36] = 0.0
    assert torch.all(enc_no_mode == 0)


def test_identity_disambiguated_from_rot90():
    """Regression for the leak: identity must not share any bits with rotation."""
    enc_id = encode_omega(None, _cpu(), spatial_dims=2)
    enc_rot = encode_omega({'type': 'rot90', 'params': {'k': 1}}, _cpu(), spatial_dims=2)
    overlap = (enc_id != 0) & (enc_rot != 0)
    assert not overlap.any(), f"Identity shares active bits with rot90 at dims {torch.nonzero(overlap)}"


def test_identity_disambiguated_from_zero_translation():
    """With dx=dy=0 translation (corner case), encoding must still differ from identity."""
    enc_id = encode_omega(None, _cpu(), spatial_dims=2)
    enc_trans = encode_omega(
        {'type': 'translate', 'params': {'dx': 0.0, 'dy': 0.0}}, _cpu(), spatial_dims=2,
    )
    assert not torch.equal(enc_id, enc_trans), \
        "Identity and zero-shift translation must encode differently"
    # Translation-active bit distinguishes them
    assert enc_trans[0, 1] == 1.0 and enc_id[0, 1] == 0.0


# --- D1: cutout size encoded ---------------------------------------------

def test_cutout_2d_encodes_size_not_center():
    params = {'cx': 0.5, 'cy': 0.5, 'size_x': 0.22, 'size_y': 0.17}
    enc = encode_omega({'type': 'cutout', 'params': params}, _cpu(), spatial_dims=2)
    assert enc[0, 2] == 1.0  # cutout active
    assert enc[0, 14] == pytest.approx(0.17)  # size_y
    assert enc[0, 15] == pytest.approx(0.22)  # size_x
    # Center coords must NOT leak (paper: only size is kept)
    assert enc[0, 11] == 0.0
    assert enc[0, 12] == 0.0
    assert enc[0, 13] == 0.0


def test_cutout_3d_encodes_all_three_sizes():
    params = {
        'cd': 0.5, 'ch': 0.5, 'cw': 0.5,
        'size_d': 0.15, 'size_h': 0.22, 'size_w': 0.18,
    }
    enc = encode_omega({'type': 'cutout', 'params': params}, _cpu(), spatial_dims=3)
    assert enc[0, 2] == 1.0  # cutout active
    assert enc[0, 14] == pytest.approx(0.22)  # size_h
    assert enc[0, 15] == pytest.approx(0.18)  # size_w
    assert enc[0, 16] == pytest.approx(0.15)  # size_d (overloaded)


def test_cutout_3d_different_sizes_yield_different_encodings():
    """Sanity: small cutout vs. large cutout must encode differently."""
    small = {'type': 'cutout', 'params': {
        'cd': 0.5, 'ch': 0.5, 'cw': 0.5,
        'size_d': 0.1, 'size_h': 0.1, 'size_w': 0.1,
    }}
    large = {'type': 'cutout', 'params': {
        'cd': 0.5, 'ch': 0.5, 'cw': 0.5,
        'size_d': 0.3, 'size_h': 0.3, 'size_w': 0.3,
    }}
    enc_s = encode_omega(small, _cpu(), spatial_dims=3)
    enc_l = encode_omega(large, _cpu(), spatial_dims=3)
    assert not torch.equal(enc_s, enc_l)


def test_cutout_and_pattern_share_dim16_safely():
    """dim 16 is overloaded: cutout 3D size_d OR pattern-ID-0 one-hot. Both should
    produce valid encodings and the model disambiguates via the active mask."""
    cutout = {'type': 'cutout', 'params': {
        'cd': 0.5, 'ch': 0.5, 'cw': 0.5,
        'size_d': 0.2, 'size_h': 0.2, 'size_w': 0.2,
    }}
    pattern = {'type': 'pattern', 'params': {'pattern_id': 0}}
    enc_c = encode_omega(cutout, _cpu(), spatial_dims=3)
    enc_p = encode_omega(pattern, _cpu(), spatial_dims=3)
    # Different active bits (disambiguates the overload)
    assert enc_c[0, 2] == 1.0 and enc_c[0, 3] == 0.0
    assert enc_p[0, 2] == 0.0 and enc_p[0, 3] == 1.0
    # dim 16: cutout carries size_d=0.2, pattern carries one-hot 1.0
    assert enc_c[0, 16] == pytest.approx(0.2)
    assert enc_p[0, 16] == 1.0


# --- Sanity: translation still encoded correctly -------------------------

def test_translation_2d_still_encodes_dx_dy():
    params = {'dx': 0.3, 'dy': -0.15}
    enc = encode_omega({'type': 'translate', 'params': params}, _cpu(), spatial_dims=2)
    assert enc[0, 1] == 1.0  # translation active
    assert enc[0, 11] == pytest.approx(0.3)
    assert enc[0, 12] == pytest.approx(-0.15)


def test_translation_3d_still_encodes_dd_dh_dw():
    params = {'dd': 0.0, 'dh': 0.1, 'dw': -0.4}
    enc = encode_omega({'type': 'translate', 'params': params}, _cpu(), spatial_dims=3)
    assert enc[0, 1] == 1.0
    assert enc[0, 11] == pytest.approx(0.0)
    assert enc[0, 12] == pytest.approx(0.1)
    assert enc[0, 13] == pytest.approx(-0.4)


