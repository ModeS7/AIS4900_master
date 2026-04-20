"""Exhaustive audit: every transform the sampler can emit must have all its
deterministic parameters encoded in omega.

Paper (Hou et al. 2025): ω must carry every parameter the denoiser needs to
predict T(d; ω) correctly. The only intentional omission per paper is cutout
CENTER (to prevent location leakage) — SIZES must be encoded.

This test enumerates every (transform_type, params) tuple the sampler in
score_aug.py produces, runs encode_omega on each, and asserts every numeric
parameter (except intentional cutout center) is reflected somewhere in the
36-dim encoding with nonzero bits that disambiguate from all other transforms.
"""
from __future__ import annotations

import torch

from medgen.augmentation.score_aug_omega import OMEGA_ENCODING_DIM, encode_omega

CPU = torch.device('cpu')


def _enc(omega, spatial_dims):
    return encode_omega(omega, CPU, spatial_dims=spatial_dims)[0]


# --- 2D spatial transforms -------------------------------------------------

def test_2d_rot90_encodes_k_for_all_k():
    for k in [1, 2, 3]:
        e = _enc({'type': 'rot90', 'params': {'k': k}}, 2)
        assert e[0] == 1.0, "spatial-active bit not set"
        assert e[4] == 1.0, "rot90 type bit not set"
        assert abs(e[10].item() - k / 3.0) < 1e-6, f"rot_k not encoded correctly for k={k}"


def test_2d_hflip_and_vflip_disambiguated():
    e_h = _enc({'type': 'hflip', 'params': {}}, 2)
    e_v = _enc({'type': 'vflip', 'params': {}}, 2)
    assert e_h[0] == 1.0 and e_h[5] == 1.0 and e_h[6] == 0.0
    assert e_v[0] == 1.0 and e_v[5] == 0.0 and e_v[6] == 1.0
    assert not torch.equal(e_h, e_v)


def test_2d_rot90_hflip_encodes_k():
    for k in [1, 3]:  # sampler only emits k=1,3
        e = _enc({'type': 'rot90_hflip', 'params': {'k': k}}, 2)
        assert e[0] == 1.0 and e[7] == 1.0
        assert abs(e[10].item() - k / 3.0) < 1e-6


# --- 3D spatial transforms -------------------------------------------------

def test_3d_rot90_encodes_axis_and_k():
    # Sampler only emits axis='d' (per our constraint), but encoder handles all 3.
    for axis, expected_dim in [('d', 4), ('h', 5), ('w', 6)]:
        for k in [1, 2, 3]:
            e = _enc({'type': 'rot90_3d', 'params': {'axis': axis, 'k': k}}, 3)
            assert e[0] == 1.0
            assert e[expected_dim] == 1.0, f"axis={axis} → dim {expected_dim} not set"
            # Other axis dims must be zero
            for other_dim in {4, 5, 6} - {expected_dim}:
                assert e[other_dim] == 0.0
            assert abs(e[10].item() - k / 3.0) < 1e-6


def test_3d_flips_disambiguated():
    e_d = _enc({'type': 'flip_d', 'params': {}}, 3)
    e_h = _enc({'type': 'flip_h', 'params': {}}, 3)
    e_w = _enc({'type': 'flip_w', 'params': {}}, 3)
    assert e_d[7] == 1.0 and e_d[8] == 0.0 and e_d[9] == 0.0
    assert e_h[7] == 0.0 and e_h[8] == 1.0 and e_h[9] == 0.0
    assert e_w[7] == 0.0 and e_w[8] == 0.0 and e_w[9] == 1.0
    assert not torch.equal(e_d, e_h) and not torch.equal(e_h, e_w)


# --- Translation -----------------------------------------------------------

def test_2d_translation_encodes_dx_dy_exactly():
    params = {'dx': 0.37, 'dy': -0.18}
    e = _enc({'type': 'translate', 'params': params}, 2)
    assert e[1] == 1.0
    assert abs(e[11].item() - 0.37) < 1e-5
    assert abs(e[12].item() - (-0.18)) < 1e-5
    # dw dim is not used in 2D → must be zero
    assert e[13] == 0.0


def test_3d_translation_encodes_dd_dh_dw_exactly():
    params = {'dd': 0.11, 'dh': -0.22, 'dw': 0.33}
    e = _enc({'type': 'translate', 'params': params}, 3)
    assert e[1] == 1.0
    assert abs(e[11].item() - 0.11) < 1e-5
    assert abs(e[12].item() - (-0.22)) < 1e-5
    assert abs(e[13].item() - 0.33) < 1e-5


# --- Cutout (D1 fix) -------------------------------------------------------

def test_2d_cutout_encodes_size_not_center():
    params = {'cx': 0.42, 'cy': 0.58, 'size_x': 0.23, 'size_y': 0.19}
    e = _enc({'type': 'cutout', 'params': params}, 2)
    assert e[2] == 1.0  # cutout active
    assert abs(e[14].item() - 0.19) < 1e-5  # size_y
    assert abs(e[15].item() - 0.23) < 1e-5  # size_x
    # Center coords must NOT leak to the denoiser
    # (verify no dims carry cx=0.42 or cy=0.58)
    nonzero_vals = e[e != 0].tolist()
    assert 0.42 not in [round(v, 2) for v in nonzero_vals]
    assert 0.58 not in [round(v, 2) for v in nonzero_vals]


def test_3d_cutout_encodes_all_three_sizes_not_centers():
    params = {
        'cd': 0.44, 'ch': 0.56, 'cw': 0.51,
        'size_d': 0.14, 'size_h': 0.21, 'size_w': 0.27,
    }
    e = _enc({'type': 'cutout', 'params': params}, 3)
    assert e[2] == 1.0
    assert abs(e[14].item() - 0.21) < 1e-5  # size_h
    assert abs(e[15].item() - 0.27) < 1e-5  # size_w
    assert abs(e[16].item() - 0.14) < 1e-5  # size_d (overloaded dim)
    # Center coords must not appear in any dim
    nonzero_vals = [round(v, 2) for v in e[e != 0].tolist()]
    for center in [0.44, 0.56, 0.51]:
        assert center not in nonzero_vals


# --- Pattern (v2 mode) -----------------------------------------------------

def test_pattern_encodes_pattern_id_one_hot():
    for pid in range(16):
        e = _enc({'type': 'pattern', 'params': {'pattern_id': pid}}, 2)
        assert e[3] == 1.0, "pattern active bit not set"
        assert e[16 + pid] == 1.0, f"pattern one-hot not set for id={pid}"
        # Only this one-hot bit in 16-31 range
        for other in range(16):
            if other != pid:
                assert e[16 + other] == 0.0


# --- Cross-transform disambiguation ----------------------------------------

def _assert_distinct(samples, spatial_dims):
    """All distinct samples emit distinct encodings within a single spatial_dims context."""
    encodings = [_enc(om, spatial_dims) for om in samples]
    seen: dict[tuple, int] = {}
    for i, enc in enumerate(encodings):
        key = tuple(enc.tolist())
        if key in seen:
            raise AssertionError(
                f"Collision: {samples[seen[key]]} and {samples[i]} produce identical "
                f"encoding at spatial_dims={spatial_dims}"
            )
        seen[key] = i


def test_all_2d_transforms_distinct():
    """A 2D training run never emits 3D transforms, so disambiguation only needs
    to hold within 2D. Audit all 2D samples."""
    samples = [
        None,  # identity
        {'type': 'rot90', 'params': {'k': 1}},
        {'type': 'rot90', 'params': {'k': 2}},
        {'type': 'rot90', 'params': {'k': 3}},
        {'type': 'hflip', 'params': {}},
        {'type': 'vflip', 'params': {}},
        {'type': 'rot90_hflip', 'params': {'k': 1}},
        {'type': 'rot90_hflip', 'params': {'k': 3}},
        {'type': 'translate', 'params': {'dx': 0.1, 'dy': 0.2}},
        {'type': 'translate', 'params': {'dx': -0.3, 'dy': 0.0}},
        {'type': 'cutout', 'params': {'cx': 0.5, 'cy': 0.5, 'size_x': 0.2, 'size_y': 0.2}},
        {'type': 'cutout', 'params': {'cx': 0.3, 'cy': 0.7, 'size_x': 0.15, 'size_y': 0.25}},
        *[{'type': 'pattern', 'params': {'pattern_id': i}} for i in range(16)],
    ]
    _assert_distinct(samples, spatial_dims=2)


def test_all_3d_transforms_distinct():
    """A 3D training run never emits 2D transforms, so disambiguation only needs
    to hold within 3D. Audit all 3D samples the sampler can emit."""
    samples = [
        None,  # identity
        # 3D rotation (sampler only emits axis='d', encoder supports all 3)
        *[{'type': 'rot90_3d', 'params': {'axis': 'd', 'k': k}} for k in [1, 2, 3]],
        *[{'type': 'rot90_3d', 'params': {'axis': 'h', 'k': k}} for k in [1, 2, 3]],
        *[{'type': 'rot90_3d', 'params': {'axis': 'w', 'k': k}} for k in [1, 2, 3]],
        # 3D flips
        {'type': 'flip_d', 'params': {}},
        {'type': 'flip_h', 'params': {}},
        {'type': 'flip_w', 'params': {}},
        # 3D translations
        {'type': 'translate', 'params': {'dd': 0.0, 'dh': 0.1, 'dw': 0.3}},
        {'type': 'translate', 'params': {'dd': 0.0, 'dh': -0.2, 'dw': -0.4}},
        # 3D cutout
        {'type': 'cutout', 'params': {'cd': 0.5, 'ch': 0.5, 'cw': 0.5,
                                       'size_d': 0.1, 'size_h': 0.2, 'size_w': 0.15}},
        {'type': 'cutout', 'params': {'cd': 0.3, 'ch': 0.4, 'cw': 0.6,
                                       'size_d': 0.2, 'size_h': 0.25, 'size_w': 0.3}},
    ]
    _assert_distinct(samples, spatial_dims=3)


def test_identity_is_orthogonal_to_every_active_encoding():
    """After D2 fix: identity = zero vector. Its L2 distance to every active
    encoding is nonzero and determined entirely by the active encoding."""
    identity = _enc(None, 3)
    assert torch.all(identity == 0)
    actives = [
        _enc({'type': 'rot90_3d', 'params': {'axis': 'd', 'k': 1}}, 3),
        _enc({'type': 'flip_d', 'params': {}}, 3),
        _enc({'type': 'translate', 'params': {'dd': 0.0, 'dh': 0.0, 'dw': 0.0}}, 3),
        _enc({'type': 'cutout', 'params': {'cd': 0.5, 'ch': 0.5, 'cw': 0.5,
                                            'size_d': 0.2, 'size_h': 0.2, 'size_w': 0.2}}, 3),
    ]
    for a in actives:
        assert (identity - a).norm() > 0.5, \
            "Identity must be clearly separated from every active encoding"


# --- Full dim budget audit -------------------------------------------------

def test_encoding_dim_budget_exactly_36():
    """Every encoding returned is exactly 36-dim — no overflow from the D1 fix."""
    samples = [
        None,
        {'type': 'rot90_3d', 'params': {'axis': 'd', 'k': 3}},
        {'type': 'translate', 'params': {'dd': 0.1, 'dh': 0.2, 'dw': 0.3}},
        {'type': 'cutout', 'params': {'cd': 0.5, 'ch': 0.5, 'cw': 0.5,
                                       'size_d': 0.2, 'size_h': 0.2, 'size_w': 0.2}},
    ]
    for om in samples:
        e = encode_omega(om, CPU, spatial_dims=3)
        assert e.shape == (1, OMEGA_ENCODING_DIM) == (1, 36)
