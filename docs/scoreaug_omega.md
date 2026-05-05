# ScoreAug omega encoding

Spec for the 36-dim ω vector that conditions the denoiser on which transform
was applied to the clean target. Implementation lives in
`src/medgen/augmentation/score_aug_omega.py`. Paper:
**Hou et al. 2025, *ScoreAug*, arXiv:2508.07926.**

## Why omega exists

ScoreAug applies a transform `T` to the clean image, runs the diffusion
forward process on the transformed image, and asks the denoiser to predict
the transformed clean image. The denoiser must know **which transform was
applied** so it can produce the correct target — otherwise it has to
average over all possible transforms, which is exactly what causes
augmentation leak in generations.

ω is the conditioning signal: a 36-dim vector that encodes both the
transform and the modality intensity scale. It is fed into the same
embedding pathway as the timestep (via `time_embed` for UNet,
`t_embedder` for DiT).

## Layout (36 dims)

```
Dims 0–3     Active mask: [spatial, translation, cutout, pattern]
Dims 4–9     Spatial type one-hot:
              2D: [rot90, hflip, vflip, rot90_hflip,    _,       _]
              3D: [rot90_d, rot90_h, rot90_w, flip_d, flip_h, flip_w]
Dim  10      rot_k normalized: k / 3.0  (k ∈ {0,1,2,3})
Dims 11–13   Translation params:
              2D: [dx, dy, _]
              3D: [dd, dh, dw]
Dims 14–15   Cutout size:
              2D: [size_y, size_x]
              3D: [size_h, size_w]   (size_d goes in dim 16, see below)
Dims 16–31   Pattern ID one-hot (16 patterns); dim 16 ALSO holds 3D cutout
             size_d (safe overload — see invariant below)
Dims 32–35   Mode one-hot (4 modalities): [bravo, flair, t1_pre, t1_gd]
```

`OMEGA_ENCODING_DIM = 36` is exported as a module constant.

## Invariants that matter

### I1 — Identity = all zeros (mode bits aside)

When no transform is applied (`omega=None`), the encoding is the zero
vector except for the mode bits at dims 32–35. **Per the paper §Sampling**:

> "set condition to zeros … to generate untransformed image"

The earlier (pre-fix) implementation set `enc[0, 0] = 1.0` for the identity
case, which collided with the "spatial active" bit set by rotations and
flips. That made identity ambiguous with a zero-parameter translation
(active mask: `[0, 1, 0, 0]`, params zero) and is what caused the
**exp23 translation leak** in generations: the denoiser learned that a
"spatial active = 1, params zero" signal could mean either "rotate by
zero" or "translate by zero" or "identity", and during generation it
hallucinated translations into the output.

Code that enforces this:
```python
if omega is None:
    return enc           # zeros (modulo mode bits)
```

### I2 — 3D cutout dim-16 overload is safe

The 3D cutout encodes its third size component in dim 16. Dim 16 is also
used as the `pattern_id == 0` slot of the pattern one-hot. The overload
is safe because:

- Cutout sets `dims[2] = 1` (cutout active) and `dims[3] = 0`.
- Pattern sets `dims[3] = 1` and `dims[2] = 0`.
- Cutout and pattern are mutually exclusive — they never co-fire in a
  single ω.

The denoiser disambiguates via the active mask (dims 0–3). If you ever
add a transform that sets *both* `dim 2` and `dim 3`, the overload breaks.

### I3 — Cutout sizes are encoded, locations are NOT

Per paper: "only the cutout size ω_c = (h, w) is kept." Center coordinates
`(cx, cy, cz)` are intentionally absent. Leaking location would let the
denoiser recover the masked region by spatial copy. Sizes are needed so
the model can learn the spatial extent of the masked area.

## Mode intensity scaling

Each modality gets a different post-noise intensity multiplier:

| Mode ID | Modality | Scale  |
|---------|----------|--------|
| 0       | bravo    | 0.85   |
| 1       | flair    | 1.15   |
| 2       | t1_pre   | 0.92   |
| 3       | t1_gd    | 1.08   |

Asymmetric around 1.0 by design — this forces the model to learn
modality-specific features rather than inverting a uniform scale. The
scale is applied to every input *after* noise addition; the model must
predict the *unscaled* target. The mode bits at dims 32–35 are the
necessary conditioning.

## Composition mode (v2)

`omega` may carry multiple transforms in a single sample (`v2=True` or
`compose=True`). Each is encoded into the same 36-dim vector by ORing
active-mask bits and summing/setting parameter slots. The encoder
function `_encode_single_transform` is called once per transform in the
list. The result is a single ω vector — **not** a sequence — so two
transforms collapse into the same buffer.

Practical implication: don't compose two transforms that overload the
same parameter slots (e.g. two cutouts of different sizes). The encoder
silently overwrites — no error.

## Where this gets used

- **Encoder:** `encode_omega()` and `encode_omega_3d()` in
  `score_aug_omega.py`. Compile-compatible: returns `[1, 36]` that
  broadcasts to `[B, 36]` so the buffer shape is constant under
  `torch.compile`.
- **Trainer call site:** `DiffusionTrainer._compute_omega_and_input()` —
  drawn during the augmentation step, fed into the denoiser.
- **Denoiser embed pathway:**
  - UNet: appended to the sinusoidal time embedding inside `time_embed`.
  - DiT: appended to `t_embedder` output.
- **Generation:** `omega=None` → all zeros (per I1). Models trained with
  ScoreAug must be sampled with this convention; mismatch causes leaks.

## Tests

Three test files cover the encoding:

| Test file                               | Covers                                        |
|-----------------------------------------|-----------------------------------------------|
| `tests/unit/test_score_aug_omega_encoding.py` | I1 (identity zeros), I2 (cutout dim overload), 11 cases |
| `tests/unit/test_score_aug_omega_coverage.py` | Audit: every transform type produces a valid encoding (14 cases) |
| `tests/unit/test_checkpoint_topology_remap.py` | Loading bare-UNet checkpoints into ScoreAug-wrapped models (5 cases) |

## History

- **March 2026 — exp23 leak diagnosis.** ScoreAug bravo model produced
  generations with structural translation artifacts even when fed real
  (perfect) seg masks. Traced to identity-as-`[1,0,..]` and to 2D cutout
  size-encoding code being run on 3D params (KeyError silently producing
  identity). See `memory/finding_exp1_1_1000_best.md` and the project
  finding on ScoreAug generation issues.
- **April 2026 — fix.** D1 (3D cutout: encode `size_d` into dim 16) and
  D2 (identity = zeros) landed with regression tests. Verified against
  the paper.
