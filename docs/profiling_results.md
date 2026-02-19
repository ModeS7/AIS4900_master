# 3D VRAM Profiling Results

---
---

# 3D DiT VRAM Profiling

Profiled on IDUN cluster, February 15, 2026.

## Hardware
- **GPU**: NVIDIA A100-SXM4-80GB
- **Driver**: 575.57.08
- **CUDA**: 12.9

## Test Configuration
- **Batch size**: 1
- **Precision**: AMP bfloat16
- **Input channels**: 1 (single modality)
- **Configs tested**: 116 (56 latent + 60 pixel)
- **Viable**: 99, **OOM**: 17

## DiT Variants

| Variant | Hidden dim | Heads | Depth | Base params |
|---------|-----------|-------|-------|-------------|
| S | 384 | 6 | 12 | ~33M |
| B | 768 | 12 | 12 | ~130M |
| L | 1024 | 16 | 24 | ~457M |
| XL | 1152 | 16 | 28 | ~674M |

Token count = (D/patch) × (H/patch) × (W/patch). Attention is O(n²) in tokens.

---

## Latent Space DiT (after VAE/VQ-VAE compression)

### DiT-S (Latent)

| Resolution | Patch | Tokens | Params | Peak VRAM | Status |
|-----------|-------|--------|--------|-----------|--------|
| 16x16x10 | 2 | 320 | 32.6M | 0.7 GB | OK |
| 16x16x20 | 2 | 640 | 32.7M | 0.7 GB | OK |
| 16x16x20 | 4 | 80 | 32.5M | 0.7 GB | OK |
| 32x32x20 | 2 | 2,560 | 33.5M | 1.0 GB | OK |
| 32x32x20 | 4 | 320 | 32.6M | 0.7 GB | OK |
| 32x32x40 | 2 | 5,120 | 34.4M | 1.7 GB | OK |
| 32x32x40 | 4 | 640 | 32.8M | 0.7 GB | OK |
| 32x32x40 | 8 | 80 | 32.9M | 0.7 GB | OK |
| 64x64x40 | 2 | 20,480 | 40.3M | 6.0 GB | OK |
| 64x64x40 | 4 | 2,560 | 33.5M | 1.0 GB | OK |
| 64x64x40 | 8 | 320 | 33.0M | 0.7 GB | OK |
| 64x64x80 | 2 | 40,960 | 48.2M | 11.7 GB | OK |
| 64x64x80 | 4 | 5,120 | 34.5M | 1.7 GB | OK |
| 64x64x80 | 8 | 640 | 33.1M | 0.7 GB | OK |

### DiT-B (Latent)

| Resolution | Patch | Tokens | Params | Peak VRAM | Status |
|-----------|-------|--------|--------|-----------|--------|
| 16x16x10 | 2 | 320 | 129.8M | 2.5 GB | OK |
| 16x16x20 | 2 | 640 | 130.0M | 2.5 GB | OK |
| 16x16x20 | 4 | 80 | 129.7M | 2.5 GB | OK |
| 32x32x20 | 2 | 2,560 | 131.5M | 2.5 GB | OK |
| 32x32x20 | 4 | 320 | 129.9M | 2.5 GB | OK |
| 32x32x40 | 2 | 5,120 | 133.5M | 3.8 GB | OK |
| 32x32x40 | 4 | 640 | 130.1M | 2.5 GB | OK |
| 32x32x40 | 8 | 80 | 130.4M | 2.5 GB | OK |
| 64x64x40 | 2 | 20,480 | 145.3M | 12.2 GB | OK |
| 64x64x40 | 4 | 2,560 | 131.6M | 2.5 GB | OK |
| 64x64x40 | 8 | 320 | 130.5M | 2.5 GB | OK |
| 64x64x80 | 2 | 40,960 | 161.0M | 23.5 GB | OK |
| 64x64x80 | 4 | 5,120 | 133.5M | 3.8 GB | OK |
| 64x64x80 | 8 | 640 | 130.8M | 2.5 GB | OK |

### DiT-L (Latent)

| Resolution | Patch | Tokens | Params | Peak VRAM | Status |
|-----------|-------|--------|--------|-----------|--------|
| 16x16x10 | 2 | 320 | 457.1M | 8.6 GB | OK |
| 16x16x20 | 2 | 640 | 457.4M | 8.6 GB | OK |
| 16x16x20 | 4 | 80 | 457.0M | 8.6 GB | OK |
| 32x32x20 | 2 | 2,560 | 459.4M | 8.6 GB | OK |
| 32x32x20 | 4 | 320 | 457.2M | 8.6 GB | OK |
| 32x32x40 | 2 | 5,120 | 462.0M | 10.1 GB | OK |
| 32x32x40 | 4 | 640 | 457.6M | 8.6 GB | OK |
| 32x32x40 | 8 | 80 | 457.9M | 8.6 GB | OK |
| 64x64x40 | 2 | 20,480 | 477.8M | 32.4 GB | OK |
| 64x64x40 | 4 | 2,560 | 459.5M | 8.6 GB | OK |
| 64x64x40 | 8 | 320 | 458.1M | 8.6 GB | OK |
| **64x64x80** | **2** | **40,960** | **498.7M** | **62.1 GB** | **OK (17.9GB free)** |
| 64x64x80 | 4 | 5,120 | 462.1M | 10.1 GB | OK |
| 64x64x80 | 8 | 640 | 458.5M | 8.6 GB | OK |

### DiT-XL (Latent)

| Resolution | Patch | Tokens | Params | Peak VRAM | Status |
|-----------|-------|--------|--------|-----------|--------|
| 16x16x10 | 2 | 320 | 674.0M | 12.7 GB | OK |
| 16x16x20 | 2 | 640 | 674.4M | 12.7 GB | OK |
| 16x16x20 | 4 | 80 | 673.9M | 12.7 GB | OK |
| 32x32x20 | 2 | 2,560 | 676.6M | 12.7 GB | OK |
| 32x32x20 | 4 | 320 | 674.2M | 12.7 GB | OK |
| 32x32x40 | 2 | 5,120 | 679.6M | 13.7 GB | OK |
| 32x32x40 | 4 | 640 | 674.5M | 12.7 GB | OK |
| 32x32x40 | 8 | 80 | 674.9M | 12.7 GB | OK |
| **64x64x40** | **2** | **20,480** | **697.2M** | **43.0 GB** | **OK** |
| 64x64x40 | 4 | 2,560 | 676.7M | 12.7 GB | OK |
| 64x64x40 | 8 | 320 | 675.2M | 12.7 GB | OK |
| 64x64x80 | 2 | 40,960 | 720.8M | --- | **OOM** |
| 64x64x80 | 4 | 5,120 | 679.7M | 13.7 GB | OK |
| 64x64x80 | 8 | 640 | 675.6M | 12.7 GB | OK |

---

## Pixel Space DiT (no compression)

### DiT-S (Pixel)

| Resolution | Patch | Tokens | Params | Peak VRAM | Status |
|-----------|-------|--------|--------|-----------|--------|
| 64x64x40 | 2 | 20,480 | 40.3M | 6.0 GB | OK |
| 64x64x40 | 4 | 2,560 | 33.5M | 1.0 GB | OK |
| 64x64x40 | 8 | 320 | 33.0M | 0.7 GB | OK |
| 64x64x80 | 2 | 40,960 | 48.2M | 11.7 GB | OK |
| 64x64x80 | 4 | 5,120 | 34.5M | 1.7 GB | OK |
| 64x64x80 | 8 | 640 | 33.1M | 0.7 GB | OK |
| **128x128x80** | **2** | **163,840** | **95.4M** | **45.6 GB** | **OK** |
| 128x128x80 | 4 | 20,480 | 40.4M | 6.0 GB | OK |
| 128x128x80 | 8 | 2,560 | 33.8M | 1.0 GB | OK |
| 128x128x160 | 2 | 327,680 | 158.3M | --- | **OOM** |
| 128x128x160 | 4 | 40,960 | 48.2M | 11.7 GB | OK |
| 128x128x160 | 8 | 5,120 | 34.8M | 1.7 GB | OK |
| 256x256x160 | 4 | 163,840 | 95.4M | 45.7 GB | OK |
| 256x256x160 | 8 | 20,480 | 40.7M | 6.1 GB | OK |

### DiT-B (Pixel)

| Resolution | Patch | Tokens | Params | Peak VRAM | Status |
|-----------|-------|--------|--------|-----------|--------|
| 64x64x40 | 2 | 20,480 | 145.3M | 12.2 GB | OK |
| 64x64x40 | 4 | 2,560 | 131.6M | 2.5 GB | OK |
| 64x64x40 | 8 | 320 | 130.5M | 2.5 GB | OK |
| 64x64x80 | 2 | 40,960 | 161.0M | 23.5 GB | OK |
| 64x64x80 | 4 | 5,120 | 133.5M | 3.8 GB | OK |
| 64x64x80 | 8 | 640 | 130.8M | 2.5 GB | OK |
| 128x128x80 | 2 | 163,840 | 255.4M | --- | **OOM** |
| 128x128x80 | 4 | 20,480 | 145.3M | 12.2 GB | OK |
| 128x128x80 | 8 | 2,560 | 132.3M | 2.6 GB | OK |
| 128x128x160 | 2 | 327,680 | 381.2M | --- | **OOM** |
| 128x128x160 | 4 | 40,960 | 161.1M | 23.5 GB | OK |
| 128x128x160 | 8 | 5,120 | 134.2M | 3.8 GB | OK |
| 256x256x160 | 4 | 163,840 | 255.4M | --- | **OOM** |
| 256x256x160 | 8 | 20,480 | 146.0M | 12.3 GB | OK |

### DiT-L (Pixel)

| Resolution | Patch | Tokens | Params | Peak VRAM | Status |
|-----------|-------|--------|--------|-----------|--------|
| 64x64x40 | 2 | 20,480 | 477.8M | 32.4 GB | OK |
| 64x64x40 | 4 | 2,560 | 459.5M | 8.6 GB | OK |
| 64x64x40 | 8 | 320 | 458.1M | 8.6 GB | OK |
| **64x64x80** | **2** | **40,960** | **498.7M** | **62.1 GB** | **OK (17.9GB free)** |
| 64x64x80 | 4 | 5,120 | 462.1M | 10.1 GB | OK |
| 64x64x80 | 8 | 640 | 458.5M | 8.6 GB | OK |
| 128x128x80 | 2 | 163,840 | 624.6M | --- | **OOM** |
| 128x128x80 | 4 | 20,480 | 477.9M | 32.4 GB | OK |
| 128x128x80 | 8 | 2,560 | 460.4M | 8.6 GB | OK |
| 128x128x160 | 2 | 327,680 | 792.3M | --- | **OOM** |
| 128x128x160 | 4 | 40,960 | 498.8M | 62.1 GB | OK (17.9GB free) |
| 128x128x160 | 8 | 5,120 | 463.1M | 10.1 GB | OK |
| 256x256x160 | 4 | 163,840 | 624.7M | --- | **OOM** |
| 256x256x160 | 8 | 20,480 | 478.8M | 32.5 GB | OK |

### DiT-XL (Pixel)

| Resolution | Patch | Tokens | Params | Peak VRAM | Status |
|-----------|-------|--------|--------|-----------|--------|
| 64x64x40 | 2 | 20,480 | 697.2M | 43.0 GB | OK |
| 64x64x40 | 4 | 2,560 | 676.7M | 12.7 GB | OK |
| 64x64x40 | 8 | 320 | 675.2M | 12.7 GB | OK |
| 64x64x80 | 2 | 40,960 | 720.8M | --- | **OOM** |
| 64x64x80 | 4 | 5,120 | 679.7M | 13.7 GB | OK |
| 64x64x80 | 8 | 640 | 675.6M | 12.7 GB | OK |
| 128x128x80 | 2 | 163,840 | 862.4M | --- | **OOM** |
| 128x128x80 | 4 | 20,480 | 697.4M | 43.0 GB | OK |
| 128x128x80 | 8 | 2,560 | 677.8M | 12.7 GB | OK |
| 128x128x160 | 2 | 327,680 | 1051.1M | --- | **OOM** |
| 128x128x160 | 4 | 40,960 | 721.0M | --- | **OOM** |
| 128x128x160 | 8 | 5,120 | 680.7M | 13.8 GB | OK |
| 256x256x160 | 4 | 163,840 | 862.5M | --- | **OOM** |
| 256x256x160 | 8 | 20,480 | 698.4M | 43.1 GB | OK |

---

## Best Viable Config Per Variant

Maximum token count that fits on A100 80GB:

| Variant | Best Resolution | Patch | Tokens | Peak VRAM | Headroom |
|---------|----------------|-------|--------|-----------|----------|
| **S** | 128x128x80 (pixel) | 2 | 163,840 | 45.6 GB | 34.4 GB |
| **B** | 64x64x80 (latent) | 2 | 40,960 | 23.5 GB | 56.5 GB |
| **L** | 64x64x80 (both) | 2 | 40,960 | 62.1 GB | 17.9 GB |
| **XL** | 64x64x40 (latent) | 2 | 20,480 | 43.0 GB | 37.0 GB |

## Token Count OOM Boundaries

| Variant | Max viable tokens (patch=2) | OOM starts at |
|---------|---------------------------|---------------|
| S | 163,840 | 327,680 |
| B | 40,960 | 163,840 |
| L | 40,960 | 163,840 |
| XL | 20,480 | 40,960 |

## Key Insights

1. **VRAM scales O(n²) with token count**: Doubling tokens roughly quadruples attention VRAM. The patch embedding layer only adds linear cost.

2. **Patch size dominates VRAM more than variant**: DiT-XL at patch=4 (13.7GB) uses less VRAM than DiT-S at patch=2 with many tokens (45.6GB). Choose patch size first, then variant.

3. **DiT-S is uniquely scalable**: Can handle 163,840 tokens (patch=2 at 128x128x80) at 45.6GB. All other variants OOM at this token count.

4. **DiT-L at 64x64x80 is the sweet spot for latent diffusion**: 40,960 tokens, 62.1GB — tight but viable on A100 80GB with 17.9GB headroom.

5. **XL gains are marginal**: XL (675M) vs L (457M) is only +48% params but halves maximum viable resolution. L is likely the better tradeoff.

6. **Latent vs pixel space have identical VRAM for same resolution**: The token count is what matters, not whether the volume is compressed. A 64x64x80 latent has the same token count as a 64x64x80 pixel volume.

7. **For exp13 (4x VQ-VAE latent, bravo_seg_cond)**: Latent shape is ~64x64x40. DiT-S/B/L/XL all fit comfortably at patch=2. DiT-L at 20,480 tokens uses only 32.4GB.

---
---

# 3D UNet VRAM Profiling (Latent Diffusion)

Profiled on IDUN cluster, January 31, 2026.

## Hardware
- **GPU**: NVIDIA A100-SXM4-80GB
- **Driver**: 575.57.08
- **CUDA**: 12.9

## Test Configuration
- **Batch size**: 1
- **Precision**: AMP bfloat16
- **Input channels**: 8 (bravo_latent + seg_latent concatenated)
- **Output channels**: 4 (noise prediction for bravo_latent only)

---

## 4x Compression Latents

**Latent shape**: `(C=8, D=40, H=64, W=64)`

4x VQ-VAE compresses 160x256x256 volumes to 40x64x64 latents.

### Results

| Config | Channels | Params | Model | Fwd | Bwd | Total | Status |
|--------|----------|--------|-------|-----|-----|-------|--------|
| 4L_tiny | [32, 64, 128, 256] | 29.0M | 0.1G | 2.0G | 2.0G | 2.0G | OK |
| 4L_small | [64, 128, 256, 512] | 166.6M | 0.7G | 5.8G | 5.8G | 5.8G | OK |
| 4L_maisi | [64, 128, 256, 512] | 166.6M | 0.7G | 5.8G | 5.8G | 5.8G | OK |
| 4L_medium | [128, 256, 512, 1024] | 666.2M | 2.6G | 11.8G | 11.8G | 12.5G | OK |
| 4L_medium_d | [128, 256, 512, 1024] | 869.5M | 3.3G | 15.1G | 15.1G | 16.3G | OK |
| **4L_large** | [256, 512, 1024, 2048] | 2664.6M | 10.0G | 31.0G | 31.0G | **49.7G** | OK |
| 4L_large_d | [256, 512, 1024, 2048] | 3477.6M | 13.0G | 39.8G | 39.8G | 64.9G | >60GB |
| 3L_tiny | [32, 64, 128] | 7.3M | 0.1G | 18.3G | 23.0G | 23.0G | OK |
| 3L_small | [64, 128, 256] | 41.7M | 0.2G | 56.4G | 65.9G | 65.9G | >60GB |
| 3L_medium+ | - | - | - | - | - | OOM | FAIL |

### Recommendation for 4x

**4L_large** (4-level UNet, 2.66B params, 49.7GB VRAM)

```yaml
model.channels: [256, 512, 1024, 2048]
model.attention_levels: [false, false, true, true]
model.num_res_blocks: [2, 2, 2, 2]
model.num_head_channels: 64
```

### Why 4-level works for 4x

- Depth=40 is divisible by 16 (2^4), so 4 downsample levels work
- 4-level UNets have smaller activation maps at each level → lower VRAM
- 3-level UNets keep larger spatial dimensions → OOM for 40x64x64

---

## 8x Compression Latents

**Latent shape**: `(C=8, D=20, H=32, W=32)`

8x VQ-VAE compresses 160x256x256 volumes to 20x32x32 latents.

### Results

| Config | Channels | Params | Model | Fwd | Bwd | Total | Status |
|--------|----------|--------|-------|-----|-----|-------|--------|
| 3L_tiny | [32, 64, 128] | 7.3M | 0.1G | 0.5G | 0.5G | 0.5G | OK |
| 3L_small | [64, 128, 256] | 41.7M | 0.2G | 1.5G | 1.5G | 1.5G | OK |
| 3L_medium | [128, 256, 512] | 166.5M | 0.7G | 2.6G | 2.6G | 3.2G | OK |
| 3L_medium_d | [128, 256, 512] | 217.2M | 0.9G | 3.4G | 3.4G | 4.1G | OK |
| 3L_large | [256, 512, 1024] | 665.8M | 2.6G | 7.0G | 7.0G | 12.5G | OK |
| 3L_large_d | [256, 512, 1024] | 868.6M | 3.3G | 9.2G | 9.2G | 16.3G | OK |
| 3L_xlarge | [384, 768, 1536] | 1497.9M | 5.7G | 13.3G | 13.3G | 28.0G | OK |
| 3L_xlarge_d | [384, 768, 1536] | 1954.1M | 7.4G | 17.4G | 17.4G | 36.6G | OK |
| **3L_huge** | [512, 1024, 2048] | 2662.8M | 10.0G | 21.4G | 22.5G | **49.7G** | OK |
| 3L_huge_d | [512, 1024, 2048] | 3473.7M | 13.0G | 28.0G | 29.4G | 64.8G | >60GB |

### Recommendation for 8x

**3L_huge** (3-level UNet, 2.66B params, 49.7GB VRAM)

```yaml
model.channels: [512, 1024, 2048]
model.attention_levels: [false, true, true]
model.num_res_blocks: [2, 2, 2]
model.num_head_channels: 64
```

### Why 3-level required for 8x

- Depth=20 is NOT divisible by 16, so 4-level UNets fail with dimension mismatch
- Depth=20 IS divisible by 8 (2^3), so 3-level works
- Smaller latent volume (20x32x32 vs 40x64x64) allows larger channel widths

---

## Key Insights

1. **UNet levels vs spatial dimensions**:
   - 4-level UNet: needs dimensions divisible by 16 (3 downsamples = 2^4 = 16x reduction)
   - 3-level UNet: needs dimensions divisible by 8 (2 downsamples = 2^3 = 8x reduction)

2. **VRAM scaling**:
   - 3-level UNets have larger intermediate activations (fewer downsamples)
   - For same channel config, 3-level uses MORE VRAM than 4-level on larger latents

3. **Optimal configs achieve ~2.6B params at ~50GB VRAM**:
   - 4x: 4L_large with [256, 512, 1024, 2048]
   - 8x: 3L_huge with [512, 1024, 2048]

4. **The `_d` suffix = deeper residual blocks**:
   - Default: `num_res_blocks=[2, 2, 2, 2]`
   - Deep (`_d`): `num_res_blocks=[3, 3, 3, 3]`
   - Adds ~30% more parameters and ~30% more VRAM

---

## SLURM Configurations

### exp9_ldm_4x_bravo.slurm
```bash
'model.channels=[256, 512, 1024, 2048]'
'model.attention_levels=[false, false, true, true]'
'model.num_res_blocks=[2, 2, 2, 2]'
model.num_head_channels=64
```

### exp9_ldm_8x_bravo.slurm
```bash
'model.channels=[512, 1024, 2048]'
'model.attention_levels=[false, true, true]'
'model.num_res_blocks=[2, 2, 2]'
model.num_head_channels=64
```

---
---

# S2D 3D UNet VRAM Profiling

Profiled on IDUN cluster, February 8, 2026.

Applies to both Space-to-Depth (exp11/exp11_1) and Haar Wavelet (exp12/exp12_1) since both produce identical shapes.

## Hardware
- **GPU**: NVIDIA H100 80GB HBM3
- **Driver**: 575.57.08
- **CUDA**: 12.9 (PyTorch built with 12.8)
- **PyTorch**: 2.9.0

## Test Configuration
- **Batch size**: 1
- **Precision**: AMP bfloat16 + gradient checkpointing
- **Mode**: Bravo (16ch input = 8ch image + 8ch seg, 8ch output)
- **Configs tested**: 62 architectures x 2 resolutions = 124 runs

## Context: S2D/Wavelet vs Latent Diffusion

| | Latent (8x VQ-VAE) | S2D / Wavelet |
|---|---|---|
| **Input** | 160x256x256 | 160x128x128 or 160x256x256 |
| **Encoded** | [B,8,20,32,32] | [B,8,80,64,64] or [B,8,80,128,128] |
| **Compression** | Lossy (learned) | Lossless (rearrangement / frequency) |
| **Spatial grid** | 20x32x32 = 20K voxels | 80x64x64 = 328K or 80x128x128 = 1.3M voxels |

S2D/Wavelet encoded volumes are **16-64x larger** than 8x latents, so VRAM is dominated by activation maps, not model parameters.

---

## exp11 / exp12: S2D/Wavelet from 128x128x160

**Encoded shape**: `[1, 8, 80, 64, 64]` (bravo: model sees 16ch in, 8ch out)

### 3-Level UNets

All 3-level configs either OOM or require extreme VRAM at this resolution. The spatial grid at level 0 (64x64x80) is too large.

| Config | Params | VRAM | Status |
|--------|--------|------|--------|
| 3L [128,256,512] no-attn | 160M | 18.4G | OK |
| 3L [128,256,512] attn-L2 | 165M | 42.1G | OK but huge |
| 3L [128,512,1024] attn-L2 | - | 75.7G* | OOM |
| 3L [256,512,512] attn-L12 | - | 9.3G* | OOM |
| 3L [256,512,1024] attn-L2 | - | 71.9G* | OOM |
| 3L [256,512,1024] attn-L12 | - | 10.8G* | OOM |

**Conclusion**: 3-level is not viable for S2D at this resolution.

### 4-Level UNets

| Config | Params | VRAM | Status |
|--------|--------|------|--------|
| 4L [64,128,256,512] no-attn | 159M | 6.8G | OK |
| 4L [64,128,256,512] attn-L3 | 165M | 7.3G | OK |
| 4L [64,128,256,512] attn-L23 | 166M | 19.1G | OK |
| 4L [64,256,512,512] attn-L23 | 271M | 35.2G | OK |
| 4L [64,256,512,1024] attn-L23 | 657M | 38.0G | OK |
| 4L [128,256,512,512] attn-L23 | 280M | 38.5G | OK |
| 4L [128,256,512,1024] attn-L23 | 666M | 41.2G | OK |
| 4L [128,256,512,1024] attn-L3 | 661M | 17.5G | OK |
| 4L [128,256,512,1024] attn-ALL | - | 5.8G* | OOM |
| 4L [128,256,512,1024] r=2,2,3,3 attn-L23 | 858M | 52.2G | OK |

**Key finding**: Attention at L2 (32x32x40) explodes VRAM. Attention at L3 only is affordable.

### 5-Level UNets (Best for S2D)

5 levels give the deepest downsampling: 64x64x80 -> 32x32x40 -> 16x16x20 -> 8x8x10 -> 4x4x5.

| Config | Params | VRAM | Notes |
|--------|--------|------|-------|
| **5L [32,64,256,512,512]** | **270M** | **5.2G** | Baseline, very efficient |
| 5L [16,32,128,256,256] | 68M | 2.2G | Too small |
| 5L [16,32,128,256,512] | 164M | 3.2G | Too small |
| 5L [32,64,128,256,512] | 166M | 4.0G | Progressive |
| **5L [64,128,256,512,512]** | **276M** | **7.5G** | Wider early layers |
| **5L [32,64,256,512,1024]** | **655M** | **12.4G** | Wider deep layers |
| **5L [64,128,256,512,1024]** | **662M** | **12.5G** | Progressive wide |
| 5L [64,128,512,512,1024] | 740M | 14.0G | Wider both |
| 5L [128,128,256,512,512] | 282M | 10.8G | Wider L0 |
| 5L [32,64,256,1024,1024] | 983M | 18.5G | Very deep |
| 5L [64,128,512,1024,1024] | 1080M | 20.3G | Large |
| 5L [128,256,512,1024,1024] | 1104M | 20.7G | XLarge |

### 5-Level Attention Sweep

Attention at early levels (L0, L1) causes OOM. Attention at L3/L4 is essentially free. L2 attention doubles VRAM.

| Attention config | Base channels | VRAM | Delta vs no-attn |
|-----------------|---------------|------|-----------------|
| **none** | [32,64,256,512,1024] | 11.9G | baseline |
| L4 only | [32,64,256,512,1024] | 12.3G | +0.4G |
| L3 only | [32,64,256,512,1024] | 12.0G | +0.1G |
| **L3+L4** | [32,64,256,512,1024] | 12.4G | **+0.5G** |
| L2+L4 | [32,64,256,512,1024] | 20.2G | +8.3G |
| L2+L3+L4 | [32,64,256,512,1024] | 20.6G | +8.7G |
| L1+L2+L3+L4 | [32,64,256,512,1024] | OOM | - |
| ALL | [32,64,256,512,1024] | OOM | - |

| Attention config | Base channels | VRAM | Delta vs no-attn |
|-----------------|---------------|------|-----------------|
| **none** | [64,128,256,512,512] | 7.1G | baseline |
| L4 only | [64,128,256,512,512] | 7.1G | +0.0G |
| L3 only | [64,128,256,512,512] | 7.5G | +0.4G |
| **L3+L4** | [64,128,256,512,512] | 7.4G | **+0.3G** |
| L2+L3+L4 | [64,128,256,512,512] | 19.4G | +12.3G |
| ALL | [64,128,256,512,512] | OOM | - |

| Attention config | Base channels | VRAM | Delta vs no-attn |
|-----------------|---------------|------|-----------------|
| **none** | [64,128,256,512,1024] | 12.0G | baseline |
| L4 only | [64,128,256,512,1024] | 12.4G | +0.4G |
| **L3+L4** | [64,128,256,512,1024] | 12.5G | **+0.5G** |
| L2+L3+L4 | [64,128,256,512,1024] | 21.6G | +9.6G |
| ALL | [64,128,256,512,1024] | OOM | - |

**Rule of thumb**: Attention at L3+L4 adds <1 GB. Attention at L2 adds ~8-12 GB. Attention at L1 or L0 = OOM.

### Multi-head Attention Comparison

More heads = slightly less VRAM (more parallelism, smaller per-head dim).

| Config | heads=32 | heads=64 |
|--------|----------|----------|
| [32,64,256,512,1024] attn-L34 | 12.4G | 12.4G |
| [64,128,256,512,512] attn-L34 | 7.4G | 7.3G |
| [64,128,256,512,1024] attn-L34 | 12.5G | 12.5G |

No significant difference at this resolution.

### Deeper Residual Blocks

| Config | Default r | VRAM | Deeper r | VRAM | Delta |
|--------|-----------|------|----------|------|-------|
| [32,64,256,512,512] | r=2 | 5.2G | r=2,2,3,3,3 | 6.8G | +1.6G |
| [64,128,256,512,1024] | r=2 | 12.5G | r=1,1,2,3,3 | 16.1G | +3.6G |
| [32,64,256,512,1024] | r=2 | 12.4G | r=2,2,2,2,2 | 12.4G | +0.0G |

### WDM-style Configs (no attention at early levels)

| Config | Params | VRAM |
|--------|--------|------|
| WDM-128 [64,128,128,256,256] no-attn | 74M | 6.7G |
| WDM-128+attn34 [64,128,128,256,256] | 77M | 6.9G |
| WDM-256 [64,128,256,512,512] no-attn | 268M | 8.0G |
| WDM-256+attn34 [64,128,256,512,512] | 279M | 8.5G |
| WDM-256-wider [64,128,256,512,1024] no-attn | 639M | 12.1G |

---

## exp11_1 / exp12_1: S2D/Wavelet from 256x256x160

**Encoded shape**: `[1, 8, 80, 128, 128]` (bravo: model sees 16ch in, 8ch out)

4x the spatial voxels compared to exp11.

### 3-Level and 4-Level: Mostly OOM

| Config | VRAM | Status |
|--------|------|--------|
| ALL 3-level configs | - | OOM |
| 4L [64,128,256,512] no-attn | 25.4G | OK |
| 4L [64,128,256,512] attn-L3 | 31.4G | OK |
| All other 4L | - | OOM |

### 5-Level UNets

| Config | Params | VRAM | Notes |
|--------|--------|------|-------|
| 5L [16,32,128,256,256] | 68M | 9.6G | Too small |
| 5L [16,32,128,256,512] | 164M | 10.2G | |
| 5L [32,64,128,256,512] | 166M | 15.3G | Progressive |
| **5L [32,64,256,512,512]** | **270M** | **19.6G** | Baseline |
| **5L [32,64,256,512,1024]** | **655M** | **21.9G** | Wider deep |
| 5L [32,64,256,512,1024] r=2,2,2,2,2 | 656M | 23.7G | Uniform r2 |
| 5L [32,64,256,512,512] r=2,2,3,3,3 | 356M | 24.6G | Deep res |
| **5L [64,128,256,512,512]** | **276M** | **29.8G** | Wider early |
| **5L [64,128,256,512,1024]** | **662M** | **32.2G** | Progressive wide |
| 5L [32,64,256,1024,1024] | 983M | 30.0G | Very deep |
| 5L [64,128,512,512,1024] | 740M | 33.7G | Wider both |
| 5L [64,128,256,512,1024] r=1,1,2,3,3 | 853M | 35.8G | Deep wide |
| 5L [64,128,512,1024,1024] | 1080M | 41.9G | Large |
| 5L [128,128,256,512,512] | 282M | 42.9G | Wide L0 |
| 5L [128,256,512,1024,1024] | - | 47.7G* | OOM |
| 5L [64,128,512,1024,1024] r=2,2,3,3,3 | 1424M | 52.9G | Large deep |

### 5-Level Attention Sweep (256x256 input)

| Attention config | Base channels | VRAM | Delta vs no-attn |
|-----------------|---------------|------|-----------------|
| **none** | [32,64,256,512,1024] | 15.6G | baseline |
| L4 only | [32,64,256,512,1024] | 15.9G | +0.3G |
| L3 only | [32,64,256,512,1024] | 21.6G | +6.0G |
| **L3+L4** (h=32) | [32,64,256,512,1024] | 18.9G | **+3.3G** |
| **L3+L4** (h=64) | [32,64,256,512,1024] | 17.4G | **+1.8G** |
| L2+L4 | [32,64,256,512,1024] | OOM | - |

| Attention config | Base channels | VRAM | Delta vs no-attn |
|-----------------|---------------|------|-----------------|
| **none** | [64,128,256,512,1024] | 25.9G | baseline |
| L4 only | [64,128,256,512,1024] | 26.2G | +0.3G |
| **L3+L4** (h=32) | [64,128,256,512,1024] | 29.1G | **+3.2G** |
| **L3+L4** | [64,128,256,512,1024] | 32.2G | +6.3G |
| L2+L3+L4 | [64,128,256,512,1024] | OOM | - |

| Attention config | Base channels | VRAM | Delta vs no-attn |
|-----------------|---------------|------|-----------------|
| **none** | [64,128,256,512,512] | 23.7G | baseline |
| L4 only | [64,128,256,512,512] | 23.8G | +0.1G |
| L3 only | [64,128,256,512,512] | 29.7G | +6.0G |
| **L3+L4** (h=32) | [64,128,256,512,512] | 26.9G | **+3.2G** |
| **L3+L4** (h=64) | [64,128,256,512,512] | 25.4G | **+1.7G** |
| L2+L3+L4 | [64,128,256,512,512] | OOM | - |

**At 256x256**: L3 attention costs ~3-6 GB (vs <1 GB at 128x128). Using heads=64 saves ~1.5 GB vs heads=32. L2 attention = OOM.

### Multi-head Attention (256x256)

More heads makes a noticeable difference at this resolution.

| Config | heads=32 | heads=64 | Savings |
|--------|----------|----------|---------|
| [32,64,256,512,1024] attn-L34 | 18.9G | 17.4G | 1.5G |
| [64,128,256,512,512] attn-L34 | 26.9G | 25.4G | 1.5G |
| [64,128,256,512,1024] attn-L34 | 29.1G | - | - |

### WDM-style Configs (256x256)

| Config | Params | VRAM |
|--------|--------|------|
| WDM-128 [64,128,128,256,256] no-attn | 74M | 25.5G |
| WDM-128+attn34 [64,128,128,256,256] | 77M | 28.5G |
| WDM-256 [64,128,256,512,512] no-attn | 268M | 27.3G |
| WDM-256+attn34 [64,128,256,512,512] | 279M | 33.4G |
| WDM-256-wider [64,128,256,512,1024] no-attn | 639M | 29.4G |

---

## VRAM Scaling: 128x128 -> 256x256 (4x spatial voxels)

Scaling ratio depends heavily on where parameters sit in the network.

| Pattern | Ratio | Why |
|---------|-------|-----|
| Narrow early, wide deep [32,64,...,1024] | 1.3-1.8x | Most params at deep levels (small activations) |
| Balanced [64,128,256,512,512] | 3.3-4.0x | Params spread evenly, early levels dominate |
| Wide early [128,...] | 3.9-4.4x | L0 activations scale with full input resolution |
| Flat [256,256,...] | >4x / OOM | Early layers at full resolution = disaster |

**Key insight**: To scale to 256x256, keep early channels narrow (32-64) and put capacity in deep levels (512-1024). This gives ~1.5-2x VRAM scaling instead of ~4x.

---

## Key Insights

1. **5-level UNets are mandatory for S2D/Wavelet**: 3L and most 4L configs OOM. The deep downsampling (80->40->20->10->5) is needed to manage the large encoded volume.

2. **Attention is the VRAM bottleneck**: Attention at L2 (16x16x20 or 32x32x40) adds 8-12 GB. Attention at L0/L1 = instant OOM. Only L3+L4 attention is practical.

3. **VRAM scales with input resolution asymmetrically**: 128->256 is 4x voxels but only 1.3-1.8x VRAM for deep-heavy configs vs 3.5-4x for wide-early configs.

4. **heads=64 saves ~1.5 GB at 256x256**: Worth using at higher resolutions. No effect at 128x128.

5. **Sweet spots** (VRAM in isolated profiling, add ~50-60% for actual training with optimizer states + gradients):

   | Resolution | Config | Params | Profile VRAM | Est. Training |
   |-----------|--------|--------|-------------|---------------|
   | 128x128 | 5L [32,64,256,512,512] baseline | 270M | 5.2G | ~8-10G |
   | 128x128 | 5L [64,128,256,512,512] wider | 276M | 7.5G | ~12-15G |
   | 128x128 | 5L [32,64,256,512,1024] attn-L34 | 655M | 12.4G | ~20-25G |
   | 128x128 | 5L [64,128,256,512,1024] attn-L34 | 662M | 12.5G | ~20-25G |
   | 256x256 | 5L [32,64,256,512,1024] attn-L4 | 650M | 15.9G | ~25-30G |
   | 256x256 | 5L [32,64,256,512,1024] attn-L34 h=64 | 655M | 17.4G | ~28-35G |
   | 256x256 | 5L [32,64,256,512,512] baseline | 270M | 19.6G | ~30-35G |
   | 256x256 | 5L [32,64,256,512,1024] wider-deep | 655M | 21.9G | ~35-40G |
   | 256x256 | 5L [64,128,256,512,1024] attn-L34 h=32 | 662M | 29.1G | ~45-55G |

6. **Comparison with latent diffusion**: S2D/Wavelet encoded volumes (80x64x64 or 80x128x128) are 16-64x larger than 8x latents (20x32x32). This means S2D models max out at ~270M-662M params vs 2.6B for latent models at the same VRAM budget. The tradeoff is lossless encoding vs lossy compression.

---

## Configs Used in Experiments

### exp11 / exp12: S2D/Wavelet 128x128x160

Uses `model=default_3d_5lvl`:
```yaml
# 5-level, [32,64,256,512,512], attn L3+L4
# ~270M params, ~5.2G profiling / ~8-10G training
model.channels: [32, 64, 256, 512, 512]
model.attention_levels: [false, false, false, true, true]
model.num_res_blocks: 2
model.num_head_channels: 32
```

### exp11_1 / exp12_1: S2D/Wavelet 256x256x160

Uses `model=default_3d_5lvl` (same):
```yaml
# ~270M params, ~19.6G profiling / ~30-35G training
model.channels: [32, 64, 256, 512, 512]
model.attention_levels: [false, false, false, true, true]
model.num_res_blocks: 2
model.num_head_channels: 32
```

---
---

# HDiT / U-ViT 3D VRAM Profiling

Profiled on IDUN cluster, February 19, 2026.

## Hardware
- **GPU**: NVIDIA A100 80GB PCIe (79.3 GB usable)
- **Driver**: 575.57.08
- **CUDA**: 12.9

## Test Configuration
- **Batch size**: 1
- **Precision**: AMP bfloat16
- **Gradient checkpointing**: Enabled
- **Input channels**: 2 (bravo mode)
- **Resolution**: 256x256x160 (full pixel space)

## Architecture Overview

### U-ViT (Bao et al., CVPR 2023)
- **Flat attention**: every block attends over ALL tokens (no hierarchy)
- Skip connections between encoder/decoder halves (like UNet but in token space)
- Token-based conditioning (timestep as prepended token, no adaLN)
- Variants use paper-defined sizes (different from DiT variants)

### HDiT (Hierarchical DiT)
- **Multi-resolution**: TokenMerge (2x2x2) reduces tokens at each level
- U-shaped structure: encoder merges, decoder splits, skip connections between levels
- Uses DiT blocks with adaLN-Zero (reuses standard DiT code)
- Variants use DiT sizes (S=384, B=768, L=1024, XL=1152)

## Variant Sizes

| Arch | Variant | Hidden | Heads | Depth/Levels |
|------|---------|--------|-------|-------------|
| U-ViT | S | 512 | 8 | 13 blocks |
| U-ViT | S-Deep | 512 | 8 | 17 blocks |
| U-ViT | M | 768 | 12 | 17 blocks |
| U-ViT | L | 1024 | 16 | 21 blocks |
| HDiT | S | 384 | 6 | configurable level_depths |
| HDiT | B | 768 | 12 | configurable level_depths |
| HDiT | L | 1024 | 16 | configurable level_depths |
| HDiT | XL | 1152 | 16 | configurable level_depths |

## Token Counts at 256x256x160

| Patch | Grid | Total tokens |
|-------|------|-------------|
| 1 | 256x256x160 | 10,485,760 |
| 2 | 128x128x80 | 1,310,720 |
| 4 | 64x64x40 | 163,840 |
| 8 | 32x32x20 | 20,480 |

For HDiT, token merging (2x2x2 = 8x reduction per level) reduces count at deeper levels:
- patch=4, 2 merge levels: 163K -> 20K -> 2.5K
- patch=4, 3 merge levels: 163K -> 20K -> 2.5K -> 320
- patch=2, 2 merge levels: 1.3M -> 163K -> 20K
- patch=2, 3 merge levels: 1.3M -> 163K -> 20K -> 2.5K

---

## U-ViT Results

### U-ViT-S (hidden=512, depth=13, heads=8)

| Patch | Tokens | Params | Peak VRAM | Status |
|-------|--------|--------|-----------|--------|
| 1 | 10.5M | - | - | Skipped (too many tokens) |
| 2 | 1.3M | 715.6M | - | **OOM** |
| **4** | **164K** | **128.5M** | **10.3 GB** | **OK (69.7 GB free)** |
| **8** | **20K** | **56.0M** | **2.1 GB** | **OK (77.9 GB free)** |

### U-ViT-S-Deep (hidden=512, depth=17, heads=8)

| Patch | Tokens | Params | Peak VRAM | Status |
|-------|--------|--------|-----------|--------|
| 1 | 10.5M | - | - | Skipped |
| 2 | 1.3M | 729.3M | - | **OOM** |
| **4** | **164K** | **142.2M** | **12.3 GB** | **OK (67.7 GB free)** |
| **8** | **20K** | **69.7M** | **2.3 GB** | **OK (77.7 GB free)** |

### U-ViT-M (hidden=768, depth=17, heads=12)

| Patch | Tokens | Params | Peak VRAM | Status |
|-------|--------|--------|-----------|--------|
| 1 | 10.5M | - | - | Skipped |
| 2 | 1.3M | 1137.3M | - | **OOM** |
| **4** | **164K** | **256.7M** | **18.5 GB** | **OK (61.5 GB free)** |
| **8** | **20K** | **148.0M** | **3.2 GB** | **OK (76.8 GB free)** |

### U-ViT-L (hidden=1024, depth=21, heads=16)

| Patch | Tokens | Params | Peak VRAM | Status |
|-------|--------|--------|-----------|--------|
| 1 | 10.5M | - | - | Skipped |
| 2 | 1.3M | 1629.0M | - | **OOM** |
| **4** | **164K** | **454.8M** | **29.0 GB** | **OK (51.0 GB free)** |
| **8** | **20K** | **309.8M** | **6.0 GB** | **OK (74.0 GB free)** |

### U-ViT Summary

U-ViT at patch=2 (1.3M tokens) OOMs for all variants due to flat O(n^2) attention. Viable options are patch=4 and patch=8 only.

| Variant | patch=4 VRAM | patch=8 VRAM |
|---------|-------------|-------------|
| S | 10.3 GB | 2.1 GB |
| S-Deep | 12.3 GB | 2.3 GB |
| M | 18.5 GB | 3.2 GB |
| L | 29.0 GB | 6.0 GB |

All variants fit comfortably at both patch=4 and patch=8. U-ViT-L at patch=4 is the largest viable config (455M params, 29 GB).

---

## HDiT Results

### HDiT-S (hidden=384, heads=6)

| Patch | level_depths | Tok L0 | Tok BN | Params | Peak VRAM | Status |
|-------|-------------|--------|--------|--------|-----------|--------|
| 1 | [1,1,1,4,1,1,1] | 10.5M | 20K | - | - | Skipped |
| **2** | **[1,2,6,2,1]** | **1.3M** | **20K** | **611.9M** | **43.4 GB** | **OK (36.6 GB free)** |
| **2** | **[2,4,6,4,2]** | **1.3M** | **20K** | **627.9M** | **47.0 GB** | **OK (33.0 GB free)** |
| **2** | **[2,4,8,4,2]** | **1.3M** | **20K** | **633.2M** | **47.1 GB** | **OK (32.9 GB free)** |
| **2** | **[1,1,2,6,2,1,1]** | **1.3M** | **3K** | **620.9M** | **43.1 GB** | **OK (36.9 GB free)** |
| **2** | **[1,2,4,6,4,2,1]** | **1.3M** | **3K** | **636.8M** | **43.6 GB** | **OK (36.4 GB free)** |
| 4 | [1,2,6,2,1] | 164K | 3K | 109.6M | 5.8 GB | OK (74.2 GB free) |
| 4 | [2,4,6,4,2] | 164K | 3K | 125.6M | 6.3 GB | OK (73.7 GB free) |
| 4 | [2,6,8,6,2] | 164K | 3K | 141.6M | 6.4 GB | OK (73.6 GB free) |
| 4 | [4,6,8,6,4] | 164K | 3K | 152.2M | 7.2 GB | OK (72.8 GB free) |
| 4 | [1,2,4,6,4,2,1] | 164K | 320 | 133.7M | 5.9 GB | OK (74.1 GB free) |
| 4 | [2,4,4,6,4,4,2] | 164K | 320 | 149.7M | 6.4 GB | OK (73.6 GB free) |

### HDiT-B (hidden=768, heads=12)

| Patch | level_depths | Tok L0 | Tok BN | Params | Peak VRAM | Status |
|-------|-------------|--------|--------|--------|-----------|--------|
| 1 | all | 10.5M | 20K | - | - | Skipped |
| 2 | [1,2,6,2,1] | 1.3M | 20K | 1299.0M | - | **OOM** |
| 2 | [2,4,6,4,2] | 1.3M | 20K | 1362.7M | - | **OOM** |
| 2 | [2,4,8,4,2] | 1.3M | 20K | 1384.0M | - | **OOM** |
| 2 | [1,1,2,6,2,1,1] | 1.3M | 3K | 1332.8M | - | **OOM** |
| 2 | [1,2,4,6,4,2,1] | 1.3M | 3K | 1396.6M | - | **OOM** |
| **4** | **[1,2,6,2,1]** | **164K** | **3K** | **294.5M** | **11.6 GB** | **OK (68.4 GB free)** |
| **4** | **[2,4,6,4,2]** | **164K** | **3K** | **358.3M** | **12.7 GB** | **OK (67.3 GB free)** |
| **4** | **[2,6,8,6,2]** | **164K** | **3K** | **422.0M** | **13.2 GB** | **OK (66.8 GB free)** |
| **4** | **[4,6,8,6,4]** | **164K** | **3K** | **464.5M** | **14.7 GB** | **OK (65.3 GB free)** |
| **4** | **[1,2,4,6,4,2,1]** | **164K** | **320** | **390.4M** | **12.0 GB** | **OK (68.0 GB free)** |
| **4** | **[2,4,4,6,4,4,2]** | **164K** | **320** | **454.2M** | **13.1 GB** | **OK (66.9 GB free)** |

### HDiT-L (hidden=1024, heads=16)

| Patch | level_depths | Tok L0 | Tok BN | Params | Peak VRAM | Status |
|-------|-------------|--------|--------|--------|-----------|--------|
| 1 | all | 10.5M | 20K | - | - | Skipped |
| 2 | all configs | 1.3M | - | 1.8-2.0B | - | **OOM** |
| **4** | **[1,2,6,2,1]** | **164K** | **3K** | **459.5M** | **15.7 GB** | **OK (64.3 GB free)** |
| **4** | **[2,4,6,4,2]** | **164K** | **3K** | **572.8M** | **17.3 GB** | **OK (62.7 GB free)** |
| **4** | **[2,6,8,6,2]** | **164K** | **3K** | **686.2M** | **18.0 GB** | **OK (62.0 GB free)** |
| **4** | **[4,6,8,6,4]** | **164K** | **3K** | **761.7M** | **20.1 GB** | **OK (59.9 GB free)** |
| **4** | **[1,2,4,6,4,2,1]** | **164K** | **320** | **629.8M** | **16.4 GB** | **OK (63.6 GB free)** |
| **4** | **[2,4,4,6,4,4,2]** | **164K** | **320** | **743.2M** | **18.0 GB** | **OK (62.0 GB free)** |

### HDiT-XL (hidden=1152, heads=16)

| Patch | level_depths | Tok L0 | Tok BN | Params | Peak VRAM | Status |
|-------|-------------|--------|--------|--------|-----------|--------|
| 1 | all | 10.5M | 20K | - | - | Skipped |
| 2 | all configs | 1.3M | - | 2.1-2.3B | - | **OOM** |
| **4** | **[1,2,6,2,1]** | **164K** | **3K** | **554.5M** | **17.8 GB** | **OK (62.2 GB free)** |
| **4** | **[2,4,6,4,2]** | **164K** | **3K** | **698.0M** | **19.6 GB** | **OK (60.4 GB free)** |
| **4** | **[2,6,8,6,2]** | **164K** | **3K** | **841.4M** | **20.4 GB** | **OK (59.6 GB free)** |
| **4** | **[4,6,8,6,4]** | **164K** | **3K** | **937.0M** | **22.9 GB** | **OK (57.1 GB free)** |
| **4** | **[1,2,4,6,4,2,1]** | **164K** | **320** | **770.0M** | **18.6 GB** | **OK (61.4 GB free)** |
| **4** | **[2,4,4,6,4,4,2]** | **164K** | **320** | **913.5M** | **20.5 GB** | **OK (59.5 GB free)** |

---

## Best Viable Config Per (Arch, Variant)

| Arch | Variant | Patch | Config | Params | Peak VRAM | Free |
|------|---------|-------|--------|--------|-----------|------|
| U-ViT | S | 4 | depth=13 | 128.5M | 10.3 GB | 69.7 GB |
| U-ViT | S-Deep | 4 | depth=17 | 142.2M | 12.3 GB | 67.7 GB |
| U-ViT | M | 4 | depth=17 | 256.7M | 18.5 GB | 61.5 GB |
| U-ViT | L | 4 | depth=21 | 454.8M | 29.0 GB | 51.0 GB |
| HDiT | S | 2 | [1,2,4,6,4,2,1] | 636.8M | 43.6 GB | 36.4 GB |
| HDiT | B | 4 | [4,6,8,6,4] | 464.5M | 14.7 GB | 65.3 GB |
| HDiT | L | 4 | [4,6,8,6,4] | 761.7M | 20.1 GB | 59.9 GB |
| HDiT | XL | 4 | [4,6,8,6,4] | 937.0M | 22.9 GB | 57.1 GB |

---

## Key Insights

1. **HDiT unlocks patch=2 at 256x256x160**: HDiT-S with patch=2 fits at 43-47 GB thanks to token merging. Flat DiT/U-ViT OOM at patch=2 (1.3M tokens with quadratic attention). This is HDiT's primary advantage.

2. **patch=2 is expensive even with HDiT**: Only HDiT-S fits at patch=2 (43-47 GB). HDiT-B and above OOM — the level-0 blocks at 1.3M tokens dominate VRAM even with just 1-2 blocks there.

3. **At patch=4, HDiT offers massive scaling headroom**: HDiT-XL [4,6,8,6,4] = 937M params at only 22.9 GB. This is 4x more parameters than the current exp16 (HDiT-S [2,4,6,4,2] = 126M) while still leaving 57 GB free.

4. **U-ViT is competitive at patch=4 and patch=8**: U-ViT-L at patch=4 = 455M params at 29 GB. Similar parameter count to HDiT-L [1,2,6,2,1] at 16 GB but more VRAM (flat attention over 164K tokens vs hierarchical). U-ViT's skip connections may provide quality benefits despite the efficiency gap.

5. **More merge levels barely save VRAM at patch=4**: HDiT-S p4 [2,4,6,4,2] (2 levels) = 6.3 GB vs [2,4,4,6,4,4,2] (3 levels) = 6.4 GB. The third merge level adds complexity but tokens are already low enough at level 1 that the extra merging doesn't help.

6. **3 merge levels matter at patch=2**: HDiT-S p2 [2,4,6,4,2] = 47.0 GB vs [1,1,2,6,2,1,1] = 43.1 GB. The extra level reduces the token count at level 1 from 20K to 3K, saving ~4 GB.

7. **VRAM is dominated by level-0 blocks**: For HDiT-S patch=2, going from [1,2,6,2,1] (1 L0 block) to [2,4,6,4,2] (2 L0 blocks) jumps from 43.4 GB to 47.0 GB (+3.6 GB). Adding 2 more bottleneck blocks (8 vs 6) only adds 0.1 GB.

8. **Comparison with flat DiT at 256x256x160**:
   - DiT-B patch=8: 146M params, 12.3 GB (20K tokens)
   - HDiT-B patch=4 [2,4,6,4,2]: 358M params, 12.7 GB (164K L0, but only 2 blocks there)
   - HDiT gets 2.5x more parameters at the same VRAM by doing most compute at reduced token counts

9. **Comparison with 3D UNet at 256x256x160**: The pixel-space UNet profiling showed that only a crippled [16,32,64,256,512,512] config fits (36.5 GB). HDiT-XL [4,6,8,6,4] has 937M params at 22.9 GB — more capable and cheaper.
