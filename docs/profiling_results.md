# 3D UNet VRAM Profiling Results

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

# S2D 3D UNet VRAM Profiling Results

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
