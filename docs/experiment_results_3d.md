# 3D Diffusion Experiment Results

Last updated: March 13, 2026. Data extracted from IDUN logs and TensorBoard runs.

---

## Quick Reference: Best Results

| Category | Experiment | Val Loss | FID (latest) | Epochs | Notes |
|----------|-----------|----------|-------------|--------|-------|
| **Bravo pixel 256 (gen)** | exp1_1 (1000ep) | 0.00230 | **19.12** | 1000 | Post-hoc eval, 27 Euler steps |
| **Bravo pixel 256 (loss)** | exp1l_1 (adj.offset) | **0.00209** | 72.52 | 500 | Best val loss |
| **Bravo pixel 256 (FID)** | exp1o_1 (PosthocEMA) | 0.00235 | **62.64** | 500 | Best in-training FID |
| **Bravo 128 ControlNet** | exp6b (Stage 2) | 0.00304 | **49.62** | 500 | Best seg-conditioned gen |
| **Bravo pixel 128** | exp8 (EMA) | 0.00227 | N/A | 500 | Best 128x128 |
| **Seg pixel 256** | exp2b_1 (input) | 0.000336 | N/A | 500 | Best seg overall |
| **LDM 4x** | exp9_1 (mid UNet) | 0.0764 | N/A | 354 | Hit time limit |

---

## Experiment Index

| ID | Type | Resolution | Architecture | Params | Strategy | Mode |
|----|------|-----------|-------------|--------|----------|------|
| exp1 | Pixel | 128x128x160 | UNet 5L | 270M | rflow | bravo |
| exp1_1 | Pixel | 256x256x160 | UNet 5L | 270M | rflow | bravo |
| exp1_chained | Pixel | 128x128x160 | UNet 5L | 270M | rflow | bravo |
| exp1c/1c_1 | Pixel+BrainNorm | 128/256x160 | UNet 5L | 270M | rflow | bravo |
| exp1e/1e_1 | Pixel+MinSNR | 128/256x160 | UNet 5L | 270M | rflow | bravo |
| exp1f/1f_1 | Pixel+EDMPrecond | 128/256x160 | UNet 5L | 270M | rflow | bravo |
| exp1g/1g_1 | Pixel+PseudoHuber | 128/256x160 | UNet 5L | 270M | rflow | bravo |
| exp1h/1h_1 | Pixel+LPIPSHuber | 128/256x160 | UNet 5L | 270M | rflow | bravo |
| exp1i/1i_1 | Pixel+ScoreAug | 128/256x160 | UNet 5L | 270M | rflow | bravo |
| exp1j/1j_1 | Pixel+GradAccum | 128/256x160 | UNet 5L | 270M | rflow | bravo |
| exp1k/1k_1 | Pixel+OffsetNoise | 128/256x160 | UNet 5L | 270M | rflow | bravo |
| exp1l/1l_1 | Pixel+AdjOffset | 128/256x160 | UNet 5L | 270M | rflow | bravo |
| exp1m/1m_1 | Pixel+GlobalNorm | 128/256x160 | UNet 5L | 270M | rflow | bravo |
| exp1o/1o_1 | Pixel+PosthocEMA | 128/256x160 | UNet 5L | 270M | rflow | bravo |
| exp1p_1 | Pixel+UniformT | 256x256x160 | UNet 5L | 270M | rflow | bravo |
| exp1n | Pixel+CFGZero* | 256x256x160 | UNet 5L | 270M | rflow | bravo |
| exp1r | Pixel+AttnDropout | 256x256x160 | UNet 5L | 270M | rflow | bravo |
| exp1s | Pixel+WeightDecay | 256x256x160 | UNet 5L | 270M | rflow | bravo |
| exp6a_1 | ControlNet S1 | 256x256x160 | UNet 5L | 270M | rflow | bravo (uncond) |
| exp6b | ControlNet S2 | 128x128x160 | UNet 5L | 270M | rflow | bravo |
| exp6b_1 | ControlNet S2 | 256x256x160 | UNet 5L | 270M | rflow | bravo |
| exp2 | Pixel | 128x128x160 | UNet 5L | 270M | rflow | seg_conditioned |
| exp2_1 | Pixel | 256x256x160 | UNet 5L | 270M | rflow | seg_conditioned |
| exp2b | Pixel | 128x128x160 | UNet 5L | 270M | rflow | seg_cond_input |
| exp2b_1 | Pixel | 256x256x160 | UNet 5L | 270M | rflow | seg_cond_input |
| exp2c | Pixel | 128x128x160 | UNet 5L | 271M | rflow | seg_cond_improved |
| exp2c_1 | Pixel | 256x256x160 | UNet 5L | 271M | rflow | seg_cond_improved |
| exp4 | Pixel+SDA | 128x128x160 | UNet 5L | 270M | rflow | bravo |
| exp5_1 | Pixel+ScoreAug | 128x128x160 | UNet 5L | 270M | rflow | bravo |
| exp6a | Pixel+ControlNet | 128x128x160 | UNet 5L | 270M | rflow | bravo (uncond) |
| exp7 | Pixel DiT | 128x128x160 | DiT-B p=8 | 135M | rflow | bravo |
| exp8 | Pixel+EMA | 128x128x160 | UNet 5L | 270M | rflow | bravo |
| exp9 | LDM 4x | latent 64x64x40 | UNet 4L | 3.48B | rflow | bravo_seg_cond |
| exp9_0 | LDM 8x small | latent 20x32x32 | UNet 3L | 167M | rflow | bravo_seg_cond |
| exp9_1 | LDM 4x mid | latent 64x64x40 | UNet 4L | 666M | rflow | bravo_seg_cond |
| exp10_1 | LDM DC-AE 8x8 | latent 8x8x32 | SiT-B p=1 | 137M | rflow | bravo_seg_cond |
| exp10_2 | LDM DC-AE 4x4 | latent 4x4x128 | SiT-B/DiT-B | 132M | rflow | bravo_seg_cond |
| exp10_3 | LDM DC-AE 2x2 | latent 2x2x512 | SiT-B/DiT-B | 131M | rflow | bravo_seg_cond |
| exp11 | S2D pixel | 128x128x160 | UNet 5L | 1.42B | rflow | bravo |
| exp11_1 | S2D pixel | 256x256x160 | UNet 5L | 1.42B | rflow | bravo |
| exp12 | Wavelet pixel | 128x128x160 | UNet 5L | 1.42B | rflow | bravo |
| exp12_1 | Wavelet pixel | 256x256x160 | UNet 5L | 1.42B | rflow | bravo |
| exp12_2 | Wavelet WDM | 128x128x160 | WDM 5L | ~77M | ddpm x0 | bravo |
| exp12_3 | Wavelet WDM | 128x128x160 | WDM 5L+attn | ~77M | rflow | bravo |
| exp12_4 | Wavelet WDM | 128x128x160 | WDM 5L+attn | ~77M | ddpm x0 | bravo |
| exp13 | LDM DiT 4x | latent 64x64x40 | DiT-S p=2 | 40M | rflow | bravo_seg_cond |
| exp14_1 | Pixel | 256x256x160 | UNet 5L | 270M | rflow | seg (unconditional) |
| exp23 | Pixel+ScoreAug | 256x256x160 | UNet 5L | 270M | rflow | bravo |

---

## Part 1: Pixel-Space Bravo Experiments

### exp1: Pixel Bravo Baseline @ 128x128x160

**Config**: UNet 5L [32,64,256,512,512], rflow, LR=1e-4, batch=1, 500 epochs

| Run | Job | GPU | Epochs | Best Val | Final MSE | Time/ep | Total | VRAM |
|-----|-----|-----|--------|----------|-----------|---------|-------|------|
| Run 1 | 23969657 | A100-SXM4 | 500 | 0.0026 | 0.0012 | 89s | 12.7h | N/A |
| Run 2 | 23989011 | A100-SXM4 | 500 | 0.00246 | 0.00146 | 70s | 10.0h | 4.1/19.6GB |

**Run 1 test results** (best model): MSE=0.003141, PSNR=32.49, MS-SSIM=0.8196
**Run 1 test results** (latest): MSE=0.007305, PSNR=30.45, MS-SSIM=0.8730
**Run 2 test**: Failed (torch.load OmegaConf bug)

**Convergence**: Healthy. MSE drops from ~0.97 to ~0.001 by epoch 500. Val loss improves steadily from 0.92 to 0.0025.

---

### exp1_1: Pixel Bravo @ 256x256x160

**Config**: UNet 5L [32,64,256,512,512], rflow, LR=1e-4, batch=1, 500 epochs

| Run | Job | GPU | Epochs | Best Val | Final MSE | Time/ep | Total | VRAM |
|-----|-----|-----|--------|----------|-----------|---------|-------|------|
| Run 1 | 23969658 | A100-SXM4 | 500 | 0.0022 | 0.0029 | 284s | 41h | N/A |
| Run 2 | 23989010 | A100-SXM4 | 500 | 0.00211 | 0.00284 | 224s | 31.4h | 3.6/43.0GB |

**Run 1 test results** (best): MSE=0.003494, PSNR=32.81, MS-SSIM=0.9090
**Run 1 test results** (latest): MSE=0.002904, PSNR=32.41, MS-SSIM=0.8897
**Run 2 test**: Failed (torch.load OmegaConf bug). OOM during generation metrics every epoch.

**256x256 is better than 128x128**: Lower val loss (0.0021 vs 0.0025), higher MS-SSIM (0.909 vs 0.820).

---

### exp1_chained: Continuation of exp1 @ 128x128x160

**Config**: Same as exp1, 100 extra epochs, auto-chained (30min segments)
- 14 chain segments, reached epoch 100/100
- Best val loss: 0.00300 (no improvement over original exp1's 0.0026)
- 1 gradient spike across all segments
- Test: Failed (no 3D bravo test loader)

---

### exp4: Pixel Bravo + SDA @ 128x128x160

**Config**: UNet 5L, rflow, LR=1e-4 + SDA (rotation+flip, noise_shift=0.1, prob=0.5)

| Job | GPU | Epochs | Best Val | Final MSE | Time/ep | Total | VRAM |
|-----|-----|--------|----------|-----------|---------|-------|------|
| 23989012 | A100-SXM4 | 500 | 0.00245 | 0.00208 | 122s | 17.6h | 5.1/22.6GB |

Healthy convergence but ~70% slower per epoch than baseline (SDA overhead). Slightly worse val loss than baseline.

---

### exp5_1: Pixel Bravo + ScoreAug @ 128x128x160

**Config**: UNet 5L, rflow, LR=1e-4 + ScoreAug (rotation+translation+cutout, omega_conditioning, no EMA, no min_snr)

| Job | GPU | Epochs | Best Val | Final MSE | Time/ep | Total | VRAM |
|-----|-----|--------|----------|-----------|---------|-------|------|
| 23989013 | A100-PCIe | 500 | 0.00213 | 0.00269 | 104s | 14.8h | 5.1/19.6GB |

**Second best 128x128 val loss** (0.00213), close to 256x256 baseline. ScoreAug helps.

---

### exp6a: Pixel Bravo ControlNet Stage 1 (Unconditional) @ 128x128x160

**Config**: UNet 5L, rflow, LR=1e-4, in_channels=1, out_channels=1 (unconditional)

| Job | GPU | Epochs | Best Val | Final MSE | Time/ep | Total | VRAM |
|-----|-----|--------|----------|-----------|---------|-------|------|
| 23989014 | A100-PCIe | 500 | 0.00234 | 0.00186 | 77s | 10.7h | 3.3/19.5GB |

Converged well. Generation metrics broken (channel mismatch in pipeline).

---

### exp7: DiT-B Pixel Bravo @ 128x128x160

**Config**: DiT-B (768 hidden, 12 depth, 12 heads), patch=8, 5120 tokens, rflow, LR=1e-4

| Job | Epochs | Best Val | Final MSE | Time/ep | Total | VRAM |
|-----|--------|----------|-----------|---------|-------|------|
| 23989812 | 2000 | 0.00234 | 0.00312 | 34s | 19.1h | 2.5/6.1GB |
| 23989813 | 500 | 0.00289 | 0.00466 | 33s | 4.8h | 2.5/6.1GB |

**Key finding**: DiT-B at patch=8 needs ~2000 epochs to match UNet at 500 epochs. At 500 epochs, val loss 0.00289 is worse than all UNet variants. But per-epoch cost is 34s vs 70-284s, and VRAM is 6GB vs 19-43GB.

Test eval failed (3D MS-SSIM/LPIPS dimension bug).

---

### exp8: Pixel Bravo + EMA @ 128x128x160

**Config**: UNet 5L, rflow, LR=1e-4 + EMA (decay=0.9999, update_after=100, update_every=10)

| Job | GPU | Epochs | Best Val | Final MSE | Time/ep | Total | VRAM |
|-----|-----|--------|----------|-----------|---------|-------|------|
| 23991162 | A100-PCIe | 500 | **0.00227** | **0.00129** | 77s | 11.1h | 5.1/20.6GB |

**Best 128x128 experiment overall.** EMA converges slower (54 best checkpoints vs ~25) but reaches the lowest final MSE and competitive val loss. Test failed (torch.load bug).

---

### Bravo Summary: Val Loss Ranking

| Rank | Experiment | Val Loss | Resolution |
|------|-----------|----------|------------|
| 1 | exp1_1 (run2) | 0.00211 | 256x256 |
| 2 | exp5_1 ScoreAug | 0.00213 | 128x128 |
| 3 | exp1_1 (run1) | 0.0022 | 256x256 |
| 4 | exp8 EMA | 0.00227 | 128x128 |
| 5 | exp7 DiT 2000ep | 0.00234 | 128x128 |
| 6 | exp6a ControlNet S1 | 0.00234 | 128x128 |
| 7 | exp4 SDA | 0.00245 | 128x128 |
| 8 | exp1 (run2) | 0.00246 | 128x128 |
| 9 | exp1 (run1) | 0.0026 | 128x128 |
| 10 | exp7 DiT 500ep | 0.00289 | 128x128 |

---

## Part 2: Pixel-Space Seg-Conditioned Experiments

### Stability Issue: 128x128 + LR=1e-4 Diverges

Three of four 128x128 seg_conditioned runs at LR=1e-4 diverged at epoch 34-37:
- exp2 (run2, 23996720): Diverged ep35, best before divergence: 0.001445
- exp2b (23996776): Diverged ep37, best before divergence: 0.001358
- exp2c (24031953): Diverged ep35, best before divergence: 0.747 (only 2 ckpts saved)

Pattern: MSE drops to ~0.0015, then jumps to ~0.6-1.0 in one epoch. Never recovers.

### exp2: Seg Size-Bin Conditioning

| Run | Res | LR | Job | Epochs | Best Val | Final MSE | Status |
|-----|-----|-----|-----|--------|----------|-----------|--------|
| run1 | 128 | 1e-4 | 23972121 | 500 | 0.000373 | 0.000351 | OK |
| run2 | 128 | 1e-4 | 23996720 | 443 | 0.001445 | 1.000 | **DIVERGED ep35** |
| exp2_1 | 256 | 5e-5 | 23996728 | 500 | **0.000337** | 0.000292 | OK |

### exp2b: Seg Input Channel Conditioning

| Run | Res | LR | Job | Epochs | Best Val | Final MSE | Status |
|-----|-----|-----|-----|--------|----------|-----------|--------|
| exp2b | 128 | 1e-4 | 23996776 | 500 | 0.001358 | 1.000 | **DIVERGED ep37** |
| exp2b_1 | 256 | 5e-5 | 23997358 | 500 | **0.000336** | 0.000289 | OK |

### exp2c: Improved Seg Conditioning

| Run | Res | LR | Job | Epochs | Best Val | Final MSE | Status |
|-----|-----|-----|-----|--------|----------|-----------|--------|
| exp2c | 128 | 1e-4 | 24031953 | 500 | 0.747 | 1.001 | **DIVERGED ep35** + ckpt fail |
| exp2c_1 | 256 | 1e-4 | 24039864 | 25 | N/A | 0.00163 | **No ckpts saved** (2GB limit) |

### Seg Summary

- **256x256 + LR=5e-5 is stable**: exp2_1 and exp2b_1 both converge to ~0.000337 val loss
- **128x128 + LR=1e-4 is unstable**: 3/4 runs diverge at epoch ~35
- **seg_conditioned vs seg_conditioned_input**: Identical results at 256x256 (0.000337 vs 0.000336)
- **exp2c_1 has LR=1e-4 at 256x256**: May diverge if run longer. Also has 2GB checkpoint limit problem.

---

## Part 3: Latent Diffusion (LDM) Experiments

### exp9: VQ-VAE 4x Latent, Large UNet (3.48B params)

**Config**: UNet 4L [256,512,1024,2048], res_blocks=[3,3,3,3], attn=[F,F,T,T], bravo_seg_cond, VQ-VAE 4x compression

Latent shape: [B, 8, 40, 64, 64] (in=8 bravo+seg, out=4 bravo velocity)

| Run | Job | GPU | Epochs | Best Val | Final MSE | Grad Spikes | VRAM | Status |
|-----|-----|-----|--------|----------|-----------|-------------|------|--------|
| run1 | 23997507 | A100-SXM4 | 500 | 0.0950 | 1.032 | 0 | 53/68GB | Completed, collapsed late |
| run2 | 24039868 | H100 | 302 | 0.0875 | 0.880 | **26,387** | 53/67GB | CANCELLED |

**Run 1**: Clean run, no gradient spikes. MSE dropped to ~0.08 early then rose to ~1.03. Best val loss 0.095.
**Run 2**: Gradient spikes started at epoch ~45 and **never stopped**. 26K out of ~31K total steps were skipped. Model frozen.

### exp9_1: VQ-VAE 4x Latent, Mid UNet (666M params)

**Config**: UNet 4L [128,256,512,1024], res_blocks=[2,2,2,2], attn=[F,F,T,T], warmup 10 epochs, grad clip 0.5

| Job | GPU | Epochs | Best Val | Final MSE | Grad Spikes | VRAM | Status |
|-----|-----|--------|----------|-----------|-------------|------|--------|
| 24036417 | A100-PCIe | 354 | **0.0764** | **0.025** | 0 | 11/65GB | **TIME LIMIT** |

**BEST non-pixel experiment.** Zero gradient spikes. MSE still decreasing at epoch 354 (0.025). Hit 14h SLURM time limit. Needs longer wall time to reach 500 epochs.

Key differences from large UNet: warmup (10 epochs), gradient clipping (0.5), smaller model (666M vs 3.48B).

### exp9: VQ-VAE 8x Latent, Large UNet (3.47B params)

**Config**: UNet 3L [512,1024,2048], res_blocks=[3,3,3], attn=[F,T,T], bravo_seg_cond, VQ-VAE 8x compression

Latent shape: [B, 8, 20, 32, 32]

| Run | Job | GPU | Epochs | Best Val | Final MSE | Grad Spikes | VRAM | Status |
|-----|-----|-----|--------|----------|-----------|-------------|------|--------|
| run1 | 23997506 | A100-SXM4 | 500 | 0.177 | 1.009 | 0 | 53/68GB | Completed, collapsed late |
| run2 | 24039867 | A100-SXM4 | 335 | 0.185 | 0.935 | **31,363** | 53/71GB | CANCELLED |

**8x compression is worse than 4x**: Best val loss 0.177 vs 0.095. More lossy compression = lower quality ceiling.

### exp9_0: VQ-VAE 8x Latent, Small UNet (167M params)

**Config**: UNet 3L [128,256,512], res_blocks=[2,2,2], attn=[F,T,T], 100 epoch quick test

| Run | Job | GPU | Epochs | Best Val | Final MSE | VRAM | Status |
|-----|-----|-----|--------|----------|-----------|------|--------|
| run1 | 23997680 | H100 | 100 | 0.198 | 0.141 | 3.8/8.1GB | OK |
| run2 | 23997808 | A100-SXM4 | 100 | 0.192 | 0.141 | 3.8/18.9GB | OK |

Quick test, clean runs. Small model at 8x still at 0.14 MSE after 100 epochs.

### LDM Pattern: Large UNets Collapse, Mid-size Works

The 3.48B param UNets (run1 of exp9 4x and 8x) completed 500 epochs but their training MSE collapsed to ~1.0, meaning the model degenerated to predicting the mean. Run2 of both had catastrophic gradient spikes.

**exp9_1 (666M mid UNet) is the exception**: zero gradient spikes, MSE still decreasing at 0.025. The key differences were warmup + gradient clipping + smaller model.

---

## Part 4: DiT/SiT Latent Experiments

### exp10_1: SiT-B + DC-AE 8x8x32

**Config**: SiT-B (768 hidden, 12 layers), latent 8×8×32, patch=1, 512 tokens

| Job | Epochs | Best Val | Final MSE | VRAM | Status |
|-----|--------|----------|-----------|------|--------|
| 23998878 | 500 | 3.146 | 1.68 | 5.9/70.1GB | Completed but poor |

Initial MSE was 31,387 (DC-AE latent magnitudes are enormous). Improved dramatically but val loss never went below 3.1. The DC-AE 8x8x32 latent space is likely too compressed (32 channels at 8x8 spatial) for meaningful generation.

### exp10_2: SiT-B/DiT-B + DC-AE 4x4x128 — FAILED

Both SiT-B (23998879) and DiT-B (24039865) crashed instantly. Architecture mismatch: 128-channel latent at 4x4 spatial doesn't work with the model's patch embedding.

### exp10_3: SiT-B/DiT-B + DC-AE 2x2x512 — FAILED

Same instant crash. 512-channel latent at 2x2 spatial is incompatible.

### exp13: DiT-S + VQ-VAE 4x (NEW, not yet run)

**Config**: DiT-S (384 hidden, 12 layers, 6 heads), patch=2, 20,480 tokens, use_compile=true, bravo_seg_cond

Previous attempt with patch=1 (163,840 tokens) was 40min/epoch, impractical. Now fixed to patch=2.

---

## Part 5: S2D and Wavelet Experiments

### Critical Failure: Mean Prediction Collapse

**ALL S2D and wavelet experiments collapsed to MSE ~1.0** (predicting the mean), regardless of resolution (128 or 256), model size, or GPU. This happened in every completed run.

The 1.42B param UNet 5L model (r=2,2,3,3,3 with [64,128,512,1024,1024]) is operating in the 8-channel wavelet/S2D encoded space [B,8,80,64,64] or [B,8,80,128,128].

### exp11: S2D @ 128x128x160

| Run | Job | GPU | Epochs | Best Val | Final MSE | Status |
|-----|-----|-----|--------|----------|-----------|--------|
| run1 | 24031954 | A100-PCIe | 500 | 0.905 | 1.006 | Ckpt corruption, mean collapse |
| run2 | 24039874 | A100-40GB | 4 | N/A | 0.291 | **OOM killed** |

### exp11_1: S2D @ 256x256x160

| Run | Job | GPU | Epochs | Best Val | Final MSE | Status |
|-----|-----|-----|--------|----------|-----------|--------|
| run1 | 24031955 | A100-SXM4 | 64 | 0.909 | 1.001 | Disk quota crash, mean collapse |
| run2 | 24039873 | H100 | 3 | 0.644 | 0.490 | **OOM killed** |

### exp12: Wavelet @ 128x128x160

| Run | Job | GPU | Epochs | Best Val | Final MSE | Status |
|-----|-----|-----|--------|----------|-----------|--------|
| run1 | 24031956 | A100-40GB | 49 | 0.905 | 1.001 | Disk quota crash, mean collapse |
| run2 | 24039876 | A100-40GB | 10 | N/A | 1.002 | **OOM killed** |

### exp12_1: Wavelet @ 256x256x160

| Run | Job | GPU | Epochs | Best Val | Final MSE | Status |
|-----|-----|-----|--------|----------|-----------|--------|
| run1 | 24031957 | A100-SXM4 | 500 | 1.001 | 1.006 | 738 ckpt failures, mean collapse |
| run2 | 24039875 | H100 | 210 | 0.186 | 1.022 | **OOM killed**, 571 grad spikes |

### Why S2D/Wavelet Failed

1. **Learning rate too high**: LR=1e-4 with 1.42B params in 8-channel space. The pixel baseline (270M params, 1-2 channel) uses the same LR.
2. **No warmup or gradient clipping**: exp9_1 (the best LDM run) used warmup=10 + grad_clip=0.5. S2D/wavelet had neither.
3. **Value range mismatch**: Wavelet subbands have very different magnitude distributions (LLL band vs HHH band). MSE on raw wavelet coefficients may not be appropriate.
4. **Mean prediction collapse**: Once MSE hits ~1.0, the model outputs near-zero predictions for all subbands. Recovery is impossible without reducing LR.

### exp12_2/12_3/12_4: WDM-style Wavelet (Not yet run)

These use the smaller WDM model (~77M params) with DDPM x0 prediction or rflow + attention. Not yet submitted.

---

## Part 6: Infrastructure Issues

### Checkpoint Corruption (Feb 10 runs)

All 24031xxx jobs suffered from `inline_container.cc:664` filesystem errors during checkpoint saves. Total: ~1,768 combined failures. Root cause: cluster filesystem not flushing properly for large files. Affected exp11, exp11_1, exp12, exp12_1, exp2c.

### 2GB File Size Limit (Feb 13 runs)

Models with >~1B params produce checkpoints exceeding 2GB when including optimizer state. The cluster filesystem (or torch.save?) caps at exactly 2,147,479,552 bytes. Affected exp11 retry, exp12 retry, exp2c_1.

### OOM Kills (Feb 13 runs)

5 runs OOM-killed on the system level (not CUDA OOM):
- S2D 128 retry (A100-40GB): 1.42B params, 34GB VRAM but total process memory exceeded 32GB RAM allocation
- S2D 256 retry (H100): Same model, generation metrics pushed total memory over limit
- Wavelet 128 retry (A100-40GB): Same
- Wavelet 256 retry (H100): Same after 210 epochs
- All used `--mem=32G` which is insufficient for 1.42B param models

### Gradient Spike Catastrophe (Feb 13 LDM reruns)

exp9 4x rerun (24039868) and exp9 8x rerun (24039867) developed catastrophic gradient spikes after 30-45 epochs. 26K and 31K steps skipped respectively. The gradient EMA froze and never recovered. Both were fresh starts (no resume). The original runs (23997507, 23997506) completed 500 epochs with zero gradient spikes on A100-SXM4.

Possible cause: Different random seed, or subtle numerical differences between A100-SXM4 and H100/A100-SXM4 (different CUDA math libraries).

### torch.load OmegaConf Bug

All newer-codebase runs (23989xxx, 23996xxx, 23997xxx) failed test evaluation with:
```
WeightsUnpickler error: Unsupported global: GLOBAL omegaconf.listconfig.ListConfig
```
Caused by PyTorch's `weights_only=True` default not supporting OmegaConf types serialized in checkpoints.

### 3D Metrics Bug

DiT exp7 runs failed test evaluation with MS-SSIM expecting 4D tensors but getting 5D (3D volumes).

---

## Part 7: Compression Models (VQ-VAE 3D)

Used by LDM experiments. Key checkpoints in runs_tb/compression_3d/multi_modality/:

| Experiment | Architecture | Compression | Used by |
|-----------|-------------|-------------|---------|
| exp8_1 | VQ-VAE 3D | 4x (256→64) | exp9, exp9_1, exp13 |
| exp7 | VQ-VAE 3D | various | earlier experiments |
| dcae_exp10_1 | DC-AE 3D | 8x8x32 | exp10_1 |

---

---

## Part 8: Sampling Improvement Evaluations (Feb 19-20, 2026)

All evaluations use **exp1_1 bravo pixel 256x256x160** (best 3D bravo model).
25 volumes generated per config, evaluated against all reference splits.

### Baseline Euler Step Sweep

| Steps | FID (all) | KID | CMMD | Time/vol |
|-------|-----------|-----|------|----------|
| 10 | 34.20 | 0.031 | 0.149 | 7.3s |
| **25** | **27.50** | **0.024** | 0.113 | 18.1s |
| 50 | 30.25 | 0.027 | **0.109** | 36.1s |

25 steps is the FID sweet spot. Beyond 25, FID degrades (error accumulation).

### Full ODE Solver Comparison (Not Yet Run)

`eval_ode_solvers.py` tests 33 configs: 5 fixed-step solvers (euler, midpoint, heun2, heun3, rk4) x 5 step counts + 4 adaptive solvers (fehlberg2, bosh3, dopri5, dopri8) x 2 tolerances. SLURM job crashed before producing results. Needs resubmission.

### DiffRS (Discriminator-Guided Reflow)

DiffRS head trained on 105 train + 105 generated volumes, 952K params. Massively overfitting (train 98% acc, val 82%).

**Full correction**: Catastrophic. FID +10 at 25 steps, FID +23 at 50 steps. NFE explosion (3.2x).
**Lightweight correction**: No improvement. FID +0.4 at 25 steps, 1.4x NFE overhead.

**Verdict**: DiffRS fails for small datasets (~200 volumes). Discriminator memorizes instead of learning density ratios.

### Restart Sampling

| Config | NFE | FID (all) | vs Euler/25 |
|--------|-----|-----------|-------------|
| Euler/25 (baseline) | 25 | 27.50 | -- |
| Restart K=1, n=3 | 28 | 32.83 | +5.3 (worse) |
| Restart K=1, n=5 | 30 | 27.50 | +0.0 (same) |

**Verdict**: No improvement. Best restart matches baseline at higher cost.

### Sampling Summary

Plain Euler at 25 steps is optimal. Neither DiffRS nor Restart Sampling helps.

---

## Part 9: Seg Generation Evaluation (exp14_1)

### Bug: Sigmoid Without Threshold

`find_optimal_steps` for exp14_1 (pixel seg 256x256x160) showed FID ~98 flat across all step counts. Training had CMMD 0.02, KID 6e-4 — massive discrepancy.

**Root cause**: Eval script applied `torch.sigmoid()` without thresholding to binary. Training does `(sigmoid > 0.5).float()`. Comparing soft probabilities vs binary masks = meaningless metrics.

**Fix applied**: Added `> 0.5` threshold in `eval_ode_solvers.py`. Results need re-evaluation.

---

## Part 10: DC-AE 1.5 Structured Latent (Feb 20, 2026)

Implementation verified correct: adaptive weight slicing, gradient isolation, shortcut handling all pass functional tests. 1110 existing tests pass.

Three 2D SLURM jobs ready for submission (exp9_1/2/3 with structured latent enabled, 12h auto-chaining).

---

## Part 11: Training Technique Ablations — 256x256x160 (March 2026)

All experiments use 270M UNet unless otherwise noted. Baseline: **exp1_1** @ 500 epochs.
In-training generation metrics: 25 Euler steps, 4 volumes, val conditioning.

### exp1_1 Baseline (reference)

| Checkpoint | Val Loss | Test MSE | Test PSNR | Test LPIPS | FID | KID | CMMD |
|-----------|----------|----------|-----------|-----------|-----|-----|------|
| Best (500ep) | 0.00211 | 0.003122 | 33.24 | 0.5616 | 91.46 | 0.0862 | 0.2347 |
| Latest (500ep) | — | — | — | — | — | — | — |
| Best (1000ep) | 0.00230 | 0.002886 | 33.12 | 0.7589 | 111.86 | 0.0921 | 0.3259 |

Post-hoc eval (25 volumes, test split): **FID=23.85 @ 23 steps** (500ep), **FID=19.12 @ 27 steps** (1000ep).

---

### exp1b_1: [-1,1] Rescaling

**Config**: Same as exp1_1 but pixel normalization to [-1, 1] instead of [0, 1].

| Checkpoint | Val Loss | Test MSE | Test PSNR | Test LPIPS | FID | KID | CMMD |
|-----------|----------|----------|-----------|-----------|-----|-----|------|
| Best | 0.00675 | 0.009399 | 34.99 | 0.5624 | 116.55 | 0.1107 | 0.2171 |
| Latest (500ep) | — | 0.015131 | 31.27 | 0.4882 | 72.00 | 0.0689 | 0.1667 |

MSE ~3x higher due to [-1,1] scale. Latest model FID (72.00) is competitive. 31 gradient spikes — higher instability than [0,1] baseline.

---

### exp1k_1: Offset Noise (σ=0.1)

**Config**: Standard offset noise (Lin et al., 2024), strength=0.1.

| Checkpoint | Val Loss | Test MSE | Test PSNR | Test LPIPS | FID | KID | CMMD |
|-----------|----------|----------|-----------|-----------|-----|-----|------|
| Best (ep 403) | 0.00261 | 0.003326 | 32.67 | 0.8495 | — | — | — |

58 gradient spikes. Generation metrics not computed in output. Val loss slightly worse than baseline.

---

### exp1l_1: Adjusted Offset Noise (σ=0.1)

**Config**: Adjusted offset noise (Everett, 2024), strength=0.1. Noise magnitude rescaled to preserve variance.

| Checkpoint | Val Loss | Test MSE | Test PSNR | Test LPIPS | FID | KID | CMMD |
|-----------|----------|----------|-----------|-----------|-----|-----|------|
| Best (ep 443) | **0.00209** | 0.002532 | 32.36 | 0.8944 | 92.49 | 0.1024 | 0.2773 |
| Latest (500ep) | — | — | — | — | 72.52 | 0.0704 | 0.2260 |

**Best val loss of any experiment (0.00209).** 13 gradient spikes. Latest FID 72.52 shows good generation quality. Training completed in 5.81h (fastest 270M run).

---

### exp1n: CFG-Zero* (Fan et al., 2025)

**Config**: Conditioning dropout prob=0.15, CFG-Zero* guidance during validation (cfg_scale=2.0).

| Checkpoint | Val Loss | Test MSE | Test PSNR | Test LPIPS | FID | KID | CMMD |
|-----------|----------|----------|-----------|-----------|-----|-----|------|
| Best | 0.00232 | 0.002761 | 32.69 | 0.8929 | 165.65 | 0.2147 | 0.3337 |
| Latest (500ep) | — | 0.004248 | 31.18 | 0.4949 | 132.83 | 0.1739 | 0.2477 |

**Poor FID despite good reconstruction.** FID 132-165 is much worse than baseline (91). CFG-Zero* guidance at scale=2.0 may be too aggressive, or the in-training eval is misconfigured. Needs post-hoc CFG scale sweep to find optimal scale. Only 1 gradient spike — very stable training.

---

### exp1o_1: Post-hoc EMA (Karras EDM2)

**Config**: PostHocEMA with sigma_rels=[0.05, 0.28], checkpoint every 5000 steps.

| Checkpoint | Val Loss | Test MSE | Test PSNR | Test LPIPS | FID | KID | CMMD |
|-----------|----------|----------|-----------|-----------|-----|-----|------|
| Best (ep 350) | 0.00235 | 0.002897 | 32.68 | 0.8119 | 80.55 | 0.0889 | 0.2226 |
| Latest (500ep) | — | 0.003812 | **33.36** | **0.5329** | **62.64** | **0.0537** | **0.1898** |

**Best in-training generation metrics across all 270M experiments.** Latest checkpoint: FID 62.64, KID 0.0537, CMMD 0.1898 — all significantly better than baseline. Best PSNR (33.36) and best LPIPS (0.5329). Only 6 gradient spikes. Training took 11.01h (extra overhead from EMA checkpointing).

---

### exp1p_1: Uniform Timestep Sampling

**Status**: Still running (338/500 epochs). Chain 3/20.

| Checkpoint | Val Loss | Notes |
|-----------|----------|-------|
| Best (ep 336) | 0.00266 | Training still in progress |

Epoch 338 showed elevated MSE (0.0082) — possible instability spike. Needs to complete for full evaluation.

---

### exp1r: Cross-Attention Dropout (p=0.1)

**Status**: Still running (323/500 epochs).

| Checkpoint | Val Loss | Notes |
|-----------|----------|-------|
| Best (ep 155) | 0.00246 | Plateaued early |

25 gradient spikes. ~101s/epoch. Val loss plateaued at epoch 155 with no improvement since.

---

### exp1s: Weight Decay (0.05)

**Status**: Still running (347/500 epochs).

| Checkpoint | Val Loss | Notes |
|-----------|----------|-------|
| Best (ep 244) | **0.00231** | Good val loss but unstable |

37 gradient spikes — 17 consecutive rapid spikes at epochs 36-44 suggest weight decay interacts poorly with optimizer early in training. Despite instability, achieved competitive val loss.

---

### exp23: ScoreAug @ 256x256x160 (1000 epochs)

**Status**: Still running (869/1000 epochs).

| Checkpoint | Val Loss | Notes |
|-----------|----------|-------|
| Best (ep 750) | **0.00200** | Still improving after 500ep |

**Best val loss of any experiment when allowed to train longer.** ScoreAug prevents the overfitting that plagues standard 270M training beyond 500 epochs. 6 gradient spikes — stable.

---

### 270M Technique Comparison (256x256x160, 500ep completed)

| Experiment | Technique | Val Loss | FID (latest) | KID | CMMD | Grad Spikes |
|-----------|-----------|----------|-------------|-----|------|-------------|
| exp1_1 | Baseline | 0.00211 | 91.46* | 0.0862 | 0.2347 | — |
| exp1b_1 | [-1,1] rescale | 0.00675 | 72.00 | 0.0689 | 0.1667 | 31 |
| exp1k_1 | Offset noise | 0.00261 | — | — | — | 58 |
| **exp1l_1** | **Adj. offset** | **0.00209** | **72.52** | **0.0704** | **0.2260** | 13 |
| exp1n | CFG-Zero* | 0.00232 | 132.83 | 0.1739 | 0.2477 | 1 |
| **exp1o_1** | **PosthocEMA** | 0.00235 | **62.64** | **0.0537** | **0.1898** | 6 |
| exp23 | ScoreAug | 0.00200† | — | — | — | 6 |

*Best checkpoint FID, not latest. †At 750 epochs, still improving.

**Winners**: PosthocEMA (best generation metrics), adjusted offset noise (best val loss), ScoreAug (best long-training val loss).

---

---

## Part 13: ControlNet Experiments (March 2026)

### exp6a_1: Stage 1 — Unconditional UNet @ 256x256x160

**Config**: 270M UNet, in=1 out=1, no seg conditioning. Train unconditional bravo generation.

| Checkpoint | Val Loss | Test MSE | Test PSNR | Test LPIPS |
|-----------|----------|----------|-----------|-----------|
| Best (ep 368) | 0.00202 | 0.003448 | 33.57 | 0.5714 |

500 epochs complete. 19 gradient spikes. Best val loss 0.00202 — competitive with best techniques. Generation metrics were computing at end of log.

---

### exp6b: Stage 2 — ControlNet @ 128x128x160

**Config**: Frozen UNet from exp6a (128x128), ControlNet encoder (~135M trainable). Seg mask injected via ControlNet residuals.

| Checkpoint | Val Loss | Test MSE | Test PSNR | Test LPIPS | FID | KID | CMMD |
|-----------|----------|----------|-----------|-----------|-----|-----|------|
| Best | 0.00304 | 0.006818 | 31.67 | 0.5036 | 52.98 | 0.0520 | 0.1799 |
| Latest (500ep) | — | 0.005155 | 31.26 | 0.5373 | **49.62** | **0.0470** | **0.1624** |

**Strong generation metrics (FID 49.62)** — ControlNet seg conditioning helps produce more realistic images.

---

### exp6b_1: Stage 2 — ControlNet @ 256x256x160

**Status**: Just started (39/500 epochs).

**Config**: Frozen UNet from exp6a_1 (256x256), ~135M trainable.

| Checkpoint | Val Loss | Notes |
|-----------|----------|-------|
| Best (ep 16) | 0.00211 | ~220s/epoch, still training |

Very early. Val loss 0.00211 at epoch 16 is promising.

---

## Part 14: Post-hoc Evaluation Results (March 2026)

### Optimal Steps (Golden Section Search, 25 vol, test split)

| Experiment | Metric | Best Steps | Best Value | Notes |
|-----------|--------|-----------|-----------|-------|
| exp1_1 (500ep) | FID | 23 | 23.85 | |
| exp1_1 (500ep) | FID_RIN | 46 | 0.674 | |
| exp1_1 (1000ep) | FID | 27 | **19.12** | Best overall |
| exp1_1 (1000ep) | FID_RIN | 49 | 0.714 | |
| exp1b (128, [-1,1]) | FID | 28 | 51.90 | |
| exp1c (128, N(0,1)) | FID | 50 | 142.36 | Gaussian norm fails |

### Time-Shift Evaluation (exp1_1, Euler/25)

| Shift Ratio | FID | Notes |
|------------|-----|-------|
| 1.0 (none) | 54.78 | No shift |
| 1.5 | 50.18 | |
| **2.0** | **49.25** | **Optimal** |
| 3.0 | 49.94 | |
| 4.0 | 53.74 | |
| 6.84 (MONAI default) | 70.90 | MONAI default is harmful |

### FreeU Grid Search (exp1_1 1000ep, Euler/25)

**Best**: backbone=1.0, skip=0.9, FID=19.87 (baseline FID=19.12 at optimal steps).
FreeU provides <1 FID point improvement — marginal benefit.

### Normalization Comparison (128x128x160)

| Normalization | Best Steps | FID |
|--------------|-----------|-----|
| [0, 1] (exp1) | 35 | 52.00 |
| [-1, 1] (exp1b) | 28 | 51.90 |
| N(0, 1) (exp1c) | 50 | 142.36 |

[0,1] and [-1,1] are equivalent. Gaussian normalization catastrophically fails.

---

## Part 15: Downstream Segmentation (March 2026)

### SegResNet Experiments

Downstream segmentation on real + synthetic bravo images using SegResNet.

| Experiment | Data | Resolution | Description |
|-----------|------|-----------|-------------|
| exp1 | Real only | 2D & 3D | Baseline |
| exp2 | Real dual | 2D & 3D | Two modalities |
| exp3 | Real + aug | 2D & 3D | Standard augmentation |
| exp4 | Real dual + aug | 2D & 3D | Combined |
| exp5 | 105 synthetic | 3D | Synthetic only |
| exp6 | All synthetic | 3D | Full synthetic training |
| exp7_1–7_6 | Mixed ratios | 3D | 25–315 synthetic volumes |

### nnU-Net Experiments

| Experiment | Training Data | Status |
|-----------|--------------|--------|
| exp3_baseline | Real only | Complete |
| exp4_baseline_dual | Real dual modality | Complete |
| exp6_synthetic | Synthetic only | Complete |
| exp7_mixed_25syn | 105 real + 25 synthetic | Chain running |
| exp7_mixed_50syn | 105 real + 50 synthetic | Chain running |
| exp7_mixed_75syn | 105 real + 75 synthetic | Chain running |
| exp7_mixed_105syn | 105 real + 105 synthetic | Chain running |
| exp7_mixed_210syn | 105 real + 210 synthetic | Chain running |
| exp7_mixed_315syn | 105 real + 315 synthetic | Chain running |
| exp7_mixed_525syn | 105 real + 525 synthetic | Chain running |

Downstream results pending completion of nnU-Net chains.

---

## Key Takeaways (Updated March 2026)

1. **Pixel-space works reliably**: All pixel UNet runs converge. Best results at 256x256 (val loss 0.0021 bravo, 0.000337 seg).

2. **PosthocEMA is the best single technique**: exp1o_1 achieves FID 62.64 — 31% better than baseline (91.46). Best PSNR, LPIPS, KID, and CMMD.

3. **Adjusted offset noise improves val loss**: exp1l_1 achieves the best validation loss (0.00209) and good FID (72.52).

4. **ScoreAug prevents overfitting beyond 500 epochs**: exp23 continues improving at 869 epochs (val loss 0.00200) while baseline overfits after ~300 epochs.

5. **ControlNet works well**: exp6b (128x128, FID 49.62) demonstrates that seg-conditioned generation via ControlNet is competitive.

6. **1000 epochs + optimal steps gives best post-hoc FID**: exp1_1 at 1000ep evaluated with 27 Euler steps achieves FID 19.12 — the best absolute score.

7. **CFG-Zero* needs tuning**: exp1n shows poor in-training FID (132-165) despite good reconstruction. The default cfg_scale=2.0 may be suboptimal. Pending CFG scale sweep.

8. **FreeU and sampling tricks provide marginal gains**: FreeU (<1 FID point), Restart Sampling (no improvement), DiffRS (negative impact). The quality ceiling is model/data limited.

9. **Time-shift ratio 2.0 is optimal**: MONAI's default time-shift (ratio ~6.84) is harmful. Ratio 2.0 gives 5.5 FID points improvement over no shift.

10. **Latent diffusion has promise but is fragile**: exp9_1 (666M mid UNet, warmup+grad clip) achieved 0.076 val loss and was still improving. Large UNets are unstable.

11. **Infrastructure is a bigger problem than algorithms**: Checkpoint corruption, OOM kills, disk quota exhaustion, and torch.load bugs have wasted more GPU hours than bad hyperparameters.
