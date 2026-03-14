# 3D Diffusion Experiment Results

Last updated: March 14, 2026. Data extracted from IDUN logs and TensorBoard runs.

---

## Quick Reference: Best Results

| Category | Experiment | Val Loss | FID | Epochs | Notes |
|----------|-----------|----------|-----|--------|-------|
| **Bravo pixel 256 (post-hoc)** | exp1_1 (1000ep) | 0.00230 | **19.12** | 1000 | Post-hoc eval, 27 Euler steps |
| **Bravo pixel 256 (loss)** | exp1l_1 (adj.offset) | **0.00209** | 72.52 | 500 | Best val loss |
| **Bravo pixel 256 (in-train)** | exp1o_1 (PosthocEMA) | 0.00235 | **62.64** | 500 | Best in-training FID (pixel) |
| **Bravo LDM 4x (FID)** | exp22_2 (DiT-L) | 0.0847 | **47.41** | 500 | Best test FID overall |
| **Bravo LDM 4x (UNet)** | exp21_2 (MAISI 167M) | 0.0767 | **50.89** | 500 | Best LDM UNet |
| **Bravo WDM** | exp19_2 (270M DDPM) | 0.178 | **67.32** | 500 | Best wavelet |
| **Bravo 128 ControlNet** | exp6b (Stage 2) | 0.00304 | **49.62** | 500 | Best seg-conditioned gen |
| **Bravo pixel 128** | exp8 (EMA) | 0.00227 | N/A | 500 | Best 128x128 |
| **Seg pixel 256** | exp2b_1 (input) | 0.000336 | N/A | 500 | Best seg overall |

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
| exp19_paper | WDM paper | wavelet 256x160 | WDM 5L | 74M | ddpm x0 | bravo |
| exp19_0 | WDM+N(0,1) | wavelet 256x160 | WDM 5L | 74M | ddpm x0 | bravo |
| exp19_1 | WDM+RFlow | wavelet 256x160 | WDM 5L | 74M | rflow | bravo |
| exp19_2 | WDM default | wavelet 256x160 | UNet 5L | 270M | ddpm x0 | bravo |
| exp19_3 | WDM large | wavelet 256x160 | UNet 5L | 662M | ddpm x0 | bravo |
| exp19_4 | WDM+RFlow x0 | wavelet 256x160 | WDM 5L | 74M | rflow x0 | bravo |
| exp19_5 | WDM+BrainNorm | wavelet 256x160 | WDM 5L | 74M | ddpm x0 | bravo |
| exp19_6 | WDM+DiT-S | wavelet 256x160 | DiT-S p=2 | ~40M | ddpm x0 | bravo |
| exp20_1 | Pixel+L3Attn | 256x256x160 | UNet 6L | 656M | rflow | bravo |
| exp20_2 | Pixel+DeepWide | 256x256x160 | UNet 6L | 855M | rflow | bravo |
| exp20_3 | Pixel+DW+L3 | 256x256x160 | UNet 6L | 855M | rflow | bravo |
| exp21_1 | LDM 4x large | latent 64x64x40 | UNet 4L | 666M | rflow | bravo_seg_cond |
| exp21_2 | LDM 4x MAISI | latent 64x64x40 | UNet 4L | 167M | rflow | bravo_seg_cond |
| exp21_3 | LDM 4x small | latent 64x64x40 | UNet 4L | 42M | rflow | bravo_seg_cond |
| exp22_1 | LDM 4x DiT-B | latent 64x64x40 | DiT-B p=2 | 136M | rflow | bravo_seg_cond |
| exp22_2 | LDM 4x DiT-L | latent 64x64x40 | DiT-L p=2 | 478M | rflow | bravo_seg_cond |
| exp22_3 | LDM 4x DiT-S | latent 64x64x40 | DiT-S p=2 | 40M | rflow | bravo_seg_cond |
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

## Part 5b: WDM Wavelet Diffusion (exp19 family, 256x256x160)

All experiments operate on wavelet-decomposed volumes [B,8,80,128,128] — 8 subbands at half resolution.

### exp19_paper: WDM True to Paper (Friedrich et al. 2024)

**Config**: 74M WDM, attention-free, channels [64,128,128,256,256], DDPM x0, LR=1e-5 (paper value), rescale [-1,1] before DWT, NO per-subband normalization, NO warmup/grad clip.

| Val Loss | Final MSE | FID | KID | CMMD | Spikes | VRAM | Status |
|----------|-----------|-----|-----|------|--------|------|--------|
| 0.00210 | 0.00193 | 81.22 | 0.0446 | 0.2846 | 9 | 20.2GB | Completed |

Low MSE but high FID. KID trajectory is **degrading** (+13.0%/100ep) — overfitting in late training.

### exp19_0: WDM DDPM N(0,1) Normalized

**Config**: 74M WDM, DDPM x0, per-subband N(0,1) normalization, gradient_clip=0.5, warmup=10.

| Val Loss | Final MSE | FID | KID | CMMD | Spikes | Status |
|----------|-----------|-----|-----|------|--------|--------|
| 0.154 | 0.323 | 131.75 | 0.120 | 0.446 | 3 | Completed |

N(0,1) normalization hurts generation quality (FID 132). KID trajectory improving (-6.3%/100ep) but absolute values poor.

### exp19_1: WDM RFlow Velocity

**Config**: Same as exp19_0 but RFlow velocity prediction instead of DDPM x0.

| Val Loss | Final MSE | FID | KID | CMMD | Spikes | Status |
|----------|-----------|-----|-----|------|--------|--------|
| 0.198 | 0.209 | 235.27 | 0.248 | 0.490 | 22 | Completed |

**Worst experiment.** RFlow velocity prediction in wavelet domain is catastrophic. KID flat (+1.5%/100ep).

### exp19_2: WDM Default UNet (270M, DDPM x0)

**Config**: Standard UNet [32,64,256,512,512], 270M params, DDPM x0, per-subband N(0,1), gradient_clip=0.5, warmup=10.

| Val Loss | Final MSE | FID | KID | CMMD | Spikes | Status |
|----------|-----------|-----|-----|------|--------|--------|
| 0.178 | 0.310 | **67.32** | **0.0437** | **0.235** | 1 | Completed |

**Best WDM experiment.** The standard UNet architecture (same as pixel baseline) outperforms the paper's WDM architecture at 270M vs 74M. KID flat (-0.07%/100ep) — fully converged. Post-hoc eval found 10 DDPM steps optimal (FID=114.65).

### exp19_3: WDM Large UNet (662M, DDPM x0)

**Config**: Larger UNet [64,128,256,512,1024], 662M params, DDPM x0.

| Val Loss | Final MSE | FID | KID | CMMD | Spikes | VRAM | Status |
|----------|-----------|-----|-----|------|--------|------|--------|
| 0.176 | 0.316 | 123.96 | 0.0999 | 0.374 | 0 | 27.7GB | Completed |

Bigger model but worse FID than exp19_2 (270M). KID trajectory still **strongly improving** (-16.4%/100ep) — undertrained at 500 epochs.

### exp19_4: WDM RFlow x0-prediction

**Config**: 74M WDM, RFlow with x0-prediction target (not velocity).

| Val Loss | Final MSE | Epochs | Spikes | Status |
|----------|-----------|--------|--------|--------|
| 0.0184 | 0.047 | 372/500 | 0 | **Timed out** |

Hit wall time at epoch 372. No generation metrics available. KID flat (-0.3%/100ep).

### exp19_5: WDM DDPM Brain-only Norm

**Config**: 74M WDM, DDPM x0, brain-only N(0,1) normalization (threshold=0.01).

| Val Loss | Final MSE | FID | KID | CMMD | Spikes | Status |
|----------|-----------|-----|-----|------|--------|--------|
| 0.0359 | 0.0449 | 92.77 | 0.0722 | 0.255 | 57 | Completed |

Brain-only norm improves MSE vs full N(0,1) but worse FID than exp19_2. 57 spikes — most unstable WDM variant. KID flat (+0.1%/100ep).

### exp19_6: Wavelet DiT-S (DDPM x0)

**Config**: DiT-S replacing WDM backbone, hidden=384, depth=12, patch=2, gradient_checkpointing=true.

| Val Loss | Final MSE | FID | KID | CMMD | Spikes | VRAM | Status |
|----------|-----------|-----|-----|------|--------|------|--------|
| 0.280 | 0.392 | 110.88 | 0.0938 | 0.409 | 3 | 7.2GB | Completed |

DiT-S underperforms UNet in wavelet domain. Very low VRAM (7.2GB). KID flat (-0.2%/100ep).

### WDM Summary

| Experiment | Architecture | Strategy | FID | KID | CMMD | KID Trend |
|-----------|-------------|----------|-----|-----|------|-----------|
| **exp19_2** | **UNet 270M** | **DDPM x0** | **67.32** | **0.0437** | **0.235** | flat |
| exp19_paper | WDM 74M | DDPM x0 | 81.22 | 0.0446 | 0.285 | degrading |
| exp19_5 | WDM 74M | DDPM x0 | 92.77 | 0.0722 | 0.255 | flat |
| exp19_6 | DiT-S ~40M | DDPM x0 | 110.88 | 0.0938 | 0.409 | flat |
| exp19_3 | UNet 662M | DDPM x0 | 123.96 | 0.0999 | 0.374 | strong improve |
| exp19_0 | WDM 74M | DDPM x0 | 131.75 | 0.1200 | 0.446 | improving |
| exp19_1 | WDM 74M | RFlow vel | 235.27 | 0.2484 | 0.490 | flat |

**Key findings**: Standard UNet architecture outperforms WDM architecture in wavelet domain. DDPM x0 prediction is essential — RFlow velocity prediction fails catastrophically in wavelet space. Per-subband N(0,1) normalization generally hurts.

---

## Part 5c: Larger Pixel UNet Experiments (exp20_1/2/3, 256x256x160)

### exp20_1: 656M UNet + L3 Attention

**Config**: UNet 6L [16,32,64,256,512,1024], L3+L4+L5 attention, 656M params, gradient_checkpointing, warmup=10, gradient_clip=0.5.

| Epochs | Val Loss | Final MSE | Spikes | VRAM | Status |
|--------|----------|-----------|--------|------|--------|
| 318/500 | 0.00240 | 0.00248 | 2 | 55.3GB | **Stalled** (1 ep/resubmit) |

Never completed — kept resubmitting but only ran 1 epoch per job (likely memory pressure or time limit at 8 min/epoch). KID trajectory **strongly improving** (-46.0%/100ep) at KID 0.060 — was still converging fast when it stalled.

### exp20_2: 855M Deep/Wide Bottleneck

**Config**: UNet 6L [16,32,64,256,512,1024], L4+L5 attention, res_blocks=[1,1,1,2,3,3], 855M params, LR=5e-5, warmup=20, gradient_clip=0.1.

| Epochs | Val Loss | Final MSE | FID | KID | CMMD | Spikes | VRAM | Status |
|--------|----------|-----------|-----|-----|------|--------|------|--------|
| 500 | 0.00239 | 0.00199 | 99.62 | 0.114 | **0.160** | 10 | 77.4GB | Completed |

**Best CMMD (0.160) of ALL experiments.** KID trajectory strong improve (-10.9%/100ep) but KID absolute still high (0.035). High FID (99.6) despite good CMMD — may need more training or different evaluation.

### exp20_3: 855M Deep/Wide + L3 Attention

**Config**: Same as exp20_2 but L3+L4+L5 attention, num_head_channels=128.

| Epochs | Val Loss | Final MSE | FID | KID | CMMD | Spikes | VRAM | Status |
|--------|----------|-----------|-----|-----|------|--------|------|--------|
| 500 | **0.00209** | 0.00203 | 98.00 | 0.110 | 0.190 | 8 | 46.3GB | Completed |

Slightly better FID than exp20_2 but worse CMMD. L3 attention adds 20K-token self-attention but uses less VRAM (46 vs 77GB) with gradient_checkpointing. KID trajectory **flat** (+0.08%/100ep) — converged.

### Large Pixel UNet Summary

These 6L UNets (656M-855M) achieve competitive val loss but poor in-training FID/KID compared to the 270M 5L baseline. The CMMD results (exp20_2: 0.160) suggest they may produce more diverse outputs. However, the huge VRAM cost (46-77GB) and long training time (~8-12 min/epoch) make them impractical for iterative experimentation.

---

## Part 5d: LDM UNet Experiments (exp21, VQ-VAE 4x)

All experiments use VQ-VAE 4x compression (exp8_1), latent shape [B,8,40,64,64], bravo_seg_cond mode (in=8 bravo+seg, out=4 bravo velocity).

### exp21_1: 666M UNet Large

**Config**: UNet 4L [128,256,512,1024], L2+L3 attention, warmup=10, gradient_clip=0.5.

| Epochs | Val Loss | Final MSE | Spikes | VRAM | Status |
|--------|----------|-----------|--------|------|--------|
| 175/500 | 0.0767 | 0.0582 | 8 | 75.1GB | **Disk quota exceeded** |

Hit disk quota at epoch 175. Similar val loss trajectory to exp21_2 before crash. Too large for available storage.

### exp21_2: 167M MAISI-style UNet

**Config**: UNet 4L [64,128,256,512], L2+L3 attention, warmup=10, gradient_clip=0.5.

| Epochs | Val Loss | Final MSE | FID | KID | CMMD | Spikes | VRAM | Status |
|--------|----------|-----------|-----|-----|------|--------|------|--------|
| 500 | 0.0767 | 0.0374 | **50.89** | **0.0413** | **0.174** | 38 | 19.1GB | Completed |

**Best LDM UNet.** Competitive with pixel baseline on FID/KID at only 19.1GB VRAM (vs 43GB for pixel 256x256). 38 gradient spikes — more unstable than pixel but manageable.

### exp21_3: 42M Small UNet

**Config**: UNet 4L [32,64,128,256], L2+L3 attention, warmup=10, gradient_clip=0.5.

| Epochs | Val Loss | Final MSE | FID | KID | CMMD | Spikes | VRAM | Status |
|--------|----------|-----------|-----|-----|------|--------|------|--------|
| 500 | 0.0817 | 0.0720 | 83.90 | 0.0852 | 0.256 | **170** | 6.1GB | Completed |

170 gradient spikes — most unstable experiment. Small model underfits latent space. Low VRAM (6.1GB).

---

## Part 5e: LDM DiT Experiments (exp22, VQ-VAE 4x)

Same compression as exp21. DiT architecture operates on 20,480 tokens (patch=2 on 64x64x40 latent).

### exp22_1: DiT-B (136M)

**Config**: DiT-B (768 hidden, 12 depth, 12 heads), patch=2, use_compile=true.

| Epochs | Val Loss | Final MSE | FID | KID | CMMD | Spikes | Time | Status |
|--------|----------|-----------|-----|-----|------|--------|------|--------|
| 500 | 0.0851 | 0.0763 | 48.99 | 0.0376 | 0.266 | 14 | 2.74h | Completed |

Strong generation metrics at very fast training (2.74h total, ~20s/epoch). FID nearly identical to exp21_2 MAISI UNet.

### exp22_2: DiT-L (478M)

**Config**: DiT-L (1024 hidden, 24 depth, 16 heads), patch=2, LR=5e-5, warmup=20, use_compile=true.

| Epochs | Val Loss | Final MSE | FID | KID | CMMD | Spikes | Time | Status |
|--------|----------|-----------|-----|-----|------|--------|------|--------|
| 500 | 0.0847 | 0.0638 | **47.41** | **0.0355** | 0.252 | 4 | 3.53h | Completed |

**Best FID (47.41) and KID (0.0355) of ALL experiments.** Only 4 gradient spikes — very stable. 3.53h total training time — 10x faster than pixel 256x256.

### exp22_3: DiT-S Long (40M, 2000 epochs)

**Config**: DiT-S (384 hidden, 12 depth, 6 heads), patch=2, 2000 epochs, use_compile=true.

| Epochs | Val Loss | Final MSE | FID | KID | CMMD | Spikes | Time | Status |
|--------|----------|-----------|-----|-----|------|--------|------|--------|
| 2000 | 0.0823 | 0.0461 | 61.14 | 0.0521 | 0.283 | 3 | 11.21h | Completed |

Small model needs more epochs but still underperforms DiT-L at 500ep. Lowest final MSE in DiT group but worse generation metrics — reconstruction quality doesn't translate to generation quality.

### LDM Architecture Comparison (VQ-VAE 4x, 500ep unless noted)

| Experiment | Architecture | Params | FID | KID | CMMD | VRAM | Time |
|-----------|-------------|--------|-----|-----|------|------|------|
| **exp22_2** | **DiT-L** | **478M** | **47.41** | **0.0355** | **0.252** | ~32GB | 3.53h |
| exp22_1 | DiT-B | 136M | 48.99 | 0.0376 | 0.266 | ~12GB | 2.74h |
| exp21_2 | UNet MAISI | 167M | 50.89 | 0.0413 | 0.174 | 19.1GB | 10.03h |
| exp22_3 | DiT-S (2000ep) | 40M | 61.14 | 0.0521 | 0.283 | ~6GB | 11.21h |
| exp21_3 | UNet Small | 42M | 83.90 | 0.0852 | 0.256 | 6.1GB | 6.56h |

DiT models train 3-4x faster than UNet equivalents in latent space. DiT-L achieves the best FID/KID, while MAISI UNet has the best CMMD.

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

**Overfitting is worse than baseline**: Lower training loss but wider train/val gap than exp1_1. Dropout helps the optimizer escape shallow minima and find deeper training loss basins, but on 105 volumes deeper minima = more memorization. The train/inference gap (no dropout at test time → full capacity) amplifies this. Generation metrics are still decent (KID 0.048, CMMD 0.191) but the technique fails as a regularizer.

---

### exp1s: Weight Decay (0.05)

**Status**: Still running (347/500 epochs).

| Checkpoint | Val Loss | Notes |
|-----------|----------|-------|
| Best (ep 244) | **0.00231** | Good val loss but unstable |

37 gradient spikes — 17 consecutive rapid spikes at epochs 36-44 suggest weight decay interacts poorly with optimizer early in training. Despite instability, achieved competitive val loss.

**Overfitting is worse than baseline**: Weight decay (AdamW-decoupled) keeps weights small and the loss landscape smooth, letting the optimizer slide into deeper training minima faster. But deeper minima on 105 volumes = more memorization. The regularization pressure is far too weak to compensate for the 2500:1 parameter-to-sample ratio. Generation metrics are still reasonable (KID 0.041, CMMD 0.190) but the technique fails as a regularizer.

---

### exp23: ScoreAug @ 256x256x160 (1000 epochs)

**Status**: Still running (869/1000 epochs).

| Checkpoint | Val Loss | Notes |
|-----------|----------|-------|
| Best (ep 750) | **0.00200** | Still improving after 500ep |

**Best val loss of any experiment when allowed to train longer.** ScoreAug is the only technique that prevents overfitting beyond 500 epochs — unlike weight decay and dropout (exp1r/1s), which only change how the model fits the same 105 volumes, ScoreAug adds genuinely new training signal via augmented volumes with known transformations, effectively increasing dataset size. 6 gradient spikes — stable.

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

## Part 16: Generation Metric Trajectory Analysis (March 14, 2026)

In-training generation metrics (KID, CMMD) are computed every epoch using 4 generated volumes at 25 Euler steps. While noisy, the **trajectory direction** over the last 100+ epochs reveals whether an experiment has converged, is still improving, or is overfitting.

Methodology: smoothed (rolling window=20), linear regression slope over last 100 data points, expressed as % change per 100 epochs.

### 2D KID Trajectory Rankings (Generation/KID_mean_val)

**Top tier — still strongly improving (slope < -10%):**

| Experiment | KID Mean | Slope | Epochs | Notes |
|-----------|---------|-------|--------|-------|
| exp23 ScoreAug | 0.059 | -50.8% | 852 | Most room to improve |
| exp20_1 656M+L3attn | 0.060 | -46.0% | 317 | Stalled before completion |
| exp8 EMA (128) | 0.044 | -42.5% | 500 | Strong 128x128 |
| exp1_1 baseline | 0.042 | -33.1% | 500 | Baseline still improving |
| exp6a_1 CtrlNet S1 | 0.026 | -27.8% | 500 | **Lowest absolute KID + improving** |
| exp1l_1 adj.offset | 0.034 | -23.6% | 500 | |
| exp6b CtrlNet S2 | 0.039 | -17.8% | 500 | |
| exp1o_1 PosthocEMA | 0.037 | -13.3% | 500 | |

**Converged — flat (-3% to +3%):**

| Experiment | KID Mean | Slope | Notes |
|-----------|---------|-------|-------|
| exp20_3 855M+L3 | 0.035 | +0.1% | Converged at 500ep |
| exp1h LPIPS+Huber (128) | 0.025 | +1.2% | Best converged 128x128 |
| exp1k offset noise (128) | 0.026 | +1.3% | |
| exp19_2 WDM 270M | 0.053 | -0.1% | Best WDM, converged |

**Degrading (slope > +10%):**

| Experiment | KID Mean | Slope | Notes |
|-----------|---------|-------|-------|
| exp1j_1 grad.accum | 0.057 | +11.6% | Overfitting |
| exp19_paper WDM | 0.070 | +13.0% | Overfitting |
| exp1c_1 brain norm | 0.458 | +28.0% | Catastrophic |

### 3D KID Trajectory Rankings (Generation_3d/KID_mean_val)

Only available for newer experiments (11 runs with >50 points).

| Experiment | 3D KID Mean | Slope | Trend |
|-----------|------------|-------|-------|
| exp6a_1 CtrlNet S1 | 0.026 | -43.3% | strong improve |
| exp23 ScoreAug | 0.061 | -41.6% | strong improve |
| exp1l_1 adj.offset | 0.031 | -26.5% | strong improve |
| exp1k_1 offset noise | 0.037 | -20.5% | strong improve |
| exp6b CtrlNet S2 | 0.040 | -19.2% | strong improve |
| exp1o_1 PosthocEMA | 0.037 | -9.6% | improving |
| exp1p_1 uniform T | 0.054 | -0.2% | flat |
| exp1s weight decay | 0.042 | +1.1% | flat |
| exp1r attn dropout | 0.051 | +2.8% | flat |

### CMMD Trajectory Rankings (Generation/CMMD_val, top 10)

| Experiment | CMMD Mean | Slope | Trend |
|-----------|----------|-------|-------|
| exp23 ScoreAug | 0.235 | -32.6% | strong improve |
| exp1p_1 uniform T | 0.243 | -22.9% | strong improve |
| exp20_1 656M+L3 | 0.260 | -22.5% | strong improve |
| exp1_1 baseline | 0.250 | -12.5% | strong improve |
| exp6a_1 CtrlNet S1 | 0.175 | -1.5% | flat (lowest absolute) |
| exp1o_1 PosthocEMA | 0.182 | -8.9% | improving |
| exp1r attn dropout | 0.191 | -7.2% | improving |
| exp6b CtrlNet S2 | 0.189 | +1.6% | flat |
| exp1s weight decay | 0.190 | -3.4% | improving |
| exp1l_1 adj.offset | 0.207 | -3.0% | flat |

### Key Trajectory Insights

1. **exp6a_1 ControlNet S1** has the lowest absolute KID (0.026) and CMMD (0.175) while still strongly improving — the unconditional generation baseline is the single best performer on in-training metrics.
2. **exp23 ScoreAug** has the steepest improvement slopes across all metrics at epoch 852 — given enough training time, it may overtake exp6a_1.
3. **exp1o_1 PosthocEMA** shows consistent improvement across all metrics but at a moderate rate — already competitive.
4. **Overfitting is visible**: exp1j_1 (grad accum), exp19_paper (WDM), and exp1c_1 (brain norm) all show degrading KID trajectories.
5. **128x128 experiments** (exp1h, exp1k, exp1e, exp1g) have the lowest absolute KID values (0.024-0.026) because 128x128 generation is fundamentally easier.
6. **No in-training generation metrics exist for LDM experiments** (exp21/22) — those results come only from test-time evaluation.

### Cross-Domain Comparison: Test FID Rankings

Combining all available test FID results (in-training for pixel, test eval for LDM/WDM):

| Rank | Experiment | Type | FID | KID | CMMD |
|------|-----------|------|-----|-----|------|
| 1 | exp1_1 (1000ep, post-hoc 27 steps) | Pixel | **19.12** | — | — |
| 2 | exp22_2 DiT-L | LDM | 47.41 | 0.0355 | 0.252 |
| 3 | exp22_1 DiT-B | LDM | 48.99 | 0.0376 | 0.266 |
| 4 | exp6b CtrlNet S2 (128) | Pixel | 49.62 | 0.0470 | 0.162 |
| 5 | exp21_2 MAISI UNet | LDM | 50.89 | 0.0413 | 0.174 |
| 6 | exp22_3 DiT-S (2000ep) | LDM | 61.14 | 0.0521 | 0.283 |
| 7 | exp1o_1 PosthocEMA | Pixel | 62.64 | 0.0537 | 0.190 |
| 8 | exp19_2 WDM 270M | WDM | 67.32 | 0.0437 | 0.235 |
| 9 | exp1l_1 adj.offset | Pixel | 72.52 | 0.0704 | 0.226 |
| 10 | exp19_paper WDM | WDM | 81.22 | 0.0446 | 0.285 |

**Caveat**: FID numbers are NOT directly comparable across evaluation setups. Post-hoc eval uses 25 volumes with optimal Euler steps. In-training eval uses 4 volumes with fixed 25 steps. LDM test eval uses val reference fallback. These rankings should be interpreted with caution.

---

## Key Takeaways (Updated March 14, 2026)

1. **Pixel-space with post-hoc eval produces the best absolute FID**: exp1_1 at 1000ep with 27 Euler steps achieves FID 19.12. However, in-training FID (91.46) is misleadingly high — post-hoc evaluation with more volumes and optimal steps reveals the true quality.

2. **LDM DiT is the most efficient approach**: exp22_2 (DiT-L 478M) achieves FID 47.41 in only 3.53h of training. DiT models in latent space train 10x faster than pixel UNets while achieving competitive generation quality.

3. **PosthocEMA is the best single pixel technique**: exp1o_1 achieves FID 62.64 and in-training KID 0.037 (still improving). Best PSNR (33.36) and LPIPS (0.5329).

4. **ScoreAug is essential for long training**: exp23 shows the steepest KID improvement (-50.8%/100ep) at epoch 852 — the only technique that prevents overfitting beyond 500 epochs.

5. **ControlNet S1 has the best in-training metrics**: exp6a_1 achieves the lowest absolute KID (0.026) and CMMD (0.175) while still strongly improving — unconditional 1-channel generation is easier to optimize.

6. **Adjusted offset noise improves val loss**: exp1l_1 achieves val loss 0.00209 with strong KID trajectory (-23.6%/100ep).

7. **Wavelet domain works but pixel is better**: exp19_2 (WDM, FID 67.32) is the best wavelet result. RFlow velocity prediction fails in wavelet space — DDPM x0 is required.

8. **CFG-Zero* needs tuning**: exp1n shows flat KID trajectory and poor FID (132-165). Pending CFG scale sweep.

9. **FreeU and sampling tricks provide marginal gains**: FreeU (<1 FID point), Restart Sampling (no improvement), DiffRS (negative impact).

10. **Time-shift ratio 2.0 is optimal**: MONAI's default (~6.84) is harmful. Ratio 2.0 gives 5.5 FID improvement.

11. **Generation metric trajectories matter more than val loss**: Val loss ranking does not predict generation quality. The best technique by val loss (exp1l_1, 0.00209) is ranked 9th by test FID. Always evaluate with generation metrics.

12. **Infrastructure remains the biggest bottleneck**: Checkpoint corruption, OOM kills, disk quota, torch.load bugs, and stalled chains have wasted more GPU hours than any hyperparameter choice.
