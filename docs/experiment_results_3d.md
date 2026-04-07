# 3D Diffusion Experiment Results

Last updated: April 7, 2026. Data extracted from IDUN logs and TensorBoard runs.

---

## Quick Reference: Best Results

| Category | Experiment | Val Loss | FID | Epochs | Notes |
|----------|-----------|----------|-----|--------|-------|
| **Bravo pixel 256 (post-hoc)** | exp1_1 (1000ep) | 0.00230 | **19.12** | 1000 | Post-hoc eval, 27 Euler steps |
| **Bravo pixel 256 (post-hoc)** | exp23 (ScoreAug) | **0.00200** | **20.38** | 1000 | Post-hoc, best RadImageNet FID (0.659) |
| **Bravo pixel 256 (in-train)** | exp1p_1 (uniform T) | 0.00256 | **58.85** | 500 | Best 256x256 in-training FID at 500ep |
| **Bravo pixel 256 (combined)** | exp24 (all techniques) | 0.00347 | **62.87** | 1000 | ScoreAug+AdjOffset+PHEMA+UniformT |
| **Bravo pixel 256 (loss)** | exp23 (ScoreAug) | **0.00200** | 62.57 | 1000 | Best val loss of any experiment |
| **Bravo LDM 4x (FID)** | exp22_2 (DiT-L) | 0.0847 | **47.41** | 500 | Best LDM FID |
| **Bravo LDM 4x (UNet)** | exp21_2 (MAISI 167M) | 0.0767 | **50.89** | 500 | Best LDM UNet |
| **Bravo LDM 4x+ScoreAug** | exp27 (DiT-L 1000ep) | — | **57.24** | 1000 | ScoreAug DiT-L |
| **Dual pixel 128** | exp1v2 (T1pre+T1gd) | — | **32.80** | 500 | Best dual FID |
| **Triple pixel 128** | exp1v3 (T1pre+T1gd+FLAIR) | — | **65.37** | 500 | Best triple FID |
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
| exp1r | Pixel+AttnDropout | 128x128x160 | UNet 5L | 270M | rflow | bravo |
| exp1s | Pixel+WeightDecay | 128x128x160 | UNet 5L | 270M | rflow | bravo |
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
| exp20_4 | Pixel Small+Attn | 256x256x160 | UNet 5L | 67M | rflow | bravo |
| exp20_5 | Pixel Mid | 256x256x160 | UNet 5L | 152M | rflow | bravo |
| exp20_6 | Pixel Tiny | 256x256x160 | UNet 4L | ~20M | rflow | bravo |
| exp20_7 | Pixel 67M NoAttn | 256x256x160 | UNet 5L | 67M | rflow | bravo |
| exp24 | Pixel Combined 270M | 256x256x160 | UNet 5L | 270M | rflow | bravo |
| exp25 | Pixel Combined 17M | 256x256x160 | UNet 6L | 17M | rflow | bravo |
| exp26 | WDM+ScoreAug 1000ep | 256x256x160 | UNet 5L | 270M | ddpm x0 | bravo |
| exp26_1 | WDM vanilla 1000ep | 256x256x160 | UNet 5L | 270M | ddpm x0 | bravo |
| exp27 | LDM DiT-L+ScoreAug | latent 64x64x40 | DiT-L p=2 | 478M | rflow | bravo_seg_cond |
| exp28 | LDM MAISI+ScoreAug | latent 64x64x40 | UNet 4L | 167M | rflow | bravo_seg_cond |
| exp28_1 | LDM MAISI+ScoreAug+Mixup | latent 64x64x40 | UNet 4L | 167M | rflow | bravo_seg_cond |
| exp28_2 | LDM MAISI+ScoreAug v2 | latent 64x64x40 | UNet 4L | 167M | rflow | bravo_seg_cond |
| exp1v2 | Pixel Dual (T1pre+T1gd) | 128x128x160 | UNet 5L | 270M | rflow | dual |
| exp1v2_1 | Pixel Dual + Joint Norm | 128x128x160 | UNet 5L | 270M | rflow | dual |
| exp1v3 | Pixel Triple (T1pre+T1gd+FLAIR) | 128x128x160 | UNet 5L | 270M | rflow | triple |
| exp1v3_1 | Pixel Triple + Joint Norm | 128x128x160 | UNet 5L | 270M | rflow | triple |
| exp12_5 | WDM Medium 250M | 128x128x160 | UNet 5L | ~250M | ddpm x0 | bravo |
| exp12_6 | WDM DiT-S 128 | 128x128x160 | DiT-S p=2 | ~40M | ddpm x0 | bravo |
| exp2e | Seg Multi-level Aux Bin | 128x128x160 | UNet 5L | 270M | rflow | seg_cond_3d |

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

### exp1p_1: Uniform Timestep Sampling — COMPLETED

**Config**: 270M UNet, rflow, LR=1e-4, sample_method=uniform. A100 80GB.

| Checkpoint | Val Loss | Test MSE | Test PSNR | Test LPIPS | FID | KID | CMMD |
|-----------|----------|----------|-----------|-----------|-----|-----|------|
| Best | 0.00256 | 0.003009 | 32.84 | 0.8018 | 62.00 | 0.0508 | 0.2228 |
| Latest (500ep) | — | 0.004160 | 33.16 | 0.6704 | **58.85** | **0.0464** | **0.1977** |

34 gradient spikes. Training: 5.81h.

Good generation metrics (FID 58.85 latest). CMMD trajectory was strongly improving (-22.9%/100ep) but 3D KID was flat (-0.2%). Simpler than logit-normal with no hyperparameters to tune.

---

### exp1r: Cross-Attention Dropout (p=0.1) @ 128x128x160 — COMPLETED

**Config**: 270M UNet, rflow, LR=1e-4 + cross-attention dropout 0.1. **128x128x160**. H100 80GB.

| Checkpoint | Val Loss | Test MSE | Test PSNR | Test LPIPS | FID | KID | CMMD |
|-----------|----------|----------|-----------|-----------|-----|-----|------|
| Best (ep 112) | 0.00246 | 0.002354 | 32.63 | 0.7802 | 82.21 | 0.0887 | 0.2667 |
| Latest (500ep) | — | 0.008438 | 30.55 | 0.4523 | 49.32 | 0.0466 | 0.1629 |

47 gradient spikes. Training: 2.48h + resume.

**Note**: 128x128x160 resolution — FID not comparable to 256x256 experiments. Best-checkpoint LPIPS (0.7802) is worst of all experiments, suggesting the dropout creates a large train/inference gap. Val loss plateaued early at epoch 112.

---

### exp1s: Weight Decay (0.05) @ 128x128x160 — COMPLETED

**Config**: 270M UNet, rflow, LR=1e-4 + AdamW weight decay 0.05. **128x128x160**. H100 80GB.

| Checkpoint | Val Loss | Test MSE | Test PSNR | Test LPIPS | FID | KID | CMMD |
|-----------|----------|----------|-----------|-----------|-----|-----|------|
| Best (ep 142) | 0.00231 | 0.005689 | 33.69 | 0.5217 | 54.60 | 0.0509 | 0.1899 |
| Latest (500ep) | — | 0.004787 | 30.86 | 0.5152 | 43.45 | 0.0323 | 0.1681 |

38 gradient spikes — 17 consecutive rapid spikes at epochs 36-44 suggest weight decay interacts poorly with optimizer early in training.

**Note**: 128x128x160 resolution — FID not comparable to 256x256 experiments. At 128x128, latest FID 43.45 compares favorably to exp8 (EMA, 128x128, no FID available) and exp5_1 (ScoreAug, 128x128, val loss 0.00213). 3D KID trajectory was flat (+1.1%/100ep).

---

### exp23: ScoreAug @ 256x256x160 (1000 epochs) — COMPLETED

**Config**: 270M UNet, rflow, LR=1e-4 + ScoreAug (rotation+flip+translation+cutout, omega conditioning). A100 80GB.

| Checkpoint | Val Loss | Test MSE | Test PSNR | Test LPIPS | FID | KID | CMMD |
|-----------|----------|----------|-----------|-----------|-----|-----|------|
| Best (ep ~750) | **0.00200** | 0.002195 | 33.04 | 0.5915 | 72.39 | 0.0579 | 0.1936 |
| Latest (1000ep) | — | 0.003152 | 33.34 | 0.5307 | **62.57** | **0.0531** | **0.1850** |

139 gradient spikes (9 optimizer skips). Training: ~5.15h (final chain segment).

**Post-hoc optimal steps (25 volumes, test split):**

| Metric | Best Steps | Best Value | Notes |
|--------|-----------|-----------|-------|
| FID (ImageNet) | **27** | **20.38** | Close to baseline's 19.12 at 1000ep |
| FID (RadImageNet) | 48 | **0.659** | Better than baseline's 0.714 |

**ScoreAug prevented overfitting AND matched or beat exp1_1 at 1000 epochs on generation metrics.** Post-hoc FID 20.38 (27 steps) is within 1.3 points of baseline's 19.12, while RadImageNet FID (0.659) actually beats baseline (0.714). The baseline overfits beyond ~500 epochs; ScoreAug maintained improving KID/CMMD trajectories throughout 1000 epochs (KID slope -50.8%/100ep, steepest of all experiments).

ScoreAug adds genuinely new training signal via augmented volumes with known transformations, effectively increasing dataset size. This is the only regularization technique that works — the problem is data-limited, not optimization-limited.

---

### 270M Technique Comparison (256x256x160)

All 500 epochs unless noted. FID/KID/CMMD are from **latest** checkpoint in-training eval (4 vol, 25 Euler steps) unless marked.

| Experiment | Technique | Val Loss | FID (latest) | KID (latest) | CMMD (latest) | Grad Spikes |
|-----------|-----------|----------|-------------|-------------|--------------|-------------|
| exp1_1 | Baseline (500ep) | 0.00211 | 51.17 | 0.033 | 0.193 | — |
| exp1_1 | Baseline (1000ep) | 0.00230 | 49.53 | 0.035 | 0.149 | — |
| exp1b_1 | [-1,1] rescale | 0.00675 | 72.00 | 0.069 | 0.167 | 31 |
| exp1k_1 | Offset noise | 0.00261 | — | — | — | 58 |
| **exp1l_1** | **Adj. offset** | **0.00209** | 72.52† | 0.070 | 0.226 | 13 |
| exp1n | CFG-Zero* | 0.00265 | 132.83 | 0.174 | 0.248 | 1 |
| **exp1o_1** | **PosthocEMA** | 0.00235 | **62.64†** | 0.054 | 0.190 | 6 |
| **exp1p_1** | **Uniform T** | 0.00256 | **58.85** | **0.046** | **0.198** | 34 |
| **exp23** | **ScoreAug (1000ep)** | **0.00200** | 62.57 | 0.053 | 0.185 | 9 |
| **exp24** | **Combined (1000ep)** | 0.00347 | 62.87 | 0.053 | 0.259 | 742 |

†Best checkpoint FID, not latest.

**Note**: exp1r (attn dropout) and exp1s (weight decay) ran at **128x128x160** — their FID numbers are not comparable to the 256x256 table above. See individual sections for their results.

### 128x128x160 Technique Comparison

| Experiment | Technique | Val Loss | FID (latest) | KID (latest) | CMMD (latest) | Grad Spikes |
|-----------|-----------|----------|-------------|-------------|--------------|-------------|
| exp1s | Weight decay | 0.00231 | **43.45** | **0.032** | **0.168** | 38 |
| exp1r | Attn dropout | 0.00246 | 49.32 | 0.047 | 0.163 | 47 |
| exp5_1 | ScoreAug | 0.00213 | — | — | — | — |
| exp8 | EMA | **0.00227** | — | — | — | — |

**Post-hoc FID rankings (25 vol, optimal Euler steps, 256x256x160 only):**

| Experiment | Best Steps | FID (ImageNet) | FID (RadImageNet) |
|-----------|-----------|---------------|------------------|
| exp1_1 (1000ep) | 27 | **19.12** | 0.714 |
| exp23 ScoreAug (1000ep) | 27 | 20.38 | **0.659** |
| exp1_1 (500ep) | 23 | 23.85 | 0.674 |

**256x256 in-training winners**: Uniform T (FID 58.85), PosthocEMA (FID 62.64†), ScoreAug (FID 62.57 at 1000ep). **Post-hoc winners**: Baseline 1000ep (FID 19.12), ScoreAug (FID 20.38, best RadImageNet 0.659). **Val loss winner**: ScoreAug (0.00200).

---

---

## Part 12b: Model Scaling Experiments (March 2026)

Testing how model size affects generation quality at 256x256x160. All use RFlow, LR=1e-4, batch=1, 500 epochs.

### exp20_4: Small UNet 67M (with L5 attention)

**Config**: UNet channels=[16,32,64,128,256], attention at level 5 only.

| Checkpoint | Val Loss | Test MSE | Test PSNR | Test LPIPS | FID | KID | CMMD |
|-----------|----------|----------|-----------|-----------|-----|-----|------|
| Best | 0.00262 | 0.003971 | 32.92 | 0.6851 | 95.85 | 0.0944 | 0.2430 |
| Latest (500ep) | — | 0.004889 | 30.77 | 0.5877 | 99.86 | 0.1104 | 0.2452 |

114 gradient spikes. A100 80GB. 9.96h.

### exp20_5: Mid UNet 152M

**Config**: UNet channels=[32,64,128,256,512], norm_num_groups=32.

| Checkpoint | Val Loss | Test MSE | Test PSNR | Test LPIPS | FID | KID | CMMD |
|-----------|----------|----------|-----------|-----------|-----|-----|------|
| Best | 0.00238 | 0.003212 | 33.13 | 0.6350 | 95.00 | 0.0895 | 0.3304 |
| Latest (500ep) | — | 0.004167 | 30.95 | 0.5574 | **72.11** | **0.0706** | **0.1885** |

76 gradient spikes. H100 80GB.

### exp20_6: Tiny UNet 17M

**Config**: UNet channels=[8,16,32,64,128,128], 6 levels, attention L4+L5.

| Checkpoint | Val Loss | Test MSE | Test PSNR | Test LPIPS | FID | KID | CMMD |
|-----------|----------|----------|-----------|-----------|-----|-----|------|
| Best | **0.00212** | 0.003290 | 33.06 | 0.5910 | 112.87 | 0.1394 | 0.2344 |
| Latest (500ep) | — | 0.004855 | 32.90 | 0.5191 | 94.75 | 0.1017 | 0.2306 |

254 gradient spikes. A100 80GB. 2.98h.

### exp20_7: 65M UNet (no attention)

**Config**: UNet channels=[8,16,32,128,256,256], no attention layers. Same as exp20_4 but with all attention disabled.

| Checkpoint | Val Loss | Test MSE | Test PSNR | Test LPIPS | FID | KID | CMMD |
|-----------|----------|----------|-----------|-----------|-----|-----|------|
| Best | 0.00222 | 0.003105 | 32.63 | 0.6132 | 112.64 | 0.1244 | 0.3127 |
| Latest (500ep) | — | 0.004998 | 33.26 | 0.5157 | 92.16 | 0.0977 | 0.2342 |

241 gradient spikes. A100 80GB. 10.42h.

### Model Scaling Summary

| Experiment | Channels | Params | Attention | Val Loss | FID (latest) | KID (latest) | CMMD (latest) | Grad Spikes |
|-----------|----------|--------|-----------|----------|-------------|-------------|--------------|-------------|
| exp1_1 (baseline) | [32,64,256,512,512] | 270M | L4+L5 | 0.00211 | 51.17 | 0.033 | 0.193 | — |
| exp20_5 | [12,24,48,192,384,384] | 152M | L4+L5 | 0.00238 | 72.11 | 0.071 | 0.189 | 2 |
| exp20_4 (w/attn) | [8,16,32,128,256,256] | 67M | L4+L5 | 0.00262 | 99.86 | 0.110 | 0.245 | 64 |
| exp20_7 (no attn) | [8,16,32,128,256,256] | 65M | none | 0.00222 | 92.16 | 0.098 | 0.234 | 56 |
| exp20_6 | [8,16,32,64,128,128] | 17M | L4+L5 | 0.00212 | 94.75 | 0.102 | 0.231 | 15 |

**Key findings:**
1. **270M → 152M drops FID from 51 to 72 (41% worse)**. Further scaling down gives diminishing returns — 67M and 20M are all in the 92-100 FID range.
2. **Val loss doesn't correlate with generation quality**: exp20_6 (20M) has the second-best val loss (0.00212) but worst FID (94.75). Small models memorize efficiently but generate poorly.
3. **Attention doesn't help at 67M**: exp20_7 (no attn, FID 92.16) slightly outperforms exp20_4 (with attn, FID 99.86). At this scale, attention parameters may be better spent on more channels.
4. **Gradient instability roughly scales inversely with model size**: 15-64 spikes (17-67M) vs 2 (152M) vs 0 (270M baseline).
5. **The 270M UNet is the right size** for this dataset. Scaling down saves VRAM but severely hurts generation quality.

---

## Part 12c: Combined Technique Experiments (March 2026)

### exp24: Combined Techniques @ 270M UNet, 1000 epochs — COMPLETED

**Config**: 270M UNet + ScoreAug + Adj. Offset + PosthocEMA + Uniform T. 256x256x160. Gradient checkpointing ON, torch.compile OFF. H100 80GB. 7 chain segments.

| Checkpoint | Val Loss | Test MSE | Test PSNR | Test LPIPS | FID | KID | CMMD |
|-----------|----------|----------|-----------|-----------|-----|-----|------|
| Best (EMA) | 0.00347 | 0.011215 | 34.24 | 0.7429 | 65.46 | 0.0542 | 0.2549 |
| Latest (1000ep, EMA) | — | 0.005543 | 32.70 | 0.8810 | **62.87** | **0.0526** | **0.2594** |

742 gradient spikes. Training: ~5 days total.

**Combined techniques ≈ ScoreAug alone.** FID 62.87 vs exp23's 62.57 — essentially identical. Adding adjusted offset noise + PosthocEMA + uniform timestep on top of ScoreAug did not improve generation quality at 1000 epochs. Val loss is worse (0.00347 vs 0.00200) and LPIPS is much worse (0.881 vs 0.531), suggesting the EMA evaluation may hurt reconstruction quality. The 742 gradient spikes (vs 139 for exp23) indicate the combined approach is less stable.

**TB trajectory**: KID still improving at epoch 1000 (min KID 0.027 at ep998, CMMD min 0.177 at ep961). The 3D metrics also improving (3D KID min 0.026 at ep980).

---

### exp25: Combined Techniques @ 17M UNet, 2000 epochs — COMPLETED

**Config**: 17.2M UNet [8,16,32,64,128,128] + ScoreAug + Adj. Offset + PosthocEMA + Uniform T + Attn Dropout + Weight Decay + warmup 10ep + gradient clip 0.5. 256x256x160. H100 80GB. 8 chain segments.

| Checkpoint | Val Loss | Test MSE | Test PSNR | Test LPIPS | FID | KID | CMMD |
|-----------|----------|----------|-----------|-----------|-----|-----|------|
| Best (EMA) | 0.00209 | 0.008342 | 33.54 | 0.7312 | 72.95 | 0.0579 | 0.3016 |
| Latest (2000ep, EMA) | — | 0.020070 | 34.20 | 0.6224 | 73.49 | 0.0565 | 0.3082 |

3,626 gradient spikes. Training: ~5 days total.

**The combined approach on a small model (17M) achieves FID ~73** — better than exp20_6 (17M vanilla, FID 95) but far from 270M results (FID 51-63). The massive gradient spike count (3,626 vs 742 for 270M exp24) confirms small models are much harder to train stably, even with gradient clipping at 0.5.

---

## Part 13: ControlNet Experiments (March 2026)

### exp6a_1: Stage 1 — Unconditional UNet @ 256x256x160

**Config**: 270M UNet, in=1 out=1, no seg conditioning. Train unconditional bravo generation.

| Checkpoint | Val Loss | Test MSE | Test PSNR | Test LPIPS |
|-----------|----------|----------|-----------|-----------|
| Best (ep 368) | 0.00202 | 0.003448 | 33.57 | 0.5714 |

500 epochs complete. 72 gradient spikes. Best val loss 0.00202 (ep 368).

---

### exp6b: Stage 2 — ControlNet @ 128x128x160

**Config**: Frozen UNet from exp6a (128x128), ControlNet encoder (~135M trainable). Seg mask injected via ControlNet residuals.

| Checkpoint | Val Loss | Test MSE | Test PSNR | Test LPIPS | FID | KID | CMMD |
|-----------|----------|----------|-----------|-----------|-----|-----|------|
| Best | 0.00304 | 0.006818 | 31.67 | 0.5036 | 52.98 | 0.0520 | 0.1799 |
| Latest (500ep) | — | 0.005155 | 31.26 | 0.5373 | **49.62** | **0.0470** | **0.1624** |

**Strong generation metrics (FID 49.62)** — ControlNet seg conditioning helps produce more realistic images.

---

### exp6b_1: Stage 2 — ControlNet @ 256x256x160 — COMPLETED

**Config**: Frozen UNet from exp6a_1 (256x256), ~135M trainable ControlNet encoder. Seg mask injected via ControlNet residuals.

| Checkpoint | Val Loss | Test MSE | Test PSNR | Test LPIPS | FID | KID | CMMD |
|-----------|----------|----------|-----------|-----------|-----|-----|------|
| Best | 0.00195 | 0.002549 | 32.99 | 0.6602 | 76.11 | 0.0749 | 0.2481 |
| Latest (500ep) | — | 0.002899 | 33.40 | 0.5920 | 82.82 | 0.0896 | 0.2242 |

0 gradient spikes. 500 epochs complete.

**Worse FID than 128x128 ControlNet (exp6b)**: FID 76-83 vs 50-53. The 256x256 resolution makes the generation task harder for the ControlNet while the frozen UNet backbone (from exp6a_1) may not be strong enough. Best val loss (0.00195) is excellent though.

---

## Part 14: Post-hoc Evaluation Results (March 2026)

### Optimal Steps (Golden Section Search, 25 vol, test split)

| Experiment | Metric | Best Steps | Best Value | Notes |
|-----------|--------|-----------|-----------|-------|
| exp1_1 (500ep) | FID | 23 | 23.85 | |
| exp1_1 (500ep) | FID_RIN | 46 | 0.674 | |
| exp1_1 (1000ep) | FID | 27 | **19.12** | Best overall |
| exp1_1 (1000ep) | FID_RIN | 49 | 0.714 | |
| exp1_1 (1000ep v2) | FID | 32 | **20.84** | Repeat run, consistent |
| exp1_1 (1000ep v2) | FID_RIN | 79 | **0.663** | Repeat run, consistent |
| **exp23 (1000ep)** | **FID** | **27** | **20.38** | ScoreAug, within 1.3 of baseline |
| **exp23 (1000ep)** | **FID_RIN** | **48** | **0.659** | **Best RadImageNet FID** |
| exp1b (128, [-1,1]) | FID | 28 | 51.90 | |
| exp1c (128, N(0,1)) | FID | 50 | 142.36 | Gaussian norm fails |

**exp1_1 v2 repeat confirms findings**: The repeat run of exp1_1 at 1000 epochs found FID 20.84 at 32 steps (vs 19.12 at 27 steps in original) and RadImageNet FID 0.663 at 79 steps (vs 0.714 at 49 steps). Small differences are expected with 25-volume evaluation variance, but both runs confirm the same pattern: ImageNet FID optimal at ~27-32 steps, RadImageNet FID improves up to ~50-79 steps. PCA pass rate is 100% across all step counts evaluated.

**ImageNet FID vs RadImageNet FID disagree on optimal steps**: For exp23, ImageNet prefers 27 steps (FID 20.38) while RadImageNet prefers 48 steps (FID 0.659). ImageNet FID degrades sharply beyond 27 steps while RadImageNet FID keeps improving. CMMD is flat (~0.110) across all step counts. Practical choice: **27-32 steps** — best ImageNet FID with acceptable RadImageNet.

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
| exp23 ScoreAug | 0.059 | -50.8% | 1000 | Completed, matched/beat exp1_1 1000ep |
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
| exp1s weight decay (128) | 0.042 | +1.1% | flat |
| exp1r attn dropout (128) | 0.051 | +2.8% | flat |

### CMMD Trajectory Rankings (Generation/CMMD_val, top 10)

| Experiment | CMMD Mean | Slope | Trend |
|-----------|----------|-------|-------|
| exp23 ScoreAug | 0.235 | -32.6% | strong improve |
| exp1p_1 uniform T | 0.243 | -22.9% | strong improve |
| exp20_1 656M+L3 | 0.260 | -22.5% | strong improve |
| exp1_1 baseline | 0.250 | -12.5% | strong improve |
| exp6a_1 CtrlNet S1 | 0.175 | -1.5% | flat (lowest absolute) |
| exp1o_1 PosthocEMA | 0.182 | -8.9% | improving |
| exp1r attn dropout (128) | 0.191 | -7.2% | improving |
| exp6b CtrlNet S2 (128) | 0.189 | +1.6% | flat |
| exp1s weight decay (128) | 0.190 | -3.4% | improving |
| exp1l_1 adj.offset | 0.207 | -3.0% | flat |

### Key Trajectory Insights

1. **exp6a_1 ControlNet S1** has the lowest absolute KID (0.026) and CMMD (0.175) while still strongly improving — the unconditional generation baseline is the single best performer on in-training metrics.
2. **exp23 ScoreAug** completed 1000 epochs with the steepest improvement slopes. Post-hoc FID 20.38 confirms generation quality matches baseline. ScoreAug prevents overfitting while maintaining quality.
3. **exp1s/1r ran at 128x128 with flat trajectories**: Weight decay (FID 43.45) and attn dropout (FID 49.32) at 128x128x160. Their KID trajectories are flat (+1.1% and +2.8%), suggesting they've converged. Not tested at 256x256.
4. **exp1o_1 PosthocEMA** shows consistent improvement across all metrics but at a moderate rate — already competitive.
5. **Overfitting is visible**: exp1j_1 (grad accum), exp19_paper (WDM), and exp1c_1 (brain norm) all show degrading KID trajectories.
6. **128x128 experiments** (exp1h, exp1k, exp1e, exp1g) have the lowest absolute KID values (0.024-0.026) because 128x128 generation is fundamentally easier.
7. **No in-training generation metrics exist for LDM experiments** (exp21/22) — those results come only from test-time evaluation.

### Cross-Domain Comparison: Test FID Rankings

**Post-hoc evaluation (25 volumes, optimal Euler steps, test split):**

| Rank | Experiment | Type | FID | FID_RIN | Steps |
|------|-----------|------|-----|---------|-------|
| 1 | exp1_1 (1000ep) | Pixel | **19.12** | 0.714 | 27 |
| 2 | exp23 ScoreAug (1000ep) | Pixel | **20.38** | **0.659** | 27 |
| 3 | exp1_1 (500ep) | Pixel | 23.85 | 0.674 | 23 |

**In-training evaluation (4 volumes, 25 Euler steps) — latest checkpoint, 256x256x160 only:**

| Rank | Experiment | Type | FID | KID | CMMD |
|------|-----------|------|-----|-----|------|
| 1 | exp22_2 DiT-L | LDM | **47.41** | 0.036 | 0.252 |
| 2 | exp22_1 DiT-B | LDM | 48.99 | 0.038 | 0.266 |
| 3 | exp1_1 Baseline (1000ep) | Pixel | 49.53 | 0.035 | **0.149** |
| 4 | exp21_2 MAISI UNet | LDM | 50.89 | 0.041 | 0.174 |
| 5 | exp1_1 Baseline (500ep) | Pixel | 51.17 | **0.033** | 0.193 |
| 6 | exp1p_1 Uniform T | Pixel | 58.85 | 0.046 | 0.198 |
| 7 | exp22_3 DiT-S (2000ep) | LDM | 61.14 | 0.052 | 0.283 |
| 8 | exp23 ScoreAug (1000ep) | Pixel | 62.57 | 0.053 | 0.185 |
| 9 | exp1o_1 PosthocEMA | Pixel | 62.64 | 0.054 | 0.190 |
| 10 | exp24 Combined (1000ep) | Pixel | 62.87 | 0.053 | 0.259 |
| 11 | exp19_2 WDM 270M | WDM | 67.32 | 0.044 | 0.235 |

**128x128x160 in-training (separate ranking, not comparable to above):**

| Experiment | FID | KID | CMMD |
|-----------|-----|-----|------|
| exp1s Weight decay | 43.45 | 0.032 | 0.168 |
| exp6b CtrlNet S2 | 49.62 | 0.047 | 0.162 |
| exp1r Attn dropout | 49.32 | 0.047 | 0.163 |

**Caveat**: FID numbers are NOT directly comparable across evaluation setups or resolutions. Post-hoc eval uses 25 volumes with optimal Euler steps. In-training eval uses 4 volumes with fixed 25 steps. LDM test eval uses val reference fallback. 128x128 FID is fundamentally different from 256x256 FID.

---

## Part 17: Multi-Modality Experiments (March-April 2026)

### exp1v2: Dual (T1 pre + T1 gd) @ 128x128x160

**Config**: 270M UNet, rflow, dual mode (in=3: noisy_t1_pre + noisy_t1_gd + seg, out=2), 500 epochs.

| Checkpoint | MS-SSIM | PSNR | LPIPS | FID | KID | CMMD | Spikes |
|-----------|---------|------|-------|-----|-----|------|--------|
| Best | 0.9439 | 33.44 | 0.7884 | **32.80** | **0.0216** | **0.1715** | 13 |
| Latest (500ep) | 0.9418 | 32.39 | 0.6395 | 35.91 | 0.0299 | 0.1756 | |

**First dual-modality experiment.** Jointly generates T1 pre-contrast and T1 post-gadolinium conditioned on segmentation mask. Strong FID (32.80) at 128x128.

### exp1v2_1: Dual + Joint Normalization @ 128x128x160

**Config**: Same as exp1v2 but with joint_normalization=true (all channels normalized together per patient).

| Checkpoint | MS-SSIM | PSNR | LPIPS | FID | KID | CMMD | Spikes |
|-----------|---------|------|-------|-----|-----|------|--------|
| Best | 0.9559 | 32.84 | 0.5267 | **24.30** | **0.0097** | 0.2267 | 11 |
| Latest (500ep) | 0.9261 | 30.64 | 0.6767 | 44.54 | 0.0376 | 0.1839 | |

**Best-checkpoint FID 24.30 is remarkable for 128x128.** Joint normalization preserves relative intensity relationships between T1 pre and T1 gd. However, latest FID (44.54) is much worse than best (24.30), suggesting overfitting.

### exp1v3: Triple (T1 pre + T1 gd + FLAIR) @ 128x128x160

**Config**: 270M UNet, rflow, triple mode (in=4: noisy_t1_pre + noisy_t1_gd + noisy_flair + seg, out=3), 500 epochs.

| Checkpoint | MS-SSIM | PSNR | LPIPS | FID | KID | CMMD | Spikes |
|-----------|---------|------|-------|-----|-----|------|--------|
| Best | 0.9370 | 32.47 | 0.9223 | 65.37 | 0.0642 | 0.2506 | 51 |
| Latest (500ep) | 0.9475 | 32.93 | 0.6537 | 79.68 | 0.0887 | 0.3276 | |

Triple mode is harder than dual — FID 65.37 vs 32.80. LPIPS on best checkpoint (0.9223) is very high, suggesting best-ckpt reconstruction is poor despite good generation metrics. 51 gradient spikes — more unstable than dual (13).

### exp1v3_1: Triple + Joint Normalization @ 128x128x160

**Config**: Same as exp1v3 but with joint_normalization=true.

| Checkpoint | MS-SSIM | PSNR | LPIPS | FID | KID | CMMD | Spikes |
|-----------|---------|------|-------|-----|-----|------|--------|
| Best | 0.9434 | 32.16 | 0.7475 | 66.57 | 0.0735 | 0.3145 | 19 |
| Latest (500ep) | 0.9412 | 32.04 | 0.6518 | 80.54 | 0.0890 | 0.3843 | |

Joint normalization does NOT help triple mode (FID 66.57 vs 65.37). Fewer gradient spikes (19 vs 51).

### Multi-Modality Summary

| Experiment | Mode | Norm | FID (best) | KID (best) | CMMD (best) | Spikes |
|-----------|------|------|-----------|-----------|------------|--------|
| **exp1v2_1** | **Dual** | **Joint** | **24.30** | **0.0097** | 0.2267 | 11 |
| exp1v2 | Dual | Independent | 32.80 | 0.0216 | **0.1715** | 13 |
| exp1v3 | Triple | Independent | 65.37 | 0.0642 | **0.2506** | 51 |
| exp1v3_1 | Triple | Joint | 66.57 | 0.0735 | 0.3145 | 19 |

Joint normalization helps dual mode (FID 24→33) but not triple. Dual mode outperforms triple significantly.

---

## Part 18: LDM ScoreAug Experiments (March 2026)

### exp27: LDM DiT-L + ScoreAug, 1000 epochs

**Config**: DiT-L 478M, VQ-VAE 4x, ScoreAug (D-axis rotation only), torch.compile OFF, gradient_checkpointing ON. 8 chain segments.

| Checkpoint | MS-SSIM | PSNR | LPIPS | FID | KID | CMMD | Spikes |
|-----------|---------|------|-------|-----|-----|------|--------|
| Best | 0.9864 | 33.30 | 0.1150 | 60.02 | 0.0493 | 0.2899 | 0 |
| Latest (1000ep) | 0.9863 | 33.96 | 0.1160 | **57.24** | **0.0477** | **0.2514** | |

**ScoreAug did NOT improve DiT-L.** FID 57.24 is worse than exp22_2 (47.41 at 500ep). Likely because torch.compile was disabled (ScoreAug's dynamic shapes break Triton), reducing training efficiency. Zero gradient spikes though — very stable.

### exp28: LDM MAISI UNet + ScoreAug, 1000 epochs

**Config**: UNet MAISI 167M, VQ-VAE 4x, ScoreAug. 2 chain segments.

| Checkpoint | MS-SSIM | PSNR | LPIPS | FID | KID | CMMD | Spikes |
|-----------|---------|------|-------|-----|-----|------|--------|
| Best | 0.9830 | 31.39 | 0.1553 | 131.66 | 0.1368 | 0.3866 | 20 |
| Latest (1000ep) | 0.9895 | 33.44 | 0.0970 | **79.91** | **0.0786** | **0.2612** | |

**ScoreAug hurt MAISI UNet.** FID 79.91 at 1000ep is much worse than exp21_2 (50.89 at 500ep). Best-checkpoint FID (131.66) is catastrophic. ScoreAug's augmentations may be too aggressive for the smaller 167M model in latent space.

### exp28_1: LDM MAISI + ScoreAug + Diffusion Mixup, 1000 epochs

**Config**: Same as exp28 + diffusion mixup (alpha=0.2, prob=0.5).

| Checkpoint | MS-SSIM | PSNR | LPIPS | FID | KID | CMMD | Spikes |
|-----------|---------|------|-------|-----|-----|------|--------|
| Best | 0.9868 | 32.68 | 0.1136 | 168.88 | 0.1857 | 0.3986 | 13 |
| Latest (1000ep) | 0.9812 | 32.74 | 0.1553 | **98.56** | **0.0960** | **0.3725** | |

**Mixup made things worse.** FID 98.56 is much worse than exp28 (79.91). Diffusion mixup in latent space with batch_size=1 may not provide useful interpolations.

### exp28_2: LDM MAISI + ScoreAug v2, 1000 epochs

**Config**: Same as exp28 but with ScoreAug v2 (stronger: stacked non-destructive + destructive transforms).

| Checkpoint | MS-SSIM | PSNR | LPIPS | FID | KID | CMMD | Spikes |
|-----------|---------|------|-------|-----|-----|------|--------|
| Best | 0.9905 | 35.24 | 0.0970 | 93.32 | 0.1057 | 0.2819 | 16 |
| Latest (1000ep) | 0.9836 | 33.51 | 0.1540 | **80.17** | **0.0776** | **0.2945** | |

Stronger augmentation (v2) is similar to v1 (FID 80.17 vs 79.91). Best checkpoint has very good reconstruction (PSNR 35.24) but poor generation.

### LDM ScoreAug Summary

| Experiment | Architecture | ScoreAug | FID (latest) | vs Baseline |
|-----------|-------------|----------|-------------|-------------|
| exp22_2 (baseline) | DiT-L 478M | None | **47.41** | — |
| exp27 | DiT-L 478M | Yes (1000ep) | 57.24 | +9.8 (worse) |
| exp21_2 (baseline) | MAISI 167M | None | **50.89** | — |
| exp28 | MAISI 167M | Yes (1000ep) | 79.91 | +29.0 (worse) |
| exp28_1 | MAISI 167M | Yes + Mixup | 98.56 | +47.7 (worse) |
| exp28_2 | MAISI 167M | Yes v2 | 80.17 | +29.3 (worse) |

**ScoreAug does NOT help in latent space.** It works for pixel-space (exp23 improved over 1000ep) but hurts LDM. Likely reasons: (1) latent space is already compact and augmentations are too destructive, (2) torch.compile disabled for DiT (training efficiency loss), (3) 167M UNet may be too small for the added augmentation complexity.

---

## Part 19: WDM Extended Experiments (March 2026)

### exp26: WDM 270M + ScoreAug, 1000 epochs

**Config**: Same as exp19_2 (best WDM) + ScoreAug, 1000 epochs. DDPM x0, per-subband N(0,1).

| Checkpoint | MS-SSIM | PSNR | LPIPS | Spikes | Status |
|-----------|---------|------|-------|--------|--------|
| Best | 0.9661 | 36.27 | 0.2521 | 1 | Gen metrics missing |

Generation metrics not available — test eval likely crashed during metric computation. Training MSE ~0.35 at epoch 950 suggests the model may have partially collapsed.

### exp26_1: WDM 270M vanilla, 1000 epochs

**Config**: Same as exp19_2, extended to 1000 epochs without ScoreAug.

| Checkpoint | MS-SSIM | PSNR | LPIPS | FID | KID | CMMD | Spikes |
|-----------|---------|------|-------|-----|-----|------|--------|
| Best | 0.9514 | 34.81 | 0.2612 | **79.03** | **0.0557** | **0.2320** | 3 |
| Latest (1000ep) | 0.9430 | 34.30 | 0.2602 | 77.28 | 0.0511 | 0.2174 | |

WDM at 1000ep (FID 77.28) is worse than at 500ep (exp19_2: FID 67.32). WDM is overfitting beyond 500 epochs — the KID trajectory was flat at 500ep, and now generation quality has degraded.

---

## Part 20: Additional Wavelet Experiments (Feb 2026)

### exp12_5: WDM Medium 250M + DDPM x0 @ 128x128x160

**Config**: UNet [64,128,256,512,512] ~250M, DDPM x0, [-1,1] rescale + raw DWT, attention at L3/L4, warmup=10, grad_clip=0.5.

| Checkpoint | MS-SSIM | PSNR | LPIPS | FID | KID | CMMD | Spikes |
|-----------|---------|------|-------|-----|-----|------|--------|
| Best | 0.8986 | 30.10 | 0.7983 | **71.11** | **0.0443** | **0.3321** | 18 |
| Latest (500ep) | 0.8403 | 28.19 | 0.7647 | — | — | — | |

Scaled up from exp12_4 (~77M) to 250M. FID 71.11 at 128x128 is competitive with exp19_2 (67.32 at 256x256).

### exp12_6: WDM DiT-S + DDPM x0 @ 128x128x160

**Config**: DiT-S (~40M), DDPM x0, [-1,1] rescale + raw DWT, gradient_checkpointing=true. Same as exp19_6 but at 128x128.

| Checkpoint | MS-SSIM | PSNR | LPIPS | FID | KID | CMMD | Spikes |
|-----------|---------|------|-------|-----|-----|------|--------|
| Best | 0.8980 | 30.19 | 0.6533 | **59.49** | **0.0385** | 0.3065 | 65 |
| Latest (500ep) | — | — | — | 60.42 | — | 0.3124 | |

DiT-S at 128x128 (FID 59.49) outperforms the 256x256 version exp19_6 (FID 110.88) — suggesting the 256x256 DiT-S was severely undertrained.

---

## Part 21: Additional Seg Experiments (Feb 2026)

### exp2e: Seg Multi-level Auxiliary Bin Prediction @ 128x128x160

**Config**: 270M UNet, rflow, seg_cond_3d, multi-level auxiliary bin prediction loss.

500 epochs completed. 411 gradient spikes — extremely unstable. Generation metrics not available (possibly crashed). Not a viable approach due to instability.

---

## Key Takeaways (Updated April 7, 2026)

1. **Pixel-space with post-hoc eval produces the best absolute FID**: exp1_1 at 1000ep achieves FID 19.12 (27 Euler steps). exp23 (ScoreAug) is close at 20.38, with better RadImageNet FID (0.659 vs 0.714). In-training FID is misleadingly high — post-hoc evaluation with more volumes and optimal steps reveals the true quality.

2. **LDM DiT is the most efficient approach**: exp22_2 (DiT-L 478M) achieves FID 47.41 in only 3.53h of training. DiT models in latent space train 10x faster than pixel UNets while achieving competitive generation quality.

3. **ScoreAug alone ≈ combined techniques**: exp24 (ScoreAug + AdjOffset + PosthocEMA + UniformT, 1000ep) achieved FID 62.87, essentially identical to exp23 (ScoreAug only, FID 62.57). Stacking techniques did not improve generation quality — but added instability (742 vs 9 gradient spikes). ScoreAug is the dominant contributor.

4. **ScoreAug helps pixel-space but HURTS latent-space**: exp23 (pixel, 1000ep) improved to FID 20.38. But exp27 (DiT-L LDM + ScoreAug, 1000ep) degraded to 57.24 vs baseline 47.41. exp28 (MAISI UNet LDM + ScoreAug) was even worse (79.91 vs 50.89). ScoreAug augmentations are too destructive for compact latent representations. Diffusion Mixup (exp28_1) and ScoreAug v2 (exp28_2) also failed in latent space.

5. **Weight decay and attn dropout ran at 128x128, not 256x256**: exp1r/1s results (FID 43-49) are not comparable to 256x256 experiments.

6. **In-training FID and post-hoc FID tell different stories**: In-training eval (4 vol, 25 steps) gives very different rankings than post-hoc eval (25 vol, optimal steps). Always do post-hoc evaluation for final claims.

7. **Model size matters more than techniques**: 270M→152M drops FID from 51→72 (41% worse). 65-67M and 17M models are all FID 92-100. Combined techniques on 17M (exp25, FID 73) can't compensate for the capacity gap.

8. **ControlNet S1 has the best in-training trajectory metrics**: exp6a_1 achieves the lowest absolute KID (0.026) and CMMD (0.175) while still strongly improving.

9. **Adjusted offset noise improves val loss**: exp1l_1 achieves val loss 0.00209 with strong KID trajectory (-23.6%/100ep).

10. **Wavelet domain works but pixel is better**: exp19_2 (WDM, FID 67.32) is the best wavelet result. RFlow velocity prediction fails in wavelet space — DDPM x0 is required.

11. **Optimal Euler steps: 27 for ImageNet FID, ~48 for RadImageNet FID**: The two metrics disagree on step count. CMMD is flat across step counts (~0.110). Practical choice: 27 steps.

12. **CFG-Zero* doesn't help**: Post-hoc CFG scale sweep shows scale=1.0 (no guidance) is optimal (FID 44.79). All scales >1.0 dramatically worse (FID 155+).

13. **FreeU and sampling tricks provide marginal gains**: FreeU (<1 FID point), Restart Sampling (no improvement), DiffRS (negative impact).

14. **Time-shift ratio 2.0 is optimal**: MONAI's default (~6.84) is harmful. Ratio 2.0 gives 5.5 FID improvement.

15. **Generation metric trajectories matter more than val loss**: Val loss ranking does not predict generation quality. exp20_6 (17M) has val loss 0.00212 (near best) but FID 94.75 (near worst). Always evaluate with generation metrics.

16. **Infrastructure remains the biggest bottleneck**: Checkpoint corruption, OOM kills, disk quota, torch.load bugs, torch.compile+ScoreAug incompatibility, and stalled chains have wasted more GPU hours than any hyperparameter choice.

17. **Dual mode works well, triple mode is harder**: exp1v2_1 (dual + joint norm) achieves FID 24.30 at 128x128 — excellent for multi-modality. Triple mode (exp1v3, FID 65.37) is significantly worse. Joint normalization helps dual but not triple.

18. **WDM overfits beyond 500 epochs**: exp26_1 (WDM 1000ep, FID 77.28) is worse than exp19_2 (WDM 500ep, FID 67.32). WDM was already converged at 500 epochs and degrades with longer training.
