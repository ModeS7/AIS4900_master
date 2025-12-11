# Pure BF16 Training Attempts

Documentation of attempts to make pure bf16 training work for diffusion models.

**Conclusion: Pure bf16 training does not produce good generation quality for diffusion models.**

## Background

Goal was to enable pure bf16 training for compatibility with NVIDIA 2:4 structured sparsity, which requires bf16 weights (not fp32 master weights with bf16 compute like AMP).

## What Works vs What Doesn't

| Aspect | Mixed Precision (AMP) | Pure BF16 |
|--------|----------------------|-----------|
| Training loss | Decreases normally | Decreases normally |
| Generation quality | Good | Black/white noise |
| Memory usage | Higher (fp32 weights) | Lower (~50% less) |
| 2:4 sparsity compatible | No (fp32 weights) | Yes (bf16 weights) |

## Attempts Made

### 6. Stochastic Rounding with Torchastic Library

**Theory:** Research paper ["Revisiting BFloat16 Training"](https://arxiv.org/abs/2010.06192) showed that stochastic rounding (instead of nearest rounding) preserves small gradient updates and achieves near-parity with fp32 training (0.1% lower to 0.2% higher accuracy).

**Changes:**
```python
# pip install torchastic
from torchastic import AdamW as StochasticAdamW, StochasticAccumulator

# In setup_model:
self.optimizer = StochasticAdamW(self.model_raw.parameters(), lr=self.learning_rate)
StochasticAccumulator.assign_hooks(self.model_raw)

# In train_step (after backward, before optimizer.step):
StochasticAccumulator.reassign_grad_buffer(self.model_raw)
```

**Result:** Still black/white images. Stochastic rounding did not fix the generation quality issue.

**Why it didn't work:** The research showing stochastic rounding success was on classification networks (ViT, ResNet), not generative models. Diffusion models may have fundamentally different precision requirements due to:
- Iterative denoising (100+ steps compound errors)
- Continuous output values (vs discrete classification)
- Velocity/noise predictions need high precision for clean reconstruction

---

### 1. Basic Pure BF16 Implementation

**Changes:**
- Added `--pure_bf16` flag to train.py
- Model and perceptual loss initialized with `dtype=torch.bfloat16`
- Disabled autocast in train_step (not needed when model is bf16)
- Cast inputs to bf16 before forward pass

**Result:** Training loss decreased normally, but validation images were black/white.

### 2. FP32 Model Copy for Generation

**Theory:** bf16 precision errors compound over 100 iterative denoising steps.

**Changes:**
```python
# In generate_validation_samples:
if self.pure_bf16:
    model_to_use = copy.deepcopy(base_model).float()
```

**Result:** Still black/white. The model learned bf16 representations during training that don't transfer to fp32.

### 3. Keep Autocast During Generation

**Theory:** Let the bf16 model run in bf16 (as trained), only keep scheduler math in fp32.

**Changes:**
```python
# In strategies.py generate():
with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
    velocity_pred = model(current_model_input, timesteps=timesteps_batch)

# Scheduler operations in fp32
velocity_pred = velocity_pred.float()
noisy_images, _ = self.scheduler.step(velocity_pred, t, noisy_images, next_timestep)
```

**Result:** Still black/white.

### 4. Velocity Convention Investigation

**Theory:** Maybe MONAI's RFlowScheduler uses opposite velocity convention.

**Attempted change:**
```python
# Changed from:
velocity_target = target_images - noise  # clean - noise

# To:
velocity_target = noise - target_images  # noise - clean
```

**Result:** Reverted - generation worked fine with original convention in mixed precision, so convention was not the issue.

### 5. Debug Analysis

Added debug prints to understand what's happening:

```
[RFlow Debug] Model dtype: torch.float32
[RFlow Debug] Initial noisy_pre: min=-3.7402, max=3.8946, mean=0.0045
[RFlow Debug] Step 0, t=100.0000: velocity min=-2.6901, max=2.7142, mean=0.0375
[RFlow Debug] Step 99, t=1.0000: velocity min=-2.7270, max=2.8327, mean=0.0494
[RFlow Debug] Final noisy_pre: min=-1.3065, max=1.5536, mean=0.0514
```

**Observations:**
- Final output range [-1.3, 1.5] instead of [0, 1]
- Mean ~0.05 instead of ~0.5 (expected for medical images)
- Values outside [0,1] get clamped → black (< 0) and white (> 1)

## Root Cause Analysis

### Why Mixed Precision Works

```
Training:
  Master weights: fp32 (full precision)
  Forward pass: bf16 (fast)
  Gradients: bf16 → accumulated in fp32
  Weight updates: Applied to fp32 weights (no precision loss)

Generation:
  Model weights: fp32
  Forward pass: bf16 via autocast
  Scheduler math: fp32
  Result: Good quality
```

### Why Pure BF16 Fails

```
Training:
  Weights: bf16 (reduced precision)
  Forward pass: bf16
  Gradients: bf16
  Weight updates: Applied to bf16 weights → SMALL UPDATES LOST/ROUNDED

Generation:
  Model weights: bf16 (already degraded from training)
  Any dtype for forward pass
  Result: Poor quality (model learned degraded representations)
```

The fundamental issue: **bf16 weight updates lose precision during training**. Small gradient updates get rounded away. Over thousands of training steps, the model learns a degraded function compared to fp32 training.

## Why Diffusion Models Are Particularly Sensitive

1. **Iterative refinement**: Generation requires 10-1000 denoising steps. Small errors compound.
2. **Continuous outputs**: Unlike LLMs (discrete tokens), diffusion models output continuous pixel values.
3. **Precision requirements**: The velocity/noise predictions need high precision to reconstruct clean images.

## Alternative Approach for 2:4 Sparsity

If 2:4 sparsity is needed:

1. **Train with mixed precision (AMP)** - maintains fp32 master weights
2. **After training, convert to bf16** for inference
3. **Apply 2:4 sparsity** to the bf16 inference model

This gives you:
- High quality training (fp32 weights)
- bf16 inference weights (2:4 sparsity compatible)
- Sparse inference acceleration on Ampere+ GPUs

Note: 2:4 sparsity primarily accelerates nn.Linear layers (Transformers), not nn.Conv2d (UNet). For UNet-based diffusion models, the speedup would be minimal anyway.

## Final Conclusion

**Pure bf16 training does not work for diffusion models** with current techniques. We tried:

1. Basic pure bf16 ❌
2. FP32 model copy for generation ❌
3. Autocast during generation ❌
4. Velocity convention changes ❌ (wrong direction)
5. Debug analysis (identified output range issue)
6. Stochastic rounding (torchastic) ❌

The fundamental issue is that **diffusion model generation is uniquely sensitive to precision**:
- Unlike classification (discrete outputs), diffusion produces continuous values
- Unlike single-pass inference, diffusion requires 10-1000 iterative steps
- Small precision errors compound multiplicatively over these steps

**For 2:4 sparsity with diffusion models, the only viable path is:**
1. Train with mixed precision (AMP) - fp32 master weights
2. After training, quantize/convert to bf16
3. Apply 2:4 pruning to the bf16 weights
4. Accept potential quality degradation at inference time

This is similar to how LLMs are quantized post-training (e.g., GPTQ, AWQ).

## Untried Options (For Future Work)

### Options Still Available

| Option | 2:4 Sparsity Compatible? | Effort | Notes |
|--------|-------------------------|--------|-------|
| Kahan + Stochastic | **Yes** | Low | Paper used both together, we only tried stochastic |
| Selective fp32 layers | **Partial** | Medium | Only bf16 layers get sparse acceleration |
| FP32 gradient accumulation | **Yes** | Medium | Training trick, weights still bf16 |
| Distillation (fewer steps) | **Yes** | High | 1-4 steps = less error compounding |
| DiT architecture | **Yes + Better** | High | nn.Linear IS accelerated by 2:4 sparsity |
| Post-training quantization | **Yes** | Medium | Quality loss from quantization |

### Critical Insight: UNet vs DiT for 2:4 Sparsity

**2:4 structured sparsity only accelerates `nn.Linear` layers, NOT `nn.Conv2d`.**

| Architecture | Primary Layers | 2:4 Sparsity Benefit |
|--------------|---------------|---------------------|
| UNet | Conv2d | **Minimal** - most ops not accelerated |
| DiT (Diffusion Transformer) | Linear | **Significant** - most ops accelerated |

**Recommendation:** When implementing DiT, revisit pure bf16 training. Transformers:
1. Handle bf16 better than CNNs
2. Actually benefit from 2:4 sparsity (Linear layers)
3. Have been successfully trained in bf16 (LLMs prove this)

### Implementation Notes for DiT + BF16

When implementing DiT, try:
```python
# 1. Stochastic rounding + Kahan summation (both together)
from torchastic import AdamW as StochasticAdamW, StochasticAccumulator

# 2. Fewer denoising steps via distillation (reduces error accumulation)
# - Consistency Models
# - Progressive Distillation
# - Rectified Flow with few-step training

# 3. If still fails: post-training quantization
# Train with AMP → quantize to bf16 → apply 2:4 pruning
```

## Files Modified (to be reverted)

- `train.py`: Added `--pure_bf16` argument
- `trainer.py`: Added `pure_bf16` parameter, stochastic optimizer, conditional dtype handling
- `strategies.py`: Various generation dtype experiments

## References

- [NVIDIA 2:4 Sparsity](https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/)
- [Revisiting BFloat16 Training](https://arxiv.org/abs/2010.06192) - Stochastic rounding + Kahan summation
- [Torchastic Library](https://github.com/lodestone-rock/torchastic) - Stochastic bf16 optimizer
- [AdamW-BF16](https://github.com/AmericanPresidentJimmyCarter/adamw-bf16) - Alternative stochastic rounding implementation
- [Min-SNR Loss Weighting](https://arxiv.org/abs/2303.09556)
- [Rectified Flow](https://arxiv.org/abs/2209.03003)
- [Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748)
