# ODE Solver Evaluation Results

**Date:** 2026-02-19
**Model:** 3D Pixel-space BRAVO RFlow (exp1_1)
**GPU:** NVIDIA A100-PCIE-40GB
**Volume:** 256x256x160 (10.5M voxels)
**Conditioning:** 25 real segmentation masks from val split
**Noise:** Deterministic (seed=42), shared across all configs
**Reference splits:** train (105 vols), test (51 vols), test_new, val, all (combined)

---

## Key Findings

### 1. Euler is the clear winner at every NFE budget

Euler outperforms all other solvers on FID and KID at every comparable NFE level. This is consistent with RFlow/flow matching literature where the model is trained with Euler discretization, so the learned velocity field implicitly compensates for Euler's discretization error.

### 2. Optimal config: euler/25 steps (NFE=25, 19s/vol)

| Config | NFE | FID | KID | CMMD | Time/vol |
|--------|-----|-----|-----|------|----------|
| euler/25 | 25 | **27.50** | **0.0240** | 0.1131 | 19.0s |
| euler/50 | 50 | 30.25 | 0.0275 | **0.1095** | 38.0s |
| euler/10 | 10 | 34.20 | 0.0328 | 0.1493 | 7.6s |

euler/25 achieves the best FID and KID. euler/50 gives marginally better CMMD but worse FID/KID at 2x the cost.

### 3. Quality DEGRADES beyond 25 Euler steps

| Euler steps | NFE | FID (vs all) |
|-------------|-----|-------------|
| 5 | 5 | 59.84 |
| 10 | 10 | 34.20 |
| **25** | **25** | **27.50** |
| 50 | 50 | 30.25 |
| 100 | 100 | 43.39 |

FID improves from 5->25 steps, then gets worse. This is a known phenomenon: the Euler solver at training-matched step counts outperforms the "true" ODE solution because the velocity field was learned to compensate for Euler discretization error. More steps = closer to the imperfect ODE = worse quality.

### 4. Higher-order fixed-step solvers are worse than Euler

At the same NFE budget (vs 'all' reference):

| NFE ~10 | FID | | NFE ~25 | FID | | NFE ~50 | FID |
|---------|-----|---|---------|-----|---|---------|-----|
| euler/10 | **34.20** | | euler/25 | **27.50** | | euler/50 | **30.25** |
| midpoint/5 | 34.77 | | midpoint/10 | 38.76 | | midpoint/25 | 37.53 |
| heun3/5 | 44.32 | | heun3/10 | 42.33 | | dopri5(1e-3) | 50.00 |
| heun2/5 | 233.61 | | rk4/5 | 95.26 | | rk4/10 | 82.30 |

Midpoint is competitive at low NFE but falls behind as steps increase. heun2 is consistently broken (likely numerical issues with 3D volumes).

### 5. Adaptive solvers perform poorly

All adaptive solvers (fehlberg2, bosh3, dopri5, dopri8) are significantly worse than Euler at comparable NFE. They accurately solve the ODE but the ODE itself is imperfect, so accuracy doesn't help.

Worst offenders:
- **fehlberg2:** FID=469 at tol=1e-3 (only 10 NFE). Catastrophically bad at low tolerances.
- **dopri8:** FID=98 at tol=1e-2 (121 NFE). Very expensive and poor quality.
- **dopri5:** FID=212 at tol=1e-2 (44 NFE). Bad.

### 6. CMMD converges faster than FID/KID

CMMD reaches ~0.11 by NFE~25 and plateaus. FID continues to change. This suggests CMMD measures a different aspect of distribution quality (semantic similarity via CLIP) that is satisfied earlier than the pixel-level statistics captured by FID.

### 7. Results are consistent across all reference splits

The ranking of solvers is identical regardless of which reference split is used (train, test, test_new, val, all). Absolute FID values vary by 2-5 points between splits but relative ordering is stable.

---

## Recommended Configurations

| Use case | Config | NFE | FID | Time/vol | Notes |
|----------|--------|-----|-----|----------|-------|
| **Production** | euler/25 | 25 | 27.5 | 19s | Best quality |
| **Fast preview** | euler/10 | 10 | 34.2 | 7.6s | ~2.5x faster, +24% FID |
| **Quick test** | euler/5 | 5 | 59.8 | 3.9s | Rough but usable |

There is no reason to use more than 25 Euler steps or any solver other than Euler for this model.

---

## Full Results Table (vs 'all' reference, sorted by NFE)

```
Config                     NFE/vol  Time/vol     FID      KID          CMMD
euler/5                         5      3.9s    59.84  0.062787     0.202311
fehlberg2(tol=1e-2)             8      5.8s   198.26  0.289727     0.435310
euler/10                       10      7.6s    34.20  0.032781     0.149327
midpoint/5                     10      7.7s    34.77  0.035300     0.126524
heun2/5                        10      7.6s   233.61  0.349224     0.507339
fehlberg2(tol=1e-3)            10      7.3s   468.98  0.716953     0.734491
fehlberg2(tol=1e-4)            13      9.7s   325.64  0.468627     0.617411
heun3/5                        15     11.3s    44.32  0.050242     0.118040
midpoint/10                    20     15.2s    38.76  0.039158     0.112641
heun2/10                       20     16.8s   144.79  0.213568     0.325269
rk4/5                          20     14.4s    95.26  0.127412     0.211555
euler/25                       25     19.0s    27.50  0.023947     0.113110  <-- BEST
heun3/10                       30     22.6s    42.33  0.045473     0.115414
rk4/10                         40     28.8s    82.30  0.114420     0.111974
bosh3(tol=1e-2)                42     30.3s   114.16  0.115772     0.231767
dopri5(tol=1e-2)               44     29.3s   211.74  0.297771     0.430733
euler/50                       50     38.0s    30.25  0.027472     0.109463
midpoint/25                    50     37.9s    37.53  0.036023     0.116220
heun2/25                       50     37.7s    94.05  0.131622     0.142603
bosh3(tol=1e-3)                52     37.3s    64.42  0.086166     0.113269
fehlberg2(tol=1e-5)            56     40.7s    63.23  0.084654     0.112452
dopri5(tol=1e-3)               68     44.9s    50.00  0.058926     0.111986
bosh3(tol=1e-4)                71     51.5s    62.35  0.082215     0.113442
heun3/25                       75     54.0s    48.46  0.056600     0.114198
fehlberg2(tol=1e-6)            88     63.6s    61.28  0.079292     0.112976
dopri5(tol=1e-4)               91     59.6s    61.17  0.079857     0.112913
euler/100                     100     82.6s    43.39  0.047593     0.111183
midpoint/50                   100     75.4s    51.95  0.062488     0.113417
heun2/50                      100     75.3s    80.02  0.112590     0.110496
rk4/25                        100     72.0s    60.64  0.079111     0.112292
dopri8(tol=1e-2)              121     79.5s    97.57  0.140846     0.141613
heun3/50                      150    107.9s    56.19  0.071442     0.112559
midpoint/100                  200    151.4s    56.68  0.073025     0.112772
heun2/100                     200    150.8s    64.14  0.086234     0.112038
rk4/50                        200    144.0s    56.76  0.072557     0.112507
dopri5(tol=1e-5)              213    140.6s    61.10  0.079332     0.113734
dopri8(tol=1e-3)              238    156.5s    71.54  0.098629     0.116685
bosh3(tol=1e-5)               265    191.5s    61.18  0.079199     0.113633
heun3/100                     300    215.9s    55.86  0.071366     0.112661
rk4/100                       400    289.6s    56.24  0.071405     0.112593
```

---

## Solver Notes

| Solver | Type | NFE/step | Verdict |
|--------|------|----------|---------|
| **euler** | Fixed, 1st order | 1 | **Best.** Matches training discretization. |
| midpoint | Fixed, 2nd order | 2 | Decent runner-up. Competitive at low NFE. |
| heun3 | Fixed, 3rd order | 3 | Mediocre. Extra accuracy hurts. |
| rk4 | Fixed, 4th order | 4 | Poor. Too many NFE for the benefit. |
| heun2 | Fixed, 2nd order | 2 | **Broken.** Catastrophically bad at low steps. |
| fehlberg2 | Adaptive, 2nd order | varies | **Broken.** Worst adaptive solver. |
| bosh3 | Adaptive, 3rd order | varies | Poor. Needs many NFE to converge. |
| dopri5 | Adaptive, 5th order | varies | Poor. Best adaptive but still worse than euler. |
| dopri8 | Adaptive, 8th order | varies | **Broken.** Most expensive, poor quality. |

---

## Experimental Details

- **40 solver configurations** evaluated (5 fixed-step solvers x 5 step counts + 4 adaptive solvers x variable tolerances, plus some extra tolerance runs)
- **25 volumes per config** = 1000 total volumes generated
- **Metrics:** FID (ResNet50), KID (ResNet50, subset_size=100), CMMD (BiomedCLIP)
- **All configs use identical noise tensors** for fair comparison
- **Reference features** cached for reuse across configs
- Tolerances 1e-4 and 1e-5 caused OOM/timeout for dopri8; removed from sweep
