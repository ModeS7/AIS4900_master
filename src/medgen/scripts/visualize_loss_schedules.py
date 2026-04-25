#!/usr/bin/env python3
"""Visualize the per-timestep loss weight schedules for every fine-tune
experiment in the project (exp32_2, exp45, exp46/b, exp47a-e, exp48a-d).

For each experiment, plots how each loss component is weighted as a function
of timestep t ∈ [0, T]:

  - MSE / base pixel loss (always present at full rate unless faded out)
  - Perceptual (LPIPS) loss
  - Shifted custom loss (L1 / Pseudo-Huber / lpips_huber, when t-shift active)

Usage:
    python -m medgen.scripts.visualize_loss_schedules \
        --output-dir runs/eval/loss_schedules
"""
import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

T = 1000  # num_train_timesteps in the project


# ─── Loss-weight functions (per t) ────────────────────────────────────
def mse_weight(t, mse_max_t=None, t_shift_max_t=None, mse_floor=0.0,
               shifted_type=None):
    """Weight applied to the MSE base loss at timestep t."""
    w = 1.0
    # mse_t_schedule (legacy, not used by 47/48 but available)
    if mse_max_t is not None and t < mse_max_t:
        w *= t / mse_max_t  # ramp 0..1
    # base_loss_t_shift: MSE blended with shifted_loss
    if t_shift_max_t is not None and t < t_shift_max_t:
        alpha = 1.0 - t / t_shift_max_t  # 0 at high-t, 1 at low-t
        if shifted_type == 'perceptual_only':
            # MSE fades to floor (no shifted pixel loss to blend with)
            w *= max(mse_floor, 1.0 - alpha)
        else:
            # MSE blended with custom pixel loss: weight = (1 - alpha)
            w *= (1.0 - alpha)
    return w


def shifted_weight(t, t_shift_max_t=None, shifted_type=None):
    """Weight on the shifted custom loss (l1 / pseudo_huber / lpips_huber)."""
    if t_shift_max_t is None or shifted_type is None:
        return 0.0
    if shifted_type == 'perceptual_only':
        return 0.0  # no pixel-domain custom loss
    if t >= t_shift_max_t:
        return 0.0
    return 1.0 - t / t_shift_max_t  # 0 at high-t, 1 at low-t


def perceptual_weight(t, perc_weight_base, perc_max_t=None, perc_schedule=None,
                       t_shift_max_t=None, shifted_type=None):
    """Total LPIPS weight at timestep t (base × t-scale × extra-ramp-if-perceptual_only)."""
    if perc_weight_base <= 0:
        return 0.0
    scale = 1.0
    if perc_schedule is not None:
        # perceptual_t_schedule = [t_on, t_full, t_off] in [0,1] units
        t_on, t_full, t_off = perc_schedule
        t_on, t_full, t_off = t_on * T, t_full * T, t_off * T
        if t < t_on or t >= t_off:
            scale = 0.0
        elif t >= t_full:
            scale = 1.0
        else:
            scale = (t - t_on) / max(1e-9, t_full - t_on)
    elif perc_max_t is not None:
        scale = max(0.0, 1.0 - t / perc_max_t)  # 1 at t=0, 0 at t=max_t

    # If perceptual_only mode: also multiply by alpha (ramps in alongside MSE fade)
    if t_shift_max_t is not None and shifted_type == 'perceptual_only':
        if t < t_shift_max_t:
            alpha = 1.0 - t / t_shift_max_t
            scale *= alpha
        else:
            scale = 0.0
    return perc_weight_base * scale


def lpips_huber_internal(t, t_shift_max_t):
    """For shifted_type='lpips_huber', the Karras formula has Huber × (1-t̃)
    plus LPIPS uniform. We render this as Huber-component vs LPIPS-component.
    Returns (huber_w, lpips_w)."""
    if t_shift_max_t is None or t >= t_shift_max_t:
        return 0.0, 0.0
    alpha = 1.0 - t / t_shift_max_t  # 0 at high-t, 1 at low-t — overall weight
    # Inside the lpips_huber formula: Huber × (1-t̃) where t̃ is normalized t in [0,1]
    t_norm_in_window = t / t_shift_max_t  # 0..1 inside the window
    huber_inner = 1.0 - t_norm_in_window  # peak at low-t inside window
    lpips_inner = 1.0  # uniform inside window
    return alpha * huber_inner, alpha * lpips_inner


# ─── Experiment registry ────────────────────────────────────────────────
# Each entry describes the loss schedule of one experiment.
EXPERIMENTS = [
    # (label, mse_max_t, perc_base, perc_max_t, perc_schedule, t_shift_max_t,
    #  shifted_type, mse_floor, curriculum_max_t, scoreaug_schedule)
    {
        'name': 'exp32_2 / extended (baseline LPIPS-lowt)',
        'family': '32',
        'perc_base': 0.1, 'perc_max_t': 250,
    },
    {
        'name': 'exp45  LPIPS-lowt + ScoreAug detail (falling)',
        'family': '45',
        'perc_base': 0.1, 'perc_max_t': 250,
        'scoreaug_falling': (0.325, 0.45),
    },
    {
        'name': 'exp46  Mamba-L + LPIPS-lowt',
        'family': '46',
        'perc_base': 0.1, 'perc_max_t': 250,
    },
    {
        'name': 'exp46b Mamba-L + LPIPS-lowt + ScoreAug (falling)',
        'family': '46',
        'perc_base': 0.1, 'perc_max_t': 250,
        'scoreaug_falling': (0.325, 0.45),
    },
    {
        'name': 'exp47a stronger LPIPS, no t-shift',
        'family': '47',
        'perc_base': 0.5, 'perc_max_t': 250,
    },
    {
        'name': 'exp47b t-shift MSE→L1 + LPIPS-lowt',
        'family': '47',
        'perc_base': 0.1, 'perc_max_t': 250,
        't_shift_max_t': 250, 'shifted_type': 'l1',
    },
    {
        'name': 'exp47c t-shift MSE→lpips_huber',
        'family': '47',
        't_shift_max_t': 250, 'shifted_type': 'lpips_huber',
    },
    {
        'name': 'exp47d t-shift MSE→Pseudo-Huber + LPIPS-lowt',
        'family': '47',
        'perc_base': 0.1, 'perc_max_t': 250,
        't_shift_max_t': 250, 'shifted_type': 'pseudo_huber',
    },
    {
        'name': 'exp47e t-shift MSE→0.01 floor + perceptual-only',
        'family': '47',
        'perc_base': 0.5,  # rises via alpha (no perc_max_t needed; alpha drives ramp)
        't_shift_max_t': 250, 'shifted_type': 'perceptual_only', 'mse_floor': 0.01,
    },
    {
        'name': 'exp48a low-t-only MSE + LPIPS@0.5',
        'family': '48',
        'perc_base': 0.5,
        'curriculum_max_t': 250,
    },
    {
        'name': 'exp48b low-t-only L1 + LPIPS',
        'family': '48',
        'perc_base': 0.1, 'base_is_l1': True,
        'curriculum_max_t': 250,
    },
    {
        'name': 'exp48c low-t-only lpips_huber',
        'family': '48',
        'base_is_lpips_huber': True,
        'curriculum_max_t': 250,
    },
    {
        'name': 'exp48d low-t-only Pseudo-Huber + LPIPS',
        'family': '48',
        'perc_base': 0.1, 'base_is_huber': True,
        'curriculum_max_t': 250,
    },
]


def evaluate(cfg, t):
    """Return dict of loss component weights at timestep t for cfg."""
    perc_base = cfg.get('perc_base', 0.0)
    perc_max_t = cfg.get('perc_max_t')
    t_shift_max_t = cfg.get('t_shift_max_t')
    shifted_type = cfg.get('shifted_type')
    mse_floor = cfg.get('mse_floor', 0.0)

    out = {}
    base_label = 'MSE'
    if cfg.get('base_is_l1'):
        base_label = 'L1'
    elif cfg.get('base_is_huber'):
        base_label = 'Pseudo-Huber'
    elif cfg.get('base_is_lpips_huber'):
        # lpips_huber as the base strategy loss (exp48c) — internally
        # equivalent to (1-t̃)·Huber + LPIPS at all t in the trained range.
        out['Huber (in lpips_huber)'] = 1.0 - t / T
        out['LPIPS (in lpips_huber)'] = 1.0
        return out

    if shifted_type == 'lpips_huber':
        # exp47c: MSE faded out, lpips_huber faded in inside [0, t_shift_max_t]
        out['MSE'] = mse_weight(t, t_shift_max_t=t_shift_max_t, shifted_type='generic_pixel')
        h, lp = lpips_huber_internal(t, t_shift_max_t)
        out['Huber (in lpips_huber)'] = h
        out['LPIPS (in lpips_huber)'] = lp
        return out

    out[base_label] = mse_weight(
        t, t_shift_max_t=t_shift_max_t,
        mse_floor=mse_floor, shifted_type=shifted_type,
    )

    sw = shifted_weight(t, t_shift_max_t=t_shift_max_t, shifted_type=shifted_type)
    if sw > 0:
        label_map = {'l1': 'L1', 'pseudo_huber': 'Pseudo-Huber'}
        out[label_map.get(shifted_type, shifted_type)] = sw

    pw = perceptual_weight(
        t, perc_base, perc_max_t=perc_max_t,
        t_shift_max_t=t_shift_max_t, shifted_type=shifted_type,
    )
    if perc_base > 0:
        out['LPIPS'] = pw

    if cfg.get('scoreaug') is not None:
        # Legacy bandpass schedule: [t_on, t_full, t_off]
        s = cfg['scoreaug']
        if t / T >= s[0] and t / T < s[2]:
            if t / T < s[1]:
                w = (t / T - s[0]) / max(1e-9, s[1] - s[0])
            else:
                w = 1.0
            out['ScoreAug (prob)'] = w
        else:
            out['ScoreAug (prob)'] = 0.0
    elif cfg.get('scoreaug_falling') is not None:
        # Falling schedule: [t_full_end, t_off] — full strength at low-t,
        # ramp down to 0 between t_full_end and t_off, off above.
        t_full_end, t_off = cfg['scoreaug_falling']
        t_norm = t / T
        if t_norm < t_full_end:
            out['ScoreAug (prob)'] = 1.0
        elif t_norm >= t_off:
            out['ScoreAug (prob)'] = 0.0
        else:
            out['ScoreAug (prob)'] = 1.0 - (t_norm - t_full_end) / max(1e-9, t_off - t_full_end)
    return out


COLORS = {
    'MSE': 'tab:blue',
    'L1': 'tab:cyan',
    'Pseudo-Huber': 'tab:purple',
    'Huber (in lpips_huber)': 'tab:purple',
    'LPIPS': 'tab:red',
    'LPIPS (in lpips_huber)': 'tab:red',
    'ScoreAug (prob)': 'tab:orange',
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='runs/eval/loss_schedules')
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(EXPERIMENTS)
    cols = 2
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.2 * rows))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes

    ts = np.linspace(0, T, 401)

    for i, cfg in enumerate(EXPERIMENTS):
        ax = axes[i]
        # Compute weights along ts. First pass collects all keys.
        per_t: list[dict] = [evaluate(cfg, float(t)) for t in ts]
        all_keys: list[str] = []
        for d in per_t:
            for k in d:
                if k not in all_keys:
                    all_keys.append(k)
        components = {k: [d.get(k, 0.0) for d in per_t] for k in all_keys}

        # Plot each component
        for k, ys in components.items():
            ax.plot(ts, ys, label=k, color=COLORS.get(k, 'gray'),
                    linewidth=2, alpha=0.85)

        # Curriculum shading (when training restricts to t < curriculum_max_t)
        cmax = cfg.get('curriculum_max_t')
        if cmax is not None:
            ax.axvspan(0, cmax, alpha=0.08, color='green',
                       label='trained-t range')
            ax.axvline(cmax, color='green', linewidth=0.8, linestyle=':', alpha=0.6)

        # t-shift transition marker
        ts_max = cfg.get('t_shift_max_t')
        if ts_max is not None:
            ax.axvline(ts_max, color='black', linewidth=0.8, linestyle='--', alpha=0.4)

        ax.set_xlim(0, T)
        ax.set_ylim(-0.05, 1.15)
        ax.set_xlabel('Timestep t')
        ax.set_ylabel('Loss weight')
        ax.set_title(cfg['name'], fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')

    # Hide unused subplots
    for j in range(n, len(axes)):
        axes[j].axis('off')

    fig.suptitle(
        'Per-timestep loss weight schedule per experiment\n'
        '(t=0: clean, t=T: noise. Trained-t shaded green; t-shift transition: dashed black.)',
        fontsize=12, y=1.0,
    )
    fig.tight_layout()
    out_path = out_dir / 'loss_schedules.png'
    fig.savefig(out_path, dpi=140, bbox_inches='tight')
    fig.savefig(out_dir / 'loss_schedules.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"Saved: {out_path}")
    print(f"Saved: {out_dir / 'loss_schedules.pdf'}")

    # Also produce a focused exp47 family plot (a-e overlay, MSE-only and LPIPS-only)
    family_cfgs = [c for c in EXPERIMENTS if c['family'] == '47']
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 4))
    for c in family_cfgs:
        ms = [mse_weight(t,
                         t_shift_max_t=c.get('t_shift_max_t'),
                         mse_floor=c.get('mse_floor', 0.0),
                         shifted_type=c.get('shifted_type'))
              for t in ts]
        ps = [perceptual_weight(
                t, c.get('perc_base', 0.0),
                perc_max_t=c.get('perc_max_t'),
                t_shift_max_t=c.get('t_shift_max_t'),
                shifted_type=c.get('shifted_type'),
            ) for t in ts]
        label = c['name'].split()[0]
        axes2[0].plot(ts, ms, label=label, linewidth=2)
        axes2[1].plot(ts, ps, label=label, linewidth=2)
    axes2[0].set_title('MSE / pixel-base weight vs t (exp47 family)')
    axes2[1].set_title('LPIPS effective weight vs t (exp47 family)')
    for ax in axes2:
        ax.set_xlabel('t')
        ax.set_ylabel('weight')
        ax.set_xlim(0, T)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig2.tight_layout()
    fig2.savefig(out_dir / 'exp47_family_overlay.png', dpi=140, bbox_inches='tight')
    plt.close(fig2)
    print(f"Saved: {out_dir / 'exp47_family_overlay.png'}")

    # ─── Loss function SHAPES (what each loss does to a residual) ─────
    e = np.linspace(-1.5, 1.5, 601)
    # MSE: ½ × e²  (matches torch.nn.MSELoss default reduction='mean' on 1 elem)
    mse = 0.5 * e ** 2
    l1 = np.abs(e)
    # Pseudo-Huber (Charbonnier-like): δ²·(√(1 + (e/δ)²) - 1)
    delta_psh_a = 0.1
    delta_psh_b = 0.3
    pseudo_huber_small = delta_psh_a ** 2 * (np.sqrt(1 + (e / delta_psh_a) ** 2) - 1)
    pseudo_huber_large = delta_psh_b ** 2 * (np.sqrt(1 + (e / delta_psh_b) ** 2) - 1)
    # True Huber: piecewise quadratic→linear
    delta_h = 0.3
    huber = np.where(np.abs(e) <= delta_h,
                     0.5 * e ** 2,
                     delta_h * (np.abs(e) - 0.5 * delta_h))
    # Charbonnier:  √(e² + ε²)  − ε   (zero at e=0)
    eps_chb = 0.01
    charbonnier = np.sqrt(e ** 2 + eps_chb ** 2) - eps_chb

    # 1D loss SHAPES: full view + zoom around zero
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 4.5))
    plot_specs = [
        ('MSE  (½·e²)', mse, 'tab:blue'),
        ('L1  (|e|)', l1, 'tab:cyan'),
        ('Pseudo-Huber  δ=0.1', pseudo_huber_small, 'tab:purple'),
        ('Pseudo-Huber  δ=0.3', pseudo_huber_large, 'tab:pink'),
        ('Huber  δ=0.3', huber, 'tab:olive'),
        ('Charbonnier  ε=0.01', charbonnier, 'tab:brown'),
    ]
    for label, ys, c in plot_specs:
        axes3[0].plot(e, ys, label=label, color=c, linewidth=2)
        axes3[1].plot(e, ys, label=label, color=c, linewidth=2)
    axes3[0].set_title('Loss function value vs residual e = prediction − target')
    axes3[0].set_xlabel('e')
    axes3[0].set_ylabel('loss(e)')
    axes3[0].set_xlim(-1.5, 1.5)
    axes3[0].set_ylim(-0.05, 1.5)
    axes3[0].grid(True, alpha=0.3)
    axes3[0].legend(fontsize=9, loc='upper center')

    axes3[1].set_title('Zoomed near e = 0 (where small residuals matter)')
    axes3[1].set_xlabel('e')
    axes3[1].set_ylabel('loss(e)')
    axes3[1].set_xlim(-0.4, 0.4)
    axes3[1].set_ylim(-0.005, 0.12)
    axes3[1].grid(True, alpha=0.3)
    axes3[1].legend(fontsize=9, loc='upper center')

    fig3.suptitle('Pixel-loss shapes used across exp32/47/48', fontsize=12, y=1.02)
    fig3.tight_layout()
    fig3.savefig(out_dir / 'loss_function_shapes.png', dpi=140, bbox_inches='tight')
    fig3.savefig(out_dir / 'loss_function_shapes.pdf', bbox_inches='tight')
    plt.close(fig3)
    print(f"Saved: {out_dir / 'loss_function_shapes.png'}")

    # 1D loss GRADIENTS: what actually drives optimization
    fig4, axes4 = plt.subplots(1, 2, figsize=(14, 4.5))
    grad_mse = e
    grad_l1 = np.sign(e)
    grad_psh_small = e / np.sqrt(1 + (e / delta_psh_a) ** 2)
    grad_psh_large = e / np.sqrt(1 + (e / delta_psh_b) ** 2)
    grad_huber = np.where(np.abs(e) <= delta_h, e, delta_h * np.sign(e))
    grad_charbonnier = e / np.sqrt(e ** 2 + eps_chb ** 2)
    grad_specs = [
        ('MSE  (e)', grad_mse, 'tab:blue'),
        ('L1  (sign e)', grad_l1, 'tab:cyan'),
        ('Pseudo-Huber  δ=0.1', grad_psh_small, 'tab:purple'),
        ('Pseudo-Huber  δ=0.3', grad_psh_large, 'tab:pink'),
        ('Huber  δ=0.3', grad_huber, 'tab:olive'),
        ('Charbonnier  ε=0.01', grad_charbonnier, 'tab:brown'),
    ]
    for label, ys, c in grad_specs:
        axes4[0].plot(e, ys, label=label, color=c, linewidth=2)
        axes4[1].plot(e, ys, label=label, color=c, linewidth=2)
    axes4[0].axhline(0, color='black', linewidth=0.5)
    axes4[0].set_title('∂loss/∂prediction — full view')
    axes4[0].set_xlabel('e')
    axes4[0].set_ylabel('gradient')
    axes4[0].set_xlim(-1.5, 1.5)
    axes4[0].set_ylim(-1.6, 1.6)
    axes4[0].grid(True, alpha=0.3)
    axes4[0].legend(fontsize=9, loc='upper left')

    axes4[1].axhline(0, color='black', linewidth=0.5)
    axes4[1].set_title('Gradient zoomed near e = 0 (the regime small residuals see)')
    axes4[1].set_xlabel('e')
    axes4[1].set_ylabel('gradient')
    axes4[1].set_xlim(-0.4, 0.4)
    axes4[1].set_ylim(-1.1, 1.1)
    axes4[1].grid(True, alpha=0.3)
    axes4[1].legend(fontsize=9, loc='upper left')

    fig4.suptitle('Gradient curves for the same losses — the discontinuity at e=0 in L1 vs the smooth bottom of Pseudo-Huber/Charbonnier',
                  fontsize=12, y=1.02)
    fig4.tight_layout()
    fig4.savefig(out_dir / 'loss_function_gradients.png', dpi=140, bbox_inches='tight')
    plt.close(fig4)
    print(f"Saved: {out_dir / 'loss_function_gradients.png'}")


if __name__ == '__main__':
    main()
