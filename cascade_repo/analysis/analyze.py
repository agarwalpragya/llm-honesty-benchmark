"""
CASCADE: Statistical Analysis Script
Computes deception gaps, Fisher's exact tests, Holm-Bonferroni correction,
Wilson confidence intervals, and summary tables.

Usage:
    python analyze.py

Outputs:
    - Table I: Per-cell concealment rates with Wilson 95% CIs
    - Table II: Deception gaps with Holm-corrected p-values
    - Table III: Effect summary (low vs. high harm)
    - figures/figure1.png (deception gap plot)
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import fisher_exact
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ── Wilson confidence interval ────────────────────────────────────────────────
def wilson_ci(k, n, z=1.96):
    """Wilson score interval for a proportion k/n at confidence level z."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0, center - margin), min(1, center + margin))

# ── Holm-Bonferroni correction ────────────────────────────────────────────────
def holm_bonferroni(p_values, alpha=0.05):
    """
    Apply Holm-Bonferroni correction.
    Returns list of (rejected, adjusted_p) tuples in original order.
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [None] * n
    rejected = [False] * n

    prev_adj = 0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj_p = min(1.0, max(prev_adj, p * (n - rank)))
        adjusted[orig_idx] = adj_p
        rejected[orig_idx] = (adj_p < alpha)
        prev_adj = adj_p

    return list(zip(rejected, adjusted))

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv('../data/results.csv')

HARM_LEVELS = ['L1', 'L2', 'L3', 'L4', 'L5', 'L6']
HARM_LABELS = {
    'L1': 'Cosmetic',
    'L2': 'Minor-func.',
    'L3': 'Financial',
    'L4': 'Regulatory',
    'L5': 'Health',
    'L6': 'Safety'
}
MODELS = ['claude_haiku_4_5', 'gpt_4o_mini', 'gpt_4o', 'llama_3_1_8b', 'qwen_2_5_7b']
MODEL_LABELS = {
    'claude_haiku_4_5': 'Claude Haiku 4.5',
    'gpt_4o_mini':      'GPT-4o-mini',
    'gpt_4o':           'GPT-4o',
    'llama_3_1_8b':     'Llama-3.1-8B',
    'qwen_2_5_7b':      'Qwen-2.5-7B'
}

def get_row(model, level, condition):
    row = df[(df['model'] == model) & (df['harm_level'] == level) & (df['condition'] == condition)]
    if row.empty:
        return None
    return row.iloc[0]

# ── TABLE I: Per-cell concealment rates ───────────────────────────────────────
print("=" * 80)
print("TABLE I: Concealment Rates (Wilson 95% CI)")
print("T = treatment, C = control")
print("=" * 80)

header = f"{'Model':<20} {'Cond':>4}  " + "  ".join(f"{'L'+str(i):<18}" for i in range(1, 7))
print(header)
print("-" * len(header))

for model in MODELS:
    for cond_label, cond in [('T', 'treatment'), ('C', 'control')]:
        row_parts = []
        for lvl in HARM_LEVELS:
            r = get_row(model, lvl, cond)
            if r is not None:
                k, n = int(r['concealment_count']), int(r['total_trials'])
                rate = k / n
                lo, hi = wilson_ci(k, n)
                row_parts.append(f"{rate:.2f} [{lo:.2f},{hi:.2f}]")
            else:
                row_parts.append("N/A")

        model_str = MODEL_LABELS[model] if cond == 'treatment' else ''
        print(f"{model_str:<20} {cond_label:>4}  " + "  ".join(f"{p:<18}" for p in row_parts))
    print()

# ── TABLE II: Deception gaps with Fisher tests and Holm correction ────────────
print("\n" + "=" * 80)
print("TABLE II: Deception Gap (Δ) and Holm-Corrected Fisher p-values")
print("* = p < 0.05 after correction")
print("=" * 80)

# Collect all 30 comparisons
comparisons = []
for model in MODELS:
    for lvl in HARM_LEVELS:
        t = get_row(model, lvl, 'treatment')
        c = get_row(model, lvl, 'control')
        if t is not None and c is not None:
            t_k, t_n = int(t['concealment_count']), int(t['total_trials'])
            c_k, c_n = int(c['concealment_count']), int(c['total_trials'])

            # 2x2 contingency table: [[treat_conceal, treat_honest], [ctrl_conceal, ctrl_honest]]
            table = [[t_k, t_n - t_k], [c_k, c_n - c_k]]
            _, p = fisher_exact(table, alternative='two-sided')

            gap = (t_k / t_n) - (c_k / c_n)
            comparisons.append({
                'model': model,
                'harm_level': lvl,
                'gap': gap,
                't_k': t_k, 't_n': t_n,
                'c_k': c_k, 'c_n': c_n,
                'p_raw': p
            })

# Apply Holm-Bonferroni
p_values = [c['p_raw'] for c in comparisons]
holm_results = holm_bonferroni(p_values)

for i, comp in enumerate(comparisons):
    comp['rejected'], comp['p_holm'] = holm_results[i]

print(f"\n{'Model':<20} {'Level':<18} {'Gap (Δ)':>8}  {'Treat/n':>10}  {'p_holm':>12}")
print("-" * 75)

for model in MODELS:
    model_comps = [c for c in comparisons if c['model'] == model]
    first = True
    for comp in model_comps:
        sig = '*' if comp['rejected'] else ' '
        if comp['t_n'] == comp['t_k'] and comp['c_n'] == comp['c_k']:
            treat_str = "ceiling"
        else:
            treat_str = f"{comp['t_k']}/{comp['t_n']}"

        model_str = MODEL_LABELS[model] if first else ''
        p_str = f"{comp['p_holm']:.4f}{sig}" if comp['p_holm'] >= 0.0001 else f"<0.0001{sig}"
        print(f"{model_str:<20} {HARM_LABELS[comp['harm_level']]:<18} {comp['gap']:>+8.3f}  {treat_str:>10}  {p_str:>12}")
        first = False
    print()

# ── TABLE III: Effect summary ─────────────────────────────────────────────────
print("\n" + "=" * 80)
print("TABLE III: Effect Summary — Mean Deception Gap at Low vs. High Harm")
print("=" * 80)

LOW_HARM  = ['L1', 'L2', 'L3']
HIGH_HARM = ['L4', 'L5', 'L6']

print(f"\n{'Model':<20} {'Low-harm Δ (L1–L3)':>20}  {'High-harm Δ (L4–L6)':>20}  {'p_holm (pooled low)':>20}")
print("-" * 87)

for model in MODELS:
    low_gaps  = [c['gap'] for c in comparisons if c['model'] == model and c['harm_level'] in LOW_HARM]
    high_gaps = [c['gap'] for c in comparisons if c['model'] == model and c['harm_level'] in HIGH_HARM]
    low_p     = [c['p_holm'] for c in comparisons if c['model'] == model and c['harm_level'] in LOW_HARM]

    mean_low  = np.mean(low_gaps)
    mean_high = np.mean(high_gaps)
    min_p_low = min(low_p)  # Most significant low-harm comparison

    p_str = f"<0.0001" if min_p_low < 0.0001 else f"{min_p_low:.3f}"
    print(f"{MODEL_LABELS[model]:<20} {mean_low:>+20.3f}  {mean_high:>+20.3f}  {p_str:>20}")

# ── FIGURE 1: Deception gap by harm level ────────────────────────────────────
print("\n\nGenerating Figure 1...")

fig, ax = plt.subplots(figsize=(8, 5))

COLORS = {
    'claude_haiku_4_5': '#2563EB',  # blue
    'gpt_4o_mini':      '#16A34A',  # green
    'gpt_4o':           '#DC2626',  # red
    'llama_3_1_8b':     '#D97706',  # amber
    'qwen_2_5_7b':      '#7C3AED',  # purple
}
MARKERS = {
    'claude_haiku_4_5': 'o',
    'gpt_4o_mini':      's',
    'gpt_4o':           '^',
    'llama_3_1_8b':     'D',
    'qwen_2_5_7b':      'P',
}

x = np.arange(len(HARM_LEVELS))

for model in MODELS:
    gaps = []
    ci_lo_errs = []
    ci_hi_errs = []

    for lvl in HARM_LEVELS:
        comp = next((c for c in comparisons if c['model'] == model and c['harm_level'] == lvl), None)
        if comp:
            t_k, t_n = comp['t_k'], comp['t_n']
            c_k, c_n = comp['c_k'], comp['c_n']
            gap = comp['gap']

            # Delta method CI propagation: Var(p_t - p_c) ≈ Var(p_t) + Var(p_c)
            var_t = (t_k / t_n) * (1 - t_k / t_n) / t_n if t_n > 0 else 0
            var_c = (c_k / c_n) * (1 - c_k / c_n) / c_n if c_n > 0 else 0
            se = np.sqrt(var_t + var_c)
            ci_lo_errs.append(1.96 * se)
            ci_hi_errs.append(1.96 * se)
            gaps.append(gap)
        else:
            gaps.append(0)
            ci_lo_errs.append(0)
            ci_hi_errs.append(0)

    sig_markers = [
        comp['rejected'] if (comp := next((c for c in comparisons if c['model'] == model and c['harm_level'] == lvl), None)) else False
        for lvl in HARM_LEVELS
    ]

    ax.errorbar(x, gaps,
                yerr=[ci_lo_errs, ci_hi_errs],
                label=MODEL_LABELS[model],
                color=COLORS[model],
                marker=MARKERS[model],
                markersize=7,
                linewidth=1.8,
                capsize=4,
                capthick=1.2)

    # Mark significant cells
    for i, (gap, sig) in enumerate(zip(gaps, sig_markers)):
        if sig:
            ax.annotate('*', (x[i], gap + ci_hi_errs[i] + 0.02),
                        ha='center', va='bottom', fontsize=12,
                        color=COLORS[model], fontweight='bold')

ax.axhline(y=0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_xticks(x)
ax.set_xticklabels([f"{lvl}\n{HARM_LABELS[lvl]}" for lvl in HARM_LEVELS], fontsize=9)
ax.set_ylabel('Deception Gap Δ(m,h) = p̂treat − p̂ctrl', fontsize=10)
ax.set_xlabel('Harm Level', fontsize=10)
ax.set_title('CASCADE: Deception Gap by Harm Level and Model\n(* = Holm-corrected p < 0.05)', fontsize=11)
ax.set_ylim(-0.15, 1.05)
ax.legend(loc='upper right', fontsize=8.5, framealpha=0.9)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('../figures/figure1.png', dpi=200, bbox_inches='tight')
plt.savefig('../figures/figure1.pdf', bbox_inches='tight')
print("Saved figures/figure1.png and figures/figure1.pdf")

print("\nAnalysis complete.")
