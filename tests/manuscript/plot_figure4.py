"""
Plot Figure 4 panels (b), (c), (d) from figure4_results.csv.

Usage:
    python plot_figure4.py [path_to_csv]

If no path given, defaults to figure4_results.csv in the current directory.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csv_path = sys.argv[1] if len(sys.argv) > 1 else 'figure4_results.csv'
df = pd.read_csv(csv_path)

traits = ['edu', 'height', 'wealth']
trait_labels = {'edu': 'Education', 'height': 'Height', 'wealth': 'Wealth'}
pair_colors = {
    ('edu', 'height'): 'tab:red',
    ('edu', 'wealth'): 'tab:blue',
    ('height', 'wealth'): 'tab:green',
}
trait_pairs = list(pair_colors.keys())

gens = np.array(sorted(df['generation'].unique()))


def mean_std(df, col, gen):
    vals = df.loc[df['generation'] == gen, col].dropna()
    return vals.mean(), vals.std()


# ═══════════════════════════════════════════════════════════════════════════
# Panel (b): Heritability inflation — one subplot per trait
# ═══════════════════════════════════════════════════════════════════════════

fig_b, axes_b = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)

for ax, t in zip(axes_b, traits):
    # h²_HE
    he_m = np.array([mean_std(df, f'h2_HE_{t}', g)[0] for g in gens])
    he_s = np.array([mean_std(df, f'h2_HE_{t}', g)[1] for g in gens])
    ax.plot(gens, he_m, 'o-', color='tab:red', label='h²(HE)')
    ax.fill_between(gens, he_m - he_s, he_m + he_s, alpha=0.15, color='tab:red')

    # h²_true
    tr_m = np.array([mean_std(df, f'h2_true_{t}', g)[0] for g in gens])
    tr_s = np.array([mean_std(df, f'h2_true_{t}', g)[1] for g in gens])
    ax.plot(gens, tr_m, 's--', color='tab:blue', label='h²(true)')
    ax.fill_between(gens, tr_m - tr_s, tr_m + tr_s, alpha=0.15, color='tab:blue')

    # R²_G from edu PGI
    col_edu = f'R2_{t}_from_edu_PGI'
    if col_edu in df.columns:
        r2e_m = np.array([mean_std(df, col_edu, g)[0] for g in gens])
        r2e_s = np.array([mean_std(df, col_edu, g)[1] for g in gens])
        ax.plot(gens, r2e_m, '^:', color='tab:green', label='R²(edu PGI)', markersize=5)
        ax.fill_between(gens, r2e_m - r2e_s, r2e_m + r2e_s, alpha=0.1, color='tab:green')

    # R²_G from height PGI
    col_hgt = f'R2_{t}_from_height_PGI'
    if col_hgt in df.columns:
        r2h_m = np.array([mean_std(df, col_hgt, g)[0] for g in gens])
        r2h_s = np.array([mean_std(df, col_hgt, g)[1] for g in gens])
        ax.plot(gens, r2h_m, 'v:', color='tab:purple', label='R²(height PGI)', markersize=5)
        ax.fill_between(gens, r2h_m - r2h_s, r2h_m + r2h_s, alpha=0.1, color='tab:purple')

    ax.set_title(trait_labels[t], fontsize=13)
    ax.set_xlabel('Generation')
    ax.set_ylim(-0.02, None)
    ax.grid(True, alpha=0.2)

axes_b[0].set_ylabel('Heritability / R²')
axes_b[0].legend(fontsize=8, loc='upper left')
fig_b.suptitle('Figure 4b — Heritability Inflation Over xAM Generations', fontsize=14)
fig_b.tight_layout()
fig_b.savefig('figure4b.png', dpi=150)
print("Saved figure4b.png")


# ═══════════════════════════════════════════════════════════════════════════
# Panel (b) — Individual trait h² plots (HE estimate vs true)
# ═══════════════════════════════════════════════════════════════════════════

for t in traits:
    fig_t, ax_t = plt.subplots(figsize=(7, 5))

    # h²_HE
    he_m = np.array([mean_std(df, f'h2_HE_{t}', g)[0] for g in gens])
    he_s = np.array([mean_std(df, f'h2_HE_{t}', g)[1] for g in gens])
    ax_t.plot(gens, he_m, 'o-', color='tab:red', label='h²(HE estimate)')
    ax_t.fill_between(gens, he_m - he_s, he_m + he_s, alpha=0.15, color='tab:red')

    # h²_true
    tr_m = np.array([mean_std(df, f'h2_true_{t}', g)[0] for g in gens])
    tr_s = np.array([mean_std(df, f'h2_true_{t}', g)[1] for g in gens])
    ax_t.plot(gens, tr_m, 's--', color='tab:blue', label='h²(true)')
    ax_t.fill_between(gens, tr_m - tr_s, tr_m + tr_s, alpha=0.15, color='tab:blue')

    ax_t.set_xlabel('Generation')
    ax_t.set_ylabel('Heritability')
    ax_t.set_title(f'{trait_labels[t]} — HE Estimated vs True h²', fontsize=13)
    ax_t.legend(fontsize=10)
    ax_t.grid(True, alpha=0.2)
    ax_t.set_ylim(-0.02, None)
    fig_t.tight_layout()
    fname = f'figure4b_{t}.png'
    fig_t.savefig(fname, dpi=150)
    print(f"Saved {fname}")


# ═══════════════════════════════════════════════════════════════════════════
# Panel (c): Genetic correlations — r_score (true) and rg_HE (estimated)
# ═══════════════════════════════════════════════════════════════════════════

fig_c, ax_c = plt.subplots(figsize=(8, 5))

for (t1, t2), color in pair_colors.items():
    label = f'{trait_labels[t1]}/{trait_labels[t2]}'

    # r_score (true PGI correlation) — solid
    col_rs = f'r_score_{t1}_{t2}'
    if col_rs in df.columns:
        rs_m = np.array([mean_std(df, col_rs, g)[0] for g in gens])
        rs_s = np.array([mean_std(df, col_rs, g)[1] for g in gens])
        valid = ~np.isnan(rs_m)
        if valid.any():
            ax_c.plot(gens[valid], rs_m[valid], 'o-', color=color,
                      label=f'r_score {label}')
            ax_c.fill_between(gens[valid], (rs_m - rs_s)[valid],
                              (rs_m + rs_s)[valid], alpha=0.15, color=color)

    # rg_HE (estimated genetic correlation) — dashed
    col_rg = f'rg_HE_{t1}_{t2}'
    if col_rg in df.columns:
        rg_m = np.array([mean_std(df, col_rg, g)[0] for g in gens])
        rg_s = np.array([mean_std(df, col_rg, g)[1] for g in gens])
        ax_c.plot(gens, rg_m, 's--', color=color,
                  label=f'rg(HE) {label}')
        ax_c.fill_between(gens, rg_m - rg_s, rg_m + rg_s,
                          alpha=0.1, color=color)

ax_c.axhline(0, color='gray', linewidth=0.5, linestyle='-')
ax_c.set_xlabel('Generation')
ax_c.set_ylabel('Correlation')
ax_c.set_title('Figure 4c — Genetic Correlations Emerge From Nothing', fontsize=13)
ax_c.legend(fontsize=8, ncol=2)
ax_c.grid(True, alpha=0.2)
fig_c.tight_layout()
fig_c.savefig('figure4c.png', dpi=150)
print("Saved figure4c.png")


# ═══════════════════════════════════════════════════════════════════════════
# Panel (d): GWAS beta correlations
# ═══════════════════════════════════════════════════════════════════════════

fig_d, ax_d = plt.subplots(figsize=(8, 5))

for (t1, t2), color in pair_colors.items():
    label = f'{trait_labels[t1]}/{trait_labels[t2]}'
    col = f'r_beta_{t1}_{t2}'
    if col not in df.columns:
        continue
    rb_m = np.array([mean_std(df, col, g)[0] for g in gens])
    rb_s = np.array([mean_std(df, col, g)[1] for g in gens])
    ax_d.plot(gens, rb_m, 'o-', color=color, label=label)
    ax_d.fill_between(gens, rb_m - rb_s, rb_m + rb_s,
                      alpha=0.15, color=color)

ax_d.axhline(0, color='gray', linewidth=0.5, linestyle='-')
ax_d.set_xlabel('Generation')
ax_d.set_ylabel('r̂_β (GWAS effect correlation)')
ax_d.set_title('Figure 4d — GWAS Effect Estimate Correlations', fontsize=13)
ax_d.legend(fontsize=9)
ax_d.grid(True, alpha=0.2)
fig_d.tight_layout()
fig_d.savefig('figure4d.png', dpi=150)
print("Saved figure4d.png")

plt.show()
