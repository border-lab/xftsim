"""Overlay Ajay's fig3 reproduction on paper curves, both HE variants.

Shows:
  Left: h²  — paper band (10-90 pct of ~1000 replicates) vs Ajay (mean ± SD of 10 seeds)
         Both HE variants (full-sample, one-per-family) shown for Ajay.
  Right: rg — same layout.

Paper scenarios 5xAM and 5xAM+VT are overlaid on Ajay's scenarios of
the same name. RM and RM+VT have no paper reference (those weren't plotted).
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTDIR = '/home/rsb/data/fig3_results'

ajay = pd.read_csv(f'{OUTDIR}/fig3_aggregated.csv')
print(f'Ajay: {len(ajay)} rows, scenarios={list(ajay.scenario.unique())}, seeds={sorted(ajay.seed.unique())}')

paper_h2 = pd.read_csv('/home/rsb/Dropbox/ftsim/round4/processed/fig3_h2_plot.csv')
paper_rg = pd.read_csv('/home/rsb/Dropbox/ftsim/round4/processed/fig3_rg_plot.csv')
paper_name = {'5xAM': '5xAM', '5xAM+VT': '5xAM + VT'}

def _curve(sub, col):
    gs = np.array(sorted(sub.gen.unique()))
    means = np.array([sub[sub.gen==g][col].mean() for g in gs])
    sds = np.array([sub[sub.gen==g][col].std() for g in gs])
    return gs, means, sds

# ═══ Plot 1: 5xAM and 5xAM+VT — full overlay ═══════════════════════════════
for scenario in ['5xAM', '5xAM+VT']:
    pname = paper_name[scenario]
    sub = ajay[ajay.scenario==scenario]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    # ── h² ──
    ax = axes[0]
    # Paper HE h² band + median
    ph2 = paper_h2[(paper_h2.scenario==pname) & (paper_h2.variable=='h2_he')].sort_values('gen')
    ax.fill_between(ph2.gen, ph2['q10.10.'], ph2['q90.90.'],
                    alpha=0.25, color='black', label='paper HE h² (10-90 pct, ~1000 reps)')
    ax.plot(ph2.gen, ph2['median.50.'], 'k-', lw=1.5, label='paper HE h² median')
    # Paper true h²
    pht = paper_h2[(paper_h2.scenario==pname) & (paper_h2.variable=='h2_true')].sort_values('gen')
    ax.plot(pht.gen, pht['median.50.'], 'k:', lw=1.5, label='paper true h² median')
    # Ajay full-sample HE h²
    gs, m, s = _curve(sub, 'mean_he_h2_full')
    ax.errorbar(gs, m, yerr=s, fmt='ro-', capsize=4, lw=2, ms=6, alpha=0.9,
                label=f'Ajay full-sample HE h² (N={len(sub.seed.unique())} seeds, mean±SD)')
    # Ajay one-per-family HE h² (gen 5 only — paper-methodology comparison point)
    sub5 = sub[sub.gen==5]
    unrel_mean = sub5.mean_he_h2_unrel.mean()
    unrel_sd = sub5.mean_he_h2_unrel.std()
    if not np.isnan(unrel_mean):
        ax.errorbar([5], [unrel_mean], yerr=[unrel_sd], fmt='bs', capsize=6, lw=2, ms=10, alpha=0.9,
                    label=f'Ajay gen-5 one-per-family HE h² (matches paper setup)')
    # Ajay true h²
    gs, m, s = _curve(sub, 'mean_h2_true')
    ax.plot(gs, m, 'g^-', ms=5, alpha=0.8, label='Ajay true h²')
    ax.set_title(f'{scenario}: HE-estimated heritability')
    ax.set_xlabel('Generation'); ax.set_ylabel('h²')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)

    # ── rg ──
    ax = axes[1]
    prb = paper_rg[(paper_rg.scenario==pname) & (paper_rg.variable=='rbeta_HE')].sort_values('gen')
    ax.fill_between(prb.gen, prb['q10.10.'], prb['q90.90.'],
                    alpha=0.25, color='black', label='paper HE rg (10-90 pct)')
    ax.plot(prb.gen, prb['median.50.'], 'k-', lw=1.5, label='paper HE rg median')
    prt = paper_rg[(paper_rg.scenario==pname) & (paper_rg.variable=='rg_true')].sort_values('gen')
    ax.plot(prt.gen, prt['median.50.'], 'k:', lw=1.5, label='paper true rg median')
    gs, m, s = _curve(sub, 'mean_he_rg_full')
    ax.errorbar(gs, m, yerr=s, fmt='ro-', capsize=4, lw=2, ms=6, alpha=0.9,
                label='Ajay full-sample HE rg (mean±SD)')
    sub5 = sub[sub.gen==5]
    unrel_rg_mean = sub5.mean_he_rg_unrel.mean()
    unrel_rg_sd = sub5.mean_he_rg_unrel.std()
    if not np.isnan(unrel_rg_mean):
        ax.errorbar([5], [unrel_rg_mean], yerr=[unrel_rg_sd], fmt='bs', capsize=6, lw=2, ms=10, alpha=0.9,
                    label='Ajay gen-5 one-per-family HE rg')
    gs, m, s = _curve(sub, 'mean_rg_true')
    ax.plot(gs, m, 'g^-', ms=5, alpha=0.8, label='Ajay true rg')
    ax.set_title(f'{scenario}: HE-estimated genetic correlation')
    ax.set_xlabel('Generation'); ax.set_ylabel('rg')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=8)

    fig.suptitle(f"Figure 3 reproduction (Ajay's new xftsim) vs paper — scenario: {scenario}")
    fig.tight_layout()
    safename = scenario.replace('+','_plus_')
    fig.savefig(f'{OUTDIR}/fig3_{safename}_overlay.png', dpi=150)
    fig.savefig(f'{OUTDIR}/fig3_{safename}_overlay.pdf')
    print(f'Saved {OUTDIR}/fig3_{safename}_overlay.png')
    plt.close(fig)

# ═══ Plot 2: RM and RM+VT (no paper reference — just show trajectories) ═══
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
for i, scenario in enumerate(['RM', 'RM+VT']):
    ax = axes[i]
    sub = ajay[ajay.scenario==scenario]
    for col, fmt, lab in [
        ('mean_he_h2_full', 'ro-', 'HE h² (full sample)'),
        ('mean_h2_true', 'g^-', 'true h²'),
    ]:
        gs, m, s = _curve(sub, col)
        if col != 'mean_h2_true':
            ax.errorbar(gs, m, yerr=s, fmt=fmt, capsize=4, lw=2, ms=6, alpha=0.9, label=f'Ajay {lab}')
        else:
            ax.plot(gs, m, fmt, ms=5, alpha=0.8, label=f'Ajay {lab}')
    # gen-5 one-per-family HE h² (paper-style methodology)
    sub5 = sub[sub.gen==5]
    unrel_mean = sub5.mean_he_h2_unrel.mean()
    unrel_sd = sub5.mean_he_h2_unrel.std()
    if not np.isnan(unrel_mean):
        ax.errorbar([5], [unrel_mean], yerr=[unrel_sd], fmt='bs', capsize=6, lw=2, ms=10, alpha=0.9,
                    label='gen-5 HE h² (one-per-family)')
    ax.axhline(0.5, color='k', ls=':', alpha=0.5, label='theory h²=0.5')
    ax.set_title(f'{scenario}: heritability'); ax.set_xlabel('Generation'); ax.set_ylabel('h²')
    ax.grid(True, alpha=0.3); ax.legend(loc='best', fontsize=8)
fig.suptitle('RM and RM+VT scenarios (no paper reference plotted for these)')
fig.tight_layout()
fig.savefig(f'{OUTDIR}/fig3_RM_scenarios.png', dpi=150)
fig.savefig(f'{OUTDIR}/fig3_RM_scenarios.pdf')
print(f'Saved {OUTDIR}/fig3_RM_scenarios.png')
plt.close(fig)

# ═══ Numeric comparison table ══════════════════════════════════════════════
print('\n=== GEN-5 COMPARISON (Ajay mean ± SD across seeds vs paper median [10-90 pct]) ===')
print(f'{"Scenario":<10}  {"Quantity":<22}  {"Paper median":>15}  {"Paper 10-90 pct":>20}  {"Ajay mean":>12}  {"Ajay SD":>10}  {"|diff|":>8}')
print('-'*115)
for scenario in ['RM', 'RM+VT', '5xAM', '5xAM+VT']:
    sub = ajay[(ajay.scenario==scenario) & (ajay.gen==5)]
    if len(sub)==0: continue
    pname = paper_name.get(scenario)
    for col_ajay, label, paper_var in [
        ('mean_he_h2_full', 'HE h² (Ajay full)', 'h2_he'),
        ('mean_he_h2_unrel', 'HE h² (Ajay 1/fam)', 'h2_he'),
        ('mean_he_rg_full', 'HE rg (Ajay full)', 'rbeta_HE'),
        ('mean_he_rg_unrel', 'HE rg (Ajay 1/fam)', 'rbeta_HE'),
    ]:
        amean = sub[col_ajay].mean(); asd = sub[col_ajay].std()
        if pname:
            pdf = paper_h2 if 'h²' in label else paper_rg
            pref = pdf[(pdf.scenario==pname) & (pdf.variable==paper_var) & (pdf.gen==5)]
            if len(pref):
                pm = float(pref['median.50.'].values[0])
                plo = float(pref['q10.10.'].values[0])
                phi = float(pref['q90.90.'].values[0])
                diff = abs(amean - pm)
                print(f'{scenario:<10}  {label:<22}  {pm:>15.4f}  [{plo:>7.4f}, {phi:>7.4f}]  {amean:>12.4f}  {asd:>10.4f}  {diff:>8.4f}')
            else:
                print(f'{scenario:<10}  {label:<22}  {"(no paper ref)":<15}  {"":<20}  {amean:>12.4f}  {asd:>10.4f}')
        else:
            print(f'{scenario:<10}  {label:<22}  {"(no paper ref)":<15}  {"":<20}  {amean:>12.4f}  {asd:>10.4f}')
    print()
