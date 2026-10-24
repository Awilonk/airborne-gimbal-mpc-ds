#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figures + numbers for the AST Sec. 4 draft from results_sec4.json / traces_sec4.npz. 2026-10-22 lyk.

    python make_sec4.py      -> fig_*.pdf/png, summary_sec4.txt, tab_results.tex (fragment)

Metrics (t >= 5 s, as sim_zr10.metrics): RMSE of theta - theta_ref (deg); saturation = share of
samples with |u| >= 49.5 levels; divergence = RMSE >= 1 deg.
'rel. M2' = paired RMSE ratio - 1 against M2 in the same env/condition/seed, averaged over seeds for
each f0 (median if a cell contains a divergent run, same rule as bench_m5), then averaged over f0.
"""
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results_sec4.json')))
T = R['test']
F0S = (0.5, 1.0, 1.5, 2.0)
SEEDS = range(100, 110)
SCEN = ('steady', 'switch')
PERTS = (0.0, 0.1, 0.2, 0.3)

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': 0.6, 'lines.linewidth': 0.9,
    'grid.linewidth': 0.4, 'savefig.dpi': 300, 'pdf.fonttype': 42,
})
W = 3.5
STY = {  # consistent mapping across figures
    'M1': dict(c='0.6', ls=':', label='M1'),
    'M2': dict(c='#4C72B0', ls='--', label='M2'),
    'M3a': dict(c='#8172B2', ls='-.', label='M3a'),
    'Xie': dict(c='#C44E52', ls=':', label='Xie'),
    'XieAd': dict(c='#DD8452', ls=(0, (3, 1, 1, 1)), label='XieAd'),
    'M4': dict(c='#55A868', ls='-.', label='M4'),
    'M5': dict(c='k', ls='-', label='M5 (proposed)'),
}


PROP = 'M5r'  # proposed = M5 with drift weight (run_sec4_rho.py); 'M5' key = kappa 0


def get(env, m, pt, f0, sc, s):
    m = PROP if m == 'M5' else ('M5' if m == 'M5k0' else m)
    return T['|'.join(map(str, (env, m, f0, sc, s, pt)))]


def stats(env, m, sc, pt=0.0, f0s=F0S):
    x = np.array([[get(env, m, pt, f, sc, s)[0] for s in SEEDS] for f in f0s])
    sat = np.array([[get(env, m, pt, f, sc, s)[1] for s in SEEDS] for f in f0s])
    b = np.array([[get(env, 'M2', pt, f, sc, s)[0] for s in SEEDS] for f in f0s])
    rel = []
    for i in range(len(f0s)):
        r = (x[i] / b[i] - 1) * 100
        rel.append(np.median(r) if (x[i] >= 1).any() else r.mean())
    return dict(mean=x.mean(), med=np.median(x), rel=float(np.mean(rel)), rel_f=rel,
                sat=100 * sat.mean(), div=int((x >= 1).sum()), all=x.ravel(),
                cell=x.mean(axis=1))


def save(fig, name):
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(HERE, name + '.' + ext), bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


def fig_time():
    D = np.load(os.path.join(HERE, 'traces_sec4.npz'))
    fig, ax = plt.subplots(2, 1, figsize=(W, 3.0), sharex=True,
                           gridspec_kw=dict(height_ratios=(1.6, 1)))
    for m in ('Xie', 'M2', 'M4', 'M5'):
        r = D['M5r' if m == 'M5' else m]
        ax[0].plot(r[:, 0], r[:, 2] - r[:, 1], color=STY[m]['c'], ls=STY[m]['ls'],
                   lw=0.6 if m == 'Xie' else 0.8, label=STY[m]['label'])
    ax[0].axvline(float(D['Tsw']), color='0.4', lw=0.5, ls='--')
    ax[0].set_ylabel('Tracking error (deg)')
    ax[0].set_ylim(-1.6, 1.6)
    ax[0].legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.22), frameon=False,
                 handlelength=2.2, columnspacing=1.0)
    r = D['M5r']
    ax[1].step(r[:, 0], r[:, 3], where='post', color='k', lw=0.5, label='$u_k$ (M5)')
    for y, ls, lab in ((50, '-', '$\\pm u_{\\max}$'), (45, ':', '$\\pm(u_{\\max}-m)$')):
        ax[1].axhline(y, color='#C44E52', ls=ls, lw=0.7, label=lab)
        ax[1].axhline(-y, color='#C44E52', ls=ls, lw=0.7)
    ax[1].axvline(float(D['Tsw']), color='0.4', lw=0.5, ls='--')
    ax[1].set_ylim(-60, 60)
    ax[1].set_ylabel('Command (level)')
    ax[1].set_xlabel('Time (s)')
    ax[1].legend(ncol=3, loc='lower center', frameon=False, fontsize=6.5)
    for a in ax:
        a.grid(alpha=0.3)
        a.set_xlim(0, 30)
    fig.align_ylabels(ax)
    save(fig, 'fig_time_switch')
    # numbers for text
    out = {}
    for m in ('M2', 'Xie', 'M4', 'M5', 'M5r', 'M5_noLIN', 'M5r_noLIN'):
        r = D[m]
        e = r[:, 2] - r[:, 1]
        seg = lambda a, b: float(np.sqrt(np.mean(e[(r[:, 0] >= a) & (r[:, 0] < b)] ** 2)))
        out[m] = dict(pre=seg(10, 20), post2=seg(20, 22), post=seg(22, 30), all=seg(5, 30),
                      sat=float(np.mean(np.abs(r[r[:, 0] >= 5, 3]) >= 49.5) * 100),
                      umax=float(np.max(np.abs(r[r[:, 0] >= 5, 3]))))
    return out


def fig_freq():
    D = np.load(os.path.join(HERE, 'traces_sec4.npz'))
    f0, tsw = float(D['f_true0']), float(D['Tsw'])
    t = D['M5r'][:, 0]
    ft = np.where(t < tsw, f0, 1.6 * f0)
    fig, ax = plt.subplots(figsize=(W, 1.8))
    ax.plot(t, ft, color='0.5', lw=1.6, alpha=0.6, label='True $f$')
    ax.plot(t, D['M5r_noLIN'][:, 8], color='#8172B2', ls='--', lw=0.8, label='Gradient law (APDO)')
    ax.plot(t, D['M5r'][:, 8], color='k', lw=0.8, label='LIN (proposed)')
    ax.axvline(tsw, color='0.4', lw=0.5, ls='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_xlim(0, 30)
    ax.grid(alpha=0.3)
    ax.legend(loc='upper left', frameon=False)
    save(fig, 'fig_freq')
    res = {}
    for k, nm in (('M5r', 'LIN'), ('M5r_noLIN', 'APDO')):
        f = D[k][:, 8]
        e = np.abs(f - ft)
        pre = (t >= 10) & (t < 20)
        post = (t >= 25) & (t < 30)
        # settling: first time after switch with |e| < 5% of 1.6 f0 and staying there
        tol = 0.05 * 1.6 * f0
        idx = np.where((t >= tsw) & (e >= tol))[0]
        ts = (t[idx[-1]] + 0.02 - tsw) if idx.size else 0.0
        if idx.size and idx[-1] == len(t) - 1:
            ts = float('nan')
        res[nm] = dict(pre=float(np.mean(e[pre])), post=float(np.mean(e[post])), settle=float(ts))
    res['f0'] = f0
    return res


def fig_box():
    meths = ('M1', 'M2', 'M3a', 'Xie', 'XieAd', 'M4', 'M5')
    fig, ax = plt.subplots(1, 2, figsize=(W, 2.0), sharey=True)
    for j, sc in enumerate(SCEN):
        data = [stats('cal', m, sc)['all'] for m in meths]
        bp = ax[j].boxplot(data, widths=0.6, whis=(0, 100), patch_artist=True,
                           medianprops=dict(color='k', lw=0.8),
                           boxprops=dict(lw=0.6), whiskerprops=dict(lw=0.6),
                           capprops=dict(lw=0.6))
        for p, m in zip(bp['boxes'], meths):
            p.set_facecolor(STY[m]['c'])
            p.set_alpha(0.55 if m != 'M5' else 0.9)
        ax[j].axhline(1.0, color='#C44E52', lw=0.5, ls=':')
        ax[j].set_yscale('log')
        ax[j].set_xticks(range(1, len(meths) + 1))
        ax[j].set_xticklabels(meths, rotation=45, ha='right')
        ax[j].set_title('(%s) %s' % ('ab'[j], 'Constant frequency' if sc == 'steady'
                                     else 'Frequency switch'), fontsize=8)
        ax[j].grid(alpha=0.3, axis='y', which='both')
    ax[0].set_ylabel('RMSE (deg)')
    save(fig, 'fig_rmse_box')


def fig_ablation():
    var = ('M5', 'M5r-noRho', 'M5r-noLIN', 'M5r-noPred', 'M5r-noMargin', 'M5r-noNom', 'M5r-noProj')
    lab = ('M5', 'noDrift', 'noLIN', 'noPred', 'noMargin', 'noNom', 'noProj')
    cases = (('cal', 'steady', 0.0), ('cal', 'switch', 0.0), ('cal', 'steady', 0.3),
             ('cal', 'switch', 0.3))
    names = ('steady', 'switch', 'steady, $\\pm$30%', 'switch, $\\pm$30%')
    cols = ('0.15', '0.45', '0.7', '0.9')
    hat = ('', '', '///', '///')
    fig, ax = plt.subplots(figsize=(W, 1.9))
    x = np.arange(len(var))
    w = 0.2
    tab = {}
    for i, (env, sc, pt) in enumerate(cases):
        v = [stats(env, m, sc, pt)['rel'] for m in var]
        tab[(env, sc, pt)] = v
        ax.bar(x + (i - 1.5) * w, v, w, color=cols[i], hatch=hat[i], edgecolor='k', lw=0.4,
               label=names[i])
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(lab)
    ax.set_ylabel('RMSE rel. to M2 (%)')
    ax.grid(alpha=0.3, axis='y')
    ax.legend(ncol=2, frameon=False, loc='lower right', fontsize=6.5)
    save(fig, 'fig_ablation')
    return tab


def fig_mismatch():
    meths = ('M2', 'M3a', 'XieAd', 'M4', 'M5')
    fig, ax = plt.subplots(1, 2, figsize=(W, 1.9), sharey=True)
    for j, sc in enumerate(SCEN):
        for m in meths:
            y = [stats('cal', m, sc, pt)['mean'] for pt in PERTS]
            ax[j].plot([100 * p for p in PERTS], y, color=STY[m]['c'], ls=STY[m]['ls'],
                       marker='o', ms=2.5, label=STY[m]['label'])
        ax[j].set_yscale('log')
        ax[j].set_xticks([0, 10, 20, 30])
        ax[j].set_xlabel('Mismatch bound (%)')
        ax[j].set_title('(%s) %s' % ('ab'[j], 'Constant frequency' if sc == 'steady'
                                     else 'Frequency switch'), fontsize=8)
        ax[j].grid(alpha=0.3, which='both')
    ax[0].set_ylabel('Mean RMSE (deg)')
    ax[1].legend(frameon=False, loc='center right', fontsize=6, bbox_to_anchor=(1.0, 0.62))
    save(fig, 'fig_mismatch')


def main():
    out = open(os.path.join(HERE, 'summary_sec4.txt'), 'w', encoding='utf-8')

    def say(s=''):
        print(s)
        out.write(s + '\n')

    say(__doc__)
    say('tuned params: ' + json.dumps(R['best']))
    meths = ('M1', 'M2', 'M3a', 'Xie', 'XieAd', 'M4', 'M5', 'M5k0', 'M5r-noLIN', 'M5r-noPred',
             'M5r-noMargin', 'M5r-noNom', 'M5r-noProj', 'M5r-conf')
    for env in ('cal', 'nom'):
        for pt in PERTS:
            for sc in SCEN:
                say('-- [%s] %s mismatch +-%d%%   mean  median  rel.M2%%  sat%%  div/40 | per-f0 mean'
                    % (env, sc, 100 * pt))
                for m in meths:
                    if m.startswith('M5r-') and pt not in (0.0, 0.3):
                        continue
                    s = stats(env, m, sc, pt)
                    say('  %-12s %.3f  %.3f  %+6.1f  %5.1f  %2d  | %s' % (
                        m, s['mean'], s['med'], s['rel'], s['sat'], s['div'],
                        ' '.join('%.3f' % c for c in s['cell'])))
    say()
    say('representative run (cal, seed 100, f0=1, switch) segment RMSE (deg):')
    for k, v in fig_time().items():
        say('  %-9s 5-30s %.3f  10-20s %.3f  20-22s %.3f  22-30s %.3f  sat %.1f%%  max|u| %.0f' % (
            k, v['all'], v['pre'], v['post2'], v['post'], v['sat'], v['umax']))
    fr = fig_freq()
    say('frequency estimate (same run, true f0 = %.4f Hz -> %.4f Hz):' % (fr['f0'], 1.6 * fr['f0']))
    for k in ('LIN', 'APDO'):
        say('  %-5s mean|err| 10-20 s %.3f Hz, 25-30 s %.3f Hz, settle(5%%) %.2f s' % (
            k, fr[k]['pre'], fr[k]['post'], fr[k]['settle']))
    fig_box()
    ab = fig_ablation()
    say('ablation rel. M2 (cal): ' + json.dumps({'|'.join(map(str, k)): [round(x, 1) for x in v]
                                                 for k, v in ab.items()}))
    fig_mismatch()
    from scipy.stats import wilcoxon
    say('drift weight, paired M5 (kappa=4) vs M5k0 (kappa=0), 40 runs:')
    for env in ('cal', 'nom'):
        for pt in (0.0, 0.3):
            for sc in SCEN:
                a_ = np.array([get(env, 'M5', pt, f, sc, s)[0] for f in F0S for s in SEEDS])
                b_ = np.array([get(env, 'M5k0', pt, f, sc, s)[0] for f in F0S for s in SEEDS])
                say('  [%s] %-6s +-%2d%%  %+.1f%%  wins %d/40  p=%.1e  max %.2f -> %.2f' % (
                    env, sc, 100 * pt, 100 * (a_.mean() / b_.mean() - 1), int((a_ < b_).sum()),
                    wilcoxon(a_, b_).pvalue, b_.max(), a_.max()))
    out.close()
    return 0


if __name__ == '__main__':
    sys.exit(main())
