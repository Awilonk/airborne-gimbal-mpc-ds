#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[drift 版: M5 均带 kappa=4, L=50] M5+LSTM 公平对比: 加 LSTM 后重调 DO 带宽 wo / 速率限 delta_L / 输出增益 alpha, 2026-10-22 lyk。

    python bench_lstm_retune.py [--jobs 14] [--out bench_lstm_retune_rho_20261022.txt]

动机: bench_lstm_20261022 沿用 M5-noLSTM 的 wo(nom 56, cal 10)。DO 终值界 ~ (delta_e + eps_h)/wo,
delta_e 为 LSTM 残差增量, 加 LSTM 后最优 wo 与 delta_L 可能变化, 应各自重调。
调参只在 seed 0-2(与 LSTM 训练 seed 0-9 重叠, 对 LSTM 有利), f0 {0.5,1,1.5,2} x {steady, switch},
目标: 平均 RMSE。每个环境(nom / cal)、每个方法独立选一组参数, 用于全部频率和场景。
  M5-noLSTM    wo in {5,10,20,28,40,56}                                          (6 组)
  M5+LSTMsim   wo x delta_L 倍数 {0.5,1,2}(相对训练时 99 分位) x alpha {0.5,1}   (36 组)
  M5+LSTMreal  同上
  lam = 0.985 固定(bench_m5 结果)。alpha 乘在投影后的 d_L 上(d_L 输出 = alpha * Pi_rate(Pi_mag(d_raw)))。
测试 seed 100-104。参考行 "(旧)": bench_lstm 的固定参数(nom wo=56, cal wo=10, 倍数 1, alpha 1)。
"""
import argparse
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
F0S = (0.5, 1.0, 1.5, 2.0)
SCEN = {'steady': 99.0, 'switch': 20.0}
MK = dict(r=1e-4, q_om=0, Np=20, Nc=3)
TUNE = range(0, 3)
TEST = range(100, 105)
LAM = 0.985
WO = (5, 10, 20, 28, 40, 56)
NPZ = {'M5-noLSTM': None, 'M5+LSTMsim': 'lstm_yaw_sim.npz', 'M5+LSTMreal': 'lstm_yaw_real.npz'}
DL0 = {k: float(np.load(os.path.join(HERE, v))['delta_L']) for k, v in NPZ.items() if v}
GRIDS = {'M5-noLSTM': [dict(wo=w, ks=1.0, al=1.0) for w in WO]}
for _k in DL0:
    GRIDS[_k] = [dict(wo=w, ks=s, al=a) for w in WO for s in (0.5, 1.0, 2.0) for a in (0.5, 1.0)]
OLD = {'nom': 56, 'cal': 10}


def key(p):
    return (p['wo'], p['ks'], p['al'])


def job(a):
    env, meth, p, f0, scen, seed = a
    from disc_pred import run
    from sim_zr10 import metrics, CAL_REAL
    from ctrl_m5 import M5Controller
    lp = NPZ[meth]
    box = {}

    def mk():
        kw = {}
        if lp:
            kw = dict(lstm=os.path.join(HERE, lp), lstm_kw=dict(delta_L=p['ks'] * DL0[meth]))
        c = M5Controller('yaw', 0.02, f_nom=f0, mpc_kw=MK, wo=p['wo'], hp_kw=dict(lam=LAM), drift=4.0, drift_L=50, **kw)
        if lp and p['al'] != 1.0:
            raw_step, al = c.ff.step, p['al']
            c.ff.step = lambda th, om, up: al * raw_step(th, om, up)
        box['c'] = c
        return c
    try:
        rec = run(mk, seed, f0, Tsw=SCEN[scen], cal=CAL_REAL if env == 'cal' else None)
        m = metrics(rec)
        c = box['c']
        rm = m['rmse'] if math.isfinite(m['rmse']) else 99.0
        mm = rec[:, 0] >= 5.0
        edr = float(np.sqrt(np.mean((rec[mm, 7] - rec[mm, 5]) ** 2)))
        return a, (min(rm, 99.0), m['sat'], c.n_dsat / max(1, c.n_step), edr)
    except Exception:
        return a, (99.0, 1.0, 1.0, 99.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=14)
    ap.add_argument('--out', default=os.path.join(HERE, 'bench_lstm_retune_rho_20261022.txt'))
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit('refuse to overwrite ' + a.out)
    out = open(a.out, 'w', encoding='utf-8')

    def say(s=''):
        print(s)
        out.write(s + '\n')
        out.flush()
        sys.stdout.flush()
    say(__doc__)
    say('训练时 delta_L(级/s): ' + ', '.join('%s %.1f' % kv for kv in DL0.items()))
    t0 = time.time()
    tasks = [(env, m, p, f0, sc, s) for env in ('nom', 'cal') for m, g in GRIDS.items()
             for p in g for f0 in F0S for sc in SCEN for s in TUNE]
    with Pool(a.jobs) as pool:
        res = pool.map(job, tasks, chunksize=4)
    score = {}
    for (env, m, p, *_), v in res:
        score.setdefault((env, m, key(p)), []).append(v[0])
    best = {}
    for env in ('nom', 'cal'):
        say('== [%s] 调参 seed 0-2 (%d 例/组), 累计 %.0f s' % (env, 8 * len(TUNE), time.time() - t0))
        for m, g in GRIDS.items():
            sc = sorted((np.mean(score[(env, m, key(p))]), i) for i, p in enumerate(g))
            best[(env, m)] = g[sc[0][1]]
            say('  %-12s 最优 wo=%-3g dL倍数=%-3g alpha=%-3g  RMSE %.4f (中位 %.4f, 最差 %.4f)' % (
                (m,) + key(best[(env, m)]) + (sc[0][0], sc[len(sc) // 2][0], sc[-1][0])))
            if m != 'M5-noLSTM':
                row = [min(np.mean(score[(env, m, key(p))]) for p in g if p['wo'] == w) for w in WO]
                say('      各 wo 最好: ' + '  '.join('%g:%.4f' % (w, r) for w, r in zip(WO, row)))
            else:
                say('      各 wo: ' + '  '.join('%g:%.4f' % (p['wo'], np.mean(score[(env, m, key(p))]))
                                              for p in g))
    METH = [(m, m) for m in GRIDS] + [(m + '(旧)', m) for m in GRIDS]
    MD = dict(METH)
    par = {}
    for env in ('nom', 'cal'):
        for lab, m in METH:
            par[(env, lab)] = (dict(wo=OLD[env], ks=1.0, al=1.0) if lab.endswith('(旧)')
                               else best[(env, m)])
    uniq = sorted({(env, MD[lab], key(par[(env, lab)]), f0, sc, s)
                   for env in ('nom', 'cal') for lab, _ in METH for f0 in F0S for sc in SCEN
                   for s in TEST})
    tasks = [(e, m, dict(wo=k[0], ks=k[1], al=k[2]), f0, sc, s) for e, m, k, f0, sc, s in uniq]
    with Pool(a.jobs) as pool:
        res = pool.map(job, tasks, chunksize=2)
    R = {(e, m, key(p), f0, sc, s): v for (e, m, p, f0, sc, s), v in res}

    def get(env, lab, f0, sc, s):
        return R[(env, MD[lab], key(par[(env, lab)]), f0, sc, s)]
    say()
    say('== 测试 seed 100-104 (40 例/环境), 累计 %.0f s' % (time.time() - t0))
    say('   基准 = 重调后的 M5-noLSTM。相对 = 配对 RMSE 比 -1 的均值; 胜 = RMSE 严格更小的例数;')
    say('   u饱和 = metrics 的 sat(指令顶约束占比); dsat = |d_hat_r|>30 拍数占比;'
        ' dr误差 = d_hat_r 对真值 d_r 的 RMSE(级, t>=5 s)')
    say('  %-5s %-17s %-14s %8s %8s %6s %7s %7s %7s' % ('env', '方法', 'wo/倍数/alpha', 'RMSE',
                                                       '相对', '胜/40', 'u饱和', 'dsat', 'dr误差'))
    for env in ('nom', 'cal'):
        for lab, m in METH:
            v = np.array([get(env, lab, f0, sc, s) for f0 in F0S for sc in SCEN for s in TEST])
            b = np.array([get(env, 'M5-noLSTM', f0, sc, s)[0] for f0 in F0S for sc in SCEN
                          for s in TEST])
            x = v[:, 0]
            say('  %-5s %-17s %-14s %8.4f %+7.1f%% %6d %6.2f%% %6.3f%% %7.2f' % (
                env, lab, '%g/%g/%g' % key(par[(env, lab)]), x.mean(), 100 * (x / b - 1).mean(),
                int((x < b).sum()), 100 * v[:, 1].mean(), 100 * v[:, 2].mean(), v[:, 3].mean()))
        say()
    say('分场景相对重调 M5-noLSTM(%, f0=0.5/1/1.5/2):')
    for env in ('nom', 'cal'):
        for sc in SCEN:
            for lab in ('M5+LSTMsim', 'M5+LSTMreal'):
                rr = []
                for f0 in F0S:
                    x = np.array([get(env, lab, f0, sc, s)[0] for s in TEST])
                    b = np.array([get(env, 'M5-noLSTM', f0, sc, s)[0] for s in TEST])
                    rr.append(100 * (x / b - 1).mean())
                say('  [%s] %-7s %-12s ' % (env, sc, lab) + ' '.join('%+6.1f' % r for r in rr))
    say('总耗时 %.0f s' % (time.time() - t0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
