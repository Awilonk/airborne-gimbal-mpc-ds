#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M5 对比: 默认仿真 + 真机标定仿真(sim_zr10.CAL_REAL), 同等调参预算。2026-10-22 lyk。

    python bench_m5.py [--jobs 6] [--out bench_m5_20261022.txt]

每个环境(nom / cal)分别调参: seed 0-2 x f0{0.5,1,1.5,2} x {steady, switch}, 目标平均 RMSE,
每个方法最多 16 组候选, 一组参数用于全部频率和场景。测试 seed 100-109。
消融(不重调, 用 M5 选出的参数只关一项):
  M5-noLIN     频率律换回 APDO
  M5-noPred    不做谐波预测(d_h=0, 只剩名义补偿 + 裕度)
  M5-noMargin  margin=0
  M5-noNom     d_r 按 M4 方式进预测(实际输入历史)
  M5-noProj    A_max=0(不投影)
失配: 对象 Kp/Kn/tau/Td 各 +-p 均匀(p=10/20/30%), 用该环境的调参结果, 不重调。
发散: RMSE >= 1 deg。
"""
import argparse
import itertools
import json
import math
import sys
import time
from multiprocessing import Pool

import numpy as np

F0S = (0.5, 1.0, 1.5, 2.0)
SCEN = {'steady': 99.0, 'switch': 20.0}
MK = dict(r=1e-4, q_om=0, Np=20, Nc=3)
TUNE = range(0, 3)
TEST = range(100, 110)
PERT = (0.1, 0.2, 0.3)
WO16 = (8, 10, 12, 14, 17, 20, 24, 28, 34, 40, 48, 56, 68, 80, 96, 112)
WO8 = (10, 14, 20, 28, 40, 56, 80, 112)

GRIDS = {
    'M1': [dict()],
    'M2': [dict(wo=w) for w in WO16],
    'M3a': [dict(wo=w, chi=c) for w in WO8 for c in (30, 60)],
    'Xie': [dict(q_w=a, q_c=b) for a, b in itertools.product((0.01, 0.03, 0.1, 0.3),
                                                             (0.1, 1.0, 10.0, 100.0))],
    'XieAd': [dict(q_w=a, q_c=b, gam=0.03) for a, b in itertools.product((0.01, 0.03, 0.1, 0.3),
                                                                         (0.1, 1.0, 10.0, 100.0))],
    'M4': [dict(wo=w, lam=l) for w in WO8 for l in (0.985, 0.995)],
    'M5': [dict(wo=w, lam=l) for w in WO8 for l in (0.985, 0.995)],
}
ABL = {'M5-noLIN': dict(freq='apdo'), 'M5-noPred': dict(pred=False),
       'M5-noMargin': dict(margin=0.0), 'M5-noNom': dict(nominal=False),
       'M5-noProj': dict(A_max=0.0)}


def build(method, p, f0):
    from zr10_ctrl import AxisController
    from ctrl_pred import PredController
    from ctrl_xie import XieController
    from ctrl_xiead import XieAdaptController
    from ctrl_m5 import M5Controller, do_bw
    if method == 'M1':
        return AxisController('yaw', 0.02, 'M1', mpc_kw=MK)
    if method == 'M2':
        return AxisController('yaw', 0.02, 'M2', mpc_kw=MK, do_kw=do_bw('yaw', p['wo']))
    if method == 'M3a':
        return AxisController('yaw', 0.02, 'M3a', mpc_kw=MK, do_kw=do_bw('yaw', p['wo']),
                              apdo_kw=dict(chi=p['chi'], f_nom=f0))
    if method == 'M4':
        return PredController('yaw', 0.02, mpc_kw=MK, do_kw=do_bw('yaw', p['wo']),
                              apdo_kw=dict(f_nom=f0), hp_kw=dict(lam=p['lam']))
    if method == 'Xie':
        return XieController('yaw', 0.02, f_nom=f0, mpc_kw=MK, **p)
    if method == 'XieAd':
        return XieAdaptController('yaw', 0.02, f_nom=f0, mpc_kw=MK, **p)
    if method.startswith('M5'):
        kw = dict(wo=p['wo'], hp_kw=dict(lam=p['lam']))
        kw.update(ABL.get(method, {}))
        return M5Controller('yaw', 0.02, f_nom=f0, mpc_kw=MK, **kw)
    raise ValueError(method)


def job(a):
    env, method, pi, p, f0, scen, seed, pert = a
    from disc_pred import run
    from sim_zr10 import metrics, CAL_REAL
    try:
        rec = run(lambda: build(method, p, f0), seed, f0, Tsw=SCEN[scen], perturb=pert,
                  cal=CAL_REAL if env == 'cal' else None)
        m = metrics(rec)
        rm = m['rmse'] if math.isfinite(m['rmse']) else 99.0
        return a, (min(rm, 99.0), m['sat'])
    except Exception:
        return a, (99.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=6)
    ap.add_argument('--out', default='bench_m5_20261022.txt')
    a = ap.parse_args()
    out = open(a.out, 'w', encoding='utf-8')

    def say(s=''):
        print(s)
        out.write(s + '\n')
        out.flush()
        sys.stdout.flush()

    say(__doc__)
    t0 = time.time()
    tasks = [(env, m, i, p, f0, sc, s, 0.0) for env in ('nom', 'cal') for m, g in GRIDS.items()
             for i, p in enumerate(g) for f0 in F0S for sc in SCEN for s in TUNE]
    with Pool(a.jobs) as pool:
        res = pool.map(job, tasks, chunksize=4)
    score = {}
    for (env, m, i, *_), (rm, sat) in res:
        score.setdefault((env, m, i), []).append(rm)
    best = {}
    for env in ('nom', 'cal'):
        say('== [%s] 调参(seed 0-2), 累计 %.0f s' % (env, time.time() - t0))
        best[env] = {}
        for m, g in GRIDS.items():
            sc = sorted((np.mean(score[(env, m, i)]), i) for i in range(len(g)))
            best[env][m] = g[sc[0][1]]
            say('  %-6s 最优 %-40s RMSE %.3f (中位 %.3f, 最差 %.3f)' % (
                m, json.dumps(best[env][m]), sc[0][0], sc[len(sc) // 2][0], sc[-1][0]))
        for k in ABL:
            best[env][k] = best[env]['M5']
    METH = list(GRIDS) + list(ABL)
    tasks = [(env, m, 0, best[env][m], f0, sc, s, pt) for env in ('nom', 'cal') for m in METH
             for pt in (0.0,) + PERT for f0 in F0S for sc in SCEN for s in TEST
             if pt == 0.0 or not m.startswith('M5-')]
    with Pool(a.jobs) as pool:
        res = pool.map(job, tasks, chunksize=4)
    R = {(env, m, pt, f0, sc, s): v for (env, m, _, _, f0, sc, s, pt), v in res}
    say()
    say('== 测试 seed 100-109, 累计 %.0f s' % (time.time() - t0))
    say('   每格: RMSE 均值 [相对同条件 M2 配对差 %, 均值] 饱和% '
        '(有发散 RMSE>=1 时标 D#, 且均值/相对差换成中位数)')
    for env in ('nom', 'cal'):
        for pt in (0.0,) + PERT:
            for sc in SCEN:
                say('-- [%s] %s  失配 +-%d%%' % (env, sc, pt * 100))
                say('  %-11s' % '' + ''.join('%-27s' % ('f0=%.1f' % f) for f in F0S) + '平均相对M2')
                for m in METH:
                    if pt and m.startswith('M5-'):
                        continue
                    cells, rels = [], []
                    for f0 in F0S:
                        x = np.array([R[(env, m, pt, f0, sc, s)][0] for s in TEST])
                        st = np.array([R[(env, m, pt, f0, sc, s)][1] for s in TEST])
                        b = np.array([R[(env, 'M2', pt, f0, sc, s)][0] for s in TEST])
                        nb = int((x >= 1.0).sum())
                        rel = (x / b - 1) * 100
                        rels.append(np.median(rel) if nb else rel.mean())
                        cells.append('%.3f[%+6.1f]%4.1f%%%s' % (np.median(x) if nb else x.mean(),
                                                               rels[-1], 100 * st.mean(),
                                                               ' D%d' % nb if nb else ''))
                    say('  %-11s' % m + ''.join('%-27s' % c for c in cells) + '%+6.1f' % np.mean(rels))
    json.dump({'best': best, 'test': {'|'.join(map(str, k)): v for k, v in R.items()}},
              open(a.out.replace('.txt', '.json'), 'w'), indent=0)
    say('总耗时 %.0f s' % (time.time() - t0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
