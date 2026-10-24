#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AST Sec. 4: add the drift-weighted M5 (kappa=4, L=50) and its ablations to results_sec4.json. 2026-10-22 lyk.

    python run_sec4_rho.py [--jobs 14]

Same protocol as run_sec4.py (seeds 100-109, f0 {0.5,1,1.5,2}, steady / switch at 20 s, mismatch
+-{0,10,20,30}%, ablations at 0 and 30%), same tuned wo / lam per env (bench_m5 'best'['M5']).
New method keys (the old 'M5*' keys are kept and are M5 with kappa = 0):
  M5r            M5 + drift weight (proposed, paper "M5")
  M5r-noRho      = M5 (kappa = 0), recomputed for a self-contained ablation
  M5r-noLIN / -noPred / -noMargin / -noNom / -noProj    one component removed from M5r
  M5r-conf       M5r + r2-based confidence weight (ctrl_m5 conf=True), evaluated for the remark
Traces: M5r and M5r-noLIN added to traces_sec4.npz (run cal, seed 100, f0 = 1, switch).
"""
import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

import numpy as np  # noqa: E402

from bench_m5 import F0S, SCEN, TEST, PERT, ABL, MK  # noqa: E402

KAPPA, L = 4.0, 50
RES = os.path.join(HERE, 'results_sec4.json')
TRC = os.path.join(HERE, 'traces_sec4.npz')
R0 = json.load(open(RES))
BEST = R0['best']
VAR = {'M5r': {}, 'M5r-noRho': dict(drift=None), 'M5r-conf': dict(conf=True)}
VAR.update({'M5r-' + k[3:]: v for k, v in ABL.items()})


def build(method, env, f0):
    from ctrl_m5 import M5Controller
    p = BEST[env]['M5']
    kw = dict(wo=p['wo'], hp_kw=dict(lam=p['lam']), drift=KAPPA, drift_L=L)
    kw.update(VAR[method])
    return M5Controller('yaw', 0.02, f_nom=f0, mpc_kw=MK, **kw)


def job(a):
    env, method, f0, scen, seed, pert = a
    from disc_pred import run
    from sim_zr10 import metrics, CAL_REAL
    try:
        rec = run(lambda: build(method, env, f0), seed, f0, Tsw=SCEN[scen], perturb=pert,
                  cal=CAL_REAL if env == 'cal' else None)
        m = metrics(rec)
        rm = m['rmse'] if math.isfinite(m['rmse']) else 99.0
        return a, (min(rm, 99.0), m['sat'], min(m['maxe'], 999.0))
    except Exception:
        return a, (99.0, 1.0, 999.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=14)
    a = ap.parse_args()
    t0 = time.time()
    tasks = [(env, m, f0, sc, s, pt) for env in ('cal', 'nom') for m in VAR
             for pt in (0.0,) + PERT for f0 in F0S for sc in SCEN for s in TEST
             if m == 'M5r' or pt in (0.0, 0.3)]
    print('%d runs' % len(tasks))
    sys.stdout.flush()
    with Pool(a.jobs) as pool:
        res = pool.map(job, tasks, chunksize=4)
    R0['test'].update({'|'.join(map(str, k)): v for k, v in res})
    R0['drift'] = dict(kappa=KAPPA, L=L)
    json.dump(R0, open(RES, 'w'), indent=0)
    print('test done %.0f s' % (time.time() - t0))
    from disc_pred import run
    from sim_zr10 import CAL_REAL
    D = dict(np.load(TRC))
    for m in ('M5r', 'M5r-noLIN'):
        D[m.replace('-', '_')] = run(lambda: build(m, 'cal', 1.0), 100, 1.0, Tsw=SCEN['switch'],
                                     cal=CAL_REAL)
    np.savez(TRC, **D)
    print('traces done %.0f s' % (time.time() - t0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
