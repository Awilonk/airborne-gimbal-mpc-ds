#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AST Sec. 4 draft: rerun of the bench_m5 test phase + representative traces. 2026-10-22 lyk.

    python run_sec4.py [--jobs 14]          # simulate (cached in results_sec4.json / traces_sec4.npz)

Tuned parameters are taken from ../bench_m5_20261022.json ('best'; tuning on seeds 0-2, budget
<= 16 candidates per method, one set for all f0 and both scenarios). No retuning here.
Test: seeds 100-109, f0 in {0.5,1,1.5,2} Hz, scenarios steady / switch (f -> 1.6 f at 20 s),
mismatch Kp/Kn/tau/Td +-{0,10,20,30}% (ablations at 0 and 30%). Env 'cal' = CAL_REAL, 'nom' = default.
Uses the controller code as it is in the parent folder (ctrl_m5.py with the d_bar = 30 clip).
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

from bench_m5 import build, F0S, SCEN, TEST, PERT, GRIDS, ABL  # noqa: E402

ENVS = ('cal', 'nom')
METH = list(GRIDS) + list(ABL)
BEST = json.load(open(os.path.join(PARENT, 'bench_m5_20261022.json')))['best']
RES = os.path.join(HERE, 'results_sec4.json')
TRC = os.path.join(HERE, 'traces_sec4.npz')


def job(a):
    env, method, f0, scen, seed, pert = a
    from disc_pred import run
    from sim_zr10 import metrics, CAL_REAL
    p = BEST[env][method]
    try:
        rec = run(lambda: build(method, p, f0), seed, f0, Tsw=SCEN[scen], perturb=pert,
                  cal=CAL_REAL if env == 'cal' else None)
        m = metrics(rec)
        rm = m['rmse'] if math.isfinite(m['rmse']) else 99.0
        return a, (min(rm, 99.0), m['sat'], min(m['maxe'], 999.0))
    except Exception:
        return a, (99.0, 1.0, 999.0)


def true_f0(seed, f0):
    """Replicates the rng draws of sim_zr10.make_dist up to the frequency factor."""
    rng = np.random.default_rng(seed)
    rng.uniform(0, 2 * np.pi, 2)
    rng.uniform(0.8, 1.2)
    rng.uniform(0.5, 1.0)
    return f0 * rng.uniform(0.9, 1.1)


def traces(env='cal', seed=100, f0=1.0, scen='switch'):
    from disc_pred import run
    from sim_zr10 import CAL_REAL
    out = {}
    for m in ('M2', 'Xie', 'M4', 'M5', 'M5-noLIN'):
        rec = run(lambda: build(m, BEST[env][m], f0), seed, f0, Tsw=SCEN[scen],
                  cal=CAL_REAL if env == 'cal' else None)
        out[m.replace('-', '_')] = rec
    out['f_true0'] = np.array(true_f0(seed, f0))
    out['Tsw'] = np.array(SCEN[scen])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=14)
    a = ap.parse_args()
    t0 = time.time()
    tasks = [(env, m, f0, sc, s, pt) for env in ENVS for m in METH for pt in (0.0,) + PERT
             for f0 in F0S for sc in SCEN for s in TEST
             if pt in (0.0, 0.3) or not m.startswith('M5-')]
    print('%d runs' % len(tasks))
    sys.stdout.flush()
    with Pool(a.jobs) as pool:
        res = pool.map(job, tasks, chunksize=4)
    json.dump({'best': BEST, 'test': {'|'.join(map(str, k)): v for k, v in res}},
              open(RES, 'w'), indent=0)
    print('test done %.0f s' % (time.time() - t0))
    np.savez(TRC, **traces())
    print('traces done %.0f s' % (time.time() - t0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
