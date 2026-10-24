#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""lstmr 训练数据: 与 lstm_data.sim_run 完全相同的闭环(M2, 同 seed/对象/扰动), 另记录
  u_f      控制器输出的浮点指令(整数化之前; M5 内部 u_hist 用的是浮点值)
  dlin     Delta_lin(k): 以 (omega, u_f) 重放 M5 lin 估计链(lead_resid.replay_lin, C-lin 参数)
theta/omega/u/y 与 lstm_data_sim.npz 逐位相同(脚本末尾核对)。

    python lstm_data_resid.py -> lstm_data_resid_sim.npz
"""
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

from lstm_data import dplus_zero_phase, remove_periodic, save, load, TS
from lead_resid import replay_lin, C_BEST

HERE = os.path.dirname(os.path.abspath(__file__))


def sim_run(env, seed, f0, scen, perturb=0.1, T=30.0):
    from sim_zr10 import Plant, make_dist, CAL_REAL
    from zr10_ctrl import AxisController
    from ctrl_m5 import do_bw
    Tsw = 20.0 if scen == 'switch' else 99.0
    rng = np.random.default_rng(seed)
    dh, dr = make_dist(rng, T, f0, Tsw, 8.0, 4.0, 2.0)
    r2 = np.random.default_rng(seed)
    r2.uniform(0, 2 * np.pi, 2)
    r2.uniform(0.8, 1.2)
    r2.uniform(0.5, 1.0)
    f_true = f0 * r2.uniform(0.9, 1.1)
    plant = Plant('yaw', rng, perturb=perturb, cal=CAL_REAL if env == 'cal' else None)
    wo = 56 if env == 'nom' else 8
    c = AxisController('yaw', TS, 'M2', mpc_kw=dict(r=1e-4, q_om=0, Np=20, Nc=3),
                       do_kw=do_bw('yaw', wo))
    Np = c.mpc.Np
    ref = lambda t: 3.0 * math.sin(0.8 * t)
    dref = lambda t: 2.4 * math.cos(0.8 * t)
    plant.th, plant.om = ref(0.0), dref(0.0)
    n = int(T / TS)
    th_m, om_m, uu, uf, drt = (np.zeros(n) for _ in range(5))
    for k in range(n):
        tk = k * TS
        jit = rng.uniform(-0.004, 0.004)
        th, om = plant.meas()
        fut = tk + TS * np.arange(1, Np + 1)
        u = c.step(th, om, np.array([ref(x) for x in fut]), np.array([dref(x) for x in fut]))
        ui = int(round(u))
        plant.cmd(tk + 0.002 + abs(jit), ui)
        plant.run(tk, tk + TS, lambda t, w: dh(t) + dr(t, w))
        th_m[k], om_m[k], uu[k], uf[k], drt[k] = th, om, ui, u, dr(tk, om)
    t = np.arange(n) * TS
    dp = dplus_zero_phase(om_m, uu)
    y = remove_periodic(t, dp, [(0, Tsw, f_true), (Tsw, 1e9, 1.6 * f_true)])
    p = C_BEST[env]
    dlin = replay_lin(om_m, uf, f0, p['wo'], p['fc'])
    return dict(theta=th_m, omega=om_m, u=uu, u_f=uf, dlin=dlin, y=y, dr_true=drt,
                name='%s_s%d_f%.1f_%s' % (env, seed, f0, scen), val=seed >= 8)


def _job(a):
    return sim_run(*a)


def main():
    out = os.path.join(HERE, 'lstm_data_resid_sim.npz')
    if os.path.exists(out):
        raise SystemExit('refuse to overwrite ' + out)
    tasks = [(env, s, f0, sc) for env in ('nom', 'cal') for s in range(10)
             for f0 in (0.5, 1.0, 1.5, 2.0) for sc in ('steady', 'switch')]
    with Pool(14) as p:
        runs = p.map(_job, tasks)
    old = load(os.path.join(HERE, 'lstm_data_sim.npz'))
    dif = max(float(np.abs(a[k] - b[k]).max()) for a, b in zip(runs, old)
              for k in ('theta', 'omega', 'u', 'y'))
    assert all(a['name'] == b['name'] and a['val'] == b['val'] for a, b in zip(runs, old))
    print('与 lstm_data_sim.npz 的 theta/omega/u/y 最大差 %.3e' % dif)
    save(out, runs)


if __name__ == '__main__':
    sys.exit(main())
