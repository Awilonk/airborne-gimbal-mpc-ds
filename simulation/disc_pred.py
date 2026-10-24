#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线判别实验: 周期扰动进预测时域(M4)能否在 1-1.5 Hz 拉开与单一 DO(M2)的差距。2026-10-21 lyk。

    python disc_pred.py [--seeds 5] [--freqs 0.5 1.0 1.5 2.0]

对象、扰动、参考、量化、抖动与 sim_zr10.simulate 完全相同(复用 Plant / make_dist)。
M2/M3a 用 tune5 的参数, M2 的 DO 带宽 lo/hi 两档每个频率取较好者(对基线有利)。
"""
import argparse
import math
import sys
import time

import numpy as np

from sim_zr10 import Plant, make_dist, metrics
from zr10_ctrl import AxisController, AXIS_ID
from ctrl_pred import PredController

MK = dict(r=1e-4, q_om=0, Np=20, Nc=3)
DO_LO = dict(l0=10, l2=0.5, L2=30)
DO_HI = dict(l0=30, l2=0.5, L2=100)
_id = AXIS_ID['yaw']
N_LEAD = int(round(_id['Td'] / 0.02)) + int(round(_id['tau'] / 0.02))


def run(make_ctrl, seed, f0, T=30.0, Ts=0.02, amp_ref=3.0, w_ref=0.8, Tsw=20.0,
        amp_h=8.0, amp_r=4.0, fric=2.0, axis='yaw', perturb=0.0, cal=None):
    rng = np.random.default_rng(seed)
    dh, dr = make_dist(rng, T, f0, Tsw, amp_h, amp_r, fric)
    plant = Plant(axis, rng, perturb=perturb, cal=cal)
    c = make_ctrl()
    Np = c.mpc.Np
    ref = lambda t: amp_ref * math.sin(w_ref * t)
    dref = lambda t: amp_ref * w_ref * math.cos(w_ref * t)
    plant.th, plant.om = ref(0.0), dref(0.0)
    n = int(T / Ts)
    rec = np.zeros((n, 9))
    for k in range(n):
        tk = k * Ts
        jit = rng.uniform(-0.004, 0.004)
        th, om = plant.meas()
        fut = tk + Ts * np.arange(1, Np + 1)
        R, dR = np.array([ref(x) for x in fut]), np.array([dref(x) for x in fut])
        u = c.step(th, om, R, dR, t=tk) if isinstance(c, PredController) or getattr(c, "wants_t", False) else c.step(th, om, R, dR)
        ui = int(round(u))
        plant.cmd(tk + 0.002 + abs(jit), ui)
        plant.run(tk, tk + Ts, lambda t, w: dh(t) + dr(t, w))
        rec[k] = (tk, ref(tk), plant.th, ui, dh(tk), dr(tk, plant.om), c.d_h, c.d_r,
                  c.apdo.freq_hz())
    return rec


def summarize(recs):
    ms = [metrics(r) for r in recs]
    rm = np.array([m['rmse'] for m in ms])
    # 频率切换后 2 s 内的 RMSE(恢复段), 与切换前稳态段
    def seg(r, a, b):
        m = (r[:, 0] >= a) & (r[:, 0] < b)
        return math.sqrt(np.mean((r[m, 2] - r[m, 1]) ** 2))
    pre = np.array([seg(r, 10, 20) for r in recs])
    post = np.array([seg(r, 20, 22) for r in recs])
    nr = np.array([m['nrmse'] for m in ms])
    sat = np.array([m['sat'] for m in ms])
    return dict(rmse=rm, pre=pre, post=post, nrmse=nr, sat=sat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--freqs', type=float, nargs='+', default=[0.5, 1.0, 1.5, 2.0])
    a = ap.parse_args()
    seeds = range(100, 100 + a.seeds)      # 与调参用的 seed 0-2 分开
    for f0 in a.freqs:
        ak = dict(chi=60, f_nom=f0 * 1.2)
        apk = dict(f_nom=f0 * 1.2)
        # 真实频率(含 make_dist 的 ±10% 随机)不易拿到, oracle 用名义 f0, 切换后 1.6 f0
        ofun = lambda t, f0=f0: f0 if t < 20.0 else 1.6 * f0
        cands = {
            'M1': lambda: AxisController('yaw', 0.02, 'M1', mpc_kw=MK),
            'M2lo': lambda: AxisController('yaw', 0.02, 'M2', mpc_kw=MK, do_kw=DO_LO),
            'M2hi': lambda: AxisController('yaw', 0.02, 'M2', mpc_kw=MK, do_kw=DO_HI),
            'M3a': lambda: AxisController('yaw', 0.02, 'M3a', mpc_kw=MK, do_kw=DO_HI, apdo_kw=ak),
            # M3a 超前同时覆盖链路延迟 nd 与内环滞后 tau/Ts(更公平的 M3a)
            'M3L': lambda: AxisController('yaw', 0.02, 'M3a', mpc_kw=MK, do_kw=DO_HI,
                                          apdo_kw=dict(ak, n_ahead=N_LEAD)),
            'M4lo':lambda: PredController('yaw', 0.02, mpc_kw=MK, do_kw=DO_LO, apdo_kw=apk),
            'M4hi': lambda: PredController('yaw', 0.02, mpc_kw=MK, do_kw=DO_HI, apdo_kw=apk),
            'M4or': lambda: PredController('yaw', 0.02, mpc_kw=MK, do_kw=DO_LO, apdo_kw=apk,
                                           oracle_f=ofun),
        }
        t0 = time.time()
        res = {k: summarize([run(fn, s, f0) for s in seeds]) for k, fn in cands.items()}
        base = min(('M2lo', 'M2hi'), key=lambda k: res[k]['rmse'].mean())
        print('== f0 %.1f Hz (切换后 %.2f Hz), %d seeds, 基线 M2=%s, %.0f s' % (
            f0, 1.6 * f0, a.seeds, base, time.time() - t0))
        b = res[base]['rmse']
        for k, r in res.items():
            rel = (r['rmse'] / b - 1) * 100
            print('  %-5s RMSE %.3f±%.3f  稳态(10-20s) %.3f  切换后2s %.3f  dNRMSE %.3f  '
                  'sat %4.1f%%  相对M2 %+5.1f%%±%.1f' % (
                      k, r['rmse'].mean(), r['rmse'].std(), r['pre'].mean(), r['post'].mean(),
                      r['nrmse'].mean(), 100 * r['sat'].mean(), rel.mean(), rel.std()))
        sys.stdout.flush()
    return 0


if __name__ == '__main__':
    sys.exit(main())
