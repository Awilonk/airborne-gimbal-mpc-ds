#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LSTM-DO 前馈的离线训练数据(sim / real), 2026-10-22 lyk。

    python lstm_data.py sim  -> lstm_data_sim.npz
    python lstm_data.py real -> lstm_data_real.npz

每条序列(一次运行)给出 theta, omega(回读), u(第 k 拍发出的指令等级), 目标 y(非周期扰动, 级)。
目标(离线, 非因果):
  d+ = J_eq*omega_dot - u_del + omega/K      (Posterior 的同一逆动力学, u_del = u_{k-nd}, nd=3)
       其中 omega、u_del 用零相位 2 阶 Butterworth 4 Hz(filtfilt)代替 Posterior 的因果一阶低通(无滞后)
  y  = d+ - 周期分量。周期分量: 已知注入基频 f(切换前 f, 切换后 1.6 f), 每段对 d+ 做
       [sin/cos(h*phi), h=1,2](乘渐入渐出包络)最小二乘拟合后减去; 无注入的运行 y = d+(已 4 Hz 低通)。
sim: disc_pred.run 同一对象/扰动, 控制器 M2(各环境 bench_m5 选出的 wo: nom 56, cal 8), 对象 perturb=0.1,
     环境 nom / cal(CAL_REAL), seed 0-9, f0 {0.5,1,1.5,2}, {steady, switch}。seed 0-7 训练, 8-9 验证。
     另存真值 dr(t, omega)(随机游走+阶跃+摩擦)仅作参考指标。
real: track_20261020/*.csv 与 gcs_log(遥测, 同一批 2026-10-21 注入实验, seed 0, f0 0.5, 切换 15 s),
      以及 ident_data/yaw*.csv(开环阶跃辨识, 无周期注入)。按运行划分训练/验证。
"""
import glob
import json
import math
import os
import sys

import numpy as np
from scipy.signal import butter, filtfilt

from zr10_ctrl import AXIS_ID

HERE = os.path.dirname(os.path.abspath(__file__))
TS = 0.02
ID = AXIS_ID['yaw']
ND = int(round(ID['Td'] / TS))
F_C = 4.0


def dplus_zero_phase(omega, u, ts=TS):
    K, tau = ID['K'], ID['tau']
    J = tau / K
    u = np.asarray(u, float)
    u_del = np.concatenate([np.zeros(ND), u[:-ND]]) if ND else u
    b, a = butter(2, F_C / (0.5 / ts))
    om_f = filtfilt(b, a, omega)
    ud_f = filtfilt(b, a, u_del)
    acc = np.gradient(om_f, ts)
    return J * acc - ud_f + om_f / K


def remove_periodic(t, d, segs, env=None):
    """segs: [(t_a, t_b, f_hz)]; 每段独立做 2 谐波最小二乘, 返回 d - 拟合。"""
    y = d.copy()
    e = np.ones_like(t) if env is None else env
    for ta, tb, f in segs:
        m = (t >= ta) & (t < tb)
        if m.sum() < 20:
            continue
        ph = 2 * np.pi * f * t[m]
        X = np.stack([e[m] * fn(h * ph) for h in (1, 2) for fn in (np.sin, np.cos)]
                     + [np.ones(m.sum())], 1)
        c, *_ = np.linalg.lstsq(X, d[m], rcond=None)
        y[m] = d[m] - X[:, :4] @ c[:4]
    return y


# ------------------------------------------------------------------ sim
def sim_run(env, seed, f0, scen, perturb=0.1, T=30.0):
    """与 disc_pred.run 相同的闭环(M2), 记录回读量。"""
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
    th_m, om_m, uu, drt = (np.zeros(n) for _ in range(4))
    for k in range(n):
        tk = k * TS
        jit = rng.uniform(-0.004, 0.004)
        th, om = plant.meas()
        fut = tk + TS * np.arange(1, Np + 1)
        u = c.step(th, om, np.array([ref(x) for x in fut]), np.array([dref(x) for x in fut]))
        ui = int(round(u))
        plant.cmd(tk + 0.002 + abs(jit), ui)
        plant.run(tk, tk + TS, lambda t, w: dh(t) + dr(t, w))
        th_m[k], om_m[k], uu[k], drt[k] = th, om, ui, dr(tk, om)
    t = np.arange(n) * TS
    dp = dplus_zero_phase(om_m, uu)
    y = remove_periodic(t, dp, [(0, Tsw, f_true), (Tsw, 1e9, 1.6 * f_true)])
    return dict(theta=th_m, omega=om_m, u=uu, y=y, dr_true=drt,
                name='%s_s%d_f%.1f_%s' % (env, seed, f0, scen), val=seed >= 8)


def _sim_job(a):
    return sim_run(*a)


def build_sim(out):
    from multiprocessing import Pool
    tasks = [(env, s, f0, sc) for env in ('nom', 'cal') for s in range(10)
             for f0 in (0.5, 1.0, 1.5, 2.0) for sc in ('steady', 'switch')]
    with Pool(12) as p:
        runs = p.map(_sim_job, tasks)
    save(out, runs)


# ------------------------------------------------------------------ real
def env_track(t, dur=30.0):
    e = np.ones_like(t)
    m = t < 2.0
    e[m] = 0.5 - 0.5 * np.cos(np.pi * t[m] / 2.0)
    m = t > dur - 2.0
    e[m] = np.maximum(0.0, 0.5 - 0.5 * np.cos(np.pi * (dur - t[m]) / 2.0))
    return e


def build_real(out):
    from zr10_track import true_f_inj
    f_true = true_f_inj(0, 0.5)
    trk = os.path.join(HERE, 'track_20261020')
    runs = []
    csv_runs = {}
    import pandas as pd
    for f in sorted(glob.glob(os.path.join(trk, '*.csv'))):
        d = pd.read_csv(f)
        csv_runs[os.path.basename(f)] = d
    # gcs 遥测: 注入运行(含 CSV 没落地的 7 次)
    gruns, cur = [], None
    for line in open(os.path.join(HERE, 'gcs_log', 'gcs_20261021_131738.jsonl'), encoding='utf-8'):
        r = json.loads(line)
        if 'start' in r:
            cur = dict(s=r['start'], rows=[])
            gruns.append(cur)
        elif 't' in r and cur is not None:
            cur['rows'].append(r)
    VAL = {'trk_yaw_M2_inj_20261021.csv', 'trk_yaw_M4_inj_wo20_20261021.csv',
           'trk_yaw_Xie_inj_x_20261021.csv', 'trk_yaw_M1_r1e-3.csv', 'yaw_mid_L5-20_h1.csv'}
    for g in gruns:
        name = g['s']['out']
        if name in csv_runs:
            d = csv_runs[name]
            t, th, om, u = (d[c].values.astype(float) for c in ('t', 'theta', 'omega', 'u'))
        else:
            R = g['rows']
            t = np.array([x['t'] for x in R])
            th = np.array([x['th'] for x in R], float)
            om = np.array([x['om'] for x in R], float)
            u = np.array([x['u'] for x in R], float)
        dp = dplus_zero_phase(om, u)
        y = remove_periodic(t, dp, [(0, 15.0, f_true), (15.0, 1e9, 1.6 * f_true)], env_track(t))
        runs.append(dict(theta=th, omega=om, u=u, y=y, name=name, val=name in VAL))
    for name in ('trk_yaw_M1.csv', 'trk_yaw_M1_r1e-3.csv'):
        d = csv_runs[name]
        th, om, u = (d[c].values.astype(float) for c in ('theta', 'omega', 'u'))
        runs.append(dict(theta=th, omega=om, u=u, y=dplus_zero_phase(om, u), name=name,
                         val=name in VAL))
    for f in sorted(glob.glob(os.path.join(HERE, '..', 'ident_data', 'yaw*.csv'))):
        d = pd.read_csv(f)
        d = d[d.kind == 'fb']
        th = np.degrees(np.unwrap(np.radians(d.yaw.values.astype(float))))
        th = th - th[0]
        om = d.ryaw_raw.values.astype(float) * 0.0847
        u = d.cmd_axis_level.values.astype(float)
        name = os.path.basename(f)
        runs.append(dict(theta=th, omega=om, u=u, y=dplus_zero_phase(om, u), name=name,
                         val=name in VAL))
    save(out, runs)


def save(out, runs):
    flat = {}
    for i, r in enumerate(runs):
        for k, v in r.items():
            flat['%d/%s' % (i, k)] = np.asarray(v)
    flat['n'] = np.array(len(runs))
    np.savez_compressed(out, **flat)
    nt = sum(len(r['y']) for r in runs if not r['val'])
    nv = sum(len(r['y']) for r in runs if r['val'])
    print('%s: %d runs, train %d / val %d samples, y std %.2f' % (
        out, len(runs), nt, nv, np.std(np.concatenate([r['y'] for r in runs]))))


def load(path):
    p = np.load(path, allow_pickle=False)
    runs = []
    for i in range(int(p['n'])):
        keys = [k.split('/', 1)[1] for k in p.files if k.startswith('%d/' % i)]
        r = {k: p['%d/%s' % (i, k)] for k in keys}
        r['name'] = str(r['name'])
        r['val'] = bool(r['val'])
        runs.append(r)
    return runs


if __name__ == '__main__':
    which = sys.argv[1]
    if which == 'sim':
        build_sim(os.path.join(HERE, 'lstm_data_sim.npz'))
    else:
        build_real(os.path.join(HERE, 'lstm_data_real.npz'))
