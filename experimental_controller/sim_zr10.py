#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ZR10 地面注入实验的离线仿真(上机前的设计验证)。2026-10-20 lyk。

    python sim_zr10.py [--trials 10] [--png out.png]

对象(1 ms 积分): 辨识的 K/tau/Td + 实测的非理想项
    死区 3/2.5 级、正负增益差 ~1-2%、角度量化 0.1 deg、角速度量化 0.0847/0.0949 deg/s、
    回读 50 Hz 带 ±4 ms 抖动、指令整数化。
注入扰动(指令通道, 单位级, 有真值):
    d_h: 两谐波, 基频 f0 在 t=Tsw 切到 1.6 f0 (对应论文仿真里的频率切换)
    d_r: 随机游走 + 阶跃 + 依赖角速度的类摩擦项
参考: 方位 A_a sin(0.8 t), 俯仰 A_e cos(0.6 t) (deg)。
"""
import argparse
import math
import sys

import numpy as np

from zr10_ctrl import AXIS_ID, AxisController

TRUE = {  # fit_asym.py 结果 + 扫频
    'yaw':   dict(Kp=1.051 * 0.805 / 1.045, Kn=1.038 * 0.805 / 1.045, dz=3.0, tau=0.109,
                  Td=0.059, q_th=0.1, q_om=0.0847),
    'pitch': dict(Kp=1.167 * 0.873 / 1.155, Kn=1.143 * 0.873 / 1.155, dz=2.5, tau=0.095,
                  Td=0.066, q_th=0.1, q_om=0.0949),
}


# 2026-10-22 真机标定选项(Plant(..., cal=CAL_REAL)), 默认 cal=None 时与原对象完全相同。
# 依据 track_20261020/trk_yaw_M{1,2,3a}_inj_20261021.csv 与 simreal.py 复现, 见 通宵_M5结论_20261022.txt:
#   lag2      指令到死区之前的额外一阶滞后(s)。22 ms 时 M1/M2/M3a/M4/M4xf 的 RMSE 与饱和率同时对上真机
#   th_noise  角度回读白噪声 std(deg, 量化前), 实测 Savitzky-Golay 残差 0.07-0.09 deg
#   om_scale  角速度回读/真实, 实测 ∫omega 比 Δtheta 小 10.5%
#   om_noise  角速度回读白噪声 std(deg/s)
# 另有可选项 om_delay(角速度回读延迟 s), K_scale / tau / dz / Td(覆盖辨识值), CAL_REAL 未用。
CAL_REAL = dict(lag2=0.022, th_noise=0.07, om_scale=1 / 1.105, om_noise=0.3)


class Plant:
    def __init__(self, axis, rng, dt=1e-3, perturb=0.0, cal=None):
        p = dict(TRUE[axis])
        self.cal = c = dict(cal or {})
        if 'K_scale' in c:
            p['Kp'] *= c['K_scale']
            p['Kn'] *= c['K_scale']
        for k in ('tau', 'dz', 'Td'):
            if k in c:
                p[k] = c[k]
        if perturb:
            for k in ('Kp', 'Kn', 'tau', 'Td'):
                p[k] *= 1 + perturb * rng.uniform(-1, 1)
        self.p, self.dt = p, dt
        self.th = self.om = 0.0
        self.qbuf = []           # (t_apply, u)
        self.u_act = 0.0
        self.x2 = 0.0
        self.rng = rng
        # 噪声用独立随机流, 不改变扰动/抖动的随机序列
        self.nrng = np.random.Generator(rng.bit_generator.jumped()) if c else None
        nd = int(round(c.get('om_delay', 0.0) / dt))
        self.omh = [0.0] * (nd + 1)

    def cmd(self, t, u):
        self.qbuf.append((t + self.p['Td'], u))

    def run(self, t0, t1, d_fun):
        p = self.p
        lag2 = self.cal.get('lag2', 0.0)
        t = t0
        while t < t1 - 1e-12:
            while self.qbuf and self.qbuf[0][0] <= t:
                self.u_act = self.qbuf.pop(0)[1]
            v = self.u_act + d_fun(t, self.om)
            if lag2:
                self.x2 += self.dt * (v - self.x2) / lag2
                v = self.x2
            if v > p['dz']:
                w_cmd = p['Kp'] * (v - p['dz'])
            elif v < -p['dz']:
                w_cmd = p['Kn'] * (v + p['dz'])
            else:
                w_cmd = 0.0
            self.om += self.dt * (w_cmd - self.om) / p['tau']
            self.th += self.dt * self.om
            if len(self.omh) > 1:
                self.omh.append(self.om)
                del self.omh[0]
            t += self.dt

    def meas(self):
        p = self.p
        c = self.cal
        if not c:
            return round(self.th / p['q_th']) * p['q_th'], round(self.om / p['q_om']) * p['q_om']
        th = self.th + c.get('th_noise', 0.0) * self.nrng.standard_normal()
        om = (self.omh[0] if len(self.omh) > 1 else self.om) * c.get('om_scale', 1.0)
        om += c.get('om_noise', 0.0) * self.nrng.standard_normal()
        return round(th / p['q_th']) * p['q_th'], round(om / p['q_om']) * p['q_om']


def make_dist(rng, T, f0, Tsw, amp_h, amp_r, fric):
    ph1, ph2 = rng.uniform(0, 2 * np.pi, 2)
    a1 = amp_h * rng.uniform(0.8, 1.2)
    a2 = 0.3 * amp_h * rng.uniform(0.5, 1.0)
    f0 = f0 * rng.uniform(0.9, 1.1)
    # 随机游走(0.1 s 节点线性插值) + 一个阶跃
    tn = np.arange(0, T + 0.2, 0.1)
    rw = np.cumsum(rng.normal(0, 0.25 * amp_r, tn.size))
    rw -= rw.mean()
    t_step, h_step = rng.uniform(0.3, 0.7) * T, amp_r * rng.choice([-1, 1])

    def phase(t):
        if t < Tsw:
            return 2 * np.pi * f0 * t
        return 2 * np.pi * f0 * Tsw + 2 * np.pi * 1.6 * f0 * (t - Tsw)

    def dh(t):
        ph = phase(t)
        return a1 * math.sin(ph + ph1) + a2 * math.sin(2 * ph + ph2)

    def dr(t, om):
        return (float(np.interp(t, tn, rw)) + (h_step if t > t_step else 0.0)
                - fric * math.tanh(om / 0.5))

    return dh, dr


def simulate(mode, axis, seed, T=40.0, Ts=0.02, amp_ref=3.0, w_ref=0.8, cosref=False,
             amp_h=8.0, amp_r=4.0, fric=2.0, f0=0.5, Tsw=20.0, perturb=0.0,
             ctrl_kw=None, log=False):
    rng = np.random.default_rng(seed)
    dh, dr = make_dist(rng, T, f0, Tsw, amp_h, amp_r, fric)
    plant = Plant(axis, rng, perturb=perturb)
    ck = dict(ctrl_kw or {})
    c = AxisController(axis, Ts, mode=mode, **ck)
    Np = c.mpc.Np
    if cosref:
        ref = lambda t: amp_ref * math.cos(w_ref * t)
        dref = lambda t: -amp_ref * w_ref * math.sin(w_ref * t)
    else:
        ref = lambda t: amp_ref * math.sin(w_ref * t)
        dref = lambda t: amp_ref * w_ref * math.cos(w_ref * t)
    plant.th = ref(0.0)
    plant.om = dref(0.0)
    n = int(T / Ts)
    rec = np.zeros((n, 9))
    t_cmd = 0.0
    for k in range(n):
        tk = k * Ts
        jit = rng.uniform(-0.004, 0.004)
        th, om = plant.meas()
        fut = tk + Ts * np.arange(1, Np + 1)
        u = c.step(th, om, np.array([ref(x) for x in fut]), np.array([dref(x) for x in fut]))
        ui = int(round(u))
        t_cmd = tk + 0.002 + abs(jit)
        plant.cmd(t_cmd, ui)
        plant.run(tk, tk + Ts, lambda t, w: dh(t) + dr(t, w))
        rec[k] = (tk, ref(tk), plant.th, ui, dh(tk), dr(tk, plant.om), c.d_h, c.d_r,
                  c.apdo.freq_hz())
    return rec


def metrics(rec, t_from=5.0):
    m = rec[:, 0] >= t_from
    e = rec[m, 2] - rec[m, 1]
    d_true = rec[m, 4] + rec[m, 5]
    d_est = rec[m, 6] + rec[m, 7]
    nr = np.sqrt(np.mean((d_est - d_true) ** 2)) / (np.ptp(d_true) + 1e-9)
    return dict(rmse=float(np.sqrt(np.mean(e ** 2))), maxe=float(np.max(np.abs(e))),
                nrmse=float(nr), sat=float(np.mean(np.abs(rec[m, 3]) >= 49.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trials', type=int, default=10)
    ap.add_argument('--png')
    ap.add_argument('--perturb', type=float, default=0.0)
    a = ap.parse_args()
    modes = ['M1', 'M2', 'M3a']
    for axis, cosref, wr, amp in (('yaw', False, 0.8, 3.0), ('pitch', True, 0.6, 2.0)):
        print('== %s  ref %.0f deg @ %.1f rad/s  (t>=5 s, %d trials)' % (axis, amp, wr, a.trials))
        for mode in modes:
            ms = [metrics(simulate(mode, axis, s, amp_ref=amp, w_ref=wr, cosref=cosref,
                                   perturb=a.perturb)) for s in range(a.trials)]
            agg = {k: (np.mean([x[k] for x in ms]), np.std([x[k] for x in ms])) for k in ms[0]}
            print('  %-4s RMSE %.3f±%.3f deg  max %.2f  dNRMSE %.3f  sat %.1f%%' % (
                mode, *agg['rmse'], agg['maxe'][0], agg['nrmse'][0], 100 * agg['sat'][0]))
    if a.png:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
        for mode, col in zip(modes, ('C0', 'C1', 'C2')):
            r = simulate(mode, 'yaw', 0)
            ax[0].plot(r[:, 0], r[:, 2] - r[:, 1], col, lw=0.8, label=mode)
            if mode == 'M3a':
                ax[1].plot(r[:, 0], r[:, 4], 'k', lw=0.8, label='d_h true')
                ax[1].plot(r[:, 0], r[:, 6], col, lw=0.8, label='APDO')
                ax[2].plot(r[:, 0], r[:, 5], 'k', lw=0.8, label='d_r true')
                ax[2].plot(r[:, 0], r[:, 7], col, lw=0.8, label='DO')
                ax[3].plot(r[:, 0], r[:, 8], col, label='f_hat')
        ax[0].set_ylabel('err deg')
        ax[1].set_ylabel('level')
        ax[2].set_ylabel('level')
        ax[3].set_ylabel('Hz')
        for x in ax:
            x.legend(loc='upper right', fontsize=7)
            x.grid(alpha=0.3)
        ax[3].set_xlabel('t s')
        fig.tight_layout()
        fig.savefig(a.png, dpi=120)
    return 0


if __name__ == '__main__':
    sys.exit(main())
