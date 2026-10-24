#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ZR10 速度接口下的 APDO + LSTM-DO + MPC 外环(单轴, 离散)。AST 论文实验, 2026-10-20 lyk。

只依赖 numpy/scipy, 兼容 N100 的 Python 3.8。仿真和实物共用本文件。

被控对象(扫频辨识, 指令单位"等级"):
    omega_dot = (-omega + K*(u(t-Td) + d)) / tau,   theta_dot = omega
等价于论文 J*theta_ddot = tau + H + d 的输入等效形式:
    J_eq = tau/K,  H_eq = -omega/K,  控制量 = u(t-Td),  d 为指令通道的输入等效扰动(级)。
"""
import math
from collections import deque

import numpy as np

# 2026-10-20 扫频辨识 (sweep_20261020/ZR10速度接口扫频辨识结果_20261020_lyk.txt)
AXIS_ID = {
    'yaw':   dict(K=0.805, tau=0.109, Td=0.059),
    'pitch': dict(K=0.873, tau=0.095, Td=0.066),
}


def c2d(K, tau, Ts):
    """x=[theta, omega], 输入 u(已延迟), 零阶保持精确离散化。"""
    a = math.exp(-Ts / tau)
    A = np.array([[1.0, tau * (1 - a)], [0.0, a]])
    B = np.array([K * (Ts - tau * (1 - a)), K * (1 - a)])
    return A, B


class AxisModel:
    """带 n_d 步输入延迟的增广模型 z=[theta, omega, u_{k-1}, ..., u_{k-nd}]。"""

    def __init__(self, K, tau, Td, Ts, cancel_drift=False):
        self.K, self.tau, self.Ts = K, tau, Ts
        self.nd = max(0, int(round(Td / Ts)))
        A2, B2 = c2d(K, tau, Ts)
        if cancel_drift:
            J = tau / K
            A2 = np.array([[1., Ts], [0., 1.]])
            B2 = np.array([.5 * Ts * Ts / J, Ts / J])
        n = 2 + self.nd
        A = np.zeros((n, n))
        B = np.zeros(n)
        A[:2, :2] = A2
        if self.nd == 0:
            B[:2] = B2
        else:
            A[:2, 1 + self.nd] = B2          # 作用于对象的是 u_{k-nd}
            B[2] = 1.0
            for i in range(3, 2 + self.nd):
                A[i, i - 1] = 1.0
        self.A, self.B, self.n = A, B, n
        self.J_eq = tau / K


class MPC:
    """单轴线性 MPC, 输入盒约束, 有界最小二乘精确求解。

    J = sum_{j=1..Np} q_th (th-ref)^2 + q_om (om-om_ref)^2 + r u^2 (+ 终端权 P)
    控制时域 Nc: 第 Nc 步以后输入保持不变(论文取 Nc=1)。
    """

    def __init__(self, model, Np=20, Nc=1, q_th=1.0, q_om=0.01, r=1e-4, p_term=1.0,
                 u_max=50.0):
        from scipy.optimize import lsq_linear
        self._lsq = lsq_linear
        self.m, self.Np, self.Nc, self.u_max = model, Np, Nc, u_max
        A, B, n = model.A, model.B, model.n
        # 预测 z_{k+j} = A^j z_k + sum_i G_{j,i} U_i
        Phi = np.zeros((Np, n, n))
        Gam = np.zeros((Np, n, Nc))
        Aj = np.eye(n)
        acc = np.zeros((n, Nc))
        for j in range(Np):
            ui = min(j, Nc - 1)
            acc = A @ acc
            acc[:, ui] += B
            Aj = A @ Aj
            Phi[j], Gam[j] = Aj, acc
        w = np.ones(Np)
        w[-1] += p_term
        sq = np.sqrt
        rows_th = sq(q_th * w)[:, None]
        rows_om = sq(q_om * w)[:, None]
        self.M = np.vstack([rows_th * Gam[:, 0, :], rows_om * Gam[:, 1, :],
                            sq(r) * np.eye(Nc)])
        self.Pth = rows_th * Phi[:, 0, :]
        self.Pom = rows_om * Phi[:, 1, :]
        self.wth, self.wom = rows_th[:, 0], rows_om[:, 0]
        self.U = np.zeros(Nc)
        self.Phi, self.Gam = Phi, Gam

    def solve(self, z, th_ref, om_ref, u_lo=None, u_hi=None):
        """th_ref/om_ref: 长度 Np, 对应 k+1..k+Np。返回第一个输入。"""
        lo = -self.u_max if u_lo is None else u_lo
        hi = self.u_max if u_hi is None else u_hi
        b = np.concatenate([self.wth * th_ref - self.Pth @ z,
                            self.wom * om_ref - self.Pom @ z,
                            np.zeros(self.Nc)])
        if lo >= hi:
            return 0.5 * (lo + hi)
        res = self._lsq(self.M, b, bounds=(lo, hi), method='bvls')
        self.U = res.x
        return float(res.x[0])


class LinearDO:
    """论文式 (lstm_do_dynamics) 的输入等效形式, 前向欧拉。

    omega_hat_dot = (-omega + K*(u_del + d_known + z2 + d_ff))/tau + l0 (omega - omega_hat)
    z2_dot        = -l2 z2 + L2 (omega - omega_hat)
    d_ff 为 LSTM 前馈估计(没有则为 0); d_known 为 APDO 周期估计。
    """

    def __init__(self, K, tau, Ts, l0=10.0, l2=5.0, L2=5.0, sub=4):
        self.K, self.tau, self.h, self.sub = K, tau, Ts / sub, sub
        self.l0, self.l2, self.L2 = l0, l2, L2
        self.om_hat = None
        self.z2 = 0.0

    def update(self, omega, u_del, d_known=0.0, d_ff=0.0):
        if self.om_hat is None:
            self.om_hat = omega
        for _ in range(self.sub):
            e = omega - self.om_hat
            dom = (-omega + self.K * (u_del + d_known + d_ff + self.z2)) / self.tau + self.l0 * e
            self.om_hat += self.h * dom
            self.z2 += self.h * (-self.l2 * self.z2 + self.L2 * e)
        return d_ff + self.z2


class Posterior:
    """d+ = J_eq*theta_ddot - u_del - H_eq 的因果滤波实现。

    对 omega 用 s/(T s+1), 对 u_del、omega 用 1/(T s+1), 三者同一滤波器, 不引入相对相位差。
    """

    def __init__(self, K, tau, Ts, f_c=4.0):
        self.J, self.K, self.Ts = tau / K, K, Ts
        self.a = math.exp(-2 * math.pi * f_c * Ts)
        self.om_f = None
        self.u_f = 0.0
        self.om_prev = None

    def update(self, omega, u_del):
        a = self.a
        if self.om_f is None:
            self.om_f, self.om_prev, self.u_f = omega, omega, u_del
        om_f_new = a * self.om_f + (1 - a) * omega
        acc = (om_f_new - self.om_f) / self.Ts
        self.om_f = om_f_new
        self.u_f = a * self.u_f + (1 - a) * u_del
        return self.J * acc - self.u_f + self.om_f / self.K


class APDO:
    """论文 APDO, 频率估计器按原代码形式(式 apdo_freq_estimator 第二行按代码修正):
        z1_dot  = -lam z1 - lam^2 y
        yh_dot  = (1 + w/lam^2) z1 + lam y + k (y - yh)
        w_dot   = rho z1 (y - yh)
    y 为 d+ 经带通预滤波后的信号。Q 滤波器 chi/(s+chi)[(1-mu) + mu e^{-t s}]。
    预测超前: 周期分量超前 n_ahead 步, 即把延迟环节的延迟改为 t - n_ahead*Ts。
    """

    def __init__(self, Ts, f_nom=0.6, lam=5.0, k=10.0, rho=0.05, bpf_v=3.0, chi=30.0,
                 mu=0.5, n_ahead=0, f_min=0.1, f_max=3.0, sub=10):
        self.Ts, self.sub = Ts, sub
        self.lam, self.k, self.rho = lam, k, rho
        self.w = (2 * math.pi * f_nom) ** 2
        self.wmin, self.wmax = (2 * math.pi * f_min) ** 2, (2 * math.pi * f_max) ** 2
        self.z1 = self.yh = 0.0
        w0 = 2 * math.pi * f_nom
        den = 4 + 2 * bpf_v * Ts + w0 ** 2 * Ts ** 2
        self.b0 = 2 * bpf_v * Ts / den
        self.a1 = (2 * w0 ** 2 * Ts ** 2 - 8) / den
        self.a2 = (4 - 2 * bpf_v * Ts + w0 ** 2 * Ts ** 2) / den
        self.bx = [0.0, 0.0]
        self.by = [0.0, 0.0]
        self.chi, self.mu, self.n_ahead = chi, mu, n_ahead
        self.buf = deque(maxlen=int(2.0 / (f_min * Ts)) + 4)
        self.q = 0.0
        self.dh = 0.0

    def freq_hz(self):
        return math.sqrt(self.w) / (2 * math.pi)

    def update(self, dplus):
        # 带通预滤波 (Tustin, 中心频率 f_nom)
        x = dplus
        y = self.b0 * x - self.b0 * self.bx[1] - self.a1 * self.by[0] - self.a2 * self.by[1]
        self.bx = [x, self.bx[0]]
        self.by = [y, self.by[0]]
        h = self.Ts / self.sub
        lam, k, rho = self.lam, self.k, self.rho
        for _ in range(self.sub):
            e = y - self.yh
            z1n = self.z1 + h * (-lam * self.z1 - lam * lam * y)
            yhn = self.yh + h * ((1 + self.w / lam ** 2) * self.z1 + lam * y + k * e)
            self.w = min(self.wmax, max(self.wmin, self.w + h * rho * self.z1 * e))
            self.z1, self.yh = z1n, yhn
        # Q 滤波器: 延迟一个周期的 d+ 与当前 d+ 加权, 再一阶低通
        self.buf.append(dplus)
        T = 2 * math.pi / math.sqrt(self.w)
        nd = int(round(T / self.Ts)) - self.n_ahead
        nd = max(1, min(nd, len(self.buf) - 1))
        d_del = self.buf[-1 - nd] if len(self.buf) > nd else dplus
        v = (1 - self.mu) * dplus + self.mu * d_del
        a = math.exp(-self.chi * self.Ts)
        self.q = a * self.q + (1 - a) * v
        self.dh = self.q
        return self.dh


class NumpyLSTM:
    """单层 LSTM + 线性输出的 numpy 前向(权重由 torch 离线训练导出), 逐步推理。"""

    def __init__(self, npz_path):
        p = np.load(npz_path)
        self.Wih, self.Whh, self.b = p['Wih'], p['Whh'], p['b']      # (4H,in),(4H,H),(4H,)
        self.Wout, self.bout = p['Wout'], p['bout']                 # (out,H),(out,)
        self.mu, self.sd = p['x_mu'], p['x_sd']
        self.y_sd = p['y_sd']
        self.H = self.Whh.shape[1]
        self.h = np.zeros(self.H)
        self.c = np.zeros(self.H)

    def step(self, x):
        x = (x - self.mu) / self.sd
        g = self.Wih @ x + self.Whh @ self.h + self.b
        H = self.H
        i = 1 / (1 + np.exp(-g[:H]))
        f = 1 / (1 + np.exp(-g[H:2 * H]))
        gg = np.tanh(g[2 * H:3 * H])
        o = 1 / (1 + np.exp(-g[3 * H:]))
        self.c = f * self.c + i * gg
        self.h = o * np.tanh(self.c)
        return (self.Wout @ self.h + self.bout) * self.y_sd


class AxisController:
    """单轴外环: mode in {'M1','M2','M3a','M3'}
    M1  MPC, 无扰动补偿
    M2  MPC + 单一线性 DO (集总扰动)
    M3a MPC + APDO(周期, 超前 nd) + 线性 DO(其余)
    M3  M3a + LSTM 前馈 (由外部提供 d_lstm, 本类只做组合)
    """

    def __init__(self, axis, Ts, mode='M3a', mpc_kw=None, do_kw=None, apdo_kw=None,
                 u_max=50.0):
        idp = AXIS_ID[axis]
        self.mode, self.Ts, self.u_max = mode, Ts, u_max
        self.model = AxisModel(idp['K'], idp['tau'], idp['Td'], Ts)
        self.mpc = MPC(self.model, u_max=u_max, **(mpc_kw or {}))
        self.do = LinearDO(idp['K'], idp['tau'], Ts, **(do_kw or {}))
        self.post = Posterior(idp['K'], idp['tau'], Ts)
        ak = dict(n_ahead=self.model.nd)
        ak.update(apdo_kw or {})
        self.apdo = APDO(Ts, **ak)
        self.u_hist = deque([0.0] * (self.model.nd + 1), maxlen=self.model.nd + 1)
        # MPC 预测用名义输入历史: 补偿后对象看到的等效输入 u + d ≈ u_nom
        self.un_hist = deque([0.0] * max(1, self.model.nd), maxlen=max(1, self.model.nd))
        self.u_nom = 0.0
        self.d_h = self.d_r = self.dplus = 0.0

    def u_delayed(self):
        return self.u_hist[0]          # 当前作用在对象上的 u_{k-nd}

    def step(self, theta, omega, th_ref, om_ref, d_lstm=0.0):
        """theta/omega: 当前测量; th_ref/om_ref: 长度 Np 的未来参考。返回发给 0x07 的等级(浮点)。"""
        u_del = self.u_delayed()
        self.dplus = self.post.update(omega, u_del)
        if self.mode == 'M1':
            d_h = d_r = 0.0
        elif self.mode == 'M2':
            d_h = 0.0
            d_r = self.do.update(omega, u_del)
        else:
            d_h = self.apdo.update(self.dplus)
            d_r = self.do.update(omega, u_del, d_known=d_h,
                                 d_ff=d_lstm if self.mode == 'M3' else 0.0)
        self.d_h, self.d_r = d_h, d_r
        z = np.zeros(self.model.n)
        z[0], z[1] = theta, omega
        hist = list(self.un_hist)           # [u_nom_{k-nd}, ..., u_nom_{k-1}]
        for i in range(self.model.nd):
            z[2 + i] = hist[-1 - i]
        # MPC 名义输入的可用范围扣除补偿量, 保证 u = u_nom - d 不越界
        comp = d_h + d_r
        lo, hi = -self.u_max + comp, self.u_max + comp
        self.u_nom = self.mpc.solve(z, th_ref, om_ref, max(lo, -self.u_max), min(hi, self.u_max))
        u = float(np.clip(self.u_nom - comp, -self.u_max, self.u_max))
        self.u_hist.append(u)
        self.un_hist.append(self.u_nom)
        return u
