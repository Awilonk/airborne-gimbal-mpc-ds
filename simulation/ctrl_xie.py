#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Xie et al. (Automatica 2026) 方法的基线实现: FODO + 已知扰动内模外推 + 扰动预览 MPC。2026-10-21 lyk。

    Xie, Yu, Qiao, Zhang, Shi, Guo, "Composite anti-disturbance control with disturbance
    prediction and utilization: A tube MPC approach", Automatica 184 (2026) 112700.

与原文对应:
  式 (2)(8)  d = V0 w + c,  w+ = W0 w  (W0 为名义频率 f_nom 及 2 次谐波的旋转矩阵, c 为常值/随机游走分量)
  式 (5)     全阶观测器 FODO, 增益取稳态 Kalman 增益(q_w, q_c 两个旋钮)
  式 (7)(11) 预测 d_{k+j} = V0 W0^j w_hat + c_hat 作为不可控输入进入 MPC
只实现名义部分(certainty equivalence), 不做 tube 收紧, 与 M4 一致, 对比公平。
W0 需要事先知道频率: f_nom 固定(原文假设), 或 oracle_f(t) 给出真频率(上界参照)。
"""
import math
from collections import deque

import numpy as np
from scipy.linalg import solve_discrete_are

from zr10_ctrl import AXIS_ID, AxisModel, c2d
from ctrl_pred import MPCdist


def rot(a):
    return np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]])


class _Freq:
    """与 APDO 接口兼容(disc_pred 记录 freq_hz)。"""

    def __init__(self):
        self.f = 0.0

    def freq_hz(self):
        return self.f


class XieController:
    wants_t = True

    def __init__(self, axis, Ts, f_nom=1.0, nh=2, q_w=0.1, q_c=1.0, q_x=(1e-6, 1e-3),
                 r_meas=(1e-3, 1e-3), mpc_kw=None, u_max=50.0, oracle_f=None):
        idp = AXIS_ID[axis]
        self.Ts, self.u_max, self.nh = Ts, u_max, nh
        self.model = AxisModel(idp['K'], idp['tau'], idp['Td'], Ts)
        self.mpc = MPCdist(self.model, u_max=u_max, **(mpc_kw or {}))
        self.A2, self.B2 = c2d(idp['K'], idp['tau'], Ts)
        self.nw = 2 * nh
        self.V = np.zeros(self.nw)
        self.V[1::2] = 1.0
        self.Q = np.diag(list(q_x) + [q_w] * self.nw + [q_c])
        self.R = np.diag(r_meas)
        self.H = np.zeros((2, 2 + self.nw + 1))
        self.H[0, 0] = self.H[1, 1] = 1.0
        self.f_nom, self.oracle_f = f_nom, oracle_f
        self._f_gain = None
        self.xh = None
        self.apdo = _Freq()
        nd = self.model.nd
        self.u_hist = deque([0.0] * (nd + 1), maxlen=nd + 1)
        self.d_h = self.d_r = self.dplus = self.u_nom = 0.0

    def _W(self, f):
        W = np.zeros((self.nw, self.nw))
        for h in range(self.nh):
            W[2 * h:2 * h + 2, 2 * h:2 * h + 2] = rot(2 * math.pi * f * (h + 1) * self.Ts)
        return W

    def _F(self, f):
        n = 2 + self.nw + 1
        F = np.zeros((n, n))
        F[:2, :2] = self.A2
        F[:2, 2:2 + self.nw] = np.outer(self.B2, self.V)
        F[:2, -1] = self.B2
        F[2:2 + self.nw, 2:2 + self.nw] = self._W(f)
        F[-1, -1] = 1.0
        return F

    def _gain(self, f):
        # 频率变化时重算稳态 Kalman 增益(固定 f 时只算一次)
        if self._f_gain is not None and abs(f - self._f_gain) < 1e-9:
            return
        F, H = self._F(f), self.H
        P = solve_discrete_are(F.T, H.T, self.Q, self.R)
        self.L = P @ H.T @ np.linalg.inv(H @ P @ H.T + self.R)
        self.F, self._f_gain = F, f

    def step(self, theta, omega, th_ref, om_ref, t=None):
        f = self.oracle_f(t) if self.oracle_f else self.f_nom
        self.apdo.f = f
        self._gain(f)
        y = np.array([theta, omega])
        if self.xh is None:
            self.xh = np.zeros(2 + self.nw + 1)
            self.xh[:2] = y
        # 预测(用作用于上一拍的 u_{k-1-nd}) + 校正
        G = np.zeros_like(self.xh)
        G[:2] = self.B2
        xp = self.F @ self.xh + G * self.u_hist[0]
        self.xh = xp + self.L @ (y - self.H @ xp)
        w, c = self.xh[2:2 + self.nw], self.xh[-1]
        Np = self.mpc.Np
        W = self._W(f)
        dh = np.zeros(Np)
        wj = w.copy()
        for j in range(Np):
            dh[j] = self.V @ wj
            wj = W @ wj
        self.d_h, self.d_r = dh[0], c
        nd = self.model.nd
        z = np.zeros(self.model.n)
        z[0], z[1] = self.xh[0], self.xh[1]
        hist = list(self.u_hist)
        for i in range(nd):
            z[2 + i] = hist[-1 - i]
        u = self.mpc.solve_d(z, th_ref, om_ref, dh + c, -self.u_max, self.u_max)
        self.u_hist.append(u)
        self.u_nom = u
        return u
