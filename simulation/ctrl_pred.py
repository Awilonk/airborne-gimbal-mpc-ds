#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M4: 分离感知 MPC(周期扰动进入预测时域)。离线判别实验用, 2026-10-21 lyk。

与 M3a 的区别:
  M3a  u = u_nom - (d_h + d_r), 扰动按当前时刻常值补偿(APDO 只超前链路延迟 nd 步)。
  M4   周期分量 d_h 用 APDO 频率 + 谐波最小二乘外推出未来 Np 步轨迹, 直接写进 MPC 预测方程;
       非周期分量 d_r(线性 DO)按常值进入预测。MPC 直接输出实际指令 u, 盒约束就是实际输入约束。
  内环滞后 tau 和链路延迟 Td 都在预测模型里, 因此周期扰动的相位滞后由 MPC 自动补偿。
"""
import math
from collections import deque

import numpy as np

from zr10_ctrl import AXIS_ID, AxisModel, MPC, LinearDO, Posterior, APDO


class MPCdist(MPC):
    """MPC + 已知扰动序列 d_k..d_{k+Np-1}(与 u_{k-nd} 同通道进入)。"""

    def __init__(self, model, **kw):
        super().__init__(model, **kw)
        A, n, Np = model.A, model.n, self.Np
        E = np.zeros(n)
        E[:2] = model.B[:2] if model.nd == 0 else A[:2, 1 + model.nd]
        Psi = np.zeros((Np, n, Np))
        acc = np.zeros((n, Np))
        for j in range(Np):
            acc = A @ acc
            acc[:, j] += E
            Psi[j] = acc
        self.Dth = self.wth[:, None] * Psi[:, 0, :]
        self.Dom = self.wom[:, None] * Psi[:, 1, :]

    def solve_d(self, z, th_ref, om_ref, dvec, lo, hi):
        b = np.concatenate([self.wth * th_ref - self.Pth @ z - self.Dth @ dvec,
                            self.wom * om_ref - self.Pom @ z - self.Dom @ dvec,
                            np.zeros(self.Nc)])
        res = self._lsq(self.M, b, bounds=(lo, hi), method='bvls')
        self.U = res.x
        return float(res.x[0])


class HarmonicPredictor:
    """以 APDO 频率估计积分出相位 phi, 对 d+ 做带遗忘因子的 RLS 谐波拟合。

    d+ 由 Posterior 的一阶低通(f_c)得到, 有相位滞后; 回归量经过同一低通, 拟合系数即对应未滤波信号。
    回归量 [sin phi, cos phi, sin 2phi, cos 2phi, 1], 常数项吸收慢变部分, 不用于预测。
    """

    def __init__(self, Ts, f_c=4.0, lam=0.985, nh=2, p_max=None):
        # p_max: 有界协方差 RLS(2026-10-22 晚加, 默认 None = 原算法)。每拍对称化 P, 迹超过 p_max 时等比缩回。
        #   低基频(0.25 Hz)时 lam=0.985 的记忆(~1.3 s)短于一个周期, 回归量不持续激励, 原算法约 40 s 后
        #   P 失去正定(cond 1e16, 迹为负), 系数发散; 有界协方差保证 P 有界正定。
        self.Ts, self.nh, self.lam = Ts, nh, lam
        self.p_max = p_max
        self.a = math.exp(-2 * math.pi * f_c * Ts)
        n = 2 * nh + 1
        self.th = np.zeros(n)
        self.P = np.eye(n) * 100.0
        self.phif = None
        self.phi = 0.0

    def _reg(self, phi):
        r = []
        for h in range(1, self.nh + 1):
            r += [math.sin(h * phi), math.cos(h * phi)]
        return np.array(r + [1.0])

    def update(self, dplus, f_hz):
        self.w = 2 * math.pi * f_hz
        self.phi += self.w * self.Ts
        x = self._reg(self.phi)
        if self.phif is None:
            self.phif = x.copy()
        self.phif = self.a * self.phif + (1 - self.a) * x
        p = self.phif
        Pp = self.P @ p
        g = Pp / (self.lam + p @ Pp)
        self.th += g * (dplus - p @ self.th)
        self.P = (self.P - np.outer(g, Pp)) / self.lam
        if self.p_max:
            self.P = 0.5 * (self.P + self.P.T)
            tr = float(np.trace(self.P))
            if not (tr <= self.p_max):          # 含 NaN
                self.P *= self.p_max / tr if tr > 0 else 0.0
                if tr <= 0:
                    self.P = np.eye(len(self.th)) * self.p_max / len(self.th)

    def predict(self, n):
        """d_h 在 k, k+1, ..., k+n-1 的预测(不含常数项)。"""
        out = np.zeros(n)
        for j in range(n):
            out[j] = self._reg(self.phi + j * self.w * self.Ts)[:-1] @ self.th[:-1]
        return out


class PredController:
    """M4 单轴外环, 接口与 AxisController 相同。"""

    def __init__(self, axis, Ts, mode='M4', mpc_kw=None, do_kw=None, apdo_kw=None,
                 hp_kw=None, u_max=50.0, oracle_f=None):
        idp = AXIS_ID[axis]
        self.mode, self.Ts, self.u_max = mode, Ts, u_max
        self.model = AxisModel(idp['K'], idp['tau'], idp['Td'], Ts)
        self.mpc = MPCdist(self.model, u_max=u_max, **(mpc_kw or {}))
        self.do = LinearDO(idp['K'], idp['tau'], Ts, **(do_kw or {}))
        self.post = Posterior(idp['K'], idp['tau'], Ts)
        self.apdo = APDO(Ts, **(apdo_kw or {}))
        self.hp = HarmonicPredictor(Ts, **(hp_kw or {}))
        self.oracle_f = oracle_f
        nd = self.model.nd
        self.u_hist = deque([0.0] * (nd + 1), maxlen=nd + 1)
        self.d_h = self.d_r = self.dplus = self.u_nom = 0.0

    def step(self, theta, omega, th_ref, om_ref, t=None):
        u_del = self.u_hist[0]
        dplus = self.post.update(omega, u_del)
        self.dplus = dplus
        self.apdo.update(dplus)
        f = self.oracle_f(t) if self.oracle_f else self.apdo.freq_hz()
        self.hp.update(dplus, f)
        Np = self.mpc.Np
        dh_fut = self.hp.predict(Np)
        self.d_h = dh_fut[0]
        self.d_r = self.do.update(omega, u_del, d_known=self.d_h)
        nd = self.model.nd
        z = np.zeros(self.model.n)
        z[0], z[1] = theta, omega
        hist = list(self.u_hist)          # [u_{k-nd}, ..., u_{k-1}, (u_k 未定)]
        for i in range(nd):
            z[2 + i] = hist[-1 - i]
        dvec = dh_fut + self.d_r
        u = self.mpc.solve_d(z, th_ref, om_ref, dvec, -self.u_max, self.u_max)
        self.u_hist.append(u)
        self.u_nom = u
        return u
