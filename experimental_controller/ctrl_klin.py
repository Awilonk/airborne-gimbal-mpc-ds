#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""约束谱 Koopman 频率辨识的线性回归版(可证明收敛), 2026-10-22 lyk。

基波谱 {e^{±iθ}}: 任意 D, 滤波后的周期分量满足 v_k + v_{k-2D} = a v_{k-D}, a = 2cos(θD)。
对 a 线性 -> 带遗忘因子 RLS(或 IV-RLS), a 投影到先验倍频程带 [2cos(θmax D), 2cos(θmin D)]。
D 由先验 f_nom 选成 θD≈π/2(回归条件最好)。前置二阶带通(LTI, 不改变单频递推)压二次谐波和低频。
"""
import math
from collections import deque

import numpy as np

from ctrl_koop import KoopController


class KoopLin:
    def __init__(self, Ts, f_nom=1.0, lam=0.995, band=(0.89, 1.77), Q=2.0, iv=0, every=1, p_min=0.0, p_max=1e2,
                 rst=0.0, mu=0.02):
        self.Ts, self.lam, self.iv = Ts, lam, iv
        self.p_min, self.p_max, self.rst, self.mu = p_min, p_max, rst, mu
        self.m2 = 1e-6
        self.r2 = 0.0
        self.D = D = max(2, int(round(1.0 / (4 * f_nom * Ts))))
        self.fmin, self.fmax = band[0] * f_nom, band[1] * f_nom
        self.amin = 2 * math.cos(2 * math.pi * self.fmax * Ts * D)
        self.amax = 2 * math.cos(2 * math.pi * self.fmin * Ts * D)
        # RBJ 带通, 中心 f_nom 的几何中点
        f0 = f_nom * math.sqrt(band[0] * band[1])
        w0 = 2 * math.pi * f0 * Ts
        al = math.sin(w0) / (2 * Q)
        a0 = 1 + al
        self.b = np.array([al, 0, -al]) / a0
        self.a = np.array([-2 * math.cos(w0), 1 - al]) / a0
        self.x = [0.0, 0.0]
        self.y = [0.0, 0.0]
        n = 2 * D + iv + 1
        self.v = deque([0.0] * n, maxlen=n)
        self.k = 0
        self.ah = 2 * math.cos(2 * math.pi * f_nom * Ts * D)
        self.P = p_max
        self.f = f_nom
        self.f2 = 2 * f_nom

    def update(self, s):
        y = (self.b[0] * s + self.b[1] * self.x[0] + self.b[2] * self.x[1]
             - self.a[0] * self.y[0] - self.a[1] * self.y[1])
        self.x = [s, self.x[0]]
        self.y = [y, self.y[0]]
        self.v.append(y)
        self.k += 1
        if self.k < 3 * len(self.v):
            return self.f
        v, D, n = self.v, self.D, len(self.v)
        g = lambda j: v[n - 1 - j]
        yk = g(0) + g(2 * D)
        phi = g(D)
        z = g(D + self.iv) if self.iv else phi
        # 归一化: m2 = phi^2 的滑动均值, 回归量/输出同除 sqrt(m2), 估计器增益与扰动幅值无关
        self.m2 = max(1e-12, (1 - self.mu) * self.m2 + self.mu * (phi * phi + 1e-9))
        sc = 1.0 / math.sqrt(self.m2)
        yk, phi, z = yk * sc, phi * sc, z * sc
        e = yk - phi * self.ah
        self.r2 = (1 - self.mu) * self.r2 + self.mu * e * e
        if self.rst and self.r2 > self.rst:                         # 残差触发协方差重置
            self.P = self.p_max
            self.r2 = 0.0
        # 标量 (IV-)RLS
        den = self.lam + phi * self.P * z
        K = self.P * z / den
        self.ah += K * (yk - phi * self.ah)
        self.P = min(self.p_max, max(self.p_min, (self.P - K * phi * self.P) / self.lam))
        self.ah = min(self.amax, max(self.amin, self.ah))          # 投影到先验带
        self.f = math.acos(self.ah / 2) / (2 * math.pi * self.Ts * D)
        self.f2 = 2 * self.f
        return self.f

    def forget(self, g=0.05):
        pass


class KoopLinController(KoopController):
    def __init__(self, axis, Ts, f_nom=1.0, kl_kw=None, **kw):
        super().__init__(axis, Ts, f_nom=f_nom, cd=False, **kw)
        self.kf = KoopLin(Ts, f_nom=f_nom, **(kl_kw or {}))
