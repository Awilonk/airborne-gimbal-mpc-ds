#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M5 试验: Xie 的 FODO + 扰动预览 MPC, 但内模频率在线自适应。2026-10-21 lyk。

频率由 FODO 自己的一次谐波状态 w1 估计: 每拍 w1 的相角增量 dphi 即 2 pi f Ts,
f_hat 用一阶低通跟踪 dphi/(2 pi Ts), 并限制在 [f_min, f_max]。
Kalman 增益在 0.02 Hz 网格上缓存, 避免每拍解 DARE。
"""
import math

import numpy as np

from ctrl_xie import XieController


class XieAdaptController(XieController):
    def __init__(self, axis, Ts, f_nom=1.0, gam=0.05, f_min=0.2, f_max=3.5, grid=0.02,
                 warm=2.0, **kw):
        super().__init__(axis, Ts, f_nom=f_nom, **kw)
        self.fh, self.gam, self.f_min, self.f_max, self.grid = f_nom, gam, f_min, f_max, grid
        self.warm, self.k = int(warm / Ts), 0
        self._cache = {}
        self.oracle_f = self._fhat
        self._t = None

    def _fhat(self, t):
        return round(self.fh / self.grid) * self.grid

    def _gain(self, f):
        key = round(f / self.grid)
        if key not in self._cache:
            self._f_gain = None
            super()._gain(f)
            self._cache[key] = (self.L, self.F)
        self.L, self.F = self._cache[key]
        self._f_gain = f

    def step(self, theta, omega, th_ref, om_ref, t=None):
        w_old = None if self.xh is None else self.xh[2:4].copy()
        u = super().step(theta, omega, th_ref, om_ref, t)
        w_new = self.xh[2:4]
        self.k += 1
        if w_old is not None and self.k > self.warm:
            # rot() 的约定: w+ = [[c, s], [-s, c]] w, 相角按 atan2(w0, w1) 前进
            a0 = math.atan2(w_old[0], w_old[1])
            a1 = math.atan2(w_new[0], w_new[1])
            dphi = (a1 - a0 + math.pi) % (2 * math.pi) - math.pi
            amp = math.hypot(*w_new)
            if amp > 0.5:
                fm = dphi / (2 * math.pi * self.Ts)
                self.fh += self.gam * (fm - self.fh)
                self.fh = min(self.f_max, max(self.f_min, self.fh))
        self.apdo.f = self.fh
        return u
