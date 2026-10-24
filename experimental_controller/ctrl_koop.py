#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在线 Koopman 扰动预测 + 延迟 MPC。2026-10-22 lyk。

两类预测器, 都替换 M4 中"APDO 频率 + 谐波 RLS"里的频率来源/外推方式, MPC/DO/Posterior 不变:

EVPredictor (M4-EV, EVOLVER 式基线, Jia et al. T-RO 2024 IV-E):
  提升函数 = 历史扰动 [s_k, s_{k-1}, ..., s_{k-n+1}], s = Posterior 输出 d+;
  滑窗普通最小二乘 EDMD 拟合 Koopman 矩阵, 固定周期刷新; Luenberger 观测器积分模型;
  未来 Np 步由 K^j 外推。谱无约束。

KoopPredictor (本文 KP, 针对 高延迟 / 量化噪声 / 频率切换 / 约束 MPC):
  (1) 约束谱: 扰动 = 常值 + nh=2 个谐波, Koopman 谱 {1, e^{±iθ}, e^{±i2θ}},
      步长 D 的延迟坐标上特征多项式 (z^D-1)·Π_h (z²-2cos(hθD)z+1) 为回文式, 所有特征值在单位圆上,
      谐波倍数关系内置; 只有一个未知量 θ, 频率直接由拟合给出, 不需要频率估计律。
  (2) 抗噪: 回归方程误差与回归量相关(低通后的量化噪声), 用滞后 L 步的工具变量消除偏差。
  (3) 变化检测: 长/短双窗频率一致性检验, 报警后长窗截短、谐波系数协方差重置。
  (4) 可信度注入(可选, 默认关): 预测按谐波解释度 ρ∈[0,1] 进入 MPC 预测时域。
  消融开关: iv / harm / cd / trust。
"""
import math
from collections import deque

import numpy as np

from ctrl_pred import PredController, HarmonicPredictor


class HarmonicPredictor2(HarmonicPredictor):
    """两个独立频率(消融 harm=False 用)。"""

    def __init__(self, Ts, **kw):
        super().__init__(Ts, nh=2, **kw)
        self.phi2 = 0.0
        self.w2 = 0.0

    def _reg(self, phi, phi2=None):
        if phi2 is None:
            phi2 = self._p2
        return np.array([math.sin(phi), math.cos(phi), math.sin(phi2), math.cos(phi2), 1.0])

    def update2(self, dplus, f1, f2):
        self.w2 = 2 * math.pi * f2
        self.phi2 += self.w2 * self.Ts
        self._p2 = self.phi2
        self.update(dplus, f1)

    def predict(self, n):
        out = np.zeros(n)
        for j in range(n):
            x = self._reg(self.phi + j * self.w * self.Ts, self.phi2 + j * self.w2 * self.Ts)
            out[j] = x[:-1] @ self.th[:-1]
        return out


class KoopFreq:
    """约束谱 Koopman 辨识: 只辨识 θ。

    v_k = s_k - s_{k-D}(去常值, 对应特征值 1)。回文式 Π_h(z²-2c_h z+1), c_h = cos(hθD):
        v_k + v_{k-4D} = b1 (v_{k-D}+v_{k-3D}) + b2 v_{k-2D}
        b1 = 2(c1+c2),  b2 = -(2+4 c1 c2)
    多个延迟步长 D 联合(小 D 无混叠, 大 D 抗噪), 带遗忘因子累积 S_zx, S_zy
    (z = 滞后 L 步的工具变量 或 x 本身), θ = argmin Σ_D ||S_zx b_D(θ) - S_zy||²/tr(S_zx)²。
    """

    def __init__(self, Ts, Ds=(4, 12), L=10, lam=0.99, f_min=0.2, f_max=4.5, iv=True,
                 harm=True, f_nom=1.0, ngrid=600, f_hp=0.25):
        # 二阶 Butterworth 高通预滤波(LTI, 不改变谐波递推关系), 去掉随机游走/阶跃/摩擦的低频部分
        wc = math.tan(math.pi * f_hp * Ts)
        k0 = 1 + math.sqrt(2) * wc + wc * wc
        self.hb = np.array([1, -2, 1]) / k0
        self.ha = np.array([2 * (wc * wc - 1), 1 - math.sqrt(2) * wc + wc * wc]) / k0
        self.hx = [0.0, 0.0]
        self.hy = [0.0, 0.0]
        self.Ts, self.Ds, self.L, self.lam, self.iv, self.harm = Ts, Ds, L, lam, iv, harm
        n = 5 * max(Ds) + L + 1
        self.s = deque([0.0] * n, maxlen=n)
        self.k = 0
        self.Szx = {D: np.eye(2) * 1e-6 for D in Ds}
        self.Szy = {D: np.zeros(2) for D in Ds}
        self.grid_f = np.linspace(f_min, f_max, ngrid)
        self.B = {}
        for D in Ds:
            th = 2 * math.pi * self.grid_f * Ts * D
            c1, c2 = np.cos(th), np.cos(2 * th)
            self.B[D] = np.stack([2 * (c1 + c2), -(2 + 4 * c1 * c2)], 1)
        self.f = f_nom
        self.f2 = 2 * f_nom
        self.f_min, self.f_max = f_min, f_max

    def _rx(self, D, off):
        s = self.s
        n = len(s)
        v = lambda j: s[n - 1 - off - j] - s[n - 1 - off - j - D]
        return v(0) + v(4 * D), np.array([v(D) + v(3 * D), v(2 * D)])

    def update(self, sk):
        if self.k == 0:
            self.hx = [sk, sk]
        yk = (self.hb[0] * sk + self.hb[1] * self.hx[0] + self.hb[2] * self.hx[1]
              - self.ha[0] * self.hy[0] - self.ha[1] * self.hy[1])
        self.hx = [sk, self.hx[0]]
        self.hy = [yk, self.hy[0]]
        self.s.append(yk)
        self.k += 1
        if self.k < self.s.maxlen:
            return self.f
        for D in self.Ds:
            y, x = self._rx(D, 0)
            z = self._rx(D, self.L)[1] if self.iv else x
            self.Szx[D] = self.lam * self.Szx[D] + np.outer(z, x)
            self.Szy[D] = self.lam * self.Szy[D] + z * y
        if self.k % 2:
            return self.f
        if self.harm:
            J = 0.0
            for D in self.Ds:
                r = self.B[D] @ self.Szx[D].T - self.Szy[D]
                J = J + np.einsum('ij,ij->i', r, r) / (np.sum(self.Szx[D] ** 2) + 1e-9)
            i = int(np.argmin(J))
            di = 0.0
            if 0 < i < len(J) - 1:                                      # 抛物线细化
                den = J[i - 1] - 2 * J[i] + J[i + 1]
                if den > 0:
                    di = 0.5 * (J[i - 1] - J[i + 1]) / den
            self.f = float(self.grid_f[i] + di * (self.grid_f[1] - self.grid_f[0]))
            self.f2 = 2 * self.f
        else:
            D = self.Ds[0]
            try:
                b = np.linalg.solve(self.Szx[D], self.Szy[D])
            except np.linalg.LinAlgError:
                return self.f
            sm, pr = b[0] / 2, -(b[1] + 2) / 4                         # c1+c2, c1 c2
            disc = sm * sm - 4 * pr
            if disc >= 0:
                cs = sorted([(sm + math.sqrt(disc)) / 2, (sm - math.sqrt(disc)) / 2], reverse=True)
                fs = [math.acos(max(-1.0, min(1.0, c))) / (2 * math.pi * self.Ts * D) for c in cs]
                self.f = float(np.clip(fs[0], self.f_min, self.f_max))
                self.f2 = float(np.clip(fs[1], self.f_min, 2 * self.f_max))
        return self.f

    def forget(self, g=0.05):
        for D in self.Ds:
            self.Szx[D] *= g
            self.Szy[D] *= g


class KoopProj:
    """约束谱 Koopman 的预测误差辨识(滑窗) + 双窗一致性变化检测。

    谱约束 {1, e^{±iθ}, e^{±i2θ}} 下, 窗内最小二乘拟合的残差 = 总能量 - 投影到 Koopman 模态上的能量,
    所以 θ = argmax_θ  |X(θ)|² + min(|X(2θ)|², |X(θ)|²),  X = Hann 窗加权的模态投影。
    min(.) 保证基波不弱于二次谐波, 消除 θ/2 的次谐波歧义。harm=False 时只用 |X(θ)|²。

    变化检测(cd=True): 长窗 N_l(抗噪)与短窗 N_s(快)各给一个 θ; 连续 m 次相对差 > tol
    判为扰动频率切换, 长窗有效长度截到 N_s 后随时间重新增长到 N_l(旧动态的数据被丢掉)。
    """

    def __init__(self, Ts, cyc=4.0, cyc_s=2.0, every=5, f_min=None, f_max=None, ngrid=300,
                 f_nom=1.0, harm=True, hp_rel=0.4, cd=True, tol=0.2, m=4):
        # 只用 f_nom(与其他方法相同的先验): 搜索带 [0.5, 2.2] f_nom, 长/短窗 cyc/cyc_s 个名义周期,
        # 高通截止 hp_rel*f_nom
        f_min = f_min or max(0.2, 0.5 * f_nom)
        f_max = f_max or min(4.5, 2.2 * f_nom)
        N = int(min(400, max(150, round(cyc / (f_nom * Ts)))))
        Ns = int(max(50, round(cyc_s / (f_nom * Ts))))
        N, Ns = N - N % 5, Ns - Ns % 5
        f_hp = hp_rel * f_nom
        self.Ts, self.N, self.Ns, self.every, self.harm = Ts, N, Ns, every, harm
        self.cd, self.tol, self.m = cd, tol, m
        self.hp = KoopFreq(Ts, Ds=(4,), f_hp=f_hp)          # 只借用其高通
        self.buf = deque([0.0] * N, maxlen=N)
        self.k = 0
        self.n_eff = 0
        self.cnt = 0
        self.changed = False
        self.n_alarm = 0
        self.grid_f = np.linspace(f_min, f_max, ngrid)
        self._E = {}
        self.f = self.f_s = f_nom
        self.f2 = 2 * f_nom

    def _mats(self, n):
        if n not in self._E:
            t = np.arange(n) * self.Ts
            w = np.hanning(n)
            self._E[n] = (w * np.exp(-2j * np.pi * self.grid_f[:, None] * t),
                          w * np.exp(-4j * np.pi * self.grid_f[:, None] * t))
        return self._E[n]

    def _est(self, x):
        E1, E2 = self._mats(len(x))
        P1 = np.abs(E1 @ x) ** 2
        J = P1 + np.minimum(np.abs(E2 @ x) ** 2, P1) if self.harm else P1
        i = int(np.argmax(J))
        di = 0.0
        if 0 < i < len(J) - 1:
            den = J[i - 1] - 2 * J[i] + J[i + 1]
            if den < 0:
                di = 0.5 * (J[i - 1] - J[i + 1]) / den
        return float(self.grid_f[i] + di * (self.grid_f[1] - self.grid_f[0]))

    def update(self, sk):
        h = self.hp
        if h.k == 0:
            h.hx = [sk, sk]
        yk = (h.hb[0] * sk + h.hb[1] * h.hx[0] + h.hb[2] * h.hx[1]
              - h.ha[0] * h.hy[0] - h.ha[1] * h.hy[1])
        h.hx = [sk, h.hx[0]]
        h.hy = [yk, h.hy[0]]
        h.k += 1
        self.buf.append(yk)
        self.k += 1
        self.n_eff = min(self.N, self.n_eff + 1)
        self.changed = False
        if self.n_eff < self.Ns or self.k % self.every:
            return self.f
        x = np.fromiter(self.buf, float)
        # 有效长度量化到 5 的倍数, 投影矩阵可缓存
        n = self.n_eff - self.n_eff % 5
        self.f = self._est(x[-n:])
        if self.cd and n >= self.N - 4:
            self.f_s = self._est(x[-self.Ns:])
            # 短窗分辨率约 1/(Ns Ts), 阈值不低于半个分辨率单元
            thr = max(self.tol * self.f, 0.5 / (self.Ns * self.Ts))
            if abs(self.f_s - self.f) > thr:
                self.cnt += 1
                if self.cnt >= self.m:
                    self.n_eff = self.Ns
                    self.f = self.f_s
                    self.cnt = 0
                    self.changed = True
                    self.n_alarm += 1
            else:
                self.cnt = 0
        self.f2 = 2 * self.f
        return self.f

    def forget(self, g=0.05):
        pass


class KoopController(PredController):
    """KP: M4 结构, 频率来自约束谱 Koopman 辨识, 带变化检测和可信度注入。"""

    def __init__(self, axis, Ts, f_nom=1.0, iv=True, harm=True, cd=True, trust=False,
                 kf_kw=None, est='proj', p_reset=10.0, **kw):
        super().__init__(axis, Ts, apdo_kw=dict(f_nom=f_nom), **kw)
        if est == 'proj':
            self.kf = KoopProj(Ts, harm=harm, f_nom=f_nom, cd=cd, **(kf_kw or {}))
        else:
            self.kf = KoopFreq(Ts, iv=iv, harm=harm, f_nom=f_nom, **(kf_kw or {}))
        if not harm:
            self.hp = HarmonicPredictor2(Ts)
        self.harm, self.cd, self.trust, self.p_reset = harm, cd, trust, p_reset
        self.ve = self.vs = None
        self.ms = 0.0
        self.rho = 1.0
        self.wants_t = True
        self.f_show = f_nom
        self.apdo = _FreqShim(self)          # 记录用: rec[:, 8] 显示 Koopman 频率

    @property
    def n_alarm(self):
        return getattr(self.kf, 'n_alarm', 0)

    def step(self, theta, omega, th_ref, om_ref, t=None):
        u_del = self.u_hist[0]
        dplus = self.post.update(omega, u_del)
        self.dplus = dplus
        f = self.kf.update(dplus)
        self.f_show = f
        if getattr(self.kf, 'changed', False):
            # 频率切换: 谐波系数协方差放大, 幅相快速重估
            self.hp.P = self.hp.P + np.eye(len(self.hp.th)) * self.p_reset
        p = self.hp.phif
        e = dplus - p @ self.hp.th if p is not None else 0.0
        if self.harm:
            self.hp.update(dplus, f)
        else:
            self.hp.update2(dplus, f, self.kf.f2)
        if self.trust:
            # 可信度: 谐波模型对 d+ 的解释度 R², 映射到 [0,1] 缩放预测
            if self.ve is None:
                self.ve, self.vs, self.ms = e * e, 1.0, dplus
            self.ms = 0.98 * self.ms + 0.02 * dplus
            self.ve = 0.95 * self.ve + 0.05 * e * e
            self.vs = 0.98 * self.vs + 0.02 * (dplus - self.ms) ** 2
            r2 = 1.0 - self.ve / (self.vs + 1e-9)
            self.rho = 0.9 * self.rho + 0.1 * min(1.0, max(0.0, (r2 - 0.2) / 0.5))
        dh_fut = self.rho * self.hp.predict(self.mpc.Np)
        return self._finish(theta, omega, th_ref, om_ref, u_del, dh_fut)

    def _finish(self, theta, omega, th_ref, om_ref, u_del, dh_fut):
        self.d_h = dh_fut[0]
        self.d_r = self.do.update(omega, u_del, d_known=self.d_h)
        z = np.zeros(self.model.n)
        z[0], z[1] = theta, omega
        hist = list(self.u_hist)
        for i in range(self.model.nd):
            z[2 + i] = hist[-1 - i]
        u = self.mpc.solve_d(z, th_ref, om_ref, dh_fut + self.d_r, -self.u_max, self.u_max)
        self.u_hist.append(u)
        self.u_nom = u
        return u


class _FreqShim:
    def __init__(self, c):
        self.c = c

    def freq_hz(self):
        return self.c.f_show


class EVController(KoopController):
    """M4-EV: EVOLVER 式在线 EDMD(历史扰动提升, 普通 LS, 固定周期刷新, 无约束谱), 观测器积分。"""

    def __init__(self, axis, Ts, f_nom=1.0, n_lift=6, N_win=150, T2=10, l_obs=0.5, **kw):
        super().__init__(axis, Ts, f_nom=f_nom, **kw)
        self.n, self.N, self.T2, self.l = n_lift, N_win, T2, l_obs
        self.buf = deque(maxlen=N_win + n_lift)
        self.Krow = None
        self.zh = np.zeros(n_lift)
        self.k = 0
        self.f_show = f_nom
        self.apdo = _FreqShim(self)
        self.a = self.post.a

    def _fit(self):
        s = np.array(self.buf)
        n = self.n
        X = np.stack([s[n - 1 - i:len(s) - 1 - i] for i in range(n)], 1)   # z_k
        y = s[n:]
        kr, *_ = np.linalg.lstsq(X, y, rcond=None)
        self.Krow = kr
        # 主导振荡频率(仅记录)
        Km = np.zeros((n, n))
        Km[0] = kr
        Km[1:, :-1] = np.eye(n - 1)
        ev = np.linalg.eigvals(Km)
        ev = ev[np.imag(ev) > 1e-6]
        if ev.size:
            e0 = ev[np.argmax(np.abs(ev))]
            self.f_show = abs(np.angle(e0)) / (2 * math.pi * self.Ts)

    def _Kz(self, z):
        return np.concatenate([[self.Krow @ z], z[:-1]])

    def step(self, theta, omega, th_ref, om_ref, t=None):
        u_del = self.u_hist[0]
        s = self.post.update(omega, u_del)
        self.dplus = s
        self.buf.append(s)
        self.k += 1
        Np = self.mpc.Np
        if self.Krow is None and len(self.buf) == self.buf.maxlen:
            self._fit()
        elif self.Krow is not None and self.k % self.T2 == 0:
            self._fit()
        if self.Krow is None:
            self.zh = np.concatenate([[s], self.zh[:-1]])
            dh_fut = np.zeros(Np)
        else:
            zp = self._Kz(self.zh)                       # 观测器: 预测 + 校正
            zp[0] += self.l * (s - zp[0])
            self.zh = zp
            traj = np.zeros(Np + 1)
            traj[0] = self.zh[1]                         # s_{k-1}
            z = self.zh.copy()
            traj[1] = z[0]
            for j in range(2, Np + 1):
                z = self._Kz(z)
                traj[j] = z[0]
            if not np.all(np.isfinite(traj)) or np.max(np.abs(traj)) > 200:
                traj = np.full(Np + 1, s)
            # 逆 Posterior 低通: d_k = (s_k - a s_{k-1})/(1-a)
            dh_fut = (traj[1:] - self.a * traj[:-1]) / (1 - self.a)
        return self._finish(theta, omega, th_ref, om_ref, u_del, dh_fut)
