#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M5: M4 的最小修改版(周期扰动预测进 MPC + 非周期扰动按名义输入补偿), 2026-10-22 lyk。

M4 的问题(见 通宵_M5结论_20261022.txt): M4 的 MPC 状态里放的是实际指令历史 u_{k-1..k-nd},
DO 估计 d_r 按常值进入整个预测时域, MPC 要把延迟窗口内"未补偿"的 d_r 一并抵消,
d_r -> u 的静态增益为 -3.15(M2 为 -1)。DO 的噪声和估计误差被放大 3 倍, 有未建模滞后时
DO-MPC 回路失去相位裕度, 在 ~10 Hz 形成极限环并顶满约束。

M5 (三处修改, 每处单独可关, 用于消融):
  (a) 非周期分量按名义输入补偿(与 M2 相同的确定性等价形式):
        v_k = u_k + d_r,k,  MPC 状态中的输入历史用 v,  u_k = v_k - d_r,k
      d_r -> u 增益为 -1, DO 误差系统与 MPC 名义闭环分离。
  (b) 周期分量 d_h 用谐波 RLS 外推 Np 步写进预测方程; 谐波系数投影到先验幅值球
        ||theta_h|| <= A_max   (投影是非扩张的, 预测误差有界: |d_h,pred| <= sqrt(nh) A_max)
      频率律可选 APDO(原文) 或 LIN(KoopLin 线性回归 + 先验带投影)。
  (c) 输入约束收紧: MPC 名义输入集 V_k = [-u_max+m+d_r, u_max-m+d_r] ∩ [-u_max, u_max] 的并集修正,
      保证 |u_k| <= u_max - m, m 为留给未建模高频成分的裕度; 盒约束下可行性对任意 m < u_max 平凡成立。
  conf: 可选的预测置信加权 rho in [0,1](谐波模型对 d+ 的解释度), 默认关。
接口与 PredController 相同: step(theta, omega, th_ref, om_ref, t=None) -> u。

2026-10-22 加 LSTM-DO(lstm=npz 路径, 默认 None 即原 M5):
  d_L,k  = LSTMFeedforward(theta_k, omega_k, u_{k-1})   冻结权重, 幅值/速率投影(lstm_ff.py), 只用 u_{k-1}
  d_hat_r = d_L + z                                       LinearDO 以 d_L 为已知输入(d_ff), z 为线性修正
  d_r_c  = clip(d_hat_r, -d_bar, d_bar)                   观测器内部用未饱和 d_hat_r; V_k 上下界与 u = v - d_r_c 用 d_r_c
  d_L 默认 0.8 d_bar; d_bar 默认 30 < u_max - margin = 45。

2026-10-22 加超前补偿(lead, 默认 'none' 即上面的行为不变; lead_kw 见 _lead_step):
  对象输入延迟 n_d=3 拍, 闭环残差 w_k = d_r,k - d_r^c_{k-n_d}。DO 仍以当前时刻估计运行, 只在补偿里加超前修正
        d_r^c_k = sat_dbar( d_hat_r,k + Delta_k ),  Delta_k = beta * Pi_rate(Pi_mag(Delta_raw))   (|Delta|<=D_max)
  'lstm2': lead_npz 两头网络 [d_r(k), d_r(k+n_d)]; now 头经 Pi_mag/Pi_rate 后 x alpha 作 d_ff, Delta_raw = lead - now
  'lstmd': lead_npz 单头网络直接给 Delta_raw = d_r(k+n_d) - d_r(k); d_ff 仍由 lstm(若给)提供
  'lin'  : 不学习, Delta_raw = n_d * LPF_fc(d_hat_r,k - d_hat_r,k-1)
  'lstmr': (2026-10-22 新增) 以 'lin' 为基线的残差: Delta_raw = Delta_lin + eps_gain * Pi_rate(Pi_mag(eps_hat)),
           eps_hat = 网络(psi_k, Delta_lin(k)), 训练目标 eps = D* - Delta_lin(train_lead_resid.py, lead_resid.py);
           |eps_hat| <= eps_max, |d eps_hat/dt| <= eps_rate 取 npz meta; eps_gain=0 即 'lin'
"""
import math
from collections import deque

import numpy as np

from zr10_ctrl import AXIS_ID, AxisModel, LinearDO, Posterior, APDO
from ctrl_pred import MPCdist, HarmonicPredictor
from ctrl_klin import KoopLin


def do_bw(axis, wo):
    """Gao (ACC 2003) 带宽参数化: 误差两极点放在 -wo。"""
    K, tau = AXIS_ID[axis]['K'], AXIS_ID[axis]['tau']
    return dict(l0=2.0 * wo, l2=0.0, L2=wo * wo * tau / K)


class _LinFreq:
    """把 KoopLin 包成 APDO 的 freq_hz() 接口(记录/遥测用)。"""

    def __init__(self, kl):
        self.kl = kl

    def freq_hz(self):
        return self.kl.f


class M5Controller:
    def __init__(self, axis, Ts, f_nom=0.5, wo=28.0, margin=5.0, A_max=12.0, freq='lin',
                 pred=True, nominal=True, conf=False, mpc_kw=None, do_kw=None, kl_kw=None,
                 hp_kw=None, apdo_kw=None, u_max=50.0, oracle_f=None, lstm=None, d_bar=30.0,
                 lstm_kw=None, lead='none', lead_npz=None, lead_kw=None, drift=None, drift_L=25):
        idp = AXIS_ID[axis]
        # drift: 按预测步的频率漂移权重 rho_j = clip(1 - drift * 2pi |f_k - f_{k-L}| (j + n_d) Ts, 0, 1)
        #   |f_k - f_{k-L}| 是频率估计未收敛程度的在线代理; 谐波外推 j 步的相位误差 ~ 2pi f~ j Ts,
        #   rho_j 把不可信的远端预测向 0 收缩。rho_j in [0,1] => |rho_j d_h_pred| <= |d_h_pred|,
        #   收紧约束与 ISS 界不变(eps_h 换成加权后的误差界)。None = 关(原 M5)。2026-10-22 lyk
        self.drift, self.drift_L = drift, drift_L
        self._fh = deque(maxlen=drift_L + 1)
        self.rho_end = 1.0
        self.Ts, self.u_max, self.margin, self.A_max = Ts, u_max, margin, A_max
        self.pred, self.nominal, self.conf = pred, nominal, conf
        self.model = AxisModel(idp['K'], idp['tau'], idp['Td'], Ts)
        self.mpc = MPCdist(self.model, u_max=u_max, **(mpc_kw or {}))
        self.do = LinearDO(idp['K'], idp['tau'], Ts, **(do_kw or do_bw(axis, wo)))
        self.post = Posterior(idp['K'], idp['tau'], Ts)
        self.hp = HarmonicPredictor(Ts, **(hp_kw or {}))
        self.freq = freq
        if freq == 'lin':
            self.kl = KoopLin(Ts, f_nom=f_nom, **(kl_kw or dict(lam=0.999, p_min=0.05)))
            self.apdo = _LinFreq(self.kl)
        else:
            self.apdo = APDO(Ts, **dict(dict(f_nom=f_nom), **(apdo_kw or {})))
        self.oracle_f = oracle_f
        self.d_bar = d_bar
        self.ff = None
        if lstm:
            from lstm_ff import LSTMFeedforward
            self.ff = LSTMFeedforward(lstm, Ts, **dict(dict(d_L=0.8 * d_bar), **(lstm_kw or {})))
        self.d_L = self.d_r_c = 0.0
        self.lead = lead
        self.lead_d = 0.0
        if lead != 'none':
            lk = dict(beta=1.0, alpha=1.0, fc=5.0, D_max=None, delta_D=None, d_L=0.8 * d_bar,
                      delta_L=None)
            lk.update(lead_kw or {})
            self.lead_ff = None
            lk.setdefault('eps_gain', 1.0)
            if lead in ('lstm2', 'lstmd', 'lstmr'):
                from lead_ff import LeadFeedforward
                import os
                if not os.path.isabs(lead_npz) and not os.path.isfile(lead_npz):
                    # 真机上 cwd 不一定是脚本目录: 相对路径找不到时按本文件目录解析
                    lead_npz = os.path.join(os.path.dirname(os.path.abspath(__file__)), lead_npz)
                if lead == 'lstmr':
                    from lead_resid import LeadResidFF
                    self.lead_ff = LeadResidFF(lead_npz)
                    pz = np.load(lead_npz)
                    for k in ('eps_max', 'eps_rate'):
                        lk.setdefault(k, None)
                        if lk[k] is None:
                            lk[k] = float(pz[k])
                else:
                    self.lead_ff = LeadFeedforward(lead_npz)
                meta = self.lead_ff.meta
                for k in ('D_max', 'delta_D', 'delta_L'):
                    if lk[k] is None:
                        lk[k] = meta[k]
            if lk['D_max'] is None:
                lk['D_max'], lk['delta_D'] = 6.0, 116.0   # sim 训练集 2x|D*| / 2x|diff D*|/Ts 99 分位
            self.lk = lk
            self._lf = math.exp(-2 * math.pi * lk['fc'] * Ts)
            self._ls = self._lprev_dr = None
            self._ldL = self._lD = self._leps = 0.0
            self.lead_dlin = self.lead_eps = 0.0
        self.n_step = self.n_dsat = 0
        nd = self.model.nd
        self.u_hist = deque([0.0] * (nd + 1), maxlen=nd + 1)   # 实际指令 u
        self.v_hist = deque([0.0] * (nd + 1), maxlen=nd + 1)   # 名义输入 v = u + d_r
        self.d_h = self.d_r = self.dplus = self.u_nom = 0.0
        self.rho = 1.0
        self._ve = self._vs = self._ms = None
        self.wants_t = True

    def _project(self):
        """谐波系数(不含常数项)投影到 ||theta_h||_2 <= A_max。"""
        th = self.hp.th
        n = len(th) - 1
        nr = float(np.linalg.norm(th[:n]))
        if nr > self.A_max:
            th[:n] *= self.A_max / nr

    def step(self, theta, omega, th_ref, om_ref, t=None):
        u_del = self.u_hist[0]
        dplus = self.post.update(omega, u_del)
        self.dplus = dplus
        Np = self.mpc.Np
        if self.pred:
            if self.freq == 'lin':
                f = self.kl.update(dplus)
            else:
                self.apdo.update(dplus)
                f = self.apdo.freq_hz()
            if self.oracle_f:
                f = self.oracle_f(t)
            p = self.hp.phif
            e = dplus - p @ self.hp.th if p is not None else 0.0
            self.hp.update(dplus, f)
            if self.A_max:
                self._project()
            if self.conf:
                if self._ve is None:
                    self._ve, self._vs, self._ms = e * e, 1.0, dplus
                self._ms = 0.98 * self._ms + 0.02 * dplus
                self._ve = 0.95 * self._ve + 0.05 * e * e
                self._vs = 0.98 * self._vs + 0.02 * (dplus - self._ms) ** 2
                r2 = 1.0 - self._ve / (self._vs + 1e-9)
                self.rho = 0.9 * self.rho + 0.1 * min(1.0, max(0.0, (r2 - 0.2) / 0.5))
            dh_fut = self.rho * self.hp.predict(Np)
            if self.drift:
                self._fh.append(f)
                df = abs(self._fh[-1] - self._fh[0])
                j = np.arange(Np) + self.model.nd
                rj = np.clip(1.0 - self.drift * 2 * math.pi * df * j * self.Ts, 0.0, 1.0)
                self.rho_end = float(rj[-1])
                dh_fut = rj * dh_fut
        else:
            dh_fut = np.zeros(Np)
        self.d_h = float(dh_fut[0])
        if self.ff is not None:
            # 因果: 只用上一拍实际发出的指令 u_{k-1}(整数化后), 不用 u_k
            u_prev = float(int(round(max(-self.u_max, min(self.u_max, self.u_hist[-1])))))
            self.d_L = self.ff.step(theta, omega, u_prev)
        lead_raw = None
        if self.lead in ('lstm2', 'lstmd'):
            u_prev = float(int(round(max(-self.u_max, min(self.u_max, self.u_hist[-1])))))
            o = self.lead_ff.step(theta, omega, u_prev)
            if self.lead == 'lstm2':
                lk, s = self.lk, self.lk['delta_L'] * self.Ts
                dm = min(max(float(o[0]), -lk['d_L']), lk['d_L'])
                dm = min(max(dm, self._ldL - s), self._ldL + s) if self.n_step else dm
                self._ldL = dm
                self.d_L = lk['alpha'] * dm
                lead_raw = float(o[1] - o[0])
            else:
                lead_raw = float(o[0])
        # 观测器以未饱和的 d_hat_r 运行(d_ff 进预测方程), 返回 d_hat_r = d_L + z
        self.d_r = self.do.update(omega, u_del, d_known=self.d_h, d_ff=self.d_L)
        if self.lead != 'none':
            if self.lead in ('lin', 'lstmr'):
                if self._lprev_dr is None:
                    self._lprev_dr, self._ls = self.d_r, 0.0
                self._ls = self._lf * self._ls + (1 - self._lf) * (self.d_r - self._lprev_dr)
                self._lprev_dr = self.d_r
                lead_raw = self.model.nd * self._ls
            lk = self.lk
            if self.lead == 'lstmr':
                self.lead_dlin = lead_raw
                if lk['eps_gain']:
                    u_prev = float(int(round(max(-self.u_max, min(self.u_max, self.u_hist[-1])))))
                    e = float(self.lead_ff.step(theta, omega, u_prev, lead_raw)[0])
                    e = min(max(e, -lk['eps_max']), lk['eps_max'])
                    if self.n_step:
                        s = lk['eps_rate'] * self.Ts
                        e = min(max(e, self._leps - s), self._leps + s)
                    self._leps = e
                    self.lead_eps = lk['eps_gain'] * e
                lead_raw = lead_raw + self.lead_eps
            D = min(max(lead_raw, -lk['D_max']), lk['D_max'])
            s = lk['delta_D'] * self.Ts
            D = min(max(D, self._lD - s), self._lD + s)
            self._lD = D
            self.lead_d = lk['beta'] * D
            d_rc = min(max(self.d_r + self.lead_d, -self.d_bar), self.d_bar)
            self.n_step += 1
            self.n_dsat += d_rc != self.d_r + self.lead_d
        else:
            d_rc = min(max(self.d_r, -self.d_bar), self.d_bar)
            self.n_step += 1
            self.n_dsat += d_rc != self.d_r
        self.d_r_c = d_rc
        nd = self.model.nd
        z = np.zeros(self.model.n)
        z[0], z[1] = theta, omega
        um = self.u_max - self.margin
        if self.nominal:
            hist = list(self.v_hist)
            for i in range(nd):
                z[2 + i] = hist[-1 - i]
            lo = max(-self.u_max, -um + d_rc)
            hi = min(self.u_max, um + d_rc)
            if lo >= hi:
                lo = hi = min(max(d_rc, -self.u_max), self.u_max)
            v = self.mpc.solve_d(z, th_ref, om_ref, dh_fut, lo, hi) if lo < hi else lo
            u = float(np.clip(v - d_rc, -um, um))
        else:
            hist = list(self.u_hist)
            for i in range(nd):
                z[2 + i] = hist[-1 - i]
            u = self.mpc.solve_d(z, th_ref, om_ref, dh_fut + d_rc, -um, um)
            v = u + d_rc
        self.u_hist.append(u)
        self.v_hist.append(v)
        self.u_nom = v
        return u
