#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""残差超前预测器 lead='lstmr' 的公共部分, 2026-10-22 lyk。

以 'lin' 超前为基线(ctrl_m5 lead='lin', 参数取 bench_lead.py 中 C-lin 的调参结果):
    Delta_lin(k) = n_d * LPF_fc(d_hat_r,k - d_hat_r,k-1)
    D*(k)        = d_r(k+n_d) - d_r(k)                      (与 lstmd 相同的零相位目标 y)
    eps(k)       = D*(k) - Delta_lin(k)                      (网络目标)
    eps_hat      -> |eps_hat| <= eps_max (= 2 x 训练集 |eps| 99 分位), 再速率投影 |d eps_hat/dt| <= eps_rate
    Delta_raw    = Delta_lin + eps_hat, 之后与其他 lead 相同走 D_max / beta / sat
网络输入: lstm_ff 的 psi_k(20 维, 只到 u_{k-1}) 再拼上 Delta_lin(k)(因果, 控制器内部量) = 21 维。
eps_hat = 0 即 C-lin。

replay_lin(): 在记录的闭环数据 (theta, omega, u) 上重放 M5 的估计链(Posterior -> KoopLin -> 谐波 RLS ->
LinearDO -> lin 低通), 得到与在环控制器相同的 Delta_lin(估计链是 (omega, u_{k-n_d}) 的因果函数, 不依赖控制律)。
"""
import math
from collections import deque

import numpy as np

from lead_ff import LeadFeedforward
from lstm_ff import frame, N_STACK

MK = dict(r=1e-4, q_om=0, Np=20, Nc=3)
LAM = 0.985
# bench_lead_20261022.txt 中 C-lin 的调参结果(seed 0-2)
C_BEST = {'nom': dict(wo=40, beta=0.5, fc=5.0), 'cal': dict(wo=5, beta=0.5, fc=2.0)}


def make_m5_lin(f0, wo, fc, beta=1.0):
    from ctrl_m5 import M5Controller
    return M5Controller('yaw', 0.02, f_nom=f0, mpc_kw=MK, wo=wo, hp_kw=dict(lam=LAM),
                        lead='lin', lead_kw=dict(beta=beta, fc=fc))


def replay_lin(omega, u, f0, wo, fc):
    """u[k] = 第 k 拍发出的指令。返回 Delta_lin(k) (未经 D_max/rate/beta)。与 M5Controller.step 逐行对应。"""
    c = make_m5_lin(f0, wo, fc)
    nd = c.model.nd
    Np = c.mpc.Np
    n = len(omega)
    out = np.zeros(n)
    ls, prev = 0.0, None
    for k in range(n):
        u_del = float(u[k - nd - 1]) if k - nd - 1 >= 0 else 0.0
        # 控制器内 u_hist 长度 nd+1, 第 k 拍 u_hist[0] = u_{k-nd-1}
        dplus = c.post.update(omega[k], u_del)
        f = c.kl.update(dplus)
        c.hp.update(dplus, f)
        if c.A_max:
            c._project()
        dh = c.rho * c.hp.predict(Np)
        d_r = c.do.update(omega[k], u_del, d_known=float(dh[0]), d_ff=0.0)
        if prev is None:
            prev, ls = d_r, 0.0
        ls = c._lf * ls + (1 - c._lf) * (d_r - prev)
        prev = d_r
        out[k] = nd * ls
    return out


class LeadResidFF(LeadFeedforward):
    """psi_k(20) + Delta_lin(k) 的 21 维输入, 单头输出 eps_hat(未投影)。"""

    def step(self, theta, omega, u_prev, dlin):
        fr = frame(theta, omega, u_prev)
        if self.buf is None:
            self.buf = deque([fr] * N_STACK, maxlen=N_STACK)
        else:
            self.buf.append(fr)
        psi = np.concatenate([self.buf[-1 - j] for j in range(N_STACK)] + [[dlin]])
        return self.s_y * self._cell((psi - self.mu) / self.sd) + self.m_y

    def load_meta(self, p):
        return {k: float(p[k]) for k in ('eps_max', 'eps_rate') if k in p.files}
