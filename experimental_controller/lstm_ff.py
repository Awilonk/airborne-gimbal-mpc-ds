#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LSTM-DO 的学习前馈 d_hat_L(纯 numpy 推理, N100 上不需要 torch)。2026-10-22 lyk。

离线训练(train_lstm.py, torch)后权重冻结, 导出 .npz。在线每拍:
    f_j   = [theta_j, omega_j, u_{j-1}, sin(theta_j), cos(theta_j)]      (theta 单位 deg, sin/cos 取弧度)
    psi_k = [f_k, f_{k-1}, f_{k-2}, f_{k-3}]                             (20 维, 起始用 f_0 重复填充)
    x_k   = (psi_k - mu_x) / sd_x
    (h_k, c_k) = LSTM(x_k, h_{k-1}, c_{k-1})                           (单层, PyTorch 门顺序 i,f,g,o)
    r_k   = W_bar h_k + b_out,   W_bar = W_out / ||W_out||_2            (谱归一化, 无激活)
    d_raw = s_y r_k + m_y                                               (s_y, m_y 为训练目标的固定标准化常数)
    d_L,k = Pi_rate( Pi_mag(d_raw) ):  |d_L,k| <= d_L,  |d_L,k - d_L,k-1| <= delta_L Ts
因果性: 第 k 拍只用到 u_{k-1}(上一拍实际发出的指令), 不用 u_k(u_k 依赖 d_hat_r,k)。
"""
import time
from collections import deque

import numpy as np

N_STACK = 4
N_F = 5


def frame(theta, omega, u_prev):
    r = np.radians(theta)
    return np.array([theta, omega, u_prev, np.sin(r), np.cos(r)], dtype=float)


def psi_offline(theta, omega, u):
    """离线批量特征, 与在线 LSTMFeedforward 完全相同的规则。u[k] 为第 k 拍发出的指令。"""
    n = len(theta)
    u_prev = np.concatenate([[0.0], np.asarray(u, float)[:-1]])
    F = np.stack([frame(theta[k], omega[k], u_prev[k]) for k in range(n)])
    out = np.zeros((n, N_STACK * N_F))
    for j in range(N_STACK):
        idx = np.maximum(np.arange(n) - j, 0)
        out[:, j * N_F:(j + 1) * N_F] = F[idx]
    return out


class LSTMFeedforward:
    def __init__(self, npz_path, Ts=0.02, d_L=None, delta_L=None):
        p = np.load(npz_path)
        self.Wih, self.Whh, self.b = p['Wih'], p['Whh'], p['b']
        Wout = p['Wout'].reshape(-1)
        self.Wbar = Wout / np.linalg.norm(Wout)          # 导出时已归一化, 这里再除一次保证 ||W_bar||_2 = 1
        self.bout = float(p['bout'])
        self.mu, self.sd = p['x_mu'], p['x_sd']
        self.s_y, self.m_y = float(p['y_sd']), float(p['y_mu'])
        self.d_L = float(p['d_L']) if d_L is None else float(d_L)
        self.delta_L = float(p['delta_L']) if delta_L is None else float(delta_L)
        self.Ts = Ts
        self.H = self.Whh.shape[1]
        self.reset()

    def reset(self):
        self.h = np.zeros(self.H)
        self.c = np.zeros(self.H)
        self.buf = None
        self.d_prev = 0.0
        self.d_raw = 0.0
        self.n = self.n_mag = self.n_rate = 0
        self.t_us = []

    def _cell(self, x):
        H = self.H
        g = self.Wih @ x + self.Whh @ self.h + self.b
        i = 1.0 / (1.0 + np.exp(-g[:H]))
        f = 1.0 / (1.0 + np.exp(-g[H:2 * H]))
        gg = np.tanh(g[2 * H:3 * H])
        o = 1.0 / (1.0 + np.exp(-g[3 * H:]))
        self.c = f * self.c + i * gg
        self.h = o * np.tanh(self.c)
        return float(self.Wbar @ self.h) + self.bout

    def step(self, theta, omega, u_prev):
        """u_prev = u_{k-1}。返回投影后的 d_L,k。"""
        t0 = time.perf_counter()
        fr = frame(theta, omega, u_prev)
        if self.buf is None:
            self.buf = deque([fr] * N_STACK, maxlen=N_STACK)
        else:
            self.buf.append(fr)
        psi = np.concatenate([self.buf[-1 - j] for j in range(N_STACK)])
        r = self._cell((psi - self.mu) / self.sd)
        d = self.s_y * r + self.m_y
        self.d_raw = d
        dm = min(max(d, -self.d_L), self.d_L)
        self.n_mag += dm != d
        s = self.delta_L * self.Ts
        dr = min(max(dm, self.d_prev - s), self.d_prev + s)
        self.n_rate += dr != dm
        self.n += 1
        self.d_prev = dr
        self.t_us.append((time.perf_counter() - t0) * 1e6)
        return dr

    def raw_sequence(self, X):
        """离线: 已标准化前的 psi 序列 X (n,20), 返回未投影输出(用于核对与 torch 一致)。"""
        self.h = np.zeros(self.H)
        self.c = np.zeros(self.H)
        out = np.array([self._cell((x - self.mu) / self.sd) for x in X])
        return self.s_y * out + self.m_y
