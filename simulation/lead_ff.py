#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""超前(lead)扰动预测的 numpy 推理, 2026-10-22 lyk。与 lstm_ff.LSTMFeedforward 同一特征/同一 LSTM 结构,
输出层为 m 维(谱归一化 W_bar = W_out/||W_out||_2, m = 1 或 2):
    kind 'lstm2': 输出 [d_r(k), d_r(k+n_d)]   (两头)
    kind 'lstmd': 输出 [d_r(k+n_d) - d_r(k)]  (直接预测超前增量)
因果: 第 k 拍特征只到 theta_k, omega_k, u_{k-1}。step() 返回未投影的原始输出(级), 投影在控制器里做。
"""
from collections import deque

import numpy as np

from lstm_ff import frame, N_STACK


class LeadFeedforward:
    def __init__(self, npz_path):
        p = np.load(npz_path)
        self.Wih, self.Whh, self.b = p['Wih'], p['Whh'], p['b']
        W = np.atleast_2d(p['Wout'])
        self.Wbar = W / np.linalg.norm(W, 2)
        self.bout = np.atleast_1d(p['bout']).astype(float)
        self.mu, self.sd = p['x_mu'], p['x_sd']
        self.s_y = np.atleast_1d(p['y_sd']).astype(float)
        self.m_y = np.atleast_1d(p['y_mu']).astype(float)
        self.kind = str(p['kind'])
        self.meta = {k: float(p[k]) for k in ('d_L', 'delta_L', 'D_max', 'delta_D') if k in p.files}
        self.H = self.Whh.shape[1]
        self.reset()

    def reset(self):
        self.h = np.zeros(self.H)
        self.c = np.zeros(self.H)
        self.buf = None

    def _cell(self, x):
        H = self.H
        g = self.Wih @ x + self.Whh @ self.h + self.b
        i = 1.0 / (1.0 + np.exp(-g[:H]))
        f = 1.0 / (1.0 + np.exp(-g[H:2 * H]))
        gg = np.tanh(g[2 * H:3 * H])
        o = 1.0 / (1.0 + np.exp(-g[3 * H:]))
        self.c = f * self.c + i * gg
        self.h = o * np.tanh(self.c)
        return self.Wbar @ self.h + self.bout

    def step(self, theta, omega, u_prev):
        fr = frame(theta, omega, u_prev)
        if self.buf is None:
            self.buf = deque([fr] * N_STACK, maxlen=N_STACK)
        else:
            self.buf.append(fr)
        psi = np.concatenate([self.buf[-1 - j] for j in range(N_STACK)])
        return self.s_y * self._cell((psi - self.mu) / self.sd) + self.m_y

    def raw_sequence(self, X):
        """X: 未标准化的 psi 序列 (n, 20)。"""
        self.reset()
        out = np.array([self._cell((x - self.mu) / self.sd) for x in X])
        return out * self.s_y + self.m_y
