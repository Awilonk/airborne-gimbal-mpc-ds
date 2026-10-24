#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""离线训练 LSTM-DO 前馈(torch), 导出 numpy 推理用 .npz。2026-10-22 lyk。

    python train_lstm.py sim  [--H 24]   -> lstm_yaw_sim.npz
    python train_lstm.py real [--H 16]   -> lstm_yaw_real.npz

网络: 单层 LSTM(H) + 谱归一化线性输出 r = (W_out/||W_out||_2) h + b_out(无激活),
d_raw = y_sd * r + y_mu(训练目标的固定标准化常数)。输入 psi_k 见 lstm_ff.py。
训练: 长 WIN 拍的窗口, 零初始状态, 前 BURN 拍不计损失; Adam, 按验证集(按运行划分)早停。
验证指标在完整序列上用 numpy 推理类(部署形式, 零初始状态从头跑)计算, 前 BURN 拍不计。
基线: 零预测; 同一 psi 特征上的线性最小二乘(训练集拟合)。
d_bar(M5 的 d_hat_r 饱和) = 30 级, d_L = 0.8 d_bar = 24 级;
delta_L = 训练目标 |diff(y)|/Ts 的 99 分位数。
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from lstm_data import load
from lstm_ff import psi_offline, LSTMFeedforward

HERE = os.path.dirname(os.path.abspath(__file__))
TS = 0.02
D_BAR = 30.0
WIN, BURN = 300, 50


class Net(nn.Module):
    def __init__(self, nin, H):
        super().__init__()
        self.lstm = nn.LSTM(nin, H, 1, batch_first=True)
        self.Wout = nn.Parameter(torch.randn(1, H) * 0.3)
        self.bout = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        h, _ = self.lstm(x)
        Wbar = self.Wout / torch.linalg.matrix_norm(self.Wout, ord=2)
        return (h @ Wbar.T).squeeze(-1) + self.bout


def windows(runs, X, Y, step):
    xs, ys = [], []
    for i, r in enumerate(runs):
        n = len(Y[i])
        for s in range(0, max(1, n - WIN + 1), step):
            if s + WIN > n:
                break
            xs.append(X[i][s:s + WIN])
            ys.append(Y[i][s:s + WIN])
    return np.array(xs, np.float32), np.array(ys, np.float32)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('which', choices=('sim', 'real'))
    ap.add_argument('--H', type=int, default=None)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    H = a.H or (24 if a.which == 'sim' else 16)
    epochs = a.epochs or (60 if a.which == 'sim' else 300)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    runs = load(os.path.join(HERE, 'lstm_data_%s.npz' % a.which))
    X = [psi_offline(r['theta'], r['omega'], r['u']) for r in runs]
    Y = [r['y'] for r in runs]
    tr = [i for i, r in enumerate(runs) if not r['val']]
    va = [i for i, r in enumerate(runs) if r['val']]
    Xtr = np.concatenate([X[i] for i in tr])
    Ytr = np.concatenate([Y[i] for i in tr])
    x_mu, x_sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    y_mu, y_sd = float(Ytr.mean()), float(Ytr.std())
    delta_L = float(np.percentile(np.concatenate([np.abs(np.diff(Y[i])) for i in tr]) / TS, 99))
    d_L = 0.8 * D_BAR
    print('[%s] H=%d, train runs %d, val runs %d, y_mu %.2f y_sd %.2f, delta_L %.1f /s, d_L %.1f'
          % (a.which, H, len(tr), len(va), y_mu, y_sd, delta_L, d_L))
    nX = [(x - x_mu) / x_sd for x in X]
    nY = [(y - y_mu) / y_sd for y in Y]
    step = 50 if a.which == 'sim' else 10
    xw, yw = windows([runs[i] for i in tr], [nX[i] for i in tr], [nY[i] for i in tr], step)
    xv, yv = windows([runs[i] for i in va], [nX[i] for i in va], [nY[i] for i in va], WIN)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Net(X[0].shape[1], H).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    xw_t, yw_t = torch.tensor(xw, device=dev), torch.tensor(yw, device=dev)
    xv_t, yv_t = torch.tensor(xv, device=dev), torch.tensor(yv, device=dev)
    best, best_state = 1e9, None
    bs = 256
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xw_t), device=dev)
        for j in range(0, len(perm), bs):
            idx = perm[j:j + bs]
            out = net(xw_t[idx])
            loss = ((out[:, BURN:] - yw_t[idx, BURN:]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            vl = float(((net(xv_t)[:, BURN:] - yv_t[:, BURN:]) ** 2).mean())
        if vl < best:
            best = vl
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if ep % max(1, epochs // 10) == 0 or ep == epochs - 1:
            print('  ep %3d train %.4f val %.4f (best %.4f) %.0f s' % (ep, float(loss), vl, best,
                                                                       time.time() - t0))
    net.load_state_dict(best_state)
    sd = best_state
    Wout = sd['Wout'].numpy()
    out = os.path.join(HERE, 'lstm_yaw_%s.npz' % a.which)
    np.savez(out, Wih=sd['lstm.weight_ih_l0'].numpy().astype(np.float64),
             Whh=sd['lstm.weight_hh_l0'].numpy().astype(np.float64),
             b=(sd['lstm.bias_ih_l0'] + sd['lstm.bias_hh_l0']).numpy().astype(np.float64),
             Wout=(Wout / np.linalg.norm(Wout, 2)).astype(np.float64),
             bout=np.float64(sd['bout'].numpy()[0]),
             x_mu=x_mu, x_sd=x_sd, y_mu=y_mu, y_sd=y_sd, d_L=d_L, delta_L=delta_L,
             H=H, n_stack=4, feat='theta,omega,u_prev,sin(theta),cos(theta) x [k..k-3]',
             src=a.which)
    # ---- 验证: 完整序列, numpy 部署推理(零初始状态), 前 BURN 拍不计
    ff = LSTMFeedforward(out, TS)
    A = np.concatenate([np.concatenate([Xtr, np.ones((len(Xtr), 1))], 1)])
    coef, *_ = np.linalg.lstsq(A, Ytr, rcond=None)
    e0, eL, eLp, eLin, yy = [], [], [], [], []
    tdiff = []
    for i in va:
        r = runs[i]
        y = Y[i]
        raw = ff.raw_sequence(X[i])
        ff.reset()
        proj = np.array([ff.step(r['theta'][k], r['omega'][k], r['u'][k - 1] if k else 0.0)
                         for k in range(len(y))])
        # torch 与 numpy 一致性
        with torch.no_grad():
            tt = net(torch.tensor(nX[i][None], dtype=torch.float32, device=dev))[0].cpu().numpy()
        tdiff.append(np.max(np.abs(tt * y_sd + y_mu - raw)))
        lin = np.concatenate([X[i], np.ones((len(y), 1))], 1) @ coef
        s = slice(BURN, None)
        yy.append(y[s]); e0.append(0 * y[s]); eL.append(raw[s]); eLp.append(proj[s]); eLin.append(lin[s])
    yy, e0, eL, eLp, eLin = map(np.concatenate, (yy, e0, eL, eLp, eLin))
    rz = rmse(yy, e0)
    print('[%s] 验证 RMSE(级): 零预测 %.3f | 线性回归(同特征) %.3f | LSTM %.3f | LSTM+投影 %.3f'
          % (a.which, rz, rmse(yy, eLin), rmse(yy, eL), rmse(yy, eLp)))
    print('      相对零预测: 线性 %.1f%%  LSTM %.1f%%  LSTM+投影 %.1f%%' % (
        100 * rmse(yy, eLin) / rz, 100 * rmse(yy, eL) / rz, 100 * rmse(yy, eLp) / rz))
    print('      torch/numpy 最大差 %.2e 级; numpy 每拍 %.1f us(中位), %.1f us(p99)' % (
        max(tdiff), np.median(ff.t_us), np.percentile(ff.t_us, 99)))
    if a.which == 'sim':
        dt_ = np.concatenate([runs[i]['dr_true'][BURN:] for i in va])
        print('      (参考) 对真值 dr: 零 %.3f 线性 %.3f LSTM+投影 %.3f' % (
            rmse(dt_, 0 * dt_), rmse(dt_, eLin), rmse(dt_, eLp)))
    print('      saved', os.path.basename(out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
