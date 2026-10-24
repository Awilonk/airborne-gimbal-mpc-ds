#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练超前扰动预测网络(torch), 导出 numpy 推理 .npz(lead_ff.py), 2026-10-22 lyk。

    python train_lead.py sim  lstm2   -> lead_lstm2_sim.npz   输出 [y_k, y_{k+nd}]
    python train_lead.py sim  lstmd   -> lead_lstmd_sim.npz   输出 [y_{k+nd} - y_k]
    python train_lead.py real lstm2|lstmd -> lead_*_real.npz
数据: lstm_data_*.npz(与 train_lstm 相同, 目标 y = 零相位非周期扰动), nd = 3。
特征 psi_k 与 lstm_ff 相同(只到 u_{k-1}); 目标右移 nd 拍, 每条序列丢掉最后 nd 拍。
两头网络的损失 = 两头 MSE + 差额 (lead - now) 的 MSE(按 D* 标准差归一), 否则差额被幅值项淹没。
投影常数(训练集): d_L = 24, delta_L = |diff y|/Ts 99 分位; D_max = 2 x |D*| 99 分位,
delta_D = 2 x |diff D*|/Ts 99 分位, D* = y_{k+nd} - y_k。
离线验证(按运行划分, 前 BURN 拍不计): D* 的 RMSE, 对比 Delta=0(零阶保持) 与线性外推
n_d (yhat_k - yhat_{k-1}) 一阶低通(fc 2/5 Hz); yhat 取 y 本身(零相位, 对外推偏乐观)或 now 头(因果)。
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from lstm_data import load, ND
from lstm_ff import psi_offline
from lead_ff import LeadFeedforward

HERE = os.path.dirname(os.path.abspath(__file__))
TS = 0.02
WIN, BURN = 300, 50


class Net(nn.Module):
    def __init__(self, nin, H, m):
        super().__init__()
        self.lstm = nn.LSTM(nin, H, 1, batch_first=True)
        self.Wout = nn.Parameter(torch.randn(m, H) * 0.3)
        self.bout = nn.Parameter(torch.zeros(m))

    def forward(self, x):
        h, _ = self.lstm(x)
        Wbar = self.Wout / torch.linalg.matrix_norm(self.Wout, ord=2)
        return h @ Wbar.T + self.bout


def windows(X, Y, step):
    xs, ys = [], []
    for x, y in zip(X, Y):
        for s in range(0, len(y) - WIN + 1, step):
            xs.append(x[s:s + WIN])
            ys.append(y[s:s + WIN])
    return np.array(xs, np.float32), np.array(ys, np.float32)


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def lin_extrap(yh, fc, nd=ND, ts=TS):
    a = math.exp(-2 * math.pi * fc * ts)
    out = np.zeros_like(yh)
    s = 0.0
    for k in range(1, len(yh)):
        s = a * s + (1 - a) * (yh[k] - yh[k - 1])
        out[k] = nd * s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('which', choices=('sim', 'real'))
    ap.add_argument('kind', choices=('lstm2', 'lstmd'))
    ap.add_argument('--H', type=int, default=None)
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    H = a.H or (24 if a.which == 'sim' else 16)
    epochs = a.epochs or (60 if a.which == 'sim' else 300)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    runs = load(os.path.join(HERE, 'lstm_data_%s.npz' % a.which))
    X, Y, YN = [], [], []
    for r in runs:
        y = r['y']
        X.append(psi_offline(r['theta'], r['omega'], r['u'])[:-ND])
        YN.append(y[:-ND])
        Y.append(np.stack([y[:-ND], y[ND:]], 1) if a.kind == 'lstm2' else (y[ND:] - y[:-ND])[:, None])
    tr = [i for i, r in enumerate(runs) if not r['val']]
    va = [i for i, r in enumerate(runs) if r['val']]
    Xtr = np.concatenate([X[i] for i in tr])
    Ytr = np.concatenate([Y[i] for i in tr])
    x_mu, x_sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    y_mu, y_sd = Ytr.mean(0), Ytr.std(0)
    Dtr = np.concatenate([runs[i]['y'][ND:] - runs[i]['y'][:-ND] for i in tr])
    meta = dict(d_L=24.0,
                delta_L=float(np.percentile(np.concatenate(
                    [np.abs(np.diff(runs[i]['y'])) for i in tr]) / TS, 99)),
                D_max=2 * float(np.percentile(np.abs(Dtr), 99)),
                delta_D=2 * float(np.percentile(np.abs(np.diff(Dtr)) / TS, 99)))
    print('[%s/%s] H=%d train %d val %d runs; y_mu %s y_sd %s; %s' % (
        a.which, a.kind, H, len(tr), len(va), np.round(y_mu, 2), np.round(y_sd, 2),
        ', '.join('%s %.2f' % kv for kv in meta.items())))
    nX = [(x - x_mu) / x_sd for x in X]
    nY = [(y - y_mu) / y_sd for y in Y]
    step = 50 if a.which == 'sim' else 10
    xw, yw = windows([nX[i] for i in tr], [nY[i] for i in tr], step)
    xv, yv = windows([nX[i] for i in va], [nY[i] for i in va], WIN)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    m = Y[0].shape[1]
    net = Net(X[0].shape[1], H, m).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    xw_t, yw_t = torch.tensor(xw, device=dev), torch.tensor(yw, device=dev)
    xv_t, yv_t = torch.tensor(xv, device=dev), torch.tensor(yv, device=dev)
    sdt = torch.tensor(y_sd, device=dev, dtype=torch.float32)
    sD = float(np.std(Dtr))

    def loss_fn(o, y):
        l = ((o[:, BURN:] - y[:, BURN:]) ** 2).mean()
        if m == 2:
            po = o[:, BURN:, 1] * sdt[1] - o[:, BURN:, 0] * sdt[0]
            py = y[:, BURN:, 1] * sdt[1] - y[:, BURN:, 0] * sdt[0]
            l = l + (((po - py) / sD) ** 2).mean()
        return l
    best, best_state = 1e9, None
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xw_t), device=dev)
        for j in range(0, len(perm), 256):
            idx = perm[j:j + 256]
            loss = loss_fn(net(xw_t[idx]), yw_t[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            vl = float(loss_fn(net(xv_t), yv_t))
        if vl < best:
            best = vl
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if ep % max(1, epochs // 6) == 0 or ep == epochs - 1:
            print('  ep %3d train %.4f val %.4f (best %.4f) %.0f s' % (
                ep, float(loss), vl, best, time.time() - t0))
    sd = best_state
    W = sd['Wout'].numpy().astype(np.float64)
    out = os.path.join(HERE, 'lead_%s_%s.npz' % (a.kind, a.which))
    if os.path.exists(out):
        raise SystemExit('refuse to overwrite ' + out)
    np.savez(out, Wih=sd['lstm.weight_ih_l0'].numpy().astype(np.float64),
             Whh=sd['lstm.weight_hh_l0'].numpy().astype(np.float64),
             b=(sd['lstm.bias_ih_l0'] + sd['lstm.bias_hh_l0']).numpy().astype(np.float64),
             Wout=W / np.linalg.norm(W, 2), bout=sd['bout'].numpy().astype(np.float64),
             x_mu=x_mu, x_sd=x_sd, y_mu=y_mu, y_sd=y_sd, kind=a.kind, nd=ND, H=H, src=a.which,
             **meta)
    ff = LeadFeedforward(out)
    Dt, DL, C2, C5, C2n, C5n, yn, rn = [], [], [], [], [], [], [], []
    for i in va:
        y = runs[i]['y']
        D = y[ND:] - y[:-ND]
        raw = ff.raw_sequence(X[i])
        dl = raw[:, 1] - raw[:, 0] if a.kind == 'lstm2' else raw[:, 0]
        dl = np.clip(dl, -meta['D_max'], meta['D_max'])
        s = slice(BURN, None)
        Dt.append(D[s])
        DL.append(dl[s])
        C2.append(lin_extrap(YN[i], 2.0)[s])
        C5.append(lin_extrap(YN[i], 5.0)[s])
        if a.kind == 'lstm2':
            C2n.append(lin_extrap(raw[:, 0], 2.0)[s])
            C5n.append(lin_extrap(raw[:, 0], 5.0)[s])
            yn.append(YN[i][s])
            rn.append(raw[s, 0])
    Dt = np.concatenate(Dt)
    r0 = rmse(Dt, 0 * Dt)

    def pc(v):
        e = rmse(Dt, np.concatenate(v))
        return '%.3f (%.0f%%)' % (e, 100 * e / r0)
    msg = ('[%s/%s] 验证 D*=y(k+%d)-y(k) RMSE(级): 零阶保持 %.3f | LSTM %s | 线性外推(y 零相位, fc2) %s'
           ' | (y, fc5) %s') % (a.which, a.kind, ND, r0, pc(DL), pc(C2), pc(C5))
    if C2n:
        yn, rn = np.concatenate(yn), np.concatenate(rn)
        msg += ' | 线性外推(now 头, fc2) %s, fc5 %s | now 头 RMSE %.3f (零预测 %.3f)' % (
            pc(C2n), pc(C5n), rmse(yn, rn), rmse(yn, 0 * yn))
    print(msg)
    with open(os.path.join(HERE, '_scratch', 'train_lead_%s_%s.log' % (a.kind, a.which)), 'w',
              encoding='utf-8') as f:
        f.write(msg + '\n')
    print('  saved', os.path.basename(out))


if __name__ == '__main__':
    sys.exit(main())
