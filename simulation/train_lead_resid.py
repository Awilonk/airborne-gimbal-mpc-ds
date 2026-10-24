#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""训练残差超前网络 lead='lstmr'(以 lin 超前为基线), 导出 lead_lstmr_sim.npz, 2026-10-22 lyk。

    python train_lead_resid.py            -> lead_lstmr_sim.npz, _scratch/train_lead_lstmr_sim.log

数据: lstm_data_resid_sim.npz(lstm_data_resid.py; 与 lstm_data_sim.npz 同一闭环逐位相同, 加 u_f/dlin 列; seed 0-7 训练, 8-9 验证, 同一 torch/numpy seed 0, 同一网络/窗口/优化器)。
Delta_lin(k): 在记录的 (omega, u_f 浮点指令) 上重放 M5 的因果估计链(lead_resid.replay_lin), 参数按环境取 C-lin 调参值
  (nom wo40 fc5, cal wo5 fc2), f_nom = 该运行的 f0。
目标 eps(k) = D*(k) - Delta_lin(k), D* = y(k+3) - y(k)。输入 = psi_k(20) + Delta_lin(k)。
meta: eps_max = 2 x |eps| 99 分位, eps_rate = 2 x |diff eps|/Ts 99 分位; D_max/delta_D 与 lstmd 相同定义(D*)。
离线表(验证集, t>=BURN 拍): 零阶保持(Delta=0)、lin(Delta_lin, Pi_Dmax)、lstmd(lead_lstmd_sim.npz, Pi_Dmax)、
lstmr(Delta_lin + Pi_eps(eps_hat), Pi_Dmax) 对 D* 的 RMSE。一致性: torch CPU float64 vs numpy 推理。
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn

from lstm_data import load, ND
from lstm_ff import psi_offline
from lead_ff import LeadFeedforward
from lead_resid import LeadResidFF, replay_lin, C_BEST
from train_lead import Net, windows, rmse, TS, BURN

HERE = os.path.dirname(os.path.abspath(__file__))


def _f0(name):
    return float(name.split('_f')[1].split('_')[0])


def _job(a):
    om, u, name = a
    p = C_BEST[name.split('_')[0]]
    return replay_lin(om, u, _f0(name), p['wo'], p['fc'])


def main():
    out = os.path.join(HERE, 'lead_lstmr_sim.npz')
    if os.path.exists(out):
        raise SystemExit('refuse to overwrite ' + out)
    H, epochs = 24, 60
    torch.manual_seed(0)
    np.random.seed(0)
    runs = load(os.path.join(HERE, 'lstm_data_resid_sim.npz'))
    t0 = time.time()
    DL = [r['dlin'] for r in runs]
    X, Y, DS, LN = [], [], [], []
    for r, dl in zip(runs, DL):
        y = r['y']
        D = y[ND:] - y[:-ND]
        dl = dl[:-ND]
        X.append(np.concatenate([psi_offline(r['theta'], r['omega'], r['u'])[:-ND], dl[:, None]], 1))
        Y.append((D - dl)[:, None])
        DS.append(D)
        LN.append(dl)
    tr = [i for i, r in enumerate(runs) if not r['val']]
    va = [i for i, r in enumerate(runs) if r['val']]
    Xtr = np.concatenate([X[i] for i in tr])
    Ytr = np.concatenate([Y[i] for i in tr])
    x_mu, x_sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    y_mu, y_sd = Ytr.mean(0), Ytr.std(0)
    Dtr = np.concatenate([DS[i] for i in tr])
    Etr = [Y[i][:, 0] for i in tr]
    meta = dict(d_L=24.0,
                delta_L=float(np.percentile(np.concatenate(
                    [np.abs(np.diff(runs[i]['y'])) for i in tr]) / TS, 99)),
                D_max=2 * float(np.percentile(np.abs(Dtr), 99)),
                delta_D=2 * float(np.percentile(np.abs(np.diff(Dtr)) / TS, 99)),
                eps_max=2 * float(np.percentile(np.abs(np.concatenate(Etr)), 99)),
                eps_rate=2 * float(np.percentile(np.concatenate([np.abs(np.diff(e)) for e in Etr]) / TS, 99)))
    print('[sim/lstmr] H=%d train %d val %d; y_mu %s y_sd %s; std(D*) %.3f std(eps) %.3f; %s' % (
        H, len(tr), len(va), np.round(y_mu, 3), np.round(y_sd, 3), Dtr.std(), np.concatenate(Etr).std(),
        ', '.join('%s %.3f' % kv for kv in meta.items())))
    nX = [(x - x_mu) / x_sd for x in X]
    nY = [(y - y_mu) / y_sd for y in Y]
    xw, yw = windows([nX[i] for i in tr], [nY[i] for i in tr], 50)
    xv, yv = windows([nX[i] for i in va], [nY[i] for i in va], 300)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = Net(X[0].shape[1], H, 1).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    xw_t, yw_t = torch.tensor(xw, device=dev), torch.tensor(yw, device=dev)
    xv_t, yv_t = torch.tensor(xv, device=dev), torch.tensor(yv, device=dev)
    loss_fn = lambda o, y: ((o[:, BURN:] - y[:, BURN:]) ** 2).mean()
    best, best_state = 1e9, None
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
        if ep % 10 == 0 or ep == epochs - 1:
            print('  ep %3d train %.4f val %.4f (best %.4f) %.0f s' % (ep, float(loss), vl, best, time.time() - t0))
    sd = best_state
    W = sd['Wout'].numpy().astype(np.float64)
    np.savez(out, Wih=sd['lstm.weight_ih_l0'].numpy().astype(np.float64),
             Whh=sd['lstm.weight_hh_l0'].numpy().astype(np.float64),
             b=(sd['lstm.bias_ih_l0'] + sd['lstm.bias_hh_l0']).numpy().astype(np.float64),
             Wout=W / np.linalg.norm(W, 2), bout=sd['bout'].numpy().astype(np.float64),
             x_mu=x_mu, x_sd=x_sd, y_mu=y_mu, y_sd=y_sd, kind='lstmr', nd=ND, H=H, src='sim',
             base='lin', c_best=str(C_BEST), **meta)
    # 一致性: torch CPU float64 vs numpy
    net64 = Net(X[0].shape[1], H, 1).double()
    net64.load_state_dict({k: v.double() for k, v in sd.items()})
    ff = LeadResidFF(out)
    dmax = 0.0
    for i in va[:6]:
        with torch.no_grad():
            o_t = net64(torch.tensor(nX[i][None], dtype=torch.float64))[0].numpy() * y_sd + y_mu
        o_n = ff.raw_sequence(X[i])
        dmax = max(dmax, float(np.abs(o_t - o_n).max()))
    print('  一致性 torch(CPU f64) vs numpy, 6 条验证序列 max|diff| = %.2e' % dmax)
    # 离线表
    ffd = LeadFeedforward(os.path.join(HERE, 'lead_lstmd_sim.npz'))
    Dm = ffd.meta['D_max']
    em = meta['eps_max']
    acc = {k: [] for k in ('D', 'lin', 'linc', 'lstmd', 'lstmr', 'lstmr_noclip')}
    for i in va:
        s = slice(BURN, None)
        D = DS[i]
        dl = LN[i]
        rd = ffd.raw_sequence(X[i][:, :20])[:, 0]
        ep_ = ff.raw_sequence(X[i])[:, 0]
        acc['D'].append(D[s])
        acc['lin'].append(dl[s])
        acc['linc'].append(np.clip(dl, -Dm, Dm)[s])
        acc['lstmd'].append(np.clip(rd, -Dm, Dm)[s])
        acc['lstmr'].append(np.clip(dl + np.clip(ep_, -em, em), -Dm, Dm)[s])
        acc['lstmr_noclip'].append((dl + ep_)[s])
    A = {k: np.concatenate(v) for k, v in acc.items()}
    r0 = rmse(A['D'], 0 * A['D'])
    pc = lambda k: '%.3f (%.0f%%)' % (rmse(A['D'], A[k]), 100 * rmse(A['D'], A[k]) / r0)
    msg = ('[sim/lstmr] 验证 D*=y(k+3)-y(k) RMSE(级, %d 条, 因果): 零阶保持 %.3f | lin %s (Pi_Dmax 后 %s) | '
           'lstmd %s | lstmr %s (不投影 %s) | 均按 Pi_Dmax=%.2f, eps_max=%.2f') % (
        len(va), r0, pc('lin'), pc('linc'), pc('lstmd'), pc('lstmr'), pc('lstmr_noclip'), Dm, em)
    msg += '\n  分环境: ' + ' | '.join(
        '%s: ZOH %.3f lin %.3f lstmd %.3f lstmr %.3f' % ((e,) + tuple(
            rmse(np.concatenate([acc['D'][j] for j, i in enumerate(va) if runs[i]['name'].startswith(e)]),
                 np.concatenate([acc[k][j] for j, i in enumerate(va) if runs[i]['name'].startswith(e)])
                 if k != 'zoh' else 0) for k in ('zoh', 'linc', 'lstmd', 'lstmr')))
        for e in ('nom', 'cal'))
    msg += '\n  一致性 torch(CPU float64) vs numpy max|diff| = %.2e' % dmax
    print(msg)
    with open(os.path.join(HERE, '_scratch', 'train_lead_lstmr_sim.log'), 'w', encoding='utf-8') as f:
        f.write(msg + '\n')
    print('  saved', os.path.basename(out))


if __name__ == '__main__':
    sys.exit(main())
