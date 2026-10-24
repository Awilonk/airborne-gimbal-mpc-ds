#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""[drift 版: M5 均带 kappa=4, L=50] 残差超前预测器 lead='lstmr' 的闭环对比(以 lin 为基线), 2026-10-22 lyk。

    python bench_lead_resid.py [--jobs 14] [--out bench_lead_resid_rho_20261022.txt]

lstmr: Delta_raw = Delta_lin + eps_gain * Pi_rate(Pi_mag(eps_hat)), eps_hat = 网络(psi_k, Delta_lin(k)),
  训练目标 eps = D* - Delta_lin(train_lead_resid.py, lead_lstmr_sim.npz, fc 固定为训练时的 C-lin 值
  nom 5 Hz / cal 2 Hz); 之后与其他 lead 一样 Pi_Dmax / rate / beta / sat_30。
  网格 wo x beta {0.5,1} x eps_gain {0.5,1}。D_max/delta_D 用与 C-lin 相同的 6.0/116.0(npz meta 为 5.97/116.3),
  这样 eps_gain=0 与同参数 C-lin 逐位相同。
  消融 R0 = lstmr 调出的同一组参数下 eps_gain=0(即该参数下的 lin 超前, 不重调)。
M5-noLSTM / B-lstmd-sim / C-lin 网格与 bench_lead.py 相同(重跑, 应复现 bench_lead_20261022.txt)。
以下为 bench_lead.py 原说明:

假设: 对象输入延迟 n_d=3 拍, 闭环残差 w_k = d_r,k - d_r^c_{k-n_d}, 其中 n_d Ts delta_r 项线性 DO 去不掉。
做法: DO 仍以当前时刻估计运行(d_ff = d_L,now, 误差分析不变), 只在补偿里加超前修正
    d_r^c_k = sat_30( d_hat_r,k + beta * Pi(Delta_k) ),  |Delta| <= D_max, |dDelta/dt| <= delta_D
  A  M5+lead-LSTM2  两头网络 [d_r(k), d_r(k+3)], now 头(投影, x alpha)作 d_ff, Delta = lead - now
  B  M5+lead-LSTMd  单头网络直接预测 Delta = d_r(k+3) - d_r(k); d_ff 无, 或用现有 LSTMsim(ff=L)
  C  M5+lead-lin    不学习: Delta = 3 * LPF_fc(d_hat_r,k - d_hat_r,k-1)
网络: train_lead.py(lstm_data_sim/real, seed 0-7 训练 8-9 验证), 权重冻结。
调参 seed 0-2, f0 {0.5,1,1.5,2} x {steady, switch}, 每环境每方法一组参数(最小平均 RMSE):
  M5-noLSTM    wo in {5,10,20,28,40,56}
  M5+LSTMsim   固定为 bench_lstm_retune 的最优(nom 56/1/1, cal 5/2/0.5), 同一 seed 0-2 调出
  A            wo x beta {0.5,1} x alpha {0,1}       (alpha=0: 只有超前修正, 无学习 d_ff)
  B            wo x beta {0.5,1} x ff {无, LSTMsim(倍数1, alpha1)}
  C            wo x beta {0.5,1} x fc {2,5} Hz
  A/B 另有真机权重(-real)版本, 同网格。
测试 seed 100-104 (40 例/环境), 基准 = 重调 M5-noLSTM。
"""
import argparse
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
F0S = (0.5, 1.0, 1.5, 2.0)
SCEN = {'steady': 99.0, 'switch': 20.0}
MK = dict(r=1e-4, q_om=0, Np=20, Nc=3)
TUNE = range(0, 3)
TEST = range(100, 105)
LAM = 0.985
ND = 3
WO = (5, 10, 20, 28, 40, 56)
DL_SIM = float(np.load(os.path.join(HERE, 'lstm_yaw_sim.npz'))['delta_L'])
LSTM_BEST = {'nom': dict(wo=56, ks=1.0, al=1.0), 'cal': dict(wo=5, ks=2.0, al=0.5)}
B2 = (0.5, 1.0)
FC_R = {'nom': 5.0, 'cal': 2.0}
GRIDS = {
    'M5-noLSTM': [dict(wo=w) for w in WO],
    'B-lstmd-sim': [dict(wo=w, beta=b, ff=f) for w in WO for b in B2 for f in (0, 1)],
    'C-lin': [dict(wo=w, beta=b, fc=f) for w in WO for b in B2 for f in (2.0, 5.0)],
    'R-lstmr': [dict(wo=w, beta=b, eg=g) for w in WO for b in B2 for g in (0.5, 1.0)],
    'R0-eps0': None,
}


def key(p):
    return tuple(sorted(p.items()))


def make(meth, p, f0, box):
    from ctrl_m5 import M5Controller
    kw = dict(f_nom=f0, mpc_kw=MK, wo=p['wo'], hp_kw=dict(lam=LAM), drift=4.0, drift_L=50)
    al = 1.0
    if meth == 'M5+LSTMsim' or (meth.startswith('B') and p['ff']):
        kw.update(lstm=os.path.join(HERE, 'lstm_yaw_sim.npz'),
                  lstm_kw=dict(delta_L=p.get('ks', 1.0) * DL_SIM))
        al = p.get('al', 1.0)
    if meth.startswith('A'):
        kw.update(lead='lstm2', lead_npz=os.path.join(HERE, 'lead_lstm2_%s.npz' % meth[-4:].strip('-')),
                  lead_kw=dict(beta=p['beta'], alpha=p['alpha']))
    elif meth.startswith('B'):
        kw.update(lead='lstmd', lead_npz=os.path.join(HERE, 'lead_lstmd_%s.npz' % meth[-4:].strip('-')),
                  lead_kw=dict(beta=p['beta']))
    elif meth.startswith('C'):
        kw.update(lead='lin', lead_kw=dict(beta=p['beta'], fc=p['fc']))
    elif meth.startswith('R'):
        kw.update(lead='lstmr', lead_npz=os.path.join(HERE, 'lead_lstmr_sim.npz'),
                  lead_kw=dict(beta=p['beta'], fc=p['fc'], D_max=6.0, delta_D=116.0, eps_gain=p['eg'] if meth == 'R-lstmr' else 0.0))
    c = M5Controller('yaw', 0.02, **kw)
    if c.ff is not None and al != 1.0:
        raw = c.ff.step
        c.ff.step = lambda th, om, up: al * raw(th, om, up)
    st = c.step
    log = box.setdefault('drc', [])

    def step(*a, **k):
        u = st(*a, **k)
        log.append((c.d_r, c.d_r_c, c.lead_d))
        return u
    c.step = step
    box['c'] = c
    return c


def job(a):
    env, meth, pk, f0, scen, seed = a
    p = dict(pk)
    if meth.startswith('R'):
        p['fc'] = FC_R[env]
    from disc_pred import run
    from sim_zr10 import metrics, CAL_REAL
    box = {}
    try:
        rec = run(lambda: make(meth, p, f0, box), seed, f0, Tsw=SCEN[scen],
                  cal=CAL_REAL if env == 'cal' else None)
        m = metrics(rec)
        c = box['c']
        rm = m['rmse'] if math.isfinite(m['rmse']) else 99.0
        L = np.array(box['drc'])
        dr = rec[:, 5]
        mm = rec[:-ND, 0] >= 5.0
        fut = dr[ND:][mm]
        e_now = float(np.sqrt(np.mean((L[:-ND, 0][mm] - fut) ** 2)))     # d_hat_r,k 对 d_r,k+3
        e_cmp = float(np.sqrt(np.mean((L[:-ND, 1][mm] - fut) ** 2)))     # d_r^c_k 对 d_r,k+3
        # 实际超前量 Delta_k = d_r^c_k - d_hat_r,k 对 D*_true = d_r,k+3 - d_r,k(仿真真值)
        Dtrue = (dr[ND:] - dr[:-ND])[mm]
        e_D = float(np.sqrt(np.mean((L[:-ND, 2][mm] - Dtrue) ** 2)))
        return a, (min(rm, 99.0), m['sat'], c.n_dsat / max(1, c.n_step), e_now, e_cmp, e_D)
    except Exception as ex:
        print('ERR', a, ex)
        return a, (99.0, 1.0, 1.0, 99.0, 99.0, 99.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jobs', type=int, default=14)
    ap.add_argument('--out', default=os.path.join(HERE, 'bench_lead_resid_rho_20261022.txt'))
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit('refuse to overwrite ' + a.out)
    out = open(a.out, 'w', encoding='utf-8')

    def say(s=''):
        print(s)
        out.write(s + '\n')
        out.flush()
        sys.stdout.flush()
    say(__doc__)
    say('离线超前预测(train_lead.py 验证集, D* = y(k+3) - y(k), 级; 括号内为相对零阶保持 Delta=0):')
    for f in ('lstmd_sim', 'lstmr_sim'):
        pth = os.path.join(HERE, '_scratch', 'train_lead_%s.log' % f)
        if os.path.exists(pth):
            say('  ' + open(pth, encoding='utf-8').read().strip())
    say('  注: "线性外推(y 零相位)" 用的是非因果零相位目标本身做差分, 对 C 偏乐观; "now 头" 为因果估计上的外推。')
    say()
    t0 = time.time()
    tasks = [(env, m, key(p), f0, sc, s) for env in ('nom', 'cal') for m, g in GRIDS.items() if g
             for p in g for f0 in F0S for sc in SCEN for s in TUNE]
    with Pool(a.jobs) as pool:
        res = pool.map(job, tasks, chunksize=4)
    score = {}
    for (env, m, pk, *_), v in res:
        score.setdefault((env, m, pk), []).append(v[0])
    best = {}
    for env in ('nom', 'cal'):
        say('== [%s] 调参 seed 0-2 (24 例/组), 累计 %.0f s' % (env, time.time() - t0))
        for m, g in GRIDS.items():
            if not g:
                best[(env, m)] = best[(env, 'R-lstmr')]
                say('  %-13s 固定为 R-lstmr 的参数, eps_gain=0 (fc %g)' % (m, FC_R[env]))
                continue
            sc = sorted((np.mean(score[(env, m, key(p))]), i) for i, p in enumerate(g))
            best[(env, m)] = g[sc[0][1]]
            row = [min(np.mean(score[(env, m, key(p))]) for p in g if p['wo'] == w) for w in WO]
            say('  %-13s 最优 %-36s RMSE %.4f | 各 wo 最好: %s' % (
                m, best[(env, m)], sc[0][0], ' '.join('%g:%.4f' % (w, r) for w, r in zip(WO, row))))
    tasks = [(env, m, key(best[(env, m)]), f0, sc, s) for env in ('nom', 'cal') for m in GRIDS
             for f0 in F0S for sc in SCEN for s in TEST]
    with Pool(a.jobs) as pool:
        res = pool.map(job, tasks, chunksize=2)
    R = {(e, m, f0, sc, s): v for (e, m, pk, f0, sc, s), v in res}
    say()
    say('== 测试 seed 100-104 (40 例/环境), 累计 %.0f s' % (time.time() - t0))
    say('   相对 = 配对 RMSE 比 -1 的均值(基准 = 重调 M5-noLSTM); 胜 = RMSE 严格更小的例数; p = Wilcoxon 符号秩(双侧);')
    say('   vsC = 相对 C-lin 的配对 % 与胜场; u饱和 = |u|>=49.5 占比; dsat = d_r^c 饱和拍数占比;')
    say('   vsB = 相对 B-lstmd-sim; e_now = RMSE(d_hat_r,k - d_r,k+3), e_cmp = RMSE(d_r^c_k - d_r,k+3),')
    say('   e_D = RMSE(Delta_k - (d_r,k+3 - d_r,k)), Delta_k = 实际注入的超前量(级, t>=5 s, d_r 为仿真真值, 含 beta)')
    say('  %-4s %-13s %8s %8s %6s %8s %13s %13s %6s %6s %6s %6s %6s' % (
        'env', '方法', 'RMSE', '相对', '胜/40', 'p', 'vsC(%/胜)', 'vsB(%/胜)', 'u饱和', 'dsat', 'e_now',
        'e_cmp', 'e_D'))
    cases = [(f0, sc, s) for f0 in F0S for sc in SCEN for s in TEST]
    for env in ('nom', 'cal'):
        b = np.array([R[(env, 'M5-noLSTM') + c][0] for c in cases])
        cc = np.array([R[(env, 'C-lin') + c][0] for c in cases])
        bb_ = np.array([R[(env, 'B-lstmd-sim') + c][0] for c in cases])
        for m in GRIDS:
            v = np.array([R[(env, m) + c] for c in cases])
            x = v[:, 0]
            try:
                p = wilcoxon(x, b).pvalue if np.any(x != b) else 1.0
            except ValueError:
                p = 1.0
            say('  %-4s %-13s %8.4f %+7.1f%% %6d %8.1e %+6.1f%%/%-5d %+6.1f%%/%-5d %5.2f%% %5.2f%% %6.2f %6.2f %6.2f' % (
                env, m, x.mean(), 100 * (x / b - 1).mean(), int((x < b).sum()), p,
                100 * (x / cc - 1).mean(), int((x < cc).sum()), 100 * (x / bb_ - 1).mean(),
                int((x < bb_).sum()), 100 * v[:, 1].mean(),
                100 * v[:, 2].mean(), v[:, 3].mean(), v[:, 4].mean(), v[:, 5].mean()))
        x = np.array([R[(env, 'R-lstmr') + c][0] for c in cases])
        try:
            pB = wilcoxon(x, bb_).pvalue
        except ValueError:
            pB = 1.0
        say('  [%s] R-lstmr vs B-lstmd-sim Wilcoxon p = %.1e' % (env, pB))
        say()
    say('分场景相对重调 M5-noLSTM(%, f0=0.5/1/1.5/2):')
    for env in ('nom', 'cal'):
        for sc in SCEN:
            for m in GRIDS:
                if m == 'M5-noLSTM':
                    continue
                rr = []
                for f0 in F0S:
                    x = np.array([R[(env, m, f0, sc, s)][0] for s in TEST])
                    bb = np.array([R[(env, 'M5-noLSTM', f0, sc, s)][0] for s in TEST])
                    rr.append(100 * (x / bb - 1).mean())
                say('  [%s] %-7s %-13s ' % (env, sc, m) + ' '.join('%+6.1f' % r for r in rr))
    say('总耗时 %.0f s' % (time.time() - t0))
    return 0


if __name__ == '__main__':
    sys.exit(main())
