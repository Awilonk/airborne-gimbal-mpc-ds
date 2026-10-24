"""Train joint paper-40 residual network on traceable identified-model data.

No hardware or ROS access. Whole trajectories, never windows from one flight,
are assigned to train (seeds 0..7) or validation (8..9). Labels are explicit
posterior-minus-APDO model-based proxies, not measured physical torque truth.
"""
import argparse,hashlib,json,math,time,os
from pathlib import Path
import numpy as np
from ctrl_m5 import M5Controller
from sim_zr10 import Plant,CAL_REAL,make_dist
from paper_features import PaperFeatures

TS=.02
MK=dict(Np=20,Nc=3,r=1e-4,q_om=0)

def _generate_task(task):return make_run(*task,T=16.)

def envelope(t,T):
    t=np.asarray(t)
    return np.where(t<2,.5-.5*np.cos(np.pi*np.maximum(t,0)/2),
           np.where(t>T-2,.5-.5*np.cos(np.pi*np.maximum(T-t,0)/2),1.))

def make_run(env,seed,f0,switch,T=16.):
    N=int(T/TS);rng=np.random.default_rng(seed+331)
    centers=rng.uniform([-60.,-30.],[60.,30.])
    plants=[Plant(ax,np.random.default_rng(seed+1000+j),perturb=.1,cal=CAL_REAL if env=='cal' else None)
            for j,ax in enumerate(('yaw','pitch'))]
    ctrls=[M5Controller(ax,TS,f_nom=f0,wo=5,lead='none',mpc_kw=MK) for ax in ('yaw','pitch')]
    fs=[make_dist(np.random.default_rng(seed*19+j+71),T,f0,T/2 if switch else 1e9,8,4,2) for j in range(2)]
    for j,p in enumerate(plants):p.th=centers[j]
    feat=PaperFeatures();prev=np.zeros(2);X=[];Y=[];DL=[];U=[]
    for k in range(N):
        t=k*TS;meas=[p.meas() for p in plants]
        qa=np.array([v[0] for v in meas]);om=np.array([v[1] for v in meas])
        X.append(feat.step(qa,om,prev));y=[];dl=[];sent=[]
        for j,(c,p) in enumerate(zip(ctrls,plants)):
            amp,w=(3.,.8) if j==0 else (2.,.6)
            fut=t+TS*np.arange(1,21)
            ref=lambda tt:envelope(tt,T)*amp*np.sin(w*tt)
            R=ref(fut);dR=(ref(fut+.001)-ref(fut-.001))/.002
            request=c.step(qa[j]-centers[j],om[j],R,dR,t=t)
            uq=int(round(request));sent.append(uq)
            y.append(c.dplus-c.d_h_raw);dl.append(c.lead_dlin)
            c.commit_sent(uq)
            p.cmd(t+.002,uq)
            dh,dr=fs[j];p.run(t,t+TS,lambda tt,ww:dh(tt)+dr(tt,ww))
            if abs(p.th-centers[j])>12:raise RuntimeError('bootstrap simulation diverged')
        prev=np.array(sent);U.append(sent);Y.append(y);DL.append(dl)
    X=np.array(X);y=np.array(Y);dl=np.array(DL);nd=3
    target=y[nd:]-y[:-nd]-dl[:-nd]
    return dict(X=X[:-nd],Y=target,U=np.array(U)[:-nd],
                name=f'{env}_s{seed}_f{f0}_{"switch" if switch else "steady"}',
                seed=seed,val=seed>=8,env=env,center=centers)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--epochs',type=int,default=45)
    ap.add_argument('--pilot',action='store_true');a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    start=time.time();runs=[]
    tasks=[(e,s,f,sw) for e in ('nom','cal') for s in range(10) for f in (.5,1.,1.5,2.) for sw in (False,True)]
    if a.pilot:tasks=[('cal',100,.5,False)]
    if a.pilot:
        runs=[make_run(*tasks[0],T=6.)]
    else:
        from concurrent.futures import ProcessPoolExecutor
        for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
        with ProcessPoolExecutor(max_workers=4) as pool:
            for n,r in enumerate(pool.map(_generate_task,tasks)):
                runs.append(r)
                if n%10==0:print('DATA',n+1,len(tasks),'elapsed_s',round(time.time()-start,1),flush=True)
    flat={}
    for i,r in enumerate(runs):
        for k,v in r.items():flat[f'{i}/{k}']=np.asarray(v)
    flat['n']=np.array(len(runs));np.savez_compressed(out/'paper40_training_data.npz',**flat)
    if a.pilot:
        print('PILOT finite',all(np.isfinite(r['X']).all() and np.isfinite(r['Y']).all() for r in runs),flush=True)
        return
    import torch
    import torch.nn as nn
    torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
    class Net(nn.Module):
        def __init__(self):
            super().__init__();self.lstm=nn.LSTM(40,24,batch_first=True)
            self.Wout=nn.Parameter(torch.randn(2,24)*.2);self.bout=nn.Parameter(torch.zeros(2))
        def forward(self,x):
            h,_=self.lstm(x);W=self.Wout/torch.linalg.matrix_norm(self.Wout,ord=2)
            return h@W.T+self.bout
    tr=[r for r in runs if not r['val']];va=[r for r in runs if r['val']]
    Xt=np.concatenate([r['X'][150:] for r in tr]);Yt=np.concatenate([r['Y'][150:] for r in tr])
    mu=Xt.mean(0);sd=np.maximum(Xt.std(0),np.tile([.1,.1,.1,.1,.5,.5,.01,.01,.01,.01],4))
    ym=Yt.mean(0);ys=np.maximum(Yt.std(0),1e-3)
    def windows(rs):
        x,y=[],[]
        for r in rs:
            for j in range(0,len(r['Y'])-128+1,128):
                x.append((r['X'][j:j+128]-mu)/sd);y.append((r['Y'][j:j+128]-ym)/ys)
        return torch.tensor(np.array(x),dtype=torch.float32),torch.tensor(np.array(y),dtype=torch.float32)
    x,y=windows(tr);xv,yv=windows(va)
    net=Net();opt=torch.optim.Adam(net.parameters(),lr=.003)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,a.epochs)
    best=float('inf');state=None;history=[]
    for ep in range(a.epochs):
        net.train();perm=torch.randperm(len(x))
        for start_idx in range(0,len(x),128):
            idx=perm[start_idx:start_idx+128];pred=net(x[idx]);loss=((pred[:,16:]-y[idx,16:])**2).mean()
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
        sched.step();net.eval()
        with torch.no_grad():vl=float(((net(xv)[:,16:]-yv[:,16:])**2).mean())
        history.append(dict(epoch=ep,train=float(loss.detach()),validation=vl))
        if vl<best:best=vl;state={k:v.detach().clone() for k,v in net.state_dict().items()}
        if ep%5==0 or ep==a.epochs-1:print('TRAIN',ep,'val',round(vl,5),'elapsed_s',round(time.time()-start,1),flush=True)
    W=state['Wout'].numpy().astype(float);W/=np.linalg.norm(W,2)
    em=np.minimum(9.,np.maximum(.5,2*np.quantile(abs(Yt),.99,axis=0)))
    diffs=np.concatenate([abs(np.diff(r['Y'][150:],axis=0))/TS for r in tr])
    er=np.maximum(10.,2*np.quantile(diffs,.99,axis=0))
    weights=out/'paper40_lead_sim.npz'
    np.savez(weights,Wih=state['lstm.weight_ih_l0'].numpy().astype(float),
             Whh=state['lstm.weight_hh_l0'].numpy().astype(float),
             b=(state['lstm.bias_ih_l0']+state['lstm.bias_hh_l0']).numpy().astype(float),
             Wout=W,bout=state['bout'].numpy().astype(float),x_mu=mu,x_sd=sd,y_mu=ym,y_sd=ys,
             kind='paper40_residual',eps_max=em,eps_rate=er,D_max=np.array([6.,6.]),
             source='identified_model_simulation',feature_units='deg,deg/s,integer-levels; trigonometry radians',
             train_seeds=np.arange(8),validation_seeds=np.array([8,9]))
    from paper_network import PaperNetwork
    pred_all=[];target_all=[];consistency=0.
    net64=Net().double();net64.load_state_dict({k:v.double() for k,v in state.items()})
    for i,r in enumerate(va):
        ff=PaperNetwork(weights);pn=np.array([ff.step_psi(xx) for xx in r['X']])
        pred_all.append(pn[150:]);target_all.append(r['Y'][150:])
        if i<2:
            with torch.no_grad():pt=net64(torch.tensor(((r['X']-mu)/sd)[None],dtype=torch.float64))[0].numpy()*ys+ym
            consistency=max(consistency,float(np.max(abs(pn-pt))))
    yy=np.concatenate(target_all);pp=np.concatenate(pred_all)
    report=dict(trajectories=len(runs),train_trajectories=len(tr),validation_trajectories=len(va),
        input_dim=40,output_dim=2,hidden_dim=24,train_seeds=list(range(8)),validation_seeds=[8,9],
        label='posterior-minus-unweighted-APDO proxy: y[k+3]-y[k]-Delta_lin[k]; no physical torque measurement',
        simulation_model='independent identified yaw/pitch speed inner loops with mismatch and calibrated nonidealities',
        validation_rmse=np.sqrt(np.mean((yy-pp)**2,axis=0)).tolist(),
        zero_residual_rmse=np.sqrt(np.mean(yy**2,axis=0)).tolist(),
        numpy_torch_max_difference=consistency,spectral_norm=float(np.linalg.norm(W,2)),
        epochs=a.epochs,history=history,elapsed_s=time.time()-start,
        dataset_sha256=hashlib.file_digest((out/'paper40_training_data.npz').open('rb'),'sha256').hexdigest(),
        weight_sha256=hashlib.file_digest(weights.open('rb'),'sha256').hexdigest(),
        limitation='New training for structural alignment; not a reproduction of manuscript numerical results or flight-domain validation.')
    (out/'training_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('DONE',json.dumps({k:v for k,v in report.items() if k!='history'},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
