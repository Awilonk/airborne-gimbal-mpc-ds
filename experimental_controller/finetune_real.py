"""Real dual-axis data adaptation. Whole acquisition runs are held out."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from identify_real import load_trial
from paper_features import PaperFeatures
from paper_network import PaperNetwork
from ctrl_m5 import M5Controller
from zr10_ctrl import AXIS_ID
from training_data_guard import validate_acquisition,validate_split,validate_output

def replay(path,ident,segment='train'):
    meta=validate_acquisition(path,segment)
    data=load_trial(path);features=PaperFeatures();prev=np.zeros(2)
    limit=float(meta['level_limit'])
    ctrls=[M5Controller(ax,.02,wo=5,lead='none',actual_limit=limit,u_max=max(50.,limit),
        mpc_kw=dict(Np=20,Nc=3,r=1e-4,q_om=0)) for ax in ('yaw','pitch')]
    X=[];Y=[];DL=[];q0=data['q'][0]
    for k in range(len(data['t'])):
        q=data['q'][k];u=data['u'][k]
        w=np.array([data['raw'][k,j]*ident[ax]['rate_calibration']['scale']+ident[ax]['rate_calibration']['bias'] for j,ax in enumerate(('yaw','pitch'))])
        X.append(features.step(q,w,prev));yy=[];dd=[]
        for j,c in enumerate(ctrls):
            c.step(q[j]-q0[j],w[j],np.zeros(20),np.zeros(20),t=k*.02)
            yy.append(c.dplus-c.d_h_raw);dd.append(c.lead_dlin)
            c.commit_sent(u[j])
        Y.append(yy);DL.append(dd);prev=u
    nd=[c.model.nd for c in ctrls];n=len(X)-max(nd)
    y=np.array(Y);dl=np.array(DL)
    target=np.stack([y[nd[j]:nd[j]+n,j]-y[:n,j]-dl[:n,j] for j in range(2)],1)
    return dict(X=np.array(X)[:n],Y=target,source=str(path),delay_samples=nd)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',required=True);ap.add_argument('--ident',required=True)
    ap.add_argument('--train',nargs='+',required=True);ap.add_argument('--validation',required=True);ap.add_argument('--out',required=True)
    a=ap.parse_args();out=Path(a.out)
    validate_split(a.train,a.validation)
    validate_output(a.base,out,Path(__file__).parent/'paper40_lead_real_finetuned.npz')
    for p in a.train:validate_acquisition(p,'train')
    validate_acquisition(a.validation,'validation')
    out.mkdir(parents=True,exist_ok=True);t0=time.time()
    ident=json.loads(Path(a.ident).read_text(encoding='utf-8'))
    for ax in ('yaw','pitch'):
        AXIS_ID[ax].update({k:ident[ax]['model'][k] for k in ('K','tau','Td')})
    train=[replay(p,ident,'train') for p in a.train];val=replay(a.validation,ident,'validation')
    flat={}
    for i,r in enumerate(train+[val]):
        for k,v in r.items():flat[f'{i}/{k}']=np.asarray(v)
    np.savez_compressed(out/'real_training_arrays.npz',**flat)
    z=np.load(a.base);H=z['Whh'].shape[1];mu,sd,ym,ys=[z[k] for k in ('x_mu','x_sd','y_mu','y_sd')]
    torch.set_num_threads(4);torch.manual_seed(23)
    class Net(nn.Module):
        def __init__(self):
            super().__init__();self.lstm=nn.LSTM(40,H,batch_first=True)
            self.Wout=nn.Parameter(torch.tensor(z['Wout'],dtype=torch.float32));self.bout=nn.Parameter(torch.tensor(z['bout'],dtype=torch.float32))
            with torch.no_grad():
                self.lstm.weight_ih_l0.copy_(torch.tensor(z['Wih']));self.lstm.weight_hh_l0.copy_(torch.tensor(z['Whh']))
                self.lstm.bias_ih_l0.copy_(torch.tensor(z['b']));self.lstm.bias_hh_l0.zero_()
        def forward(self,x):
            h,_=self.lstm(x);return h@(self.Wout/torch.linalg.matrix_norm(self.Wout,ord=2)).T+self.bout
    def windows(rs):
        xs=[];ys_=[]
        for r in rs:
            for st in range(100,len(r['Y'])-128+1,64):
                xs.append((r['X'][st:st+128]-mu)/sd);ys_.append((r['Y'][st:st+128]-ym)/ys)
        return torch.tensor(np.array(xs),dtype=torch.float32),torch.tensor(np.array(ys_),dtype=torch.float32)
    x,y=windows(train);xv,yv=windows([val]);net=Net();opt=torch.optim.Adam(net.parameters(),lr=.001)
    best=float('inf');state=None;history=[]
    for ep in range(100):
        net.train();perm=torch.randperm(len(x))
        losses=[]
        for st in range(0,len(x),32):
            ii=perm[st:st+32];loss=((net(x[ii])[:,16:]-y[ii,16:])**2).mean()
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
            losses.append(float(loss.detach()))
        net.eval()
        with torch.no_grad():vl=float(((net(xv)[:,16:]-yv[:,16:])**2).mean())
        history.append(dict(epoch=ep+1,train=float(np.mean(losses)),validation=vl))
        if vl<best:best=vl;state={k:v.detach().clone() for k,v in net.state_dict().items()}
    W=state['Wout'].numpy().astype(float);W/=np.linalg.norm(W,2)
    Ytr=np.concatenate([r['Y'][150:] for r in train]);diff=np.concatenate([abs(np.diff(r['Y'][150:],axis=0))/.02 for r in train])
    # Conservative projected bounds for the first hardware validation; not fitted on its outcome.
    em=np.minimum(2.,np.maximum(.5,2*np.quantile(abs(Ytr),.99,axis=0)))
    er=np.minimum(80.,np.maximum(10.,2*np.quantile(diff,.99,axis=0)))
    weights=out/'paper40_lead_real_finetuned.npz'
    np.savez(weights,Wih=state['lstm.weight_ih_l0'].numpy().astype(float),Whh=state['lstm.weight_hh_l0'].numpy().astype(float),
        b=(state['lstm.bias_ih_l0']+state['lstm.bias_hh_l0']).numpy().astype(float),Wout=W,bout=state['bout'].numpy().astype(float),
        x_mu=mu,x_sd=sd,y_mu=ym,y_sd=ys,kind='paper40_residual',eps_max=em,eps_rate=er,D_max=[6.,6.],
        source='simulation_pretraining_then_real_ground_finetuning',delay_samples=val['delay_samples'])
    p=PaperNetwork(weights);pred=np.array([p.step_psi(xx) for xx in val['X']]);s=slice(150,None)
    before=PaperNetwork(a.base);base=np.array([before.step_psi(xx) for xx in val['X']])
    report=dict(train_sources=a.train,validation_source=a.validation,feature_dim=40,outputs=2,
        validation_role='held out from gradients; used for early stopping; final Proposed ground run is a separate test',
        label='model-based posterior-minus-APDO proxy, future increment minus causal linear lead; not independent torque truth',
        delay_samples=val['delay_samples'],epochs=100,best_validation_loss=best,
        baseline_pretrained_rmse=np.sqrt(np.mean((base[s]-val['Y'][s])**2,axis=0)).tolist(),
        finetuned_rmse=np.sqrt(np.mean((pred[s]-val['Y'][s])**2,axis=0)).tolist(),
        zero_residual_rmse=np.sqrt(np.mean(val['Y'][s]**2,axis=0)).tolist(),
        eps_max=em.tolist(),eps_rate=er.tolist(),spectral_norm=float(np.linalg.norm(W,2)),
        pretrained_sha256=hashlib.sha256(Path(a.base).read_bytes()).hexdigest(),
        final_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),elapsed_s=time.time()-t0,history=history)
    (out/'real_training_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
