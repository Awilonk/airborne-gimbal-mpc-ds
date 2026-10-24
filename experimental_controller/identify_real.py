"""Identify both speed interfaces from saved, real, disarmed chirp records."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.signal import lfilter
from durable_log import recover_records
from training_data_guard import validate_acquisition,validate_split

TS=.02

def load_trial(path):
    rows=recover_records(path)
    if not rows:raise ValueError('empty trial')
    t=np.array([r['t'] for r in rows]);grid=np.arange(t[0],t[-1],TS)
    ix=np.clip(np.searchsorted(t,grid,side='right')-1,0,len(t)-1)
    q=np.degrees(np.unwrap(np.radians(np.array([r['cmd_frame'][:2] for r in rows])),axis=0))
    raw=np.array([r['rate_raw'][:2] for r in rows],float)
    u=np.array([[r['u_yaw'],r['u_pitch']] for r in rows],float)
    return dict(t=grid,q=np.stack([np.interp(grid,t,q[:,j]) for j in range(2)],1),
                raw=np.stack([np.interp(grid,t,raw[:,j]) for j in range(2)],1),u=u[ix],
                source=str(path),n_original=len(rows),max_gap_s=float(np.max(np.diff(t))))

def calibrate_rate(data,j):
    if isinstance(data,list):
        X=[];Y=[]
        for d in data:
            t,q,raw=d['t'],d['q'][:,j],d['raw'][:,j]
            integ=np.r_[0,np.cumsum((raw[:-1]+raw[1:])*.5*TS)]
            for lag in (10,20,40):
                X.append(np.stack([integ[lag:]-integ[:-lag],t[lag:]-t[:-lag]],1));Y.append(q[lag:]-q[:-lag])
        X=np.concatenate(X);Y=np.concatenate(Y);beta=np.linalg.lstsq(X,Y,rcond=None)[0]
        r2=1-float(np.sum((X@beta-Y)**2)/np.sum((Y-Y.mean())**2))
        if not .03<abs(beta[0])<.3 or r2<.6:raise RuntimeError('rate calibration unreliable')
        return dict(scale=float(beta[0]),bias=float(beta[1]),integrated_angle_r2=r2,
                    method='angle increments vs integrated raw rate + bias, using training trajectories only')
    t,q,raw=data['t'],data['q'][:,j],data['raw'][:,j]
    integral=np.r_[0,np.cumsum((raw[:-1]+raw[1:])*.5*TS)]
    X=[];Y=[]
    for lag in (10,20,40):
        X.append(np.stack([integral[lag:]-integral[:-lag],t[lag:]-t[:-lag]],1))
        Y.append(q[lag:]-q[:-lag])
    X=np.concatenate(X);Y=np.concatenate(Y)
    beta=np.linalg.lstsq(X,Y,rcond=None)[0]
    r2=1-float(np.sum((X@beta-Y)**2)/np.sum((Y-Y.mean())**2))
    if not .03<abs(beta[0])<.3 or r2<.6:raise RuntimeError('rate-to-angle calibration unreliable')
    return dict(scale=float(beta[0]),bias=float(beta[1]),integrated_angle_r2=r2,
                method='least squares of angle increments vs integrated raw rate and elapsed time; training trajectory only')

def simulate(w0,u,model):
    a=np.exp(-TS/model['tau']);b=model['K']*(1-a)
    v=np.sign(u)*np.maximum(0,np.abs(u)-model['deadzone'])
    nd=model['nd'];delayed=np.r_[np.zeros(nd),v[:len(v)-nd]] if nd else v
    out=np.zeros(len(u));out[0]=w0
    for k in range(len(out)-1):out[k+1]=a*out[k]+b*delayed[k]+model.get('c',0)
    return out

def identify(data,j,cal):
    ds=data if isinstance(data,list) else [data]
    choices=[]
    for nd in range(10):
        for dz in np.arange(0,6.01,.5):
            XX=[];YY=[]
            for d in ds:
                w=d['raw'][:,j]*cal['scale']+cal['bias'];u=d['u'][:,j]
                ve=np.sign(u)*np.maximum(0,np.abs(u)-dz)
                ix=np.arange(max(100,nd),len(w)-100-1)
                XX.append(np.stack([w[ix],ve[ix-nd],np.ones(len(ix))],1));YY.append(w[ix+1])
            X=np.concatenate(XX)
            a,b,c=np.linalg.lstsq(X,np.concatenate(YY),rcond=None)[0]
            if not .2<a<.995 or b<=0:continue
            tau=-TS/np.log(a);K=b/(1-a)
            if not .03<tau<.4 or not .2<K<2.5:continue
            model=dict(K=float(K),tau=float(tau),Td=nd*TS,nd=nd,deadzone=float(dz),c=float(c))
            errors=[]
            for d in ds:
                w=d['raw'][:,j]*cal['scale']+cal['bias'];u=d['u'][:,j]
                pred=simulate(w[0],u,model);m=(d['t']>=2)&(d['t']<=28)
                errors.append(w[m]-pred[m])
            model['train_rate_rmse']=float(np.sqrt(np.mean(np.concatenate(errors)**2)))
            choices.append(model)
    if not choices:raise RuntimeError('no acceptable positive-gain identified model')
    return min(choices,key=lambda m:m['train_rate_rmse'])

def metrics(data,j,cal,model):
    w=data['raw'][:,j]*cal['scale']+cal['bias'];pred=simulate(w[0],data['u'][:,j],model)
    m=(data['t']>=2)&(data['t']<=28)
    rmse=float(np.sqrt(np.mean((w[m]-pred[m])**2)))
    r2=1-float(np.sum((w[m]-pred[m])**2)/np.sum((w[m]-w[m].mean())**2))
    return dict(rate_rmse_deg_s=rmse,rate_r2=r2)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--train',required=True);ap.add_argument('--validation',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--extra-train',action='append',default=[])
    a=ap.parse_args();train_paths=[a.train]+a.extra_train
    validate_split(train_paths,a.validation)
    for p in train_paths:validate_acquisition(p,'train')
    validate_acquisition(a.validation,'validation')
    tr=[load_trial(v) for v in train_paths];va=load_trial(a.validation);report={}
    for j,ax in enumerate(('yaw','pitch')):
        cal=calibrate_rate(tr,j);model=identify(tr,j,cal);validation=metrics(va,j,cal,model)
        report[ax]=dict(rate_calibration=cal,model=model,validation=validation,
                       validation_used_for_fit=False,accepted=validation['rate_r2']>.5)
    report.update(train_source=[a.train]+a.extra_train,validation_source=a.validation,
                  note='Identified effective speed-command model, including sensor/inner-loop effects; no physical torque identification.')
    Path(a.out).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
