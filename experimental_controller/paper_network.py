"""Frozen, joint 40-input / 2-output LSTM, numpy-only inference on N100."""
import numpy as np
from paper_features import PaperFeatures

class PaperNetwork:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as z:
            self.Wih=np.array(z['Wih']);self.Whh=np.array(z['Whh']);self.b=np.array(z['b'])
            if self.Wih.shape[1]!=40:raise ValueError('paper network requires exactly 40 inputs')
            W=np.asarray(z['Wout'])
            if W.ndim!=2 or W.shape[0]!=2:raise ValueError('paper network requires two output axes')
            norm=np.linalg.norm(W,2)
            if not np.isfinite(norm) or norm<=0:raise ValueError('invalid output spectral norm')
            self.Wbar=W/norm;self.bout=np.array(z['bout'])
            self.mu=np.array(z['x_mu']);self.sd=np.array(z['x_sd'])
            self.y_mu=np.array(z['y_mu']);self.y_sd=np.array(z['y_sd'])
            self.meta={k:np.array(z[k]).tolist() for k in ('eps_max','eps_rate','D_max','kind') if k in z.files}
        if self.mu.shape!=(40,) or self.sd.shape!=(40,) or np.any(self.sd<=0):
            raise ValueError('invalid 40-dimensional normalization')
        self.H=self.Whh.shape[1]
        if self.Wih.shape!=(4*self.H,40) or self.Wbar.shape!=(2,self.H):
            raise ValueError('inconsistent LSTM dimensions')
        self.raw_bound_l2=float(np.linalg.norm(self.y_sd[:,None]*self.Wbar,2)*np.sqrt(self.H)+
                                np.linalg.norm(self.y_sd*self.bout+self.y_mu))
        self.reset()

    def reset(self):
        self.h=np.zeros(self.H);self.c=np.zeros(self.H);self.features=PaperFeatures()

    def step(self,q_deg,w_deg_s,u_prev):
        self.last_psi=self.features.step(q_deg,w_deg_s,u_prev)
        return self.step_psi(self.last_psi)

    def step_psi(self,psi):
        x=np.asarray(psi,dtype=float)
        if x.shape!=(40,) or not np.isfinite(x).all():raise ValueError('finite 40-vector required')
        gates=self.Wih@((x-self.mu)/self.sd)+self.Whh@self.h+self.b
        H=self.H
        sig=lambda a:1./(1.+np.exp(-np.clip(a,-60,60)))
        i,f,g,o=sig(gates[:H]),sig(gates[H:2*H]),np.tanh(gates[2*H:3*H]),sig(gates[3*H:])
        self.c=f*self.c+i*g;self.h=o*np.tanh(self.c)
        result=(self.Wbar@self.h+self.bout)*self.y_sd+self.y_mu
        if not np.isfinite(result).all():raise FloatingPointError('nonfinite network output')
        return result
