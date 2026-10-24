"""Paper-aligned Proposed core; isolated from the legacy M5 deployment.

step computes the decision at k. commit_sent advances the observer k -> k+1
AFTER publishing the quantized command. The ZR10 adapter's J_eq=tau/K and
H_eq=-omega/K are equivalent speed-inner-loop coefficients, not physical torque.
"""
import math
from collections import deque
import numpy as np
from zr10_ctrl import AXIS_ID, AxisModel, LinearDO, Posterior
from ctrl_pred import MPCdist, HarmonicPredictor
from ctrl_klin import KoopLin
from paper_posterior import PaperPosterior

def do_bw(axis, wo):
    p=AXIS_ID[axis]
    return dict(l0=2*wo,l2=0.,L2=wo*wo*p['tau']/p['K'])

class M5Controller:
    def __init__(self,axis,Ts,f_nom=.5,wo=5.,margin=5.,A_max=12.,
                 pred=True,nominal=True,mpc_kw=None,do_kw=None,kl_kw=None,hp_kw=None,
                 u_max=50.,d_bar=30.,drift=4.,drift_L=50,lead='external40',
                 lead_fc=2.,D_max=6.,eps_max=9.,eps_rate=280.,soft_kw=None,
                 actual_limit=45.):
        if not nominal or lead not in ('none','external40'):
            raise ValueError('this isolated release supports Proposed only')
        if not 0<d_bar<u_max-margin or not 0<actual_limit<=u_max:
            raise ValueError('invalid input headroom')
        self.axis,self.Ts,self.pred=axis,Ts,pred
        p=AXIS_ID[axis];self.K=p['K']
        self.model=AxisModel(p['K'],p['tau'],p['Td'],Ts,cancel_drift=True)
        if soft_kw:
            from soft_mpc import SoftMPCdist
            self.mpc=SoftMPCdist(self.model,u_max=u_max,soft_kw=soft_kw,**(mpc_kw or {}))
        else:self.mpc=MPCdist(self.model,u_max=u_max,**(mpc_kw or {}))
        self.do=LinearDO(p['K'],p['tau'],Ts,**(do_kw or do_bw(axis,wo)))
        self.post=PaperPosterior(p['K'],p['tau'],Ts)
        self.hp=HarmonicPredictor(Ts,**(hp_kw or {}))
        self.kl=KoopLin(Ts,f_nom=f_nom,**(kl_kw or dict(lam=.999,p_min=.05)))
        self.u_max,self.margin,self.actual_limit=u_max,margin,actual_limit
        self.d_bar,self.A_max=d_bar,A_max
        self.drift,self.drift_L=drift,drift_L
        self._fh=deque(maxlen=drift_L+1)
        self.lead,self.D_max,self.eps_max,self.eps_rate=lead,D_max,eps_max,eps_rate
        self._lead_alpha=math.exp(-2*math.pi*lead_fc*Ts)
        self._prev_dr=None;self._inc=0.;self.lead_eps=0.
        n=self.model.nd+1
        self.u_hist=deque([0.]*n,maxlen=n)
        self.v_hist=deque([0.]*n,maxlen=n)
        self.H_hist=deque([0.]*n,maxlen=n)
        self.n_step=0;self._pending=None
        self.d_h=self.d_h_raw=self.d_h_mpc=self.d_r=self.d_r_c=self.dplus=self.dplus_raw=0.
        self.d_L=self.u_nom=self.lead_d=self.lead_dlin=self.H=0.
        self.rho_end=self.rho_now=1.
        self.posterior_valid=False
        self.u_post_used=self.u_do_used=self.h_delay_residual=self.quantization_residual=0.
        self.bounds=(-u_max,u_max)

    def step(self,theta,omega,th_ref,om_ref,t=None,eps_raw=0.,d_L=0.):
        if self._pending is not None:
            raise RuntimeError('previous command was not committed after sending')
        self.H=-omega/self.K
        self.u_post_used=self.u_hist[0] # u[k-1-nd], previous interval
        self.dplus=self.post.update(omega,self.u_post_used)
        self.dplus_raw=self.post.raw
        self.posterior_valid=self.n_step>0
        Np=self.mpc.Np
        if self.pred:
            f=self.kl.update(self.dplus)
            self.hp.update(self.dplus,f)
            n=len(self.hp.th)-1
            nr=np.linalg.norm(self.hp.th[:n])
            if nr>self.A_max:self.hp.th[:n]*=self.A_max/nr
            raw=self.hp.predict(Np)
            self.d_h=self.d_h_raw=float(raw[0])
            self._fh.append(f)
            df=abs(self._fh[-1]-self._fh[0])
            rho=np.clip(1-self.drift*2*math.pi*df*(np.arange(Np)+self.model.nd)*self.Ts,0,1)
            dh_fut=rho*raw
            self.rho_now,self.rho_end=float(rho[0]),float(rho[-1])
        else:
            self.d_h=self.d_h_raw=0.;dh_fut=np.zeros(Np)
        self.d_h_mpc=float(dh_fut[0])
        self.d_L=float(d_L)
        self.d_r=self.d_L+self.do.z2 # z_k BEFORE the k -> k+1 propagation
        inc=0. if self._prev_dr is None else self.d_r-self._prev_dr
        self._prev_dr=self.d_r
        self._inc=self._lead_alpha*self._inc+(1-self._lead_alpha)*inc
        self.lead_dlin=self.model.nd*self._inc
        if self.lead=='external40':
            eps=float(np.clip(eps_raw,-self.eps_max,self.eps_max))
            self.lead_eps=float(np.clip(eps,self.lead_eps-self.eps_rate*self.Ts,
                                        self.lead_eps+self.eps_rate*self.Ts))
            self.lead_d=float(np.clip(self.lead_dlin+self.lead_eps,-self.D_max,self.D_max))
        else:self.lead_eps=self.lead_d=0.
        self.d_r_c=float(np.clip(self.d_r+self.lead_d,-self.d_bar,self.d_bar))
        z=np.zeros(self.model.n);z[:2]=theta,omega
        for i in range(self.model.nd):z[2+i]=self.v_hist[-1-i]
        um=self.u_max-self.margin;dr=self.d_r_c;H=self.H
        # Intersect paper V_k with the actual speed-interface command bound.
        lo=max(-self.u_max,-um+dr,-self.actual_limit+H+dr)
        hi=min(self.u_max,um+dr,self.actual_limit+H+dr)
        if not np.isfinite([lo,hi]).all() or lo>=hi:
            raise RuntimeError('empty input box: loss of assumed H headroom')
        self.bounds=float(lo),float(hi)
        v=self.mpc.solve_d(z,th_ref,om_ref,dh_fut,lo,hi)
        u=float(v-H-dr)
        if not np.isfinite(u) or abs(u)>self.actual_limit+1e-6:
            raise FloatingPointError('actual command violates solved box')
        self.u_nom=float(v)
        self._pending=dict(omega=float(omega),u=u,H=H,d_h=self.d_h_raw,d_L=self.d_L)
        return u

    def commit_sent(self,sent):
        if self._pending is None:raise RuntimeError('no pending command')
        p=self._pending;sent=float(sent)
        if not np.isfinite(sent) or abs(sent)>self.actual_limit:
            raise ValueError('invalid actual sent level')
        nd=self.model.nd
        self.u_do_used=self.u_hist[-nd] if nd else sent
        h_del=self.H_hist[-nd] if nd else p['H']
        self.h_delay_residual=p['H']-h_del
        self.quantization_residual=sent-p['u']
        self.do.update(p['omega'],self.u_do_used,d_known=p['d_h'],d_ff=p['d_L'])
        self.u_hist.append(sent);self.v_hist.append(self.u_nom);self.H_hist.append(p['H'])
        self.n_step+=1;self._pending=None

    def diagnostics(self):
        vals={k:getattr(self,k) for k in ('dplus','dplus_raw','d_h_raw','d_h_mpc','d_r','d_L','d_r_c',
              'lead_dlin','lead_eps','lead_d','rho_now','rho_end','H','u_nom','u_post_used',
              'u_do_used','h_delay_residual','quantization_residual','posterior_valid')}
        vals.update(f_hat=self.kl.f,cov_min=getattr(self.hp,'cov_min',100.),
                    cov_max=getattr(self.hp,'cov_max',100.),cov_projections=self.hp.cov_projections,
                    v_lo=self.bounds[0],v_hi=self.bounds[1],
                    do_omega_hat=self.do.om_hat,do_z2=self.do.z2,
                    posterior_previous_omega=self.post.prev_omega,posterior_previous_H=self.post.prev_H,
                    harmonic_state=self.hp.th.tolist(),harmonic_covariance=self.hp.P.tolist(),
                    frequency_rls_a=self.kl.ah,frequency_rls_P=self.kl.P,
                    frequency_filter_x=list(self.kl.x),frequency_filter_y=list(self.kl.y),
                    observer_timing='post commit k+1; d_r and lead fields are decision k')
        vals.update(getattr(self.mpc,'last_diag',{}))
        return vals
