"""Causal angle-consistent rate feedback for the angle-hold task.

SDK angle and joint gyro need not share a frame on a moving base. This adapter
estimates the derivative of the *reported angle* from that angle and committed
SDK commands. It does not assume unverified IMU mount signs. A nuisance constant
input state lets actual unmodeled motion correct the prediction. That state is
never supplied as disturbance compensation to any controller.

Raw SDK angles/rates remain logged; raw angle/rate hard guards and the frozen
network's original feature stream are independent of this estimator. This
adapter does not establish an inertial LOS calibration or remove physical
motion. Commands are never filtered after solving the MPC. All methods share
the adapter for hold; tracking retains its previously accepted raw rate path.
The experimental high-dynamic observer failed a ground rate guard and is not
enabled by this release.
"""
from collections import deque
import math

import numpy as np
from zr10_ctrl import c2d


class AngleRateObserver:
    def __init__(self,K,tau,Td,Ts,bandwidth,actual_limit):
        values=np.asarray([K,tau,Td,Ts,bandwidth,actual_limit],float)
        if not np.isfinite(values).all() or min(K,tau,Ts,bandwidth,actual_limit)<=0 or Td<0:
            raise ValueError('finite positive angle-feedback model and bandwidth required')
        if bandwidth*Ts>1.2:
            raise ValueError('angle-feedback bandwidth exceeds verified discrete range')
        self.K,self.tau,self.Ts,self.bandwidth,self.actual_limit=map(float,(K,tau,Ts,bandwidth,actual_limit))
        self.nd=max(0,int(round(Td/Ts)))
        a,b=c2d(K,tau,Ts)
        self.A=np.eye(3);self.A[:2,:2]=a;self.A[:2,2]=b
        self.B=np.r_[b,0.]
        # Dual Ackermann placement for the CURRENT observer (I-L*C)*A.
        dual=self.A.T;vector=self.A[0,:]
        controllability=np.stack((vector,dual@vector,dual@dual@vector),axis=1)
        pole=math.exp(-bandwidth*Ts)
        polynomial=dual@dual@dual-3*pole*(dual@dual)+3*pole*pole*dual-pole**3*np.eye(3)
        self.L=np.linalg.solve(controllability.T,np.array([0.,0.,1.]))@polynomial
        if not np.isfinite(self.L).all():raise ValueError('nonfinite angle-feedback gains')
        self.history=deque([0.]*(self.nd+1),maxlen=self.nd+1)
        self.state=None;self.pending=False;self.n=0
        self.innovation=self.input_used=0.

    def update(self,angle):
        angle=float(angle)
        if not math.isfinite(angle):raise ValueError('nonfinite angle measurement')
        if self.pending:raise RuntimeError('commit sent input before next angle-feedback update')
        self.input_used=float(self.history[0])
        if self.state is None:
            # The first angle anchors the state; no frame-inconsistent gyro
            # is used to initialize its derivative or nuisance disturbance.
            self.state=np.array([angle,0.,0.]);self.innovation=0.
        else:
            prior=self.A@self.state+self.B*self.input_used
            self.innovation=angle-float(prior[0])
            self.state=prior+self.L*self.innovation
        if not np.isfinite(self.state).all():raise FloatingPointError('nonfinite angle-feedback state')
        self.pending=True
        return self.state.copy()

    def commit_sent(self,sent):
        if not self.pending:raise RuntimeError('no angle-feedback decision to commit')
        sent=float(sent)
        if not math.isfinite(sent) or abs(sent)>self.actual_limit or sent!=round(sent):
            raise ValueError('invalid committed angle-feedback command')
        self.history.append(sent);self.pending=False;self.n+=1

    def diagnostics(self):
        return dict(kind='angle_input_observer_v1',bandwidth_rad_s=self.bandwidth,
            nominal_sample_time_s=self.Ts,input_delay_samples=self.nd,
            state=self.state.tolist() if self.state is not None else None,
            gains=self.L.tolist(),innovation_deg=self.innovation,previous_interval_input=self.input_used,
            input_timing='u[k-1-nd] from committed integer controller contribution',
            network_rate_source='raw_calibrated_sdk',raw_rate_guard=True,
            angle_control_source='raw_sdk',rate_control_source='angle_consistent_model_observer',
            disturbance_state_used_for_control=False)


def make_angle_feedback(cfg):
    settings=cfg.get('feedback_estimator')
    if settings is None:return None
    if not isinstance(settings,dict) or settings.get('kind')!='angle_input_observer_v1':
        raise ValueError('unknown angle-feedback estimator')
    if settings.get('network_features')!='raw_calibrated_sdk' or settings.get('raw_rate_guard') is not True:
        raise ValueError('raw guards and frozen network feature stream must be retained')
    if settings.get('scope')!='angle_hold_only':raise ValueError('only angle-hold feedback adaptation has been commissioned')
    if np.any(np.asarray(cfg['reference_amplitude_deg'],float)!=0.):return None
    bandwidth=float(settings['hold_bandwidth_rad_s'])
    observers=[]
    for axis in ('yaw','pitch'):
        p=cfg['identified'][axis]['model']
        observers.append(AngleRateObserver(p['K'],p['tau'],p['Td'],.02,bandwidth,cfg['actual_level_limit']))
    return observers
