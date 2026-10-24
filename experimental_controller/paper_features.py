"""Paper Eq.(25): 4 x [qa,wa,qe,we,u_a_prev,u_e_prev,sin qa,sin qe,cos qa,cos qe]."""
from collections import deque
import numpy as np

class PaperFeatures:
    def __init__(self):self.buf=None

    def step(self,q_deg,w_deg_s,u_prev):
        q=np.asarray(q_deg,dtype=float);w=np.asarray(w_deg_s,dtype=float);u=np.asarray(u_prev,dtype=float)
        if q.shape!=(2,) or w.shape!=(2,) or u.shape!=(2,):raise ValueError('two axes required')
        r=np.radians(q)
        frame=np.array([q[0],w[0],q[1],w[1],u[0],u[1],np.sin(r[0]),np.sin(r[1]),np.cos(r[0]),np.cos(r[1])])
        if not np.isfinite(frame).all():raise ValueError('nonfinite network feature')
        if self.buf is None:self.buf=deque([frame.copy() for _ in range(4)],maxlen=4)
        else:self.buf.append(frame)
        return np.concatenate([self.buf[-1-i] for i in range(4)])
