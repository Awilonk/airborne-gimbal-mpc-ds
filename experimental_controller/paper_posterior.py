import math

class PaperPosterior:
    """Eq.(18) first, then an explicitly documented causal 4-Hz noise filter.

    H_{k-1} and u_sent[k-1-nd] have separate histories from the forward DO.
    The raw inverse-dynamics value is logged; the initial derivative is invalid.
    """
    def __init__(self,K,tau,Ts,f_c=4.):
        self.K,self.J,self.Ts=K,tau/K,Ts
        self.a=math.exp(-2*math.pi*f_c*Ts)
        self.prev_omega=self.prev_H=None;self.raw=self.filtered=0.;self.valid=False

    def update(self,omega,u_prev_delayed):
        self.valid=self.prev_omega is not None
        if self.valid:self.raw=self.J*(omega-self.prev_omega)/self.Ts-u_prev_delayed-self.prev_H
        else:self.raw=0.
        self.filtered=self.a*self.filtered+(1-self.a)*self.raw
        self.prev_omega=float(omega);self.prev_H=-omega/self.K
        return self.filtered
