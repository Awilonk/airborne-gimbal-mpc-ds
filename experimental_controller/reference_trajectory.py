"""Shared relative two-axis reference; angles in degrees, frequency in rad/s."""
import numpy as np


def envelope_and_rate(t,duration):
    if not np.isfinite(duration) or duration<=0:raise ValueError('positive finite duration required')
    t=np.asarray(t,dtype=float)
    if not np.isfinite(t).all():raise ValueError('finite reference time required')
    ramp=min(2.,duration/2.)
    edge=np.clip(np.minimum(t,duration-t),0.,ramp)
    env=.5-.5*np.cos(np.pi*edge/ramp)
    slope=.5*np.pi/ramp*np.sin(np.pi*edge/ramp)
    slope=np.where(t<duration/2.,slope,-slope)
    active=(t>0)&(t<duration)
    return np.where(active,env,0.),np.where(active,slope,0.)


class AngularReference:
    def __init__(self,amplitude,omega,phase=(0.,0.)):
        self.amplitude=np.asarray(amplitude,dtype=float)
        self.omega=np.asarray(omega,dtype=float)
        self.phase=np.asarray(phase,dtype=float)
        if any(v.shape!=(2,) or not np.isfinite(v).all() for v in (self.amplitude,self.omega,self.phase)):
            raise ValueError('reference parameters must be finite two-axis vectors')
        if np.any(self.amplitude<0) or np.any(self.omega<=0):raise ValueError('invalid amplitude or frequency')

    def value(self,t,duration=30.):
        t=np.asarray(t,dtype=float);env,_=envelope_and_rate(t,duration)
        return env[...,None]*self.amplitude*np.sin(t[...,None]*self.omega+self.phase)

    def rate(self,t,duration=30.):
        t=np.asarray(t,dtype=float);env,slope=envelope_and_rate(t,duration)
        phase=t[...,None]*self.omega+self.phase
        return self.amplitude*(slope[...,None]*np.sin(phase)+env[...,None]*self.omega*np.cos(phase))

    def acceleration(self,t,duration=30.):
        """Piecewise second derivative; choose zero envelope curvature at joins."""
        t=np.asarray(t,dtype=float);env,slope=envelope_and_rate(t,duration)
        ramp=min(2.,duration/2.);edge=np.minimum(t,duration-t)
        curvature=np.where((edge>0)&(edge<ramp),.5*(np.pi/ramp)**2*np.cos(np.pi*edge/ramp),0.)
        phase=t[...,None]*self.omega+self.phase
        return self.amplitude*((curvature[...,None]-env[...,None]*self.omega**2)*np.sin(phase)+
            2*slope[...,None]*self.omega*np.cos(phase))
