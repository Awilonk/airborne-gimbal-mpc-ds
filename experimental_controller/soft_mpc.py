"""Input-hard / state-soft MPC. Epigraph slacks carry a LINEAR L1 penalty.

The fast path solves the paper box problem and checks all state inequalities.
When a state constraint is active, solve its convex QP with explicit slacks.
No claim is made that the configured penalty exceeds all hard-QP multipliers.
"""
import time
import numpy as np
from ctrl_pred import MPCdist
from small_soft_qp import PairedSoftQP

class SoftMPCdist(MPCdist):
    def __init__(self,model,soft_kw,**kw):
        super().__init__(model,**kw)
        self.soft=dict(soft_kw);self.last_diag=dict(qp_success=True,qp_iterations=0,qp_objective_gap=0.,qp_solver='box_active_set',
            soft_feasibility_residual=0.,slack_mechanical_max=0.,slack_speed_max=0.,slack_fov_max=0.,mpc_ms=0.)
        if (not all(np.isfinite(self.soft[k]) for k in ('theta_lo','theta_hi','penalty','fov_half','omega_lim'))
                or not np.isfinite(self.soft.get('theta_offset',0.))
                or self.soft['penalty']<=0 or self.soft['fov_half']<=0 or self.soft['omega_lim']<=0):
            raise ValueError('positive soft-constraint settings required')
        self.soft_solver=PairedSoftQP(self.M.T@self.M,
            np.vstack((self.Gam[:,0,:],self.Gam[:,1,:],self.Gam[:,0,:])),self.soft['penalty'])

    def solve_d(self,z,th_ref,om_ref,dvec,lo,hi):
        t0=time.perf_counter();N,C=self.Np,self.Nc;s=self.soft
        if not all(np.isfinite(np.asarray(a)).all() for a in (z,th_ref,om_ref,dvec,lo,hi)) or lo>hi:
            raise ValueError('nonfinite soft MPC input or reversed input bounds')
        u0=super().solve_d(z,th_ref,om_ref,dvec,lo,hi)
        Gq,Gw=self.Gam[:,0,:],self.Gam[:,1,:]
        q0=self.Phi[:,0,:]@z+self.Psi[:,0,:]@dvec
        w0=self.Phi[:,1,:]@z+self.Psi[:,1,:]@dvec
        off=s.get('theta_offset',0.)
        lows=[np.full(N,s['theta_lo']-off),np.full(N,-s['omega_lim']),np.asarray(th_ref)-s['fov_half']]
        highs=[np.full(N,s['theta_hi']-off),np.full(N,s['omega_lim']),np.asarray(th_ref)+s['fov_half']]
        bases=[q0,w0,q0];Gs=[Gq,Gw,Gq]
        def violations(U):
            return [np.maximum(0,np.maximum(l-(b+G@U),(b+G@U)-h))
                    for l,h,b,G in zip(lows,highs,bases,Gs)]
        vs=violations(self.U)
        # The exact box minimizer is also a lower bound for the soft objective.
        # Thus its L1 penalty alone is a valid gap certificate, including tiny
        # round-off violations at a state boundary.
        qp_success=True;iterations=0;residual=0.;gap=float(s['penalty'])*sum(float(v.sum()) for v in vs)
        algorithm='box_active_set'
        if gap>1e-5:
            algorithm='paired_primal_dual'
            target=np.concatenate([self.wth*th_ref-self.Pth@z-self.Dth@dvec,
                                   self.wom*om_ref-self.Pom@z-self.Dom@dvec,np.zeros(C)])
            try:
                control,diag=self.soft_solver.solve(-self.M.T@target,
                    np.concatenate([l-b for l,b in zip(lows,bases)]),
                    np.concatenate([h-b for h,b in zip(highs,bases)]),lo,hi,self.U)
            except (RuntimeError,ValueError,np.linalg.LinAlgError) as ex:
                self.last_diag=dict(qp_success=False,qp_error=str(ex),qp_iterations=0,
                    qp_solver=algorithm,qp_objective_gap=None,soft_feasibility_residual=None,mpc_ms=(time.perf_counter()-t0)*1000)
                self.last_diag.update(getattr(ex,'diagnostics',{}))
                raise
            iterations=diag['iterations'];residual=diag['residual'];gap=diag['objective_gap']
            self.U=control;u0=float(self.U[0]);vs=violations(self.U)
        self.last_diag=dict(qp_success=qp_success,qp_iterations=iterations,soft_feasibility_residual=residual,
                            qp_objective_gap=gap,qp_solver=algorithm,
                            slack_mechanical_max=float(vs[0].max()),slack_speed_max=float(vs[1].max()),
                            slack_fov_max=float(vs[2].max()),mpc_ms=(time.perf_counter()-t0)*1000)
        return u0
