"""Small, certified convex QP with one L1 slack per two-sided state row.

The Newton system eliminates all diagonal slack blocks, leaving only the
control-horizon square matrix. No dense (control + slack) QP is assembled.
"""
import numpy as np


class SoftQPFailure(RuntimeError):
    def __init__(self,message,iterations,gap,residual):
        super().__init__(message)
        self.diagnostics=dict(qp_iterations=iterations,qp_objective_gap=gap,
                              soft_feasibility_residual=residual)


class PairedSoftQP:
    def __init__(self, hessian, state_matrix, penalty, max_iterations=40):
        self.H=np.asarray(hessian,dtype=float)/float(penalty)
        self.G=np.asarray(state_matrix,dtype=float)
        self.penalty=float(penalty)
        self.max_iterations=int(max_iterations)
        if not (np.isfinite(self.H).all() and np.isfinite(self.G).all()
                and np.isfinite(self.penalty) and self.penalty>0):
            raise ValueError('nonfinite soft QP matrices/penalty')
        np.linalg.cholesky(self.H)

    @staticmethod
    def _step(positive, direction, fraction=1.):
        negative=direction<0
        if not np.any(negative):return 1.
        return min(1.,fraction*float(np.min(-positive[negative]/direction[negative])))

    def solve(self, linear, lower, upper, lo, hi, initial):
        """Return control and a global objective-gap certificate, or raise.

        Objective is .5 U'H U + linear'U + penalty * sum(max(0,
        lower-GU, GU-upper)). The input box is hard, including fixed boxes.
        Certificate uses a feasible L1 dual and minimizes its supporting
        hyperplane over the exact input box. It bounds suboptimality in the
        original (unscaled) objective units, not just KKT feasibility.
        """
        f=np.asarray(linear,dtype=float)/self.penalty
        lower=np.asarray(lower,dtype=float);upper=np.asarray(upper,dtype=float)
        initial=np.asarray(initial,dtype=float)
        if not (np.isfinite(f).all() and np.isfinite(lower).all()
                and np.isfinite(upper).all() and np.isfinite(initial).all()
                and np.isfinite(lo) and np.isfinite(hi) and lo<=hi):
            raise ValueError('nonfinite or reversed soft QP input bounds')
        n=len(f);k=len(lower)
        if lo==hi:
            return np.full(n,lo),dict(iterations=0,objective_gap=0.,residual=0.)
        # Center/scale the hard input box to [-1,1] for numerical conditioning.
        radius=.5*(hi-lo);center=np.full(n,.5*(hi+lo))
        H=self.H*(radius*radius);f=radius*(f+self.H@center)
        G=self.G*radius
        low=lower-self.G@center;high=upper-self.G@center
        x=np.clip((initial-center)/radius,-.95,.95)
        y=G@x;s=np.maximum(0.,np.maximum(y-high,low-y))+1.
        b=np.concatenate((high,-low,np.zeros(k),np.ones(2*n)))
        v=np.concatenate((high-y+s,y-low+s,s,1.-x,1.+x))
        lam=np.concatenate((np.full(2*k,.1),np.full(k,.8),np.full(2*n,.1)))

        def ax(dx,ds):
            gd=G@dx
            return np.concatenate((gd-ds,-gd-ds,-ds,dx,-dx))

        gap=np.inf;residual=np.inf
        for iteration in range(self.max_iterations+1):
            # Force an exactly hard-box feasible primal before certification.
            candidate=np.clip(x,-1.,1.)
            yc=G@candidate
            alpha=np.maximum(lam[:k],0.);beta=np.maximum(lam[k:2*k],0.)
            pair_scale=np.maximum(1.,alpha+beta)
            alpha=alpha/pair_scale;beta=beta/pair_scale
            slack=np.maximum(0.,np.maximum(yc-high,low-yc))
            gradient=H@candidate+f+G.T@(alpha-beta)
            # All terms are nonnegative analytically; summing this way avoids
            # cancellation of large unavoidable slack constants.
            complement=slack-alpha*(yc-high)-beta*(low-yc)
            stationarity=gradient*candidate+np.abs(gradient)
            gap=self.penalty*float(np.sum(np.maximum(complement,0.))+
                                   np.sum(np.maximum(stationarity,0.)))
            if np.isfinite(gap) and gap<=1e-5:
                control=np.clip(center+radius*candidate,lo,hi)
                return control,dict(iterations=iteration,objective_gap=gap,residual=0.)
            if iteration==self.max_iterations:break

            rp=ax(x,s)+v-b
            rd_u=H@x+f+G.T@(lam[:k]-lam[k:2*k])+lam[3*k:3*k+n]-lam[3*k+n:]
            rd_s=1.-lam[:k]-lam[k:2*k]-lam[2*k:3*k]
            residual=max(float(np.max(np.abs(rp))),float(np.max(np.abs(rd_u))),
                         float(np.max(np.abs(rd_s))))
            d=lam/v
            du,dl,dz=d[:k],d[k:2*k],d[2*k:3*k]
            total=du+dl+dz;difference=dl-du
            # Stable Schur coefficient: avoid subtraction of nearly equal
            # large terms when an individual state row becomes active.
            weight=(4.*du*dl+(du+dl)*dz)/total
            K=H+G.T@(weight[:,None]*G)+np.diag(d[3*k:3*k+n]+d[3*k+n:])

            def direction(rc):
                t=(rc-lam*rp)/v
                rhs_s=-rd_s-t[:k]-t[k:2*k]-t[2*k:3*k]
                rhs_u=-rd_u+G.T@(t[:k]-t[k:2*k])+t[3*k:3*k+n]-t[3*k+n:]
                dx=np.linalg.solve(K,rhs_u-G.T@(difference*rhs_s/total))
                ds=(rhs_s-difference*(G@dx))/total
                dv=-rp-ax(dx,ds)
                dlambda=(-rc-lam*dv)/v
                return dx,ds,dv,dlambda

            mu=float(v@lam)/len(v)
            dx,ds,dv,dlambda=direction(v*lam)
            ap=self._step(v,dv);ad=self._step(lam,dlambda)
            mu_aff=float((v+ap*dv)@(lam+ad*dlambda))/len(v)
            sigma=min(1.,max(0.,mu_aff/mu))**3
            dx,ds,dv,dlambda=direction(v*lam+dv*dlambda-sigma*mu)
            ap=self._step(v,dv,.995);ad=self._step(lam,dlambda,.995)
            x=x+ap*dx;s=s+ap*ds;v=v+ap*dv;lam=lam+ad*dlambda
            if not (np.isfinite(x).all() and np.isfinite(s).all()
                    and np.isfinite(v).all() and np.isfinite(lam).all()
                    and np.all(v>0) and np.all(lam>0)):
                raise SoftQPFailure('soft QP nonfinite/nonpositive interior iterate',iteration+1,gap,residual)
        raise SoftQPFailure('soft QP uncertified after %d iterations: gap %.3g residual %.3g'%
                            (self.max_iterations,gap,residual),self.max_iterations,gap,residual)
