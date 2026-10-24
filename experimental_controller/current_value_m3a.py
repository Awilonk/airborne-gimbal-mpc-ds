"""Original APDO current-value + linear DO baseline in the common speed adapter.

This is not a harmonic predictor: APDO's causal Q-filter output is subtracted
once at the current decision; the entire MPC disturbance horizon is zero.
Original zr10_ctrl.APDO is used with n_ahead=0. There is no KoopLin/harmonic fit,
frequency-drift weighting, learned residual, or linear delay lead.

The fresh common controller transfers its existing model, soft MPC, LinearDO,
PaperPosterior, command histories and exact settings to this adapter. It cannot
be constructed from an already running controller. commit_sent is inherited,
so posterior interval indexing and post-publication DO propagation are common
to all methods, using the actual quantized controller contribution.
"""
import numpy as np

from ctrl_m5 import M5Controller
from zr10_ctrl import APDO


class CurrentValueM3a(M5Controller):
    def __init__(self, common):
        if not isinstance(common, M5Controller) or common.n_step or common._pending is not None:
            raise ValueError("M3a requires a fresh unused common controller")
        # Transfer ownership of the freshly constructed common components. The
        # caller replaces that unused shell; no live baseline state is changed.
        self.__dict__ = common.__dict__.copy()
        self.pred = False
        self.lead = 'none'
        self.drift = 0.
        self.apdo = APDO(self.Ts, f_nom=self.kl.f, n_ahead=0)
        self.compensation_current_raw = self.compensation_current_used = 0.

    def step(self, theta, omega, th_ref, om_ref, t=None, eps_raw=0., d_L=0.):
        if self._pending is not None:
            raise RuntimeError('previous command was not committed after sending')
        if d_L != 0:
            raise ValueError('M3a has no external learned disturbance feedforward')
        self.H = -omega/self.K
        self.u_post_used = self.u_hist[0]  # previous interval, k-1-nd
        self.dplus = self.post.update(omega, self.u_post_used)
        self.dplus_raw = self.post.raw
        self.posterior_valid = self.n_step > 0
        current = float(self.apdo.update(self.dplus))
        if not np.isfinite(current):
            raise FloatingPointError('nonfinite original APDO current value')
        self.d_h = self.d_h_raw = current
        self.d_h_mpc = 0.
        self.d_L = 0.
        self.d_r = self.do.z2  # residual state at k, before commit advances it
        self.d_r_c = float(np.clip(self.d_r, -self.d_bar, self.d_bar))
        self.lead_dlin = self.lead_eps = self.lead_d = 0.
        self.rho_now = self.rho_end = 1.
        self.compensation_current_raw = current+self.d_r_c
        self.compensation_current_used = float(np.clip(self.compensation_current_raw, -self.d_bar, self.d_bar))
        z = np.zeros(self.model.n)
        z[:2] = theta, omega
        for i in range(self.model.nd):
            z[2+i] = self.v_hist[-1-i]
        um = self.u_max-self.margin
        comp, H = self.compensation_current_used, self.H
        lo = max(-self.u_max, -um+comp, -self.actual_limit+H+comp)
        hi = min(self.u_max, um+comp, self.actual_limit+H+comp)
        if not np.isfinite([lo, hi]).all() or lo >= hi:
            raise RuntimeError('empty input box: loss of assumed H headroom')
        self.bounds = float(lo), float(hi)
        # Current compensation is external. Do not add it to a prediction
        # horizon as well, which would count the periodic estimate twice.
        v = self.mpc.solve_d(z, th_ref, om_ref, np.zeros(self.mpc.Np), lo, hi)
        u = float(v-H-comp)
        if not np.isfinite(u) or abs(u) > self.actual_limit+1e-6:
            raise FloatingPointError('actual command violates solved box')
        self.u_nom = float(v)
        self._pending = dict(omega=float(omega), u=u, H=H, d_h=current, d_L=0.)
        return u

    def diagnostics(self):
        values = super().diagnostics()
        # These common-shell estimators exist but are never stepped by M3a;
        # report original APDO states under their actual names instead.
        for key in ('harmonic_state', 'harmonic_covariance', 'frequency_rls_a',
                    'frequency_rls_P', 'frequency_filter_x', 'frequency_filter_y'):
            values.pop(key, None)
        values.update(periodic_observer='original_apdo_current_value',
            periodic_prediction=False, current_value_periodic_compensation=True,
            linear_lead=False, learned_residual=False, apdo_n_ahead=self.apdo.n_ahead,
            f_hat=self.apdo.freq_hz(), apdo_frequency_w=self.apdo.w,
            apdo_z1=self.apdo.z1, apdo_yh=self.apdo.yh, apdo_q=self.apdo.q,
            apdo_dh=self.apdo.dh, apdo_bpf_x=list(self.apdo.bx), apdo_bpf_y=list(self.apdo.by),
            apdo_buffer_count=len(self.apdo.buf),
            apdo_history_source='logged dplus sequence from initial zero state',
            compensation_current_raw=self.compensation_current_raw,
            compensation_current_used=self.compensation_current_used,
            harmonic_covariance_applicable=False,
            covariance_source='inactive common-shell placeholder; original APDO has no covariance',
            observer_timing='APDO and compensation decision k; linear DO post-commit k+1')
        return values
