"""Independent PID, cascade LADRC and plain nominal MPC speed-level baselines.

The identified plant is q_dot=w, tau*w_dot=-w+K*u(t-Td). K is deg/s
per SDK speed level; no motor torque or physical inertia is assumed. References
and measurements are relative angles in deg and calibrated rates in deg/s.

Every step(k) must be followed exactly once by commit_sent(integer controller
contribution). Unmeasured virtual injection must never be passed to commit.
The observer propagates over [k,k+1] with u[k-nd], while MPC augments its current
state with the committed history. Input delays are rounded to integer samples,
as in the common AxisModel. eps_raw and d_L are accepted but never consumed.

PID uses only Kp*e+I+Kd*e_dot, with e=r(t)-q and e_dot=r_dot(t)-w.
The derivative of tracking error is a PID term, not inverse-model feedforward.
No reference acceleration or nominal model feedforward enters PID. Conditional
integration and exponential back-calculation both run at commit, using the
actual integer sent input. LADRC adds
r_ddot to its rate feedback acceleration. Supplied current reference
acceleration is preferred; standalone callers may use a causal difference of
current reference rates, over elapsed t (fixed Ts only when t is absent), with
first acceleration zero. Neither method differentiates measured omega.

LADRC closes an outer proportional angle loop around first-order rate ADRC.
The standard linear ESO observes calibrated rate, with total acceleration
f=-w/tau+unmeasured effects and b0=K/tau. A ZOH discrete current observer puts
both error poles at exp(-wo*Ts), avoiding forward Euler instability. See Herbst
and Madonski, Sections III-A/III-C: https://arxiv.org/html/2211.07309v2 . Limited
actual input feedback also gives LADRC its usual windup protection.

Plain MPC uses AxisModel(cancel_drift=False) and existing input-hard/state-soft
QP code. Its disturbance horizon is identically zero, with no ESO, DO, neural
network, H cancellation, periodic estimator, or learned compensation.

Optional tuning: cfg['engineering_baselines'][method] holds scalar overrides;
an inner 'yaw'/'pitch' dict overrides common values. Effective numeric settings,
gain derivations, units and timing conventions are JSON-safe and logged.
"""
import copy
import math
from collections import deque
from types import SimpleNamespace

import numpy as np

from zr10_ctrl import AxisModel


SAMPLE_TIME_S = .02
ENGINEERING_METHODS = ('PID', 'LADRC', 'MPC')
_TUNING_KEYS = dict(
    PID={'closed_loop_rad_s', 'damping_ratio', 'integral_pole_ratio', 'kp', 'ki', 'kd',
         'antiwindup_time_s', 'integral_limit_levels', 'reference_feedforward', 'acceleration_feedforward'},
    LADRC={'angle_bandwidth_rad_s', 'rate_bandwidth_rad_s', 'observer_bandwidth_rad_s',
           'acceleration_feedforward'},
    MPC={'Np', 'Nc', 'q_th', 'q_om', 'r', 'p_term', 'soft_penalty'})


def _number(value, name, minimum=None, strictly_positive=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('%s must be numeric' % name)
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('%s must be finite' % name)
    if strictly_positive and result <= 0.:
        raise ValueError('%s must be positive' % name)
    if minimum is not None and result < minimum:
        raise ValueError('%s below minimum' % name)
    return result


def _options(cfg, method, axis):
    settings = cfg.get('engineering_baselines', {}).get(method, {})
    if not isinstance(settings, dict):
        raise ValueError('engineering baseline tuning must be a dict')
    merged = {key: value for key, value in settings.items() if key not in ('yaw', 'pitch')}
    axis_values = settings.get(axis, {})
    if not isinstance(axis_values, dict):
        raise ValueError('axis tuning must be a dict')
    merged.update(axis_values)
    unknown = sorted(set(merged)-_TUNING_KEYS[method])
    if unknown:
        raise ValueError('unknown %s tuning keys: %s' % (method, ', '.join(unknown)))
    return merged


class _EngineeringController:
    def __init__(self, method, axis, cfg, origin, index):
        self.method, self.axis, self.Ts = method, axis, SAMPLE_TIME_S
        self.actual_limit = _number(cfg['actual_level_limit'], 'actual_level_limit', strictly_positive=True)
        if self.actual_limit > 90. or self.actual_limit != round(self.actual_limit):
            raise ValueError('actual_level_limit must be an integer in [1,90]')
        model = cfg['identified'][axis]['model']
        self.K = _number(model['K'], 'identified K', strictly_positive=True)
        self.tau = _number(model['tau'], 'identified tau', strictly_positive=True)
        self.Td = _number(model['Td'], 'identified Td', minimum=0.)
        self.model = AxisModel(self.K, self.tau, self.Td, self.Ts, cancel_drift=False)
        self.u_hist = deque([0.] * max(1, self.model.nd), maxlen=max(1, self.model.nd))
        self.mpc = SimpleNamespace(Np=20)
        self.options = _options(cfg, method, axis)
        self._pending = None
        self.previous_current_rate = self.previous_reference_time = None
        self.reference_acceleration = 0.
        self.reference_acceleration_source = 'not_used'
        self.n_step = 0
        self.requested = self.sent = self.unsaturated = self.quantization_residual = 0.
        self.controller_settings = dict(method=method, axis=axis, sample_time_s=self.Ts,
            identified_model=dict(K=self.K, tau=self.tau, Td=self.Td),
            input_delay_samples=int(self.model.nd), represented_delay_s=self.model.nd*self.Ts,
            initial_command_history=[0.]*max(1, self.model.nd),
            actual_level_limit=self.actual_limit, reference_horizon=20,
            origin_deg=_number(origin, 'origin'), coordinate_frame='origin-relative degrees',
            units=dict(angle='deg', rate='deg/s', input='SDK speed level'),
            plant='q_dot=w; tau*w_dot=-w+K*u(t-Td)',
            observer_enabled=method == 'LADRC', learned_residual=False,
            periodic_prediction=False, linear_lead=False,
            external_eps_raw_used=False, external_d_L_used=False,
            commit_input='actual integer controller contribution; excludes unmeasured injection',
            delay_rounding='nearest integer Ts, matching AxisModel',
            current_reference_required=method != 'MPC')
        if method != 'MPC':
            self.acceleration_feedforward = self.options.get('acceleration_feedforward', method == 'LADRC')
            if not isinstance(self.acceleration_feedforward, bool):
                raise ValueError('acceleration_feedforward must be boolean')
            self.controller_settings.update(acceleration_feedforward=self.acceleration_feedforward,
                reference_acceleration_rule='supplied current analytic acceleration preferred; else backward difference of committed current_rate over elapsed t, first zero; Ts fallback only if time absent',
                reference_acceleration_uses_measured_rate=False)

    def _inputs(self, theta, omega, th_ref, om_ref, current_ref, current_rate):
        if self._pending is not None:
            raise RuntimeError('previous command was not committed after sending')
        theta, omega = _number(theta, 'theta'), _number(omega, 'omega')
        r, dr = np.asarray(th_ref, dtype=float), np.asarray(om_ref, dtype=float)
        if r.shape != (20,) or dr.shape != (20,) or not np.isfinite(r).all() or not np.isfinite(dr).all():
            raise ValueError('finite reference horizons of length 20 required')
        if self.method != 'MPC':
            if current_ref is None or current_rate is None:
                raise ValueError('current_ref and current_rate at measurement t are required')
            current_ref = _number(current_ref, 'current_ref')
            current_rate = _number(current_rate, 'current_rate')
        return theta, omega, r, dr, current_ref, current_rate

    def _request(self, unsaturated, **pending):
        unsaturated = _number(unsaturated, 'controller output')
        self.unsaturated = unsaturated
        self.requested = float(np.clip(unsaturated, -self.actual_limit, self.actual_limit))
        self._pending = dict(pending, unsaturated=unsaturated, requested=self.requested)
        return self.requested

    def _reference_acceleration(self, current_rate, t, supplied):
        current_time = None if t is None else _number(t, 'reference time')
        if supplied is not None:
            acceleration = _number(supplied, 'current_acceleration')
            source = 'supplied_current_reference_acceleration'
        elif self.previous_current_rate is None:
            acceleration, source = 0., 'first_current_rate_sample_zero'
        else:
            interval = self.Ts
            source = 'causal_current_rate_difference_fixed_Ts_time_unavailable'
            if current_time is not None and self.previous_reference_time is not None:
                interval = current_time-self.previous_reference_time
                if interval <= 0.:
                    raise ValueError('reference time must increase for causal acceleration fallback')
                source = 'causal_current_rate_difference_elapsed_t'
            acceleration = (current_rate-self.previous_current_rate)/interval
        self.reference_acceleration = float(acceleration)
        self.reference_acceleration_source = source
        return current_time

    def commit_sent(self, sent):
        if self._pending is None:
            raise RuntimeError('no pending command')
        sent = _number(sent, 'actual sent level')
        if abs(sent) > self.actual_limit or sent != round(sent):
            raise ValueError('sent input must be an integer within actual_level_limit')
        delayed = self.u_hist[0] if self.model.nd else sent
        self._advance(sent, delayed)
        self.quantization_residual = sent-self._pending['requested']
        self.sent = sent
        self.u_hist.append(sent)
        if 'current_rate' in self._pending:
            self.previous_current_rate = self._pending['current_rate']
            self.previous_reference_time = self._pending['reference_time']
        self.n_step += 1
        self._pending = None

    def _advance(self, sent, delayed):
        pass

    def diagnostics(self):
        return dict(method=self.method, controller_settings=copy.deepcopy(self.controller_settings),
            n_step=int(self.n_step), pending_command=self._pending is not None,
            requested_levels=float(self.requested), unsaturated_levels=float(self.unsaturated),
            actual_sent_levels=float(self.sent), quantization_residual=float(self.quantization_residual),
            saturated=bool(abs(self.unsaturated) > self.actual_limit),
            command_history=list(self.u_hist),
            input_lo=-self.actual_limit, input_hi=self.actual_limit)


class PIDController(_EngineeringController):
    """Feedback-only PID, rate-error derivative and quantized antiwindup."""
    def __init__(self, axis, cfg, origin, index):
        super().__init__('PID', axis, cfg, origin, index)
        options = self.options
        default_wc = min(6., .4/max(self.Td, self.Ts))
        wc = _number(options.get('closed_loop_rad_s', default_wc), 'closed_loop_rad_s', strictly_positive=True)
        damping = _number(options.get('damping_ratio', .9), 'damping_ratio', strictly_positive=True)
        pole_ratio = _number(options.get('integral_pole_ratio', .25), 'integral_pole_ratio', strictly_positive=True)
        integral_pole = pole_ratio*wc
        # Match tau*s^3+(1+K*Kd)*s^2+K*Kp*s+K*Ki to
        # tau*(s+integral_pole)*(s^2+2*damping*wc*s+wc^2).
        gains = dict(kp=self.tau*(wc*wc+2*damping*wc*integral_pole)/self.K,
            ki=self.tau*integral_pole*wc*wc/self.K,
            kd=max(0., (self.tau*(2*damping*wc+integral_pole)-1.)/self.K))
        self.kp, self.ki, self.kd = [_number(options.get(key, gains[key]), key, minimum=0.)
                                    for key in ('kp', 'ki', 'kd')]
        tracking_time = _number(options.get('antiwindup_time_s', max(self.tau, self.Td+self.Ts)),
                                'antiwindup_time_s', strictly_positive=True)
        self.aw_alpha = 1.-math.exp(-self.Ts/tracking_time)
        self.integral_limit = _number(options.get('integral_limit_levels', self.actual_limit),
                                      'integral_limit_levels', strictly_positive=True)
        if options.get('reference_feedforward', False) is not False or self.acceleration_feedforward is not False:
            raise ValueError('PID is feedback-only; reference/acceleration feedforward must be false')
        self.reference_feedforward = False
        self.integral = self.error = self.rate_error = self.p_term = self.d_term = self.feedforward = 0.
        self.antiwindup_tracking_error = 0.
        self.integral_used = 0.
        self.controller_settings.update(kp=self.kp, ki=self.ki, kd=self.kd,
            closed_loop_rad_s=wc, damping_ratio=damping, integral_pole_ratio=pole_ratio,
            initial_gain_design='identified lag plant, cubic pole matching; no-delay initial design',
            derivative='current_rate - calibrated omega; no finite difference of angle',
            controller_variant='plain_pid_v1', control_formula='Kp*error + integral_used + Kd*error_derivative',
            reference_feedforward=False, acceleration_feedforward=False,
            reference_acceleration_rule='not_used',
            feedforward_formula='0; no reference velocity or acceleration model feedforward',
            antiwindup='conditional integration + exponential actual-sent back-calculation at commit',
            quantization_tracking_deadband_levels=.5,
            antiwindup_time_s=tracking_time, antiwindup_alpha=self.aw_alpha,
            integral_limit_levels=self.integral_limit,
            configured_gain_overrides=[key for key in ('kp', 'ki', 'kd') if key in options])

    def step(self, theta, omega, th_ref, om_ref, t=None, eps_raw=0., d_L=0.,
             current_ref=None, current_rate=None, current_acceleration=None):
        theta, omega, r, dr, current_ref, current_rate = self._inputs(
            theta, omega, th_ref, om_ref, current_ref, current_rate)
        self.error = current_ref-theta
        self.rate_error = current_rate-omega
        self.p_term = self.kp*self.error
        self.d_term = self.kd*self.rate_error
        self.integral_used = self.integral
        # Reference acceleration and inverse-model reference commands are never
        # evaluated by plain PID. current_rate only forms the D tracking error.
        return self._request(self.p_term+self.d_term+self.integral_used, error=self.error)

    def _advance(self, sent, delayed):
        pending = self._pending
        if self.ki == 0.:
            self.integral = self.antiwindup_tracking_error = 0.
            return
        delta_i = self.Ts*self.ki*pending['error']
        if (pending['unsaturated'] >= self.actual_limit and delta_i > 0.) or (
                pending['unsaturated'] <= -self.actual_limit and delta_i < 0.):
            delta_i = 0.
        tracking_error = sent-pending['unsaturated']
        # Ordinary nearest-integer quantization is not actuator saturation.
        # Back-calculating sublevel errors continuously can cancel Ki*e and
        # create a permanent angle offset. Keep normal integral action in that
        # quantization cell; still use actual sent feedback for saturation or
        # any publication change larger than the normal half-level rounding.
        if abs(pending['unsaturated']) <= self.actual_limit and abs(sent-pending['requested']) <= .5:
            tracking_error = 0.
        self.antiwindup_tracking_error = float(tracking_error)
        self.integral = float(np.clip(self.integral+delta_i+
            self.aw_alpha*tracking_error, -self.integral_limit, self.integral_limit))

    def diagnostics(self):
        values = super().diagnostics()
        values.update(error_deg=float(self.error), rate_error_deg_s=float(self.rate_error),
            proportional_levels=float(self.p_term), derivative_levels=float(self.d_term),
            feedforward_levels=float(self.feedforward), integral_levels=float(self.integral),
            integral_used_levels=float(self.integral_used),
            antiwindup_tracking_error=float(self.antiwindup_tracking_error),
            reference_acceleration_deg_s2=float(self.reference_acceleration),
            reference_acceleration_source=self.reference_acceleration_source,
            integrator_timing='integral_levels is post-commit k+1; P,D,integral_used are decision k')
        return values


class LADRCController(_EngineeringController):
    """Outer angle P feedback and first-order rate LADRC with a linear ESO."""
    def __init__(self, axis, cfg, origin, index):
        super().__init__('LADRC', axis, cfg, origin, index)
        options = self.options
        self.angle_bandwidth = _number(options.get('angle_bandwidth_rad_s', min(6., .36/max(self.Td, self.Ts))),
            'angle_bandwidth_rad_s', strictly_positive=True)
        self.rate_bandwidth = _number(options.get('rate_bandwidth_rad_s', min(12., .72/max(self.Td, self.Ts))),
            'rate_bandwidth_rad_s', strictly_positive=True)
        observer_bandwidth = _number(options.get('observer_bandwidth_rad_s', 3.*self.rate_bandwidth),
            'observer_bandwidth_rad_s', strictly_positive=True)
        self.b0 = self.K/self.tau
        p = math.exp(-observer_bandwidth*self.Ts)
        self.eso_gain = np.array([1.-p*p, (1.-p)**2/self.Ts])
        self.eso_state = np.zeros(2)
        self.eso_prediction = np.zeros(2)
        self.eso_initialized = False
        self.observer_input_used = self.innovation = self.rate_target = 0.
        self.angle_error = self.rate_error = 0.
        self.controller_settings.update(b0=self.b0,
            angle_bandwidth_rad_s=self.angle_bandwidth, rate_bandwidth_rad_s=self.rate_bandwidth,
            observer_bandwidth_rad_s=observer_bandwidth,
            eso_order=2, eso_output='calibrated angular rate',
            eso_total_disturbance='angular acceleration, includes -omega/tau',
            eso_current_gain=self.eso_gain.tolist(), eso_error_poles=[p, p],
            eso_discretization='ZOH integrator augmentation + discrete current correction',
            eso_initialization='first measured calibrated rate; total disturbance zero',
            control_formula='w_target=current_rate+angle_bw*(current_ref-theta); u=(current_acceleration_if_enabled+rate_bw*(w_target-w_hat)-f_hat)/b0',
            observer_input_timing='commit(k) predicts k+1 using quantized u[k-nd]; step(k+1) corrects with w[k+1]',
            observer_source='https://arxiv.org/html/2211.07309v2#S3')

    def step(self, theta, omega, th_ref, om_ref, t=None, eps_raw=0., d_L=0.,
             current_ref=None, current_rate=None, current_acceleration=None):
        theta, omega, r, dr, current_ref, current_rate = self._inputs(
            theta, omega, th_ref, om_ref, current_ref, current_rate)
        reference_time = self._reference_acceleration(current_rate, t, current_acceleration)
        if not self.eso_initialized:
            self.eso_prediction = np.array([omega, 0.])
            self.eso_initialized = True
        self.innovation = omega-self.eso_prediction[0]
        self.eso_state = self.eso_prediction+self.eso_gain*self.innovation
        self.angle_error = current_ref-theta
        self.rate_target = current_rate+self.angle_bandwidth*self.angle_error
        self.rate_error = self.rate_target-self.eso_state[0]
        acceleration = self.reference_acceleration if self.acceleration_feedforward else 0.
        return self._request((acceleration+self.rate_bandwidth*self.rate_error-self.eso_state[1])/self.b0,
            current_rate=current_rate, reference_time=reference_time)

    def _advance(self, sent, delayed):
        self.observer_input_used = float(delayed)
        self.eso_prediction = self.eso_state.copy()
        self.eso_prediction[0] += self.Ts*(self.eso_state[1]+self.b0*delayed)
        if not np.isfinite(self.eso_prediction).all():
            raise FloatingPointError('nonfinite LADRC observer')

    def diagnostics(self):
        values = super().diagnostics()
        values.update(eso_current_state=self.eso_state.tolist(),
            eso_next_prediction=self.eso_prediction.tolist(),
            estimated_total_acceleration=float(self.eso_state[1]),
            observer_input_used=float(self.observer_input_used),
            rate_innovation_deg_s=float(self.innovation),
            rate_target_deg_s=float(self.rate_target), angle_error_deg=float(self.angle_error),
            rate_error_deg_s=float(self.rate_error),
            reference_acceleration_deg_s2=float(self.reference_acceleration),
            reference_acceleration_source=self.reference_acceleration_source)
        return values


class NominalMPCController(_EngineeringController):
    """Plain speed-interface nominal MPC, without disturbance estimation."""
    def __init__(self, axis, cfg, origin, index):
        super().__init__('MPC', axis, cfg, origin, index)
        from soft_mpc import SoftMPCdist
        options = self.options
        if options.get('Np', 20) != 20:
            raise ValueError('common comparison prediction horizon is 20')
        nc = options.get('Nc', 3)
        if isinstance(nc, bool) or int(nc) != nc or not 1 <= nc <= 5:
            raise ValueError('MPC Nc must be an integer in [1,5]')
        weights = dict(Np=20, Nc=int(nc),
            q_th=_number(options.get('q_th', 1.), 'q_th', strictly_positive=True),
            q_om=_number(options.get('q_om', 0.), 'q_om', minimum=0.),
            r=_number(options.get('r', cfg.get('mpc_r', 1e-4)), 'r', strictly_positive=True),
            p_term=_number(options.get('p_term', 1.), 'p_term', minimum=0.))
        radius = _number(cfg['mechanical_envelope_half_deg'][index], 'mechanical envelope', strictly_positive=True)
        soft = dict(theta_lo=float(origin)-radius, theta_hi=float(origin)+radius,
            theta_offset=float(origin),
            omega_lim=_number(cfg.get('angular_rate_limit_deg_s', 20.), 'rate limit', strictly_positive=True),
            fov_half=_number(cfg['mission_fov_half_deg'][index], 'FOV half angle', strictly_positive=True),
            penalty=_number(options.get('soft_penalty', 1000.), 'soft_penalty', strictly_positive=True))
        self.mpc = SoftMPCdist(self.model, soft_kw=soft, u_max=self.actual_limit, **weights)
        self.mpc_state = np.zeros(self.model.n)
        self.zero_disturbance = np.zeros(20)
        self.controller_settings.update(weights, soft_constraints=copy.deepcopy(soft),
            speed_interface_input_max=self.actual_limit,
            cancel_inner_loop_drift=False, disturbance_horizon='identically zero',
            model_discretization='exact ZOH first-order velocity lag + integer input delay',
            prediction_input_history='actual quantized controller commands',
            objective='angle/rate quadratic tracking, input quadratic effort, terminal tracking weight',
            constraint_solver='existing SoftMPCdist input-hard/state-soft with L1 epigraph penalty')

    def step(self, theta, omega, th_ref, om_ref, t=None, eps_raw=0., d_L=0.,
             current_ref=None, current_rate=None, current_acceleration=None):
        theta, omega, r, dr, current_ref, current_rate = self._inputs(
            theta, omega, th_ref, om_ref, current_ref, current_rate)
        self.mpc_state[:2] = theta, omega
        for i in range(self.model.nd):
            self.mpc_state[2+i] = self.u_hist[-1-i]
        requested = self.mpc.solve_d(self.mpc_state, r, dr, self.zero_disturbance,
                                      -self.actual_limit, self.actual_limit)
        return self._request(requested)

    def diagnostics(self):
        values = super().diagnostics()
        values.update(mpc_state=self.mpc_state.tolist(), disturbance_horizon=self.zero_disturbance.tolist())
        values.update(self.mpc.last_diag)
        return values


def create_engineering_controller(method, axis, cfg, origin_j, index):
    """Build one fresh independent controller; cfg is never mutated."""
    if method not in ENGINEERING_METHODS:
        raise ValueError('unknown engineering method')
    if axis not in ('yaw', 'pitch') or index != ('yaw', 'pitch').index(axis):
        raise ValueError('axis/index mismatch')
    classes = dict(PID=PIDController, LADRC=LADRCController, MPC=NominalMPCController)
    return classes[method](axis, cfg, origin_j, index)
