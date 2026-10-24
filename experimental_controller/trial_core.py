"""Shared frozen method protocol, complete balanced blocks and hover guard."""
import copy,itertools,math,random
import numpy as np
from ground_proposed import DualProposed

ENGINEERING_METHODS=('PID','LADRC','MPC')
METHODS=('M2','M3a','NoLearn','NoDrift','Proposed','GRU')+ENGINEERING_METHODS
LABELS={'M2':'M2：MPC＋线性DO','M3a':'M3a：原始APDO当前值＋线性DO',
        'NoLearn':'Proposed：关闭学习残差','NoDrift':'Proposed：关闭频漂权重','Proposed':'完整 Proposed',
        'PID':'朴素PID：无前馈＋抗积分饱和','LADRC':'LADRC：线性ESO','MPC':'MPC：无扰动观测器',
        'GRU':'Proposed：LSTM 换为 GRU'}
SCENARIOS=('none','constant','switch')
ORDER_SEED=20261023
COMPARISONS={
    'hover_baseline_comparison':('NoLearn','PID','LADRC','Proposed'),
    'baseline_neural_comparison':('PID','LADRC','GRU','Proposed'),
    'comparison':('M2','NoLearn','Proposed'),
    'ground_comparison':('M2','M3a','NoLearn','Proposed'),
    'ground_switch_comparison':('M2','M3a','NoLearn','NoDrift','Proposed'),
    'carrier_comparison':('M2','Proposed'),
    'engineering_comparison':('PID','LADRC','MPC','M2','Proposed'),
    'full_ground_comparison':('PID','LADRC','MPC','M2','NoLearn','M3a','Proposed'),
    'full_ground_switch_comparison':('PID','LADRC','MPC','M2','NoLearn','M3a','NoDrift','Proposed'),
}

def method_settings(method):
    """JSON-safe behavior declarations for round manifests and diagnostics."""
    if method not in METHODS:raise ValueError('unknown method')
    if method in ENGINEERING_METHODS:
        settings=dict(controller_family=method,periodic_prediction=False,linear_lead=False,learned_residual=False,
            drift_weight=0.,periodic_observer='none',disturbance_observer='linear_eso' if method=='LADRC' else 'none',
            current_value_periodic_compensation=False,apdo_n_ahead=None,mpc_enabled=method=='MPC',
            tuning_status='model_initialized_pending_independent_ground_tuning_and_acceptance')
        if method=='PID':settings.update(controller_variant='plain_pid_v1',reference_feedforward=False,acceleration_feedforward=False)
        return settings
    prediction=method in ('NoLearn','NoDrift','Proposed','GRU')
    settings=dict(periodic_prediction=prediction,linear_lead=prediction,
        learned_residual=method in ('NoDrift','Proposed','GRU'),
        drift_weight=0. if method in ('NoDrift','M3a') else 4.,
        periodic_observer='original_apdo_current_value' if method=='M3a' else 'kooplin_harmonic_rls' if prediction else 'none',
        current_value_periodic_compensation=method=='M3a',
        apdo_n_ahead=0 if method=='M3a' else None)
    if method=='GRU':settings['network_architecture']='GRU'
    return settings

def build_suite(stage,selection,scenario,repeats=3,task='track'):
    if stage not in ('ground','hover'):raise ValueError('unknown stage')
    if scenario not in SCENARIOS:raise ValueError('unknown scenario')
    if stage=='hover' and scenario!='none':raise ValueError('hover first-stage tests do not permit active input injection')
    if selection not in ('single',)+tuple(COMPARISONS)+METHODS:raise ValueError('unknown selection')
    if isinstance(repeats,bool) or repeats not in (1,3,6):raise ValueError('repeat count must be 1, 3 or 6')
    if task not in ('hold','track'):raise ValueError('unknown gimbal task')
    if selection in ('ground_comparison','ground_switch_comparison','M3a','NoDrift','engineering_comparison',
            'full_ground_comparison','full_ground_switch_comparison','baseline_neural_comparison','GRU','MPC') and stage!='ground':
        raise ValueError('selected comparison is ground only')
    if selection=='hover_baseline_comparison' and scenario!='none':raise ValueError('hover baseline comparison requires no active injection')
    if selection in ('ground_switch_comparison','full_ground_switch_comparison') and scenario!='switch':
        raise ValueError('switch comparison requires ground/switch')
    if selection=='carrier_comparison' and stage!='hover':
        raise ValueError('carrier_comparison requires airborne stage mapped to hover')
    methods=list(COMPARISONS[selection]) if selection in COMPARISONS else ['Proposed' if selection=='single' else selection]
    rng=random.Random(ORDER_SEED);runs=[]
    if len(methods)==3 and repeats==6:
        blocks=[list(p) for p in itertools.permutations(methods)];rng.shuffle(blocks)
    elif len(methods)>3 and repeats==6:
        blocks=[]
        for _ in range(3):
            order=methods.copy();rng.shuffle(order);blocks.extend([order,order[::-1]])
        rng.shuffle(blocks)
    else:
        base=methods.copy();rng.shuffle(base)
        blocks=[base[i%len(base):]+base[:i%len(base)] for i in range(repeats)]
        rng.shuffle(blocks)
    for rep,block in enumerate(blocks,1):
        for position,method in enumerate(block,1):
            runs.append(dict(method=method,stage=stage,scenario=scenario,repeat=rep,duration_s=30,task=task,
                order_seed=ORDER_SEED,block_position=position,block_methods=block.copy()))
    return runs

LEGACY_INJECTION=dict(fundamental_levels=2.,second_harmonic_levels=0.,step_levels=1.,base_hz=.5,after_switch_hz=.8,switch_time_s=15.)


def injection_bound(settings):
    value=sum(float(settings[k]) for k in ('fundamental_levels','second_harmonic_levels','step_levels'))
    if not np.isfinite(value) or any(float(settings[k])<0 for k in ('fundamental_levels','second_harmonic_levels','step_levels')):
        raise ValueError('invalid injection amplitudes')
    return int(math.ceil(value))


def injection(t,scenario,settings=None):
    if scenario=='none':return [0,0]
    if scenario not in SCENARIOS:raise ValueError('unknown injection')
    taper=min(1.,max(0.,t/2),max(0.,(30-t)/2))
    # 1.6x switch, matching the manuscript ratio and staying in the frozen
    # KoopLin admissible band (.445--.885 Hz at f_nom=.5). Hardware switches
    # at 15 s to retain a longer post-switch window within each 30-s round.
    s=LEGACY_INJECTION if settings is None else settings
    switch=float(s['switch_time_s']);base=float(s['base_hz'])
    phase=2*math.pi*(base*min(t,switch)+(float(s['after_switch_hz']) if scenario=='switch' else base)*max(t-switch,0))
    aper=float(s['step_levels'])*(1. if 8<=t<12 else -1. if 19<=t<24 else 0.)
    return [int(round(taper*(float(s['fundamental_levels'])*math.sin(phase+p)+
        float(s['second_harmonic_levels'])*math.sin(2*phase+p+.3)+aper))) for p in (0.,.7)]

def wire_command(control,disturbance,limit):
    u=np.asarray(control)+np.asarray(disturbance)
    if not np.isfinite(u).all() or np.max(abs(u))>limit or np.any(u!=np.rint(u)):
        raise RuntimeError('final wire command outside bounded integer range')
    return u.astype(int).tolist()

def task_config(cfg,task,scenario='none'):
    """Resolve the shared tracking envelope without widening hold operation."""
    if task not in ('hold','track'):raise ValueError('unknown gimbal task')
    inner=copy.deepcopy(cfg)
    if task=='hold':inner['reference_amplitude_deg']=[0.,0.]
    if task=='track' or scenario!='none':
        overrides=inner.get('tracking_limits',{})
        for key in ('actual_level_limit','mechanical_envelope_half_deg','angular_rate_limit_deg_s','nominal_input_max'):
            if key in overrides:inner[key]=copy.deepcopy(overrides[key])
        if 'tracking_injection' in inner:inner['injection']=copy.deepcopy(inner['tracking_injection'])
    return inner


class TrialCore:
    def __init__(self,cfg,origin,method='Proposed',scenario='none',task='track',network_factory=None):
        if method not in METHODS or scenario not in SCENARIOS:raise ValueError('invalid method/scenario')
        inner=task_config(cfg,task,scenario)
        self.method,self.scenario=method,scenario;self.wire_limit=inner['actual_level_limit']
        self.injection_settings=copy.deepcopy(inner.get('injection',LEGACY_INJECTION))
        self.injection_bound=injection_bound(self.injection_settings)
        self.control_reserve=self.injection_bound if scenario!='none' else 0
        if self.control_reserve>=self.wire_limit:raise ValueError('injection leaves no control headroom')
        if scenario!='none':inner['actual_level_limit']=self.wire_limit-self.control_reserve
        if method in ENGINEERING_METHODS:
            from engineering_baselines import create_engineering_controller
            self.core=DualProposed(inner,origin,controller_factory=lambda axis,cfg,origin_j,index:
                create_engineering_controller(method,axis,cfg,origin_j,index))
        else:
            if method=='GRU':
                if network_factory is None:
                    from gru_network import GRUNetwork
                    network_factory=GRUNetwork
                self.core=DualProposed(inner,origin,network_factory=network_factory,
                    network_weights=inner.get('gru_weights','paper40_gru_real_finetuned.npz'),network_architecture='GRU')
            else:self.core=DualProposed(inner,origin)
        if method=='M2':
            for c in self.core.ctrl:c.pred=False;c.lead='none'
        elif method=='NoDrift':
            for c in self.core.ctrl:c.drift=0.
        elif method=='M3a':
            from current_value_m3a import CurrentValueM3a
            self.core.ctrl=[CurrentValueM3a(c) for c in self.core.ctrl]

    def prepare(self,smp,t,T=30.,reference_func=None):
        used=None if self.method in ('Proposed','NoDrift','GRU') else [0.,0.]
        p=self.core.prepare(smp,t,T,eps_override=used,reference_func=reference_func)
        p['network_used']=None if self.method in ENGINEERING_METHODS else p['raw'] if used is None else used
        p['injection']=injection(t,self.scenario,self.injection_settings)
        p['wire_sent']=wire_command(p['sent'],p['injection'],self.wire_limit)
        return p

    def commit(self,packet):
        # Input injection represents an unmeasured virtual plant disturbance.
        # The observer sees the actual quantized CONTROLLER contribution only;
        # final SDK wire levels are separately logged and hard-bounded.
        return self.core.commit(packet)

def validate_hover(s,anchor_xy=None,anchor_z=None,locked_mode=None):
    if s.get('armed') is not True or not s.get('connected') or s.get('state_age',99)>2:
        raise RuntimeError('hover requires fresh connected armed flight state')
    if s.get('mode') not in ('LOITER','GUIDED') or (locked_mode and s['mode']!=locked_mode):
        raise RuntimeError('pilot mode changed or not position-hold mode')
    if s.get('pose_age',99)>.3 or s.get('imu_age',99)>.2:
        raise RuntimeError('hover pose/IMU stale')
    if s.get('rc_age',99)>1.5:raise RuntimeError('pilot RC feedback stale')
    if s.get('battery_age',99)>5 or (s.get('vcell') or 0)<3.65:
        raise RuntimeError('hover battery low or stale')
    if not s.get('vg') or s.get('vg_age',99)>2 or s.get('ego') is not False:
        raise RuntimeError('position input unhealthy or trajectory bridge enabled')
    vals=np.r_[s.get('xy',[np.nan,np.nan]),s.get('z',np.nan),s.get('rpy',[np.nan]*3)]
    if not np.isfinite(vals).all():raise RuntimeError('hover state nonfinite')
    if not .6<=s['z']<=1.6:raise RuntimeError('hover local height outside 0.6–1.6 m')
    if max(abs(np.asarray(s['rpy'][:2])))>12:raise RuntimeError('hover roll/pitch exceeds 12 deg')
    if anchor_xy is not None and np.linalg.norm(np.array(s['xy'])-anchor_xy)>.5:
        raise RuntimeError('hover horizontal drift >0.5 m')
    if anchor_z is not None and abs(s['z']-anchor_z)>.3:raise RuntimeError('hover height drift >0.3 m')
