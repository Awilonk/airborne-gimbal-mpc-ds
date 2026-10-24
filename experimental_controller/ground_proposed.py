"""Proposed only, both axes, 30 seconds, disarmed ground entry point."""
import os
for _k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[_k]='1'
import argparse,hashlib,json,time,subprocess
from pathlib import Path
import numpy as np
from ctrl_m5 import M5Controller
from zr10_ctrl import AXIS_ID
from paper_network import PaperNetwork
from durable_log import DurableRun
from ground_io import GroundLink,PerformanceLease
from reference_trajectory import AngularReference,envelope_and_rate

HERE=Path(__file__).resolve().parent

def env(t,T):
    return envelope_and_rate(t,T)[0]

class DualProposed:
    def __init__(self,cfg,origin,controller_factory=None,network_factory=None,network_weights=None,network_architecture='LSTM'):
        self.cfg=cfg;self.origin=np.asarray(origin,float);self.prev=np.zeros(2)
        self.rate_limit=float(cfg.get('angular_rate_limit_deg_s',20.))
        if not np.isfinite(self.rate_limit) or self.rate_limit<=0:raise ValueError('positive angular rate limit required')
        self.reference=AngularReference(cfg['reference_amplitude_deg'],cfg['reference_w_rad_s'],cfg.get('reference_phase_rad',[0.,0.]))
        from angle_feedback import make_angle_feedback
        self.feedback=make_angle_feedback(cfg)
        self.network_architecture=network_architecture if controller_factory is None else None
        self.network_weights=(network_weights or cfg['weights']) if controller_factory is None else None
        self.net=(network_factory or PaperNetwork)(HERE/self.network_weights) if controller_factory is None else None
        self.network_weight_sha256=None
        if self.net is not None:
            self.network_weight_sha256=getattr(self.net,'weight_sha256',None)
            if self.network_weight_sha256 is None and (HERE/self.network_weights).is_file():
                self.network_weight_sha256=hashlib.sha256((HERE/self.network_weights).read_bytes()).hexdigest()
        self.ctrl=[]
        self.native_current_reference=controller_factory is not None
        for j,ax in enumerate(('yaw','pitch')):
            if controller_factory is not None:
                self.ctrl.append(controller_factory(ax,cfg,float(self.origin[j]),j));continue
            p=cfg['identified'][ax]
            AXIS_ID[ax].update({k:p['model'][k] for k in ('K','tau','Td')})
            rad=cfg['mechanical_envelope_half_deg'][j]
            soft=dict(theta_lo=self.origin[j]-rad,theta_hi=self.origin[j]+rad,
                theta_offset=self.origin[j],omega_lim=self.rate_limit,fov_half=cfg['mission_fov_half_deg'][j],penalty=1000.)
            self.ctrl.append(M5Controller(ax,.02,wo=5,mpc_kw=dict(Np=20,Nc=3,r=cfg.get('mpc_r',1e-4),q_om=0.),
                soft_kw=soft,eps_max=cfg['eps_max'][j],eps_rate=cfg['eps_rate'][j],D_max=6.,lead='external40',
                actual_limit=cfg.get('actual_level_limit',45),u_max=cfg.get('nominal_input_max',50.)))

    def prepare(self,smp,t,T,eps_override=None,reference_func=None):
        q=np.asarray(smp['cmd_frame'][:2],float)
        if abs(q[0])>155:raise RuntimeError('yaw approaching verified 160 degree stop')
        rel=(q-self.origin+180)%360-180
        w=np.array([smp['rate_raw'][j]*self.cfg['identified'][ax]['rate_calibration']['scale']+
                    self.cfg['identified'][ax]['rate_calibration']['bias'] for j,ax in enumerate(('yaw','pitch'))])
        if not (np.isfinite(q).all() and np.isfinite(rel).all() and np.isfinite(w).all()):
            raise RuntimeError('nonfinite gimbal angle/rate feedback')
        if np.any(abs(rel)>np.asarray(self.cfg['mechanical_envelope_half_deg'])):
            raise RuntimeError('ground excursion outside configured envelope')
        if np.any(abs(w)>self.rate_limit):raise RuntimeError('ground angular rate outside envelope')
        feedback=None;control_w=w
        if self.feedback is not None:
            estimates=[observer.update(float(angle)) for observer,angle in zip(self.feedback,rel)]
            control_w=np.asarray([state[1] for state in estimates])
            # The raw-rate guard above is never replaced by a conditioned rate.
            if np.any(abs(control_w)>self.rate_limit):raise RuntimeError('estimated angular rate outside envelope')
            feedback=[observer.diagnostics() for observer in self.feedback]
        raw=self.net.step(q,w,self.prev) if self.net is not None else None;requests=[];refs=[]
        external=None
        if reference_func is not None:
            external=[]
            for tt in np.r_[t,t+.02*np.arange(1,21)]:
                absolute=np.asarray(reference_func(float(tt)),float)
                value=(absolute-self.origin+180)%360-180
                if not np.isfinite(value).all() or np.any(abs(value)>np.asarray(self.cfg['mechanical_envelope_half_deg'])):
                    raise RuntimeError('inertial pointing reference outside verified angular envelope')
                external.append(value)
            external=np.asarray(external)
        if external is None:
            fut=t+.02*np.arange(1,21)
            references=self.reference.value(fut,T);derivatives=self.reference.rate(fut,T)
            current_reference=self.reference.value(t,T)
            if self.native_current_reference:
                current_rate=self.reference.rate(t,T);current_acceleration=self.reference.acceleration(t,T)
        else:
            gradient=np.gradient(external,.02,axis=0,edge_order=2)
            references=external[1:];derivatives=gradient[1:]
            current_reference=external[0]
            if self.native_current_reference:
                current_rate=gradient[0];current_acceleration=np.gradient(gradient,.02,axis=0,edge_order=2)[0]
        for j,c in enumerate(self.ctrl):
            R=references[:,j];dR=derivatives[:,j];current=float(current_reference[j])
            eps=(raw[j] if raw is not None else 0.) if eps_override is None else eps_override[j]
            native=dict(current_ref=current,current_rate=float(current_rate[j]),
                current_acceleration=float(current_acceleration[j])) if self.native_current_reference else {}
            requests.append(c.step(rel[j],control_w[j],R,dR,t=t,eps_raw=eps,**native));refs.append(current)
        sent=np.rint(requests).astype(int)
        # Snapshot immediately after inference; later inference/commit cannot mutate the decision record.
        network_state=None if self.net is None else dict(architecture=self.network_architecture,
            hidden_state=self.net.h.tolist(),cell_state=self.net.c.tolist() if hasattr(self.net,'c') else None,
            weight_sha256=self.network_weight_sha256)
        return dict(q=q.tolist(),rel=rel.tolist(),omega=w.tolist(),raw=raw.tolist() if raw is not None else None,
                    omega_control=control_w.tolist(),feedback=feedback,
                    requested=list(map(float,requests)),sent=sent.tolist(),ref=refs,
                    network_state=network_state,network_projected=[float(c.lead_eps) for c in self.ctrl] if self.net is not None else None,
                    psi40=self.net.last_psi.tolist() if self.net is not None else None)

    def commit(self,packet):
        for c,u in zip(self.ctrl,packet['sent']):c.commit_sent(u)
        if self.feedback is not None:
            for observer,u in zip(self.feedback,packet['sent']):observer.commit_sent(u)
        self.prev=np.array(packet['sent'],float)
        diagnostics=[c.diagnostics() for c in self.ctrl]
        for j,diag in enumerate(diagnostics):
            diag['measurement_feedback']=packet.get('feedback')[j] if packet.get('feedback') else None
        return diagnostics

FIELDS=['t','wall_time','dev_seq','device_dt_s','sample_age_ms','q_abs','q_rel','omega','ref',
        'u_requested','u_sent','network_raw','psi40','calc_ms','armed','mode','imu','diag_yaw','diag_pitch']

def make_row(t,smp,packet,diag,calc,armed,mode,imu=None):
    return dict(t=t,wall_time=time.time(),dev_seq=smp['dev_seq'],device_dt_s=smp.get('dt'),
        sample_age_ms=1000*(time.monotonic()-smp.get('t_mono',time.monotonic())),
        q_abs=packet['q'],q_rel=packet['rel'],omega=packet['omega'],ref=packet['ref'],
        u_requested=packet['requested'],u_sent=packet['sent'],network_raw=packet['raw'],psi40=packet['psi40'],
        calc_ms=calc,armed=armed,mode=mode,imu=imu,diag_yaw=diag[0],diag_pitch=diag[1])

def summarize(rows,T,reason,simulated=False,actual_limit=45):
    r=[v for v in rows if 5<=v['t']<=T-2]
    errs=np.array([np.array(v['q_rel'])-v['ref'] for v in r]) if r else None
    rms=np.sqrt(np.mean(errs**2,axis=0)).tolist() if r else [None,None]
    calcs=[v['calc_ms'] for v in rows] or [0]
    return dict(method='Proposed-Paper40',mode='M5',axis='both',tag='Proposed-Paper40-real',
        stage='software_simulation' if simulated else 'ground_test',simulated=simulated,
        aborted=bool(reason),reason=reason,n=len(rows),t_end=rows[-1]['t'] if rows else 0.,dur=T,
        evaluation_samples=len(r),rmse=rms[0],rmse_yaw=rms[0],
        rmse_pitch=rms[1],max_error=np.max(abs(errs),axis=0).tolist() if r else [None,None],
        calc_p95=float(np.quantile(calcs,.95)),calc_max=float(max(calcs)),
        sat=float(np.mean([max(abs(np.array(v['u_sent'])))>=actual_limit for v in rows])) if rows else 0.,
        actual_level_limit=actual_limit,
        max_actual_levels=np.max(abs(np.array([v['u_sent'] for v in rows])),axis=0).tolist() if rows else [0,0],
        max_abs_delta=[max([abs(v[k]['lead_d']) for v in rows if isinstance(v[k].get('lead_d'),(int,float))],default=None) for k in ('diag_yaw','diag_pitch')],
        min_covariance=[min([v[k]['cov_min'] for v in rows if isinstance(v[k].get('cov_min'),(int,float))],default=None) for k in ('diag_yaw','diag_pitch')],
        video_off_during_real_test=not simulated,flight_commands_sent=False)

def selftest(cfg,root,T):
    from sim_zr10 import Plant,CAL_REAL
    origin=[115.,0.];core=DualProposed(cfg,origin)
    plants=[]
    for j,ax in enumerate(('yaw','pitch')):
        p=cfg['identified'][ax]['model'];oldK=.805 if j==0 else .873
        cal=dict(lag2=.015,th_noise=.04,om_noise=.12,om_scale=1.,K_scale=p['K']/oldK,
                 tau=p['tau'],Td=p['Td'],dz=p['deadzone'])
        pl=Plant(ax,np.random.default_rng(100+j),cal=cal);pl.th=origin[j];plants.append(pl)
    log=DurableRun(root,dict(stage='software_simulation',config=cfg),FIELDS);rows=[];reason=''
    try:
        for k in range(int(T/.02)):
            t=k*.02;q=[];rates=[]
            for j,ax in enumerate(('yaw','pitch')):
                th,om=plants[j].meas();cal=cfg['identified'][ax]['rate_calibration']
                q.append(th);rates.append(int(round((om-cal['bias'])/cal['scale'])))
            smp=dict(cmd_frame=q+[0],rate_raw=rates+[0],dev_seq=k,dt=.02,t_mono=time.monotonic())
            start=time.perf_counter();packet=core.prepare(smp,t,T);diag=core.commit(packet)
            calc=(time.perf_counter()-start)*1000
            row=make_row(t,smp,packet,diag,calc,False,'SIMULATED');rows.append(row);log.submit(row)
            for j,p in enumerate(plants):
                p.cmd(t+.002,packet['sent'][j]);p.run(t,t+.02,lambda tt,ww:0.)
    except BaseException as ex:reason=repr(ex)
    summary=summarize(rows,T,reason,True,cfg.get('actual_level_limit',45));log.close(summary)
    print('SELFTEST '+json.dumps(dict(summary,out=str(log.path))),flush=True)
    return 2 if reason else 0

def real_test(cfg,root):
    rows=[];reason='';T=30.
    # Prioritize this short, sleeping userspace loop under the live mapping load.
    # No realtime scheduler and no global CPU/governor changes.
    scheduling=subprocess.run(['sudo','-n','/usr/bin/renice','-n','-10','-p',str(os.getpid())],
        capture_output=True,text=True,timeout=3)
    if scheduling.returncode or os.getpriority(os.PRIO_PROCESS,0)!=-10:
        raise RuntimeError('cannot establish experiment process scheduling priority')
    with PerformanceLease(),GroundLink(level_limit=cfg.get('actual_level_limit',45)) as io:
        from ground_center import center_if_needed
        centered=center_if_needed(io,cfg,str(Path(root)/'preparation'))
        origin=io.sample['cmd_frame'][:2]
        # Prewarm on a separate instance, preserving the real observer initial state.
        warm=DualProposed(cfg,origin)
        smp=dict(io.sample)
        for k in range(10):p=warm.prepare(smp,0.,T);warm.commit(p)
        core=DualProposed(cfg,origin)
        meta=dict(stage='ground_test',method='Proposed-Paper40-real',config=cfg,origin_deg=origin,nice=-10,cpu_governor='performance_temporarily',
             video_service_stopped=io.cam.stopped,flight_commands=False,centering_log=centered,
             units='angles deg; rates deg/s; inputs integer speed levels; trig radians',
             camera_fov_calibrated=False,mechanical_envelope_is_selected_test_region=True,
             weight_sha256=hashlib.sha256((HERE/cfg['weights']).read_bytes()).hexdigest(),
             sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')})
        log=DurableRun(root,meta,FIELDS,sync_period=.2)
        print('GROUND_START '+json.dumps(dict(out=str(log.path),origin=origin,video_off=io.cam.stopped)),flush=True)
        io.telemetry({'start':dict(axis='both',mode='M5',tag='Proposed-Paper40-real',amp=3,w=.8,dur=30,
                                   hover=False,inj=False,out=str(log.path))})
        start=time.monotonic();last=io.sample['dev_seq']
        try:
            while True:
                smp=io.next(last);last=smp['dev_seq'];t=time.monotonic()-start
                if t>=T:break
                tick=time.perf_counter();cpu_tick=time.thread_time();packet=core.prepare(smp,t,T)
                prepared_ms=(time.perf_counter()-tick)*1000
                if prepared_ms>cfg['compute_limit_ms']:
                    raise RuntimeError('control computation deadline exceeded before send: wall %.3f ms, CPU %.3f ms'%(prepared_ms,1000*(time.thread_time()-cpu_tick)))
                log.assert_healthy();io.send(*packet['sent']);diag=core.commit(packet)
                calc=(time.perf_counter()-tick)*1000
                row=make_row(t,smp,packet,diag,calc,io.state.armed,io.state.mode,io.imu)
                log.submit(row);rows.append(row)
                io.telemetry(dict(t=round(t,3),th=packet['rel'][0],ref=packet['ref'][0],
                    pitch=packet['rel'][1],ref_pitch=packet['ref'][1],u=packet['sent'][0],u_pitch=packet['sent'][1],
                    lv=packet['sent'][0],dh=diag[0]['d_h_raw'],dr=diag[0]['d_r'],f=diag[0]['f_hat'],
                    calc=calc,age=row['sample_age_ms'],sat=int(max(abs(np.array(packet['sent'])))>=cfg.get('actual_level_limit',45)),
                    fm=io.state.mode,arm=io.state.armed))
                if len(rows)%250==0:print('GROUND_PROGRESS',round(t,1),len(rows),flush=True)
        except BaseException as ex:reason=repr(ex)
        finally:
            io.stop();summary=summarize(rows,T,reason,False,cfg.get('actual_level_limit',45));log.close(summary)
            io.telemetry({'summary':summary});print('SUMMARY '+json.dumps(dict(summary,out=str(log.path))),flush=True)
    return 2 if reason else 0

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',default=str(HERE/'proposed_config.json'))
    ap.add_argument('--root',default='/home/ak/ast_proposed_ground_runs_20261023')
    ap.add_argument('--selftest',action='store_true');ap.add_argument('--selftest-seconds',type=float,default=60.)
    a=ap.parse_args();cfg=json.loads(Path(a.config).read_text(encoding='utf-8'))
    return selftest(cfg,a.root,a.selftest_seconds) if a.selftest else real_test(cfg,a.root)

if __name__=='__main__':raise SystemExit(main())
