"""Bounded slow retreat from the observed/vendor +/-160 degree yaw stops."""
import time
from durable_log import DurableRun

def retreat_level(q,start):
    if abs(q-start)>25:raise RuntimeError('centering excursion exceeds 25 degrees')
    if abs(q)>161:raise RuntimeError('yaw outside verified SDK range')
    if start>145:return -6 if q>140 else 0
    if start< -145:return 6 if q< -140 else 0
    return 0

def center_if_needed(io,cfg,root):
    start=float(io.sample['cmd_frame'][0])
    if abs(start)<=145:return None
    log=DurableRun(root,dict(stage='ground_centering',origin=start,target=140 if start>0 else -140,
        video_service_stopped=io.cam.stopped,flight_commands=False),['t','sample','u_yaw','u_pitch'])
    print('GROUND_CENTERING from %.1f toward %.1f deg, speed level 6'%(start,140 if start>0 else -140),flush=True)
    tick=time.monotonic();last=io.sample['dev_seq'];reason=''
    try:
        while True:
            smp=io.next(last);last=smp['dev_seq'];t=time.monotonic()-tick
            if t>18:raise RuntimeError('slow yaw centering timed out')
            q=float(smp['cmd_frame'][0]);u=retreat_level(q,start)
            cal=cfg['identified']['yaw']['rate_calibration']
            if abs(smp['rate_raw'][0]*cal['scale']+cal['bias'])>8:
                raise RuntimeError('centering angular rate >8 deg/s')
            log.assert_healthy();io.send(u,0);log.submit(dict(t=t,sample=smp,u_yaw=u,u_pitch=0))
            if u==0:break
    except BaseException as ex:reason=repr(ex);raise
    finally:
        io.stop();log.close(dict(stage='ground_centering',aborted=bool(reason),reason=reason))
    time.sleep(.25);io.last_consumed=None
    print('GROUND_CENTERING_DONE %.1f deg'%io.sample['cmd_frame'][0],flush=True)
    return str(log.path)
