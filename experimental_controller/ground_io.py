"""Ground-only gimbal IO. There are no flight publishers/services in this module."""
import json,math,os,signal,socket,subprocess,threading,time
from pathlib import Path
from device_sequence import device_sequence_delta

class PerformanceLease:
    """Temporary CPU performance governor, restored on every normal/error exit."""
    def __init__(self):self.saved={}
    def __enter__(self):
        try:
            for p in sorted(Path('/sys/devices/system/cpu/cpufreq').glob('policy*/scaling_governor')):
                old=p.read_text().strip()
                if old!='performance':
                    self.saved[p]=old
                    subprocess.run(['sudo','-n','/usr/bin/tee',str(p)],input='performance\n',
                        text=True,stdout=subprocess.DEVNULL,check=True,timeout=3)
                if p.read_text().strip()!='performance':raise RuntimeError('CPU performance governor unavailable')
            if not self.saved and not list(Path('/sys/devices/system/cpu/cpufreq').glob('policy*')):
                raise RuntimeError('CPU frequency policy unavailable')
            return self
        except BaseException:self.__exit__(None,None,None);raise
    def __exit__(self,*args):
        for p,value in self.saved.items():
            subprocess.run(['sudo','-n','/usr/bin/tee',str(p)],input=value+'\n',text=True,
                stdout=subprocess.DEVNULL,check=True,timeout=3)
        self.saved={}

def validate_ground(armed,connected,state_age,feedback_age):
    if armed is not False:raise RuntimeError('flight controller armed or state unknown')
    if not connected or state_age>2.:raise RuntimeError('flight state stale/disconnected')
    if feedback_age>.15:raise RuntimeError('gimbal feedback stale')

def validate_sample(smp,now,previous):
    ts=smp.get('t_mono')
    if ts is None or not math.isfinite(ts) or not 0<=now-ts<=.15:
        raise RuntimeError('gimbal source timestamp stale/invalid')
    if previous is not None:
        dt=ts-previous['t_mono']
        if device_sequence_delta(smp['dev_seq'],previous['dev_seq'])<=0 or dt<=0:
            raise RuntimeError('gimbal sample out of order')
        if dt>.05:raise RuntimeError('gimbal source sample gap >50 ms')


def centering_driver_connected(subscribers):
    return any(topic=='/gimbal/cmd_center' and '/gimbal_node' in nodes
               for topic,nodes in subscribers)


def exclusive_gimbal_publishers(published):
    # /shoot and rosbridge keep cmd_angle advertised while idle. Actual angle
    # messages remain a stop condition through the command monitor below.
    return [v for v in published if v[0] in ('/gimbal/cmd_rate','/gimbal/cmd_center') and v[1]]

def chirp_command(t,T,deviation,phase,f0=.5,f1=4.,amplitude=8.,level_limit=10):
    if not all(math.isfinite(v) for v in (t,T,deviation,phase,f0,f1,amplitude,level_limit)) or T<=0 or min(f0,f1)<=0 or amplitude<0 or not 0<level_limit<=100:
        raise ValueError('invalid chirp settings')
    e=min(1.,max(0.,.5-.5*math.cos(math.pi*min(t,T-t)/2))) if min(t,T-t)<2 else 1.
    e=max(0.,e)
    f=f0+(f1-f0)*min(max(t/T,0),1)
    raw=e*amplitude*math.sin(phase+2*math.pi*(f0*t+.5*(f1-f0)*t*t/T))
    u=int(round(max(-level_limit,min(level_limit,raw-2.*deviation))))
    return u,raw,f

class CameraQuiet:
    def __init__(self):self.was_active=False;self.stopped=False
    def enter(self):
        check=subprocess.run(['systemctl','is-active','cam-web.service'],capture_output=True,text=True,timeout=3)
        self.was_active=check.stdout.strip()=='active'
        if self.was_active:
            subprocess.run(['sudo','-n','systemctl','stop','cam-web.service'],check=True,timeout=6)
        check=subprocess.run(['systemctl','is-active','cam-web.service'],capture_output=True,text=True,timeout=3)
        if check.stdout.strip()=='active':raise RuntimeError('video service still active')
        self.stopped=True
    def close(self):
        if self.was_active and self.stopped:
            subprocess.run(['sudo','-n','systemctl','start','cam-web.service'],check=True,timeout=6)
            self.stopped=False

class GroundLink:
    def __init__(self,level_limit=45):
        self.level_limit=level_limit;self.cv=threading.Condition();self.sample=None
        self.state=None;self.state_t=0.;self.sample_t=0.;self.count=0;self.reason=''
        self.pub=None;self.center_pub=None;self.saved_stream=None;self.imu=None;self.cam=CameraQuiet()
        self.last_consumed=None
        addr=os.environ.get('GCS_ADDR','127.0.0.1:15601').split(':')
        self.tel_dest=(addr[0],int(addr[1]));self.sk=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.sk.setblocking(False)

    def __enter__(self):
        try:
            self.cam.enter()
            import rospy
            from geometry_msgs.msg import Vector3
            from std_msgs.msg import String,Bool
            from mavros_msgs.msg import State
            from sensor_msgs.msg import Imu
            self.rospy=rospy;self.Vector3=Vector3
            rospy.init_node('ast_ground_proposed',anonymous=True,disable_signals=True)
            pubs=rospy.get_master().getSystemState()[2][0]
            others=exclusive_gimbal_publishers(pubs)
            if others:raise RuntimeError('gimbal command endpoint occupied: '+repr(others))
            def state_cb(m):
                with self.cv:self.state=m;self.state_t=time.monotonic();self.cv.notify_all()
            def sample_cb(m):
                try:s=json.loads(m.data)
                except (ValueError,TypeError):return
                with self.cv:
                    self.sample=s;self.sample_t=time.monotonic();self.count+=1;self.cv.notify_all()
            rospy.Subscriber('/mavros/state',State,state_cb,queue_size=2)
            rospy.Subscriber('/gimbal/sample',String,sample_cb,queue_size=3)
            def imu_cb(m):
                self.imu=dict(t=time.monotonic(),q=[m.orientation.x,m.orientation.y,m.orientation.z,m.orientation.w],
                              ros_stamp=m.header.stamp.to_sec(),
                              linear_acceleration=[m.linear_acceleration.x,m.linear_acceleration.y,m.linear_acceleration.z],
                              angular_velocity=[m.angular_velocity.x,m.angular_velocity.y,m.angular_velocity.z])
            rospy.Subscriber('/mavros/imu/data',Imu,imu_cb,queue_size=2)
            self.pub=rospy.Publisher('/gimbal/cmd_rate',Vector3,queue_size=1)
            # Advertise before the feedback warmup so rosbag discovers its type
            # before the single non-latched native command is published.
            self.center_pub=rospy.Publisher('/gimbal/cmd_center',Bool,queue_size=1,latch=False)
            def cmd_cb(m):
                caller=m._connection_header.get('callerid','')
                if caller!=rospy.get_name():self.reason='external gimbal command received'
            rospy.Subscriber('/gimbal/cmd_rate',Vector3,cmd_cb,queue_size=1)
            rospy.Subscriber('/gimbal/cmd_angle',Vector3,cmd_cb,queue_size=1)
            rospy.Subscriber('/gimbal/cmd_center',Bool,cmd_cb,queue_size=1)
            self.saved_stream=rospy.get_param('/gimbal_node/stream_hz',4)
            timeout=rospy.get_param('/gimbal_node/rate_timeout',.5)
            if not 0<timeout<=.5:raise RuntimeError('driver command watchdog not configured <=0.5 s')
            rospy.set_param('/gimbal_node/stream_hz',6)
            if hasattr(self,'setup_extra'):self.setup_extra()
            for sig in (signal.SIGINT,signal.SIGTERM):
                signal.signal(sig,lambda s,f:setattr(self,'reason','signal '+str(s)))
            deadline=time.monotonic()+7.
            while time.monotonic()<deadline:
                t=time.monotonic();n=self.count
                time.sleep(1.)
                if self.count-n>=40 and self.state is not None:
                    self.guard();break
            else:raise RuntimeError('fresh 40-Hz gimbal feedback/flight state not ready')
            return self
        except BaseException:
            self.close();raise

    def guard(self):
        if self.reason:raise RuntimeError(self.reason)
        now=time.monotonic();s=self.state
        validate_ground(s.armed if s else None,s.connected if s else False,now-self.state_t,now-self.sample_t)
        if self.sample is not None:validate_sample(self.sample,now,None)
        try:
            while True:
                data,source=self.sk.recvfrom(64)
                if data.strip()==b'STOP' and source[0] in (self.tel_dest[0],'127.0.0.1'):
                    raise RuntimeError('ground station STOP')
        except BlockingIOError:pass

    def next(self,last_seq):
        while True:
            self.guard()
            with self.cv:
                smp=self.sample
                if smp and smp.get('dev_seq')!=last_seq:
                    validate_sample(smp,time.monotonic(),self.last_consumed)
                    self.last_consumed=dict(smp)
                    return dict(smp)
                self.cv.wait(.005)

    def send(self,yaw,pitch=0):
        self.guard()
        if any(int(v)!=v or abs(v)>self.level_limit for v in (yaw,pitch)):
            raise RuntimeError('invalid bounded integer gimbal command')
        self.pub.publish(self.Vector3(x=yaw,y=pitch,z=0.))

    def telemetry(self,data):
        try:self.sk.sendto(json.dumps(data,allow_nan=False).encode('utf-8'),self.tel_dest)
        except (OSError,ValueError):pass

    def request_center(self,pre_publish_guard=None):
        """Same native home as the web button, with live guards and no camera IO."""
        from std_msgs.msg import Bool
        self.guard()
        if self.center_pub is None:
            self.center_pub=self.rospy.Publisher('/gimbal/cmd_center',Bool,queue_size=1,latch=False)
        deadline=time.monotonic()+1.5
        while True:
            self.guard()
            subscribers=self.rospy.get_master().getSystemState()[2][1]
            # Driver, our competing-command monitor and the independent recorder.
            if centering_driver_connected(subscribers) and self.center_pub.get_num_connections()>=3:break
            if time.monotonic()>deadline:raise RuntimeError('native centering driver unavailable')
            time.sleep(.01)
        self.guard()
        if pre_publish_guard is not None:pre_publish_guard()
        self.center_pub.publish(Bool(data=True))

    def stop(self):
        if self.pub is not None:
            for _ in range(3):
                self.pub.publish(self.Vector3());time.sleep(.02)

    def close(self):
        self.stop()
        if self.saved_stream is not None:
            self.rospy.set_param('/gimbal_node/stream_hz',self.saved_stream);self.saved_stream=None
        self.cam.close()
        try:self.sk.close()
        except OSError:pass

    def __exit__(self,*args):self.close()
