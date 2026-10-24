"""Nonblocking control logging with fsynced CSV/JSONL checkpoints.

Only records at/before checkpoint byte offsets are claimed durable. Power loss
can still lose the tail since the last sync; the configured period is recorded.
"""
import copy,csv,json,os,queue,threading,time,uuid
from pathlib import Path

def sync_dir(path):
    if os.name=='nt':return
    fd=os.open(str(path),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',encoding='utf-8',newline='') as f:
        json.dump(value,f,ensure_ascii=False,allow_nan=False,indent=2)
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path);sync_dir(path.parent)

class DurableRun:
    def __init__(self,root,metadata,fields,sync_period=.2,queue_size=4096,max_stall=1.):
        root=Path(root);root.mkdir(parents=True,exist_ok=True)
        self.path=root/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
        self.path.mkdir();sync_dir(root)
        self.fields=['seq']+list(fields);self.period=sync_period;self.max_stall=max_stall
        self.q=queue.Queue(maxsize=queue_size);self.error=None;self.durable_seq=-1
        self.next_seq=0;self.closed=False;self.progress=time.monotonic();self.cv=threading.Condition()
        atomic_json(self.path/'manifest.json',dict(metadata,sync_period_s=sync_period,
                   power_loss_tail='only the checkpoint prefix is guaranteed by this software protocol'))
        self.thread=threading.Thread(target=self._writer,name='durable-log',daemon=True);self.thread.start()

    def assert_healthy(self):
        if self.error is not None:raise RuntimeError('log writer failed: '+repr(self.error))
        if self.next_seq-1>self.durable_seq and time.monotonic()-self.progress>self.max_stall:
            raise RuntimeError('log writer fsync progress stalled')

    def submit(self,record):
        if self.closed:raise RuntimeError('logger already closed')
        self.assert_healthy()
        if set(record)!=set(self.fields[1:]):raise ValueError('log schema mismatch')
        row=copy.deepcopy(record);row['seq']=self.next_seq
        try:self.q.put_nowait(row)
        except queue.Full:raise RuntimeError('log queue full; stop controller')
        self.next_seq+=1;return row['seq']

    def _writer(self):
        try:
            with (self.path/'control.jsonl').open('wb') as jf,(self.path/'control.csv').open('w',encoding='utf-8',newline='') as cf:
                csvout=csv.DictWriter(cf,fieldnames=self.fields);csvout.writeheader()
                last=-1;sync_t=time.monotonic();finishing=False
                def checkpoint():
                    nonlocal sync_t
                    jf.flush();os.fsync(jf.fileno());cf.flush();os.fsync(cf.fileno())
                    atomic_json(self.path/'checkpoint.json',dict(last_seq=last,jsonl_bytes=jf.tell(),
                                csv_bytes=os.fstat(cf.fileno()).st_size,wall_time=time.time()))
                    with self.cv:
                        self.durable_seq=last;self.progress=time.monotonic();self.cv.notify_all()
                    sync_t=time.monotonic()
                checkpoint()
                while not finishing:
                    try:row=self.q.get(timeout=max(.001,self.period-(time.monotonic()-sync_t)))
                    except queue.Empty:row='timeout'
                    if row is None:finishing=True
                    elif row!='timeout':
                        jf.write((json.dumps(row,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode('utf-8'))
                        csvout.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(list,dict)) else v for k,v in row.items()})
                        last=row['seq']
                    if finishing or time.monotonic()-sync_t>=self.period:checkpoint()
        except BaseException as ex:
            self.error=ex
            with self.cv:self.cv.notify_all()

    def wait_durable(self,seq,timeout=3.):
        end=time.monotonic()+timeout
        with self.cv:
            while self.durable_seq<seq:
                self.assert_healthy();left=end-time.monotonic()
                if left<=0:raise TimeoutError('durable logging timeout')
                self.cv.wait(min(left,.05))

    def close(self,summary):
        if self.closed:return
        self.closed=True
        try:self.q.put(None,timeout=2.)
        except queue.Full:raise RuntimeError('unable to finish log writer')
        self.thread.join(timeout=4.)
        if self.thread.is_alive():raise RuntimeError('log writer did not finish')
        if self.error is not None:raise RuntimeError('log writer failed: '+repr(self.error))
        if self.durable_seq!=self.next_seq-1:raise RuntimeError('not all submitted samples durable')
        atomic_json(self.path/'summary.json',dict(summary,record_count=self.next_seq,durable_seq=self.durable_seq))

def recover_records(path):
    path=Path(path);cp=json.loads((path/'checkpoint.json').read_text(encoding='utf-8'))
    with (path/'control.jsonl').open('rb') as f:data=f.read(cp['jsonl_bytes'])
    rows=[json.loads(v) for v in data.splitlines()]
    if len(rows)!=cp['last_seq']+1 or [r['seq'] for r in rows]!=list(range(len(rows))):
        raise ValueError('checkpoint prefix is not contiguous')
    return rows
