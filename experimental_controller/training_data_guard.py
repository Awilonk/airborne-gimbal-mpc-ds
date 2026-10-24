"""Keep dedicated identification acquisitions separate from evaluation data."""
import json,math
from pathlib import Path
from durable_log import recover_records


def validate_acquisition(path,expected_segment):
    path=Path(path)
    meta=json.loads((path/'manifest.json').read_text(encoding='utf-8'))
    if (meta.get('stage')!='identification_acquisition' or
            meta.get('data_role','identification')!='identification'):
        raise ValueError('dedicated identification data required; evaluation data cannot train models')
    if expected_segment not in ('train','validation') or meta.get('segment')!=expected_segment:
        raise ValueError('identification train/validation segment mismatch')
    if 'training_allowed' in meta and meta['training_allowed'] is not (expected_segment=='train'):
        raise ValueError('identification training permission does not match segment')
    summary=json.loads((path/'summary.json').read_text(encoding='utf-8'))
    rows=recover_records(path)
    n=len(rows)
    reported_end=float(summary.get('t_end',0))
    if (summary.get('aborted') is not False or summary.get('reason') or
            summary.get('complete',True) is not True or n<1400 or
            summary.get('record_count')!=n or summary.get('durable_seq')!=n-1 or
            summary.get('n')!=n or not math.isfinite(reported_end) or reported_end<29.5):
        raise ValueError('complete durable 30-second identification acquisition required')
    limit=float(meta.get('level_limit',10))
    if not math.isfinite(limit) or not 0<limit<=100:raise ValueError('invalid identification input limit')
    previous=None
    for row in rows:
        t=float(row['t'])
        values=[t]+list(row['cmd_frame'][:2])+list(row['rate_raw'][:2])+[row['u_yaw'],row['u_pitch']]
        if (row.get('armed') is not False or not all(math.isfinite(float(v)) for v in values)
                or any(abs(row[k])>limit for k in ('u_yaw','u_pitch'))):
            raise ValueError('invalid/disarmed identification sample required')
        if previous is not None and not 0<t-previous<=.05:
            raise ValueError('identification timestamp gap or ordering invalid')
        previous=t
    if rows[-1]['t']<29.5 or abs(rows[-1]['t']-reported_end)>.05:raise ValueError('complete identification timeline required')
    return meta


def validate_split(trainpaths,validation):
    resolved=[Path(p).resolve() for p in trainpaths]
    if len(set(resolved))!=len(resolved) or Path(validation).resolve() in resolved:
        raise ValueError('train/validation overlap is forbidden')


def validate_output(base,out,active):
    target=(Path(out)/'paper40_lead_real_finetuned.npz').resolve()
    if target in (Path(base).resolve(),Path(active).resolve()):
        raise ValueError('training output cannot overwrite frozen active/base weights')
    if target.exists():raise ValueError('candidate weight already exists; use a new output directory')
