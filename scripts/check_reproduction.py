from pathlib import Path
import json,sys
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
m=json.loads((ROOT/'data/flight/metrics.json').read_text())
for row in m['methods']:
 a=np.loadtxt(ROOT/row['file'],delimiter=',',skiprows=1)
 mask=(a[:,0]>=5)&(a[:,0]<=28);e=a[mask,1:3]-a[mask,3:5]
 actual=np.sqrt(np.mean(e**2,axis=0));joint=np.sqrt(np.mean(np.sum(e**2,axis=1)))
 assert np.allclose(actual,[row['yaw_rmse_deg'],row['pitch_rmse_deg']],atol=1e-12)
 assert abs(joint-row['two_axis_rmse_deg'])<1e-12
sys.path.insert(0,str(ROOT/'simulation'))
sys.path.insert(0,str(ROOT/'simulation/paper_sec4_20261022'))
from run_sec4_rho import build
from disc_pred import run
from sim_zr10 import CAL_REAL
r=run(lambda:build('M5r','cal',1.0),100,1.0,Tsw=20.0,cal=CAL_REAL)
frozen=np.load(ROOT/'simulation/paper_sec4_20261022/traces_sec4.npz')['M5r']
assert r.shape==frozen.shape
assert np.allclose(r,frozen,atol=1e-8,rtol=1e-8),float(np.max(abs(r-frozen)))
print('Four flight RMSEs and the 30-s MPC-DS simulation trace match the frozen paper data.')
