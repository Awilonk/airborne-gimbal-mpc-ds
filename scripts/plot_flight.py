"""Compact Figure 8 review proof with paired yaw/pitch vector RMSE."""
from pathlib import Path
from datetime import datetime, timezone, timedelta
import json,hashlib,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

W=Path(__file__).resolve().parent
ROOT=W.parent
R=ROOT
OUT=R/'figures';OUT.mkdir(exist_ok=True)
meta=json.loads((ROOT/'data/flight/metrics.json').read_text(encoding='utf-8'))
methods=['PID','LADRC','NoLearn','Proposed']
names=['PID','LADRC','MPC-DS w/o LR','MPC-DS (Proposed)']
ticks=['PID','LADRC','MPC-DS\nw/o LR','MPC-DS\n(Proposed)']
colors=['#009E73','#D55E00','#7B5294','#0072B2']
hatches=['xx','\\\\','//','']
records=[]
for method in methods:
 row=next(v for v in meta['methods'] if v['raw_id']==method)
 a=np.loadtxt(ROOT/row['file'],delimiter=',',skiprows=1)
 t=a[:,0];q=a[:,1:3];ref=a[:,3:5];e=q-ref;mask=(t>=5)&(t<=28)
 rms=np.sqrt(np.mean(e[mask]**2,axis=0));joint=float(np.sqrt(np.mean(np.sum(e[mask]**2,axis=1))))
 assert np.allclose(rms,[row['yaw_rmse_deg'],row['pitch_rmse_deg']],atol=1e-12)
 records.append(dict(method=method,t=t,q=q,ref=ref,e=e,mask=mask,rms=rms,joint=joint,n=int(mask.sum())))
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
 'mathtext.fontset':'stix','font.size':8,'axes.labelsize':8.2,'axes.titlesize':8.3,
 'legend.fontsize':7.7,'xtick.labelsize':7.8,'ytick.labelsize':7.8,
 'axes.linewidth':.65,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
 'axes.unicode_minus':False,'savefig.facecolor':'white'})
def style(ax):
 ax.spines[['top','right']].set_visible(False)
 ax.tick_params(direction='out',length=2.5,width=.6,pad=2)
 ax.grid(True,color='#dddddd',lw=.4,alpha=.7,zorder=0)

fig=plt.figure(figsize=(7.12,3.65))
phase=np.linspace(0,2*np.pi,721)
extent=float(np.ceil(max(np.max(abs(r['e'][r['mask']])) for r in records)*2)/2)
for k,(r,col) in enumerate(zip(records,colors)):
 row,column=divmod(k,2)
 ax=fig.add_axes([.05+column*.265,.57 if row==0 else .15,.237,.36]);style(ax)
 ax.plot(r['q'][r['mask'],0],r['q'][r['mask'],1],color=col,lw=.55,alpha=.9)
 ax.plot(8*np.sin(phase),8*np.cos(phase),color='#222222',ls=(0,(4,2.5)),lw=.8)
 idx=[int(np.argmin(abs(r['t']-tt))) for tt in 10+np.arange(4)/(4*.6)]
 ax.add_collection(LineCollection(np.array([[r['ref'][j],r['q'][j]] for j in idx]),colors='#333333',linewidths=.75,zorder=5))
 ax.scatter(r['ref'][idx,0],r['ref'][idx,1],s=8,facecolors='white',edgecolors='#333333',lw=.6,zorder=6)
 ax.set_aspect('equal',adjustable='box')
 ax.set(xlim=(-10.2,10.2),ylim=(-10.2,10.2),xticks=[-8,0,8],yticks=[-8,0,8])
 ax.set_title(f'({chr(97+k)}) {names[k]}',loc='left',pad=4)
 ax.text(0,.85,r'RMSE$_{\mathrm{2D}}$',ha='center',va='center',fontsize=7.8,color='#333333')
 ax.text(0,-1.2,f'{r["joint"]:.3f}°',ha='center',va='center',fontsize=9.4,fontweight='bold',color=col)
 if column==0:ax.set_ylabel('Pitch (°)',labelpad=1.5)
 if row==1:ax.set_xlabel('Yaw (°)',labelpad=1.5)
 else:ax.tick_params(labelbottom=False)

for j,axis in enumerate(['Yaw','Pitch']):
 ax=fig.add_axes([.655,.57 if j==0 else .15,.325,.36]);style(ax)
 vals=[r['e'][r['mask'],j] for r in records]
 vp=ax.violinplot(vals,positions=[1,2,3,4],widths=.72,showmeans=False,showmedians=False,showextrema=False,points=180,bw_method='scott')
 for k,(body,col,hatch) in enumerate(zip(vp['bodies'],colors,hatches)):
  rgb=np.array(matplotlib.colors.to_rgb(col));body.set_facecolor(.23*rgb+.77);body.set_edgecolor(.5*rgb+.5);body.set_alpha(1)
  body.set_linewidth(.75);body.set_hatch(hatch)
  lo,mid,hi=np.quantile(vals[k],[.25,.5,.75])
  ax.plot([k+1,k+1],[lo,hi],color='#222222',lw=1.8,zorder=5)
  ax.scatter(k+1,mid,s=10,c='white',edgecolors='#222222',lw=.6,zorder=6)
  ax.text(k+1,extent+.6,f'{records[k]["rms"][j]:.3f}',ha='center',va='top',fontsize=7.7)
 ax.axhline(0,color='#666666',ls=(0,(4,3)),lw=.6)
 ax.set(xlim=(.4,4.6),ylim=(-extent,extent+1.5),xticks=[1,2,3,4],xticklabels=ticks,
  yticks=[-3,0,3],ylabel='Error (°)')
 ax.set_title(f'({chr(101+j)}) {axis} error',loc='left',pad=4)
 ax.text(.02,.985,'RMSE (°)',transform=ax.transAxes,va='top',fontsize=7.7)
 if j==0:ax.tick_params(labelbottom=False)
 ax.tick_params(axis='x',labelsize=7.6)

fig.legend(handles=[Line2D([],[],color='#222222',ls=(0,(4,2.5)),lw=.9,label='Reference'),
 Line2D([],[],color='#333333',marker='o',mfc='white',ms=3,lw=.75,label='Same-time pair')],
 loc='lower left',bbox_to_anchor=(.065,.008),ncol=2,frameon=False,columnspacing=1.4,fontsize=7.7)
fig.text(.98,.025,'LR: learned lead residual',ha='right',va='bottom',fontsize=7.7)
fig.canvas.draw();renderer=fig.canvas.get_renderer()
for txt in fig.findobj(matplotlib.text.Text):
 if txt.get_visible() and txt.get_text():
  b=txt.get_window_extent(renderer)
  assert b.x0>=-1 and b.y0>=-1 and b.x1<=fig.bbox.x1+1 and b.y1<=fig.bbox.y1+1,(txt.get_text(),b)
stem='fig_hover_v6_20261024_lyk'
for ext in ('pdf','svg','png'):fig.savefig(OUT/(stem+'.'+ext),dpi=400)
plt.close(fig)
audit={'state':'User-approved naming rule: MPC first; included in v6',
 'window_s':[5,28],'units':'degree','joint_formula':'sqrt(mean(error_yaw_deg**2 + error_pitch_deg**2))',
 'joint_metric_interpretation':'RMSE of synchronous distance in the yaw-pitch angular coordinate plane; no division by two',
 'previous_figure_size_in':[7.12,4.5],'candidate_figure_size_in':[7.12,3.65],
 'height_reduction_pct':100*(1-3.65/4.5),'filtering':False,'resampling':False,
 'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data/flight').glob('*.csv')},
 'methods':[{'raw_method':r['method'],'candidate_name':names[k],'samples':r['n'],
  'yaw_rmse_deg':float(r['rms'][0]),'pitch_rmse_deg':float(r['rms'][1]),'two_axis_rmse_deg':r['joint']} for k,r in enumerate(records)],
 'outputs':[str(OUT/(stem+'.'+e)) for e in ('pdf','svg','png')]}
(W/'fig8_metrics_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(audit,ensure_ascii=False,indent=2))
