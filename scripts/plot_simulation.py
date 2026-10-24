"""Restyle the CURRENT frozen simulation results, never reuse old-paper numbers."""
from pathlib import Path
import hashlib, importlib.util, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, FixedLocator, FixedFormatter

W=Path(__file__).resolve().parent
S=W.parent/'simulation/paper_sec4_20261022'
R=W.parent
F=R/'figures';F.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('sec4_frozen',S/'make_sec4.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
D=np.load(S/'traces_sec4.npz')
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],
 'font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':8.5,'legend.fontsize':8,
 'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':.65,
 'lines.linewidth':1,'xtick.direction':'in','ytick.direction':'in',
 'xtick.top':True,'ytick.right':True,'grid.linewidth':.35,
 'pdf.fonttype':42,'ps.fonttype':42,'mathtext.fontset':'stix'})
sty={
 'M1':('#7f7f7f',':'),'M2':('#2185c5','--'),'M3a':('#9271ad','-.'),
 'Xie':('#929292',(0,(4,2))),'XieAd':('#d58a32',(0,(3,1,1,1))),
 'M4':('#79a640','-.'),'M5':('#df3446','-')}
audit={'data_sources':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [S/'results_sec4.json',S/'traces_sec4.npz']},
 'old_paper_data_used':False,'raw_series_filtered':False,'figures':{},'quantities':{}}


method_names={'M1':'MPC','M2':'MPC-DO','M3a':'MPC-APDO',
 'Xie':'MPC with fixed frequency','XieAd':'MPC with adaptive frequency',
 'M4':'MPC-APDO (preview)','M5':'MPC-DS'}
tick_names={'M1':'MPC','M2':'MPC-DO','M3a':'MPC-APDO',
 'Xie':'MPC with\nfixed frequency','XieAd':'MPC with\nadaptive frequency',
 'M4':'MPC-APDO\n(preview)','M5':'MPC-DS'}

def finish(fig,name):
 stem=name+'_v6_20261024_lyk'
 fig.savefig(F/(stem+'.pdf'),bbox_inches='tight',pad_inches=.035)
 fig.savefig(F/(stem+'.png'),dpi=350,bbox_inches='tight',pad_inches=.035)
 audit['figures'][name]=str(F/(stem+'.pdf'))
 plt.close(fig)

def frame(ax):
 ax.grid(axis='y',alpha=.25);ax.set_axisbelow(True)

# Preserve the seven-method, forty-run box summaries, including extrema.
meths=('M1','M2','M3a','Xie','XieAd','M4','M5')
fig,axs=plt.subplots(1,2,figsize=(7.12,2.65),sharey=True)
fig.subplots_adjust(left=.075,right=.99,bottom=.29,top=.9,wspace=.12)
for j,sc in enumerate(base.SCEN):
 data=[base.stats('cal',m,sc)['all'] for m in meths]
 audit['quantities']['box_'+sc]=[x.tolist() for x in data]
 bp=axs[j].boxplot(data,widths=.62,whis=(0,100),patch_artist=True,
  medianprops={'color':'#292929','lw':.85},boxprops={'lw':.7},
  whiskerprops={'lw':.65},capprops={'lw':.65})
 for box,m in zip(bp['boxes'],meths):box.set_facecolor(sty[m][0]);box.set_alpha(.47 if m!='M5' else .7)
 axs[j].axhline(1,color='#555555',ls=':',lw=.8)
 axs[j].set_yscale('log');axs[j].set_ylim(.07,1.65)
 axs[j].yaxis.set_major_locator(FixedLocator([.1,.2,.5,1]));axs[j].yaxis.set_major_formatter(FixedFormatter(['0.1','0.2','0.5','1']))
 axs[j].yaxis.set_minor_formatter(NullFormatter())
 axs[j].set_xticks(np.arange(1,8),[tick_names[m] for m in meths],rotation=38,ha='right',rotation_mode='anchor')
 axs[j].set_title(('(a) Constant frequency','(b) Frequency switch')[j],pad=7)
 frame(axs[j])
axs[0].set_ylabel('RMSE (°)')
finish(fig,'fig_rmse_box')

# Show complete unfiltered trace and a pre-switch detail window, all four methods.
fig,axs=plt.subplots(2,1,figsize=(3.55,3.8),sharex=True,gridspec_kw={'height_ratios':(1.6,1)})
fig.subplots_adjust(left=.17,right=.98,bottom=.12,top=.89,hspace=.34)
for m in ('Xie','M2','M4','M5'):
 r=D['M5r' if m=='M5' else m];c,ls=sty[m]
 axs[0].plot(r[:,0],r[:,2]-r[:,1],color=c,ls=ls,lw=.65 if m=='Xie' else .9,
  alpha=.8 if m=='Xie' else 1,label=method_names[m],zorder=4 if m=='M5' else 2)
 audit['quantities']['trace_'+m]={'points':len(r),'rmse_5_30':float(np.sqrt(np.mean((r[(r[:,0]>=5),2]-r[(r[:,0]>=5),1])**2)))}
axs[0].set_ylim(-3.15,3.9);axs[0].set_yticks([-2,0,2]);axs[0].set_ylabel('Tracking error (°)')
axs[0].legend(ncol=2,loc='lower left',bbox_to_anchor=(-.04,1.01),frameon=False,handlelength=2.3,columnspacing=1.5)
axs[0].text(.02,.95,'(a)',transform=axs[0].transAxes,va='top')
zoom=axs[0].inset_axes([.49,.68,.48,.29])
for m in ('Xie','M2','M4','M5'):
 r=D['M5r' if m=='M5' else m];c,ls=sty[m]
 zoom.plot(r[:,0],r[:,2]-r[:,1],color=c,ls=ls,lw=.65)
zoom.set_xlim(16,18);zoom.set_ylim(-1.6,1.6);zoom.set_xticks([16,17,18]);zoom.set_yticks([-1,0,1]);zoom.tick_params(labelsize=7.6,pad=1)
zoom.set_facecolor('white');zoom.grid(alpha=.15)
r=D['M5r']
axs[1].step(r[:,0],r[:,3],where='post',color=sty['M5'][0],lw=.65,label='$u_k$')
for y,ls,lab in [(50,'--','$\\pm u_{\\max}$'),(45,':','$\\pm(u_{\\max}-m)$')]:
 axs[1].axhline(y,color='#444444',ls=ls,lw=.8,label=lab);axs[1].axhline(-y,color='#444444',ls=ls,lw=.8)
axs[1].set_ylim(-75,65);axs[1].set_yticks([-50,0,50]);axs[1].set_ylabel('Command (level)');axs[1].set_xlabel('Time (s)')
axs[1].text(.02,.07,'(b)',transform=axs[1].transAxes,va='bottom')
axs[1].legend(ncol=3,loc='lower center',bbox_to_anchor=(.5,1.01),frameon=False,fontsize=7.6,columnspacing=.7,handlelength=1.8)
for ax in axs:ax.axvline(float(D['Tsw']),color='#555555',ls='--',lw=.7);ax.set_xlim(0,30);frame(ax)
finish(fig,'fig_time_switch')

# Frequency estimator: same frozen record, complete transient retained.
fig,ax=plt.subplots(figsize=(3.55,2.25));fig.subplots_adjust(left=.16,right=.98,bottom=.23,top=.96)
t=D['M5r'][:,0];ft=np.where(t<float(D['Tsw']),float(D['f_true0']),1.6*float(D['f_true0']))
ax.plot(t,ft,color='#333333',lw=1.2,ls=':',label='True frequency')
ax.plot(t,D['M5r_noLIN'][:,8],color='#2185c5',ls='--',lw=.95,label='Gradient law (APDO)')
ax.plot(t,D['M5r'][:,8],color=sty['M5'][0],lw=1,label='LIN (proposed)')
ax.axvline(float(D['Tsw']),color='#555555',ls='--',lw=.65)
ax.set_xlim(0,30);ax.set_xlabel('Time (s)');ax.set_ylabel('Frequency (Hz)');frame(ax)
ax.legend(loc='upper left',frameon=False,fontsize=7.8)
finish(fig,'fig_freq')

# Preserve the original paired relative-RMSE estimator and the four cases.
var=('M5','M5r-noRho','M5r-noLIN','M5r-noPred','M5r-noMargin','M5r-noNom','M5r-noProj')
lab=('MPC-DS','w/o drift\nweighting','w/ gradient\nfrequency estimation','w/o periodic\nprediction','w/o input\ntightening','w/ horizon-held\naperiodic estimate','w/o amplitude\nprojection')
cases=(('cal','steady',0.),('cal','switch',0.),('cal','steady',.3),('cal','switch',.3))
names=('Constant frequency','Frequency switch','Constant, ±30%','Switch, ±30%')
fig,ax=plt.subplots(figsize=(7.12,2.55));fig.subplots_adjust(left=.08,right=.99,bottom=.3,top=.79)
for i,(env,sc,pt) in enumerate(cases):
 vals=[base.stats(env,m,sc,pt)['rel'] for m in var]
 audit['quantities']['ablation_'+str((env,sc,pt))]=vals
 ax.bar(np.arange(7)+(i-1.5)*.2,vals,.2,color=('#66a8d5','#a5c779','#a9d2ed','#d3e6b6')[i],
  edgecolor='#333333',lw=.4,hatch='///' if i>=2 else '',label=names[i])
ax.axhline(0,color='#333333',lw=.7);ax.set_xticks(range(7),lab,rotation=0,ha='center')
ax.set_ylabel('RMSE relative to MPC-DO (%)');ax.legend(ncol=2,frameon=False,loc='lower left',bbox_to_anchor=(-.04,1.01),fontsize=7.7,columnspacing=1)
frame(ax);finish(fig,'fig_ablation')

fig,axs=plt.subplots(1,2,figsize=(7.12,2.4),sharey=True)
fig.subplots_adjust(left=.075,right=.99,bottom=.22,top=.67,wspace=.14)
for j,sc in enumerate(base.SCEN):
 for m in ('M2','M3a','XieAd','M4','M5'):
  y=[base.stats('cal',m,sc,pt)['mean'] for pt in base.PERTS]
  audit['quantities']['mismatch_'+sc+'_'+m]=y
  c,ls=sty[m];axs[j].plot([0,10,20,30],y,color=c,ls=ls,marker='o',ms=3,mfc='white' if m!='M5' else c,mew=.75,label=method_names[m])
 axs[j].set_yscale('log');axs[j].set_xticks([0,10,20,30]);axs[j].set_xlabel('Mismatch bound (%)')
 axs[j].yaxis.set_major_locator(FixedLocator([.1,.2,.5,1]));axs[j].yaxis.set_major_formatter(FixedFormatter(['0.1','0.2','0.5','1']))
 axs[j].yaxis.set_minor_formatter(NullFormatter())
 axs[j].set_title(('(a) Constant frequency','(b) Frequency switch')[j],pad=7);frame(axs[j])
axs[0].set_ylabel('Mean RMSE (°)')
handles,labels=axs[1].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='upper center',bbox_to_anchor=(.53,1.0),frameon=False,fontsize=7.8,handlelength=1.6,columnspacing=.7)
finish(fig,'fig_mismatch')
(W/'simulation_style_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
print('Five current-data simulation figures exported; original figures retained.')


