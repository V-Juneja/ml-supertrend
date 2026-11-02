"""Declared-portfolio Monte Carlo stress. No account ledger or historical returns used.
The generated shocks are conditional risk scenarios, not simulated market-data evidence.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def metrics(r):
 loss=-r;v=float(np.quantile(loss,.95));return dict(var95_pct=v*100,es95_pct=float(loss[loss>=v].mean()*100),prob_loss10_pct=float((loss>.10).mean()*100),median_return_pct=float(np.median(r)*100))
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output-dir',type=Path,default=Path('outputs/risk'));args=ap.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
 seed=20261101;N=100000;rng=np.random.default_rng(seed);common=rng.normal(size=(N,1));idio=rng.normal(size=(N,6));rows=[]
 for days in [15,126]:
  for exposure in [.2,.4,.6,.75]:
   for sigma in [.25,.45,.65]:
    for rho in [0,.35,.65,.9]:
     z=np.sqrt(rho)*common+np.sqrt(1-rho)*idio
     stock=np.exp(-.5*sigma*sigma*days/252+sigma*np.sqrt(days/252)*z)
     ret=exposure*(stock.mean(axis=1)-1)
     rows.append(dict(days=days,exposure=exposure,sigma=sigma,rho=rho,**metrics(ret)))
 grid=pd.DataFrame(rows);grid.to_csv(out/'terminal_stress.csv',index=False)
 count=40000;days=15;sigma=.45;rho=.65;daily=[];paths={}
 models=['Gaussian','Student-t(5)','10% common gap','20% common gap']
 for j,model in enumerate(models):
  rng=np.random.default_rng(seed+100+j);prices=np.ones((count,6));stock=np.ones((count,days+1))
  for t in range(days):
   z=np.sqrt(rho)*rng.normal(size=(count,1))+np.sqrt(1-rho)*rng.normal(size=(count,6))
   if model=='Student-t(5)':z*=np.sqrt(3/rng.chisquare(5,size=(count,1)))
   prices*=np.exp(-.5*sigma*sigma/252+sigma/np.sqrt(252)*z)
   if t==0 and 'gap' in model:prices*=.8 if model.startswith('20') else .9
   stock[:,t+1]=prices.mean(axis=1)
  for exposure in [.4,.6,.75]:
   eq=1+exposure*(stock-1);dd=1-eq/np.maximum.accumulate(eq,axis=1);guard=np.ones_like(eq);halted=np.zeros(count,bool);peak=np.ones(count)
   for t in range(days):
    guard[:,t+1]=np.where(halted,guard[:,t],eq[:,t+1]);peak=np.maximum(peak,guard[:,t+1]);halted|=guard[:,t+1]/peak-1<=-.06
   daily.append(dict(model=model,exposure=exposure,prob_drawdown6_pct=float((dd.max(axis=1)>=.06).mean()*100),guard_es95_pct=metrics(guard[:,-1]-1)['es95_pct'],**metrics(eq[:,-1]-1)))
   if model=='Gaussian' and exposure==.4:paths['fan']=eq
 daily=pd.DataFrame(daily);daily.to_csv(out/'daily_risk.csv',index=False)
 shocks=pd.DataFrame([dict(exposure=w,stock_loss_pct=k*100,account_loss_pct=w*k*100) for w in [.2,.4,.6,.75] for k in [.05,.1,.2,.3,.5]]);shocks.to_csv(out/'deterministic_shocks.csv',index=False)
 # Stop-fill stress: planned risk across the book vs worsening loss/stop multiples.
 stops=pd.DataFrame([dict(aggregate_planned_risk_pct=r,loss_to_planned_multiple=m,account_loss_pct=r*m) for r in [1,2,3] for m in [1,1.5,2,3]]);stops.to_csv(out/'stop_overshoot.csv',index=False)
 base=grid[(grid.days==15)&(grid.sigma==.45)&(grid.rho==.65)].to_dict('records')
 summary=dict(seed=seed,terminal_draws_per_case=N,terminal_cases=len(grid),daily_paths_per_model=count,daily_cases=len(daily),daily_seeds=list(range(seed+100,seed+104)),days=days,baseline=base,daily=daily.to_dict('records'),assumptions='Six equal stock holdings, frozen shares, fixed uninvested balance, no interest, no fresh signals, no costs; Gaussian expected stock drift zero. Shared t5 scale gives standardized heavy-tailed log innovations, without a finite expected price drift. Guard liquidates at daily close after 6% drawdown. These scenarios are not strategy backtests.')
 (out/'risk_summary.json').write_text(json.dumps(summary,indent=2))
 plt.rcParams.update({'font.family':'serif','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 fig,axs=plt.subplots(2,2,figsize=(7,4.4),layout='constrained');ax=axs[0,0];q=np.quantile(paths['fan'],[.05,.25,.5,.75,.95],axis=0)*100
 ax.fill_between(range(16),q[0],q[4],color='.88',label='5th-95th percentile');ax.fill_between(range(16),q[1],q[3],color='.65',label='25th-75th');ax.plot(q[2],c='black',lw=1);ax.axhline(94,c='.4',ls='--',lw=.7);ax.set(title='A  Fifteen-session Monte Carlo paths',xlabel='Trading sessions',ylabel='Portfolio value (start = 100)');ax.legend(fontsize=6,frameon=False)
 ax=axs[0,1];g=grid[(grid.days==15)&(grid.exposure==.4)].pivot(index='sigma',columns='rho',values='es95_pct');ax.imshow(g,cmap='Greys',vmin=0,vmax=16,aspect='auto');ax.set_xticks(range(4),['0','.35','.65','.90']);ax.set_yticks(range(3),['25%','45%','65%']);ax.set(title='B  Expected shortfall, worst 5% (%)',xlabel='Log-shock correlation',ylabel='Annual stock volatility')
 for i in range(3):
  for j in range(4):ax.text(j,i,f'{g.values[i,j]:.1f}',ha='center',va='center',color='white' if g.values[i,j]>8 else 'black')
 ax=axs[1,0];d=daily[daily.exposure==.4];x=np.arange(4);ax.barh(x-.16,d.es95_pct,height=.32,color='.65',label='No liquidation');ax.barh(x+.16,d.guard_es95_pct,height=.32,color='.2',label='6% daily-close trigger');ax.set_yticks(x,models);ax.invert_yaxis();ax.set(title='C  Liquidation reduces, but cannot cap, tail loss',xlabel='15-session expected shortfall (%)');ax.legend(fontsize=6,frameon=False)
 ax=axs[1,1];x=np.arange(3);width=.19
 for j,w in enumerate([.2,.4,.6,.75]):ax.bar(x+(j-1.5)*width,[w*k*100 for k in [.1,.2,.3]],width,color=["#dddddd","#aaaaaa","#777777","#444444"][j],label=f'{w:.0%} exposure')
 ax.set_xticks(x,['-10%','-20%','-30%']);ax.set(title='D  Correlated equity gap stress',xlabel='Simultaneous stock-book shock',ylabel='Account loss (%)');ax.legend(fontsize=6,frameon=False)
 fig.savefig(out/'declared_risk_dashboard.pdf',bbox_inches='tight');fig.savefig(out/'declared_risk_dashboard.png',dpi=450,bbox_inches='tight');plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,1.5),layout='constrained')
 for r,c in [(1,'.7'),(2,'.35'),(3,'black')]:
  a=stops[stops.aggregate_planned_risk_pct==r];ax.plot(a.loss_to_planned_multiple,a.account_loss_pct,'o-',ms=3,c=c,label=f'{r}% planned account risk')
 ax.axhline(6,ls='--',c='.5',lw=.7);ax.set(xlabel='Realized loss / planned stop loss',ylabel='Account loss (%)',title='Gap and slippage sensitivity of planned stop budgets');ax.legend(fontsize=6,frameon=False,ncol=3)
 fig.savefig(out/'stop_budget.pdf',bbox_inches='tight');plt.close(fig)
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
