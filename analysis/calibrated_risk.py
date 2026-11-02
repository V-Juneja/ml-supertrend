"""Cross-asset Monte Carlo and block-bootstrap risk from provider-sourced 30m bars.
Input directory: one SYMBOL.csv per asset, columns Datetime,open,high,low,close,volume.
Datetime is a bar-OPEN timestamp with explicit UTC offset (the provider cache contract).
No account returns or positions are used. No generated price data are presented as evidence.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def measure(eq):
 loss=1-eq[:,-1];v=np.quantile(loss,.95);dd=1-eq/np.maximum.accumulate(eq,axis=1);hit=dd.max(axis=1)>=.06;p=hit.mean();se=np.sqrt(p*(1-p)/len(eq))
 guard=np.ones_like(eq);halt=np.zeros(len(eq),bool);peak=np.ones(len(eq))
 for t in range(1,eq.shape[1]):
  guard[:,t]=np.where(halt,guard[:,t-1],eq[:,t]);peak=np.maximum(peak,guard[:,t]);halt|=guard[:,t]/peak-1<=-.06
 gl=1-guard[:,-1];gv=np.quantile(gl,.95)
 return dict(var95_pct=float(v*100),es95_pct=float(loss[loss>=v].mean()*100),prob_drawdown6_pct=float(p*100),mc_probability_se_pp=float(se*100),guard_es95_pct=float(gl[gl>=gv].mean()*100),max_drawdown95_pct=float(np.quantile(dd.max(axis=1),.95)*100))
def main():
 a=argparse.ArgumentParser(description=__doc__);a.add_argument('bars_dir',type=Path);a.add_argument('--output-dir',type=Path,default=Path('outputs/calibrated_risk'));args=a.parse_args();out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
 names=['DELL','MSFT','JPM','UNH','GE','XOM'];closes={};hashes={}
 for symbol in names:
  p=args.bars_dir/(symbol+'.csv');d=pd.read_csv(p);d['Datetime']=pd.to_datetime(d.Datetime,utc=True);d=d.set_index('Datetime').sort_index()
  if d.index.duplicated().any():raise ValueError('Duplicate bar timestamps')
  # Local trading date, using only observed regular-session bars. No missing prices filled.
  local=d.index.tz_convert('America/New_York');mask=(local.hour*60+local.minute>=570)&(local.hour*60+local.minute<960);d=d.loc[mask];local=d.index.tz_convert('America/New_York')
  closes[symbol]=d.close.groupby(local.date).last();hashes[symbol]=hashlib.sha256(p.read_bytes()).hexdigest()
 prices=pd.DataFrame(closes).dropna();returns=np.log(prices/prices.shift()).dropna()
 if len(returns)<40:raise ValueError('Insufficient common daily history')
 x=returns.to_numpy();centered=x-x.mean(axis=0);lw=LedoitWolf().fit(x);cov=lw.covariance_;vol=np.sqrt(np.diag(cov));corr=cov/np.outer(vol,vol);chol=np.linalg.cholesky(cov);rho=.90;crisis_cov=np.outer(vol*1.5,vol*1.5)*(rho+(1-rho)*np.eye(6));crisis_chol=np.linalg.cholesky(crisis_cov)
 pd.DataFrame(cov,index=names,columns=names).to_csv(out/'shrinkage_covariance.csv');pd.DataFrame(corr,index=names,columns=names).to_csv(out/'shrinkage_correlation.csv');returns.to_csv(out/'observed_daily_log_returns.csv')
 N=50000;H=15;seed=20261102;rows=[];fan=None
 models=['Calibrated Gaussian','Student-t(5)','5-session block bootstrap','Crisis: vol x1.5, rho .90','20% common opening gap']
 for j,model in enumerate(models):
  rng=np.random.default_rng(seed+j)
  if 'bootstrap' in model:
   starts=rng.integers(0,len(x),size=(N,3));ix=(starts[:,:,None]+np.arange(5))%len(x);shocks=centered[ix.reshape(N,H)]
   # The empirical innovations have zero sample log mean; no expected-price-drift claim.
   increments=shocks-.5*np.diag(cov)
  else:
   mat=crisis_chol if model.startswith('Crisis') else chol;vv=np.diag(crisis_cov) if model.startswith('Crisis') else np.diag(cov)
   z=rng.normal(size=(N,H,6))
   if model=='Student-t(5)':z*=np.sqrt(3/rng.chisquare(5,size=(N,H,1)))
   increments=z@mat.T-.5*vv
  if model.startswith('20%'):increments[:,0,:]+=np.log(.8)
  stock=np.exp(np.cumsum(increments,axis=1)).mean(axis=2);stock=np.column_stack([np.ones(N),stock])
  for w in [.4,.6,.75]:
   eq=1+w*(stock-1);rows.append(dict(model=model,exposure=w,**measure(eq)))
   if j==0 and w==.4:fan=eq
 result=pd.DataFrame(rows);result.to_csv(out/'calibrated_scenarios.csv',index=False)
 summary=dict(symbols=names,calibration_start=str(prices.index[0]),calibration_end=str(prices.index[-1]),daily_return_observations=len(returns),shrinkage_intensity=float(lw.shrinkage_),annualized_stock_volatility=dict(zip(names,(vol*np.sqrt(252)).tolist())),simulation_paths=N,sessions=H,base_seed=seed,source_sha256=hashes,results=rows,scope='Illustrative cross-sector risk basket, not an actual position list or a strategy replay. Zero price drift is imposed only in Gaussian models. Bootstrap and t stress do not estimate expected price returns; parameter uncertainty and real execution are not captured.')
 (out/'calibrated_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
 plt.rcParams.update({'font.family':'serif','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 fig,axs=plt.subplots(2,2,figsize=(7,4.4),layout='constrained');ax=axs[0,0];q=np.quantile(fan,[.05,.25,.5,.75,.95],axis=0)*100;ax.fill_between(range(16),q[0],q[4],color='.86',label='5th-95th percentile');ax.fill_between(range(16),q[1],q[3],color='.6',label='25th-75th');ax.plot(q[2],c='black',lw=1);ax.set(title='A  Calibrated Monte Carlo fan',xlabel='Trading sessions',ylabel='Account value (start = 100)');ax.legend(fontsize=6,frameon=False)
 labels=['Gaussian','t(5)','Block bootstrap','Crisis','20% gap'];ax=axs[0,1];y=np.arange(5)
 for k,w in enumerate([.4,.6,.75]):
  v=result[result.exposure==w].es95_pct;ax.barh(y+(k-1)*.24,v,height=.24,color=['#168a66','#6e9d91','#b7c9c5'][k],label=f'{w:.0%} exposure')
 ax.set_yticks(y,labels);ax.invert_yaxis();ax.set(title='B  Tail-risk model comparison',xlabel='15-session expected shortfall (%)');ax.legend(fontsize=6,frameon=False)
 ax=axs[1,0];ax.imshow(corr,cmap='Greys',vmin=-1,vmax=1);asset_labels=[f'Asset {chr(65+i)}' for i in range(6)];ax.set_xticks(range(6),asset_labels,rotation=45);ax.set_yticks(range(6),asset_labels);ax.set_title('C  Ledoit-Wolf shrinkage correlation')
 for i in range(6):
  for k in range(6):ax.text(k,i,f'{corr[i,k]:.2f}',ha='center',va='center',fontsize=6,color='white' if corr[i,k]>.55 else 'black')
 ax=axs[1,1];r=result[result.exposure==.4];ax.barh(y,r.es95_pct,color='.7',label='No liquidation');ax.plot(r.guard_es95_pct,y,'ko',ms=3,label='6% daily-close trigger');ax.set_yticks(y,labels);ax.invert_yaxis();ax.set(title='D  Effect of liquidation rule',xlabel='15-session expected shortfall (%)');ax.legend(fontsize=6,frameon=False)
 fig.savefig(out/'calibrated_dashboard.pdf',bbox_inches='tight');fig.savefig(out/'calibrated_dashboard.png',dpi=450,bbox_inches='tight');plt.close(fig)
if __name__=='__main__':main()
