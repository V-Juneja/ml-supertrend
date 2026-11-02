"""Plot actual provider-sourced OHLCV and learning state. Never generates price data."""
from pathlib import Path
import argparse,sys,json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'src')]
from run_signals import load_bars
from ml_supertrend import MLSupertrend,SupertrendConfig

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('--output-dir',type=Path,default=Path('outputs/chart'));p.add_argument('--symbol',required=True);p.add_argument('--provider',required=True);p.add_argument('--bars',type=int,default=78);args=p.parse_args()
 df=load_bars(args.input);res=MLSupertrend(SupertrendConfig(enable_tick_pressure=False,lock_atr_bands=False)).run(df)
 fixed=MLSupertrend(SupertrendConfig(enable_tick_pressure=False,enable_adaptive=False)).run(df)
 n=min(args.bars,len(df));a=res.iloc[-n:];b=fixed.iloc[-n:];x=np.arange(n);out=args.output_dir;out.mkdir(parents=True,exist_ok=True)
 plt.rcParams.update({'font.family':'serif','font.size':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 def price(ax,legend=True):
  for i,r in enumerate(a.itertuples()):
   color='#157c77' if r.close>=r.open else '#b54141';ax.vlines(i,r.low,r.high,color=color,lw=.65);ax.add_patch(Rectangle((i-.3,min(r.open,r.close)),.6,max(abs(r.close-r.open),.015),facecolor=color,edgecolor=color,lw=.4))
  ax.plot(x,a.upper_band,color='#157c77',lw=1.2,label='Adaptive lower boundary');ax.plot(x,a.lower_band,color='#b54141',lw=1.2,label='Adaptive upper boundary');ax.fill_between(x,a.upper_band,a.lower_band,color='#536ca1',alpha=.06)
  ax.plot(x,b.upper_band,color='.55',lw=.65,ls='--',label='Fixed baseline boundaries');ax.plot(x,b.lower_band,color='.55',lw=.65,ls='--')
  for col,mark,color,offset in [('buy_signal','^','#006c61',-.5),('sell_signal','v','#b02929',.5)]:
   mask=a[col].to_numpy(dtype=bool);ax.scatter(x[mask],a.close.to_numpy()[mask]+offset,marker=mark,c=color,s=22,zorder=5)
  ticks=np.arange(0,n,max(1,n//6));ax.set_xticks(ticks,[a.index[i].tz_convert('America/New_York').strftime('%d %b\n%H:%M') for i in ticks]);ax.set_ylabel('Price (USD)');ax.grid(axis='y',alpha=.12)
  if legend:ax.legend(loc='upper left',ncol=3,fontsize=6,frameon=False)
 fig,axs=plt.subplots(3,1,figsize=(9,6),sharex=True,gridspec_kw={'height_ratios':[3,1,1]},layout='constrained');price(axs[0]);axs[0].set_title(f'{args.symbol} | Completed 30-minute bars | Adaptive vs fixed volatility bands')
 axs[1].plot(x,a.adaptive_multiplier,color='#5a4586',lw=1.2);axs[1].axhline(1.4,color='.5',ls='--',lw=.7);axs[1].set_ylabel('Learned\nmultiplier')
 axs[2].plot(x,a.adaptive_atr_period,color='#1d607b',lw=1.2);axs[2].axhline(24,color='.5',ls='--',lw=.7);axs[2].set_ylabel('Learned ATR\nlength (bars)');axs[2].set_xlabel('Completed candle close (America/New_York; overnight gaps omitted)')
 fig.savefig(out/'adaptive_bands.png',dpi=400,bbox_inches='tight');fig.savefig(out/'adaptive_bands.pdf',bbox_inches='tight');plt.close(fig)
 fig,ax=plt.subplots(figsize=(7,1.7),layout='constrained');price(ax);ax.set_title(f'{args.symbol}: adaptive signal bands and fixed baseline',fontsize=9);fig.savefig(out/'adaptive_bands_report.pdf',bbox_inches='tight');fig.savefig(out/'adaptive_bands_report.png',dpi=450,bbox_inches='tight');plt.close(fig)
 metadata=dict(symbol=args.symbol,provider=args.provider,input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),warmup_start=df.index[0].isoformat(),warmup_end=df.index[-1].isoformat(),shown_start=a.index[0].isoformat(),shown_end=a.index[-1].isoformat(),bars_in_full_run=len(df),displayed_bars=n,engine='Python research engine; not a TradingView rendering',tick_pressure=False,adaptive_plot_bands=True)
 (out/'chart_provenance.json').write_text(json.dumps(metadata,indent=2));print(json.dumps(metadata,indent=2))
if __name__=='__main__':main()
