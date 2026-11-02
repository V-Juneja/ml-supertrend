"""Parameter and chronological-prefix checks on user-supplied completed market bars."""
from pathlib import Path
import argparse,sys,json,hashlib
import pandas as pd
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R),str(R/'src')]
from run_signals import load_bars
from ml_supertrend import MLSupertrend,SupertrendConfig

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('--output-dir',type=Path,default=Path('outputs/model_checks'));a=p.parse_args();df=load_bars(a.input);out=a.output_dir;out.mkdir(parents=True,exist_ok=True)
 rows=[]
 for length in [20,24,28]:
  for mult in [1.2,1.4,1.6]:
   model=MLSupertrend(SupertrendConfig(atr_period=length,multiplier=mult,enable_tick_pressure=False));r=model.run(df)
   rows.append(dict(initial_atr=length,initial_multiplier=mult,buy_count=int(r.buy_signal.sum()),sell_count=int(r.sell_signal.sum()),completed_long_probes=model._long_total,completed_short_probes=model._short_total))
 pd.DataFrame(rows).to_csv(out/'parameter_checks.csv',index=False)
 base=MLSupertrend(SupertrendConfig(enable_tick_pressure=False)).run(df);cols=['buy_signal','sell_signal','trend_signal','adaptive_multiplier','adaptive_atr_period'];prefixes=[]
 for n in [200,400,650]:
  if n<len(df):
   prefix=MLSupertrend(SupertrendConfig(enable_tick_pressure=False)).run(df.iloc[:n]);pd.testing.assert_frame_equal(base[cols].iloc[:n],prefix[cols]);prefixes.append(n)
 fixed=MLSupertrend(SupertrendConfig(enable_tick_pressure=False,enable_adaptive=False)).run(df)
 summary=dict(bars=len(df),start=df.index[0].isoformat(),end=df.index[-1].isoformat(),input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),prefix_checks=prefixes,adaptive_buy=int(base.buy_signal.sum()),adaptive_sell=int(base.sell_signal.sum()),fixed_buy=int(fixed.buy_signal.sum()),fixed_sell=int(fixed.sell_signal.sum()),purpose='Implementation checks only; no portfolio return or Pine parity claim')
 (out/'checks.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
