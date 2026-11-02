"""Re-run the archived ranked-email replay with different fixed per-name caps."""
from pathlib import Path
import argparse, hashlib, json
import pandas as pd
from replay_archived_signals import replay, metrics


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('signals',type=Path)
    ap.add_argument('bars',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args()
    signals=pd.read_csv(a.signals); bars=pd.read_csv(a.bars)
    rows=[]
    for pct in (0.05,0.10,0.15):
        daily,fills=replay(signals,bars,fee_bps=5,slippage_bps=5,position_notional=100000*pct)
        m=metrics(daily,fills)
        open_names=set(); peak_positions=0
        for fill in fills.itertuples():
            if fill.action=='BUY': open_names.add(fill.symbol)
            else: open_names.discard(fill.symbol)
            peak_positions=max(peak_positions,len(open_names))
        rows.append({'per_name_cap_pct':100*pct,'per_name_dollars_at_start':100000*pct,
          'peak_positions':peak_positions,'average_gross_exposure_pct':100*daily.exposure.mean(),
          'ending_equity':m['final_value'],'total_return_pct':100*m['total_return'],
          'max_drawdown_pct':100*m['max_drawdown'],'sharpe_zero_rf':m['daily_sharpe_zero_rf'],
          'closed_trades':m['sell_fills'],'win_rate_pct':100*m['closed_trade_win_rate'],
          'positive_months':m['positive_months'],'negative_months':m['negative_months']})
    out=pd.DataFrame(rows);a.output.parent.mkdir(parents=True,exist_ok=True);out.to_csv(a.output,index=False)
    a.output.with_suffix('.json').write_text(json.dumps({
      'signals_sha256':hashlib.sha256(a.signals.read_bytes()).hexdigest(),
      'bars_sha256':hashlib.sha256(a.bars.read_bytes()).hexdigest(),
      'start':str(daily.date.min().date()),'end':str(daily.date.max().date()),
      'sessions':int(len(daily)),'fee_bps_per_side':5,'slippage_bps_per_side':5,
      'position_tickets':'fixed at 5%, 10%, and 15% of $100,000 starting equity',
      'max_names':10,'scope':'Saved daily BUY rankings; identical exit rules and next-session open fills across size cases.'
    },indent=2)+'\n')
    print(out.to_string(index=False))

if __name__=='__main__':main()
