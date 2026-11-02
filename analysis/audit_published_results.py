"""Reconcile the published simulated ledger, daily equity and monthly returns."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent/'results/archived_replay'

def main():
    d=pd.read_csv(P/'10bps_daily.csv',parse_dates=['date']);f=pd.read_csv(P/'10bps_fills.csv');m=pd.read_csv(P/'actual_monthly_returns.csv');s=pd.read_csv(P/'sensitivity.csv');v=json.loads((P/'verification.json').read_text())
    assert len(d)==290 and len(f)==v['matched_fills']==1254
    assert f.action.eq('BUY').sum()==632 and f.action.eq('SELL').sum()==622
    cash=100000.;hold={}
    for x in f.itertuples():
        if x.action=='BUY':
            assert x.symbol not in hold
            hold[x.symbol]=x.shares;cash-=x.amount
            np.testing.assert_allclose(x.amount,x.notional+x.fee,rtol=1e-12)
        else:
            np.testing.assert_allclose(hold.pop(x.symbol),x.shares,rtol=1e-12)
            cash+=x.amount;np.testing.assert_allclose(x.amount,x.notional-x.fee,rtol=1e-12)
        np.testing.assert_allclose(cash,x.cash_after,atol=1e-7)
        assert cash>=-1e-7 and len(hold)<=10
    assert len(hold)==10
    np.testing.assert_allclose(cash,d.cash.iloc[-1],atol=1e-7)
    np.testing.assert_allclose(d.total_value,d.cash+d.holdings_value,rtol=1e-12)
    assert (d.cash>=-1e-7).all()
    monthly=d.set_index('date').total_value.resample('ME').last().pct_change();monthly.iloc[0]=d.set_index('date').total_value.resample('ME').last().iloc[0]/100000-1
    np.testing.assert_allclose(monthly,m['return'],rtol=1e-12)
    for row in s.itertuples():
        q=pd.read_csv(P/(row.case+'_daily.csv'));np.testing.assert_allclose(q.total_value.iloc[-1]/100000-1,row.total_return,rtol=1e-12)
        np.testing.assert_allclose(q.total_value,q.cash+q.holdings_value,rtol=1e-12)
    e=pd.read_csv(P/'closed_trade_expectancy.csv');z=json.loads((P/'expectancy_summary.json').read_text())
    assert len(e)==len(f[f.action.eq('SELL')])==622
    np.testing.assert_allclose(e.gross_return_pct-e.round_trip_cost_pct,e.net_return_pct,atol=1e-10)
    np.testing.assert_allclose(e.net_pnl.to_numpy(),f.loc[f.action.eq('SELL'),'profit'].to_numpy(),atol=1e-8)
    np.testing.assert_allclose(e.net_return_pct.mean(),z['mean_net_return_pct'],atol=1e-12)
    print('PASS: 1,254 fills; cash ledger; 10 open holdings; 290 daily marks; monthly compounding; four sensitivity curves.')
if __name__=='__main__':main()
