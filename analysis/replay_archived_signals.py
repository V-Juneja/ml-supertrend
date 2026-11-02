"""Offline replay of previously archived top-ten BUY lists on provider-supplied bars.

This historical workflow is separate from the declared 30-minute ML Supertrend
portfolio. It consumes saved rankings; it does not reconstruct or train them.
No network, mail or brokerage access is used.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import pandas as pd


def replay(signals, bars, fee_bps=5., slippage_bps=5., extra_delay=0):
    required={'Date','Rank','Symbol','Score','Signal'}
    if not required.issubset(signals):raise ValueError('Signal columns: '+', '.join(sorted(required)))
    if not {'date','symbol','open','close'}.issubset(bars):raise ValueError('Bar columns: date,symbol,open,close')
    if extra_delay<0 or fee_bps<0 or slippage_bps<0:raise ValueError('Nonnegative costs/delay required')
    signals=signals.copy();bars=bars.copy()
    signals['Date']=pd.to_datetime(signals.Date).dt.normalize();bars['date']=pd.to_datetime(bars.date).dt.normalize()
    if signals.duplicated(['Date','Rank']).any() or bars.duplicated(['date','symbol']).any():raise ValueError('Duplicate input keys')
    if not signals.Signal.eq('BUY').all():raise ValueError('Archive must contain BUY rankings')
    if not np.isfinite(bars[['open','close']]).all().all() or (bars[['open','close']]<=0).any().any():raise ValueError('Invalid prices')
    # QQQ supplies the recorded replay calendar, not a trading signal.
    sessions=sorted(bars.loc[bars.symbol.eq('QQQ'),'date'].unique())
    if not sessions:raise ValueError('Include QQQ bars to define the session calendar')
    sessions=list(map(pd.Timestamp,sessions));p=bars.set_index(['date','symbol'])
    groups={day:g.sort_values('Rank') for day,g in signals.groupby('Date')}
    schedule={}
    for day in sorted(groups):
        eligible=[t for t in sessions if t>day]
        if len(eligible)>extra_delay:schedule[eligible[extra_delay]]=day
    cash=100000.;hold={};fills=[];daily=[];prior={};active=None
    fee=fee_bps/10000;slip=slippage_bps/10000
    for day in sessions:
        if day in schedule:active=schedule[day]
        if active is None:continue
        current={}
        for row in groups[active].itertuples():
            if row.Score>=0 and row.Symbol not in current:current[row.Symbol]=(row.Score,row.Rank)
            if len(current)==10:break
        for symbol,h in list(hold.items()):
            if symbol not in prior:continue
            stop=prior[symbol]<=h['entry']*.95;target=prior[symbol]>=h['entry']*1.15
            if stop or target or symbol not in current:
                market=float(p.at[(day,symbol),'open']);price=market*(1-slip);notional=h['shares']*price;charge=notional*fee;proceeds=notional-charge;profit=proceeds-h['basis'];cash+=proceeds
                fills.append(dict(date=day,symbol=symbol,action='SELL',shares=h['shares'],market_price=market,price=price,notional=notional,fee=charge,slippage_cost=h['shares']*(market-price),amount=proceeds,profit=profit,profit_pct=100*profit/h['basis'],cash_after=cash))
                del hold[symbol]
        for symbol,(score,rank) in current.items():
            if symbol in hold or cash<10000-1e-7:continue
            amount=min(10000.,cash);market=float(p.at[(day,symbol),'open']);price=market*(1+slip);shares=amount/(price*(1+fee));notional=shares*price;charge=notional*fee;debit=notional+charge;cash-=debit
            hold[symbol]=dict(shares=shares,entry=price,basis=debit)
            fills.append(dict(date=day,symbol=symbol,action='BUY',shares=shares,market_price=market,price=price,notional=notional,fee=charge,slippage_cost=shares*(price-market),amount=debit,score=score,rank=rank,cash_after=cash))
        holdings=sum(h['shares']*float(p.at[(day,s),'close']) for s,h in hold.items());value=cash+holdings
        daily.append(dict(date=day,total_value=value,cash=cash,holdings_value=holdings,exposure=holdings/value))
        prior={s:float(p.at[(day,s),'close']) for s in hold}
    return pd.DataFrame(daily),pd.DataFrame(fills)


def metrics(daily,fills):
    eq=daily.set_index('date').total_value;ret=eq.pct_change();ret.iloc[0]=eq.iloc[0]/100000-1
    dd=eq/np.maximum(eq.cummax(),100000)-1;days=(eq.index[-1]-eq.index[0]).days
    monthly=eq.resample('ME').last().pct_change();monthly.iloc[0]=eq.resample('ME').last().iloc[0]/100000-1
    closed=fills[fills.action.eq('SELL')]
    return dict(start=str(eq.index[0].date()),end=str(eq.index[-1].date()),sessions=len(eq),final_value=float(eq.iloc[-1]),total_return=float(eq.iloc[-1]/100000-1),annualized_growth=float((eq.iloc[-1]/100000)**(365.2425/days)-1),max_drawdown=float(dd.min()),daily_sharpe_zero_rf=float(np.sqrt(252)*ret.mean()/ret.std(ddof=1)),buy_fills=int(fills.action.eq('BUY').sum()),sell_fills=len(closed),closed_trade_win_rate=float(closed.profit.gt(0).mean()),positive_months=int(monthly.gt(0).sum()),negative_months=int(monthly.lt(0).sum()))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('signals',type=Path);ap.add_argument('bars',type=Path);ap.add_argument('--output-dir',type=Path,default=Path('outputs/replay'));a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True)
    signals=pd.read_csv(a.signals);bars=pd.read_csv(a.bars);rows=[]
    for label,fee,slip,delay in [('zero_cost',0,0,0),('10bps',5,5,0),('20bps',10,10,0),('10bps_extra_session',5,5,1)]:
        daily,fills=replay(signals,bars,fee,slip,delay);daily.to_csv(a.output_dir/(label+'_daily.csv'),index=False);fills.to_csv(a.output_dir/(label+'_fills.csv'),index=False)
        rows.append(dict(case=label,fee_bps=fee,slippage_bps=slip,extra_signal_delay_sessions=delay,**metrics(daily,fills)))
    pd.DataFrame(rows).to_csv(a.output_dir/'sensitivity.csv',index=False)
    (a.output_dir/'provenance.json').write_text(json.dumps(dict(signal_sha256=hashlib.sha256(a.signals.read_bytes()).hexdigest(),bars_sha256=hashlib.sha256(a.bars.read_bytes()).hexdigest(),scope='Saved-ranking replay, not a reconstruction of historical model signals. Last list carries forward. Fractional shares, no dividends/cash interest/borrowing. Extra-delay case delays ranking eligibility one session; stop/target exits retain next-open execution.'),indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
