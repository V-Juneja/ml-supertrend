"""Offline, single-symbol signal generation from completed OHLCV bars."""
from pathlib import Path
import argparse
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from ml_supertrend import MLSupertrend, SupertrendConfig

def load_bars(path):
    df=pd.read_csv(path)
    required=['time','open','high','low','close','volume']
    if set(required)-set(df):raise ValueError('Required columns: '+', '.join(required))
    # Require explicit timezone information; never infer local time from naive timestamps.
    if not df.time.astype(str).str.match(r'.*(Z|[+-]\d{2}:?\d{2})$').all():raise ValueError('time must include UTC Z or an explicit UTC offset')
    df['time']=pd.to_datetime(df.time,utc=True)
    if df.time.isna().any() or df.time.duplicated().any() or not df.time.is_monotonic_increasing:raise ValueError('Times must be unique and increasing')
    for col in required[1:]:df[col]=pd.to_numeric(df[col],errors='raise')
    if not np.isfinite(df[required[1:]].to_numpy()).all():raise ValueError('OHLCV must be finite')
    if (df[['open','high','low','close']]<=0).any().any() or (df.volume<0).any():raise ValueError('Prices must be positive; volume nonnegative')
    if (df.high<df[['open','close','low']].max(axis=1)).any() or (df.low>df[['open','close','high']].min(axis=1)).any():raise ValueError('Invalid candle high/low')
    if len(df)<100:raise ValueError('Provide at least 100 completed bars; retain a consistent warm-up history')
    if (df.time>pd.Timestamp.now(tz='UTC')).any():raise ValueError('Future candle close time')
    return df.set_index('time')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--output',type=Path,default=Path('outputs/signals.csv'))
    parser.add_argument('--fixed',action='store_true',help='Disable adaptive learning for a fixed-setting comparison; this does not run a portfolio backtest')
    args=parser.parse_args();df=load_bars(args.input)
    cfg=SupertrendConfig(signal_mode='Reversal',enable_tick_pressure=False,enable_adaptive=not args.fixed)
    result=MLSupertrend(cfg).run(df)
    args.output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(args.output)
    print(f'{len(result)} completed bars; {int(result.buy_signal.sum())} BUY and {int(result.sell_signal.sum())} SELL signals. Saved {args.output}')
if __name__=='__main__':main()
